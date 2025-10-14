"""
DataGenius PRO - Data Loader (PRO)
Universal data loader supporting multiple formats + compressed + SQL/URL
"""

from __future__ import annotations

import io
import json
import csv
from pathlib import Path
from typing import Union, Optional, Literal, Dict, Any, Tuple
from urllib.parse import urlparse

import pandas as pd
import polars as pl
from loguru import logger

from config.settings import settings
from config.constants import (
    SUPPORTED_FILE_EXTENSIONS,
    MAX_PREVIEW_ROWS,
)


class DataLoader:
    """
    Universal data loader with support for:
    - CSV (+ .csv.gz), Excel (XLSX, XLS)
    - JSON / JSONL / NDJSON, Parquet
    - URLs
    - SQL databases (query or table)
    """

    def __init__(self, use_polars: bool = False):
        """
        Initialize data loader

        Args:
            use_polars: Use Polars instead of Pandas (faster for large datasets)
        """
        self.use_polars = use_polars
        self.logger = logger.bind(component="DataLoader")

    # ------------------------- Public API -------------------------

    def load(
        self,
        filepath: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load data from file

        Args:
            filepath: Path to data file
            file_type: File type (auto-detected if None)
            **kwargs: Additional arguments for specific loaders

        Returns:
            DataFrame (pandas or polars)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Normalize file type + compression
        norm_type, compression = self._normalize_file_type(path, file_type)

        # Validate user-visible type (allow compressed forms)
        if norm_type not in SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {norm_type}. "
                f"Supported: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
            )

        self.logger.info(f"Loading data from {path} (type: {norm_type}, compression={compression or 'none'})")

        if norm_type == ".csv":
            df = self._load_csv(path, compression=compression, **kwargs)
        elif norm_type in [".xlsx", ".xls"]:
            df = self._load_excel(path, **kwargs)
        elif norm_type == ".json":
            # auto-detect jsonl/ndjson by suffix if not overridden
            is_lines = path.suffix.lower() in (".jsonl", ".ndjson")
            df = self._load_json(path, lines=is_lines, **kwargs)
        elif norm_type == ".parquet":
            df = self._load_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {norm_type}")

        self.logger.success(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        return df

    def load_from_url(
        self,
        url: str,
        file_type: str = ".csv",
        timeout: int = 30,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load data from URL

        Args:
            url: URL to data file
            file_type: File type ('.csv', '.json', '.jsonl', '.ndjson', '.parquet')
            timeout: HTTP timeout in seconds
            **kwargs: Additional arguments

        Returns:
            DataFrame
        """
        self._validate_url(url)
        self.logger.info(f"Loading data from URL: {url} (type: {file_type})")

        # Polars potrafi czytać z URL (HTTP(S)) w read_csv/read_parquet.
        if file_type == ".csv":
            if self.use_polars:
                return pl.read_csv(url, **kwargs)
            return pd.read_csv(url, on_bad_lines="skip", low_memory=False, **kwargs)

        if file_type in (".json", ".jsonl", ".ndjson"):
            if self.use_polars:
                if file_type in (".jsonl", ".ndjson"):
                    # Polars 0.20+: read_ndjson umie URL
                    try:
                        return pl.read_ndjson(url, **kwargs)
                    except Exception:
                        # fallback: requests -> DataFrame
                        import requests
                        resp = requests.get(url, timeout=timeout)
                        resp.raise_for_status()
                        lines = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
                        return pl.DataFrame(lines)
                else:
                    # JSON array/dict
                    import requests
                    resp = requests.get(url, timeout=timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    return pl.DataFrame(data)
            else:
                if file_type in (".jsonl", ".ndjson"):
                    return pd.read_json(url, lines=True, **kwargs)
                return pd.read_json(url, **kwargs)

        if file_type == ".parquet":
            if self.use_polars:
                return pl.read_parquet(url, **kwargs)
            return pd.read_parquet(url, **kwargs)

        raise ValueError(f"URL loading not supported for {file_type}")

    def load_from_sql(
        self,
        query_or_table: str,
        connection: Optional[Any] = None,
        con_str: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from SQL (pandas only).

        Args:
            query_or_table: SQL query or table name
            connection: existing SQLAlchemy Connection/Engine (preferred)
            con_str: connection string (fallback)
            params: query parameters
            limit: optional LIMIT applied if provided and query looks like table name
            **kwargs: forwarded to read_sql

        Returns:
            pandas.DataFrame
        """
        if self.use_polars:
            self.logger.warning("SQL loading uses pandas; result returned as pandas DataFrame.")
        try:
            import sqlalchemy as sa
        except Exception as e:
            raise ImportError("SQL loading requires SQLAlchemy installed.") from e

        engine = None
        if connection is None:
            if not con_str:
                # Use settings.DATABASE_URL if configured
                con_str = settings.get_database_url()
            engine = sa.create_engine(con_str)
            connection = engine.connect()

        try:
            if self._looks_like_table_name(query_or_table):
                q = sa.text(f"SELECT * FROM {query_or_table}" + (f" LIMIT {int(limit)}" if limit else ""))
                df = pd.read_sql(q, connection, params=params, **kwargs)
            else:
                df = pd.read_sql(sa.text(query_or_table), connection, params=params, **kwargs)
            return df
        finally:
            try:
                if engine is not None:
                    connection.close()
                    engine.dispose()
            except Exception:
                pass

    def save(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        filepath: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Save DataFrame to file

        Args:
            df: DataFrame to save
            filepath: Output filepath
            file_type: File type (auto-detected if None)
            **kwargs: Additional arguments
        """
        filepath = Path(filepath)
        if file_type is None:
            file_type = filepath.suffix.lower()

        self.logger.info(f"Saving data to {filepath} (type: {file_type})")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if file_type == ".csv":
            if isinstance(df, pl.DataFrame):
                df.write_csv(filepath, **kwargs)
            else:
                df.to_csv(filepath, index=False, **kwargs)

        elif file_type in [".xlsx", ".xls"]:
            if isinstance(df, pl.DataFrame):
                df.to_pandas().to_excel(filepath, index=False, **kwargs)
            else:
                df.to_excel(filepath, index=False, **kwargs)

        elif file_type == ".json":
            if isinstance(df, pl.DataFrame):
                # Polars ma write_json; w razie starszej wersji fallback
                try:
                    df.write_json(filepath, **kwargs)
                except Exception:
                    Path(filepath).write_text(json.dumps(df.to_pandas().to_dict(orient="records")), encoding="utf-8")
            else:
                df.to_json(filepath, orient="records", force_ascii=False, **kwargs)

        elif file_type == ".parquet":
            if isinstance(df, pl.DataFrame):
                df.write_parquet(filepath, **kwargs)
            else:
                # jawny fallback silnika
                try:
                    df.to_parquet(filepath, index=False, engine="pyarrow", **kwargs)
                except Exception:
                    df.to_parquet(filepath, index=False, engine="fastparquet", **kwargs)

        else:
            raise ValueError(f"Unsupported file type for saving: {file_type}")

        self.logger.success(f"Data saved to {filepath}")

    def load_sample(self, dataset_name: str) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load sample dataset

        Args:
            dataset_name: Name of sample dataset (iris, titanic, house_prices)

        Returns:
            DataFrame
        """
        sample_path = settings.ROOT_DIR / "data" / "samples" / f"{dataset_name}.csv"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample dataset not found: {dataset_name}")
        return self.load(sample_path)

    def get_preview(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        n_rows: int = MAX_PREVIEW_ROWS
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Get preview of DataFrame"""
        return df.head(n_rows)

    def get_info(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
        """
        Get detailed information about DataFrame
        """
        if isinstance(df, pl.DataFrame):
            try:
                mem_bytes = df.estimated_size()
            except Exception:
                mem_bytes = None
            return {
                "n_rows": df.height,
                "n_columns": df.width,
                "columns": df.columns,
                "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                "memory_usage": (mem_bytes / 1024**2) if mem_bytes is not None else None,  # MB
            }
        else:
            return {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                "memory_usage": float(df.memory_usage(deep=True).sum() / 1024**2),  # MB
                "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            }

    @staticmethod
    def to_pandas(df: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
        """Convert any DataFrame to pandas"""
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    @staticmethod
    def to_polars(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """Convert any DataFrame to polars"""
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        return df

    # ------------------------- Internals -------------------------

    def _normalize_file_type(self, path: Path, explicit: Optional[str]) -> Tuple[str, Optional[str]]:
        """
        Return (normalized_type, compression) where type ∈ SUPPORTED_FILE_EXTENSIONS.
        Accepts compressed variants like *.csv.gz => ('.csv', 'gzip')
        """
        if explicit:
            return explicit.lower(), None

        # handle multi-suffix like .csv.gz
        suff = [s.lower() for s in path.suffixes]
        if not suff:
            return path.suffix.lower(), None

        compression = None
        if suff[-1] in (".gz", ".bz2", ".zip", ".xz", ".zst"):
            compression = {
                ".gz": "gzip",
                ".bz2": "bz2",
                ".zip": "zip",
                ".xz": "xz",
                ".zst": "zstd",
            }.get(suff[-1], None)
            # previous suffix should be the data type
            if len(suff) >= 2:
                base = suff[-2]
            else:
                base = ".csv"  # guess
        else:
            base = suff[-1]

        # Map jsonl/ndjson to .json
        if base in (".jsonl", ".ndjson"):
            base = ".json"

        return base, compression

    def _load_csv(
        self,
        filepath: Path,
        encoding: str = "utf-8",
        delimiter: Optional[str] = None,
        compression: Optional[str] = None,
        sniff_delimiter: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Load CSV file (robust, with optional delimiter sniffing)"""
        sep = delimiter
        if sep is None and sniff_delimiter:
            try:
                with open(filepath, "r", encoding=encoding, newline="") as f:
                    sample = f.read(64_000)
                sniff = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = sniff.delimiter
                self.logger.info(f"Detected delimiter: {repr(sep)}")
            except Exception:
                sep = ","

        try:
            if self.use_polars:
                # Polars rozpoznaje kompresję po rozszerzeniu, ale przekażmy sep jeśli wykryty
                return pl.read_csv(
                    filepath,
                    separator=sep or ",",
                    encoding=encoding,
                    **kwargs
                )
            else:
                return pd.read_csv(
                    filepath,
                    encoding=encoding,
                    sep=sep or ",",
                    compression=compression,
                    on_bad_lines="skip",
                    low_memory=False,
                    **kwargs
                )
        except UnicodeDecodeError:
            self.logger.warning(f"Failed with {encoding}, trying latin-1")
            if self.use_polars:
                return pl.read_csv(filepath, separator=sep or ",", encoding="latin-1", **kwargs)
            else:
                return pd.read_csv(
                    filepath,
                    encoding="latin-1",
                    sep=sep or ",",
                    compression=compression,
                    on_bad_lines="skip",
                    low_memory=False,
                    **kwargs
                )

    def _load_excel(
        self,
        filepath: Path,
        sheet_name: Union[str, int] = 0,
        engine: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load Excel file (pandas)"""
        try:
            return pd.read_excel(filepath, sheet_name=sheet_name, engine=engine, **kwargs)
        except Exception as e:
            # Spróbuj innego silnika (openpyxl/xlrd)
            engines = ["openpyxl", "xlrd"]
            for eng in engines:
                try:
                    return pd.read_excel(filepath, sheet_name=sheet_name, engine=eng, **kwargs)
                except Exception:
                    continue
            raise e

    def _load_json(
        self,
        filepath: Path,
        lines: Optional[bool] = None,
        orient: str = "records",
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load JSON/NDJSON
        - lines=True -> NDJSON/JSONL
        """
        if self.use_polars:
            # Polars: prefer read_ndjson dla linii
            if lines:
                try:
                    return pl.read_ndjson(filepath, **kwargs)
                except Exception:
                    # fallback: manual
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = [json.loads(line) for line in f if line.strip()]
                    return pl.DataFrame(data)
            else:
                # read_json jest dostępny w nowszych Polars; w razie czego fallback
                try:
                    return pl.read_json(filepath, **kwargs)
                except Exception:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return pl.DataFrame(data)
        else:
            if lines is None:
                # heurystyka: jeśli plik ma wiele linii zaczynających się od '{'
                with open(filepath, "r", encoding="utf-8") as f:
                    head = "".join([next(f) for _ in range(5)]).strip()
                lines = "\n" in head and head.lstrip().startswith("{")
            return pd.read_json(filepath, lines=lines, orient=None if lines else orient, **kwargs)

    def _load_parquet(
        self,
        filepath: Path,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Load Parquet file"""
        if self.use_polars:
            return pl.read_parquet(filepath, **kwargs)
        try:
            return pd.read_parquet(filepath, engine="pyarrow", **kwargs)
        except Exception:
            return pd.read_parquet(filepath, engine="fastparquet", **kwargs)

    # ------------------------- Helpers -------------------------

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http(s) URLs are supported")

    @staticmethod
    def _looks_like_table_name(text: str) -> bool:
        # brak spacji/SELECT/sredników — surowa heurystyka
        t = text.strip().lower()
        return " " not in t and ";" not in t and not t.startswith(("select", "with"))
        

# Global loader instance
_data_loader: Optional[DataLoader] = None


def get_data_loader(use_polars: bool = False) -> DataLoader:
    """Get global data loader instance"""
    global _data_loader

    if _data_loader is None:
        _data_loader = DataLoader(use_polars=use_polars)

    return _data_loader
