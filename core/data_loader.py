# utils/data_loader.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Data Loader v7.0                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE UNIVERSAL DATA LOADER                                        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Multi-Format Support (CSV/Excel/JSON/Parquet)                         ║
║  ✓ Compression Support (gzip/bz2/zip/xz/zstd)                            ║
║  ✓ Pandas & Polars Support                                               ║
║  ✓ URL Loading                                                           ║
║  ✓ SQL Database Support                                                  ║
║  ✓ Smart Delimiter Detection                                             ║
║  ✓ Encoding Fallback                                                     ║
║  ✓ Sample Datasets                                                       ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Data Loader Structure:
```
    DataLoader
    ├── File Loading
    │   ├── CSV (+ compressed)
    │   ├── Excel (XLSX, XLS)
    │   ├── JSON (+ JSONL/NDJSON)
    │   └── Parquet
    ├── URL Loading
    │   ├── HTTP/HTTPS support
    │   ├── Format detection
    │   └── Timeout handling
    ├── SQL Loading
    │   ├── Query execution
    │   ├── Table reading
    │   └── Connection management
    └── Utilities
        ├── Format conversion
        ├── Preview generation
        └── Info extraction
```

Features:
    Multi-Format:
        • CSV with compression (gz, bz2, zip, xz, zst)
        • Excel (XLSX, XLS) with engine fallback
        • JSON (standard, JSONL, NDJSON)
        • Parquet with engine selection
    
    Smart Loading:
        • Automatic format detection
        • Delimiter sniffing
        • Encoding fallback (UTF-8 → Latin-1)
        • Compression detection
    
    Dual Engine:
        • Pandas (default, compatible)
        • Polars (faster for large data)
        • Seamless conversion
    
    Advanced Features:
        • URL loading (HTTP/HTTPS)
        • SQL database support
        • Sample datasets
        • Preview generation
        • Metadata extraction

Usage:
```python
    from utils.data_loader import DataLoader, get_data_loader
    
    # Get loader instance
    loader = get_data_loader()
    
    # Load from file
    df = loader.load("data.csv")
    df = loader.load("data.csv.gz")  # Compressed
    df = loader.load("data.xlsx")
    df = loader.load("data.json")
    df = loader.load("data.parquet")
    
    # Load from URL
    df = loader.load_from_url(
        "https://example.com/data.csv",
        file_type=".csv"
    )
    
    # Load from SQL
    df = loader.load_from_sql(
        "SELECT * FROM users WHERE active = :active",
        params={"active": True},
        limit=1000
    )
    
    # Save data
    loader.save(df, "output.parquet")
    loader.save(df, "output.csv.gz")
    
    # Load sample
    df = loader.load_sample("iris")
    
    # Get info
    info = loader.get_info(df)
    preview = loader.get_preview(df, n_rows=10)
```

Dependencies:
    • pandas
    • polars (optional)
    • sqlalchemy (for SQL support)
    • openpyxl/xlrd (for Excel)
    • pyarrow/fastparquet (for Parquet)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

import pandas as pd
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = ["DataLoader", "get_data_loader"]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SUPPORTED_FILE_EXTENSIONS = [
    ".csv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".parquet"
]

MAX_PREVIEW_ROWS = 100

COMPRESSION_MAP = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".zip": "zip",
    ".xz": "xz",
    ".zst": "zstd"
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Loader
# ═══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    📦 **Universal Data Loader**
    
    Loads data from multiple formats with smart detection.
    
    Supported Formats:
      • CSV (+ .csv.gz, .csv.bz2, etc.)
      • Excel (XLSX, XLS)
      • JSON (standard, JSONL, NDJSON)
      • Parquet
      • URLs (HTTP/HTTPS)
      • SQL databases
    
    Features:
      • Automatic format detection
      • Compression support
      • Smart delimiter detection
      • Encoding fallback
      • Pandas/Polars support
    
    Usage:
```python
        loader = DataLoader()
        
        # Load from file
        df = loader.load("data.csv")
        df = loader.load("data.csv.gz")
        df = loader.load("data.xlsx")
        
        # Load from URL
        df = loader.load_from_url("https://example.com/data.csv")
        
        # Load from SQL
        df = loader.load_from_sql("SELECT * FROM users")
```
    """
    
    def __init__(self, use_polars: bool = False):
        """
        Initialize data loader.
        
        Args:
            use_polars: Use Polars instead of Pandas (faster for large data)
        """
        self.use_polars = use_polars
        self.logger = logger.bind(component="DataLoader")
        
        # Check if polars available
        if use_polars:
            try:
                import polars as pl
                self._pl = pl
            except ImportError:
                self.logger.warning(
                    "Polars not available, falling back to Pandas"
                )
                self.use_polars = False
                self._pl = None
        else:
            self._pl = None
    
    # ───────────────────────────────────────────────────────────────────
    # Public API - Loading
    # ───────────────────────────────────────────────────────────────────
    
    def load(
        self,
        filepath: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Any]:
        """
        📂 **Load Data from File**
        
        Loads data with automatic format detection.
        
        Args:
            filepath: Path to data file
            file_type: File type (auto-detected if None)
            **kwargs: Format-specific arguments
        
        Returns:
            DataFrame (pandas or polars)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format unsupported
        
        Example:
```python
            df = loader.load("data.csv")
            df = loader.load("data.csv.gz")
            df = loader.load("data.xlsx", sheet_name="Sheet1")
            df = loader.load("data.json", orient="records")
```
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Normalize file type + compression
        norm_type, compression = self._normalize_file_type(path, file_type)
        
        # Validate format
        if norm_type not in SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {norm_type}. "
                f"Supported: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
            )
        
        self.logger.info(
            f"Loading data from {path} "
            f"(type: {norm_type}, compression: {compression or 'none'})"
        )
        
        # Load based on type
        if norm_type == ".csv":
            df = self._load_csv(path, compression=compression, **kwargs)
        elif norm_type in [".xlsx", ".xls"]:
            df = self._load_excel(path, **kwargs)
        elif norm_type == ".json":
            # Auto-detect JSONL/NDJSON by suffix
            is_lines = path.suffix.lower() in (".jsonl", ".ndjson")
            df = self._load_json(path, lines=is_lines, **kwargs)
        elif norm_type == ".parquet":
            df = self._load_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {norm_type}")
        
        self.logger.success(
            f"Data loaded: {len(df)} rows × {len(df.columns)} columns"
        )
        
        return df
    
    def load_from_url(
        self,
        url: str,
        file_type: str = ".csv",
        timeout: int = 30,
        **kwargs
    ) -> Union[pd.DataFrame, Any]:
        """
        🌐 **Load Data from URL**
        
        Downloads and loads data from HTTP/HTTPS URL.
        
        Args:
            url: URL to data file
            file_type: File type (.csv, .json, .parquet)
            timeout: HTTP timeout in seconds
            **kwargs: Format-specific arguments
        
        Returns:
            DataFrame
        
        Example:
```python
            df = loader.load_from_url(
                "https://example.com/data.csv",
                file_type=".csv"
            )
```
        """
        self._validate_url(url)
        self.logger.info(f"Loading from URL: {url} (type: {file_type})")
        
        if file_type == ".csv":
            if self.use_polars:
                return self._pl.read_csv(url, **kwargs)
            return pd.read_csv(
                url,
                on_bad_lines="skip",
                low_memory=False,
                **kwargs
            )
        
        if file_type in (".json", ".jsonl", ".ndjson"):
            if self.use_polars:
                if file_type in (".jsonl", ".ndjson"):
                    try:
                        return self._pl.read_ndjson(url, **kwargs)
                    except Exception:
                        # Fallback: requests
                        import requests
                        resp = requests.get(url, timeout=timeout)
                        resp.raise_for_status()
                        lines = [
                            json.loads(line)
                            for line in resp.text.splitlines()
                            if line.strip()
                        ]
                        return self._pl.DataFrame(lines)
                else:
                    import requests
                    resp = requests.get(url, timeout=timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    return self._pl.DataFrame(data)
            else:
                if file_type in (".jsonl", ".ndjson"):
                    return pd.read_json(url, lines=True, **kwargs)
                return pd.read_json(url, **kwargs)
        
        if file_type == ".parquet":
            if self.use_polars:
                return self._pl.read_parquet(url, **kwargs)
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
        🗄️ **Load Data from SQL**
        
        Executes SQL query or reads table.
        
        Args:
            query_or_table: SQL query or table name
            connection: SQLAlchemy connection/engine
            con_str: Connection string (if no connection)
            params: Query parameters
            limit: Optional row limit
            **kwargs: Additional read_sql arguments
        
        Returns:
            pandas DataFrame
        
        Example:
```python
            # Query
            df = loader.load_from_sql(
                "SELECT * FROM users WHERE active = :active",
                params={"active": True},
                limit=1000
            )
            
            # Table
            df = loader.load_from_sql("users", limit=100)
```
        """
        if self.use_polars:
            self.logger.warning(
                "SQL loading uses pandas; returning pandas DataFrame"
            )
        
        try:
            import sqlalchemy as sa
        except ImportError as e:
            raise ImportError(
                "SQL loading requires sqlalchemy. "
                "Install with: pip install sqlalchemy"
            ) from e
        
        engine = None
        
        if connection is None:
            if not con_str:
                # Try to get from settings
                try:
                    from config.settings import settings
                    con_str = settings.get_database_url()
                except Exception:
                    raise ValueError(
                        "Either connection or con_str must be provided"
                    )
            
            engine = sa.create_engine(con_str)
            connection = engine.connect()
        
        try:
            # Check if table name or query
            if self._looks_like_table_name(query_or_table):
                query = f"SELECT * FROM {query_or_table}"
                if limit:
                    query += f" LIMIT {int(limit)}"
                df = pd.read_sql(sa.text(query), connection, params=params, **kwargs)
            else:
                df = pd.read_sql(
                    sa.text(query_or_table),
                    connection,
                    params=params,
                    **kwargs
                )
            
            return df
        
        finally:
            if engine is not None:
                try:
                    connection.close()
                    engine.dispose()
                except Exception:
                    pass
    
    def load_sample(self, dataset_name: str) -> Union[pd.DataFrame, Any]:
        """
        📊 **Load Sample Dataset**
        
        Loads built-in sample dataset.
        
        Args:
            dataset_name: Name of sample (iris, titanic, house_prices)
        
        Returns:
            DataFrame
        
        Example:
```python
            df = loader.load_sample("iris")
```
        """
        try:
            from config.settings import settings
            base_path = settings.BASE_PATH
        except Exception:
            base_path = Path(".")
        
        sample_path = base_path / "data" / "samples" / f"{dataset_name}.csv"
        
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Sample dataset not found: {dataset_name}"
            )
        
        return self.load(sample_path)
    
    # ───────────────────────────────────────────────────────────────────
    # Public API - Saving
    # ───────────────────────────────────────────────────────────────────
    
    def save(
        self,
        df: Union[pd.DataFrame, Any],
        filepath: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        💾 **Save DataFrame to File**
        
        Saves data with automatic format detection.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            file_type: File type (auto-detected if None)
            **kwargs: Format-specific arguments
        
        Example:
```python
            loader.save(df, "output.csv")
            loader.save(df, "output.parquet")
            loader.save(df, "output.csv.gz")
```
        """
        filepath = Path(filepath)
        
        if file_type is None:
            file_type = filepath.suffix.lower()
        
        self.logger.info(f"Saving data to {filepath} (type: {file_type})")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Get underlying type if compressed
        if file_type in COMPRESSION_MAP:
            # Get base extension
            suffixes = [s.lower() for s in filepath.suffixes]
            if len(suffixes) >= 2:
                file_type = suffixes[-2]
        
        if file_type == ".csv":
            if self._is_polars_df(df):
                df.write_csv(filepath, **kwargs)
            else:
                df.to_csv(filepath, index=False, **kwargs)
        
        elif file_type in [".xlsx", ".xls"]:
            if self._is_polars_df(df):
                df.to_pandas().to_excel(filepath, index=False, **kwargs)
            else:
                df.to_excel(filepath, index=False, **kwargs)
        
        elif file_type == ".json":
            if self._is_polars_df(df):
                try:
                    df.write_json(filepath, **kwargs)
                except Exception:
                    import json
                    filepath.write_text(
                        json.dumps(
                            df.to_pandas().to_dict(orient="records"),
                            ensure_ascii=False
                        ),
                        encoding="utf-8"
                    )
            else:
                df.to_json(
                    filepath,
                    orient="records",
                    force_ascii=False,
                    **kwargs
                )
        
        elif file_type == ".parquet":
            if self._is_polars_df(df):
                df.write_parquet(filepath, **kwargs)
            else:
                try:
                    df.to_parquet(
                        filepath,
                        index=False,
                        engine="pyarrow",
                        **kwargs
                    )
                except Exception:
                    df.to_parquet(
                        filepath,
                        index=False,
                        engine="fastparquet",
                        **kwargs
                    )
        
        else:
            raise ValueError(f"Unsupported file type for saving: {file_type}")
        
        self.logger.success(f"Data saved to {filepath}")
    
    # ───────────────────────────────────────────────────────────────────
    # Public API - Utilities
    # ───────────────────────────────────────────────────────────────────
    
    def get_preview(
        self,
        df: Union[pd.DataFrame, Any],
        n_rows: int = MAX_PREVIEW_ROWS
    ) -> Union[pd.DataFrame, Any]:
        """
        👁️ **Get DataFrame Preview**
        
        Returns first N rows.
        
        Args:
            df: DataFrame
            n_rows: Number of rows
        
        Returns:
            Preview DataFrame
        """
        return df.head(n_rows)
    
    def get_info(self, df: Union[pd.DataFrame, Any]) -> Dict[str, Any]:
        """
        ℹ️ **Get DataFrame Information**
        
        Returns detailed metadata.
        
        Args:
            df: DataFrame
        
        Returns:
            Info dictionary
        
        Example:
```python
            info = loader.get_info(df)
            print(f"Rows: {info['n_rows']}")
            print(f"Columns: {info['n_columns']}")
            print(f"Memory: {info['memory_usage']} MB")
```
        """
        if self._is_polars_df(df):
            try:
                mem_bytes = df.estimated_size()
            except Exception:
                mem_bytes = None
            
            return {
                "n_rows": df.height,
                "n_columns": df.width,
                "columns": df.columns,
                "dtypes": {
                    col: str(dtype)
                    for col, dtype in zip(df.columns, df.dtypes)
                },
                "memory_usage": (
                    mem_bytes / 1024**2 if mem_bytes is not None else None
                )
            }
        else:
            return {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                "memory_usage": float(
                    df.memory_usage(deep=True).sum() / 1024**2
                ),
                "missing_values": {
                    k: int(v) for k, v in df.isnull().sum().to_dict().items()
                }
            }
    
    @staticmethod
    def to_pandas(df: Union[pd.DataFrame, Any]) -> pd.DataFrame:
        """
        🔄 **Convert to Pandas**
        
        Converts any DataFrame to pandas.
        """
        if hasattr(df, 'to_pandas'):
            return df.to_pandas()
        return df
    
    @staticmethod
    def to_polars(df: Union[pd.DataFrame, Any]) -> Any:
        """
        🔄 **Convert to Polars**
        
        Converts any DataFrame to polars.
        """
        try:
            import polars as pl
            
            if isinstance(df, pd.DataFrame):
                return pl.from_pandas(df)
            return df
        except ImportError:
            raise ImportError(
                "Polars not installed. Install with: pip install polars"
            )
    
    # ───────────────────────────────────────────────────────────────────
    # Internal - Format Loaders
    # ───────────────────────────────────────────────────────────────────
    
    def _load_csv(
        self,
        filepath: Path,
        encoding: str = "utf-8",
        delimiter: Optional[str] = None,
        compression: Optional[str] = None,
        sniff_delimiter: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, Any]:
        """Load CSV file with smart detection."""
        sep = delimiter
        
        # Sniff delimiter if requested
        if sep is None and sniff_delimiter:
            try:
                with open(filepath, "r", encoding=encoding, newline="") as f:
                    sample = f.read(64_000)
                
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
                
                self.logger.info(f"Detected delimiter: {repr(sep)}")
            except Exception:
                sep = ","
        
        try:
            if self.use_polars:
                return self._pl.read_csv(
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
                return self._pl.read_csv(
                    filepath,
                    separator=sep or ",",
                    encoding="latin-1",
                    **kwargs
                )
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
        """Load Excel file with engine fallback."""
        try:
            return pd.read_excel(
                filepath,
                sheet_name=sheet_name,
                engine=engine,
                **kwargs
            )
        except Exception as e:
            # Try different engines
            engines = ["openpyxl", "xlrd"]
            
            for eng in engines:
                try:
                    return pd.read_excel(
                        filepath,
                        sheet_name=sheet_name,
                        engine=eng,
                        **kwargs
                    )
                except Exception:
                    continue
            
            raise e
    
    def _load_json(
        self,
        filepath: Path,
        lines: Optional[bool] = None,
        orient: str = "records",
        **kwargs
    ) -> Union[pd.DataFrame, Any]:
        """Load JSON/JSONL/NDJSON."""
        if self.use_polars:
            if lines:
                try:
                    return self._pl.read_ndjson(filepath, **kwargs)
                except Exception:
                    # Fallback: manual
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = [
                            json.loads(line)
                            for line in f
                            if line.strip()
                        ]
                    return self._pl.DataFrame(data)
            else:
                try:
                    return self._pl.read_json(filepath, **kwargs)
                except Exception:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return self._pl.DataFrame(data)
        else:
            # Auto-detect lines format
            if lines is None:
                with open(filepath, "r", encoding="utf-8") as f:
                    head = "".join([next(f, "") for _ in range(5)]).strip()
                lines = "\n" in head and head.lstrip().startswith("{")
            
            return pd.read_json(
                filepath,
                lines=lines,
                orient=None if lines else orient,
                **kwargs
            )
    
    def _load_parquet(
        self,
        filepath: Path,
        **kwargs
    ) -> Union[pd.DataFrame, Any]:
        """Load Parquet file with engine fallback."""
        if self.use_polars:
            return self._pl.read_parquet(filepath, **kwargs)
        
        try:
            return pd.read_parquet(filepath, engine="pyarrow", **kwargs)
        except Exception:
            return pd.read_parquet(filepath, engine="fastparquet", **kwargs)
    
    # ───────────────────────────────────────────────────────────────────
    # Internal - Helpers
    # ───────────────────────────────────────────────────────────────────
    
    def _normalize_file_type(
        self,
        path: Path,
        explicit: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        """
        Normalize file type and detect compression.
        
        Returns:
            (normalized_type, compression)
        """
        if explicit:
            return explicit.lower(), None
        
        # Handle multi-suffix like .csv.gz
        suffixes = [s.lower() for s in path.suffixes]
        
        if not suffixes:
            return path.suffix.lower(), None
        
        compression = None
        
        # Check if last suffix is compression
        if suffixes[-1] in COMPRESSION_MAP:
            compression = COMPRESSION_MAP[suffixes[-1]]
            
            # Get base extension
            if len(suffixes) >= 2:
                base = suffixes[-2]
            else:
                base = ".csv"  # Default guess
        else:
            base = suffixes[-1]
        
        # Map JSONL/NDJSON to JSON
        if base in (".jsonl", ".ndjson"):
            base = ".json"
        
        return base, compression
    
    def _validate_url(self, url: str) -> None:
        """Validate URL scheme."""
        parsed = urlparse(url)
        
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP(S) URLs are supported")
    
    @staticmethod
    def _looks_like_table_name(text: str) -> bool:
        """Heuristic check if text is table name vs query."""
        t = text.strip().lower()
        
        return (
            " " not in t and
            ";" not in t and
            not t.startswith(("select", "with", "insert", "update", "delete"))
        )
    
    def _is_polars_df(self, df: Any) -> bool:
        """Check if DataFrame is Polars."""
        return self.use_polars and hasattr(df, 'write_csv')


# ═══════════════════════════════════════════════════════════════════════════
# Global Instance
# ═══════════════════════════════════════════════════════════════════════════

_data_loader: Optional[DataLoader] = None


def get_data_loader(use_polars: bool = False) -> DataLoader:
    """
    🏭 **Get Data Loader (Singleton)**
    
    Returns global data loader instance.
    
    Args:
        use_polars: Use Polars instead of Pandas
    
    Returns:
        DataLoader instance
    """
    global _data_loader
    
    if _data_loader is None:
        _data_loader = DataLoader(use_polars=use_polars)
    
    return _data_loader


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Data Loader v{__version__} - Self Test")
    print("="*80)
    
    # Initialize loader
    print("\n1. Initializing Loader...")
    loader = get_data_loader()
    print(f"   Engine: {'Polars' if loader.use_polars else 'Pandas'}")
    
    # Test CSV loading
    print("\n2. Testing CSV Loading...")
    try:
        # Create test CSV
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        ) as f:
            f.write("A,B,C\n1,2,3\n4,5,6\n")
            temp_path = f.name
        
        df = loader.load(temp_path)
        assert len(df) == 2, "CSV loading failed"
        print("   ✓ CSV loading works")
        
        # Cleanup
        Path(temp_path).unlink()
    except Exception as e:
        print(f"   ✗ CSV loading failed: {e}")
    
    # Test format detection
    print("\n3. Testing Format Detection...")
    try:
        formats = [
            ("test.csv", ".csv", None),
            ("test.csv.gz", ".csv", "gzip"),
            ("test.xlsx", ".xlsx", None),
            ("test.json", ".json", None),
            ("test.parquet", ".parquet", None)
        ]
        
        for filename, expected_type, expected_comp in formats:
            path = Path(filename)
            detected_type, detected_comp = loader._normalize_file_type(path, None)
            assert detected_type == expected_type, f"Type detection failed for {filename}"
            assert detected_comp == expected_comp, f"Compression detection failed for {filename}"
        
        print("   ✓ Format detection works")
    except Exception as e:
        print(f"   ✗ Format detection failed: {e}")
    
    # Test DataFrame info
    print("\n4. Testing DataFrame Info...")
    try:
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "C": [1.1, 2.2, 3.3]
        })
        
        info = loader.get_info(df)
        assert info["n_rows"] == 3, "Row count incorrect"
        assert info["n_columns"] == 3, "Column count incorrect"
        print(f"   ✓ Info extraction works")
        print(f"     Rows: {info['n_rows']}")
        print(f"     Columns: {info['n_columns']}")
        print(f"     Memory: {info['memory_usage']:.2f} MB")
    except Exception as e:
        print(f"   ✗ Info extraction failed: {e}")
    
    # Test preview
    print("\n5. Testing Preview...")
    try:
        df = pd.DataFrame({"A": range(200)})
        preview = loader.get_preview(df, n_rows=10)
        assert len(preview) == 10, "Preview size incorrect"
        print("   ✓ Preview works")
    except Exception as e:
        print(f"   ✗ Preview failed: {e}")
    
    # Test conversion
    print("\n6. Testing DataFrame Conversion...")
    try:
        df_pandas = pd.DataFrame({"A": [1, 2, 3]})
        
        # Pandas → Pandas
        converted = DataLoader.to_pandas(df_pandas)
        assert isinstance(converted, pd.DataFrame), "Pandas conversion failed"
        print("   ✓ Pandas conversion works")
        
        # Test Polars if available
        try:
            import polars as pl
            df_polars = DataLoader.to_polars(df_pandas)
            assert hasattr(df_polars, 'to_pandas'), "Polars conversion failed"
            print("   ✓ Polars conversion works")
        except ImportError:
            print("   ⚠ Polars not available, skipping")
    
    except Exception as e:
        print(f"   ✗ Conversion failed: {e}")
    
    # Test save/load
    print("\n7. Testing Save/Load...")
    try:
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"]
        })
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV
            csv_path = Path(tmpdir) / "test.csv"
            loader.save(df, csv_path)
            df_loaded = loader.load(csv_path)
            assert len(df_loaded) == 3, "CSV save/load failed"
            print("   ✓ CSV save/load works")
            
            # Test JSON
            json_path = Path(tmpdir) / "test.json"
            loader.save(df, json_path)
            df_loaded = loader.load(json_path)
            assert len(df_loaded) == 3, "JSON save/load failed"
            print("   ✓ JSON save/load works")
    
    except Exception as e:
        print(f"   ✗ Save/load failed: {e}")
    
    # Test delimiter sniffing
    print("\n8. Testing Delimiter Sniffing...")
    try:
        import tempfile
        
        # Create semicolon-delimited CSV
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        ) as f:
            f.write("A;B;C\n1;2;3\n4;5;6\n")
            temp_path = f.name
        
        df = loader._load_csv(Path(temp_path), sniff_delimiter=True)
        assert len(df.columns) == 3, "Delimiter sniffing failed"
        print("   ✓ Delimiter sniffing works")
        
        Path(temp_path).unlink()
    except Exception as e:
        print(f"   ✗ Delimiter sniffing failed: {e}")
    
    # Test table name detection
    print("\n9. Testing Table Name Detection...")
    try:
        assert loader._looks_like_table_name("users") == True
        assert loader._looks_like_table_name("SELECT * FROM users") == False
        assert loader._looks_like_table_name("my_table") == True
        assert loader._looks_like_table_name("WITH cte AS ...") == False
        print("   ✓ Table name detection works")
    except Exception as e:
        print(f"   ✗ Table name detection failed: {e}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from utils.data_loader import DataLoader, get_data_loader

# === Get Loader Instance ===
loader = get_data_loader()

# Use Polars for large datasets
loader_polars = get_data_loader(use_polars=True)

# === Load from File ===

# CSV
df = loader.load("data.csv")

# Compressed CSV
df = loader.load("data.csv.gz")

# Excel
df = loader.load("data.xlsx", sheet_name="Sheet1")

# JSON
df = loader.load("data.json")

# JSONL/NDJSON
df = loader.load("data.jsonl")

# Parquet
df = loader.load("data.parquet")

# Auto-detect format
df = loader.load("data.unknown_extension", file_type=".csv")

# === Load from URL ===

df = loader.load_from_url(
    "https://example.com/data.csv",
    file_type=".csv"
)

df = loader.load_from_url(
    "https://api.example.com/data.json",
    file_type=".json"
)

# === Load from SQL ===

# Query
df = loader.load_from_sql(
    "SELECT * FROM users WHERE active = :active",
    params={"active": True},
    limit=1000
)

# Table
df = loader.load_from_sql("users", limit=100)

# Custom connection
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host/db")
df = loader.load_from_sql(
    "SELECT * FROM orders",
    connection=engine
)

# === Load Sample Dataset ===

df_iris = loader.load_sample("iris")
df_titanic = loader.load_sample("titanic")

# === Save Data ===

# CSV
loader.save(df, "output.csv")

# Compressed CSV
loader.save(df, "output.csv.gz")

# Excel
loader.save(df, "output.xlsx")

# JSON
loader.save(df, "output.json")

# Parquet
loader.save(df, "output.parquet")

# === Get Info & Preview ===

# Get detailed info
info = loader.get_info(df)
print(f"Rows: {info['n_rows']}")
print(f"Columns: {info['n_columns']}")
print(f"Memory: {info['memory_usage']} MB")
print(f"Missing: {info['missing_values']}")

# Get preview
preview = loader.get_preview(df, n_rows=10)
print(preview)

# === Convert Between Engines ===

# Any DataFrame → Pandas
df_pandas = DataLoader.to_pandas(df)

# Any DataFrame → Polars
df_polars = DataLoader.to_polars(df_pandas)

# === Advanced CSV Loading ===

# Custom delimiter
df = loader.load("data.csv", delimiter=";")

# Auto-detect delimiter
df = loader.load("data.csv", sniff_delimiter=True)

# Custom encoding
df = loader.load("data.csv", encoding="latin-1")

# Skip bad lines
df = loader.load("data.csv", on_bad_lines="skip")

# === Advanced Excel Loading ===

# Specific sheet
df = loader.load("data.xlsx", sheet_name="Sheet2")

# Sheet by index
df = loader.load("data.xlsx", sheet_name=1)

# Skip rows
df = loader.load("data.xlsx", skiprows=3)

# === Advanced JSON Loading ===

# JSONL/NDJSON
df = loader.load("data.jsonl", lines=True)

# Nested JSON
df = loader.load("data.json", orient="records")

# === Error Handling ===

try:
    df = loader.load("missing.csv")
except FileNotFoundError as e:
    print(f"File not found: {e}")

try:
    df = loader.load("data.unknown")
except ValueError as e:
    print(f"Unsupported format: {e}")

# === Integration with Pipeline ===

from utils.data_loader import get_data_loader

class DataPipeline:
    def __init__(self):
        self.loader = get_data_loader()
    
    def load_and_process(self, filepath):
        # Load
        df = self.loader.load(filepath)
        
        # Get info
        info = self.loader.get_info(df)
        print(f"Loaded {info['n_rows']} rows")
        
        # Process
        df_processed = self.process(df)
        
        # Save
        self.loader.save(df_processed, "output.parquet")
        
        return df_processed

# === Multi-Format Support ===

loader = get_data_loader()

# Load from various sources
sources = [
    "sales.csv",
    "customers.xlsx",
    "products.json",
    "inventory.parquet",
    "https://api.example.com/data.csv"
]

dataframes = []
for source in sources:
    if source.startswith("http"):
        df = loader.load_from_url(source)
    else:
        df = loader.load(source)
    dataframes.append(df)

# Combine
import pandas as pd
combined = pd.concat(dataframes, ignore_index=True)
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)
