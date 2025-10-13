"""
DataGenius PRO - Data Loader
Universal data loader supporting multiple formats
"""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Union, Optional, Literal, Dict, Any
from loguru import logger
import json
from config.settings import settings
from config.constants import (
    SUPPORTED_FILE_EXTENSIONS,
    MAX_PREVIEW_ROWS,
)


class DataLoader:
    """
    Universal data loader with support for:
    - CSV, Excel (XLSX, XLS)
    - JSON, Parquet
    - SQL databases
    """
    
    def __init__(self, use_polars: bool = False):
        """
        Initialize data loader
        
        Args:
            use_polars: Use Polars instead of Pandas (faster for large datasets)
        """
        self.use_polars = use_polars
        self.logger = logger.bind(component="DataLoader")
    
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
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Auto-detect file type
        if file_type is None:
            file_type = filepath.suffix.lower()
        
        # Validate file type
        if file_type not in SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {file_type}. "
                f"Supported: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
            )
        
        self.logger.info(f"Loading data from {filepath} (type: {file_type})")
        
        # Load based on file type
        if file_type == ".csv":
            df = self._load_csv(filepath, **kwargs)
        elif file_type in [".xlsx", ".xls"]:
            df = self._load_excel(filepath, **kwargs)
        elif file_type == ".json":
            df = self._load_json(filepath, **kwargs)
        elif file_type == ".parquet":
            df = self._load_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        self.logger.success(
            f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns"
        )
        
        return df
    
    def _load_csv(
        self,
        filepath: Path,
        encoding: str = "utf-8",
        delimiter: str = ",",
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Load CSV file"""
        
        try:
            if self.use_polars:
                return pl.read_csv(
                    filepath,
                    encoding=encoding,
                    separator=delimiter,
                    **kwargs
                )
            else:
                return pd.read_csv(
                    filepath,
                    encoding=encoding,
                    sep=delimiter,
                    **kwargs
                )
        except UnicodeDecodeError:
            # Try different encodings
            self.logger.warning(f"Failed with {encoding}, trying latin-1")
            if self.use_polars:
                return pl.read_csv(filepath, encoding="latin-1", **kwargs)
            else:
                return pd.read_csv(filepath, encoding="latin-1", **kwargs)
    
    def _load_excel(
        self,
        filepath: Path,
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """Load Excel file (only pandas supports Excel)"""
        
        return pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            **kwargs
        )
    
    def _load_json(
        self,
        filepath: Path,
        orient: str = "records",
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Load JSON file"""
        
        if self.use_polars:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pl.DataFrame(data)
        else:
            return pd.read_json(
                filepath,
                orient=orient,
                **kwargs
            )
    
    def _load_parquet(
        self,
        filepath: Path,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Load Parquet file"""
        
        if self.use_polars:
            return pl.read_parquet(filepath, **kwargs)
        else:
            return pd.read_parquet(filepath, **kwargs)
    
    def load_from_url(
        self,
        url: str,
        file_type: str = ".csv",
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load data from URL
        
        Args:
            url: URL to data file
            file_type: File type
            **kwargs: Additional arguments
        
        Returns:
            DataFrame
        """
        
        self.logger.info(f"Loading data from URL: {url}")
        
        if file_type == ".csv":
            if self.use_polars:
                return pl.read_csv(url, **kwargs)
            else:
                return pd.read_csv(url, **kwargs)
        elif file_type == ".json":
            if self.use_polars:
                import requests
                data = requests.get(url).json()
                return pl.DataFrame(data)
            else:
                return pd.read_json(url, **kwargs)
        else:
            raise ValueError(f"URL loading not supported for {file_type}")
    
    def load_sample(self, dataset_name: str) -> pd.DataFrame:
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
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file type
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
                df.write_json(filepath, **kwargs)
            else:
                df.to_json(filepath, orient="records", **kwargs)
        
        elif file_type == ".parquet":
            if isinstance(df, pl.DataFrame):
                df.write_parquet(filepath, **kwargs)
            else:
                df.to_parquet(filepath, index=False, **kwargs)
        
        else:
            raise ValueError(f"Unsupported file type for saving: {file_type}")
        
        self.logger.success(f"Data saved to {filepath}")
    
    def get_preview(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        n_rows: int = MAX_PREVIEW_ROWS
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Get preview of DataFrame
        
        Args:
            df: DataFrame
            n_rows: Number of rows to preview
        
        Returns:
            Preview DataFrame
        """
        
        if isinstance(df, pl.DataFrame):
            return df.head(n_rows)
        else:
            return df.head(n_rows)
    
    def get_info(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
        """
        Get detailed information about DataFrame
        
        Args:
            df: DataFrame
        
        Returns:
            Dictionary with DataFrame info
        """
        
        if isinstance(df, pl.DataFrame):
            return {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": df.columns,
                "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                "memory_usage": df.estimated_size("mb"),
            }
        else:
            return {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
                "missing_values": df.isnull().sum().to_dict(),
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


# Global loader instance
_data_loader: Optional[DataLoader] = None


def get_data_loader(use_polars: bool = False) -> DataLoader:
    """Get global data loader instance"""
    global _data_loader
    
    if _data_loader is None:
        _data_loader = DataLoader(use_polars=use_polars)
    
    return _data_loader