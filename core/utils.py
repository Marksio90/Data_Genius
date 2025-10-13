"""
DataGenius PRO - Utility Functions
Common utility functions used across the application
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import hashlib
import json
from datetime import datetime
import pickle
from loguru import logger
from config.settings import settings


def setup_logging(name: str = "DataGenius") -> logger:
    """
    Setup logging for a module
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    from config.logging_config import get_logger
    return get_logger(name)


def generate_session_id() -> str:
    """Generate unique session ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    return f"{timestamp}_{random_suffix}"


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Generate hash of DataFrame for caching
    
    Args:
        df: DataFrame to hash
    
    Returns:
        MD5 hash string
    """
    # Use pandas hash function
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    return df_hash


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file
    
    Args:
        obj: Object to save
        filepath: Output filepath
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    
    logger.info(f"Saved object to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file
    
    Args:
        filepath: Pickle filepath
    
    Returns:
        Loaded object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    
    logger.info(f"Loaded object from {filepath}")
    return obj


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output filepath
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load dictionary from JSON file
    
    Args:
        filepath: JSON filepath
    
    Returns:
        Loaded dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON from {filepath}")
    return data


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes to human-readable string
    
    Args:
        bytes_size: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage
    
    Args:
        value: Value (0-1 or 0-100)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    if value <= 1:
        value *= 100
    return f"{value:.{decimals}f}%"


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format number with thousands separator
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted number string
    """
    if isinstance(value, int):
        return f"{value:,}"
    else:
        return f"{value:,.{decimals}f}"


def detect_column_type(series: pd.Series) -> str:
    """
    Detect semantic type of column
    
    Args:
        series: Pandas Series
    
    Returns:
        Column type: numeric, categorical, datetime, text, id
    """
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Check for numeric
    if pd.api.types.is_numeric_dtype(series):
        # Check if it's likely an ID column
        if series.nunique() == len(series) and series.name and "id" in series.name.lower():
            return "id"
        return "numeric"
    
    # Check for categorical
    if pd.api.types.is_categorical_dtype(series) or series.dtype == "object":
        n_unique = series.nunique()
        n_total = len(series)
        
        # If almost all values are unique, it's likely text/ID
        if n_unique / n_total > 0.9:
            return "text"
        else:
            return "categorical"
    
    return "unknown"


def infer_problem_type(
    target: pd.Series,
    threshold: int = 20
) -> str:
    """
    Infer problem type from target column
    
    Args:
        target: Target column
        threshold: Max unique values for classification
    
    Returns:
        "classification" or "regression"
    """
    n_unique = target.nunique()
    
    # Remove missing values
    target_clean = target.dropna()
    
    # Check if numeric
    if pd.api.types.is_numeric_dtype(target_clean):
        # If few unique values, likely classification
        if n_unique <= threshold:
            return "classification"
        else:
            return "regression"
    else:
        # Non-numeric is always classification
        return "classification"


def split_features_target(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
    
    Returns:
        Tuple of (X, y)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get detailed memory usage of DataFrame
    
    Args:
        df: DataFrame
    
    Returns:
        Dictionary with memory info
    """
    memory_by_column = df.memory_usage(deep=True)
    total_memory = memory_by_column.sum()
    
    return {
        "total_mb": total_memory / 1024**2,
        "by_column": {
            col: mem / 1024**2
            for col, mem in memory_by_column.items()
        },
        "per_row_bytes": total_memory / len(df) if len(df) > 0 else 0,
    }


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of DataFrame by downcasting numeric types
    
    Args:
        df: Input DataFrame
        verbose: Print memory reduction info
    
    Returns:
        DataFrame with reduced memory usage
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        logger.info(
            f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
            f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names (remove special characters, lowercase, etc.)
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with cleaned column names
    """
    df = df.copy()
    
    # Remove special characters and convert to lowercase
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    
    # Handle duplicate column names
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            f"{dup}_{i}" if i != 0 else dup
            for i in range(sum(cols == dup))
        ]
    df.columns = cols
    
    return df


def sample_dataframe(
    df: pd.DataFrame,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample DataFrame
    
    Args:
        df: Input DataFrame
        n: Number of samples
        frac: Fraction of samples (0-1)
        random_state: Random seed
    
    Returns:
        Sampled DataFrame
    """
    if n is not None:
        n = min(n, len(df))
        return df.sample(n=n, random_state=random_state)
    elif frac is not None:
        return df.sample(frac=frac, random_state=random_state)
    else:
        raise ValueError("Either n or frac must be specified")


def get_categorical_columns(
    df: pd.DataFrame,
    max_unique: int = 50
) -> List[str]:
    """
    Get list of categorical columns
    
    Args:
        df: Input DataFrame
        max_unique: Maximum unique values for categorical
    
    Returns:
        List of categorical column names
    """
    categorical = []
    
    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            if df[col].nunique() <= max_unique:
                categorical.append(col)
    
    return categorical


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numeric columns
    
    Args:
        df: Input DataFrame
    
    Returns:
        List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def timer(func):
    """Decorator to time function execution"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} executed in {end - start:.2f}s")
        return result
    
    return wrapper