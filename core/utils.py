"""
DataGenius PRO - Utility Functions
Common utility functions used across the application
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pandas.api import types as ptypes  # future-proof dtype checks
from config.settings import settings


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging(name: str = "DataGenius") -> logger.__class__:
    """
    Setup logging for a module.

    Returns a loguru logger bound with given name.
    """
    from config.logging_config import get_logger
    return get_logger(name)


# ---------------------------------------------------------------------
# Session & hashing
# ---------------------------------------------------------------------
def generate_session_id() -> str:
    """Generate unique session ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    return f"{ts}_{rand}"


def _normalize_for_hash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dtypes for stable hashing:
    - datetimes -> int64 (ns)
    - categories -> strings
    - bool -> int8
    """
    out = df.copy()

    for col in out.columns:
        s = out[col]
        # datetime -> int64 ns (preserves NaT as NaN)
        if ptypes.is_datetime64_any_dtype(s):
            out[col] = s.view("int64")
        # categorical -> string
        elif ptypes.is_categorical_dtype(s):
            out[col] = s.astype("string")
        # boolean -> int8
        elif ptypes.is_bool_dtype(s):
            out[col] = s.astype("Int8")
        # objects: leave as-is; hash_pandas_object poradzi sobie
    return out


def hash_dataframe(
    df: pd.DataFrame,
    *,
    sort_columns: bool = True,
    include_index: bool = True,
    normalize_dtypes: bool = True
) -> str:
    """
    Generate a stable MD5 hash for a DataFrame.

    Args:
        df: DataFrame to hash
        sort_columns: Sort columns for order-invariant hash
        include_index: Include index in the hash
        normalize_dtypes: Normalize types (datetimes, categoricals, bools) for stability

    Returns:
        MD5 hash hex string
    """
    if df is None or df.empty:
        return hashlib.md5(b"").hexdigest()

    work = _normalize_for_hash(df) if normalize_dtypes else df

    if sort_columns:
        work = work.reindex(sorted(work.columns), axis=1)

    # Important: include dtype + column names in the signature
    meta_blob = json.dumps(
        {
            "columns": list(map(str, work.columns)),
            "dtypes": [str(t) for t in work.dtypes],
            "shape": work.shape,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode()

    values_hash = pd.util.hash_pandas_object(work, index=include_index).values
    md5 = hashlib.md5()
    md5.update(meta_blob)
    md5.update(values_hash)
    return md5.hexdigest()


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """Save object to pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded object from {filepath}")
    return obj


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load dictionary from JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
def format_bytes(bytes_size: int) -> str:
    """Format bytes to human-readable string."""
    size = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} EB"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage. Accepts either fraction (0-1) or percent (0-100).
    Negative values preserved.
    """
    v = float(value)
    if -1.0 <= v <= 1.0:
        v *= 100.0
    return f"{v:.{decimals}f}%"


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """Format number with thousands separator."""
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):,.{decimals}f}"


# ---------------------------------------------------------------------
# Type detection & inference
# ---------------------------------------------------------------------
def detect_column_type(series: pd.Series) -> str:
    """
    Detect semantic type of column: numeric, categorical, datetime, text, id, unknown.
    """
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    if ptypes.is_numeric_dtype(series):
        if series.nunique(dropna=True) == len(series.dropna()) and series.name and "id" in str(series.name).lower():
            return "id"
        return "numeric"
    if ptypes.is_categorical_dtype(series) or series.dtype == "object":
        n_unique = series.nunique(dropna=True)
        n_total = series.size
        if n_total > 0 and (n_unique / n_total) > 0.9:
            return "text"
        return "categorical"
    return "unknown"


def infer_problem_type(target: pd.Series, threshold: int = 20) -> str:
    """
    Infer problem type ("classification" or "regression") from target column.
    """
    target_clean = target.dropna()
    n_unique = target_clean.nunique()
    if ptypes.is_numeric_dtype(target_clean):
        return "classification" if n_unique <= threshold else "regression"
    return "classification"


# ---------------------------------------------------------------------
# DataFrame split & info
# ---------------------------------------------------------------------
def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """Get detailed memory usage of DataFrame."""
    memory_by_column = df.memory_usage(deep=True)
    total_memory = memory_by_column.sum()
    return {
        "total_mb": total_memory / 1024**2,
        "by_column": {col: mem / 1024**2 for col, mem in memory_by_column.items()},
        "per_row_bytes": (total_memory / len(df)) if len(df) > 0 else 0,
    }


def df_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compact overview of a DataFrame (useful for logs/UI)."""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "missing_pct": float(df.isna().sum().sum() / (df.size) * 100) if df.size else 0.0,
    }


# ---------------------------------------------------------------------
# Memory optimization
# ---------------------------------------------------------------------
def reduce_memory_usage(
    df: pd.DataFrame,
    verbose: bool = True,
    convert_categoricals: bool = True,
    max_cat_unique: int = 50,
    cat_ratio_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numerics and optionally converting objects to category.

    - Integers (no NaN) -> smallest int/uint
    - Floats -> float32 if fits
    - Bool -> Int8
    - Objects with low cardinality -> category (optional)
    """
    before = df.memory_usage(deep=True).sum() / 1024**2
    out = df.copy()

    for col in out.columns:
        s = out[col]

        # Booleans -> Int8 (nullable-safe)
        if ptypes.is_bool_dtype(s):
            out[col] = s.astype("Int8")
            continue

        # Integers (nullable-safe)
        if ptypes.is_integer_dtype(s):
            # Pandas integer dtypes are already compact; skip explicit downcast
            out[col] = pd.to_numeric(s, downcast="integer")
            continue

        # Floats
        if ptypes.is_float_dtype(s):
            c_min, c_max = s.min(skipna=True), s.max(skipna=True)
            try:
                if np.isfinite([c_min, c_max]).all() and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    out[col] = s.astype(np.float32)
            except Exception:
                pass
            continue

        # Objects -> category (optional)
        if convert_categoricals and s.dtype == object:
            nunique = s.nunique(dropna=True)
            if nunique > 0:
                ratio = nunique / max(1, len(s))
                if nunique <= max_cat_unique and ratio <= cat_ratio_threshold:
                    out[col] = s.astype("category")

    after = out.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        pct = (before - after) / before * 100 if before > 0 else 0.0
        logger.info(f"Memory usage reduced from {before:.2f} MB to {after:.2f} MB ({pct:.1f}% reduction)")

    return out


# ---------------------------------------------------------------------
# Cleaning & sampling
# ---------------------------------------------------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names (trim, lower, spaces->_, strip non-alnum, ensure uniqueness).
    """
    out = df.copy()
    cols = (
        pd.Index(out.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    # Ensure uniqueness while preserving order
    seen: Dict[str, int] = {}
    uniq = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            uniq.append(c)
        else:
            seen[c] += 1
            uniq.append(f"{c}_{seen[c]}")
    out.columns = uniq
    return out


def sample_dataframe(
    df: pd.DataFrame,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample DataFrame (mutually exclusive n / frac).
    """
    if (n is None and frac is None) or (n is not None and frac is not None):
        raise ValueError("Provide either n or frac (exclusively).")
    if frac is not None and not (0 < frac <= 1):
        raise ValueError("frac must be in (0, 1].")
    if n is not None:
        n = min(int(n), len(df))
        return df.sample(n=n, random_state=random_state)
    return df.sample(frac=frac, random_state=random_state)


# ---------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------
def get_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> List[str]:
    """Get list of categorical columns by dtype and uniqueness."""
    cols: List[str] = []
    for col in df.columns:
        s = df[col]
        if s.dtype == "object" or ptypes.is_categorical_dtype(s):
            if s.nunique(dropna=True) <= max_unique:
                cols.append(col)
    return cols


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def top_n_categories(series: pd.Series, n: int = 10) -> pd.DataFrame:
    """Return top-N categories with counts and percentages."""
    vc = series.value_counts(dropna=False).head(n)
    pct = (vc / max(1, len(series))) * 100
    return pd.DataFrame({"value": vc.index.astype(str), "count": vc.values, "pct": pct.values})


# ---------------------------------------------------------------------
# Converters & parsing
# ---------------------------------------------------------------------
def ensure_datetime(series: pd.Series, utc: bool = False, errors: str = "coerce") -> pd.Series:
    """Parse a Series to datetime with robust defaults."""
    s = pd.to_datetime(series, errors=errors, utc=utc, infer_datetime_format=True)
    return s


def safe_to_numeric(series: pd.Series, downcast: Optional[str] = None) -> pd.Series:
    """
    Convert Series to numeric; non-convertible values become NaN.
    downcast in {"integer", "signed", "unsigned", "float"} if desired.
    """
    s = pd.to_numeric(series, errors="coerce", downcast=downcast)
    return s


# ---------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------
def timer(func):
    """Decorator to time function execution."""
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
