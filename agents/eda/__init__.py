# agents/eda/__init__.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — EDA Package                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Exploratory Data Analysis module with lazy exports & lightweight helpers: ║
║    ✓ EDAExplorer (advanced exploratory analysis)                          ║
║    ✓ EDAReporter (professional EDA reports)                               ║
║    ✓ DataQualityAgent (quality assessment)                                ║
║    ✓ quick_overview() — fast dataset snapshot                             ║
║    ✓ memory_usage() — per-column memory breakdown                         ║
║    ✓ Lazy loading (imports only when needed)                              ║
║    ✓ Zero external dependencies for helpers                               ║
╚════════════════════════════════════════════════════════════════════════════╝

Quick Start:
    from agents.eda import quick_overview, memory_usage
    overview = quick_overview(df, max_cols=50)
    mem = memory_usage(df)
    
    from agents.eda import EDAReporter  # Lazy import (on first use)
    reporter = EDAReporter()
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple, Any, Optional
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Lazy Exports Registry
# ═══════════════════════════════════════════════════════════════════════════

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "EDAExplorer": ("agents.eda.explorer", "EDAExplorer"),
    "EDAReporter": ("agents.eda.reporter", "EDAReporter"),
    "DataQualityAgent": ("agents.eda.quality", "DataQualityAgent"),
}

__all__ = tuple(list(_LAZY_EXPORTS.keys()) + ["quick_overview", "memory_usage"])


def __getattr__(name: str) -> Any:
    """
    Lazy attribute resolution for optional EDA components.
    
    Allows importing classes without loading heavy modules until first use.
    
    Example:
        from agents.eda import EDAReporter  # No import until this line
        reporter = EDAReporter()  # Actual loading happens here
    """
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        try:
            module: ModuleType = import_module(mod_name)
        except ImportError as e:
            raise AttributeError(
                f"Optional EDA component '{name}' not available\n"
                f"  → Failed to import '{mod_name}'\n"
                f"  → Original error: {e}"
            ) from e
        except Exception as e:
            raise AttributeError(
                f"Optional EDA component '{name}' not available\n"
                f"  → Unexpected error loading '{mod_name}': {type(e).__name__}: {e}"
            ) from e
        
        try:
            obj = getattr(module, symbol)
        except AttributeError as e:
            raise AttributeError(
                f"Optional EDA component '{name}' not available\n"
                f"  → Symbol '{symbol}' not found in '{mod_name}'\n"
                f"  → Available: {dir(module)[:5]}..."
            ) from e
        
        # Cache for future access
        globals()[name] = obj
        return obj
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list:
    """Provide comprehensive attribute listing."""
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Lightweight Helpers (No External Dependencies)
# ═══════════════════════════════════════════════════════════════════════════

def memory_usage(df: pd.DataFrame, unit: str = "mb") -> pd.Series:
    """
    Calculate memory usage per column with automatic totaling.
    
    Features:
      • Per-column breakdown (deep counting for strings/categories)
      • Automatic '__TOTAL__' row for quick reference
      • Configurable units (mb, kb, bytes)
      • Defensive: handles empty DataFrames gracefully
      • Zero side-effects on input DataFrame
    
    Args:
        df: Input DataFrame
        unit: Output unit ("mb", "kb", or "bytes")
    
    Returns:
        pd.Series with per-column usage + '__TOTAL__' row
    
    Example:
        >>> mem = memory_usage(df)
        >>> print(mem.tail(3))
        column_a       2.345
        column_b       1.234
        __TOTAL__      3.579
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    
    # Empty DataFrame handling
    if df.empty:
        empty_result = pd.Series({"__TOTAL__": 0.0}, dtype=float)
        return empty_result
    
    # Unit conversion factors
    unit_lower = unit.lower().strip()
    if unit_lower == "bytes":
        divisor = 1
        unit_suffix = " bytes"
    elif unit_lower == "kb":
        divisor = 1024
        unit_suffix = " KB"
    else:  # default to MB
        divisor = 1024 ** 2
        unit_suffix = " MB"
    
    # Get per-column memory usage (deep=True for accurate string counting)
    try:
        per_col = df.memory_usage(deep=True)
    except Exception:
        per_col = df.memory_usage(deep=False)
    
    # Convert to string index (handle int/mixed index types)
    per_col.index = per_col.index.astype(str)
    
    # Convert to requested unit
    per_col_converted = (per_col / divisor).astype(float).round(3)
    
    # Compute total
    total_value = float(per_col_converted.sum())
    
    # Add total row
    total_row = pd.Series({"__TOTAL__": total_value}, dtype=float)
    
    return pd.concat([per_col_converted, total_row])


def quick_overview(
    df: pd.DataFrame,
    max_cols: int = 50,
    include_numeric_stats: bool = True,
    include_categorical_top: bool = True
) -> Dict[str, Any]:
    """
    Fast high-level dataset overview with defensive error handling.
    
    Computes:
      • Shape & basic metadata
      • Data types distribution
      • Missing value percentages
      • Cardinality (unique values)
      • Numeric statistics (for numeric columns)
      • Categorical distributions (top 5 values)
      • Memory footprint
    
    Args:
        df: Input DataFrame
        max_cols: Maximum columns to analyze (safety limit)
        include_numeric_stats: Include mean/std/min/max for numerics
        include_categorical_top: Include top-5 values for categories
    
    Returns:
        Dictionary with overview data (JSON-serializable)
    
    Example:
        >>> overview = quick_overview(df)
        >>> print(overview["shape"])
        {'rows': 1000, 'cols': 25}
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        return {"error": f"Expected pd.DataFrame, got {type(df).__name__}"}
    
    if df.empty:
        return {"error": "DataFrame is empty"}
    
    n_rows, n_cols_total = df.shape
    
    # Column subset (safety limit)
    max_cols_safe = max(0, int(max_cols))
    if max_cols_safe == 0:
        return {"error": "max_cols must be > 0"}
    
    cols_to_analyze = list(df.columns[:max_cols_safe])
    df_sample = df[cols_to_analyze]
    
    # ─── Basic Metadata
    metadata = {
        "shape": {
            "rows": int(n_rows),
            "cols": int(n_cols_total),
            "analyzed_cols": len(cols_to_analyze),
        },
        "preview_cols": [str(c) for c in cols_to_analyze],
    }
    
    # ─── Data Types
    try:
        dtypes_dict = {str(c): str(t) for c, t in df_sample.dtypes.items()}
        metadata["dtypes"] = dtypes_dict
    except Exception as e:
        metadata["dtypes"] = {"error": str(e)[:100]}
    
    # ─── Missing Data
    try:
        missing_pct = (df_sample.isna().sum() / max(1, n_rows) * 100).round(2)
        metadata["missing_pct"] = {str(c): float(v) for c, v in missing_pct.items()}
    except Exception as e:
        metadata["missing_pct"] = {"error": str(e)[:100]}
    
    # ─── Cardinality
    try:
        nunique = df_sample.nunique(dropna=True)
        metadata["cardinality"] = {str(c): int(v) for c, v in nunique.items()}
    except Exception as e:
        metadata["cardinality"] = {"error": str(e)[:100]}
    
    # ─── Numeric Statistics
    numeric_stats = {}
    if include_numeric_stats:
        try:
            num_cols = df_sample.select_dtypes(include=[np.number]).columns
            
            if len(num_cols) > 0:
                # Use describe for quick stats
                try:
                    desc = df_sample[num_cols].describe(
                        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                        include="all"
                    ).round(6).T
                    
                    numeric_stats = {
                        str(c): {
                            str(k): (None if pd.isna(v) else float(v))
                            for k, v in row.items()
                        }
                        for c, row in desc.to_dict(orient="index").items()
                    }
                except Exception:
                    # Fallback: minimal stats
                    for c in num_cols:
                        try:
                            s = pd.to_numeric(df_sample[c], errors="coerce")
                            s = s.replace([np.inf, -np.inf], np.nan).dropna()
                            
                            if len(s) > 0:
                                numeric_stats[str(c)] = {
                                    "count": int(s.count()),
                                    "mean": round(float(s.mean()), 6),
                                    "std": round(float(s.std(ddof=1) if len(s) > 1 else 0.0), 6),
                                    "min": round(float(s.min()), 6),
                                    "25%": round(float(s.quantile(0.25)), 6),
                                    "50%": round(float(s.median()), 6),
                                    "75%": round(float(s.quantile(0.75)), 6),
                                    "max": round(float(s.max()), 6),
                                }
                            else:
                                numeric_stats[str(c)] = {"count": 0}
                        except Exception:
                            numeric_stats[str(c)] = {"error": "stats_failed"}
        
        except Exception as e:
            numeric_stats = {"error": str(e)[:100]}
    
    metadata["numeric_stats"] = numeric_stats
    
    # ─── Categorical Top Values
    categorical_top = {}
    if include_categorical_top:
        try:
            cat_cols = df_sample.select_dtypes(
                include=["object", "category", "string"]
            ).columns
            
            for c in cat_cols:
                try:
                    vc = df_sample[c].value_counts(dropna=False).head(5)
                    categorical_top[str(c)] = {str(k): int(v) for k, v in vc.items()}
                except Exception:
                    categorical_top[str(c)] = {}
        
        except Exception as e:
            categorical_top = {"error": str(e)[:100]}
    
    metadata["categorical_top_values"] = categorical_top
    
    # ─── Memory Usage
    try:
        mem_total = memory_usage(df_sample)["__TOTAL__"]
        metadata["memory_mb"] = float(round(mem_total, 2))
    except Exception:
        try:
            mem_bytes = df_sample.memory_usage(deep=True).sum()
            metadata["memory_mb"] = float(round(mem_bytes / (1024 ** 2), 2))
        except Exception:
            metadata["memory_mb"] = None
    
    return metadata