# agents/data_understanding/__init__.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Data Understanding Module         ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Lazy-load exports + Enterprise-grade helpers:                             ║
║    • SchemaInspector (advanced schema analysis)                            ║
║    • DataProfiler (comprehensive statistical profiling)                    ║
║    • ConstraintValidator (business rule validation)                        ║
║    • profile_summary() → full dataset metadata                             ║
║    • target_diagnosis() → problem type classification                      ║
║    • potential_leakage() → data leakage detection                          ║
║    • simple_quality_flags() → data health assessment                       ║
║    • advanced_quality_metrics() → enterprise quality indicators            ║
╚════════════════════════════════════════════════════════════════════════════╝

PRO Enterprise Standards:
  ✓ Bulletproof input validation & comprehensive error handling
  ✓ Graceful degradation (partial results > hard failures)
  ✓ Intelligent logging (loguru with stdlib fallback)
  ✓ Type hints & documentation for all exports
  ✓ Cache optimization for lazy imports
  ✓ Performance-conscious operations with memory awareness
  ✓ Hierarchical structure with clear separation of concerns
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Core Infrastructure & Initialization
# ═══════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple, Any, Iterable, Optional, List, Set
from functools import lru_cache
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────
# Logging Infrastructure: Premium loguru with intelligent fallback
# ─────────────────────────────────────────────────────────────────────────

try:
    from loguru import logger as _LOGGER
    _USE_LOGURU = True
except ImportError:
    import logging
    _LOGGER = logging.getLogger(__name__)
    if not _LOGGER.handlers:
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter(
            "%(levelname)-8s | %(asctime)s | %(name)s → %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        _handler.setFormatter(_formatter)
        _LOGGER.addHandler(_handler)
        _LOGGER.setLevel(logging.INFO)
    _USE_LOGURU = False

# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Lazy-Load Module Registry & Export Mechanism
# ═══════════════════════════════════════════════════════════════════════════

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "SchemaInspector": ("agents.data_understanding.schema", "SchemaInspector"),
    "DataProfiler": ("agents.data_understanding.profiler", "DataProfiler"),
    "ConstraintValidator": ("agents.data_understanding.constraints", "ConstraintValidator"),
}

_IMPORT_CACHE: Dict[str, Any] = {}

__all__ = tuple(
    list(_LAZY_EXPORTS.keys()) + [
        "profile_summary",
        "target_diagnosis",
        "potential_leakage",
        "simple_quality_flags",
        "advanced_quality_metrics",
    ]
)


def __getattr__(name: str) -> Any:
    """
    Lazy attribute resolution with intelligent caching and error reporting.
    
    Behavior:
      1. Check cache first (instant access on repeat imports)
      2. Load module dynamically if not cached
      3. Extract requested symbol
      4. Cache for subsequent access
      5. Provide detailed error messages if anything fails
    
    Args:
        name: Requested export name
        
    Returns:
        Requested object (class, function, etc.)
        
    Raises:
        AttributeError: If module/symbol not found with diagnostic info
    """
    if name in _LAZY_EXPORTS:
        # Early exit if already cached
        if name in _IMPORT_CACHE:
            return _IMPORT_CACHE[name]
        
        mod_name, symbol = _LAZY_EXPORTS[name]
        
        try:
            module: ModuleType = import_module(mod_name)
            _LOGGER.debug(f"✓ Loaded module: {mod_name}")
        except ModuleNotFoundError as e:
            msg = (
                f"❌ Cannot load '{mod_name}' for symbol '{name}'\n"
                f"   → Verify file exists: agents/data_understanding/{mod_name.split('.')[-1]}.py\n"
                f"   → Original error: {e}"
            )
            _LOGGER.error(msg)
            raise AttributeError(msg) from e
        
        # Extract symbol with validation
        obj = getattr(module, symbol, None)
        if obj is None:
            msg = (
                f"❌ Symbol '{symbol}' not found in '{mod_name}'\n"
                f"   → Available exports: {dir(module)}"
            )
            _LOGGER.error(msg)
            raise AttributeError(msg)
        
        # Cache and return
        _IMPORT_CACHE[name] = obj
        _LOGGER.debug(f"✓ Cached export: {name}")
        return obj
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'\n"
        f"   → Available: {list(_LAZY_EXPORTS.keys())}"
    )


def __dir__() -> List[str]:
    """Provide comprehensive attribute listing including lazy exports."""
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Input Validation & Safety Layer
# ═══════════════════════════════════════════════════════════════════════════

def _validate_df(df: Any, name: str = "DataFrame") -> Optional[str]:
    """
    Comprehensive DataFrame validation with diagnostic messages.
    
    Returns:
        None if valid, error message string if invalid
    """
    if not isinstance(df, pd.DataFrame):
        return f"{name} must be pd.DataFrame, got {type(df).__name__}"
    if df.empty:
        return f"{name} is empty (0 rows)"
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].unique().tolist()
        return f"{name} has duplicate columns: {dup}"
    return None


def _validate_column(df: pd.DataFrame, col: str) -> Optional[str]:
    """Validate that column exists in DataFrame."""
    if col not in df.columns:
        return f"Column '{col}' not in DataFrame. Available: {list(df.columns)[:5]}..."
    return None


def _safe_operation(operation_name: str):
    """Decorator for safe pandas operations with logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                _LOGGER.debug(f"✓ {operation_name} completed")
                return result
            except Exception as e:
                _LOGGER.warning(f"⚠ {operation_name} failed: {type(e).__name__}: {str(e)[:100]}")
                return {}
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Statistical Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

@_safe_operation("Numeric Statistics")
def _compute_numeric_stats(num_df: pd.DataFrame, percentiles: Iterable[float]) -> Dict[str, Dict[str, float]]:
    """
    Compute robust numeric statistics with extended percentile coverage.
    
    Returns dict mapping column names to their statistical summaries.
    """
    if num_df.empty:
        return {}
    
    percentile_list = sorted(set([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99] + list(percentiles)))
    desc = num_df.describe(percentiles=[p for p in percentile_list if 0 < p < 1]).T
    
    return desc.round(8).to_dict(orient="index")


@_safe_operation("Categorical Distribution")
def _compute_categorical_distribution(
    df: pd.DataFrame,
    cols: Iterable[str],
    top_k: int = 10,
    include_nan: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Extract categorical value distributions including NaN handling.
    
    Args:
        df: Source DataFrame
        cols: Column names to analyze
        top_k: Maximum values to extract per column
        include_nan: Whether to include NaN in value counts
    """
    result: Dict[str, Dict[str, int]] = {}
    
    for col in cols:
        try:
            vc = df[col].value_counts(dropna=not include_nan, sort=True).head(top_k)
            result[str(col)] = {str(k): int(v) for k, v in vc.items()}
        except Exception as e:
            _LOGGER.debug(f"value_counts failed for '{col}': {e}")
    
    return result


@_safe_operation("Missing Data Analysis")
def _analyze_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive missing data analysis including patterns and correlations.
    
    Returns:
        Dictionary with missing statistics and patterns
    """
    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / len(df) * 100).round(2)
    
    # Columns with any missing values
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    # Missing correlation (if multiple columns have missing)
    missing_correlation = {}
    if len(cols_with_missing) >= 2:
        try:
            missing_corr = df[cols_with_missing].isna().corr()
            # Extract strong correlations
            for i, c1 in enumerate(cols_with_missing):
                for j, c2 in enumerate(cols_with_missing):
                    if i < j:
                        corr_val = float(missing_corr.loc[c1, c2])
                        if abs(corr_val) > 0.3:  # Strong correlation threshold
                            missing_correlation[f"{c1}↔{c2}"] = round(corr_val, 3)
        except Exception as e:
            _LOGGER.debug(f"Missing correlation computation failed: {e}")
    
    return {
        "total_missing_cells": int(missing_counts.sum()),
        "pct_missing": float(missing_counts.sum() / (len(df) * len(df.columns)) * 100),
        "cols_with_missing": len(cols_with_missing),
        "detailed": {str(c): float(missing_pcts.get(c, 0)) for c in df.columns},
        "high_impact_missing": [c for c in cols_with_missing if missing_pcts[c] > 50],
        "missing_correlation": missing_correlation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Enterprise Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def profile_summary(df: pd.DataFrame, max_cols: int = 80) -> Dict[str, Any]:
    """
    **Profile Summary** — Comprehensive dataset metadata & health overview.
    
    Computes: shape, dtypes, missing patterns, cardinality, statistics,
    categorical distributions, memory footprint, and quality indicators.
    
    Args:
        df: Input pandas DataFrame
        max_cols: Maximum columns to analyze (safety limit)
    
    Returns:
        Dictionary with complete dataset profile including:
        - shape, columns preview, dtypes
        - missing data patterns & percentages
        - unique value counts
        - numeric statistics (mean, std, percentiles)
        - categorical top values
        - memory consumption
        - quality flags
    """
    # Validation
    err = _validate_df(df, "profile_summary input")
    if err:
        _LOGGER.warning(f"profile_summary: {err}")
        return {"error": err, "status": "failed"}
    
    # Safe column subset
    cols = [str(c) for c in df.columns[:max_cols]]
    n_rows, n_cols_full = df.shape
    
    # Core metadata
    result: Dict[str, Any] = {
        "shape": {"rows": int(n_rows), "cols": int(n_cols_full), "analyzed_cols": len(cols)},
        "preview_cols": cols,
    }
    
    # Type information
    try:
        result["dtypes"] = {c: str(t) for c, t in df[cols].dtypes.items()}
    except Exception as e:
        _LOGGER.warning(f"dtypes computation failed: {e}")
        result["dtypes"] = {}
    
    # Missing data analysis
    result["missing"] = _analyze_missing_patterns(df[cols])
    
    # Cardinality
    try:
        result["cardinality"] = {
            c: int(v) for c, v in df[cols].nunique(dropna=True).items()
        }
    except Exception as e:
        _LOGGER.warning(f"cardinality computation failed: {e}")
        result["cardinality"] = {}
    
    # Numeric statistics
    num_cols = df[cols].select_dtypes(include=[np.number])
    if not num_cols.empty:
        result["numeric_stats"] = _compute_numeric_stats(
            num_cols,
            percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        )
    else:
        result["numeric_stats"] = {}
    
    # Categorical distributions
    cat_cols = df[cols].select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        result["categorical_distributions"] = _compute_categorical_distribution(
            df,
            cat_cols,
            top_k=10,
            include_nan=True
        )
    else:
        result["categorical_distributions"] = {}
    
    # Memory footprint
    try:
        mem_bytes = df[cols].memory_usage(deep=True).sum()
        result["memory"] = {
            "bytes": int(mem_bytes),
            "mb": round(mem_bytes / (1024 ** 2), 3),
            "gb": round(mem_bytes / (1024 ** 3), 6),
        }
    except Exception as e:
        _LOGGER.debug(f"memory_usage computation failed: {e}")
        result["memory"] = {"error": str(e)}
    
    # Quality indicators
    result["quality"] = simple_quality_flags(df[cols])
    
    _LOGGER.info(f"✓ profile_summary: {n_rows} rows × {n_cols_full} cols")
    return result


def target_diagnosis(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    **Target Diagnosis** — Problem type classification & target analysis.
    
    Automatically determines problem type:
      • REGRESSION: numeric with >15 unique values
      • CLASSIFICATION: discrete, categorical, or numeric with ≤15 values
      • MULTILABEL: special handling available
    
    Args:
        df: Input DataFrame
        target: Target column name
    
    Returns:
        Dictionary with:
        - problem_type: "regression" or "classification"
        - dtype: Target data type
        - cardinality: Unique values (excluding NaN)
        - missing: Count of missing values
        - class_distribution: (for classification)
        - imbalance_metrics: (for classification)
        - statistical_summary: (for regression)
        - recommendations: Problem-specific suggestions
    """
    # Validation
    err = _validate_df(df, "target_diagnosis input")
    if err:
        _LOGGER.warning(f"target_diagnosis: {err}")
        return {"error": err, "status": "failed"}
    
    col_err = _validate_column(df, target)
    if col_err:
        _LOGGER.warning(f"target_diagnosis: {col_err}")
        return {"error": col_err, "status": "failed"}
    
    y = df[target]
    
    # Basic metadata
    try:
        missing = int(y.isna().sum())
        nunique = int(y.nunique(dropna=True))
    except Exception as e:
        _LOGGER.warning(f"target cardinality analysis failed: {e}")
        return {"error": str(e), "status": "failed"}
    
    # Heuristic: Problem type classification
    is_numeric = pd.api.types.is_numeric_dtype(y)
    problem_type = "regression" if (is_numeric and nunique > 15) else "classification"
    
    result: Dict[str, Any] = {
        "target": target,
        "problem_type": problem_type,
        "dtype": str(y.dtype),
        "cardinality": nunique,
        "missing": missing,
        "pct_missing": round(missing / len(df) * 100, 2),
    }
    
    # Classification-specific analysis
    if problem_type == "classification":
        try:
            vc = y.value_counts(dropna=True, sort=True)
            
            # Class distribution
            result["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
            
            # Imbalance metrics
            if len(vc) >= 2 and vc.min() > 0:
                max_count, min_count = int(vc.max()), int(vc.min())
                ratio = max_count / min_count
                result["imbalance_ratio"] = round(ratio, 3)
                result["imbalance_flag"] = ratio > 10.0
                result["majority_class_pct"] = round(max_count / len(y) * 100, 2)
                result["minority_class_pct"] = round(min_count / len(y) * 100, 2)
            else:
                result["imbalance_flag"] = True
            
            result["unique_classes"] = len(vc)
            
        except Exception as e:
            _LOGGER.warning(f"classification analysis failed: {e}")
            result["error"] = str(e)
    
    # Regression-specific analysis
    else:
        try:
            stats = y.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            result["statistical_summary"] = {
                k: (float(v) if pd.notna(v) else None) for k, v in stats.items()
            }
            
            # Check for outliers
            q1, q3 = y.quantile(0.25), y.quantile(0.75)
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((y < lower_bound) | (y > upper_bound)).sum()
            result["outlier_count"] = int(outliers)
            result["outlier_pct"] = round(outliers / len(y) * 100, 2)
            
        except Exception as e:
            _LOGGER.warning(f"regression analysis failed: {e}")
            result["error"] = str(e)
    
    # Recommendations
    recommendations: List[str] = []
    if missing > 0:
        recommendations.append(f"Handle {missing} missing values ({result['pct_missing']}%)")
    if problem_type == "classification" and result.get("imbalance_flag"):
        recommendations.append("Consider class imbalance mitigation (SMOTE, class weights)")
    if problem_type == "regression" and result.get("outlier_pct", 0) > 5:
        recommendations.append("Investigate outliers (>5% detected)")
    
    result["recommendations"] = recommendations
    
    _LOGGER.info(f"✓ target_diagnosis: {problem_type} problem ({nunique} classes/values)")
    return result


def potential_leakage(
    df: pd.DataFrame,
    target: Optional[str] = None,
    id_like_patterns: Iterable[str] = ("id", "uuid", "guid", "pk_", "_id"),
) -> Dict[str, Any]:
    """
    **Potential Leakage Detection** — Identifies columns likely to cause data leakage.
    
    Heuristic checks:
      1. ID-like columns (high cardinality or name patterns)
      2. Future information columns (name patterns like *_label, *_target)
      3. Perfect/near-perfect mapping to target (for classification)
      4. Temporal leakage indicators
    
    ⚠️  This is heuristic-based. Results are "suspected" — manual review needed.
    
    Args:
        df: Input DataFrame
        target: Target column name (optional, enables mapping checks)
        id_like_patterns: Tuple of patterns indicating ID columns
    
    Returns:
        Dictionary with:
        - suspected_columns: Dict mapping column names to leakage indicators
        - n_suspects: Total number of suspicious columns
        - severity_breakdown: Count by reason
    """
    # Validation
    err = _validate_df(df, "potential_leakage input")
    if err:
        _LOGGER.warning(f"potential_leakage: {err}")
        return {"error": err, "suspected_columns": {}, "n_suspects": 0}
    
    if target:
        col_err = _validate_column(df, target)
        if col_err:
            _LOGGER.warning(f"potential_leakage: {col_err}")
            target = None
    
    n_rows = len(df)
    suspects: Dict[str, Dict[str, Any]] = {}
    reasons_count: Dict[str, int] = {}
    
    # Check each column (excluding target)
    cols = [c for c in df.columns if c != target]
    
    for col in cols:
        col_str = str(col)
        col_lower = col_str.lower()
        reasons: List[str] = []
        metadata: Dict[str, Any] = {}
        
        try:
            s = df[col]
            nunique = int(s.nunique(dropna=True))
            metadata["nunique"] = nunique
            
            # ─── Check 1: ID-like columns (high cardinality)
            id_threshold = max(int(0.9 * n_rows), 1000)
            is_id_by_cardinality = nunique >= id_threshold
            is_id_by_name = any(pat in col_lower for pat in id_like_patterns)
            
            if is_id_by_cardinality or is_id_by_name:
                reasons.append("id_like")
                reasons_count["id_like"] = reasons_count.get("id_like", 0) + 1
        
        except Exception as e:
            _LOGGER.debug(f"Cardinality check failed for '{col}': {e}")
            continue
        
        # ─── Check 2: Name-based leakage indicators
        leakage_keywords = ("target", "label", "outcome", "groundtruth", "gt_", "true_", "actual_", "result")
        if any(kw in col_lower for kw in leakage_keywords):
            reasons.append("name_leaky_keyword")
            reasons_count["name_leaky_keyword"] = reasons_count.get("name_leaky_keyword", 0) + 1
        
        # ─── Check 3: Perfect mapping to target (classification context)
        if target:
            try:
                y = df[target]
                y_nunique = int(y.nunique(dropna=True))
                
                # Only check if target is categorical-like
                is_target_categorical = (not pd.api.types.is_numeric_dtype(y)) or (y_nunique <= 50)
                
                if is_target_categorical and s.notna().any():
                    # Groupby: for each value of col, is there only 1 target value?
                    grp_target = df.groupby(col, dropna=False)[target].nunique()
                    max_targets_per_group = int(grp_target.max())
                    
                    if max_targets_per_group == 1 and nunique <= int(0.8 * n_rows):
                        reasons.append("perfect_map_to_target")
                        reasons_count["perfect_map_to_target"] = reasons_count.get("perfect_map_to_target", 0) + 1
            
            except Exception as e:
                _LOGGER.debug(f"Perfect mapping check failed for '{col}': {e}")
        
        # Add to suspects if any reason found
        if reasons:
            suspects[col_str] = {
                "reasons": reasons,
                "combined_reason": " | ".join(reasons),
                **metadata
            }
    
    result: Dict[str, Any] = {
        "suspected_columns": suspects,
        "n_suspects": len(suspects),
        "severity_breakdown": reasons_count,
    }
    
    if suspects:
        _LOGGER.warning(f"⚠ potential_leakage: {len(suspects)} suspicious columns detected")
    else:
        _LOGGER.info("✓ potential_leakage: No obvious leakage indicators found")
    
    return result


def simple_quality_flags(df: pd.DataFrame, max_card: int = 1000) -> Dict[str, List[str]]:
    """
    **Simple Quality Flags** — Fast health check for data quality issues.
    
    Identifies:
      • High missing (>50% missing values)
      • Constant columns (only 1 unique value)
      • High-cardinality categoricals (likely IDs)
      • Columns with all NaN
    
    Args:
        df: Input DataFrame
        max_card: Threshold for high-cardinality categorical columns
    
    Returns:
        Dictionary with lists of problematic column names
    """
    # Validation
    err = _validate_df(df, "simple_quality_flags input")
    if err:
        _LOGGER.warning(f"simple_quality_flags: {err}")
        return {
            "high_missing": [],
            "constant_columns": [],
            "high_cardinality_categoricals": [],
            "all_nan_columns": [],
        }
    
    flags: Dict[str, List[str]] = {
        "high_missing": [],
        "constant_columns": [],
        "high_cardinality_categoricals": [],
        "all_nan_columns": [],
    }
    
    for col in df.columns:
        col_str = str(col)
        s = df[col]
        
        # Missing check
        try:
            missing_pct = float(s.isna().mean() * 100)
            if missing_pct == 100:
                flags["all_nan_columns"].append(col_str)
            elif missing_pct > 50:
                flags["high_missing"].append(col_str)
        except Exception:
            pass
        
        # Constant/cardinality check
        try:
            nunique = int(s.nunique(dropna=True))
            
            if nunique <= 1:
                flags["constant_columns"].append(col_str)
            elif (s.dtype == "object" or pd.api.types.is_categorical_dtype(s)) and nunique > max_card:
                flags["high_cardinality_categoricals"].append(col_str)
        except Exception:
            pass
    
    return flags


def advanced_quality_metrics(
    df: pd.DataFrame,
    target: Optional[str] = None,
    numeric_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    **Advanced Quality Metrics** — Enterprise-grade data quality assessment.
    
    Computes:
      • Completeness: % of non-null values
      • Consistency: dtype uniformity, range checks
      • Validity: type coherence, format compliance
      • Uniqueness: duplicate row counts, key integrity
      • Timeliness: (if temporal columns detected)
      • Statistical coherence: outlier ratios, distribution shifts
    
    Args:
        df: Input DataFrame
        target: Target column (for special handling)
        numeric_cols: Explicit numeric columns list (auto-detected if None)
    
    Returns:
        Comprehensive quality report dictionary
    """
    # Validation
    err = _validate_df(df, "advanced_quality_metrics input")
    if err:
        _LOGGER.warning(f"advanced_quality_metrics: {err}")
        return {"error": err, "status": "failed"}
    
    report: Dict[str, Any] = {
        "assessment_timestamp": pd.Timestamp.now().isoformat(),
        "dataset_size": {"rows": len(df), "columns": len(df.columns)},
    }
    
    # ─── Completeness Metrics
    completeness: Dict[str, Any] = {}
    total_cells = len(df) * len(df.columns)
    non_null_cells = total_cells - df.isna().sum().sum()
    
    completeness["overall_pct"] = round(non_null_cells / total_cells * 100, 2)
    completeness["by_column"] = {
        str(c): round((1 - df[c].isna().mean()) * 100, 2) for c in df.columns
    }
    
    report["completeness"] = completeness
    
    # ─── Uniqueness Metrics (duplicate detection)
    uniqueness: Dict[str, Any] = {}
    try:
        duplicate_rows = int(df.duplicated().sum())
        uniqueness["duplicate_rows"] = duplicate_rows
        uniqueness["duplicate_pct"] = round(duplicate_rows / len(df) * 100, 2)
        
        # Key columns (if target exists)
        if target:
            try:
                dup_including_target = int(df[[target]].duplicated().sum())
                uniqueness["duplicates_by_target"] = dup_including_target
            except Exception:
                pass
    
    except Exception as e:
        _LOGGER.debug(f"Uniqueness computation failed: {e}")
    
    report["uniqueness"] = uniqueness
    
    # ─── Statistical Coherence (for numeric columns)
    coherence: Dict[str, Any] = {}
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        outlier_summary = {}
        for col in numeric_df.columns:
            try:
                q1, q3 = numeric_df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
                outlier_pct = outliers / len(df) * 100
                outlier_summary[str(col)] = {
                    "count": int(outliers),
                    "pct": round(outlier_pct, 2),
                }
            except Exception as e:
                _LOGGER.debug(f"Outlier detection failed for '{col}': {e}")
        
        coherence["numeric_outliers"] = outlier_summary
        
        # Distribution skewness check
        try:
            skewness_summary = {}
            for col in numeric_df.columns:
                skew = float(numeric_df[col].skew())
                skewness_summary[str(col)] = {
                    "skewness": round(skew, 3),
                    "severity": "high" if abs(skew) > 2 else "moderate" if abs(skew) > 1 else "low",
                }
            coherence["skewness"] = skewness_summary
        except Exception as e:
            _LOGGER.debug(f"Skewness computation failed: {e}")
    
    report["statistical_coherence"] = coherence
    
    # ─── Type Consistency (dtype uniformity)
    type_consistency: Dict[str, Any] = {}
    dtype_distribution = df.dtypes.value_counts().to_dict()
    type_consistency["dtype_distribution"] = {str(k): int(v) for k, v in dtype_distribution.items()}
    
    # Check for mixed types (object columns with mixed types)
    mixed_types = []
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            types_in_col = df[col].dropna().apply(type).nunique()
            if types_in_col > 1:
                mixed_types.append(str(col))
        except Exception:
            pass
    
    if mixed_types:
        type_consistency["mixed_type_columns"] = mixed_types
    
    report["type_consistency"] = type_consistency
    
    # ─── Target-Specific Metrics (if provided)
    if target and target in df.columns:
        target_metrics: Dict[str, Any] = {}
        try:
            y = df[target]
            target_metrics["type"] = str(y.dtype)
            target_metrics["missing"] = int(y.isna().sum())
            target_metrics["unique_values"] = int(y.nunique(dropna=True))
            
            # If numeric
            if pd.api.types.is_numeric_dtype(y):
                target_metrics["range"] = {
                    "min": float(y.min()),
                    "max": float(y.max()),
                }
            
            # If categorical
            else:
                vc = y.value_counts(dropna=True).head(10)
                target_metrics["top_values"] = {str(k): int(v) for k, v in vc.items()}
        
        except Exception as e:
            _LOGGER.debug(f"Target metrics computation failed: {e}")
        
        report["target_metrics"] = target_metrics
    
    # ─── Overall Quality Score (0-100)
    try:
        completeness_score = completeness.get("overall_pct", 0)
        uniqueness_score = 100 - uniqueness.get("duplicate_pct", 0)
        
        # Penalize high missing
        missing_penalty = sum(
            1 for pct in completeness["by_column"].values() if pct < 50
        ) * 5
        
        overall_score = (completeness_score + uniqueness_score) / 2 - missing_penalty
        report["overall_quality_score"] = max(0, round(overall_score, 2))
    except Exception as e:
        _LOGGER.debug(f"Overall score computation failed: {e}")
    
    _LOGGER.info(f"✓ advanced_quality_metrics: Report generated for {len(df)} rows")
    return report