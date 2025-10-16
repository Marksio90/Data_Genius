# agents/eda/statistical_analyzer.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Statistical Analyzer              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade comprehensive statistical analysis:                      ║
║    ✓ Overall dataset metrics (shape, memory, sparsity)                    ║
║    ✓ Numeric features profiling (mean, std, quantiles, CV, variance)      ║
║    ✓ Categorical features analysis (cardinality, mode, dominance)         ║
║    ✓ Distribution analysis (normality tests, skewness, kurtosis)          ║
║    ✓ Quality flags (zero variance, near-constant, monotonicity)           ║
║    ✓ Multi-method normality testing (Shapiro/D'Agostino/Anderson)         ║
║    ✓ Intelligent sampling for large datasets (300k row limit)             ║
║    ✓ Column caps for performance (3000 columns per type)                  ║
║    ✓ Outlier detection via IQR method                                     ║
║    ✓ Actionable recommendations with emoji indicators                     ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "overall": {
        "n_rows": int,
        "n_columns": int,
        "n_numeric": int,
        "n_categorical": int,
        "memory_mb": float,
        "sparsity": float,
    },
    "numeric_features": {
        "n_features": int,
        "features": {
            col: {
                "count": int,
                "mean": float,
                "std": float,
                "min": float,
                "q01": float,
                "q25": float,
                "median": float,
                "q75": float,
                "q99": float,
                "max": float,
                "skewness": float,
                "kurtosis": float,
                "variance": float,
                "range": float,
                "iqr": float,
                "cv": float | None,
                "zero_variance": bool,
                "near_constant": bool,
                "monotonic": "increasing" | "decreasing" | None,
            }
        },
        "summary": {
            "highest_variance": str | None,
            "lowest_variance": str | None,
            "avg_skewness": float,
            "zero_variance_features": List[str],
            "near_constant_features": List[str],
            "high_cv_features": List[str],
        },
    },
    "categorical_features": {
        "n_features": int,
        "features": {
            col: {
                "count": int,
                "n_unique": int,
                "mode": str | None,
                "mode_frequency": int,
                "mode_percentage": float,
                "top_k_values": Dict[str, int],
                "is_binary": bool,
                "cardinality": "high" | "medium" | "low",
                "majority_share": float,
            }
        },
        "summary": {
            "high_cardinality_features": List[str],
            "dominant_classes_features": List[str],
        },
    },
    "distributions": {
        col: {
            "distribution_type": "normal" | "symmetric" | "right_skewed" | "left_skewed",
            "is_normal": bool | None,
            "normality_test": "shapiro" | "dagostino" | "anderson" | None,
            "p_value": float | None,
            "skewness": float,
            "kurtosis": float,
            "has_outliers": bool,
            "heavy_tails": bool,
            "high_skewness": bool,
        }
    },
    "recommendations": List[str],
    "summary": {
        "n_zero_variance": int,
        "n_near_constant": int,
        "n_high_cv": int,
        "n_high_cardinality": int,
        "n_dominant_categorical": int,
    },
    "telemetry": {
        "elapsed_ms": float,
        "timings_ms": {
            "overall": float,
            "numeric": float,
            "categorical": float,
            "distributions": float,
        },
        "sampled_for_tests": bool,
        "sample_info": {"from_rows": int, "to_rows": int} | None,
        "caps": {
            "numeric_cols_total": int,
            "numeric_cols_used": int,
            "numeric_cols_cap": int,
            "categorical_cols_total": int,
            "categorical_cols_used": int,
            "categorical_cols_cap": int,
        },
        "skipped_columns": {
            "numeric_empty": List[str],
            "categorical_empty": List[str],
        },
    },
    "version": "5.0-kosmos-enterprise",
}
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from scipy import stats
except ImportError:
    stats = None
    logger.warning("⚠ scipy not available — normality tests disabled")

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
except ImportError:
    # Fallback for testing
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data = None
            self.errors = []
            self.warnings = []
        
        def add_error(self, msg: str):
            self.errors.append(msg)
        
        def add_warning(self, msg: str):
            self.warnings.append(msg)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StatisticalAnalyzerConfig:
    """Enterprise configuration for statistical analysis."""
    
    # Normality testing
    normality_alpha: float = 0.05          # Significance threshold
    max_shapiro_n: int = 5_000             # Max samples for Shapiro-Wilk test
    max_rows_for_tests: int = 300_000      # Sampling limit for normality tests
    random_state: int = 42
    
    # Distribution thresholds
    high_cardinality_threshold: int = 50   # >50 unique values → high cardinality
    skew_high_abs: float = 1.0             # |skew| > 1 → high skewness
    kurt_high_abs: float = 3.0             # |excess kurtosis| > 3 → heavy tails
    cv_warn: float = 1.0                   # CV > 1 → high variability
    near_constant_ratio: float = 0.98      # ≥98% same value → near-constant
    
    # Output configuration
    top_k_values: int = 5                  # Top K values for categorical features
    
    # Safety guards
    min_non_na_numeric: int = 3            # Minimum non-NA for numeric calculations
    min_non_na_categorical: int = 3        # Minimum non-NA for categorical stats
    max_numeric_cols: int = 3000           # Soft limit on numeric columns
    max_categorical_cols: int = 3000       # Soft limit on categorical columns
    
    # Preprocessing
    strip_object_whitespace: bool = True
    replace_empty_string_with_nan: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str):
    """Decorator for operation timing with intelligent logging."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(f"⏱ {operation_name}: {elapsed_ms:.2f}ms")
        return wrapper
    return decorator


def _safe_operation(operation_name: str, default_value: Any = None):
    """Decorator for defensive operations with fallback values."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠ {operation_name} failed: {type(e).__name__}: {str(e)[:80]}")
                return default_value
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Statistical Analyzer Agent
# ═══════════════════════════════════════════════════════════════════════════

class StatisticalAnalyzer(BaseAgent):
    """
    **StatisticalAnalyzer** — Enterprise comprehensive statistical analysis.
    
    Responsibilities:
      1. Overall dataset metrics (shape, memory, sparsity)
      2. Numeric features profiling with quantiles & quality flags
      3. Categorical features analysis with cardinality assessment
      4. Distribution analysis with multi-method normality testing
      5. Quality flags (zero variance, near-constant, monotonicity)
      6. Outlier detection via IQR method
      7. Actionable recommendations generation
      8. Intelligent sampling for large datasets
      9. Column caps for performance optimization
      10. Zero side-effects on input DataFrame
    
    Features:
      • Multi-method normality testing (Shapiro/D'Agostino/Anderson)
      • Coefficient of variation (CV) analysis
      • Monotonicity detection
      • Near-constant feature identification
      • High cardinality detection
      • Dominant class detection
      • Heavy tails & skewness flags
    """
    
    def __init__(self, config: Optional[StatisticalAnalyzerConfig] = None) -> None:
        """Initialize analyzer with optional custom configuration."""
        super().__init__(
            name="StatisticalAnalyzer",
            description="Comprehensive statistical analysis of dataset features"
        )
        self.config = config or StatisticalAnalyzerConfig()
        self._log = logger.bind(agent="StatisticalAnalyzer")
        warnings.filterwarnings("ignore")
        
        # Check scipy availability
        if stats is None:
            self._log.warning("⚠ scipy unavailable — normality tests will be skipped")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            data: pd.DataFrame
        
        Optional:
            include_numeric: List[str] — whitelist numeric columns
            include_categorical: List[str] — whitelist categorical columns
        """
        if "data" not in kwargs:
            raise ValueError("Required parameter 'data' not provided")
        
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError(f"'data' must be pd.DataFrame, got {type(kwargs['data']).__name__}")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("StatisticalAnalyzer.execute")
    def execute(
        self,
        data: pd.DataFrame,
        include_numeric: Optional[List[str]] = None,
        include_categorical: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            data: Input DataFrame (not modified)
            include_numeric: Optional whitelist for numeric columns
            include_categorical: Optional whitelist for categorical columns
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with statistical analysis (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t0_total = time.perf_counter()
        
        try:
            # Input validation
            if data is None or not isinstance(data, pd.DataFrame):
                msg = "Invalid input: expected pandas DataFrame"
                result.add_error(msg)
                self._log.error(msg)
                result.data = self._empty_payload()
                return result
            
            if data.empty:
                self._log.warning("⚠ Empty DataFrame provided")
                result.add_warning("Empty DataFrame — statistical analysis skipped")
                payload = self._empty_payload()
                payload["telemetry"]["elapsed_ms"] = round((time.perf_counter() - t0_total) * 1000, 1)
                result.data = payload
                return result
            
            cfg = self.config
            
            # ─── Data Preparation (zero side-effects)
            df = self._prepare_dataframe(data)
            
            # ─── 1. Overall Statistics
            t0 = time.perf_counter()
            overall_stats = self._get_overall_statistics(df)
            t_overall = (time.perf_counter() - t0) * 1000
            
            # ─── 2. Numeric Features Analysis
            t0 = time.perf_counter()
            num_df, num_caps_meta = self._select_numeric_features(df, include_numeric)
            numeric_stats = self._analyze_numeric_features(num_df)
            t_numeric = (time.perf_counter() - t0) * 1000
            
            # ─── 3. Categorical Features Analysis
            t0 = time.perf_counter()
            cat_df, cat_caps_meta = self._select_categorical_features(df, include_categorical)
            categorical_stats = self._analyze_categorical_features(cat_df)
            t_categorical = (time.perf_counter() - t0) * 1000
            
            # ─── 4. Distribution Analysis (with sampling)
            t0 = time.perf_counter()
            dist_df, sampled, sample_info = self._maybe_sample_for_tests(num_df)
            distributions = self._analyze_distributions(dist_df)
            t_distributions = (time.perf_counter() - t0) * 1000
            
            # ─── 5. Generate Recommendations
            recommendations = self._build_recommendations(
                numeric_stats=numeric_stats,
                categorical_stats=categorical_stats,
                distributions=distributions
            )
            
            # ─── 6. Compile Summary
            summary = self._compile_summary(numeric_stats, categorical_stats)
            
            # ─── Assemble Result
            elapsed_ms = round((time.perf_counter() - t0_total) * 1000, 1)
            
            result.data = {
                "overall": overall_stats,
                "numeric_features": numeric_stats,
                "categorical_features": categorical_stats,
                "distributions": distributions,
                "recommendations": recommendations,
                "summary": summary,
                "telemetry": {
                    "elapsed_ms": elapsed_ms,
                    "timings_ms": {
                        "overall": round(t_overall, 1),
                        "numeric": round(t_numeric, 1),
                        "categorical": round(t_categorical, 1),
                        "distributions": round(t_distributions, 1),
                    },
                    "sampled_for_tests": bool(sampled),
                    "sample_info": sample_info,
                    "caps": {
                        "numeric_cols_total": num_caps_meta["caps"]["total"],
                        "numeric_cols_used": num_caps_meta["caps"]["used"],
                        "numeric_cols_cap": num_caps_meta["caps"]["cap"],
                        "categorical_cols_total": cat_caps_meta["caps"]["total"],
                        "categorical_cols_used": cat_caps_meta["caps"]["used"],
                        "categorical_cols_cap": cat_caps_meta["caps"]["cap"],
                    },
                    "skipped_columns": {
                        "numeric_empty": num_caps_meta["skipped_empty"],
                        "categorical_empty": cat_caps_meta["skipped_empty"],
                    },
                },
                "version": "5.0-kosmos-enterprise",
            }
            
            self._log.success(
                f"✓ Statistical analysis complete | "
                f"numeric={len(num_df.columns)} categorical={len(cat_df.columns)} | "
                f"elapsed={elapsed_ms:.1f}ms"
            )
        
        except Exception as e:
            msg = f"Statistical analysis failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            result.data = self._empty_payload()
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Data Preparation (Zero Side-Effects)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("data_preparation")
    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for analysis with normalization.
        
        Operations:
          • Deep copy (no side-effects)
          • Replace ±Inf with NaN
          • Strip whitespace from object columns
          • Convert empty strings to NaN
        
        Returns:
            Prepared DataFrame copy
        """
        cfg = self.config
        df = data.copy()
        
        # Inf → NaN conversion
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Object column preprocessing
        obj_cols = df.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            # Strip whitespace
            if cfg.strip_object_whitespace:
                try:
                    for col in obj_cols:
                        df[col] = df[col].astype(str).str.strip()
                except Exception as e:
                    self._log.debug(f"Whitespace stripping failed: {e}")
            
            # Empty string → NaN
            if cfg.replace_empty_string_with_nan:
                try:
                    df[obj_cols] = df[obj_cols].replace("", np.nan)
                except Exception as e:
                    self._log.debug(f"Empty string replacement failed: {e}")
        
        return df
    
    # ───────────────────────────────────────────────────────────────────
    # Overall Statistics
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("overall_statistics")
    @_safe_operation("overall_statistics", default_value={
        "n_rows": 0, "n_columns": 0, "n_numeric": 0, "n_categorical": 0,
        "memory_mb": 0.0, "sparsity": 0.0
    })
    def _get_overall_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute overall dataset statistics.
        
        Returns comprehensive dataset-level metrics.
        """
        n_rows = int(len(df))
        n_cols = int(len(df.columns))
        
        # Memory usage
        try:
            memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
        except Exception:
            memory_mb = float(df.memory_usage().sum() / 1024**2)
        
        # Sparsity (missing + zeros)
        total_cells = max(1, n_rows * n_cols)
        sparsity = float(df.isna().sum().sum() / total_cells)
        
        # Type counts
        n_numeric = int(len(df.select_dtypes(include=[np.number]).columns))
        n_categorical = int(len(df.select_dtypes(include=["object", "category"]).columns))
        
        return {
            "n_rows": n_rows,
            "n_columns": n_cols,
            "n_numeric": n_numeric,
            "n_categorical": n_categorical,
            "memory_mb": round(memory_mb, 2),
            "sparsity": round(sparsity, 4),
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Numeric Features Selection & Analysis
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("numeric_selection")
    def _select_numeric_features(
        self,
        df: pd.DataFrame,
        include: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Select numeric features with intelligent filtering.
        
        Guards:
          • Whitelist filtering (if provided)
          • Column cap for performance
          • Minimum non-NA requirement
        
        Returns:
            Tuple of (filtered_df, metadata_dict)
        """
        cfg = self.config
        
        # Get numeric columns
        num_df = df.select_dtypes(include=[np.number])
        total = int(num_df.shape[1])
        
        # Whitelist filter
        if include:
            keep = [c for c in include if c in num_df.columns]
            num_df = num_df.loc[:, keep]
        
        # Column cap
        capped = num_df.shape[1] > cfg.max_numeric_cols
        if capped:
            self._log.warning(
                f"⚠ Limiting numeric columns: {num_df.shape[1]} → {cfg.max_numeric_cols}"
            )
            num_df = num_df.iloc[:, :cfg.max_numeric_cols]
        
        # Filter empty columns (insufficient data)
        skipped_empty: List[str] = []
        for col in list(num_df.columns):
            if num_df[col].count() < cfg.min_non_na_numeric:
                skipped_empty.append(str(col))
                num_df = num_df.drop(columns=[col])
        
        metadata = {
            "caps": {
                "total": total,
                "used": int(num_df.shape[1]),
                "cap": cfg.max_numeric_cols if capped else int(num_df.shape[1]),
            },
            "skipped_empty": skipped_empty,
        }
        
        return num_df, metadata
    
    @_timeit("numeric_analysis")
    def _analyze_numeric_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze numeric features comprehensively.
        
        Computes:
          • Descriptive statistics (mean, std, quantiles)
          • Quality flags (zero variance, near-constant, monotonicity)
          • Coefficient of variation (CV)
          • IQR and range
        
        Returns:
            Dictionary with features stats and summary
        """
        cfg = self.config
        
        if df.shape[1] == 0:
            return {
                "n_features": 0,
                "features": {},
                "summary": {"message": "No numeric features found"},
            }
        
        features_stats: Dict[str, Dict[str, Any]] = {}
        zero_variance_list: List[str] = []
        near_constant_list: List[str] = []
        high_cv_list: List[str] = []
        
        for col in df.columns:
            stats_dict = self._compute_numeric_stats(df[col], col)
            
            if stats_dict is None:
                continue
            
            features_stats[str(col)] = stats_dict
            
            # Track quality flags
            if stats_dict.get("zero_variance"):
                zero_variance_list.append(str(col))
            
            if stats_dict.get("near_constant"):
                near_constant_list.append(str(col))
            
            cv = stats_dict.get("cv")
            if cv is not None and cv > cfg.cv_warn:
                high_cv_list.append(str(col))
        
        # Generate summary
        summary = self._generate_numeric_summary(
            features_stats=features_stats,
            zero_variance=zero_variance_list,
            near_constant=near_constant_list,
            high_cv=high_cv_list
        )
        
        return {
            "n_features": len(df.columns),
            "features": features_stats,
            "summary": summary,
        }
    
    @_safe_operation("numeric_stats", default_value=None)
    def _compute_numeric_stats(
        self,
        series: pd.Series,
        col_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compute comprehensive statistics for numeric series.
        
        Returns:
            Dictionary with all numeric statistics or None if insufficient data
        """
        cfg = self.config
        
        # Coerce to numeric
        s = pd.to_numeric(series, errors="coerce")
        s_nona = s.dropna()
        
        # Guard: insufficient data
        if len(s_nona) < cfg.min_non_na_numeric:
            return None
        
        # Descriptive statistics
        count = int(s_nona.count())
        mean = float(s_nona.mean())
        std = float(s_nona.std(ddof=1)) if len(s_nona) > 1 else 0.0
        variance = float(s_nona.var(ddof=1)) if len(s_nona) > 1 else 0.0
        
        # Quantiles
        q01 = float(s_nona.quantile(0.01))
        q25 = float(s_nona.quantile(0.25))
        median = float(s_nona.median())
        q75 = float(s_nona.quantile(0.75))
        q99 = float(s_nona.quantile(0.99))
        
        # Min/Max
        mn = float(s_nona.min())
        mx = float(s_nona.max())
        
        # Derived metrics
        rng = float(mx - mn)
        iqr = float(q75 - q25)
        cv = float(std / mean) if mean != 0 else None
        
        # Moments
        skewness = float(s_nona.skew()) if len(s_nona) > 2 else 0.0
        kurtosis = float(s_nona.kurtosis()) if len(s_nona) > 3 else 0.0
        
        # Quality flags
        zero_variance = bool(variance == 0.0)
        
        # Near-constant detection
        mode_freq = int(s_nona.value_counts(dropna=False).iloc[0]) if len(s_nona) > 0 else 0
        near_constant = bool((mode_freq / max(1, len(s_nona))) >= cfg.near_constant_ratio)
        
        # Monotonicity detection
        monotonic = None
        try:
            if s_nona.is_monotonic_increasing:
                monotonic = "increasing"
            elif s_nona.is_monotonic_decreasing:
                monotonic = "decreasing"
        except Exception:
            monotonic = None
        
        return {
            "count": count,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(mn, 6),
            "q01": round(q01, 6),
            "q25": round(q25, 6),
            "median": round(median, 6),
            "q75": round(q75, 6),
            "q99": round(q99, 6),
            "max": round(mx, 6),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "variance": round(variance, 6),
            "range": round(rng, 6),
            "iqr": round(iqr, 6),
            "cv": round(cv, 4) if cv is not None else None,
            "zero_variance": zero_variance,
            "near_constant": near_constant,
            "monotonic": monotonic,
        }
    
    def _generate_numeric_summary(
        self,
        features_stats: Dict[str, Dict[str, Any]],
        zero_variance: List[str],
        near_constant: List[str],
        high_cv: List[str]
    ) -> Dict[str, Any]:
        """Generate summary statistics for numeric features."""
        
        if not features_stats:
            return {}
        
        # Find highest/lowest variance
        variances = {k: v.get("variance", 0.0) for k, v in features_stats.items()}
        non_empty = {k: v for k, v in variances.items() if v is not None}
        
        highest_var = max(non_empty, key=non_empty.get) if non_empty else None
        lowest_var = min(non_empty, key=non_empty.get) if non_empty else None
        
        # Average skewness
        skewnesses = [v.get("skewness", 0.0) for v in features_stats.values()]
        avg_skewness = float(np.mean(skewnesses)) if skewnesses else 0.0
        
        return {
            "highest_variance": highest_var,
            "lowest_variance": lowest_var,
            "avg_skewness": round(avg_skewness, 4),
            "zero_variance_features": zero_variance,
            "near_constant_features": near_constant,
            "high_cv_features": high_cv,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Categorical Features Selection & Analysis
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("categorical_selection")
    def _select_categorical_features(
        self,
        df: pd.DataFrame,
        include: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Select categorical features with intelligent filtering.
        
        Guards:
          • Whitelist filtering (if provided)
          • Column cap for performance
          • Minimum non-NA requirement
        
        Returns:
            Tuple of (filtered_df, metadata_dict)
        """
        cfg = self.config
        
        # Get categorical columns
        cat_df = df.select_dtypes(include=["object", "category"])
        total = int(cat_df.shape[1])
        
        # Whitelist filter
        if include:
            keep = [c for c in include if c in cat_df.columns]
            cat_df = cat_df.loc[:, keep]
        
        # Column cap
        capped = cat_df.shape[1] > cfg.max_categorical_cols
        if capped:self._log.warning(f"⚠ Limiting categorical columns: {cat_df.shape[1]} → {cfg.max_categorical_cols}")
        cat_df = cat_df.iloc[:, :cfg.max_categorical_cols]
    # Filter empty columns (insufficient data)
    skipped_empty: List[str] = []
    for col in list(cat_df.columns):
        if cat_df[col].count() < cfg.min_non_na_categorical:
            skipped_empty.append(str(col))
            cat_df = cat_df.drop(columns=[col])
    
    metadata = {
        "caps": {
            "total": total,
            "used": int(cat_df.shape[1]),
            "cap": cfg.max_categorical_cols if capped else int(cat_df.shape[1]),
        },
        "skipped_empty": skipped_empty,
    }
    
    return cat_df, metadata

@_timeit("categorical_analysis")
def _analyze_categorical_features(self, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze categorical features comprehensively.
    
    Computes:
      • Cardinality (unique values count)
      • Mode and frequency
      • Top-K most frequent values
      • Binary detection
      • Majority class share
    
    Returns:
        Dictionary with features stats and summary
    """
    cfg = self.config
    
    if df.shape[1] == 0:
        return {
            "n_features": 0,
            "features": {},
            "summary": {
                "high_cardinality_features": [],
                "dominant_classes_features": [],
            },
        }
    
    features_stats: Dict[str, Dict[str, Any]] = {}
    high_cardinality_list: List[str] = []
    dominant_list: List[str] = []
    
    for col in df.columns:
        stats_dict = self._compute_categorical_stats(df[col], col)
        
        if stats_dict is None:
            continue
        
        features_stats[str(col)] = stats_dict
        
        # Track quality flags
        if stats_dict.get("cardinality") == "high":
            high_cardinality_list.append(str(col))
        
        if stats_dict.get("majority_share", 0.0) > 80.0:
            dominant_list.append(str(col))
    
    return {
        "n_features": len(df.columns),
        "features": features_stats,
        "summary": {
            "high_cardinality_features": high_cardinality_list,
            "dominant_classes_features": dominant_list,
        },
    }

@_safe_operation("categorical_stats", default_value=None)
def _compute_categorical_stats(
    self,
    series: pd.Series,
    col_name: str
) -> Optional[Dict[str, Any]]:
    """
    Compute comprehensive statistics for categorical series.
    
    Returns:
        Dictionary with all categorical statistics or None if insufficient data
    """
    cfg = self.config
    
    # Convert to string
    s = series.astype("string")
    s_nona = s.dropna()
    
    # Guard: insufficient data
    if len(s_nona) < cfg.min_non_na_categorical:
        return None
    
    # Value counts
    value_counts = s_nona.value_counts()
    
    # Basic statistics
    count = int(len(s_nona))
    n_unique = int(s_nona.nunique())
    
    # Mode
    try:
        mode_series = s_nona.mode()
        mode_val = str(mode_series.iloc[0]) if not mode_series.empty else None
    except Exception:
        mode_val = None
    
    # Mode frequency and percentage
    mode_frequency = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
    mode_percentage = float((mode_frequency / max(1, count)) * 100.0)
    
    # Top-K values
    top_k = {
        str(k): int(v)
        for k, v in value_counts.head(cfg.top_k_values).to_dict().items()
    }
    
    # Binary detection
    is_binary = bool(n_unique == 2)
    
    # Cardinality classification
    if n_unique > cfg.high_cardinality_threshold:
        cardinality = "high"
    elif n_unique > int(cfg.high_cardinality_threshold * 0.4):
        cardinality = "medium"
    else:
        cardinality = "low"
    
    return {
        "count": count,
        "n_unique": n_unique,
        "mode": mode_val,
        "mode_frequency": mode_frequency,
        "mode_percentage": round(mode_percentage, 2),
        "top_k_values": top_k,
        "is_binary": is_binary,
        "cardinality": cardinality,
        "majority_share": round(mode_percentage, 2),
    }

# ───────────────────────────────────────────────────────────────────
# Distribution Analysis with Normality Testing
# ───────────────────────────────────────────────────────────────────

@_timeit("sampling_for_tests")
def _maybe_sample_for_tests(
    self,
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, bool, Optional[Dict[str, int]]]:
    """
    Sample DataFrame for normality tests if too large.
    
    Returns:
        Tuple of (sampled_df, was_sampled, sample_info)
    """
    cfg = self.config
    
    if len(df) > cfg.max_rows_for_tests:
        self._log.info(
            f"Sampling for distribution tests: {len(df):,} → {cfg.max_rows_for_tests:,} rows"
        )
        
        sampled_df = df.sample(
            n=cfg.max_rows_for_tests,
            random_state=cfg.random_state
        )
        
        sample_info = {
            "from_rows": int(len(df)),
            "to_rows": int(cfg.max_rows_for_tests),
        }
        
        return sampled_df, True, sample_info
    
    return df, False, None

@_timeit("distribution_analysis")
def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze distributions with multi-method normality testing.
    
    Methods:
      • Shapiro-Wilk test (N ≤ 5,000)
      • D'Agostino K² test (N > 5,000)
      • Anderson-Darling test (fallback)
    
    Returns:
        Dictionary with distribution characteristics per column
    """
    cfg = self.config
    
    if df.shape[1] == 0:
        return {}
    
    distributions: Dict[str, Dict[str, Any]] = {}
    
    for col in df.columns:
        dist_dict = self._analyze_column_distribution(df[col], col)
        
        if dist_dict is not None:
            distributions[str(col)] = dist_dict
    
    return distributions

@_safe_operation("column_distribution", default_value=None)
def _analyze_column_distribution(
    self,
    series: pd.Series,
    col_name: str
) -> Optional[Dict[str, Any]]:
    """
    Analyze distribution characteristics for a single column.
    
    Returns:
        Dictionary with distribution analysis or None if insufficient data
    """
    cfg = self.config
    
    # Coerce to numeric and drop NA
    s = pd.to_numeric(series, errors="coerce").dropna()
    
    # Guard: insufficient data
    if len(s) < max(10, cfg.min_non_na_numeric):
        return None
    
    # Compute moments
    skewness = float(s.skew()) if len(s) > 2 else 0.0
    kurtosis = float(s.kurtosis()) if len(s) > 3 else 0.0
    
    # Normality testing (if scipy available)
    is_normal: Optional[bool] = None
    test_name: Optional[str] = None
    p_value: Optional[float] = None
    
    if stats is not None:
        is_normal, test_name, p_value = self._test_normality(s)
    
    # Distribution type classification
    if is_normal:
        dist_type = "normal"
    elif abs(skewness) < 0.5:
        dist_type = "symmetric"
    elif skewness > 0.5:
        dist_type = "right_skewed"
    else:
        dist_type = "left_skewed"
    
    # Additional characteristics
    has_outliers = self._check_outliers_iqr(s)
    heavy_tails = bool(abs(kurtosis) > cfg.kurt_high_abs)
    high_skewness = bool(abs(skewness) > cfg.skew_high_abs)
    
    return {
        "distribution_type": dist_type,
        "is_normal": is_normal,
        "normality_test": test_name,
        "p_value": round(p_value, 6) if p_value is not None else None,
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "has_outliers": has_outliers,
        "heavy_tails": heavy_tails,
        "high_skewness": high_skewness,
    }

@_safe_operation("normality_test", default_value=(None, None, None))
def _test_normality(
    self,
    series: pd.Series
) -> Tuple[Optional[bool], Optional[str], Optional[float]]:
    """
    Test normality using appropriate method based on sample size.
    
    Returns:
        Tuple of (is_normal, test_name, p_value)
    """
    cfg = self.config
    n = len(series)
    
    # Method 1: Shapiro-Wilk (small to medium samples)
    if n <= cfg.max_shapiro_n:
        try:
            stat, p = stats.shapiro(series)
            is_normal = bool(p > cfg.normality_alpha)
            return is_normal, "shapiro", float(p)
        except Exception:
            pass
    
    # Method 2: D'Agostino K² (large samples)
    try:
        k2, p = stats.normaltest(series, nan_policy="omit")
        is_normal = bool(p > cfg.normality_alpha)
        return is_normal, "dagostino", float(p)
    except Exception:
        pass
    
    # Method 3: Anderson-Darling (fallback)
    try:
        result = stats.anderson(series, dist="norm")
        # Compare statistic to 5% critical value (index 2)
        critical_5pct = float(result.critical_values[2])
        is_normal = bool(result.statistic < critical_5pct)
        return is_normal, "anderson", None  # No p-value for Anderson
    except Exception:
        pass
    
    return None, None, None

@_safe_operation("outlier_check", default_value=False)
def _check_outliers_iqr(self, series: pd.Series) -> bool:
    """
    Check for outliers using IQR method.
    
    Returns:
        True if outliers detected, False otherwise
    """
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    
    # Guard: zero IQR
    if iqr <= 0:
        return False
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    outliers = ((series < lower) | (series > upper)).sum()
    return bool(outliers > 0)

# ───────────────────────────────────────────────────────────────────
# Recommendations Generation
# ───────────────────────────────────────────────────────────────────

@_timeit("recommendations")
def _build_recommendations(
    self,
    numeric_stats: Dict[str, Any],
    categorical_stats: Dict[str, Any],
    distributions: Dict[str, Any]
) -> List[str]:
    """
    Generate actionable recommendations based on analysis.
    
    Categories:
      • Zero variance features
      • Near-constant features
      • High CV features
      • Skewed distributions
      • Heavy-tailed distributions
      • High cardinality features
      • Dominant class features
    
    Returns:
        Deduplicated list of recommendations
    """
    rec: List[str] = []
    
    # ─── Numeric Features Issues
    num_summary = numeric_stats.get("summary", {}) or {}
    
    # Zero variance
    zero_vars = num_summary.get("zero_variance_features", []) or []
    if zero_vars:
        vars_str = ", ".join(str(v) for v in zero_vars[:3])
        if len(zero_vars) > 3:
            vars_str += f" (+{len(zero_vars)-3} more)"
        rec.append(
            f"❄️ Remove zero-variance features: {vars_str}. "
            "These provide no information for modeling."
        )
    
    # Near-constant
    near_const = num_summary.get("near_constant_features", []) or []
    if near_const:
        const_str = ", ".join(str(v) for v in near_const[:3])
        if len(near_const) > 3:
            const_str += f" (+{len(near_const)-3} more)"
        rec.append(
            f"🧊 Near-constant features detected (≥98% same value): {const_str}. "
            "Consider dropping or using target encoding if informative."
        )
    
    # High CV
    high_cv = num_summary.get("high_cv_features", []) or []
    if high_cv:
        cv_str = ", ".join(str(v) for v in high_cv[:3])
        if len(high_cv) > 3:
            cv_str += f" (+{len(high_cv)-3} more)"
        rec.append(
            f"📈 High coefficient of variation (CV > 1.0): {cv_str}. "
            "Consider: (1) Robust scaling (median/MAD), "
            "(2) Log/Yeo-Johnson transformation, or (3) Tree-based models."
        )
    
    # ─── Distribution Issues
    skewed = [c for c, d in distributions.items() if d.get("high_skewness")]
    if skewed:
        skew_str = ", ".join(str(v) for v in skewed[:3])
        if len(skewed) > 3:
            skew_str += f" (+{len(skewed)-3} more)"
        rec.append(
            f"↔️ Highly skewed distributions (|skew| > 1.0): {skew_str}. "
            "Apply transformations: log, Box-Cox, or Yeo-Johnson. "
            "For tree-based models, skewness is less problematic."
        )
    
    heavy_tails = [c for c, d in distributions.items() if d.get("heavy_tails")]
    if heavy_tails:
        tail_str = ", ".join(str(v) for v in heavy_tails[:3])
        if len(heavy_tails) > 3:
            tail_str += f" (+{len(heavy_tails)-3} more)"
        rec.append(
            f"🪙 Heavy-tailed distributions (|kurtosis| > 3.0): {tail_str}. "
            "Prefer robust methods: (1) Huber loss, "
            "(2) Quantile regression, (3) Winsorization, or (4) Tree-based models."
        )
    
    # ─── Categorical Features Issues
    cat_summary = categorical_stats.get("summary", {}) or {}
    
    # High cardinality
    high_card = cat_summary.get("high_cardinality_features", []) or []
    if high_card:
        card_str = ", ".join(str(v) for v in high_card[:3])
        if len(high_card) > 3:
            card_str += f" (+{len(high_card)-3} more)"
        rec.append(
            f"🏷️ High cardinality categorical features (>50 unique): {card_str}. "
            "Consider: (1) Target encoding, (2) CatBoost/LightGBM native support, "
            "(3) Frequency encoding, or (4) Embedding layers."
        )
    
    # Dominant classes
    dominant = cat_summary.get("dominant_classes_features", []) or []
    if dominant:
        dom_str = ", ".join(str(v) for v in dominant[:3])
        if len(dominant) > 3:
            dom_str += f" (+{len(dominant)-3} more)"
        rec.append(
            f"⚖️ Dominant class imbalance (>80% majority): {dom_str}. "
            "Consider: (1) Combining rare classes, "
            "(2) Class weights, (3) SMOTE/oversampling, or (4) Anomaly detection."
        )
    
    # ─── Positive Message
    if not rec:
        rec.append(
            "✅ Feature distributions appear healthy — "
            "variance is sufficient, no extreme skewness or cardinality issues. "
            "Ready for feature engineering and modeling."
        )
    
    # Deduplicate and return
    return list(dict.fromkeys([r for r in rec if r]))

# ───────────────────────────────────────────────────────────────────
# Summary Compilation
# ───────────────────────────────────────────────────────────────────

def _compile_summary(
    self,
    numeric_stats: Dict[str, Any],
    categorical_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Compile high-level summary statistics."""
    
    num_summary = numeric_stats.get("summary", {}) or {}
    cat_summary = categorical_stats.get("summary", {}) or {}
    
    zero_vars = num_summary.get("zero_variance_features", []) or []
    near_const = num_summary.get("near_constant_features", []) or []
    high_cv = num_summary.get("high_cv_features", []) or []
    high_card = cat_summary.get("high_cardinality_features", []) or []
    dominant = cat_summary.get("dominant_classes_features", []) or []
    
    return {
        "n_zero_variance": int(len(zero_vars)),
        "n_near_constant": int(len(near_const)),
        "n_high_cv": int(len(high_cv)),
        "n_high_cardinality": int(len(high_card)),
        "n_dominant_categorical": int(len(dominant)),
    }

# ───────────────────────────────────────────────────────────────────
# Empty Payload (Fallback)
# ───────────────────────────────────────────────────────────────────

@staticmethod
def _empty_payload() -> Dict[str, Any]:
    """Generate empty payload for failed/invalid input."""
    
    return {
        "overall": {
            "n_rows": 0,
            "n_columns": 0,
            "n_numeric": 0,
            "n_categorical": 0,
            "memory_mb": 0.0,
            "sparsity": 0.0,
        },
        "numeric_features": {
            "n_features": 0,
            "features": {},
            "summary": {},
        },
        "categorical_features": {
            "n_features": 0,
            "features": {},
            "summary": {
                "high_cardinality_features": [],
                "dominant_classes_features": [],
            },
        },
        "distributions": {},
        "recommendations": [
            "Provide valid data for statistical analysis"
        ],
        "summary": {
            "n_zero_variance": 0,
            "n_near_constant": 0,
            "n_high_cv": 0,
            "n_high_cardinality": 0,
            "n_dominant_categorical": 0,
        },
        "telemetry": {
            "elapsed_ms": 0.0,
            "timings_ms": {
                "overall": 0.0,
                "numeric": 0.0,
                "categorical": 0.0,
                "distributions": 0.0,
            },
            "sampled_for_tests": False,
            "sample_info": None,
            "caps": {
                "numeric_cols_total": 0,
                "numeric_cols_used": 0,
                "numeric_cols_cap": 0,
                "categorical_cols_total": 0,
                "categorical_cols_used": 0,
                "categorical_cols_cap": 0,
            },
            "skipped_columns": {
                "numeric_empty": [],
                "categorical_empty": [],
            },
        },
        "version": "5.0-kosmos-enterprise",
    }