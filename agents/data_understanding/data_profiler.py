# agents/data_understanding/profiler.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Data Profiler                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Comprehensive data profiling & quality assessment engine:                 ║
║    ✓ Quality scoring (delegated to DataValidator)                          ║
║    ✓ Statistical profiling (shape, types, memory, heuristic datetime)      ║
║    ✓ Quality issues detection (10+ categories with severity levels)        ║
║    ✓ Feature characteristics (smart categorization)                        ║
║    ✓ Correlation analysis (numeric, pearson/spearman, safe sampling)       ║
║    ✓ Dataset hash for caching & comparisons                                ║
║    ✓ Enterprise-grade error handling & performance monitoring              ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "quality_score": float (0..100),
    "quality_details": Dict[str, Any] (from DataValidator),
    "statistical_profile": Dict[str, Any] (comprehensive dataset metadata),
    "quality_issues": List[Dict[str, Any]] (typed issues with severity),
    "feature_characteristics": Dict[str, List[str]] (smart feature categories),
    "correlations": Dict[str, Any] (correlation matrix + high correlations),
}
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple, Literal, Callable, Set
from functools import wraps
import time
import hashlib
import json
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies (provided in project)
try:
    from core.base_agent import BaseAgent, AgentResult
    from core.data_validator import DataValidator
except ImportError:
    # Graceful fallback for testing
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data = None
            self.errors = []
        
        def add_error(self, msg: str):
            self.errors.append(msg)
    
    class DataValidator:
        def get_data_quality_score(self, df):
            return 75.0, {"status": "fallback_mode"}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProfilerConfig:
    """Enterprise profiler configuration with sensible defaults."""
    
    # Missing data thresholds
    high_missing_col_threshold: float = 0.50
    high_missing_flag_threshold: float = 0.30
    
    # Cardinality & constant detection
    high_cardinality_ratio: float = 0.90
    quasi_constant_ratio: float = 0.995
    id_like_unique_ratio: float = 0.98
    
    # Outlier detection (IQR-based)
    outlier_iqr_factor: float = 3.0
    outlier_row_ratio_flag: float = 0.05
    
    # Duplicate & correlation thresholds
    duplicates_row_ratio_mid: float = 0.10
    corr_abs_threshold: float = 0.80
    corr_method: Literal["pearson", "spearman"] = "pearson"
    
    # Performance & safety limits
    max_corr_rows: int = 200_000
    max_corr_cols: int = 200
    corr_nan_policy: Literal["pairwise", "drop"] = "pairwise"
    
    # Text & datetime heuristics
    text_heavy_avg_len: int = 64
    object_datetime_parse_ratio: float = 0.95
    
    # Features & metadata
    enable_mixed_type_check: bool = True
    include_dataset_hash: bool = True
    round_corr: int = 6
    
    # Performance monitoring
    enable_timing: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Data Models (Type-Safe Structures)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StatisticalProfile:
    """Immutable statistical profile snapshot."""
    n_rows: int
    n_columns: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    memory_mb: float
    duplicates: Dict[str, float]
    missing_data: Dict[str, Any]
    dtypes: Dict[str, str]
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class QualityIssue:
    """Typed quality issue representation."""
    type: str
    severity: Literal["info", "low", "medium", "high"]
    column: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with metadata flattened."""
        return {
            "type": self.type,
            "severity": self.severity,
            "column": self.column,
            "description": self.description,
            **self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utilities & Decorators
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str) -> Callable:
    """Decorator for performance monitoring with intelligent logging."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self: "DataProfiler", *args: Any, **kwargs: Any) -> Any:
            if not self.config.enable_timing:
                return fn(self, *args, **kwargs)
            
            t_start = time.perf_counter()
            try:
                return fn(self, *args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(f"⏱ {operation_name}: {elapsed_ms:.2f}ms")
        
        return wrapper
    return decorator


def _safe_operation(operation_name: str, default_value: Any = None) -> Callable:
    """Decorator for defensive operations with fallback values."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠ {operation_name} failed: {type(e).__name__}: {str(e)[:80]}")
                return default_value
        
        return wrapper
    return decorator


@contextmanager
def _suppress_warnings():
    """Context manager for selective warning suppression."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


def _dataset_hash(df: pd.DataFrame, sample_rows: int = 100_000) -> str:
    """
    Generate stable dataset signature for caching & comparisons.
    
    Combines:
      • Column names
      • Up to N sample rows
      • SHA1 hash for deterministic output
    
    Returns deterministic 16-character hash.
    """
    try:
        sample = df if len(df) <= sample_rows else df.sample(n=sample_rows, random_state=42)
        h = hashlib.sha1()
        
        # Hash column names
        h.update("|".join(map(str, df.columns)).encode("utf-8"))
        
        # Hash sample data
        h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
        
        return f"h{h.hexdigest()[:16]}"
    except Exception as e:
        logger.debug(f"hash computation fallback: {e}")
        return f"h{hash((tuple(df.columns), df.shape)) & 0xFFFFFFFF:016X}"


def _is_object_dtype(s: pd.Series) -> bool:
    """Check if series has object or string dtype."""
    return s.dtype == "object" or s.dtype.name == "string"


def _coerce_object_datetime_like(
    s: pd.Series,
    parse_ratio: float = 0.95
) -> Tuple[pd.Series, bool]:
    """
    Heuristic datetime detection for object/string columns.
    
    Returns:
        Tuple[parsed_series, is_datetime]
        - If parse_ratio% of values are parseable to datetime: returns coerced copy + True
        - Otherwise: returns original series + False
        
    Note: No side-effects on input DataFrame.
    """
    if not _is_object_dtype(s):
        return s, False
    
    try:
        sample = s.dropna().astype(str)
        if sample.empty:
            return s, False
        
        # Attempt parse on sample
        parsed_sample = pd.to_datetime(sample, errors="coerce", utc=False)
        success_ratio = float(parsed_sample.notna().sum() / len(parsed_sample))
        
        if success_ratio >= parse_ratio:
            # Parse full series
            parsed_full = pd.to_datetime(s.astype(str), errors="coerce", utc=False)
            return parsed_full, True
        
        return s, False
    
    except Exception as e:
        logger.debug(f"datetime coercion failed: {e}")
        return s, False


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Profiler Agent Class
# ═══════════════════════════════════════════════════════════════════════════

class DataProfiler(BaseAgent):
    """
    **DataProfiler** — Enterprise-grade data profiling & quality assessment.
    
    Responsibilities:
      1. Quality scoring via DataValidator delegation
      2. Statistical profiling (shape, types, memory, cardinality)
      3. Quality issue detection (10+ categories with severity levels)
      4. Feature characteristics categorization
      5. Correlation analysis (numeric, safe sampling)
      6. Dataset hashing for caching & change detection
      7. Comprehensive error handling & performance monitoring
    
    Output format is guaranteed stable (see module docstring).
    """
    
    def __init__(self, config: Optional[ProfilerConfig] = None) -> None:
        """Initialize profiler with optional custom configuration."""
        super().__init__(
            name="DataProfiler",
            description="Enterprise-grade data profiling & quality assessment"
        )
        
        self.config = config or ProfilerConfig()
        self._initialize_validator()
    
    def _initialize_validator(self) -> None:
        """Initialize DataValidator with fallback to degraded mode."""
        try:
            self.validator = DataValidator()
            logger.info("✓ DataValidator initialized")
        except Exception as e:
            logger.warning(f"⚠ DataValidator unavailable: {e} (degraded mode)")
            
            # Null validator for fallback
            class _DegradedValidator:
                def get_data_quality_score(self, *_args, **_kwargs) -> Tuple[float, Dict]:
                    return 50.0, {"status": "validator_unavailable", "mode": "degraded"}
            
            self.validator = _DegradedValidator()
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Raises:
            ValueError: If 'data' missing or wrong type
            TypeError: If not a pandas DataFrame
        """
        if "data" not in kwargs:
            raise ValueError("Required parameter 'data' not provided")
        
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError(f"'data' must be pd.DataFrame, got {type(kwargs['data']).__name__}")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Profile data comprehensively.
        
        Args:
            data: Input DataFrame for profiling
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with profiling information following the output contract.
        """
        result = AgentResult(agent_name=self.name)
        
        try:
            # Input validation
            if data is None or not isinstance(data, pd.DataFrame):
                logger.warning("⚠ DataProfiler: received invalid input (not a DataFrame)")
                result.data = self._empty_payload()
                return result
            
            if data.empty:
                logger.warning("⚠ DataProfiler: received empty DataFrame")
                result.data = self._empty_payload()
                return result
            
            t_start = time.perf_counter()
            
            with _suppress_warnings():
                # 1. Quality Score (delegated to DataValidator)
                quality_score, quality_details = self._get_quality_score(data)
                
                # 2. Statistical Profile
                statistical_profile = self._get_statistical_profile(data)
                
                # 3. Quality Issues
                quality_issues = self._identify_quality_issues(data)
                
                # 4. Feature Characteristics
                feature_characteristics = self._get_feature_characteristics(data)
                
                # 5. Correlations
                correlations = self._get_correlations(data)
            
            # Assemble result
            result.data = {
                "quality_score": float(quality_score),
                "quality_details": quality_details,
                "statistical_profile": statistical_profile,
                "quality_issues": quality_issues,
                "feature_characteristics": feature_characteristics,
                "correlations": correlations,
            }
            
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.success(
                f"✓ DataProfiler complete in {elapsed:.1f}ms | "
                f"rows={data.shape[0]:,} cols={data.shape[1]} | "
                f"quality={quality_score:.1f}/100"
            )
        
        except Exception as e:
            msg = f"Data profiling failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            logger.exception(f"❌ {msg}")
            result.data = self._empty_payload()
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Quality Score (Delegation)
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("quality_score_retrieval", (0.0, {"error": "retrieval_failed"}))
    def _get_quality_score(self, df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """
        Delegate quality scoring to DataValidator.
        
        Returns:
            Tuple of (score: float, details: Dict)
        """
        try:
            score, details = self.validator.get_data_quality_score(df)
            return float(score), details
        except Exception as e:
            logger.warning(f"Quality score computation failed: {e}")
            return 50.0, {"error": "validator_exception", "message": str(e)[:100]}
    
    # ───────────────────────────────────────────────────────────────────
    # Statistical Profile
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("statistical_profile")
    def _get_statistical_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute comprehensive statistical profile.
        
        Includes:
          • Shape & cardinality
          • Type distribution (numeric, categorical, datetime)
          • Missing data patterns
          • Duplicate row detection
          • Memory footprint
          • Optional dataset hash
        """
        n_rows = len(df)
        n_cols = len(df.columns)
        
        # Type detection
        num_cols = set(df.select_dtypes(include=[np.number]).columns)
        cat_cols = set(df.select_dtypes(include=["object", "category", "string"]).columns)
        
        # Heuristic datetime detection (on copies - no side effects)
        dt_like_cols: Set[str] = set()
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                dt_like_cols.add(str(col))
            elif _is_object_dtype(s):
                _, is_dt = _coerce_object_datetime_like(s, self.config.object_datetime_parse_ratio)
                if is_dt:
                    dt_like_cols.add(str(col))
        
        # Memory computation
        try:
            memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
        except Exception:
            memory_mb = float(df.memory_usage().sum() / 1024**2)
        
        # Duplicate analysis
        n_duplicates = int(df.duplicated().sum())
        pct_duplicates = (n_duplicates / n_rows * 100) if n_rows > 0 else 0.0
        
        # Missing data analysis
        total_missing = int(df.isna().sum().sum())
        total_cells = n_rows * n_cols if (n_rows > 0 and n_cols > 0) else 1
        pct_missing = (total_missing / total_cells * 100)
        
        cols_missing = df.isna().sum()
        cols_missing = {str(c): int(v) for c, v in cols_missing[cols_missing > 0].items()}
        
        # Dtype mapping
        dtypes_map = {str(c): str(t) for c, t in df.dtypes.items()}
        
        # Metadata
        meta: Dict[str, Any] = {}
        if self.config.include_dataset_hash:
            meta["dataset_hash"] = _dataset_hash(df)
        
        profile = StatisticalProfile(
            n_rows=int(n_rows),
            n_columns=int(n_cols),
            n_numeric=int(len(num_cols)),
            n_categorical=int(len(cat_cols) - len(dt_like_cols)),
            n_datetime=int(len(dt_like_cols)),
            memory_mb=round(memory_mb, 2),
            duplicates={
                "n_duplicates": n_duplicates,
                "pct_duplicates": round(pct_duplicates, 2),
            },
            missing_data={
                "total_missing": total_missing,
                "pct_missing": round(pct_missing, 2),
                "columns_with_missing": cols_missing,
            },
            dtypes=dtypes_map,
            meta=meta,
        )
        
        return profile.to_dict()
    
    # ───────────────────────────────────────────────────────────────────
    # Quality Issues Detection (10+ categories)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("quality_issues")
    def _identify_quality_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify data quality issues across 10+ categories with severity levels.
        
        Categories:
          1. high_missing_data (>threshold)
          2. constant_column (1 unique value)
          3. quasi_constant (dominant value >99.5%)
          4. high_cardinality (object cols with >90% unique)
          5. id_like (almost all unique values)
          6. duplicates (duplicate rows)
          7. outliers (IQR-based for numeric)
          8. mixed_type_heuristic (numbers + text in object col)
          9. text_heavy (average string length >= threshold)
          10. datetime_monotonic (temporal series increasing/decreasing)
        """
        cfg = self.config
        issues: List[QualityIssue] = []
        n_rows = max(1, len(df))
        
        # ─── Category 1: High Missing Data
        self._detect_high_missing(df, issues, n_rows)
        
        # ─── Category 2-3: Constant & Quasi-Constant
        self._detect_constant_columns(df, issues)
        
        # ─── Category 4-5: Cardinality Issues
        self._detect_cardinality_issues(df, issues, n_rows)
        
        # ─── Category 6: Duplicates
        self._detect_duplicate_rows(df, issues, n_rows)
        
        # ─── Category 7: Outliers (numeric)
        self._detect_outliers(df, issues)
        
        # ─── Category 8: Mixed Types
        if cfg.enable_mixed_type_check:
            self._detect_mixed_types(df, issues)
        
        # ─── Category 9: Text-Heavy
        self._detect_text_heavy(df, issues)
        
        # ─── Category 10: Datetime Monotonicity
        self._detect_datetime_monotonic(df, issues)
        
        # Convert to dicts
        return [issue.to_dict() for issue in issues]
    
    @_safe_operation("high_missing_detection")
    def _detect_high_missing(
        self,
        df: pd.DataFrame,
        issues: List[QualityIssue],
        n_rows: int
    ) -> None:
        """Detect columns with high missing data."""
        cfg = self.config
        missing_counts = df.isna().sum()
        
        for col in missing_counts[missing_counts > cfg.high_missing_col_threshold * n_rows].index:
            pct = float(missing_counts[col] / n_rows * 100)
            issues.append(QualityIssue(
                type="high_missing_data",
                severity="high",
                column=str(col),
                description=f"Missing data: {pct:.1f}% (threshold: {cfg.high_missing_col_threshold*100:.0f}%)",
                metadata={"missing_pct": round(pct, 2)},
            ))
    
    @_safe_operation("constant_detection")
    def _detect_constant_columns(
        self,
        df: pd.DataFrame,
        issues: List[QualityIssue]
    ) -> None:
        """Detect constant and quasi-constant columns."""
        cfg = self.config
        
        for col in df.columns:
            s = df[col]
            try:
                nunique = int(s.nunique(dropna=True))
            except Exception:
                continue
            
            # Constant column
            if nunique <= 1:
                issues.append(QualityIssue(
                    type="constant_column",
                    severity="medium",
                    column=str(col),
                    description="Column has ≤1 unique value",
                ))
            
            # Quasi-constant
            else:
                try:
                    top_freq = s.value_counts(dropna=False).iloc[0]
                    if len(s) > 0 and (top_freq / len(s)) >= cfg.quasi_constant_ratio:
                        issues.append(QualityIssue(
                            type="quasi_constant",
                            severity="low",
                            column=str(col),
                            description=f"Dominant value: {top_freq/len(s)*100:.1f}% (threshold: {cfg.quasi_constant_ratio*100:.1f}%)",
                            metadata={"dominant_freq_ratio": round(top_freq / len(s), 3)},
                        ))
                except Exception:
                    pass
    
    @_safe_operation("cardinality_detection")
    def _detect_cardinality_issues(
        self,
        df: pd.DataFrame,
        issues: List[QualityIssue],
        n_rows: int
    ) -> None:
        """Detect high cardinality and ID-like columns."""
        cfg = self.config
        
        for col in df.columns:
            try:
                nunique = int(df[col].nunique(dropna=True))
            except Exception:
                continue
            
            if n_rows <= 0:
                continue
            
            unique_ratio = nunique / n_rows
            
            # High cardinality (object/category/string)
            if df[col].dtype in ("object", "category", "string") and nunique > cfg.high_cardinality_ratio * n_rows:
                issues.append(QualityIssue(
                    type="high_cardinality",
                    severity="low",
                    column=str(col),
                    description=f"High cardinality: {nunique} unique values ({unique_ratio*100:.1f}%)",
                    metadata={"nunique": nunique, "ratio": round(unique_ratio, 3)},
                ))
            
            # ID-like (almost all unique)
            if unique_ratio >= cfg.id_like_unique_ratio:
                issues.append(QualityIssue(
                    type="id_like",
                    severity="info",
                    column=str(col),
                    description=f"ID-like column: {unique_ratio*100:.1f}% unique values",
                    metadata={"unique_ratio": round(unique_ratio, 3)},
                ))
    
    @_safe_operation("duplicate_detection")
    def _detect_duplicate_rows(
        self,
        df: pd.DataFrame,
        issues: List[QualityIssue],
        n_rows: int
    ) -> None:
        """Detect duplicate rows."""
        cfg = self.config
        n_dups = int(df.duplicated().sum())
        
        if n_dups > 0:
            dup_ratio = n_dups / n_rows
            severity = "medium" if dup_ratio > cfg.duplicates_row_ratio_mid else "low"
            
            issues.append(QualityIssue(
                type="duplicates",
                severity=severity,
                description=f"Duplicate rows: {n_dups} ({dup_ratio*100:.2f}%)",
                metadata={"n_duplicates": n_dups, "pct_duplicates": round(dup_ratio*100, 2)},
            ))
    
    @_safe_operation("outlier_detection")
    def _detect_outliers(self, df: pd.DataFrame, issues: List[QualityIssue]) -> None:
        """Detect outliers in numeric columns using IQR method."""
        cfg = self.config
        
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if s.empty:
                continue
            
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            
            if iqr <= 0:
                continue
            
            lower = q1 - cfg.outlier_iqr_factor * iqr
            upper = q3 + cfg.outlier_iqr_factor * iqr
            
            n_outliers = int(((s < lower) | (s > upper)).sum())
            outlier_ratio = n_outliers / len(s) if len(s) > 0 else 0
            
            if outlier_ratio > cfg.outlier_row_ratio_flag:
                issues.append(QualityIssue(
                    type="outliers",
                    severity="low",
                    column=str(col),
                    description=f"Outliers detected: {n_outliers} ({outlier_ratio*100:.2f}%)",
                    metadata={
                        "n_outliers": n_outliers,
                        "outlier_pct": round(outlier_ratio*100, 2),
                        "bounds": {"lower": round(lower, 4), "upper": round(upper, 4)},
                    },
                ))
    
    @_safe_operation("mixed_type_detection")
    def _detect_mixed_types(self, df: pd.DataFrame, issues: List[QualityIssue]) -> None:
        """Detect mixed types (numeric + text) in object columns."""
        for col in df.select_dtypes(include=["object", "string"]).columns:
            s = df[col].dropna()
            if s.empty:
                continue
            
            def _is_numeric_like(x: Any) -> bool:
                if isinstance(x, (int, float, np.integer, np.floating)):
                    return True
                if isinstance(x, str):
                    try:
                        float(x.replace(",", "."))
                        return True
                    except (ValueError, AttributeError):
                        return False
                return False
            
            has_numeric = s.map(_is_numeric_like).any()
            has_alpha = s.map(lambda x: isinstance(x, str) and any(c.isalpha() for c in x)).any()
            
            if has_numeric and has_alpha:
                issues.append(QualityIssue(
                    type="mixed_type_heuristic",
                    severity="low",
                    column=str(col),
                    description="Column appears to have mixed types (numeric + text)",
                ))
    
    @_safe_operation("text_heavy_detection")
    def _detect_text_heavy(self, df: pd.DataFrame, issues: List[QualityIssue]) -> None:
        """Detect columns with heavy text (long average string length)."""
        cfg = self.config
        
        for col in df.select_dtypes(include=["object", "string"]).columns:
            s = df[col].dropna().astype(str)
            if s.empty:
                continue
            
            avg_len = float(s.str.len().mean())
            if avg_len >= cfg.text_heavy_avg_len:
                issues.append(QualityIssue(
                    type="text_heavy",
                    severity="info",
                    column=str(col),
                    description=f"Heavy text column: average length {avg_len:.1f}",
                    metadata={"avg_length": round(avg_len, 1)},
                ))
    
    @_safe_operation("datetime_monotonic_detection")
    def _detect_datetime_monotonic(self, df: pd.DataFrame, issues: List[QualityIssue]) -> None:
        """Detect datetime columns that are monotonically increasing/decreasing."""
        for col in df.columns: 
            s = df[col]
            
            # Check if datetime type
            if pd.api.types.is_datetime64_any_dtype(s):
                sdt = s.dropna()
            elif _is_object_dtype(s):
                # Attempt heuristic parsing
                coerced, is_dt = _coerce_object_datetime_like(s, self.config.object_datetime_parse_ratio)
                if not is_dt:
                    continue
                sdt = coerced.dropna()
            else:
                continue
            
            # Check monotonicity
            if len(sdt) > 1 and (sdt.is_monotonic_increasing or sdt.is_monotonic_decreasing):
                direction = "increasing" if sdt.is_monotonic_increasing else "decreasing"
                issues.append(QualityIssue(
                    type="datetime_monotonic",
                    severity="info",
                    column=str(col),
                    description=f"Datetime column is monotonically {direction}",
                    metadata={"direction": direction},
                ))
    
    # ───────────────────────────────────────────────────────────────────
    # Feature Characteristics Categorization
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("feature_characteristics")
    def _get_feature_characteristics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features by type and characteristics.
        
        Returns dictionary with categories:
          • numeric: numeric dtypes
          • categorical: object/category/string (non-datetime)
          • datetime: temporal types + heuristic datetime
          • binary: exactly 2 unique values
          • constant: ≤1 unique value
          • quasi_constant: dominant value >99.5%
          • high_cardinality: >90% unique for object cols
          • id_like: >98% unique values
          • high_missing: >30% missing values
          • text_heavy: average string length >= threshold
        """
        cfg = self.config
        chars: Dict[str, List[str]] = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "binary": [],
            "constant": [],
            "quasi_constant": [],
            "high_cardinality": [],
            "id_like": [],
            "high_missing": [],
            "text_heavy": [],
        }
        
        n_rows = len(df)
        
        for col in df.columns:
            col_str = str(col)
            s = df[col]
            
            # ─── Type Detection
            if pd.api.types.is_numeric_dtype(s):
                chars["numeric"].append(col_str)
            elif pd.api.types.is_datetime64_any_dtype(s):
                chars["datetime"].append(col_str)
            elif _is_object_dtype(s):
                _, is_dt = _coerce_object_datetime_like(s, cfg.object_datetime_parse_ratio)
                if is_dt:
                    chars["datetime"].append(col_str)
                else:
                    chars["categorical"].append(col_str)
            else:
                chars["categorical"].append(col_str)
            
            # ─── Cardinality Analysis
            try:
                nunique = int(s.nunique(dropna=True))
                nunique_all = int(s.nunique(dropna=False))
            except Exception:
                nunique = nunique_all = 0
            
            if nunique_all <= 1:
                chars["constant"].append(col_str)
            elif nunique == 2:
                chars["binary"].append(col_str)
            elif n_rows > 0 and (nunique > cfg.high_cardinality_ratio * n_rows):
                chars["high_cardinality"].append(col_str)
            
            # ─── Quasi-Constant Detection
            if nunique_all > 1:
                try:
                    top_freq = s.value_counts(dropna=False).iloc[0]
                    if (top_freq / len(s)) >= cfg.quasi_constant_ratio:
                        chars["quasi_constant"].append(col_str)
                except Exception:
                    pass
            
            # ─── ID-Like Detection
            if n_rows > 0 and (nunique / n_rows) >= cfg.id_like_unique_ratio:
                chars["id_like"].append(col_str)
            
            # ─── Missing Data
            if n_rows > 0:
                missing_pct = s.isna().sum() / n_rows
                if missing_pct > cfg.high_missing_flag_threshold:
                    chars["high_missing"].append(col_str)
            
            # ─── Text-Heavy Detection
            if _is_object_dtype(s):
                text_s = s.dropna().astype(str)
                if not text_s.empty:
                    avg_len = float(text_s.str.len().mean())
                    if avg_len >= cfg.text_heavy_avg_len:
                        chars["text_heavy"].append(col_str)
        
        return chars
    
    # ───────────────────────────────────────────────────────────────────
    # Correlation Analysis (Numeric)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("correlations")
    def _get_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute correlation matrix for numeric features.
        
        Features:
          • Safe sampling for large datasets (>200k rows)
          • Configurable method (pearson/spearman)
          • NaN handling policies (pairwise/drop)
          • Automatic inf/NaN cleaning
          • High correlation detection (threshold configurable)
          • Soft limits on column count (O(k²) safety)
        
        Returns:
            Dict with correlation_matrix, high_correlations, metadata
        """
        cfg = self.config
        
        # Get numeric columns
        num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        k = len(num_cols_all)
        
        # Check if insufficient data
        if k < 2:
            return {
                "n_numeric_features": k,
                "correlation_matrix": None,
                "high_correlations": [],
                "n_high_correlations": 0,
            }
        
        # Apply soft column limit
        num_cols = num_cols_all[:cfg.max_corr_cols] if k > cfg.max_corr_cols else num_cols_all
        
        if k > cfg.max_corr_cols:
            logger.warning(
                f"⚠ Correlation: limiting columns from {k} to {len(num_cols)} (O(k²) safety)"
            )
        
        # Sample rows if necessary
        if len(df) > cfg.max_corr_rows:
            df_corr = df[num_cols].sample(n=cfg.max_corr_rows, random_state=42).copy()
            logger.debug(f"⚠ Correlation: sampling {cfg.max_corr_rows} rows from {len(df)}")
        else:
            df_corr = df[num_cols].copy()
        
        # Apply NaN policy
        if cfg.corr_nan_policy == "drop":
            df_corr = df_corr.dropna()
        
        # Check if data remains
        if df_corr.empty or df_corr.shape[1] < 2:
            return {
                "n_numeric_features": len(num_cols),
                "correlation_matrix": None,
                "high_correlations": [],
                "n_high_correlations": 0,
            }
        
        # Clean inf/NaN (replace with NaN)
        df_corr = df_corr.replace([np.inf, -np.inf], np.nan)
        
        # Compute correlation
        try:
            corr_matrix = df_corr.corr(method=cfg.corr_method, numeric_only=True)
        except Exception:
            logger.warning("⚠ Correlation method failed, falling back to pearson")
            corr_matrix = df_corr.corr(method="pearson", numeric_only=True)
        
        # Clean correlation matrix
        if cfg.corr_nan_policy == "drop":
            corr_matrix = corr_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
        
        # Check if data remains after cleaning
        if corr_matrix.empty or corr_matrix.shape[1] < 2:
            return {
                "n_numeric_features": len(num_cols),
                "correlation_matrix": None,
                "high_correlations": [],
                "n_high_correlations": 0,
            }
        
        # Sanitize values
        corr_matrix = corr_matrix.fillna(0.0).clip(-1.0, 1.0).round(cfg.round_corr)
        
        # Find high correlations
        high_corr: List[Dict[str, Any]] = []
        cols = corr_matrix.columns
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = float(corr_matrix.iloc[i, j])
                if abs(val) >= cfg.corr_abs_threshold:
                    high_corr.append({
                        "feature1": str(cols[i]),
                        "feature2": str(cols[j]),
                        "correlation": val,
                    })
        
        return {
            "n_numeric_features": len(num_cols),
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": high_corr,
            "n_high_correlations": len(high_corr),
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Empty Payload (Fallback)
    # ───────────────────────────────────────────────────────────────────
    
    def _empty_payload(self) -> Dict[str, Any]:
        """Generate empty/default payload for failed/empty inputs."""
        return {
            "quality_score": 0.0,
            "quality_details": {"reason": "empty_or_invalid_input"},
            "statistical_profile": {
                "n_rows": 0,
                "n_columns": 0,
                "n_numeric": 0,
                "n_categorical": 0,
                "n_datetime": 0,
                "memory_mb": 0.0,
                "duplicates": {"n_duplicates": 0, "pct_duplicates": 0.0},
                "missing_data": {
                    "total_missing": 0,
                    "pct_missing": 0.0,
                    "columns_with_missing": {},
                },
                "dtypes": {},
                "meta": {},
            },
            "quality_issues": [],
            "feature_characteristics": {
                "numeric": [],
                "categorical": [],
                "datetime": [],
                "binary": [],
                "constant": [],
                "quasi_constant": [],
                "high_cardinality": [],
                "id_like": [],
                "high_missing": [],
                "text_heavy": [],
            },
            "correlations": {
                "n_numeric_features": 0,
                "correlation_matrix": None,
                "high_correlations": [],
                "n_high_correlations": 0,
            },
        }