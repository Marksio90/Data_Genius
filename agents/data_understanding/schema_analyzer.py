# agents/data_understanding/schema_analyzer.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Schema Analyzer                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise data structure analysis & metadata extraction:                 ║
║    ✓ Comprehensive column profiling (dtype, cardinality, statistics)       ║
║    ✓ Semantic type detection (numeric, categorical, datetime)             ║
║    ✓ Quality heuristics (mixed types, spaces, normalization)              ║
║    ✓ Memory usage breakdown & downcast recommendations                    ║
║    ✓ Smart primary key detection (single & composite)                     ║
║    ✓ Type casting recommendations (object→numeric/datetime)               ║
║    ✓ Encoder recommendations (OneHot, Binary, TargetEnc, DateParts)      ║
║    ✓ Schema fingerprinting for change detection & caching                 ║
║    ✓ Graceful error handling with defensive programming                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "basic_info": {
        "n_rows": int,
        "n_columns": int,
        "column_names": List[str],
        "shape": Tuple[int, int],
        "size": int,
    },
    "columns": List[ColumnInfoDict],  # Detailed per-column metadata
    "dtypes_summary": Dict[str, int],  # dtype distribution
    "memory_info": {
        "total_mb": float,
        "per_row_bytes": float,
        "by_column_mb": Dict[str, float],
    },
    "suggestions": {
        "potential_primary_keys": List[str],
        "potential_casts": List[{"column": str, "from": str, "to": str}],
        "downcast_hints": List[{"column": str, "from": str, "to": str, "est_saving_mb": float}],
        "encoder_recommendations": List[{"column": str, "encoder": str, "reason": str}],
        "warnings": List[str],
    },
    "schema_fingerprint": str,  # SHA1 hash for change detection
}
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from functools import wraps

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
    from core.utils import (
        detect_column_type,
        get_numeric_columns,
        get_categorical_columns,
    )
except ImportError:
    # Fallback for standalone usage
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
    
    def detect_column_type(series):
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "categorical"
    
    def get_numeric_columns(df):
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(df):
        return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SchemaAnalyzerConfig:
    """Enterprise configuration for schema analysis."""
    
    # Data quality thresholds
    high_missing_ratio_flag: float = 0.30
    high_cardinality_ratio: float = 0.90
    quasi_constant_ratio: float = 0.995
    
    # Primary key detection
    potential_pk_unique_ratio: float = 0.999
    potential_pk_max_nulls: int = 0
    enable_composite_pk_search: bool = True
    max_cols_for_composite_pk: int = 30
    max_rows_for_composite_pk: int = 100_000
    
    # Type detection & conversion
    id_like_unique_ratio: float = 0.98
    text_heavy_avg_len: int = 64
    cast_object_numeric_ratio: float = 0.90
    cast_object_datetime_ratio: float = 0.90
    
    # Performance & safety
    numeric_zero_count_limit: int = 10**9
    include_dataset_hash: bool = True
    truncate_log_chars: int = 500


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Data Models (Type-Safe)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnInfo:
    """Comprehensive column metadata."""
    name: str
    dtype: str
    semantic_type: Optional[str]
    n_unique: int
    n_missing: int
    missing_pct: float
    n_zeros: int
    extras: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str) -> Callable:
    """Decorator for operation timing and performance monitoring."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t_start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(f"⏱ {operation_name}: {elapsed_ms:.2f}ms")
        
        return wrapper
    return decorator


def _dataset_hash(df: pd.DataFrame, sample_rows: int = 100_000) -> str:
    """
    Generate stable dataset signature for caching & change detection.
    
    Combines:
      • Column names
      • Up to N sample rows
      • Deterministic SHA1 hash
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


def _safe_numeric_coerce(series: pd.Series) -> pd.Series:
    """
    Safely coerce series to numeric (handles decimals with commas, preserves NaN).
    """
    try:
        s = pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
        return s
    except Exception:
        return pd.to_numeric(series, errors="coerce")


def _safe_json_str(obj: Any, limit: int = 500) -> str:
    """Safely convert object to truncated JSON string for logging."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    
    if len(s) <= limit:
        return s
    return s[:limit] + f"...(+{len(s)-limit} chars)"


def _is_mixed_object_types(s: pd.Series) -> bool:
    """
    Heuristic: mixed types in object column (e.g., numeric strings + alpha).
    """
    if s.dtype != "object":
        return False
    
    ss = s.dropna().astype(str)
    if ss.empty:
        return False
    
    # Check for numeric-like values
    has_numeric_like = ss.map(
        lambda x: x.replace(".", "", 1).replace("-", "", 1).isdigit()
    ).any()
    
    # Check for alphabetic values
    has_alpha_like = ss.map(lambda x: any(c.isalpha() for c in x)).any()
    
    return bool(has_numeric_like and has_alpha_like)


def _is_id_like(s: pd.Series, ratio_threshold: float = 0.98) -> bool:
    """Detect ID-like columns (almost all unique values)."""
    try:
        n_unique = int(s.nunique(dropna=True))
        n_total = int(len(s))
        
        if n_total <= 0:
            return False
        
        return (n_unique / n_total) >= ratio_threshold
    
    except Exception:
        return False


def _has_leading_trailing_spaces(s: pd.Series) -> bool:
    """Check if object column contains leading/trailing spaces."""
    if s.dtype != "object":
        return False
    
    try:
        ss = s.dropna().astype(str)
        if ss.empty:
            return False
        
        # Compare original vs stripped
        return bool((ss != ss.str.strip()).any())
    
    except Exception:
        return False


def _is_non_normalized_categories(s: pd.Series) -> bool:
    """
    Heuristic: detect non-normalized categories (case/spacing variants).
    
    Example: "Apple", "apple", "APPLE" would be detected as non-normalized.
    """
    if s.dtype not in ("object",) and not pd.api.types.is_categorical_dtype(s):
        return False
    
    try:
        ss = s.dropna().astype(str)
        if ss.empty:
            return False
        
        # Normalize: lowercase + strip
        normalized = ss.str.strip().str.lower()
        
        # If normalized has fewer unique values → originals had variants
        return int(normalized.nunique()) < int(ss.nunique())
    
    except Exception:
        return False


def _object_numeric_ratio(s: pd.Series) -> float:
    """Estimate ratio of parseable numeric values in object column."""
    try:
        ss = _safe_numeric_coerce(s.dropna().astype(str))
        return float(ss.notna().mean()) if len(ss) else 0.0
    except Exception:
        return 0.0


def _object_datetime_ratio(s: pd.Series) -> float:
    """Estimate ratio of parseable datetime values in object column."""
    try:
        ss = pd.to_datetime(s.dropna().astype(str), errors="coerce", utc=False)
        return float(ss.notna().mean()) if len(ss) else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Schema Analyzer Agent
# ═══════════════════════════════════════════════════════════════════════════

class SchemaAnalyzer(BaseAgent):
    """
    **SchemaAnalyzer** — Enterprise data structure and schema analysis.
    
    Responsibilities:
      1. Comprehensive column profiling (dtypes, cardinality, statistics)
      2. Semantic type detection (numeric, categorical, datetime)
      3. Quality heuristics (mixed types, spaces, normalization)
      4. Memory analysis & downcast recommendations
      5. Primary key detection (single & composite)
      6. Type casting recommendations
      7. Encoder recommendations (OneHot, Binary, TargetEnc, DateParts)
      8. Schema fingerprinting for change detection
      9. Stable 1:1 contract output
      10. Zero side-effects on input DataFrame
    
    Output format is deterministic and consistent.
    """
    
    def __init__(self, config: Optional[SchemaAnalyzerConfig] = None) -> None:
        """Initialize analyzer with optional custom configuration."""
        super().__init__(
            name="SchemaAnalyzer",
            description="Analyzes data structure and column types"
        )
        self.config = config or SchemaAnalyzerConfig()
        self._log = logger.bind(agent="SchemaAnalyzer")
        warnings.filterwarnings("ignore")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            data: pd.DataFrame
        """
        if "data" not in kwargs:
            raise ValueError("Required parameter 'data' not provided")
        
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError(f"'data' must be pd.DataFrame, got {type(kwargs['data']).__name__}")
        
        # Note: we accept empty DataFrames (return empty payload)
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("SchemaAnalyzer.execute")
    def execute(self, data: pd.DataFrame, **kwargs: Any) -> AgentResult:
        """
        Analyze data schema comprehensively.
        
        Args:
            data: Input DataFrame (not modified)
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with schema information (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        
        try:
            # Handle empty DataFrame
            if data is None or data.empty:
                self._log.warning("⚠ Received empty DataFrame")
                result.data = self._empty_payload()
                return result
            
            # 1. Basic information
            basic_info = self._get_basic_info(data)
            
            # 2. Detailed column analysis
            columns_info = self._analyze_columns(data)
            
            # 3. Data types summary
            dtypes_summary = self._get_dtypes_summary(data)
            
            # 4. Memory usage breakdown
            memory_info = self._get_memory_info(data)
            
            # 5. Heuristic suggestions
            suggestions = self._get_suggestions(data, columns_info)
            
            # 6. Schema fingerprint
            schema_fingerprint = self._schema_fingerprint(data)
            
            # Add dataset hash to metadata if configured
            if self.config.include_dataset_hash:
                if "meta" not in basic_info:
                    basic_info["meta"] = {}
                basic_info["meta"]["dataset_hash"] = _dataset_hash(data)
            
            # Assemble result
            result.data = {
                "basic_info": basic_info,
                "columns": columns_info,
                "dtypes_summary": dtypes_summary,
                "memory_info": memory_info,
                "suggestions": suggestions,
                "schema_fingerprint": schema_fingerprint,
            }
            
            self._log.success(
                f"✓ Schema analysis complete | "
                f"cols={len(data.columns)} rows={len(data):,} "
                f"memory={memory_info['total_mb']:.1f}MB"
            )
        
        except Exception as e:
            msg = f"Schema analysis failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            result.data = self._empty_payload()
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Basic Information
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("basic_info")
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic DataFrame metadata."""
        return {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns.tolist()],
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "size": int(df.size),
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Column Analysis (Per-Column Metadata)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("analyze_columns")
    def _analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze each column in detail.
        
        Includes:
          • Basic stats (unique, missing, zeros)
          • Type-specific statistics (numeric, categorical, datetime)
          • Quality heuristics (mixed types, spaces, cardinality, etc.)
        """
        cfg = self.config
        columns_info: List[Dict[str, Any]] = []
        n_rows = max(1, len(df))
        
        for col in df.columns:
            s = df[col]
            
            # Basic dtype and cardinality
            try:
                dtype_str = str(s.dtype)
            except Exception:
                dtype_str = "unknown"
            
            try:
                n_unique = int(s.nunique(dropna=True))
            except Exception:
                n_unique = 0
            
            try:
                n_missing = int(s.isna().sum())
            except Exception:
                n_missing = 0
            
            missing_pct = float((n_missing / n_rows) * 100.0) if n_rows > 0 else 0.0
            
            # Count zeros (only for numeric/bool)
            n_zeros = 0
            try:
                if pd.api.types.is_bool_dtype(s):
                    n_zeros = int((~s.fillna(False)).sum())
                elif pd.api.types.is_numeric_dtype(s):
                    sn = pd.to_numeric(s, errors="coerce")
                    n_zeros = int((sn == 0).sum())
                
                if n_zeros > cfg.numeric_zero_count_limit:
                    n_zeros = 0
            except Exception:
                n_zeros = 0
            
            # Semantic type detection
            try:
                semantic_type = detect_column_type(s)
            except Exception:
                semantic_type = None
            
            # Initialize extras
            extras: Dict[str, Any] = {}
            
            # Type-specific analysis
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                extras.update(self._get_numeric_stats(s))
            
            elif s.dtype in ("object",) or pd.api.types.is_categorical_dtype(s):
                extras.update(self._get_categorical_stats(s))
                
                # Additional heuristics for categorical/object
                try:
                    # Quasi-constant detection
                    vc = s.dropna().astype(str).value_counts()
                    if not vc.empty and len(s.dropna()) > 0:
                        top_freq = int(vc.iloc[0])
                        if (top_freq / len(s.dropna())) >= cfg.quasi_constant_ratio and s.nunique(dropna=False) > 1:
                            extras["quasi_constant"] = True
                    
                    # Mixed types
                    if _is_mixed_object_types(s):
                        extras["mixed_type"] = True
                    
                    # Text-heavy
                    ss = s.dropna().astype(str)
                    if not ss.empty and float(ss.str.len().mean()) >= cfg.text_heavy_avg_len:
                        extras["text_heavy"] = True
                    
                    # Leading/trailing spaces
                    if _has_leading_trailing_spaces(s):
                        extras["has_leading_trailing_spaces"] = True
                    
                    # Non-normalized categories
                    if _is_non_normalized_categories(s):
                        extras["non_normalized_categories"] = True
                
                except Exception as e:
                    logger.debug(f"categorical extras failed for '{col}': {e}")
            
            # Datetime analysis
            if pd.api.types.is_datetime64_any_dtype(s):
                extras.update(self._get_datetime_stats(s))
                
                # Monotonicity check
                try:
                    sd = pd.to_datetime(s, errors="coerce").dropna()
                    if len(sd) > 1 and (sd.is_monotonic_increasing or sd.is_monotonic_decreasing):
                        extras["monotonic"] = True
                except Exception:
                    pass
            
            # ID-like detection
            try:
                if _is_id_like(s, cfg.id_like_unique_ratio):
                    extras["id_like"] = True
            except Exception:
                pass
            
            # High cardinality (only for categorical/object)
            try:
                if (s.dtype in ("object",) or pd.api.types.is_categorical_dtype(s)) and n_rows > 0:
                    if (n_unique > cfg.high_cardinality_ratio * n_rows):
                        extras["high_cardinality"] = True
            except Exception:
                pass
            
            # Create column info
            col_info = ColumnInfo(
                name=str(col),
                dtype=dtype_str,
                semantic_type=semantic_type,
                n_unique=n_unique,
                n_missing=n_missing,
                missing_pct=missing_pct,
                n_zeros=n_zeros,
                extras=extras,
            )
            
            columns_info.append(col_info.to_dict())
        
        return columns_info
    
    # ───────────────────────────────────────────────────────────────────
    # Type-Specific Statistics
    # ───────────────────────────────────────────────────────────────────
    
    def _get_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for numeric/boolean columns."""
        try:
            if pd.api.types.is_bool_dtype(series):
                s = series.astype("Int8").astype(float)
            else:
                s = _safe_numeric_coerce(series)
        except Exception:
            s = _safe_numeric_coerce(series)
        
        # Check if all NaN after coercion
        if s.notna().sum() == 0:
            return {
                "mean": None, "std": None, "min": None, "max": None,
                "median": None, "q25": None, "q75": None,
                "skewness": None, "kurtosis": None,
            }
        
        # Clean inf values
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        
        if s.empty:
            return {
                "mean": None, "std": None, "min": None, "max": None,
                "median": None, "q25": None, "q75": None,
                "skewness": None, "kurtosis": None,
            }
        
        return {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std(ddof=1)) if s.count() > 1 else 0.0, 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "median": round(float(s.median()), 6),
            "q25": round(float(s.quantile(0.25)), 6),
            "q75": round(float(s.quantile(0.75)), 6),
            "skewness": round(float(s.skew()) if s.count() > 2 else 0.0, 3),
            "kurtosis": round(float(s.kurtosis()) if s.count() > 3 else 0.0, 3),
        }
    
    def _get_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for categorical/object columns."""
        try:
            s = series.astype("string")
            vc = s.value_counts(dropna=True)
        except Exception:
            vc = series.value_counts(dropna=True)
        
        # Mode
        mode_val = None
        try:
            mode_series = series.mode(dropna=True)
            if not mode_series.empty:
                mode_val = str(mode_series.iloc[0])
        except Exception:
            pass
        
        n_categories = int(len(vc))
        is_binary = bool(n_categories == 2)
        
        return {
            "mode": mode_val,
            "top_values": {str(k): int(v) for k, v in vc.head(5).to_dict().items()},
            "n_categories": n_categories,
            "is_binary": is_binary,
            "high_cardinality": bool(n_categories > 0.90 * max(1, len(series))),
        }
    
    def _get_datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for datetime columns."""
        try:
            s = pd.to_datetime(series, errors="coerce")
            s = s.dropna()
        except Exception:
            return {
                "min_date": None,
                "max_date": None,
                "date_range_days": None,
                "maybe_mixed_timezones": False,
            }
        
        if s.empty:
            return {
                "min_date": None,
                "max_date": None,
                "date_range_days": None,
                "maybe_mixed_timezones": False,
            }
        
        min_d = s.min()
        max_d = s.max()
        
        # Detect mixed timezones (heuristic)
        tz_mixed = False
        try:
            has_tz = getattr(series.dtype, "tz", None) is not None
            has_naive = any(
                getattr(x, "tzinfo", None) is None
                for x in series.dropna().tolist()
            )
            tz_mixed = bool(has_tz and has_naive)
        except Exception:
            pass
        
        return {
            "min_date": str(min_d),
            "max_date": str(max_d),
            "date_range_days": int((max_d - min_d).days),
            "maybe_mixed_timezones": tz_mixed,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Data Types Summary
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("dtypes_summary")
    def _get_dtypes_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Summarize pandas dtypes distribution."""
        try:
            vc = df.dtypes.value_counts()
            return {str(dtype): int(count) for dtype, count in vc.items()}
        except Exception:
            # Fallback
            types = [str(t) for t in df.dtypes]
            vc = pd.Series(types).value_counts()
            return {str(dtype): int(count) for dtype, count in vc.items()}
    
    # ───────────────────────────────────────────────────────────────────
    # Memory Usage
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("memory_info")
    def _get_memory_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute memory usage breakdown."""
        try:
            mem = df.memory_usage(deep=True)
        except Exception:
            mem = df.memory_usage(deep=False)
        
        total_bytes = int(mem.sum())
        
        return {
            "total_mb": round(total_bytes / (1024**2), 2),
            "per_row_bytes": round(total_bytes / max(1, len(df)), 2),
            "by_column_mb": {
                str(col): round(val / (1024**2), 3)
                for col, val in mem.items()
            },
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Heuristic Suggestions
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("suggestions")
    def _get_suggestions(
        self,
        df: pd.DataFrame,
        columns_info: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate heuristic suggestions for:
          • Primary keys (single & composite)
          • Type casting (object→numeric/datetime)
          • Memory downcasting
          • Feature encoding recommendations
          • Data quality warnings
        """
        cfg = self.config
        n_rows = max(1, len(df))
        
        suggestions: Dict[str, Any] = {
            "potential_primary_keys": [],
            "potential_casts": [],
            "downcast_hints": [],
            "encoder_recommendations": [],
            "warnings": [],
        }
        
        # Get column lists
        cat_cols = get_categorical_columns(df)
        num_cols = get_numeric_columns(df)
        
        # ─── Primary Key Detection (Single Column)
        self._find_primary_keys_single(df, columns_info, n_rows, suggestions)
        
        # ─── Primary Key Detection (Composite)
        if cfg.enable_composite_pk_search:
            self._find_primary_keys_composite(df, columns_info, n_rows, suggestions)
        
        # ─── Type Casting Recommendations
        self._find_type_casts(df, cat_cols, suggestions)
        
        # ─── Memory Downcast Hints
        self._find_downcasts(df, num_cols, suggestions)
        
        # ─── Data Quality Warnings
        self._find_warnings(columns_info, suggestions)
        
        # ─── Encoder Recommendations
        self._find_encoders(df, columns_info, suggestions)
        
        return suggestions
    
    # agents/data_understanding/schema_analyzer.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Schema Analyzer                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise data structure analysis & metadata extraction:                 ║
║    ✓ Comprehensive column profiling (dtype, cardinality, statistics)       ║
║    ✓ Semantic type detection (numeric, categorical, datetime)             ║
║    ✓ Quality heuristics (mixed types, spaces, normalization)              ║
║    ✓ Memory usage breakdown & downcast recommendations                    ║
║    ✓ Smart primary key detection (single & composite)                     ║
║    ✓ Type casting recommendations (object→numeric/datetime)               ║
║    ✓ Encoder recommendations (OneHot, Binary, TargetEnc, DateParts)      ║
║    ✓ Schema fingerprinting for change detection & caching                 ║
║    ✓ Graceful error handling with defensive programming                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "basic_info": {
        "n_rows": int,
        "n_columns": int,
        "column_names": List[str],
        "shape": Tuple[int, int],
        "size": int,
    },
    "columns": List[ColumnInfoDict],  # Detailed per-column metadata
    "dtypes_summary": Dict[str, int],  # dtype distribution
    "memory_info": {
        "total_mb": float,
        "per_row_bytes": float,
        "by_column_mb": Dict[str, float],
    },
    "suggestions": {
        "potential_primary_keys": List[str],
        "potential_casts": List[{"column": str, "from": str, "to": str}],
        "downcast_hints": List[{"column": str, "from": str, "to": str, "est_saving_mb": float}],
        "encoder_recommendations": List[{"column": str, "encoder": str, "reason": str}],
        "warnings": List[str],
    },
    "schema_fingerprint": str,  # SHA1 hash for change detection
}
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from functools import wraps

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
    from core.utils import (
        detect_column_type,
        get_numeric_columns,
        get_categorical_columns,
    )
except ImportError:
    # Fallback for standalone usage
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
    
    def detect_column_type(series):
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "categorical"
    
    def get_numeric_columns(df):
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(df):
        return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SchemaAnalyzerConfig:
    """Enterprise configuration for schema analysis."""
    
    # Data quality thresholds
    high_missing_ratio_flag: float = 0.30
    high_cardinality_ratio: float = 0.90
    quasi_constant_ratio: float = 0.995
    
    # Primary key detection
    potential_pk_unique_ratio: float = 0.999
    potential_pk_max_nulls: int = 0
    enable_composite_pk_search: bool = True
    max_cols_for_composite_pk: int = 30
    max_rows_for_composite_pk: int = 100_000
    
    # Type detection & conversion
    id_like_unique_ratio: float = 0.98
    text_heavy_avg_len: int = 64
    cast_object_numeric_ratio: float = 0.90
    cast_object_datetime_ratio: float = 0.90
    
    # Performance & safety
    numeric_zero_count_limit: int = 10**9
    include_dataset_hash: bool = True
    truncate_log_chars: int = 500


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Data Models (Type-Safe)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnInfo:
    """Comprehensive column metadata."""
    name: str
    dtype: str
    semantic_type: Optional[str]
    n_unique: int
    n_missing: int
    missing_pct: float
    n_zeros: int
    extras: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str) -> Callable:
    """Decorator for operation timing and performance monitoring."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t_start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(f"⏱ {operation_name}: {elapsed_ms:.2f}ms")
        
        return wrapper
    return decorator


def _dataset_hash(df: pd.DataFrame, sample_rows: int = 100_000) -> str:
    """
    Generate stable dataset signature for caching & change detection.
    
    Combines:
      • Column names
      • Up to N sample rows
      • Deterministic SHA1 hash
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


def _safe_numeric_coerce(series: pd.Series) -> pd.Series:
    """
    Safely coerce series to numeric (handles decimals with commas, preserves NaN).
    """
    try:
        s = pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
        return s
    except Exception:
        return pd.to_numeric(series, errors="coerce")


def _safe_json_str(obj: Any, limit: int = 500) -> str:
    """Safely convert object to truncated JSON string for logging."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    
    if len(s) <= limit:
        return s
    return s[:limit] + f"...(+{len(s)-limit} chars)"


def _is_mixed_object_types(s: pd.Series) -> bool:
    """
    Heuristic: mixed types in object column (e.g., numeric strings + alpha).
    """
    if s.dtype != "object":
        return False
    
    ss = s.dropna().astype(str)
    if ss.empty:
        return False
    
    # Check for numeric-like values
    has_numeric_like = ss.map(
        lambda x: x.replace(".", "", 1).replace("-", "", 1).isdigit()
    ).any()
    
    # Check for alphabetic values
    has_alpha_like = ss.map(lambda x: any(c.isalpha() for c in x)).any()
    
    return bool(has_numeric_like and has_alpha_like)


def _is_id_like(s: pd.Series, ratio_threshold: float = 0.98) -> bool:
    """Detect ID-like columns (almost all unique values)."""
    try:
        n_unique = int(s.nunique(dropna=True))
        n_total = int(len(s))
        
        if n_total <= 0:
            return False
        
        return (n_unique / n_total) >= ratio_threshold
    
    except Exception:
        return False


def _has_leading_trailing_spaces(s: pd.Series) -> bool:
    """Check if object column contains leading/trailing spaces."""
    if s.dtype != "object":
        return False
    
    try:
        ss = s.dropna().astype(str)
        if ss.empty:
            return False
        
        # Compare original vs stripped
        return bool((ss != ss.str.strip()).any())
    
    except Exception:
        return False


def _is_non_normalized_categories(s: pd.Series) -> bool:
    """
    Heuristic: detect non-normalized categories (case/spacing variants).
    
    Example: "Apple", "apple", "APPLE" would be detected as non-normalized.
    """
    if s.dtype not in ("object",) and not pd.api.types.is_categorical_dtype(s):
        return False
    
    try:
        ss = s.dropna().astype(str)
        if ss.empty:
            return False
        
        # Normalize: lowercase + strip
        normalized = ss.str.strip().str.lower()
        
        # If normalized has fewer unique values → originals had variants
        return int(normalized.nunique()) < int(ss.nunique())
    
    except Exception:
        return False


def _object_numeric_ratio(s: pd.Series) -> float:
    """Estimate ratio of parseable numeric values in object column."""
    try:
        ss = _safe_numeric_coerce(s.dropna().astype(str))
        return float(ss.notna().mean()) if len(ss) else 0.0
    except Exception:
        return 0.0


def _object_datetime_ratio(s: pd.Series) -> float:
    """Estimate ratio of parseable datetime values in object column."""
    try:
        ss = pd.to_datetime(s.dropna().astype(str), errors="coerce", utc=False)
        return float(ss.notna().mean()) if len(ss) else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Schema Analyzer Agent
# ═══════════════════════════════════════════════════════════════════════════

class SchemaAnalyzer(BaseAgent):
    """
    **SchemaAnalyzer** — Enterprise data structure and schema analysis.
    
    Responsibilities:
      1. Comprehensive column profiling (dtypes, cardinality, statistics)
      2. Semantic type detection (numeric, categorical, datetime)
      3. Quality heuristics (mixed types, spaces, normalization)
      4. Memory analysis & downcast recommendations
      5. Primary key detection (single & composite)
      6. Type casting recommendations
      7. Encoder recommendations (OneHot, Binary, TargetEnc, DateParts)
      8. Schema fingerprinting for change detection
      9. Stable 1:1 contract output
      10. Zero side-effects on input DataFrame
    
    Output format is deterministic and consistent.
    """
    
    def __init__(self, config: Optional[SchemaAnalyzerConfig] = None) -> None:
        """Initialize analyzer with optional custom configuration."""
        super().__init__(
            name="SchemaAnalyzer",
            description="Analyzes data structure and column types"
        )
        self.config = config or SchemaAnalyzerConfig()
        self._log = logger.bind(agent="SchemaAnalyzer")
        warnings.filterwarnings("ignore")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            data: pd.DataFrame
        """
        if "data" not in kwargs:
            raise ValueError("Required parameter 'data' not provided")
        
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError(f"'data' must be pd.DataFrame, got {type(kwargs['data']).__name__}")
        
        # Note: we accept empty DataFrames (return empty payload)
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("SchemaAnalyzer.execute")
    def execute(self, data: pd.DataFrame, **kwargs: Any) -> AgentResult:
        """
        Analyze data schema comprehensively.
        
        Args:
            data: Input DataFrame (not modified)
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with schema information (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        
        try:
            # Handle empty DataFrame
            if data is None or data.empty:
                self._log.warning("⚠ Received empty DataFrame")
                result.data = self._empty_payload()
                return result
            
            # 1. Basic information
            basic_info = self._get_basic_info(data)
            
            # 2. Detailed column analysis
            columns_info = self._analyze_columns(data)
            
            # 3. Data types summary
            dtypes_summary = self._get_dtypes_summary(data)
            
            # 4. Memory usage breakdown
            memory_info = self._get_memory_info(data)
            
            # 5. Heuristic suggestions
            suggestions = self._get_suggestions(data, columns_info)
            
            # 6. Schema fingerprint
            schema_fingerprint = self._schema_fingerprint(data)
            
            # Add dataset hash to metadata if configured
            if self.config.include_dataset_hash:
                if "meta" not in basic_info:
                    basic_info["meta"] = {}
                basic_info["meta"]["dataset_hash"] = _dataset_hash(data)
            
            # Assemble result
            result.data = {
                "basic_info": basic_info,
                "columns": columns_info,
                "dtypes_summary": dtypes_summary,
                "memory_info": memory_info,
                "suggestions": suggestions,
                "schema_fingerprint": schema_fingerprint,
            }
            
            self._log.success(
                f"✓ Schema analysis complete | "
                f"cols={len(data.columns)} rows={len(data):,} "
                f"memory={memory_info['total_mb']:.1f}MB"
            )
        
        except Exception as e:
            msg = f"Schema analysis failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            result.data = self._empty_payload()
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Basic Information
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("basic_info")
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic DataFrame metadata."""
        return {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns.tolist()],
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "size": int(df.size),
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Column Analysis (Per-Column Metadata)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("analyze_columns")
    def _analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze each column in detail.
        
        Includes:
          • Basic stats (unique, missing, zeros)
          • Type-specific statistics (numeric, categorical, datetime)
          • Quality heuristics (mixed types, spaces, cardinality, etc.)
        """
        cfg = self.config
        columns_info: List[Dict[str, Any]] = []
        n_rows = max(1, len(df))
        
        for col in df.columns:
            s = df[col]
            
            # Basic dtype and cardinality
            try:
                dtype_str = str(s.dtype)
            except Exception:
                dtype_str = "unknown"
            
            try:
                n_unique = int(s.nunique(dropna=True))
            except Exception:
                n_unique = 0
            
            try:
                n_missing = int(s.isna().sum())
            except Exception:
                n_missing = 0
            
            missing_pct = float((n_missing / n_rows) * 100.0) if n_rows > 0 else 0.0
            
            # Count zeros (only for numeric/bool)
            n_zeros = 0
            try:
                if pd.api.types.is_bool_dtype(s):
                    n_zeros = int((~s.fillna(False)).sum())
                elif pd.api.types.is_numeric_dtype(s):
                    sn = pd.to_numeric(s, errors="coerce")
                    n_zeros = int((sn == 0).sum())
                
                if n_zeros > cfg.numeric_zero_count_limit:
                    n_zeros = 0
            except Exception:
                n_zeros = 0
            
            # Semantic type detection
            try:
                semantic_type = detect_column_type(s)
            except Exception:
                semantic_type = None
            
            # Initialize extras
            extras: Dict[str, Any] = {}
            
            # Type-specific analysis
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                extras.update(self._get_numeric_stats(s))
            
            elif s.dtype in ("object",) or pd.api.types.is_categorical_dtype(s):
                extras.update(self._get_categorical_stats(s))
                
                # Additional heuristics for categorical/object
                try:
                    # Quasi-constant detection
                    vc = s.dropna().astype(str).value_counts()
                    if not vc.empty and len(s.dropna()) > 0:
                        top_freq = int(vc.iloc[0])
                        if (top_freq / len(s.dropna())) >= cfg.quasi_constant_ratio and s.nunique(dropna=False) > 1:
                            extras["quasi_constant"] = True
                    
                    # Mixed types
                    if _is_mixed_object_types(s):
                        extras["mixed_type"] = True
                    
                    # Text-heavy
                    ss = s.dropna().astype(str)
                    if not ss.empty and float(ss.str.len().mean()) >= cfg.text_heavy_avg_len:
                        extras["text_heavy"] = True
                    
                    # Leading/trailing spaces
                    if _has_leading_trailing_spaces(s):
                        extras["has_leading_trailing_spaces"] = True
                    
                    # Non-normalized categories
                    if _is_non_normalized_categories(s):
                        extras["non_normalized_categories"] = True
                
                except Exception as e:
                    logger.debug(f"categorical extras failed for '{col}': {e}")
            
            # Datetime analysis
            if pd.api.types.is_datetime64_any_dtype(s):
                extras.update(self._get_datetime_stats(s))
                
                # Monotonicity check
                try:
                    sd = pd.to_datetime(s, errors="coerce").dropna()
                    if len(sd) > 1 and (sd.is_monotonic_increasing or sd.is_monotonic_decreasing):
                        extras["monotonic"] = True
                except Exception:
                    pass
            
            # ID-like detection
            try:
                if _is_id_like(s, cfg.id_like_unique_ratio):
                    extras["id_like"] = True
            except Exception:
                pass
            
            # High cardinality (only for categorical/object)
            try:
                if (s.dtype in ("object",) or pd.api.types.is_categorical_dtype(s)) and n_rows > 0:
                    if (n_unique > cfg.high_cardinality_ratio * n_rows):
                        extras["high_cardinality"] = True
            except Exception:
                pass
            
            # Create column info
            col_info = ColumnInfo(
                name=str(col),
                dtype=dtype_str,
                semantic_type=semantic_type,
                n_unique=n_unique,
                n_missing=n_missing,
                missing_pct=missing_pct,
                n_zeros=n_zeros,
                extras=extras,
            )
            
            columns_info.append(col_info.to_dict())
        
        return columns_info
    
    # ───────────────────────────────────────────────────────────────────
    # Type-Specific Statistics
    # ───────────────────────────────────────────────────────────────────
    
    def _get_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for numeric/boolean columns."""
        try:
            if pd.api.types.is_bool_dtype(series):
                s = series.astype("Int8").astype(float)
            else:
                s = _safe_numeric_coerce(series)
        except Exception:
            s = _safe_numeric_coerce(series)
        
        # Check if all NaN after coercion
        if s.notna().sum() == 0:
            return {
                "mean": None, "std": None, "min": None, "max": None,
                "median": None, "q25": None, "q75": None,
                "skewness": None, "kurtosis": None,
            }
        
        # Clean inf values
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        
        if s.empty:
            return {
                "mean": None, "std": None, "min": None, "max": None,
                "median": None, "q25": None, "q75": None,
                "skewness": None, "kurtosis": None,
            }
        
        return {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std(ddof=1)) if s.count() > 1 else 0.0, 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "median": round(float(s.median()), 6),
            "q25": round(float(s.quantile(0.25)), 6),
            "q75": round(float(s.quantile(0.75)), 6),
            "skewness": round(float(s.skew()) if s.count() > 2 else 0.0, 3),
            "kurtosis": round(float(s.kurtosis()) if s.count() > 3 else 0.0, 3),
        }
    
    def _get_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for categorical/object columns."""
        try:
            s = series.astype("string")
            vc = s.value_counts(dropna=True)
        except Exception:
            vc = series.value_counts(dropna=True)
        
        # Mode
        mode_val = None
        try:
            mode_series = series.mode(dropna=True)
            if not mode_series.empty:
                mode_val = str(mode_series.iloc[0])
        except Exception:
            pass
        
        n_categories = int(len(vc))
        is_binary = bool(n_categories == 2)
        
        return {
            "mode": mode_val,
            "top_values": {str(k): int(v) for k, v in vc.head(5).to_dict().items()},
            "n_categories": n_categories,
            "is_binary": is_binary,
            "high_cardinality": bool(n_categories > 0.90 * max(1, len(series))),
        }
    
    def _get_datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for datetime columns."""
        try:
            s = pd.to_datetime(series, errors="coerce")
            s = s.dropna()
        except Exception:
            return {
                "min_date": None,
                "max_date": None,
                "date_range_days": None,
                "maybe_mixed_timezones": False,
            }
        
        if s.empty:
            return {
                "min_date": None,
                "max_date": None,
                "date_range_days": None,
                "maybe_mixed_timezones": False,
            }
        
        min_d = s.min()
        max_d = s.max()
        
        # Detect mixed timezones (heuristic)
        tz_mixed = False
        try:
            has_tz = getattr(series.dtype, "tz", None) is not None
            has_naive = any(
                getattr(x, "tzinfo", None) is None
                for x in series.dropna().tolist()
            )
            tz_mixed = bool(has_tz and has_naive)
        except Exception:
            pass
        
        return {
            "min_date": str(min_d),
            "max_date": str(max_d),
            "date_range_days": int((max_d - min_d).days),
            "maybe_mixed_timezones": tz_mixed,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Data Types Summary
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("dtypes_summary")
    def _get_dtypes_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Summarize pandas dtypes distribution."""
        try:
            vc = df.dtypes.value_counts()
            return {str(dtype): int(count) for dtype, count in vc.items()}
        except Exception:
            # Fallback
            types = [str(t) for t in df.dtypes]
            vc = pd.Series(types).value_counts()
            return {str(dtype): int(count) for dtype, count in vc.items()}
    
    # ───────────────────────────────────────────────────────────────────
    # Memory Usage
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("memory_info")
    def _get_memory_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute memory usage breakdown."""
        try:
            mem = df.memory_usage(deep=True)
        except Exception:
            mem = df.memory_usage(deep=False)
        
        total_bytes = int(mem.sum())
        
        return {
            "total_mb": round(total_bytes / (1024**2), 2),
            "per_row_bytes": round(total_bytes / max(1, len(df)), 2),
            "by_column_mb": {
                str(col): round(val / (1024**2), 3)
                for col, val in mem.items()
            },
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Heuristic Suggestions
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("suggestions")
    def _get_suggestions(
        self,
        df: pd.DataFrame,
        columns_info: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate heuristic suggestions for:
          • Primary keys (single & composite)
          • Type casting (object→numeric/datetime)
          • Memory downcasting
          • Feature encoding recommendations
          • Data quality warnings
        """
        cfg = self.config
        n_rows = max(1, len(df))
        
        suggestions: Dict[str, Any] = {
            "potential_primary_keys": [],
            "potential_casts": [],
            "downcast_hints": [],
            "encoder_recommendations": [],
            "warnings": [],
        }
        
        # Get column lists
        cat_cols = get_categorical_columns(df)
        num_cols = get_numeric_columns(df)
        
        # ─── Primary Key Detection (Single Column)
        self._find_primary_keys_single(df, columns_info, n_rows, suggestions)
        
        # ─── Primary Key Detection (Composite)
        if cfg.enable_composite_pk_search:
            self._find_primary_keys_composite(df, columns_info, n_rows, suggestions)
        
        # ─── Type Casting Recommendations
        self._find_type_casts(df, cat_cols, suggestions)
        
        # ─── Memory Downcast Hints
        self._find_downcasts(df, num_cols, suggestions)
        
        # ─── Data Quality Warnings
        self._find_warnings(columns_info, suggestions)
        
        # ─── Encoder Recommendations
        self._find_encoders(df, columns_info, suggestions)
        
        return suggestions
    