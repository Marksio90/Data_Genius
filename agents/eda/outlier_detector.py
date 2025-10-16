# agents/eda/outlier_detector.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Outlier Detector                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade multi-method outlier detection:                          ║
║    ✓ IQR Method (Tukey's fences with configurable factor)                 ║
║    ✓ Z-Score Method (classical parametric detection)                      ║
║    ✓ Robust Z-Score (Median + MAD for heavy tails)                        ║
║    ✓ Isolation Forest (unsupervised anomaly detection)                    ║
║    ✓ Intelligent column filtering (binary, constant, excluded)            ║
║    ✓ Robust scaling with median/IQR normalization                         ║
║    ✓ Intelligent sampling (500k row limit for IF)                         ║
║    ✓ Contamination auto-estimation from IQR baseline                      ║
║    ✓ Per-method guards (min samples, zero variance, IQR=0)                ║
║    ✓ Union-based row aggregation across methods                           ║
║    ✓ Comprehensive telemetry with method timings                          ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "iqr_method": {
        "method": str,
        "description": str,
        "params": {"iqr_factor": float},
        "columns": {
            col: {
                "n_outliers": int,
                "percentage": float,
                "lower_bound": float,
                "upper_bound": float,
                "outlier_indices": List[Any],
            }
        },
        "n_columns_with_outliers": int,
        "guards": {
            "skipped_constant": List[str],
            "skipped_low_n": List[str],
        },
    },
    "zscore_method": {
        "method": str,
        "description": str,
        "params": {"threshold": float},
        "columns": {
            col: {
                "n_outliers": int,
                "percentage": float,
                "threshold": float,
                "outlier_indices": List[Any],
            }
        },
        "n_columns_with_outliers": int,
        "guards": {
            "skipped_constant": List[str],
            "skipped_low_n": List[str],
        },
    },
    "robust_zscore_method": {
        "method": str,
        "description": str,
        "params": {"threshold": float},
        "columns": {
            col: {
                "n_outliers": int,
                "percentage": float,
                "threshold": float,
                "median": float,
                "mad": float,
                "outlier_indices": List[Any],
            }
        },
        "n_columns_with_outliers": int,
        "guards": {
            "skipped_zero_mad": List[str],
            "skipped_low_n": List[str],
        },
    },
    "isolation_forest": {
        "method": str,
        "description": str,
        "n_outliers": int,
        "percentage": float,
        "outlier_indices": List[Any],
        "contamination": float,
        "rows_used": int,
        "n_features_used": int,
        "scaled": bool,
    } | None,
    "summary": {
        "total_outliers_rows_union": int,
        "by_method": {
            "IQR": int,
            "Z-Score": int,
            "RobustZ": int,
            "Isolation Forest": int,
        },
        "n_columns_with_outliers": int,
        "methods_used": List[str],
        "most_outliers": {"column": str, "n_outliers": int} | None,
        "example_outlier_indices": List[Any],
    },
    "recommendations": List[str],
    "telemetry": {
        "elapsed_ms": float,
        "timings_ms": {
            "iqr": float,
            "zscore": float,
            "robust_z": float,
            "iforest": float,
        },
        "sampled_iforest": bool,
        "iforest_sample_info": {"from_rows": int, "to_rows": int} | None,
        "caps": {
            "numeric_cols_total": int,
            "numeric_cols_used": int,
            "numeric_cols_cap": int,
        },
        "skipped_columns": {
            "non_numeric": List[str],
            "binary_like": List[str],
            "constant": List[str],
            "excluded": List[str],
        },
    },
    "version": "5.0-kosmos-enterprise",
}
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
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
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None
    logger.warning("⚠ sklearn not available — Isolation Forest disabled")

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
class OutlierDetectorConfig:
    """Enterprise configuration for outlier detection."""
    
    # IQR Method
    iqr_factor: float = 1.5                 # Tukey's fences multiplier
    
    # Z-Score Method (classical)
    zscore_threshold: float = 3.0           # Standard deviations threshold
    
    # Robust Z-Score (Median + MAD)
    robust_z_threshold: float = 3.5         # MAD-based threshold
    
    # Isolation Forest
    enable_isolation_forest: bool = True
    min_rows_for_if: int = 100              # Minimum rows to run IF
    if_random_state: int = 42
    if_n_jobs: int = -1                     # Parallel processing
    if_contam_min: float = 0.01             # Minimum contamination
    if_contam_max: float = 0.20             # Maximum contamination
    if_max_rows: int = 500_000              # Sampling limit for IF
    if_max_features: int = 500              # Column limit for IF
    
    # Output & Safety
    max_indices_return: int = 25            # Max example indices per method
    clip_inf_to_nan: bool = True            # Convert ±Inf → NaN
    min_non_na_per_col: int = 5             # Minimum non-NaN values required
    skip_binary_like: bool = True           # Skip columns with ≤2 unique values
    max_numeric_cols: int = 3000            # Soft limit on numeric columns
    exclude_columns: Tuple[str, ...] = ()   # Columns to exclude by name
    
    # Constants
    robust_z_constant: float = 0.6745       # MAD consistency constant


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
# SECTION: Main Outlier Detector Agent
# ═══════════════════════════════════════════════════════════════════════════

class OutlierDetector(BaseAgent):
    """
    **OutlierDetector** — Enterprise multi-method outlier detection.
    
    Responsibilities:
      1. IQR-based outlier detection (Tukey's fences)
      2. Z-Score outlier detection (parametric)
      3. Robust Z-Score detection (Median + MAD for heavy tails)
      4. Isolation Forest anomaly detection (unsupervised)
      5. Intelligent column filtering & guards
      6. Robust scaling for Isolation Forest
      7. Contamination auto-estimation
      8. Union-based aggregation across methods
      9. Comprehensive telemetry with method timings
      10. Zero side-effects on input DataFrame
    
    Features:
      • Binary/constant column skipping
      • Per-method minimum sample guards
      • Inf → NaN conversion
      • Intelligent sampling for large datasets
      • Median/IQR robust scaling
      • Contamination estimation from IQR baseline
    """
    
    def __init__(self, config: Optional[OutlierDetectorConfig] = None) -> None:
        """Initialize detector with optional custom configuration."""
        super().__init__(
            name="OutlierDetector",
            description="Detects outliers using multiple robust methods"
        )
        self.config = config or OutlierDetectorConfig()
        self._log = logger.bind(agent="OutlierDetector")
        warnings.filterwarnings("ignore")
        
        # Check Isolation Forest availability
        if self.config.enable_isolation_forest and IsolationForest is None:
            self._log.warning("⚠ Isolation Forest requested but sklearn unavailable")
            self.config = dataclass.replace(self.config, enable_isolation_forest=False)
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            data: pd.DataFrame
        
        Optional:
            include_columns: List[str] — whitelist columns
            override_if_contamination: float — force IF contamination
        """
        if "data" not in kwargs:
            raise ValueError("Required parameter 'data' not provided")
        
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError(f"'data' must be pd.DataFrame, got {type(kwargs['data']).__name__}")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("OutlierDetector.execute")
    def execute(
        self,
        data: pd.DataFrame,
        include_columns: Optional[List[str]] = None,
        override_if_contamination: Optional[float] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Detect outliers using multiple methods comprehensively.
        
        Args:
            data: Input DataFrame (not modified)
            include_columns: Optional whitelist of columns to analyze
            override_if_contamination: Force Isolation Forest contamination rate
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with outlier detection analysis (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t0_total = time.perf_counter()
        
        try:
            # Input validation
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                msg = "Invalid input: expected non-empty pandas DataFrame"
                result.add_error(msg)
                self._log.error(msg)
                result.data = self._empty_payload()
                return result
            
            cfg = self.config
            
            # ─── Select Numeric Subset with Guards
            num_df, caps_meta = self._select_numeric_subset(
                data=data,
                include_columns=include_columns,
                exclude_columns=cfg.exclude_columns,
                max_cols=cfg.max_numeric_cols,
                skip_binary_like=cfg.skip_binary_like
            )
            
            if num_df.shape[1] == 0:
                self._log.warning("⚠ No suitable numeric columns after filtering")
                result.add_warning("No suitable numeric columns for outlier detection")
                payload = self._empty_payload()
                payload["telemetry"]["elapsed_ms"] = round((time.perf_counter() - t0_total) * 1000, 1)
                payload["telemetry"]["caps"] = caps_meta["caps"]
                payload["telemetry"]["skipped_columns"] = caps_meta["skipped"]
                result.data = payload
                return result
            
            # Prepare data (clip inf, no side-effects)
            df_num = num_df.copy()
            if cfg.clip_inf_to_nan:
                df_num = df_num.replace([np.inf, -np.inf], np.nan)
            
            # ─── Method 1: IQR Detection
            t0 = time.perf_counter()
            iqr_dict, mask_iqr = self._detect_iqr_outliers(df_num)
            t_iqr = (time.perf_counter() - t0) * 1000
            
            # ─── Method 2: Z-Score Detection
            t0 = time.perf_counter()
            z_dict, mask_z = self._detect_zscore_outliers(df_num)
            t_z = (time.perf_counter() - t0) * 1000
            
            # ─── Method 3: Robust Z-Score Detection
            t0 = time.perf_counter()
            rz_dict, mask_rz = self._detect_robust_zscore_outliers(df_num)
            t_rz = (time.perf_counter() - t0) * 1000
            
            # ─── Method 4: Isolation Forest Detection (optional)
            t0 = time.perf_counter()
            if_dict, mask_if, sampled_if, sample_info_if = None, None, False, None
            
            if cfg.enable_isolation_forest and len(df_num) >= cfg.min_rows_for_if:
                # Estimate contamination from IQR baseline or use override
                contamination = self._estimate_contamination(
                    iqr_dict=iqr_dict,
                    n_rows=len(df_num),
                    override=override_if_contamination
                )
                
                if_dict, mask_if, sampled_if, sample_info_if = self._detect_isolation_forest_outliers(
                    df=df_num,
                    contamination=contamination
                )
            
            t_if = (time.perf_counter() - t0) * 1000
            
            # ─── Aggregate Results
            summary = self._create_summary(
                iqr_dict=iqr_dict,
                z_dict=z_dict,
                rz_dict=rz_dict,
                if_dict=if_dict,
                mask_iqr=mask_iqr,
                mask_z=mask_z,
                mask_rz=mask_rz,
                mask_if=mask_if
            )
            
            # ─── Generate Recommendations
            recommendations = self._get_recommendations(
                iqr_dict=iqr_dict,
                z_dict=z_dict,
                rz_dict=rz_dict,
                if_dict=if_dict,
                summary=summary
            )
            
            # ─── Assemble Result
            elapsed_ms = round((time.perf_counter() - t0_total) * 1000, 1)
            
            result.data = {
                "iqr_method": iqr_dict,
                "zscore_method": z_dict,
                "robust_zscore_method": rz_dict,
                "isolation_forest": if_dict,
                "summary": summary,
                "recommendations": recommendations,
                "telemetry": {
                    "elapsed_ms": elapsed_ms,
                    "timings_ms": {
                        "iqr": round(t_iqr, 1),
                        "zscore": round(t_z, 1),
                        "robust_z": round(t_rz, 1),
                        "iforest": round(t_if, 1),
                    },
                    "sampled_iforest": bool(sampled_if),
                    "iforest_sample_info": sample_info_if,
                    "caps": caps_meta["caps"],
                    "skipped_columns": caps_meta["skipped"],
                },
                "version": "5.0-kosmos-enterprise",
            }
            
            self._log.success(
                f"✓ Outlier detection complete | "
                f"rows_flagged={summary['total_outliers_rows_union']:,} | "
                f"cols_analyzed={caps_meta['caps']['numeric_cols_used']} | "
                f"methods={','.join(summary['methods_used'])} | "
                f"elapsed={elapsed_ms:.1f}ms"
            )
        
        except Exception as e:
            msg = f"Outlier detection failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            result.data = self._empty_payload()
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Numeric Column Selection & Filtering
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("numeric_subset_selection")
    def _select_numeric_subset(
        self,
        data: pd.DataFrame,
        include_columns: Optional[List[str]],
        exclude_columns: Tuple[str, ...],
        max_cols: int,
        skip_binary_like: bool
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Select and filter numeric columns with intelligent guards.
        
        Guards:
          • Exclude specified columns
          • Include whitelist (if provided)
          • Skip binary-like columns (≤2 unique values)
          • Skip constant columns (≤1 unique value)
          • Apply column cap for performance
        
        Returns:
            Tuple of (filtered_df, metadata_dict)
        """
        all_numeric = data.select_dtypes(include=[np.number]).copy()
        total_num = all_numeric.shape[1]
        
        skipped_excluded: List[str] = []
        skipped_binary: List[str] = []
        skipped_constant: List[str] = []
        
        # ─── Exclude by name
        if exclude_columns:
            present_excl = [c for c in exclude_columns if c in all_numeric.columns]
            skipped_excluded.extend(present_excl)
            all_numeric = all_numeric.drop(columns=present_excl, errors="ignore")
        
        # ─── Whitelist filter
        if include_columns:
            keep = [c for c in include_columns if c in all_numeric.columns]
            all_numeric = all_numeric.loc[:, keep]
        
        # ─── Skip binary-like
        if skip_binary_like and all_numeric.shape[1] > 0:
            nunique = all_numeric.nunique(dropna=True)
            binary_cols = nunique[nunique <= 2].index.tolist()
            if binary_cols:
                skipped_binary = binary_cols
                all_numeric = all_numeric.drop(columns=binary_cols, errors="ignore")
        
        # ─── Skip constant columns
        if all_numeric.shape[1] > 0:
            nunique_clean = all_numeric.apply(
                lambda s: pd.to_numeric(s, errors="coerce")
            ).nunique(dropna=True)
            const_cols = nunique_clean[nunique_clean <= 1].index.tolist()
            if const_cols:
                skipped_constant = const_cols
                all_numeric = all_numeric.drop(columns=const_cols, errors="ignore")
        
        # ─── Apply column cap
        capped = all_numeric.shape[1] > max_cols
        if capped:
            self._log.warning(f"⚠ Limiting columns: {all_numeric.shape[1]} → {max_cols}")
            all_numeric = all_numeric.iloc[:, :max_cols]
        
        caps_meta = {
            "caps": {
                "numeric_cols_total": int(total_num),
                "numeric_cols_used": int(all_numeric.shape[1]),
                "numeric_cols_cap": int(max_cols) if capped else int(all_numeric.shape[1]),
            },
            "skipped": {
                "non_numeric": [],  # Not tracked in this method
                "binary_like": skipped_binary,
                "constant": skipped_constant,
                "excluded": skipped_excluded,
            },
        }
        
        return all_numeric, caps_meta
    
    # ───────────────────────────────────────────────────────────────────
    # Method 1: IQR Outlier Detection
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("iqr_detection")
    def _detect_iqr_outliers(
        self,
        df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], pd.Series]:
        """
        Detect outliers using IQR method (Tukey's fences).
        
        Formula:
          outlier if x < Q1 - factor*IQR or x > Q3 + factor*IQR
        
        Returns:
            Tuple of (payload_dict, row_mask_series)
        """
        cfg = self.config
        outliers_by_column: Dict[str, Dict[str, Any]] = {}
        row_mask = pd.Series(False, index=df.index)
        skipped_constant: List[str] = []
        skipped_low_n: List[str] = []
        
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s_nona = s.dropna()
            
            # Guard: insufficient data
            if len(s_nona) < cfg.min_non_na_per_col:
                skipped_low_n.append(str(col))
                continue
            
            q1, q3 = s_nona.quantile(0.25), s_nona.quantile(0.75)
            iqr = q3 - q1
            
            # Guard: zero IQR (constant)
            if iqr <= 0:
                skipped_constant.append(str(col))
                continue
            
            lower = q1 - cfg.iqr_factor * iqr
            upper = q3 + cfg.iqr_factor * iqr
            
            mask = (s < lower) | (s > upper)
            n_out = int(mask.sum())
            
            if n_out > 0:
                row_mask |= mask.fillna(False)
                outliers_by_column[str(col)] = {
                    "n_outliers": n_out,
                    "percentage": round(float(n_out / len(s) * 100.0), 2),
                    "lower_bound": round(float(lower), 6),
                    "upper_bound": round(float(upper), 6),
                    "outlier_indices": self._head_indices(mask[mask].index, cfg.max_indices_return),
                }
        
        payload = {
            "method": f"IQR (factor={cfg.iqr_factor})",
            "description": f"Outliers: < Q1-{cfg.iqr_factor}*IQR or > Q3+{cfg.iqr_factor}*IQR",
            "params": {"iqr_factor": float(cfg.iqr_factor)},
            "columns": outliers_by_column,
            "n_columns_with_outliers": len(outliers_by_column),
            "guards": {
                "skipped_constant": skipped_constant,
                "skipped_low_n": skipped_low_n,
            },
        }
        
        return payload, row_mask
    
    # ───────────────────────────────────────────────────────────────────
    # Method 2: Z-Score Outlier Detection
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("zscore_detection")
    def _detect_zscore_outliers(
        self,
        df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], pd.Series]:
        """
        Detect outliers using classical Z-score method.
        
        Formula:
          outlier if |z-score| > threshold, where z = (x - μ) / σ
        
        Returns:
            Tuple of (payload_dict, row_mask_series)
        """
        cfg = self.config
        threshold = cfg.zscore_threshold
        outliers_by_column: Dict[str, Dict[str, Any]] = {}
        row_mask = pd.Series(False, index=df.index)
        skipped_constant: List[str] = []
        skipped_low_n: List[str] = []
        
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s_nona = s.dropna()
            
            # Guard: insufficient data
            if len(s_nona) < cfg.min_non_na_per_col:
                skipped_low_n.append(str(col))
                continue
            
            # Guard: zero standard deviation
            std = float(s_nona.std(ddof=1)) if len(s_nona) > 1 else 0.0
            if std == 0.0:
                skipped_constant.append(str(col))
                continue
            
            # Compute z-scores
            z = np.abs((s - s_nona.mean()) / std)
            mask = z > threshold
            n_out = int(mask.sum(skipna=True))
            
            if n_out > 0:
                row_mask |= mask.fillna(False)
                outliers_by_column[str(col)] = {
                    "n_outliers": n_out,
                    "percentage": round(float(n_out / len(s) * 100.0), 2),
                    "threshold": float(threshold),
                    "outlier_indices": self._head_indices(mask[mask].index, cfg.max_indices_return),
                }
        
        payload = {
            "method": "Z-Score",
            "description": f"Outliers: |z-score| > {threshold}",
            "params": {"threshold": float(threshold)},
            "columns": outliers_by_column,
            "n_columns_with_outliers": len(outliers_by_column),
            "guards": {
                "skipped_constant": skipped_constant,
                "skipped_low_n": skipped_low_n,
            },
        }
        
        return payload, row_mask
    
    # ───────────────────────────────────────────────────────────────────
    # Method 3: Robust Z-Score Detection (Median + MAD)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("robust_zscore_detection")
    def _detect_robust_zscore_outliers(
        self,
        df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], pd.Series]:
        """
        Detect outliers using Robust Z-score (Median + MAD).
        
        Formula:
          robust_z = 0.6745 * (x - median) / MAD
          outlier if |robust_z| > threshold
        
        Robust to heavy tails and outliers themselves.
        
        Returns:
            Tuple of (payload_dict, row_mask_series)
        """
        cfg = self.config
        threshold = cfg.robust_z_threshold
        c = cfg.robust_z_constant  # 0.6745 for consistency with normal distribution
        outliers_by_column: Dict[str, Dict[str, Any]] = {}
        row_mask = pd.Series(False, index=df.index)
        skipped_zero_mad: List[str] = []
        skipped_low_n: List[str] = []
        
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s_nona = s.dropna()
            
            # Guard: insufficient data
            if len(s_nona) < cfg.min_non_na_per_col:
                skipped_low_n.append(str(col))
                continue
            
            # Compute median and MAD
            med = float(np.median(s_nona))
            mad = float(np.median(np.abs(s_nona - med)))
            
            # Guard: zero MAD (no dispersion)
            if mad == 0.0:
                skipped_zero_mad.append(str(col))
                continue
            
            # Compute robust z-scores
            rz = np.abs(c * (s - med) / mad)
            mask = rz > threshold
            n_out = int(mask.sum(skipna=True))
            
            if n_out > 0:
                row_mask |= mask.fillna(False)
                outliers_by_column[str(col)] = {
                    "n_outliers": n_out,
                    "percentage": round(float(n_out / len(s) * 100.0), 2),
                    "threshold": float(threshold),
                    "median": round(med, 6),
                    "mad": round(mad, 6),
                    "outlier_indices": self._head_indices(mask[maskRetryMContinue].index, cfg.max_indices_return),
}
    payload = {
        "method": "Robust Z-Score (Median+MAD)",
        "description": f"Outliers: |{c}*(x-median)/MAD| > {threshold}",
        "params": {"threshold": float(threshold)},
        "columns": outliers_by_column,
        "n_columns_with_outliers": len(outliers_by_column),
        "guards": {
            "skipped_zero_mad": skipped_zero_mad,
            "skipped_low_n": skipped_low_n,
        },
    }
    
    return payload, row_mask

# ───────────────────────────────────────────────────────────────────
# Method 4: Isolation Forest Detection
# ───────────────────────────────────────────────────────────────────

@_timeit("isolation_forest_detection")
@_safe_operation("isolation_forest", default_value=(None, None, False, None))
def _detect_isolation_forest_outliers(
    self,
    df: pd.DataFrame,
    contamination: float
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.Series], bool, Optional[Dict[str, int]]]:
    """
    Detect outliers using Isolation Forest with robust scaling.
    
    Features:
      • Intelligent sampling for large datasets
      • Robust scaling (median/IQR normalization)
      • Median imputation for NaN values
      • Feature limiting for high-dimensional data
    
    Returns:
        Tuple of (payload|None, row_mask|None, sampled_flag, sample_info|None)
    """
    cfg = self.config
    
    if IsolationForest is None:
        self._log.warning("⚠ Isolation Forest unavailable (sklearn not installed)")
        return None, None, False, None
    
    sampled = False
    sample_info = None
    X = df.copy()
    
    # ─── Sampling Safety
    if len(X) > cfg.if_max_rows:
        sampled = True
        sample_info = {
            "from_rows": int(len(X)),
            "to_rows": int(cfg.if_max_rows),
        }
        self._log.info(
            f"Isolation Forest sampling: {len(X):,} → {cfg.if_max_rows:,} rows"
        )
        X = X.sample(n=cfg.if_max_rows, random_state=cfg.if_random_state)
    
    # ─── Feature Limiting
    if X.shape[1] > cfg.if_max_features:
        self._log.warning(
            f"⚠ Isolation Forest feature limit: {X.shape[1]} → {cfg.if_max_features} cols"
        )
        X = X.iloc[:, :cfg.if_max_features]
    
    # ─── Robust Scaling (Median/IQR)
    med = X.median(numeric_only=True)
    q75 = X.quantile(0.75, numeric_only=True)
    q25 = X.quantile(0.25, numeric_only=True)
    iqr = q75 - q25
    
    # Guard against zero IQR
    scale = iqr.replace(0.0, 1.0)
    
    X_scaled = (X - med) / scale
    
    # Median imputation for NaN values
    X_scaled = X_scaled.fillna(0.0)  # 0 represents median after scaling
    
    # ─── Fit Isolation Forest
    iso = IsolationForest(
        contamination=float(contamination),
        random_state=cfg.if_random_state,
        n_jobs=cfg.if_n_jobs,
        bootstrap=False,  # Use all samples
        max_samples="auto"
    )
    
    predictions = iso.fit_predict(X_scaled)
    
    # -1 indicates outlier, 1 indicates inlier
    mask = pd.Series(predictions == -1, index=X_scaled.index)
    
    payload = {
        "method": "Isolation Forest",
        "description": "Unsupervised anomaly detection with robust scaling (median/IQR)",
        "n_outliers": int(mask.sum()),
        "percentage": round(float(mask.mean() * 100.0), 2),
        "outlier_indices": self._head_indices(mask[mask].index, cfg.max_indices_return),
        "contamination": round(float(contamination), 4),
        "rows_used": int(len(X_scaled)),
        "n_features_used": int(X_scaled.shape[1]),
        "scaled": True,
    }
    
    # Reindex mask to match original DataFrame
    mask_full = mask.reindex(df.index, fill_value=False)
    
    return payload, mask_full, sampled, sample_info

# ───────────────────────────────────────────────────────────────────
# Contamination Estimation
# ───────────────────────────────────────────────────────────────────

def _estimate_contamination(
    self,
    iqr_dict: Dict[str, Any],
    n_rows: int,
    override: Optional[float]
) -> float:
    """
    Estimate contamination rate for Isolation Forest.
    
    Strategy:
      1. If override provided: use it (clamped to valid range)
      2. Otherwise: estimate from IQR baseline (average % outliers)
      3. Clamp to [if_contam_min, if_contam_max]
    
    Returns:
        Contamination rate (float between 0 and 1)
    """
    cfg = self.config
    
    # Override if provided
    if override is not None:
        try:
            ov = float(override)
            contamination = float(np.clip(ov, cfg.if_contam_min, cfg.if_contam_max))
            self._log.info(f"Using override contamination: {contamination:.4f}")
            return contamination
        except Exception as e:
            self._log.warning(f"⚠ Invalid contamination override: {e}")
    
    # Estimate from IQR baseline
    cols = iqr_dict.get("columns", {})
    if not cols or n_rows <= 0:
        return max(cfg.if_contam_min, 0.01)
    
    percentages = [float(v.get("percentage", 0.0)) for v in cols.values()]
    mean_pct = float(np.mean(percentages)) if percentages else 0.0
    
    # Convert percentage to ratio
    estimated = mean_pct / 100.0
    
    # Clamp to valid range
    contamination = float(np.clip(estimated, cfg.if_contam_min, cfg.if_contam_max))
    
    self._log.debug(f"Estimated contamination from IQR: {contamination:.4f}")
    return contamination

# ───────────────────────────────────────────────────────────────────
# Summary Generation (Union-Based Aggregation)
# ───────────────────────────────────────────────────────────────────

@_timeit("summary_generation")
def _create_summary(
    self,
    iqr_dict: Dict[str, Any],
    z_dict: Dict[str, Any],
    rz_dict: Dict[str, Any],
    if_dict: Optional[Dict[str, Any]],
    mask_iqr: Optional[pd.Series],
    mask_z: Optional[pd.Series],
    mask_rz: Optional[pd.Series],
    mask_if: Optional[pd.Series]
) -> Dict[str, Any]:
    """
    Create comprehensive summary with union-based row aggregation.
    
    Aggregation:
      • total_outliers_rows_union: OR across all method masks
      • by_method: per-method outlier row counts
      • n_columns_with_outliers: unique columns flagged (columnwise methods)
      • most_outliers: column with most outliers across methods
    
    Returns:
        Summary dictionary
    """
    cfg = self.config
    
    # Determine methods used
    methods_used = ["IQR", "Z-Score", "RobustZ"]
    if if_dict is not None:
        methods_used.append("Isolation Forest")
    
    # Compute union of columns with outliers (from columnwise methods)
    cols_with_outliers = (
        set(iqr_dict.get("columns", {}).keys()) |
        set(z_dict.get("columns", {}).keys()) |
        set(rz_dict.get("columns", {}).keys())
    )
    
    # Compute union of row masks (OR aggregation)
    union_mask = None
    for mask in (mask_iqr, mask_z, mask_rz, mask_if):
        if mask is None:
            continue
        union_mask = mask if union_mask is None else (union_mask | mask)
    
    total_union = int(union_mask.sum()) if union_mask is not None else 0
    
    # Per-method outlier row counts
    by_method = {
        "IQR": int(mask_iqr.sum()) if mask_iqr is not None else 0,
        "Z-Score": int(mask_z.sum()) if mask_z is not None else 0,
        "RobustZ": int(mask_rz.sum()) if mask_rz is not None else 0,
        "Isolation Forest": int(mask_if.sum()) if mask_if is not None else 0,
    }
    
    # Example outlier indices (for inspection)
    example_indices: List[Any] = []
    for mask in (mask_iqr, mask_z, mask_rz, mask_if):
        if mask is not None and mask.any():
            example_indices.extend(
                self._head_indices(mask[mask].index, cfg.max_indices_return)
            )
    
    # Deduplicate and limit
    example_indices = list(dict.fromkeys(example_indices))[:cfg.max_indices_return]
    
    # Find column with most outliers
    most_iqr = self._get_most_outliers_column(iqr_dict)
    most_z = self._get_most_outliers_column(z_dict)
    most_rz = self._get_most_outliers_column(rz_dict)
    
    candidates = [x for x in (most_iqr, most_z, most_rz) if x is not None]
    most = max(candidates, key=lambda d: d["n_outliers"], default=None) if candidates else None
    
    return {
        "total_outliers_rows_union": total_union,
        "by_method": by_method,
        "n_columns_with_outliers": len(cols_with_outliers),
        "methods_used": methods_used,
        "most_outliers": most,
        "example_outlier_indices": example_indices,
    }

def _get_most_outliers_column(
    self,
    method_payload: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract column with most outliers from method payload.
    
    Returns:
        Dict with {"column": str, "n_outliers": int} or None
    """
    cols = (method_payload or {}).get("columns", {})
    if not cols:
        return None
    
    max_col, meta = max(
        cols.items(),
        key=lambda kv: kv[1].get("n_outliers", 0)
    )
    
    return {
        "column": str(max_col),
        "n_outliers": int(meta.get("n_outliers", 0)),
    }

# ───────────────────────────────────────────────────────────────────
# Recommendations Generation
# ───────────────────────────────────────────────────────────────────

@_timeit("recommendations")
def _get_recommendations(
    self,
    iqr_dict: Dict[str, Any],
    z_dict: Dict[str, Any],
    rz_dict: Dict[str, Any],
    if_dict: Optional[Dict[str, Any]],
    summary: Dict[str, Any]
) -> List[str]:
    """
    Generate actionable recommendations based on detection results.
    
    Categories:
      • Overall outlier presence warnings
      • Method-specific insights (IQR, Z-score, Robust Z, IF)
      • Treatment strategies (winsorization, transformation, filtering)
      • Validation best practices
    
    Returns:
        Deduplicated list of recommendations
    """
    rec: List[str] = []
    
    n_cols = summary.get("n_columns_with_outliers", 0)
    total_rows = summary.get("total_outliers_rows_union", 0)
    
    # ─── No Outliers Detected
    if total_rows == 0:
        rec.append(
            "✅ No outliers detected across all methods — data appears clean, "
            "you can skip outlier treatment steps."
        )
        return rec
    
    # ─── Overall Detection Warning
    if n_cols > 0:
        rec.append(
            f"🔎 Detected outliers in {n_cols} numeric column(s), "
            f"affecting {total_rows} row(s) total (union across methods). "
            "Consider data cleaning before model training."
        )
    
    # ─── IQR-Specific Recommendations
    if iqr_dict.get("columns"):
        most = self._get_most_outliers_column(iqr_dict)
        if most:
            rec.append(
                f"📦 IQR method: Most outliers in '{most['column']}' ({most['n_outliers']} values). "
                "Consider: (1) Winsorization (capping at percentiles), "
                "(2) Log/Yeo-Johnson transformation, or (3) Robust scaling."
            )
    
    # ─── Z-Score Recommendations
    if z_dict.get("n_columns_with_outliers", 0) > 0:
        rec.append(
            "📏 Z-Score method flagged extreme values (>3σ). "
            "For parametric models: standardization may help. "
            "For tree-based models: outliers often less problematic."
        )
    
    # ─── Robust Z-Score Recommendations
    if rz_dict.get("n_columns_with_outliers", 0) > 0:
        rec.append(
            "🛡️ Robust Z-Score detected heavy-tailed distributions. "
            "Prefer robust methods: (1) Median/MAD scaling, "
            "(2) Huber loss, (3) Quantile regression, or (4) Tree-based models."
        )
    
    # ─── Isolation Forest Recommendations
    if if_dict and if_dict.get("n_outliers", 0) > 0:
        contamination = if_dict.get("contamination", 0.0)
        rec.append(
            f"🤖 Isolation Forest detected {if_dict['n_outliers']} multivariate anomalies "
            f"(contamination={contamination:.2%}). "
            "Consider: (1) Creating 'is_anomaly' indicator feature, "
            "(2) Separate anomaly model, or (3) Filtering if data quality issue."
        )
    
    # ─── General Best Practices
    rec.append(
        "💡 Best practices: "
        "(1) Validate outlier removal impact on holdout metrics, "
        "(2) Never remove outliers from test set, "
        "(3) Document all cleaning decisions, "
        "(4) Consider domain expertise before automatic removal."
    )
    
    rec.append(
        "🧪 Treatment options ranked by severity: "
        "(1) Investigate & correct data errors, "
        "(2) Transform features (log, Box-Cox), "
        "(3) Winsorize/cap at percentiles, "
        "(4) Use robust models (trees, robust loss), "
        "(5) Remove only if justified by domain knowledge."
    )
    
    # Deduplicate and return
    return list(dict.fromkeys([r for r in rec if r]))

# ───────────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────────

@staticmethod
def _head_indices(idx: pd.Index, n: int) -> List[Any]:
    """
    Extract first n indices as list (for payload examples).
    
    Args:
        idx: Pandas Index
        n: Maximum number of indices to return
    
    Returns:
        List of index values (serializable)
    """
    try:
        head = idx[:n]
        # Convert to native Python types for JSON serialization
        return [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in head]
    except Exception:
        return []

# ───────────────────────────────────────────────────────────────────
# Empty Payload (Fallback)
# ───────────────────────────────────────────────────────────────────

def _empty_payload(self) -> Dict[str, Any]:
    """Generate empty payload for failed/invalid input."""
    cfg = self.config
    
    return {
        "iqr_method": {
            "method": f"IQR (factor={cfg.iqr_factor})",
            "description": "No analysis performed",
            "params": {"iqr_factor": cfg.iqr_factor},
            "columns": {},
            "n_columns_with_outliers": 0,
            "guards": {"skipped_constant": [], "skipped_low_n": []},
        },
        "zscore_method": {
            "method": "Z-Score",
            "description": "No analysis performed",
            "params": {"threshold": cfg.zscore_threshold},
            "columns": {},
            "n_columns_with_outliers": 0,
            "guards": {"skipped_constant": [], "skipped_low_n": []},
        },
        "robust_zscore_method": {
            "method": "Robust Z-Score (Median+MAD)",
            "description": "No analysis performed",
            "params": {"threshold": cfg.robust_z_threshold},
            "columns": {},
            "n_columns_with_outliers": 0,
            "guards": {"skipped_zero_mad": [], "skipped_low_n": []},
        },
        "isolation_forest": None,
        "summary": {
            "total_outliers_rows_union": 0,
            "by_method": {
                "IQR": 0,
                "Z-Score": 0,
                "RobustZ": 0,
                "Isolation Forest": 0,
            },
            "n_columns_with_outliers": 0,
            "methods_used": ["IQR", "Z-Score", "RobustZ"],
            "most_outliers": None,
            "example_outlier_indices": [],
        },
        "recommendations": [
            "Provide valid numeric data for outlier detection"
        ],
        "telemetry": {
            "elapsed_ms": 0.0,
            "timings_ms": {
                "iqr": 0.0,
                "zscore": 0.0,
                "robust_z": 0.0,
                "iforest": 0.0,
            },
            "sampled_iforest": False,
            "iforest_sample_info": None,
            "caps": {
                "numeric_cols_total": 0,
                "numeric_cols_used": 0,
                "numeric_cols_cap": 0,
            },
            "skipped_columns": {
                "non_numeric": [],
                "binary_like": [],
                "constant": [],
                "excluded": [],
            },
        },
        "version": "5.0-kosmos-enterprise",
    }

