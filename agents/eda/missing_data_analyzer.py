# agents/eda/missing_data_analyzer.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Missing Data Analyzer             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade missing data pattern analysis:                           ║
║    ✓ Multi-level analysis (dataset, columns, rows)                        ║
║    ✓ Missing mask correlation detection (O(k²) safety)                     ║
║    ✓ MAR/MNAR signal identification with heuristics                       ║
║    ✓ Co-missing pattern discovery with fallback                           ║
║    ✓ Intelligent sampling (500k row limit)                                ║
║    ✓ Severity classification + strategy recommendations                   ║
║    ✓ Soft time budgeting with adaptive column reduction                   ║
║    ✓ Indicator-like column detection                                      ║
║    ✓ Empty string & whitespace normalization                              ║
║    ✓ Comprehensive telemetry + debug metadata                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "summary": {
        "total_cells": int,
        "total_missing": int,
        "missing_percentage": float,
        "n_columns_with_missing": int,
        "n_rows_with_missing": int,
        "complete_rows": int,
    },
    "columns": [
        {
            "column": str,
            "n_missing": int,
            "missing_percentage": float,
            "dtype": str,
            "severity": "low" | "medium" | "high" | "critical",
            "suggested_strategy": str,
            "is_indicator_like": bool,
            "n_unique_non_na": int,
            "top_co_missing_with": [
                {"column": str, "mask_corr": float}
            ],  # Top 3 correlated missing patterns
        }
    ],
    "rows": {
        "top_rows_with_many_missing": [
            {
                "row_index": int | str,
                "n_missing": int,
                "missing_cols": List[str],  # Limited to max_return_indices
            }
        ],
        "max_missing_in_row": int,
    },
    "patterns": {
        "correlated": [
            {
                "column": str,
                "correlated_with": str,
                "mask_corr": float,
            }
        ],
        "blocks": {
            "rows_ge_threshold": int,
            "threshold": int,
        },
        "mar_mnar_signals": [
            {
                "column": str,
                "signal": "MAR_like" | "MNAR_like",
                "reason": str,
            }
        ],
    },
    "recommendations": List[str],
    "telemetry": {
        "elapsed_ms": float,
        "mask_cols_analyzed": int,
        "sampled": bool,
        "sample_info": {"from_rows": int, "to_rows": int} | None,
        "soft_time_budget_ms": int,
        "soft_time_exceeded": bool,
        "caps": {
            "corr_mask_cols_requested": int,
            "corr_mask_cols_used": int,
            "corr_mask_cols_cap": int,
        },
        "debug": {
            "object_empty_as_na": bool,
            "strip_whitespace_in_object": bool,
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
class MissingDataConfig:
    """Enterprise configuration for missing data analysis."""
    
    # Severity classification thresholds
    low_threshold_pct: float = 5.0          # <5% missing → low
    medium_threshold_pct: float = 20.0      # <20% missing → medium
    high_threshold_pct: float = 50.0        # <50% missing → high, else critical
    drop_column_over_pct: float = 70.0      # >70% missing → recommend drop
    
    # Row pattern detection
    row_missing_block_threshold: int = 3    # ≥3 missing in row → block pattern
    
    # Mask correlation analysis
    corr_mask_threshold: float = 0.70       # |r| > 0.7 → correlated missing patterns
    max_corr_cols_for_masks: int = 400      # Soft limit (O(k²) safety)
    
    # Top lists limits
    top_rows_limit: int = 10                # Top rows with most missing values
    top_co_missing_with: int = 3            # Top co-missing columns per column
    fallback_top_co_missing: int = 3        # Fallback when no correlation above threshold
    max_return_indices_per_rowlist: int = 50  # Max columns to return per row
    
    # Sampling configuration
    enable_sampling: bool = True
    sample_rows: int = 500_000
    random_state: int = 42
    
    # Object column preprocessing
    treat_empty_string_as_na: bool = True
    strip_whitespace_in_object: bool = True
    
    # Performance & time budgeting
    soft_time_budget_ms: int = 45_000       # Soft time limit (informational)
    min_rows_for_mask_corr: int = 2_000     # Minimum rows for correlation analysis
    
    # Adaptive column reduction at time budget percentages
    time_budget_70pct_reduction: float = 0.6  # Reduce to 60% of columns at 70% time
    time_budget_90pct_reduction: float = 0.4  # Reduce to 40% of columns at 90% time


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
# SECTION: Main Missing Data Analyzer Agent
# ═══════════════════════════════════════════════════════════════════════════

class MissingDataAnalyzer(BaseAgent):
    """
    **MissingDataAnalyzer** — Enterprise missing data pattern analysis.
    
    Responsibilities:
      1. Multi-level analysis: dataset summary, per-column, per-row
      2. Missing mask correlation detection with O(k²) safety
      3. MAR/MNAR signal identification using heuristics
      4. Co-missing pattern discovery with intelligent fallback
      5. Severity classification + strategy recommendations
      6. Indicator-like column detection
      7. Intelligent sampling for large datasets
      8. Soft time budgeting with adaptive column reduction
      9. Comprehensive telemetry + debug metadata
      10. Zero side-effects on input DataFrame
    
    Features:
      • Empty string & whitespace normalization
      • Inf → NaN conversion
      • Adaptive correlation analysis based on time budget
      • Fallback top co-missing when no strong correlations
      • Row block pattern detection
      • Type-aware imputation strategy recommendations
    """
    
    def __init__(self, config: Optional[MissingDataConfig] = None) -> None:
        """Initialize analyzer with optional custom configuration."""
        super().__init__(
            name="MissingDataAnalyzer",
            description="Analyzes missing data patterns and suggests handling strategies"
        )
        self.config = config or MissingDataConfig()
        self._log = logger.bind(agent="MissingDataAnalyzer")
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
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("MissingDataAnalyzer.execute")
    def execute(self, data: pd.DataFrame, **kwargs: Any) -> AgentResult:
        """
        Analyze missing data patterns comprehensively.
        
        Args:
            data: Input DataFrame (not modified)
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with missing data analysis (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t0 = time.perf_counter()
        
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
                result.add_warning("Empty DataFrame — no missing data analysis performed")
                payload = self._empty_payload()
                payload["telemetry"]["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                result.data = payload
                return result
            
            cfg = self.config
            
            # ─── Data Preparation (zero side-effects)
            df = self._prepare_dataframe(data)
            
            # ─── Intelligent Sampling
            df_work, sampled, sample_info = self._maybe_sample(df)
            
            # ─── Analysis Pipeline
            # 1. Dataset-level summary
            summary = self._get_missing_summary(df_work)
            
            # Check if there's any missing data
            if summary["total_missing"] == 0:
                self._log.success("✓ No missing data found!")
                result.data = self._build_no_missing_payload(df_work, t0, sampled, sample_info)
                return result
            
            # 2. Column-level analysis (severity, strategy, co-missing patterns)
            column_analysis = self._analyze_missing_by_column(df_work)
            
            # 3. Row-level analysis (block patterns, top rows)
            rows_info = self._analyze_missing_by_row(df_work)
            
            # 4. Pattern detection with time guard (correlations, MAR/MNAR)
            patterns, mask_cols_analyzed, corr_caps_meta = self._identify_patterns_with_time_guard(
                df=df_work,
                rows_info=rows_info,
                column_analysis=column_analysis,
                started_at=t0
            )
            
            # 5. Generate recommendations
            recommendations = self._get_recommendations(
                column_analysis=column_analysis,
                rows_info=rows_info,
                summary=summary,
                patterns=patterns
            )
            
            # ─── Assemble Result
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            
            result.data = {
                "summary": summary,
                "columns": column_analysis,
                "rows": rows_info,
                "patterns": patterns,
                "recommendations": recommendations,
                "telemetry": {
                    "elapsed_ms": elapsed_ms,
                    "mask_cols_analyzed": int(mask_cols_analyzed),
                    "sampled": bool(sampled),
                    "sample_info": sample_info,
                    "soft_time_budget_ms": cfg.soft_time_budget_ms,
                    "soft_time_exceeded": bool(elapsed_ms > cfg.soft_time_budget_ms),
                    "caps": {
                        "corr_mask_cols_requested": corr_caps_meta["requested"],
                        "corr_mask_cols_used": corr_caps_meta["used"],
                        "corr_mask_cols_cap": corr_caps_meta["cap"],
                    },
                    "debug": {
                        "object_empty_as_na": cfg.treat_empty_string_as_na,
                        "strip_whitespace_in_object": cfg.strip_whitespace_in_object,
                    },
                },
                "version": "5.0-kosmos-enterprise",
            }
            
            self._log.success(
                f"✓ Missing data analysis complete | "
                f"missing={summary['total_missing']:,} ({summary['missing_percentage']:.2f}%) | "
                f"cols_affected={summary['n_columns_with_missing']} | "
                f"elapsed={elapsed_ms:.1f}ms"
            )
        
        except Exception as e:
            msg = f"Missing data analysis failed: {type(e).__name__}: {str(e)}"
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
          • Shallow copy (no side-effects)
          • Strip whitespace from object columns
          • Convert empty strings to NaN
          • Replace ±Inf with NaN
        
        Returns:
            Prepared DataFrame copy
        """
        cfg = self.config
        df = data.copy(deep=False)  # Shallow copy: no side-effects
        
        # Object column preprocessing
        obj_cols = df.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            # Strip whitespace
            if cfg.strip_whitespace_in_object:
                try:
                    for col in obj_cols:
                        df[col] = df[col].astype(str).str.strip()
                except Exception as e:
                    self._log.debug(f"Whitespace stripping failed: {e}")
            
            # Empty string → NaN
            if cfg.treat_empty_string_as_na:
                try:
                    df[obj_cols] = df[obj_cols].replace("", np.nan)
                except Exception as e:
                    self._log.debug(f"Empty string replacement failed: {e}")
        
        # Inf → NaN conversion
        try:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        except Exception as e:
            self._log.debug(f"Inf replacement failed: {e}")
        
        return df
    
    # ───────────────────────────────────────────────────────────────────
    # Intelligent Sampling
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("sampling")
    def _maybe_sample(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, bool, Optional[Dict[str, int]]]:
        """
        Sample DataFrame if too large for performance.
        
        Returns:
            Tuple of (sampled_df, was_sampled, sample_info)
        """
        cfg = self.config
        
        if not cfg.enable_sampling or len(df) <= cfg.sample_rows:
            return df, False, None
        
        # Sampling required
        sampled = True
        sample_info = {
            "from_rows": int(len(df)),
            "to_rows": int(cfg.sample_rows),
        }
        
        self._log.info(
            f"Sampling for missing analysis: {len(df):,} → {cfg.sample_rows:,} rows"
        )
        
        try:
            df_sampled = df.sample(
                n=cfg.sample_rows,
                random_state=cfg.random_state
            )
            return df_sampled, sampled, sample_info
        
        except Exception as e:
            self._log.warning(f"⚠ Sampling failed, using full dataset: {e}")
            return df, False, None
    
    # ───────────────────────────────────────────────────────────────────
    # Dataset-Level Summary
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("dataset_summary")
    @_safe_operation("dataset_summary", default_value={
        "total_cells": 0, "total_missing": 0, "missing_percentage": 0.0,
        "n_columns_with_missing": 0, "n_rows_with_missing": 0, "complete_rows": 0
    })
    def _get_missing_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute dataset-level missing data summary.
        
        Returns comprehensive statistics about missing values across the dataset.
        """
        rows, cols = int(df.shape[0]), int(df.shape[1])
        total_cells = rows * cols
        
        if total_cells == 0:
            return {
                "total_cells": 0,
                "total_missing": 0,
                "missing_percentage": 0.0,
                "n_columns_with_missing": 0,
                "n_rows_with_missing": 0,
                "complete_rows": rows,
            }
        
        # Compute missing mask
        isna = df.isna()
        total_missing = int(isna.values.sum())
        missing_percentage = float((total_missing / max(1, total_cells)) * 100.0)
        
        # Column statistics
        n_cols_with_missing = int((isna.sum(axis=0) > 0).sum())
        
        # Row statistics
        row_missing_mask = isna.any(axis=1)
        n_rows_with_missing = int(row_missing_mask.sum())
        complete_rows = int((~row_missing_mask).sum())
        
        return {
            "total_cells": total_cells,
            "total_missing": total_missing,
            "missing_percentage": round(missing_percentage, 2),
            "n_columns_with_missing": n_cols_with_missing,
            "n_rows_with_missing": n_rows_with_missing,
            "complete_rows": complete_rows,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Column-Level Analysis
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("column_analysis")
    def _analyze_missing_by_column(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze missing data patterns per column.
        
        Includes:
          • Severity classification
          • Imputation strategy recommendations
          • Indicator-like column detection
          • Co-missing pattern discovery
          • Fallback top co-missing when no strong correlations
        
        Returns:
            List of column analysis dictionaries (sorted by missing percentage)
        """
        cfg = self.config
        out: List[Dict[str, Any]] = []
        n = max(1, len(df))
        
        # Compute missing counts
        isna = df.isna()
        n_missing_per_col = isna.sum(axis=0)
        cols_with_missing = [c for c in df.columns if int(n_missing_per_col.get(c, 0)) > 0]
        
        if not cols_with_missing:
            return []
        
        # Build missing mask correlation matrix (if applicable)
        mask_matrix = self._build_mask_correlation_matrix(df, cols_with_missing)
        
        # Analyze each column with missing data
        for col in df.columns:
            n_missing = int(n_missing_per_col.get(col, 0))
            if n_missing == 0:
                continue
            
            s = df[col]
            missing_pct = float((n_missing / n) * 100.0)
            dtype_str = str(s.dtype)
            
            # Severity classification
            severity = self._classify_severity(missing_pct)
            
            # Strategy recommendation
            strategy = self._recommend_strategy(s, missing_pct)
            
            # Indicator-like detection
            non_na_unique = int(s.dropna().nunique())
            is_indicator_like = bool(
                non_na_unique <= 2 and missing_pct >= cfg.medium_threshold_pct
            )
            
            # Co-missing pattern discovery
            top_co = self._find_co_missing_patterns(col, mask_matrix)
            
            out.append({
                "column": str(col),
                "n_missing": n_missing,
                "missing_percentage": round(missing_pct, 2),
                "dtype": dtype_str,
                "severity": severity,
                "suggested_strategy": strategy,
                "is_indicator_like": is_indicator_like,
                "n_unique_non_na": non_na_unique,
                "top_co_missing_with": top_co,
            })
        
        # Sort by missing percentage (descending)
        out.sort(key=lambda x: x["missing_percentage"], reverse=True)
        return out
    
    @_safe_operation("mask_correlation_matrix", default_value=None)
    def _build_mask_correlation_matrix(
        self,
        df: pd.DataFrame,
        cols_with_missing: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        Build correlation matrix of missing masks.
        
        Features:
          • Column limiting for O(k²) safety
          • Zero-variance column removal
          • Defensive error handling
        
        Returns:
            DataFrame with mask correlations or None if insufficient data
        """
        cfg = self.config
        
        if len(cols_with_missing) < 2:
            return None
        
        # Limit columns for O(k²) safety
        k_max = min(len(cols_with_missing), cfg.max_corr_cols_for_masks)
        selected_cols = cols_with_missing[:k_max]
        
        # Build binary mask matrix
        mask_matrix = df[selected_cols].isna().astype(float)
        
        # Remove zero-variance columns
        std = mask_matrix.std(axis=0, ddof=0)
        keep_cols = std[std > 0].index
        
        if len(keep_cols) < 2:
            return None
        
        mask_matrix = mask_matrix[keep_cols]
        return mask_matrix
    
    @_safe_operation("co_missing_patterns", default_value=[])
    def _find_co_missing_patterns(
        self,
        col: str,
        mask_matrix: Optional[pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Find columns with correlated missing patterns.
        
        Strategy:
          1. Find correlations above threshold
          2. If none found, use fallback top-K by absolute correlation
        
        Returns:
            List of {"column": str, "mask_corr": float}
        """
        cfg = self.config
        
        if mask_matrix is None or col not in mask_matrix.columns:
            return []
        
        # Compute correlations with target column
        v = mask_matrix[col]
        if float(v.std(ddof=0)) <= 0.0:
            return []
        
        corr = mask_matrix.corrwith(v).drop(index=col, errors="ignore")
        corr_abs = corr.abs().dropna()
        
        if corr_abs.empty:
            return []
        
        corr_abs = corr_abs.sort_values(ascending=False)
        
        # Strategy 1: Above threshold
        top_pairs = corr_abs[corr_abs >= cfg.corr_mask_threshold].head(cfg.top_co_missing_with)
        if not top_pairs.empty:
            return [
                {"column": str(k), "mask_corr": round(float(corr[k]), 4)}
                for k in top_pairs.index
            ]
        
        # Strategy 2: Fallback top-K
        fallback = corr_abs.head(cfg.fallback_top_co_missing)
        return [
            {"column": str(k), "mask_corr": round(float(corr[k]), 4)}
            for k in fallback.index
        ]
    
    # ───────────────────────────────────────────────────────────────────
    # Row-Level Analysis
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("row_analysis")
    @_safe_operation("row_analysis", default_value={
        "top_rows_with_many_missing": [],
        "max_missing_in_row": 0
    })
    def _analyze_missing_by_row(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing data patterns per row.
        
        Returns:
          • Top rows with most missing values
          • Maximum missing values in any single row
          • Column names limited by max_return_indices
        """
        cfg = self.config
        
        isna = df.isna()
        n_missing_per_row = isna.sum(axis=1).astype(int)
        
        max_missing = int(n_missing_per_row.max())
        
        if max_missing == 0:
            return {
                "top_rows_with_many_missing": [],
                "max_missing_in_row": 0,
            }
        
        # Find top rows with most missing values
        top_idx = n_missing_per_row.sort_values(ascending=False).head(cfg.top_rows_limit).index
        
        top_rows: List[Dict[str, Any]] = []
        for idx in top_idx:
            # Format index (handle various types)
            idx_out = int(idx) if isinstance(idx, (int, np.integer)) else str(idx)
            
            # Get missing columns (limited for payload size)
            missing_cols_full = df.columns[isna.loc[idx]].tolist()
            missing_cols = [
                str(c) for c in missing_cols_full[:cfg.max_return_indices_per_rowlist]
            ]
            
            top_rows.append({
                "row_index": idx_out,
                "n_missing": int(n_missing_per_row.loc[idx]),
                "missing_cols": missing_cols,
            })
        
        return {
            "top_rows_with_many_missing": top_rows,
            "max_missing_in_row": max_missing,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Pattern Detection with Time Guard
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("pattern_detection")
    def _identify_patterns_with_time_guard(
        self,
        df: pd.DataFrame,
        rows_info: Dict[str, Any],
        column_analysis: List[Dict[str, Any]],
        started_at: float
    ) -> Tuple[Dict[str, Any], int, Dict[str, int]]:
        """
        Identify missing data patterns with adaptive time budgeting.
        
        Features:
          • Row block pattern detection
          • Mask correlation analysis with O(k²) safety
          • MAR/MNAR signal identification
          • Adaptive column reduction at 70% and 90% time budget
        
        Returns:
            Tuple of (patterns_dict, mask_cols_analyzed, caps_metadata)
        """
        cfg = self.config
        
        patterns = {
            "correlated": [],
            "blocks": {
                "rows_ge_threshold": 0,
                "threshold": int(cfg.row_missing_block_threshold),
            },
            "mar_mnar_signals": [],
        }
        
        # ─── Row Block Patterns
        isna_rows = df.isna().sum(axis=1)
        patterns["blocks"]["rows_ge_threshold"] = int(
            (isna_rows >= cfg.row_missing_block_threshold).sum()
        )
        
        # ─── Adaptive Column Selection for Correlation Analysis
        cols_with_missing = [c["column"] for c in column_analysis]
        requested = len(cols_with_missing)
        k_max = min(requested, cfg.max_corr_cols_for_masks)
        
        # Check elapsed time and reduce columns if needed
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        
        if elapsed_ms > cfg.soft_time_budget_ms * 0.9:
            k_max = max(2, int(k_max * cfg.time_budget_90pct_reduction))
            self._log.debug(f"Time budget 90% exceeded, reducing to {k_max} columns")
        elif elapsed_ms > cfg.soft_time_budget_ms * 0.7:
            k_max = max(2, int(k_max * cfg.time_budget_70pct_reduction))
            self._log.debug(f"Time budget 70% exceeded, reducing to {k_max} columns")
        
        mask_cols = cols_with_missing[:k_max]
        mask_cols_analyzed = len(mask_cols)
        
        caps_meta = {
            "requested": int(requested),
            "used": int(mask_cols_analyzed),
            "cap": int(cfg.max_corr_cols_for_masks),
        }
        
        # ─── Mask Correlation Analysis
        if mask_cols_analyzed >= 2:
            correlated = self._compute_mask_correlations(df, mask_cols)
            patterns["correlated"] = correlated
        
        # ─── MAR/MNAR Signal Detection
        mar_mnar_signals = self._detect_mar_mnar_signals(
            column_analysis=column_analysis,
            correlated_pairs=patterns["correlated"]
        )
        patterns["mar_mnar_signals"] = mar_mnar_signals
        
        return patterns, mask_cols_analyzed, caps_meta
    
    @_safe_operation("mask_correlations", default_value=[])
    def _compute_mask_correlations(
        self,
        df: pd.DataFrame,
        mask_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Compute correlations between missing masks.
        
        Returns:
            List of correlated column pairs with correlation coefficients
        """
        cfg = self.config
        
        # Build mask matrix
        mask = df[mask_cols].isna().astype(float)
        
        # Remove zero-variance columns
        std = mask.std(axis=0, ddof=0)
        keep = std[std > 0].index
        mask = mask[keep]
        
        if len(mask.columns) < 2:
            return []
        
        # Compute correlations
        corr = mask.corr()
        cols = list(corr.columns)
        
        # Find strong correlations
        correlated: List[Dict[str, Any]] = []
        for i in range(len(cols)):
            row_vals = corr.iloc[i, i+1:].dropna()
            if row_vals.empty:
                continue
            
            strong = row_vals[abs(row_vals) >= cfg.corr_mask_threshold]
            for other_col, val in strong.items():
                correlated.append({
                    "column": str(cols[i]),
                    "correlated_with": str(other_col),
                    "mask_corr": round(float(val), 4),
                })
        
        return correlated
    
    @_safe_operation("mar_mnar_detection", default_value=[])
    def _detect_mar_mnar_signals(
        self,
        column_analysis: List[Dict[str, Any]],
        correlated_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect MAR (Missing At Random) and MNAR (Missing Not At Random) signals.
        
        Heuristics:
          • MNAR: Indicator-like columns with high missing percentage
          • MAR: Columns with strongly correlated missing patterns
        
        Returns:
            List of MAR/MNAR signals with reasoning
        """
        cfg = self.config
        signals: List[Dict[str, Any]] = []
        
        # ─── MNAR Detection (Indicator-like columns)
        indicator_like_cols = {
            c["column"] for c in column_analysis
            if c.get("is_indicator_like") and c["missing_percentage"] >= cfg.medium_threshold_pct
        }
        
        for col in indicator_like_cols:
            signals.append({
                "column": col,
                "signal": "MNAR_like",
                "reason": "High missing percentage with indicator-like nature (≤2 unique non-NA values)",
            })
        
        # ─── MAR Detection (Correlated missing patterns)
        if correlated_pairs:
            involved_cols = set()
            for pair in correlated_pairs:
                involved_cols.add(pair["column"])
                involved_cols.add(pair["correlated_with"])
            
            for col in sorted(involved_cols):
                signals.append({
                    "column": col,
                    "signal": "MAR_like",
                    "reason": f"Missing pattern strongly correlated with other columns (|r| ≥ {cfg.corr_mask_threshold})",
                })
        
        return signals
    
    # ───────────────────────────────────────────────────────────────────
    # Severity Classification & Strategy Recommendations
    # ───────────────────────────────────────────────────────────────────
    
    def _classify_severity(self, missing_pct: float) -> str:
        """
        Classify missing data severity.
        
        Thresholds:
          • <5%: low
          • <20%: medium
          • <50%: high
          • ≥50%: critical
        """
        cfg = self.config
        
        if missing_pct < cfg.low_threshold_pct:
            return "low"
        elif missing_pct < cfg.medium_threshold_pct:
            return "medium"
        elif missing_pct < cfg.high_threshold_pct:
            return "high"
        else:
            return "critical"
    
    def _recommend_strategy(self, series: pd.Series, missing_pct: float) -> str:
        """
        Recommend imputation strategy based on column type and missing percentage.
        
        Strategies:
          • Datetime: Time-aware forward/backward fill
          • Numeric (low missing): Mean/median imputation
          • Numeric (high missing): KNN/MICE/interpolation
          • Categorical (low missing): Mode or missing category
          • Categorical (high missing): Missing indicator or category
          • High missing (>70%): Consider dropping column
        """
        cfg = self.config
        
        # Drop recommendation for severe missingness
        if missing_pct >= cfg.drop_column_over_pct:
            return "consider_dropping_column"
        
        # Datetime columns
        if pd.api.types.is_datetime64_any_dtype(series):
            return "time_aware_ffill_bfill"
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            if missing_pct < cfg.low_threshold_pct:
                return "mean_or_median_imputation"
            else:
                return "knn_mice_or_interpolation"
        
        # Categorical columns
        if series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
            if missing_pct < (cfg.low_threshold_pct + 5.0):
                return "mode_or_missing_category"
            else:
                return "missing_indicator_or_category"
        
        # Default fallback
        return "forward_fill_or_domain_specific"
    
    # ───────────────────────────────────────────────────────────────────
    # Recommendations Generation
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("recommendations")
    def _get_recommendations(
        self,
        column_analysis: List[Dict[str, Any]],
        rows_info: Dict[str, Any],
        summary: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        
        Categories:
          • Severity-based warnings (critical/high missing)
          • Type-specific strategies (numeric/categorical/datetime)
          • Row block pattern warnings
          • Correlation-based suggestions
          • MAR/MNAR handling guidance
        
        Returns:
            Deduplicated list of recommendations
        """
        cfg = self.config
        rec: List[str] = []
        
        # ─── Severity Buckets
        critical = [c for c in column_analysis if c["severity"] == "critical"]
        high = [c for c in column_analysis if c["severity"] == "high"]
        medium = [c for c in column_analysis if c["severity"] == "medium"]
        
        if critical:
            critical_names = ", ".join(c["column"] for c in critical[:3])
            if len(critical) > 3:
                critical_names += f" (+{len(critical)-3} more)"
            rec.append(
                f"🚨 {len(critical)} column(s) with ≥{cfg.high_threshold_pct}% missing: {critical_names}. "
                "Consider dropping or using models tolerant to missing data + missing indicators."
            )
        
        if high:
            high_names = ", ".join(c["column"] for c in high[:3])
            if len(high) > 3:
                high_names += f" (+{len(high)-3} more)"
            rec.append(
                f"⚠️ {len(high)} column(s) with {cfg.medium_threshold_pct}–{cfg.high_threshold_pct}% missing: {high_names}. "
                "Use advanced imputation (KNN/MICE) or consider 'missingness as feature'."
            )
        
        if medium:
            rec.append(
                f"ℹ️ {len(medium)} column(s) with {cfg.low_threshold_pct}–{cfg.medium_threshold_pct}% missing. "
                "Standard imputation methods (mean/median/mode) should suffice."
            )
        
        # ─── Type-Specific Strategies
        numeric_missing = [
            c for c in column_analysis
            if ("int" in c["dtype"].lower() or "float" in c["dtype"].lower())
        ]
        categorical_missing = [
            c for c in column_analysis
            if (c["dtype"] == "object" or "category" in c["dtype"].lower())
        ]
        datetime_missing = [
            c for c in column_analysis
            if "datetime" in c["dtype"].lower()
        ]
        
        if numeric_missing:
            rec.append(
                f"📊 {len(numeric_missing)} numeric column(s) with missing data. "
                "Strategies: median/quantile for skewed data, KNN for multivariate patterns, "
                "interpolation for time series."
            )
        
        if categorical_missing:
            rec.append(
                f"📝 {len(categorical_missing)} categorical column(s) with missing data. "
                "Strategies: mode for low missingness, 'Missing' category + indicator for moderate/high missingness."
            )
        
        if datetime_missing:
            rec.append(
                f"⏱️ {len(datetime_missing)} datetime column(s) with missing data. "
                "Prefer forward/backward fill within groups, or domain-specific temporal logic."
            )
        
        # ─── Row Block Patterns
        max_missing_row = rows_info.get("max_missing_in_row", 0)
        if max_missing_row >= cfg.row_missing_block_threshold:
            rec.append(
                f"🧱 Rows with ≥{cfg.row_missing_block_threshold} missing values detected "
                f"(max: {max_missing_row} in single row). "
                "Check data collection/merging logic, JOIN operations, and key integrity."
            )
        
        # ─── Correlated Missing Patterns
        if patterns.get("correlated"):
            n_corr = len(patterns["correlated"])
            rec.append(
                f"🔗 {n_corr} strongly correlated missing pattern(s) detected (|r| ≥ {cfg.corr_mask_threshold}). "
                "Consider multivariate imputation (KNN/MICE) that models co-missing relationships."
            )
        
        # ─── MAR/MNAR Signals
        mar_signals = [
            s for s in patterns.get("mar_mnar_signals", [])
            if s.get("signal") == "MAR_like"
        ]
        mnar_signals = [
            s for s in patterns.get("mar_mnar_signals", [])
            if s.get("signal") == "MNAR_like"
        ]
        
        if mar_signals:
            mar_cols = ", ".join(s["column"] for s in mar_signals[:3])
            if len(mar_signals) > 3:
                mar_cols += f" (+{len(mar_signals)-3} more)"
            rec.append(
                f"🧪 MAR-like pattern detected in: {mar_cols}. "
                "Missing depends on other observed variables. "
                "Use multivariate imputation and validate post-imputation distributions."
            )
        
        if mnar_signals:
            mnar_cols = ", ".join(s["column"] for s in mnar_signals[:3])
            if len(mnar_signals) > 3:
                mnar_cols += f" (+{len(mnar_signals)-3} more)"
            rec.append(
                f"🕳️ MNAR-like pattern detected in: {mnar_cols}. "
                "Missing may depend on unobserved data. "
                "Consider: (1) modeling missingness itself (indicators), "
                "(2) analyzing data collection process, (3) sensitivity analysis, or (4) dropping if unreliable."
            )
        
        # ─── General Best Practices
        if summary.get("missing_percentage", 0.0) > 0:
            rec.append(
                "💡 Best practices: (1) Create missing indicators for medium/high missingness, "
                "(2) Compare pre/post imputation distributions, "
                "(3) Use cross-validation to validate imputation quality, "
                "(4) Consider models robust to missing data (XGBoost, LightGBM with native support)."
            )
        
        # Deduplicate and return
        return list(dict.fromkeys([r for r in rec if r]))
    
    # ───────────────────────────────────────────────────────────────────
    # Payload Construction
    # ───────────────────────────────────────────────────────────────────
    
    def _build_no_missing_payload(
        self,
        df: pd.DataFrame,
        started_at: float,
        sampled: bool,
        sample_info: Optional[Dict[str, int]]
    ) -> Dict[str, Any]:
        """Build payload when no missing data is found."""
        cfg = self.config
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        
        rows, cols = df.shape
        total_cells = rows * cols
        
        return {
            "summary": {
                "total_cells": total_cells,
                "total_missing": 0,
                "missing_percentage": 0.0,
                "n_columns_with_missing": 0,
                "n_rows_with_missing": 0,
                "complete_rows": rows,
            },
            "columns": [],
            "rows": {
                "top_rows_with_many_missing": [],
                "max_missing_in_row": 0,
            },
            "patterns": {
                "correlated": [],
                "blocks": {
                    "rows_ge_threshold": 0,
                    "threshold": cfg.row_missing_block_threshold,
                },
                "mar_mnar_signals": [],
            },
            "recommendations": [
                "✅ No missing data detected — you can skip imputation steps."
            ],
            "telemetry": {
                "elapsed_ms": elapsed_ms,
                "mask_cols_analyzed": 0,
                "sampled": sampled,
                "sample_info": sample_info,
                "soft_time_budget_ms": cfg.soft_time_budget_ms,
                "soft_time_exceeded": bool(elapsed_ms > cfg.soft_time_budget_ms),
                "caps": {
                    "corr_mask_cols_requested": 0,
                    "corr_mask_cols_used": 0,
                    "corr_mask_cols_cap": cfg.max_corr_cols_for_masks,
                },
                "debug": {
                    "object_empty_as_na": cfg.treat_empty_string_as_na,
                    "strip_whitespace_in_object": cfg.strip_whitespace_in_object,
                },
            },
            "version": "5.0-kosmos-enterprise",
        }
    
    @staticmethod
    def _empty_payload() -> Dict[str, Any]:
        """Generate empty payload for failed/invalid input."""
        return {
            "summary": {
                "total_cells": 0,
                "total_missing": 0,
                "missing_percentage": 0.0,
                "n_columns_with_missing": 0,
                "n_rows_with_missing": 0,
                "complete_rows": 0,
            },
            "columns": [],
            "rows": {
                "top_rows_with_many_missing": [],
                "max_missing_in_row": 0,
            },
            "patterns": {
                "correlated": [],
                "blocks": {
                    "rows_ge_threshold": 0,
                    "threshold": 3,
                },
                "mar_mnar_signals": [],
            },
            "recommendations": [
                "Provide valid data for missing data analysis"
            ],
            "telemetry": {
                "elapsed_ms": 0.0,
                "mask_cols_analyzed": 0,
                "sampled": False,
                "sample_info": None,
                "soft_time_budget_ms": 0,
                "soft_time_exceeded": False,
                "caps": {
                    "corr_mask_cols_requested": 0,
                    "corr_mask_cols_used": 0,
                    "corr_mask_cols_cap": 0,
                },
                "debug": {
                    "object_empty_as_na": False,
                    "strip_whitespace_in_object": False,
                },
            },
            "version": "5.0-kosmos-enterprise",
        }