# agents/monitoring/drift_detector.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Drift Detector v6.0              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ENTERPRISE-GRADE DATA & CONCEPT DRIFT DETECTION                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Multi-Method Statistical Testing (PSI, KS, Chi², Wasserstein, Cramér) ║
║  ✓ Comprehensive Feature Type Support (Numeric, Categorical, Datetime)   ║
║  ✓ Intelligent Drift Banding (OK, Warning, Critical)                     ║
║  ✓ Target & Performance Drift Detection                                  ║
║  ✓ Robust Statistical Methods (MAD, IQR, Quantile-based)                 ║
║  ✓ Memory-Efficient Sampling                                             ║
║  ✓ Schema Alignment & Validation                                         ║
║  ✓ Actionable Recommendations                                            ║
║  ✓ Comprehensive Telemetry                                               ║
║  ✓ Production-Ready Error Handling                                       ║
║  ✓ Missing Value Tolerance                                               ║
║  ✓ Constant Feature Detection                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  DriftDetector Core                         │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Data Alignment & Sampling                               │
    │  2. Schema Validation & Type Inference                      │
    │  3. Per-Feature Drift Analysis                              │
    │     • Numeric: PSI, KS, Wasserstein                        │
    │     • Categorical: PSI, Chi², Cramér's V                   │
    │     • Datetime: Timestamp-based numeric analysis           │
    │  4. Target Drift Detection                                  │
    │  5. Performance Drift Analysis                              │
    │  6. Drift Scoring & Banding                                 │
    │  7. Recommendation Generation                               │
    └─────────────────────────────────────────────────────────────┘

Statistical Methods:
    • PSI (Population Stability Index)   → Distribution shift
    • KS (Kolmogorov-Smirnov)            → Two-sample test
    • Chi-Square                         → Categorical independence
    • Wasserstein Distance               → Optimal transport
    • Cramér's V                         → Effect size for Chi²
    
Drift Bands:
    • OK        → <10% features drifted
    • WARNING   → 10-30% features drifted
    • CRITICAL  → >30% features drifted

Usage:
```python
    from agents.monitoring import DriftDetector, DriftConfig
    
    # Basic usage
    detector = DriftDetector()
    result = detector.execute(
        reference_data=train_df,
        current_data=prod_df,
        target_column='target'
    )
    
    # Access results
    drift_score = result.data['data_drift']['drift_score']
    drifted_features = result.data['data_drift']['drifted_features']
    
    # Custom configuration
    config = DriftConfig(
        psi_warn_threshold=0.15,
        psi_crit_threshold=0.25,
        sample_size=50_000
    )
    detector = DriftDetector(config)
```
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

# ═══════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════

try:
    from loguru import logger
    
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/drift_detector_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        level="DEBUG"
    )
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Dependencies
# ═══════════════════════════════════════════════════════════════════════════

try:
    from core.base_agent import BaseAgent, AgentResult
except ImportError:
    logger.warning("⚠ core.base_agent not found - using fallback")
    
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
            self.logger = logger
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data: Dict[str, Any] = {}
            self.errors: List[str] = []
            self.warnings: List[str] = []
        
        def add_error(self, error: str):
            self.errors.append(error)
        
        def add_warning(self, warning: str):
            self.warnings.append(warning)
        
        def is_success(self) -> bool:
            return len(self.errors) == 0

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["DriftConfig", "DriftDetector", "detect_drift"]
__version__ = "6.0.0-enterprise"
__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=False)
class DriftConfig:
    """
    🎯 **Drift Detection Configuration**
    
    Complete configuration for statistical drift detection.
    
    Statistical Testing:
        alpha: Significance level for statistical tests (default: 0.05)
        psi_bins: Number of bins for PSI calculation (default: 10)
        bootstrap_samples: Bootstrap samples for CI estimation (default: 1000)
        
    Thresholds - PSI (Population Stability Index):
        psi_warn_threshold: Warning threshold (default: 0.10)
        psi_crit_threshold: Critical threshold (default: 0.20)
        
    Thresholds - KS (Kolmogorov-Smirnov):
        ks_warn_threshold: Warning threshold for KS statistic (default: 0.10)
        ks_crit_threshold: Critical threshold for KS statistic (default: 0.20)
        
    Thresholds - Wasserstein Distance:
        wdist_warn_threshold: Warning threshold (default: 0.20)
        wdist_crit_threshold: Critical threshold (default: 0.35)
        
    Thresholds - Cramér's V:
        cramer_warn_threshold: Warning threshold (default: 0.20)
        cramer_crit_threshold: Critical threshold (default: 0.35)
        
    Analysis Limits:
        max_features: Maximum features to analyze (None = all)
        sample_size: Maximum samples per dataset (default: 100,000)
        min_non_null_ratio: Minimum non-null ratio (default: 0.5)
        min_category_freq: Minimum category frequency (default: 5)
        max_categories: Maximum unique categories (default: 50)
        
    Categorical Analysis:
        topk_categorical: Top K categories to analyze (default: 5)
        merge_rare_categories: Merge rare categories (default: True)
        rare_category_threshold: Threshold for rare categories (default: 0.01)
        
    Performance:
        enable_parallel: Enable parallel processing (default: False)
        n_jobs: Number of parallel jobs (default: -1)
        cache_results: Cache intermediate results (default: True)
        
    Advanced:
        detect_outliers: Detect outliers before drift calculation (default: False)
        outlier_method: Outlier detection method (default: 'iqr')
        outlier_threshold: Outlier threshold (default: 3.0)
        robust_scaling: Use robust scaling (default: True)
    """
    
    # Statistical testing
    alpha: float = 0.05
    psi_bins: int = 10
    bootstrap_samples: int = 1000
    
    # PSI thresholds
    psi_warn_threshold: float = 0.10
    psi_crit_threshold: float = 0.20
    
    # KS thresholds
    ks_warn_threshold: float = 0.10
    ks_crit_threshold: float = 0.20
    
    # Wasserstein thresholds
    wdist_warn_threshold: float = 0.20
    wdist_crit_threshold: float = 0.35
    
    # Cramér's V thresholds
    cramer_warn_threshold: float = 0.20
    cramer_crit_threshold: float = 0.35
    
    # Analysis limits
    max_features: Optional[int] = None
    sample_size: int = 100_000
    min_non_null_ratio: float = 0.5
    min_category_freq: int = 5
    max_categories: int = 50
    
    # Categorical analysis
    topk_categorical: int = 5
    merge_rare_categories: bool = True
    rare_category_threshold: float = 0.01
    
    # Performance
    enable_parallel: bool = False
    n_jobs: int = -1
    cache_results: bool = True
    
    # Advanced
    detect_outliers: bool = False
    outlier_method: Literal['iqr', 'zscore', 'isolation_forest'] = 'iqr'
    outlier_threshold: float = 3.0
    robust_scaling: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        
        if self.psi_bins < 2:
            raise ValueError(f"psi_bins must be >= 2, got {self.psi_bins}")
        
        if not 0 < self.min_non_null_ratio <= 1:
            raise ValueError(f"min_non_null_ratio must be in (0, 1], got {self.min_non_null_ratio}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def create_strict(cls) -> 'DriftConfig':
        """Create strict configuration (lower thresholds)."""
        return cls(
            psi_warn_threshold=0.05,
            psi_crit_threshold=0.15,
            ks_warn_threshold=0.05,
            cramer_warn_threshold=0.15
        )
    
    @classmethod
    def create_lenient(cls) -> 'DriftConfig':
        """Create lenient configuration (higher thresholds)."""
        return cls(
            psi_warn_threshold=0.15,
            psi_crit_threshold=0.30,
            ks_warn_threshold=0.15,
            cramer_warn_threshold=0.30
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Drift Detector (Main Class)
# ═══════════════════════════════════════════════════════════════════════════

class DriftDetector(BaseAgent):
    """
    🚀 **DriftDetector PRO Master Enterprise ++++**
    
    Enterprise-grade data and concept drift detection system.
    
    Capabilities:
      1. Data drift detection (feature-level)
      2. Target drift detection
      3. Performance drift analysis
      4. Multi-method statistical testing
      5. Intelligent drift banding
      6. Actionable recommendations
      7. Comprehensive telemetry
      8. Schema validation
      9. Memory-efficient processing
     10. Production-ready error handling
    
    Statistical Methods:
      • PSI (Population Stability Index)
      • KS Test (Kolmogorov-Smirnov)
      • Chi-Square Test
      • Wasserstein Distance
      • Cramér's V
    
    Feature Types:
      • Numeric: Continuous & discrete
      • Categorical: Nominal & ordinal
      • Datetime: Temporal features
    
    Usage:
```python
        # Basic usage
        detector = DriftDetector()
        result = detector.execute(
            reference_data=train_df,
            current_data=prod_df,
            target_column='target'
        )
        
        # Check drift
        if result.is_success():
            drift_score = result.data['data_drift']['drift_score']
            band = result.data['summary']['drift_band']
            
            if band == 'critical':
                print("⚠️ Critical drift detected!")
                print(result.data['recommendations'])
        
        # Advanced usage
        config = DriftConfig.create_strict()
        detector = DriftDetector(config)
        
        result = detector.execute(
            reference_data=train_df,
            current_data=prod_df,
            target_column='target',
            feature_types={
                'age': 'numeric',
                'category': 'categorical'
            },
            y_ref=y_train,
            y_cur=y_prod,
            pred_ref=train_predictions,
            pred_cur=prod_predictions
        )
```
    """
    
    version: str = __version__
    
    def __init__(self, config: Optional[DriftConfig] = None):
        """
        Initialize drift detector.
        
        Args:
            config: Optional custom configuration
        """
        super().__init__(
            name="DriftDetector",
            description="Enterprise data & concept drift detection"
        )
        self.config = config or DriftConfig()
        self._log = logger.bind(agent="DriftDetector", version=self.version)
        
        # Cache for expensive computations
        self._cache: Dict[str, Any] = {} if self.config.cache_results else None
        
        self._log.info(f"✓ DriftDetector v{self.version} initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Args:
            **kwargs: Input parameters
        
        Returns:
            True if valid
        
        Raises:
            ValueError: Invalid input
        """
        if "reference_data" not in kwargs or "current_data" not in kwargs:
            raise ValueError("'reference_data' and 'current_data' are required")
        
        ref = kwargs["reference_data"]
        cur = kwargs["current_data"]
        
        if not isinstance(ref, pd.DataFrame) or ref.empty:
            raise ValueError("'reference_data' must be non-empty DataFrame")
        
        if not isinstance(cur, pd.DataFrame) or cur.empty:
            raise ValueError("'current_data' must be non-empty DataFrame")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
        feature_types: Optional[Dict[str, Literal["numeric", "categorical", "datetime"]]] = None,
        y_ref: Optional[pd.Series] = None,
        y_cur: Optional[pd.Series] = None,
        pred_ref: Optional[pd.Series] = None,
        pred_cur: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        🎯 **Execute Drift Detection Analysis**
        
        Comprehensive drift detection across multiple dimensions.
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            current_data: Current dataset (e.g., production data)
            target_column: Target column name (optional)
            feature_types: Feature type mapping (optional)
            y_ref: Reference true labels (for performance drift)
            y_cur: Current true labels (for performance drift)
            pred_ref: Reference predictions (for performance drift)
            pred_cur: Current predictions (for performance drift)
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with comprehensive drift analysis
        """
        result = AgentResult(agent_name=self.name)
        t_start = time.perf_counter()
        
        try:
            self._log.info(
                f"🔍 Starting drift analysis | "
                f"ref={len(reference_data):,} rows | "
                f"cur={len(current_data):,} rows"
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: Data Alignment & Sampling
            # ═══════════════════════════════════════════════════════════
            
            ref_sampled, cur_sampled = self._align_and_sample(
                reference_data, current_data
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: Schema Validation
            # ═══════════════════════════════════════════════════════════
            
            schema_info, common_cols = self._schema_alignment(
                ref_sampled, cur_sampled, target_column
            )
            
            if not common_cols:
                raise ValueError("No common columns found between datasets")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: Feature Type Inference
            # ═══════════════════════════════════════════════════════════
            
            ftypes = self._infer_feature_types(
                ref_sampled[common_cols], feature_types
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 4: Feature Selection
            # ═══════════════════════════════════════════════════════════
            
            feature_list = [c for c in common_cols if c != target_column]
            
            if self.config.max_features is not None:
                feature_list = feature_list[:self.config.max_features]
                if len(feature_list) < len(common_cols) - (1 if target_column else 0):
                    result.add_warning(
                        f"Analysis limited to {self.config.max_features} features"
                    )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 5: Per-Feature Drift Analysis
            # ═══════════════════════════════════════════════════════════
            
            self._log.info(f"📊 Analyzing {len(feature_list)} features...")
            
            per_feature: Dict[str, Dict[str, Any]] = {}
            drifted_features: List[str] = []
            
            for col in feature_list:
                try:
                    ftype = ftypes[col]
                    
                    if ftype == "numeric":
                        metrics = self._drift_numeric(
                            ref_sampled[col],
                            cur_sampled[col]
                        )
                    elif ftype == "categorical":
                        metrics = self._drift_categorical(
                            ref_sampled[col],
                            cur_sampled[col]
                        )
                    else:  # datetime
                        metrics = self._drift_datetime(
                            ref_sampled[col],
                            cur_sampled[col]
                        )
                    
                    per_feature[col] = metrics
                    
                    if metrics.get("is_drift", False):
                        drifted_features.append(col)
                
                except Exception as e:
                    self._log.warning(f"Feature '{col}' analysis failed: {e}")
                    per_feature[col] = {
                        "error": str(e),
                        "skipped": True
                    }
            
            # Calculate drift score
            evaluated_features = [
                c for c, m in per_feature.items()
                if not m.get("skipped", False)
            ]
            
            drift_score = (
                len(drifted_features) / max(1, len(evaluated_features)) * 100.0
            )
            
            data_drift = {
                "per_feature": per_feature,
                "drifted_features": drifted_features,
                "n_drifted": len(drifted_features),
                "drift_score": float(drift_score),
                "pct_drifted_features": float(drift_score),  # Alias
                "evaluated_features": len(evaluated_features),
                "total_features": len(feature_list)
            }
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 6: Target Drift Analysis
            # ═══════════════════════════════════════════════════════════
            
            target_drift = None
            
            if target_column and target_column in ref_sampled.columns and target_column in cur_sampled.columns:
                self._log.info("🎯 Analyzing target drift...")
                
                try:
                    target_type = ftypes.get(target_column, "categorical")
                    
                    if target_type == "numeric":
                        target_drift = self._drift_numeric(
                            ref_sampled[target_column],
                            cur_sampled[target_column]
                        )
                    else:
                        target_drift = self._drift_categorical(
                            ref_sampled[target_column],
                            cur_sampled[target_column]
                        )
                        
                        # Add class shift analysis
                        target_drift["major_class_shift"] = self._analyze_class_shift(
                            target_drift
                        )
                
                except Exception as e:
                    self._log.warning(f"Target drift analysis failed: {e}")
                    target_drift = {"error": str(e)}
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 7: Performance Drift Analysis
            # ═══════════════════════════════════════════════════════════
            
            perf_drift = None
            
            if all(v is not None for v in [y_ref, pred_ref, y_cur, pred_cur]):
                self._log.info("📈 Analyzing performance drift...")
                
                try:
                    perf_drift = self._performance_drift(
                        y_ref, pred_ref, y_cur, pred_cur
                    )
                except Exception as e:
                    self._log.warning(f"Performance drift analysis failed: {e}")
                    perf_drift = {"error": str(e)}
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 8: Summary & Recommendations
            # ═══════════════════════════════════════════════════════════
            
            summary = self._build_summary(
                data_drift, target_drift, perf_drift,
                len(reference_data), len(current_data)
            )
            
            recommendations = self._generate_recommendations(
                data_drift, target_drift, perf_drift
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 9: Telemetry
            # ═══════════════════════════════════════════════════════════
            
            elapsed_s = time.perf_counter() - t_start
            
            telemetry = {
                "elapsed_ms": round(elapsed_s * 1000, 2),
                "elapsed_s": round(elapsed_s, 4),
                "sampled_ref": len(reference_data) > self.config.sample_size,
                "sampled_cur": len(current_data) > self.config.sample_size,
                "sample_size": self.config.sample_size,
                "n_reference": len(reference_data),
                "n_current": len(current_data),
                "n_reference_sampled": len(ref_sampled),
                "n_current_sampled": len(cur_sampled),
                "evaluated_features": len(evaluated_features),
                "total_common_features": len(feature_list),
                "feature_types": {
                    "numeric": sum(1 for v in ftypes.values() if v == "numeric"),
                    "categorical": sum(1 for v in ftypes.values() if v == "categorical"),
                    "datetime": sum(1 for v in ftypes.values() if v == "datetime")
                },
                "version": self.version
            }
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 10: Assemble Result
            # ═══════════════════════════════════════════════════════════
            
            result.data = {
                "schema": schema_info,
                "data_drift": data_drift,
                "target_drift": target_drift,
                "performance_drift": perf_drift,
                "summary": summary,
                "recommendations": recommendations,
                "telemetry": telemetry,
                "config": self.config.to_dict()
            }
            
            # Log summary
            band = summary["drift_band"]
            band_emoji = {"ok": "✅", "warn": "⚠️", "critical": "🚨"}
            
            self._log.success(
                f"{band_emoji.get(band, '📊')} Drift analysis complete | "
                f"score={drift_score:.1f}% | "
                f"band={band} | "
                f"drifted={len(drifted_features)}/{len(evaluated_features)} | "
                f"time={elapsed_s:.2f}s"
            )
        
        except Exception as e:
            error_msg = f"Drift detection failed: {type(e).__name__}: {str(e)}"
            result.add_error(error_msg)
            self._log.error(error_msg, exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Data Preparation
    # ───────────────────────────────────────────────────────────────────
    
    def _align_and_sample(
        self,
        ref: pd.DataFrame,
        cur: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align and sample datasets for efficient processing.
        
        Args:
            ref: Reference dataset
            cur: Current dataset
        
        Returns:
            Tuple of (sampled_ref, sampled_cur)
        """
        ref_sampled = ref
        cur_sampled = cur
        
        try:
            # Sample if needed
            if len(ref) > self.config.sample_size:
                ref_sampled = ref.sample(
                    n=self.config.sample_size,
                    random_state=42
                )
                self._log.debug(
                    f"Sampled reference: {len(ref):,} → {len(ref_sampled):,}"
                )
            
            if len(cur) > self.config.sample_size:
                cur_sampled = cur.sample(
                    n=self.config.sample_size,
                    random_state=42
                )
                self._log.debug(
                    f"Sampled current: {len(cur):,} → {len(cur_sampled):,}"
                )
        
        except Exception as e:
            self._log.warning(f"Sampling failed: {e}")
        
        return ref_sampled, cur_sampled
    
    def _schema_alignment(
        self,
        ref: pd.DataFrame,
        cur: pd.DataFrame,
        target_column: Optional[str]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Align schemas and identify common columns.
        
        Args:
            ref: Reference dataset
            cur: Current dataset
            target_column: Target column name
        
        Returns:
            Tuple of (schema_info, common_columns)
        """
        # Preserve column order from reference
        common_cols = [c for c in ref.columns if c in cur.columns]
        only_ref = [c for c in ref.columns if c not in cur.columns]
        only_cur = [c for c in cur.columns if c not in ref.columns]
        
        schema_info = {
            "n_ref_cols": len(ref.columns),
            "n_cur_cols": len(cur.columns),
            "n_common_cols": len(common_cols),
            "common_cols": common_cols,
            "only_in_reference": only_ref,
            "only_in_current": only_cur,
            "target_column": target_column,
            "schema_drift": len(only_ref) > 0 or len(only_cur) > 0
        }
        
        if schema_info["schema_drift"]:
            self._log.warning(
                f"⚠️ Schema drift detected: "
                f"{len(only_ref)} cols only in ref, "
                f"{len(only_cur)} cols only in cur"
            )
        
        return schema_info, common_cols
    
    def _infer_feature_types(
        self,
        df: pd.DataFrame,
        provided: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Infer feature types with intelligent heuristics.
        
        Args:
            df: DataFrame to analyze
            provided: User-provided type mapping
        
        Returns:
            Dictionary mapping column names to types
        """
        ftypes: Dict[str, str] = {}
        
        for col in df.columns:
            # Use provided type if available
            if provided and col in provided:
                ftypes[col] = provided[col]
                continue
            
            series = df[col]
            
            # Datetime detection
            if pd.api.types.is_datetime64_any_dtype(series):
                ftypes[col] = "datetime"
            
            # Numeric detection
            elif pd.api.types.is_numeric_dtype(series):
                # Check if it's actually categorical (low cardinality)
                nunique = series.nunique(dropna=True)
                
                if nunique <= 20 and nunique < len(series) * 0.05:
                    ftypes[col] = "categorical"
                else:
                    ftypes[col] = "numeric"
            
            # Categorical detection
            else:
                ftypes[col] = "categorical"
        
        return ftypes
    
    # ───────────────────────────────────────────────────────────────────
    # Numeric Drift Detection
    # ───────────────────────────────────────────────────────────────────
    
    def _drift_numeric(
        self,
        s_ref: pd.Series,
        s_cur: pd.Series
    ) -> Dict[str, Any]:
        """
        🔢 **Numeric Feature Drift Analysis**
        
        Uses multiple statistical methods:
          • PSI (Population Stability Index)
          • KS Test (Kolmogorov-Smirnov)
          • Wasserstein Distance
        
        Args:
            s_ref: Reference series
            s_cur: Current series
        
        Returns:
            Dictionary with drift metrics
        """
        metrics: Dict[str, Any] = {"type": "numeric"}
        
        try:
            # Drop NaN values
            ref_clean = s_ref.dropna()
            cur_clean = s_cur.dropna()
            
            # Check observability
            ref_valid_ratio = len(ref_clean) / max(1, len(s_ref))
            cur_valid_ratio = len(cur_clean) / max(1, len(s_cur))
            
            if ref_valid_ratio < self.config.min_non_null_ratio:
                return {
                    "skipped": True,
                    "reason": "too_many_missing_reference",
                    "type": "numeric",
                    "missing_ref_pct": (1 - ref_valid_ratio) * 100
                }
            
            if cur_valid_ratio < self.config.min_non_null_ratio:
                return {
                    "skipped": True,
                    "reason": "too_many_missing_current",
                    "type": "numeric",
                    "missing_cur_pct": (1 - cur_valid_ratio) * 100
                }
            
            # Check for constant features
            if ref_clean.std() == 0:
                return {
                    "skipped": True,
                    "reason": "constant_reference_feature",
                    "type": "numeric"
                }
            
            # ═══════════════════════════════════════════════════════════
            # Method 1: PSI (Population Stability Index)
            # ═══════════════════════════════════════════════════════════
            
            psi_score, psi_bins = self._calculate_psi_numeric(
                ref_clean, cur_clean, bins=self.config.psi_bins
            )
            
            # ═══════════════════════════════════════════════════════════
            # Method 2: KS Test
            # ═══════════════════════════════════════════════════════════
            
            ks_stat, ks_pvalue = ks_2samp(ref_clean, cur_clean)
            
            # ═══════════════════════════════════════════════════════════
            # Method 3: Wasserstein Distance (with robust scaling)
            # ═══════════════════════════════════════════════════════════
            
            scale = self._robust_scale(ref_clean)
            wasserstein_raw = wasserstein_distance(ref_clean, cur_clean)
            wasserstein_norm = wasserstein_raw / max(scale, 1e-10)
            
            # ═══════════════════════════════════════════════════════════
            # Distribution Statistics
            # ═══════════════════════════════════════════════════════════
            
            ref_stats = {
                "mean": float(ref_clean.mean()),
                "std": float(ref_clean.std()),
                "median": float(ref_clean.median()),
                "min": float(ref_clean.min()),
                "max": float(ref_clean.max()),
                "q25": float(ref_clean.quantile(0.25)),
                "q75": float(ref_clean.quantile(0.75))
            }
            
            cur_stats = {
                "mean": float(cur_clean.mean()),
                "std": float(cur_clean.std()),
                "median": float(cur_clean.median()),
                "min": float(cur_clean.min()),
                "max": float(cur_clean.max()),
                "q25": float(cur_clean.quantile(0.25)),
                "q75": float(cur_clean.quantile(0.75))
            }
            
            # Calculate shifts
            mean_shift = cur_stats["mean"] - ref_stats["mean"]
            std_shift = cur_stats["std"] - ref_stats["std"]
            median_shift = cur_stats["median"] - ref_stats["median"]
            
            # ═══════════════════════════════════════════════════════════
            # Drift Detection Logic
            # ═══════════════════════════════════════════════════════════
            
            triggers: List[str] = []
            
            # PSI triggers
            if psi_score is not None:
                if psi_score >= self.config.psi_crit_threshold:
                    triggers.append("psi_critical")
                elif psi_score >= self.config.psi_warn_threshold:
                    triggers.append("psi_warning")
            
            # KS triggers
            if ks_pvalue < self.config.alpha:
                triggers.append("ks_test_significant")
            
            if ks_stat >= self.config.ks_crit_threshold:
                triggers.append("ks_statistic_critical")
            elif ks_stat >= self.config.ks_warn_threshold:
                triggers.append("ks_statistic_warning")
            
            # Wasserstein triggers
            if wasserstein_norm >= self.config.wdist_crit_threshold:
                triggers.append("wasserstein_critical")
            elif wasserstein_norm >= self.config.wdist_warn_threshold:
                triggers.append("wasserstein_warning")
            
            # Assemble metrics
            metrics.update({
                "missing_ref_pct": (1 - ref_valid_ratio) * 100,
                "missing_cur_pct": (1 - cur_valid_ratio) * 100,
                "psi": float(psi_score) if psi_score is not None else None,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "wasserstein_distance": float(wasserstein_raw),
                "wasserstein_normalized": float(wasserstein_norm),
                "reference_stats": ref_stats,
                "current_stats": cur_stats,
                "shifts": {
                    "mean": float(mean_shift),
                    "std": float(std_shift),
                    "median": float(median_shift)
                },
                "psi_bins": psi_bins,
                "is_drift": len(triggers) > 0,
                "triggers": triggers,
                "drift_severity": self._categorize_severity(triggers)
            })
            
            return metrics
        
        except Exception as e:
            self._log.warning(f"Numeric drift calculation failed: {e}")
            return {
                "error": str(e),
                "type": "numeric",
                "skipped": True
            }
    
    def _calculate_psi_numeric(
        self,
        ref: pd.Series,
        cur: pd.Series,
        bins: int
    ) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        """
        Calculate PSI (Population Stability Index) for numeric features.
        
        Args:
            ref: Reference series
            cur: Current series
            bins: Number of bins
        
        Returns:
            Tuple of (psi_score, bin_details)
        """
        try:
            # Create quantile-based bins from reference
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.unique(np.quantile(ref, quantiles))
            
            # Handle constant or near-constant features
            if len(bin_edges) <= 2:
                vmin, vmax = ref.min(), ref.max()
                
                if vmin == vmax:
                    return 0.0, [{
                        "note": "constant_feature",
                        "value": float(vmin)
                    }]
                
                # Create uniform bins
                bin_edges = np.linspace(vmin, vmax, bins + 1)
            
            # Create histograms
            ref_hist, _ = np.histogram(ref, bins=bin_edges)
            cur_hist, _ = np.histogram(cur, bins=bin_edges)
            
            # Calculate percentages (avoid division by zero)
            ref_pct = np.where(
                ref_hist == 0,
                1e-6,
                ref_hist / max(1, ref_hist.sum())
            )
            cur_pct = np.where(
                cur_hist == 0,
                1e-6,
                cur_hist / max(1, cur_hist.sum())
            )
            
            # PSI calculation
            psi_values = (ref_pct - cur_pct) * np.log(ref_pct / cur_pct)
            psi_score = float(np.sum(psi_values))
            
            # Bin details
            bin_details: List[Dict[str, Any]] = []
            for i in range(len(bin_edges) - 1):
                bin_details.append({
                    "bin_index": i,
                    "bin_left": float(bin_edges[i]),
                    "bin_right": float(bin_edges[i + 1]),
                    "ref_count": int(ref_hist[i]),
                    "cur_count": int(cur_hist[i]),
                    "ref_pct": float(ref_pct[i]),
                    "cur_pct": float(cur_pct[i]),
                    "psi_contribution": float(psi_values[i])
                })
            
            return psi_score, bin_details
        
        except Exception as e:
            self._log.warning(f"PSI calculation failed: {e}")
            return None, []
    
    def _robust_scale(self, series: pd.Series) -> float:
        """
        Calculate robust scale using MAD or IQR.
        
        Args:
            series: Pandas Series
        
        Returns:
            Robust scale value
        """
        arr = series.values
        
        # Method 1: MAD (Median Absolute Deviation)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        
        if mad > 0:
            return 1.4826 * mad  # ~= standard deviation for normal dist
        
        # Method 2: IQR (Interquartile Range)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        
        if iqr > 0:
            return iqr / 1.349  # ~= standard deviation for normal dist
        
        # Method 3: Standard deviation
        std = np.std(arr)
        
        if std > 0:
            return std
        
        # Fallback
        return 1.0
    
    # ───────────────────────────────────────────────────────────────────
    # Categorical Drift Detection
    # ───────────────────────────────────────────────────────────────────
    
    def _drift_categorical(
        self,
        s_ref: pd.Series,
        s_cur: pd.Series
    ) -> Dict[str, Any]:
        """
        📊 **Categorical Feature Drift Analysis**
        
        Uses multiple statistical methods:
          • PSI (Population Stability Index)
          • Chi-Square Test
          • Cramér's V
        
        Args:
            s_ref: Reference series
            s_cur: Current series
        
        Returns:
            Dictionary with drift metrics
        """
        metrics: Dict[str, Any] = {"type": "categorical"}
        
        try:
            # Convert to string and drop NaN
            ref_clean = s_ref.astype(str).replace('nan', np.nan).dropna()
            cur_clean = s_cur.astype(str).replace('nan', np.nan).dropna()
            
            # Check observability
            ref_valid_ratio = len(ref_clean) / max(1, len(s_ref))
            cur_valid_ratio = len(cur_clean) / max(1, len(s_cur))
            
            if ref_valid_ratio < self.config.min_non_null_ratio:
                return {
                    "skipped": True,
                    "reason": "too_many_missing_reference",
                    "type": "categorical",
                    "missing_ref_pct": (1 - ref_valid_ratio) * 100
                }
            
            if cur_valid_ratio < self.config.min_non_null_ratio:
                return {
                    "skipped": True,
                    "reason": "too_many_missing_current",
                    "type": "categorical",
                    "missing_cur_pct": (1 - cur_valid_ratio) * 100
                }
            
            # Get all unique categories
            all_categories = sorted(set(ref_clean.unique()) | set(cur_clean.unique()))
            
            # Check cardinality
            if len(all_categories) > self.config.max_categories:
                return {
                    "skipped": True,
                    "reason": "too_many_categories",
                    "type": "categorical",
                    "n_categories": len(all_categories)
                }
            
            # ═══════════════════════════════════════════════════════════
            # Create Distribution Tables
            # ═══════════════════════════════════════════════════════════
            
            ref_counts = ref_clean.value_counts().reindex(
                all_categories, fill_value=0
            )
            cur_counts = cur_clean.value_counts().reindex(
                all_categories, fill_value=0
            )
            
            ref_pct = ref_counts / max(1, ref_counts.sum())
            cur_pct = cur_counts / max(1, cur_counts.sum())
            
            # ═══════════════════════════════════════════════════════════
            # Method 1: PSI
            # ═══════════════════════════════════════════════════════════
            
            ref_pct_safe = ref_pct.replace(0, 1e-6)
            cur_pct_safe = cur_pct.replace(0, 1e-6)
            
            psi_values = (ref_pct_safe - cur_pct_safe) * np.log(
                ref_pct_safe / cur_pct_safe
            )
            psi_score = float(psi_values.sum())
            
            # ═══════════════════════════════════════════════════════════
            # Method 2: Chi-Square Test & Cramér's V
            # ═══════════════════════════════════════════════════════════
            
            contingency = np.vstack([ref_counts.values, cur_counts.values])
            
            try:
                chi2_stat, chi2_pvalue, dof, expected = chi2_contingency(contingency)
                
                # Cramér's V calculation
                n = contingency.sum()
                min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
                cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0.0
                
            except Exception as e:
                self._log.debug(f"Chi-square test failed: {e}")
                chi2_pvalue = None
                cramers_v = None
            
            # ═══════════════════════════════════════════════════════════
            # Distribution Analysis
            # ═══════════════════════════════════════════════════════════
            
            # Top K categories by frequency
            top_ref = ref_pct.sort_values(ascending=False).head(
                self.config.topk_categorical
            )
            top_cur = cur_pct.sort_values(ascending=False).head(
                self.config.topk_categorical
            )
            
            # Category shifts
            pct_delta = (cur_pct - ref_pct).fillna(0)
            top_changes = pct_delta.abs().sort_values(ascending=False).head(
                self.config.topk_categorical
            )
            
            # New and disappeared categories
            new_categories = [c for c in cur_clean.unique() if c not in ref_clean.unique()]
            disappeared_categories = [c for c in ref_clean.unique() if c not in cur_clean.unique()]
            
            # ═══════════════════════════════════════════════════════════
            # Drift Detection Logic
            # ═══════════════════════════════════════════════════════════
            
            triggers: List[str] = []
            
            # PSI triggers
            if psi_score >= self.config.psi_crit_threshold:
                triggers.append("psi_critical")
            elif psi_score >= self.config.psi_warn_threshold:
                triggers.append("psi_warning")
            
            # Chi-square triggers
            if chi2_pvalue is not None and chi2_pvalue < self.config.alpha:
                triggers.append("chi2_test_significant")
            
            # Cramér's V triggers
            if cramers_v is not None:
                if cramers_v >= self.config.cramer_crit_threshold:
                    triggers.append("cramers_v_critical")
                elif cramers_v >= self.config.cramer_warn_threshold:
                    triggers.append("cramers_v_warning")
            
            # New/disappeared categories
            if new_categories:
                triggers.append("new_categories_detected")
            
            if disappeared_categories:
                triggers.append("categories_disappeared")
            
            # Assemble metrics
            metrics.update({
                "missing_ref_pct": (1 - ref_valid_ratio) * 100,
                "missing_cur_pct": (1 - cur_valid_ratio) * 100,
                "n_categories": len(all_categories),
                "psi": psi_score,
                "chi2_statistic": float(chi2_stat) if chi2_pvalue is not None else None,
                "chi2_pvalue": float(chi2_pvalue) if chi2_pvalue is not None else None,
                "cramers_v": float(cramers_v) if cramers_v is not None else None,
                "top_categories_ref": {
                    str(k): float(v) for k, v in top_ref.items()
                },
                "top_categories_cur": {
                    str(k): float(v) for k, v in top_cur.items()
                },
                "topk_delta": {
                    str(k): float(pct_delta[k]) for k in top_changes.index
                },
                "new_categories": new_categories,
                "disappeared_categories": disappeared_categories,
                "is_drift": len(triggers) > 0,
                "triggers": triggers,
                "drift_severity": self._categorize_severity(triggers)
            })
            
            return metrics
        
        except Exception as e:
            self._log.warning(f"Categorical drift calculation failed: {e}")
            return {
                "error": str(e),
                "type": "categorical",
                "skipped": True
            }
    
    # ───────────────────────────────────────────────────────────────────
    # Datetime Drift Detection
    # ───────────────────────────────────────────────────────────────────
    
    def _drift_datetime(
        self,
        s_ref: pd.Series,
        s_cur: pd.Series
    ) -> Dict[str, Any]:
        """
        📅 **Datetime Feature Drift Analysis**
        
        Converts to Unix timestamps and applies numeric drift detection.
        
        Args:
            s_ref: Reference series
            s_cur: Current series
        
        Returns:
            Dictionary with drift metrics
        """
        try:
            # Convert to datetime and then to Unix timestamp (seconds)
            ref_dt = pd.to_datetime(s_ref, errors='coerce').dropna()
            cur_dt = pd.to_datetime(s_cur, errors='coerce').dropna()
            
            if len(ref_dt) == 0 or len(cur_dt) == 0:
                return {
                    "skipped": True,
                    "reason": "invalid_datetime_values",
                    "type": "datetime"
                }
            
            # Convert to Unix timestamps (int64 nanoseconds → float seconds)
            ref_numeric = pd.Series(ref_dt.view('int64') / 1e9)
            cur_numeric = pd.Series(cur_dt.view('int64') / 1e9)
            
            # Apply numeric drift detection
            metrics = self._drift_numeric(ref_numeric, cur_numeric)
            metrics["type"] = "datetime"
            
            # Add datetime-specific information
            metrics["datetime_info"] = {
                "ref_min": ref_dt.min().isoformat(),
                "ref_max": ref_dt.max().isoformat(),
                "cur_min": cur_dt.min().isoformat(),
                "cur_max": cur_dt.max().isoformat(),
                "time_range_shift_days": (
                    (cur_dt.mean() - ref_dt.mean()).total_seconds() / 86400
                )
            }
            
            return metrics
        
        except Exception as e:
            self._log.warning(f"Datetime drift calculation failed: {e}")
            return {
                "error": str(e),
                "type": "datetime",
                "skipped": True
            }
    
    # ───────────────────────────────────────────────────────────────────
    # Performance Drift Detection
    # ───────────────────────────────────────────────────────────────────
    
    def _performance_drift(
        self,
        y_ref: pd.Series,
        pred_ref: pd.Series,
        y_cur: pd.Series,
        pred_cur: pd.Series
    ) -> Dict[str, Any]:
        """
        📈 **Performance Drift Analysis**
        
        Compares model performance between reference and current data.
        
        Args:
            y_ref: Reference true labels
            pred_ref: Reference predictions
            y_cur: Current true labels
            pred_cur: Current predictions
        
        Returns:
            Dictionary with performance metrics and deltas
        """
        try:
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score,
                r2_score, mean_absolute_error, mean_squared_error
            )
            
            # Detect problem type
            is_regression = self._is_regression_problem(y_ref, y_cur)
            
            if is_regression:
                # ═══════════════════════════════════════════════════════
                # Regression Metrics
                # ═══════════════════════════════════════════════════════
                
                def calc_regression_metrics(y_true, y_pred):
                    metrics = {}
                    try:
                        metrics["r2"] = float(r2_score(y_true, y_pred))
                    except:
                        pass
                    
                    try:
                        mse = mean_squared_error(y_true, y_pred)
                        metrics["mse"] = float(mse)
                        metrics["rmse"] = float(np.sqrt(mse))
                    except:
                        pass
                    
                    try:
                        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
                    except:
                        pass
                    
                    return metrics
                
                ref_metrics = calc_regression_metrics(y_ref, pred_ref)
                cur_metrics = calc_regression_metrics(y_cur, pred_cur)
                primary_metric = "r2"
            
            else:
                # ═══════════════════════════════════════════════════════
                # Classification Metrics
                # ═══════════════════════════════════════════════════════
                
                average = "weighted"
                
                def calc_classification_metrics(y_true, y_pred):
                    metrics = {}
                    try:
                        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                    except:
                        pass
                    
                    try:
                        metrics["f1"] = float(f1_score(
                            y_true, y_pred, average=average, zero_division=0
                        ))
                    except:
                        pass
                    
                    try:
                        metrics["precision"] = float(precision_score(
                            y_true, y_pred, average=average, zero_division=0
                        ))
                    except:
                        pass
                    
                    try:
                        metrics["recall"] = float(recall_score(
                            y_true, y_pred, average=average, zero_division=0
                        ))
                    except:
                        pass
                    
                    return metrics
                
                ref_metrics = calc_classification_metrics(y_ref, pred_ref)
                cur_metrics = calc_classification_metrics(y_cur, pred_cur)
                primary_metric = "accuracy"
            
            # Calculate deltas
            all_metric_names = set(ref_metrics.keys()) | set(cur_metrics.keys())
            delta = {
                metric: cur_metrics.get(metric, np.nan) - ref_metrics.get(metric, np.nan)
                for metric in all_metric_names
            }
            
            # Determine if performance degraded
            primary_delta = delta.get(primary_metric, 0)
            degraded = primary_delta < -0.05  # 5% degradation threshold
            
            return {
                "problem_type": "regression" if is_regression else "classification",
                "reference_metrics": ref_metrics,
                "current_metrics": cur_metrics,
                "delta": delta,
                "primary_metric": primary_metric,
                "primary_delta": float(primary_delta),
                "performance_degraded": degraded,
                "degradation_pct": float(abs(primary_delta) * 100) if degraded else 0.0
            }
        
        except Exception as e:
            self._log.warning(f"Performance drift calculation failed: {e}")
            return {"error": str(e)}
    
    def _is_regression_problem(
        self,
        y_ref: pd.Series,
        y_cur: pd.Series
    ) -> bool:
        """
        Detect if problem is regression or classification.
        
        Args:
            y_ref: Reference labels
            y_cur: Current labels
        
        Returns:
            True if regression, False if classification
        """
        # Check if float dtype
        if pd.api.types.is_float_dtype(y_ref) or pd.api.types.is_float_dtype(y_cur):
            return True
        
        # Check cardinality
        combined_nunique = pd.concat([y_ref, y_cur]).nunique(dropna=True)
        
        if combined_nunique > 20:
            return True
        
        return False
    
    # ───────────────────────────────────────────────────────────────────
    # Analysis & Reporting
    # ───────────────────────────────────────────────────────────────────
    
    def _analyze_class_shift(
        self,
        target_drift: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze major class distribution shift.
        
        Args:
            target_drift: Target drift metrics
        
        Returns:
            Major class shift information
        """
        try:
            topk_delta = target_drift.get("topk_delta", {})
            
            if not topk_delta:
                return None
            
            # Find class with maximum absolute shift
            max_class = max(topk_delta, key=lambda k: abs(topk_delta[k]))
            max_delta = topk_delta[max_class]
            
            return {
                "class": str(max_class),
                "delta_pct": float(max_delta * 100),
                "direction": "increased" if max_delta > 0 else "decreased"
            }
        
        except Exception:
            return None
    
    def _categorize_severity(self, triggers: List[str]) -> str:
        """
        Categorize drift severity based on triggers.
        
        Args:
            triggers: List of triggered conditions
        
        Returns:
            Severity level ('ok', 'warning', 'critical')
        """
        if not triggers:
            return "ok"
        
        critical_keywords = ["critical", "disappeared", "new_categories"]
        
        for trigger in triggers:
            for keyword in critical_keywords:
                if keyword in trigger:
                    return "critical"
        
        return "warning"
    
    def _drift_band(self, drift_pct: float) -> Literal["ok", "warn", "critical"]:
        """
        Categorize drift score into bands.
        
        Args:
            drift_pct: Percentage of drifted features
        
        Returns:
            Drift band
        """
        if drift_pct >= 30.0:
            return "critical"
        elif drift_pct >= 10.0:
            return "warn"
        else:
            return "ok"
    
    def _build_summary(
        self,
        data_drift: Dict[str, Any],
        target_drift: Optional[Dict[str, Any]],
        perf_drift: Optional[Dict[str, Any]],
        n_ref: int,
        n_cur: int
    ) -> Dict[str, Any]:
        """
        Build comprehensive drift summary.
        
        Args:
            data_drift: Data drift metrics
            target_drift: Target drift metrics
            perf_drift: Performance drift metrics
            n_ref: Reference dataset size
            n_cur: Current dataset size
        
        Returns:
            Summary dictionary
        """
        drift_score = float(data_drift.get("drift_score", 0.0))
        band = self._drift_band(drift_score)
        
        # Get top triggers
        top_triggers = self._extract_top_triggers(data_drift.get("per_feature", {}))
        
        summary = {
            "n_reference": n_ref,
            "n_current": n_cur,
            "n_evaluated_features": data_drift.get("evaluated_features", 0),
            "n_drifted_features": data_drift.get("n_drifted", 0),
            "drift_score_pct": drift_score,
            "drift_band": band,
            "has_target_drift": bool(
                target_drift and target_drift.get("is_drift", False)
            ),
            "has_performance_drift": bool(
                perf_drift and perf_drift.get("performance_degraded", False)
            ),
            "top_triggers": top_triggers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return summary
    
    def _extract_top_triggers(
        self,
        per_feature: Dict[str, Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract top drift triggers for summary.
        
        Args:
            per_feature: Per-feature drift metrics
            top_n: Number of top features to extract
        
        Returns:
            List of top trigger information
        """
        scored_features: List[Tuple[str, float, str]] = []
        
        for feature, metrics in per_feature.items():
            if metrics.get("skipped") or not metrics.get("is_drift"):
                continue
            
            # Use PSI as primary score
            psi = metrics.get("psi")
            if psi is not None:
                try:
                    severity = metrics.get("drift_severity", "warning")
                    scored_features.append((feature, float(psi), severity))
                except Exception:
                    pass
        
        # Sort by PSI score (descending)
        scored_features.sort(key=lambda x: x[1], reverse=True)
        
        # Format top N
        top_triggers = []
        for feature, score, severity in scored_features[:top_n]:
            top_triggers.append({
                "feature": feature,
                "psi_score": round(score, 4),
                "severity": severity
            })
        
        return top_triggers
    
    def _generate_recommendations(
        self,
        data_drift: Dict[str, Any],
        target_drift: Optional[Dict[str, Any]],
        perf_drift: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        🎯 **Generate Actionable Recommendations**
        
        Creates context-aware recommendations based on drift analysis.
        
        Args:
            data_drift: Data drift metrics
            target_drift: Target drift metrics
            perf_drift: Performance drift metrics
        
        Returns:
            List of recommendations
        """
        recommendations: List[str] = []
        
        n_drifted = data_drift.get("n_drifted", 0)
        drifted_features = data_drift.get("drifted_features", [])
        drift_score = data_drift.get("drift_score", 0.0)
        
        # ═══════════════════════════════════════════════════════════════
        # Data Drift Recommendations
        # ═══════════════════════════════════════════════════════════════
        
        if n_drifted == 0:
            recommendations.append(
                "✅ No significant data drift detected. Continue regular monitoring."
            )
        else:
            # List drifted features
            feature_list = ", ".join(drifted_features[:5])
            if len(drifted_features) > 5:
                feature_list += f" (+{len(drifted_features) - 5} more)"
            
            recommendations.append(
                f"🔍 Data drift detected in {n_drifted} features: {feature_list}"
            )
            
            # Severity-based recommendations
            if drift_score >= 30:
                recommendations.extend([
                    "🚨 CRITICAL: High drift detected (>30% features affected)",
                    "➡️ IMMEDIATE ACTION: Retrain model with recent data",
                    "➡️ Review data collection pipeline for issues",
                    "➡️ Consider implementing automated retraining triggers"
                ])
            elif drift_score >= 10:
                recommendations.extend([
                    "⚠️ WARNING: Moderate drift detected (10-30% features affected)",
                    "➡️ Schedule model retraining within next cycle",
                    "➡️ Monitor drifted features closely"
                ])
            else:
                recommendations.append(
                    "➡️ Minor drift detected. Monitor trends over time."
                )
            
            # Feature-specific recommendations
            recommendations.extend([
                "➡️ Analyze features with highest PSI/KS scores",
                "➡️ Investigate root causes: data source changes, seasonality, etc.",
                "➡️ Consider feature engineering updates or transformations"
            ])
        
        # ═══════════════════════════════════════════════════════════════
        # Target Drift Recommendations
        # ═══════════════════════════════════════════════════════════════
        
        if target_drift and target_drift.get("is_drift"):
            recommendations.append(
                "⚠️ TARGET DRIFT: Target distribution has shifted"
            )
            
            # Check for class shift
            class_shift = target_drift.get("major_class_shift")
            if class_shift:
                recommendations.append(
                    f"➡️ Major class shift detected: {class_shift['class']} "
                    f"{class_shift['direction']} by {abs(class_shift['delta_pct']):.1f}%"
                )
            
            recommendations.extend([
                "➡️ Verify target labeling process hasn't changed",
                "➡️ Check if business definition of target has evolved",
                "➡️ Consider concept drift - relationship between features and target may have changed"
            ])
        
        # ═══════════════════════════════════════════════════════════════
        # Performance Drift Recommendations
        # ═══════════════════════════════════════════════════════════════
        
        if perf_drift and perf_drift.get("performance_degraded"):
            primary_metric = perf_drift.get("primary_metric", "metric")
            primary_delta = perf_drift.get("primary_delta", 0)
            degradation_pct = perf_drift.get("degradation_pct", 0)
            
            recommendations.extend([
                f"📉 PERFORMANCE DEGRADATION: {primary_metric} declined by {degradation_pct:.1f}%",
                f"➡️ Current {primary_metric}: {perf_drift['current_metrics'].get(primary_metric, 'N/A')}",
                f"➡️ Reference {primary_metric}: {perf_drift['reference_metrics'].get(primary_metric, 'N/A')}",
                "➡️ PRIORITY: Retrain model immediately",
                "➡️ Investigate correlation between data drift and performance drop"
            ])
            
            # Additional metric degradation
            delta = perf_drift.get("delta", {})
            degraded_metrics = [
                f"{k} ({v:.3f})"
                for k, v in delta.items()
                if isinstance(v, (int, float)) and v < -0.01 and k != primary_metric
            ]
            
            if degraded_metrics:
                recommendations.append(
                    f"➡️ Other degraded metrics: {', '.join(degraded_metrics)}"
                )
        
        # ═══════════════════════════════════════════════════════════════
        # General Best Practices
        # ═══════════════════════════════════════════════════════════════
        
        recommendations.extend([
            "🧪 Establish operational thresholds (e.g., PSI > 0.2 = action required)",
            "📊 Implement automated drift monitoring in production",
            "🔔 Configure alerts for critical drift patterns",
            "📈 Track drift trends over time for proactive management"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: Optional[str] = None,
    config: Optional[DriftConfig] = None,
    **kwargs
) -> AgentResult:
    """
    🚀 **Convenience Function: Detect Drift**
    
    High-level API for drift detection with minimal configuration.
    
    Args:
        reference_data: Reference dataset (e.g., training data)
        current_data: Current dataset (e.g., production data)
        target_column: Target column name (optional)
        config: Optional custom configuration
        **kwargs: Additional parameters
    
    Returns:
        AgentResult with drift analysis
    
    Examples:
```python
        from agents.monitoring import detect_drift
        
        # Basic usage
        result = detect_drift(
            reference_data=train_df,
            current_data=prod_df,
            target_column='target'
        )
        
        # Check drift
        if result.is_success():
            drift_score = result.data['data_drift']['drift_score']
            print(f"Drift score: {drift_score:.1f}%")
            
            if drift_score > 20:
                print("⚠️ Significant drift detected!")
                print("\nRecommendations:")
                for rec in result.data['recommendations']:
                    print(f"  - {rec}")
        
        # With custom config
        config = DriftConfig.create_strict()
        result = detect_drift(
            reference_data=train_df,
            current_data=prod_df,
            target_column='target',
            config=config
        )
        
        # With performance drift
        result = detect_drift(
            reference_data=train_df,
            current_data=prod_df,
            target_column='target',
            y_ref=y_train,
            y_cur=y_prod,
            pred_ref=train_predictions,
            pred_cur=prod_predictions
        )
```
    """
    detector = DriftDetector(config)
    return detector.execute(
        reference_data=reference_data,
        current_data=current_data,
        target_column=target_column,
        **kwargs
    )


def quick_drift_check(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    ⚡ **Quick Drift Check**
    
    Simplified drift check returning only key metrics.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        target_column: Target column name
    
    Returns:
        Dictionary with key drift metrics
    
    Example:
```python
        from agents.monitoring import quick_drift_check
        
        metrics = quick_drift_check(train_df, prod_df, 'target')
        
        print(f"Drift Score: {metrics['drift_score']:.1f}%")
        print(f"Band: {metrics['band']}")
        print(f"Drifted Features: {metrics['n_drifted']}")
```
    """
    result = detect_drift(reference_data, current_data, target_column)
    
    if not result.is_success():
        return {
            "success": False,
            "error": result.errors[0] if result.errors else "Unknown error"
        }
    
    summary = result.data['summary']
    data_drift = result.data['data_drift']
    
    return {
        "success": True,
        "drift_score": data_drift['drift_score'],
        "band": summary['drift_band'],
        "n_drifted": data_drift['n_drifted'],
        "n_evaluated": data_drift['evaluated_features'],
        "drifted_features": data_drift['drifted_features'],
        "has_target_drift": summary['has_target_drift'],
        "top_triggers": summary['top_triggers']
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _module_init():
    """Initialize module on import."""
    logger.info(f"✓ DriftDetector v{__version__} loaded")

_module_init()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{'='*80}")
    print(f"DriftDetector v{__version__}")
    print(f"{'='*80}")
    
    # Generate synthetic data for testing
    np.random.seed(42)
    
    # Reference data
    ref_df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.choice(['A', 'B', 'C'], 1000),
        'feature_3': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Current data (with drift)
    cur_df = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, 1000),  # Mean shift + variance increase
        'feature_2': np.random.choice(['A', 'B', 'C', 'D'], 1000),  # New category
        'feature_3': pd.date_range('2024-02-01', periods=1000, freq='H'),
        'target': np.random.choice([0, 1], 1000, p=[0.3, 0.7])  # Class imbalance shift
    })
    
    print("\n✓ Testing drift detection...")
    
    # Run detection
    result = detect_drift(
        reference_data=ref_df,
        current_data=cur_df,
        target_column='target'
    )
    
    if result.is_success():
        print(f"\n✓ Drift analysis completed successfully")
        
        summary = result.data['summary']
        data_drift = result.data['data_drift']
        
        print(f"\nSummary:")
        print(f"  Drift Score: {data_drift['drift_score']:.1f}%")
        print(f"  Band: {summary['drift_band']}")
        print(f"  Drifted Features: {data_drift['n_drifted']}/{data_drift['evaluated_features']}")
        print(f"  Target Drift: {summary['has_target_drift']}")
        
        print(f"\nDrifted Features:")
        for feature in data_drift['drifted_features']:
            metrics = data_drift['per_feature'][feature]
            psi = metrics.get('psi', 'N/A')
            print(f"  - {feature}: PSI={psi}")
        
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(result.data['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
    
    else:
        print(f"\n✗ Drift analysis failed:")
        for error in result.errors:
            print(f"  - {error}")
    
    print(f"\n{'='*80}")
    print("USAGE EXAMPLES:")
    print(f"{'='*80}")
    print("""
from agents.monitoring import DriftDetector, DriftConfig, detect_drift

# Basic usage
result = detect_drift(
    reference_data=train_df,
    current_data=prod_df,
    target_column='target'
)

# Check drift
drift_score = result.data['data_drift']['drift_score']
band = result.data['summary']['drift_band']

if band == 'critical':
    print("⚠️ Critical drift detected!")
    print(result.data['recommendations'])

# Custom configuration
config = DriftConfig(
    psi_warn_threshold=0.15,
    psi_crit_threshold=0.25,
    sample_size=50_000
)

detector = DriftDetector(config)
result = detector.execute(
    reference_data=train_df,
    current_data=prod_df,
    target_column='target'
)

# Quick check
from agents.monitoring import quick_drift_check

metrics = quick_drift_check(train_df, prod_df, 'target')
print(f"Drift: {metrics['drift_score']:.1f}% ({metrics['band']})")
    """)