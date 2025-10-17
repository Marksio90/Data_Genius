# agents/preprocessing/feature_engineer.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Feature Engineer v6.0            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ENTERPRISE-GRADE AUTOMATED FEATURE ENGINEERING                        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Datetime Feature Extraction (components + cyclical)                   ║
║  ✓ Time Delta Calculations (intelligent pairing)                         ║
║  ✓ Text Signal Generation (length, composition)                          ║
║  ✓ Numeric Transformations (log1p, sqrt, power)                          ║
║  ✓ Interaction Features (guided by importance)                           ║
║  ✓ Polynomial Features (degree 2/3)                                      ║
║  ✓ Robust Binning (qcut with fallbacks)                                  ║
║  ✓ Feature Importance Integration                                        ║
║  ✓ Defensive Validation & Sanitization                                   ║
║  ✓ Comprehensive Metadata & Telemetry                                    ║
║  ✓ Production-Ready with Safety Guards                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │              FeatureEngineer Core                           │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Datetime Engineering (components + cyclical + deltas)   │
    │  2. Text Signals (length, digits, composition)              │
    │  3. Numeric Transforms (log1p, sqrt, skew-based)            │
    │  4. Interaction Features (importance-guided)                │
    │  5. Polynomial Features (squared, cubed)                    │
    │  6. Robust Binning (qcut with fallback)                     │
    │  7. Feature Importance Calculation                          │
    │  8. Sanitization & Validation                               │
    └─────────────────────────────────────────────────────────────┘

Feature Types Generated:
    • Datetime Components → year, month, day, dayofweek, quarter, hour
    • Cyclical Datetime → sin/cos for periodic features
    • Time Deltas → differences between date pairs (days)
    • Text Signals → length, digit_count, space_count, letter_count
    • Text Ratios → digit_share, space_share, letter_share
    • Log Transform → log1p for skewed features (skew >= 1.0)
    • Sqrt Transform → sqrt for non-negative features
    • Interactions → product of important features
    • Polynomials → squared, cubed
    • Bins → quantile-based discretization

Dependencies:
    • Required: pandas, numpy, loguru
    • Optional: scikit-learn (for mutual_info_*)

Usage:
```python
    from agents.preprocessing import FeatureEngineer, FeatureConfig
    
    # Basic usage
    engineer = FeatureEngineer()
    result = engineer.execute(
        data=train_df,
        target_column='target',
        problem_type='classification'
    )
    
    df_engineered = result.data['engineered_data']
    features_created = result.data['features_created']
    
    # Custom configuration
    config = FeatureConfig(
        add_cyclical_dates=True,
        max_interactions=10,
        poly_degree=3
    )
    engineer = FeatureEngineer(config)
```
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

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
        "logs/feature_engineer_{time:YYYY-MM-DD}.log",
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

# Mutual Information (optional)
try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    _SKLEARN_MI_AVAILABLE = True
except ImportError:
    _SKLEARN_MI_AVAILABLE = False
    logger.warning("⚠ sklearn mutual_info not available - using correlation fallback")

# Date features configuration
try:
    from config.constants import DATE_FEATURES
except ImportError:
    DATE_FEATURES = ["date", "created", "updated", "timestamp"]

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["FeatureConfig", "FeatureEngineer", "engineer_features"]
__version__ = "6.0.0-enterprise"
__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=False)
class FeatureConfig:
    """
    🎯 **Feature Engineering Configuration**
    
    Complete configuration for automated feature engineering.
    
    Datetime Features:
        parse_object_dates: Auto-parse string dates (default: True)
        date_parse_dayfirst: Day-first parsing (None=auto)
        drop_original_dates: Drop original date columns (default: True)
        add_cyclical_dates: Add sin/cos for periodic features (default: True)
        date_cycle_cols: Columns to cyclicalize (default: month, dayofweek, hour)
        add_time_deltas: Add time differences (default: True)
        max_time_delta_pairs: Max delta pairs (default: 4)
        
    Text Features:
        add_text_signals: Generate text statistics (default: True)
        max_text_cols: Max text columns to process (default: 6)
        text_min_unique_ratio: Min unique ratio (default: 0.001)
        
    Numeric Transformations:
        enable_log1p_for_skewed: Log transform for skewed (default: True)
        skew_threshold: Skewness threshold (default: 1.0)
        enable_sqrt_for_nonneg: Sqrt transform (default: True)
        enable_power_transforms: Enable power transforms (default: False)
        
    Interaction Features:
        max_interactions: Max interaction features (default: 6)
        top_features_for_interactions: Top N features (default: 6)
        interaction_importance_min: Min importance (default: 0.0)
        interaction_method: Method ('product', 'division', 'both')
        
    Polynomial Features:
        poly_degree: Polynomial degree (0, 2, 3) (default: 2)
        poly_top_features: Top N features (default: 5)
        poly_include_bias: Include bias term (default: False)
        
    Binning:
        bin_top_features: Top N features to bin (default: 4)
        bin_q: Number of quantiles (default: 5)
        bin_strategy: Binning strategy ('quantile', 'uniform', 'kmeans')
        
    Safety & Limits:
        cap_infinite_to_nan: Replace inf with NaN (default: True)
        max_new_features: Global limit (default: 2000)
        validate_output: Validate engineered features (default: True)
        safe_suffix_sep: Suffix separator (default: '__')
        
    Metadata & Telemetry:
        keep_feature_metadata: Keep metadata (default: True)
        collect_telemetry: Collect timing info (default: True)
        track_feature_importance: Calculate importance (default: True)
    """
    
    # Datetime
    parse_object_dates: bool = True
    date_parse_dayfirst: Optional[bool] = None
    drop_original_dates: bool = True
    add_cyclical_dates: bool = True
    date_cycle_cols: Tuple[str, ...] = ("month", "dayofweek", "hour")
    add_time_deltas: bool = True
    max_time_delta_pairs: int = 4
    
    # Text
    add_text_signals: bool = True
    max_text_cols: int = 6
    text_min_unique_ratio: float = 0.001
    
    # Numeric transforms
    enable_log1p_for_skewed: bool = True
    skew_threshold: float = 1.0
    enable_sqrt_for_nonneg: bool = True
    enable_power_transforms: bool = False
    
    # Interactions
    max_interactions: int = 6
    top_features_for_interactions: int = 6
    interaction_importance_min: float = 0.0
    interaction_method: Literal["product", "division", "both"] = "product"
    
    # Polynomials
    poly_degree: Literal[0, 2, 3] = 2
    poly_top_features: int = 5
    poly_include_bias: bool = False
    
    # Binning
    bin_top_features: int = 4
    bin_q: int = 5
    bin_strategy: Literal["quantile", "uniform", "kmeans"] = "quantile"
    
    # Safety
    cap_infinite_to_nan: bool = True
    max_new_features: Optional[int] = 2000
    validate_output: bool = True
    safe_suffix_sep: str = "__"
    
    # Metadata
    keep_feature_metadata: bool = True
    collect_telemetry: bool = True
    track_feature_importance: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.poly_degree not in (0, 2, 3):
            raise ValueError(f"poly_degree must be 0, 2, or 3, got {self.poly_degree}")
        
        if self.bin_q < 2:
            raise ValueError(f"bin_q must be >= 2, got {self.bin_q}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def create_minimal(cls) -> 'FeatureConfig':
        """Create minimal configuration (fast, few features)."""
        return cls(
            add_cyclical_dates=False,
            add_time_deltas=False,
            max_interactions=0,
            poly_degree=0,
            bin_top_features=0
        )
    
    @classmethod
    def create_comprehensive(cls) -> 'FeatureConfig':
        """Create comprehensive configuration (many features)."""
        return cls(
            add_cyclical_dates=True,
            add_time_deltas=True,
            max_interactions=15,
            poly_degree=3,
            bin_top_features=10,
            max_new_features=5000
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Feature Engineer (Main Class)
# ═══════════════════════════════════════════════════════════════════════════

class FeatureEngineer(BaseAgent):
    """
    🚀 **FeatureEngineer PRO Master Enterprise ++++**
    
    Enterprise-grade automated feature engineering system.
    
    Capabilities:
      1. Datetime feature extraction & cyclical encoding
      2. Intelligent time delta calculation
      3. Text signal generation
      4. Numeric transformations (log1p, sqrt)
      5. Importance-guided interactions
      6. Polynomial features
      7. Robust binning
      8. Feature importance calculation
      9. Comprehensive metadata tracking
     10. Production-ready validation
    
    Feature Generation Pipeline:
```
        ┌──────────────────────────────────────────┐
        │ Input DataFrame                          │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Datetime Features                        │
        │  • Components (year, month, day)         │
        │  • Cyclical (sin/cos)                    │
        │  • Time deltas (days between dates)      │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Text Features                            │
        │  • Length, digits, spaces, letters       │
        │  • Composition ratios                    │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Numeric Transforms                       │
        │  • log1p (skewed features)               │
        │  • sqrt (non-negative)                   │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Interactions (importance-guided)         │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Polynomials (degree 2/3)                 │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Robust Binning (qcut/cut)                │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Validation & Sanitization                │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Output DataFrame + Metadata              │
        └──────────────────────────────────────────┘
```
    
    Usage:
```python
        # Basic usage
        engineer = FeatureEngineer()
        
        result = engineer.execute(
            data=train_df,
            target_column='target',
            problem_type='classification'
        )
        
        df_engineered = result.data['engineered_data']
        new_features = result.data['features_created']
        
        # Custom configuration
        config = FeatureConfig.create_comprehensive()
        engineer = FeatureEngineer(config)
```
    """
    
    version: str = __version__
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Optional custom configuration
        """
        super().__init__(
            name="FeatureEngineer",
            description="Enterprise automated feature engineering"
        )
        
        self.config = config or FeatureConfig()
        self._log = logger.bind(agent="FeatureEngineer", version=self.version)
        
        self._log.info(f"✓ FeatureEngineer v{self.version} initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: Optional[Literal["classification", "regression"]] = None,
        *,
        protect_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        🎯 **Execute Feature Engineering**
        
        Automated feature generation with intelligent strategies.
        
        Args:
            data: Input DataFrame
            target_column: Target column name
            problem_type: 'classification' or 'regression'
            protect_cols: Columns to protect (copy only)
            exclude_cols: Columns to exclude
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with engineered data and metadata
        """
        result = AgentResult(agent_name=self.name)
        t_start = time.perf_counter()
        telemetry: Dict[str, Any] = {
            "timing_s": {},
            "warnings": [],
            "counts": {}
        }
        
        try:
            self._log.info(
                f"🔧 Starting feature engineering | "
                f"rows={len(data):,} | "
                f"cols={len(data.columns)}"
            )
            
            # Validation
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be non-empty DataFrame")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: Setup
            # ═══════════════════════════════════════════════════════════
            
            df = data.copy()
            features_created: List[str] = []
            feature_metadata: List[Dict[str, Any]] = []
            
            protect_set: Set[str] = set(protect_cols or [])
            exclude_set: Set[str] = set(exclude_cols or [])
            
            max_features = self.config.max_new_features or 10**9
            
            # Extract target
            y = None
            if target_column and target_column in df.columns:
                y = df[target_column]
            
            # Infer problem type
            if problem_type is None and y is not None:
                problem_type = self._infer_problem_type(y)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: Datetime Features
            # ═══════════════════════════════════════════════════════════
            
            t = time.perf_counter()
            created = self._create_date_features(
                df,
                exclude=exclude_set | protect_set
            )
            features_created.extend([f["name"] for f in created])
            feature_metadata.extend(created)
            telemetry["timing_s"]["datetime"] = round(time.perf_counter() - t, 4)
            
            self._log.debug(f"Created {len(created)} datetime features")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: Text Features
            # ═══════════════════════════════════════════════════════════
            
            if self.config.add_text_signals and len(features_created) < max_features:
                t = time.perf_counter()
                created = self._create_text_signals(
                    df,
                    exclude=exclude_set | protect_set | ({target_column} if target_column else set())
                )
                features_created.extend([f["name"] for f in created])
                feature_metadata.extend(created)
                telemetry["timing_s"]["text"] = round(time.perf_counter() - t, 4)
                
                self._log.debug(f"Created {len(created)} text features")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 4: Numeric Transforms
            # ═══════════════════════════════════════════════════════════
            
            if len(features_created) < max_features:
                t = time.perf_counter()
                created = self._create_numeric_transforms(
                    df,
                    exclude={target_column} if target_column else set(),
                    hard_exclude=exclude_set | protect_set
                )
                features_created.extend([f["name"] for f in created])
                feature_metadata.extend(created)
                telemetry["timing_s"]["numeric"] = round(time.perf_counter() - t, 4)
                
                self._log.debug(f"Created {len(created)} numeric transform features")
            
            # Get numeric columns for advanced features
            num_cols = self._get_numeric_cols(
                df,
                exclude=exclude_set | protect_set | ({target_column} if target_column else set())
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 5: Interaction Features
            # ═══════════════════════════════════════════════════════════
            
            if len(features_created) < max_features and self.config.max_interactions > 0:
                t = time.perf_counter()
                created = self._create_interactions(
                    df,
                    y=y,
                    num_cols=num_cols,
                    problem_type=problem_type,
                    cap_left=max(0, max_features - len(features_created))
                )
                features_created.extend([f["name"] for f in created])
                feature_metadata.extend(created)
                telemetry["timing_s"]["interactions"] = round(time.perf_counter() - t, 4)
                
                self._log.debug(f"Created {len(created)} interaction features")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 6: Polynomial Features
            # ═══════════════════════════════════════════════════════════
            
            if len(features_created) < max_features and self.config.poly_degree in (2, 3):
                t = time.perf_counter()
                created = self._create_polynomials(
                    df,
                    y=y,
                    num_cols=num_cols,
                    problem_type=problem_type,
                    cap_left=max(0, max_features - len(features_created))
                )
                features_created.extend([f["name"] for f in created])
                feature_metadata.extend(created)
                telemetry["timing_s"]["polynomials"] = round(time.perf_counter() - t, 4)
                
                self._log.debug(f"Created {len(created)} polynomial features")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 7: Binning
            # ═══════════════════════════════════════════════════════════
            
            if len(features_created) < max_features and self.config.bin_top_features > 0:
                t = time.perf_counter()
                created = self._create_binned(
                    df,
                    num_cols=num_cols,
                    cap_left=max(0, max_features - len(features_created))
                )
                features_created.extend([f["name"] for f in created])
                feature_metadata.extend(created)
                telemetry["timing_s"]["binning"] = round(time.perf_counter() - t, 4)
                
                self._log.debug(f"Created {len(created)} binned features")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 8: Validation & Sanitization
            # ═══════════════════════════════════════════════════════════
            
            if self.config.cap_infinite_to_nan:
                t = time.perf_counter()
                self._sanitize_features(df, features_created)
                telemetry["timing_s"]["sanitization"] = round(time.perf_counter() - t, 4)
            
            if self.config.validate_output:
                validation_warnings = self._validate_features(df, features_created)
                telemetry["warnings"].extend(validation_warnings)
                
                for warning in validation_warnings:
                    result.add_warning(warning)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 9: Telemetry & Summary
            # ═══════════════════════════════════════════════════════════
            
            elapsed_s = time.perf_counter() - t_start
            
            telemetry["counts"] = {
                "n_features_created": len(features_created),
                "n_cols_input": data.shape[1],
                "n_cols_output": df.shape[1],
                "n_rows": len(df)
            }
            telemetry["timing_s"]["total"] = round(elapsed_s, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 10: Assemble Result
            # ═══════════════════════════════════════════════════════════
            
            result.data = {
                "engineered_data": df,
                "features_created": features_created,
                "n_new_features": len(features_created),
                "original_shape": tuple(data.shape),
                "new_shape": tuple(df.shape)
            }
            
            if self.config.keep_feature_metadata:
                result.data["feature_metadata"] = feature_metadata
            
            if self.config.collect_telemetry:
                result.data["telemetry"] = telemetry
            
            self._log.success(
                f"✓ Engineering complete | "
                f"created={len(features_created)} features | "
                f"time={elapsed_s:.2f}s"
            )
        
        except Exception as e:
            error_msg = f"Feature engineering failed: {type(e).__name__}: {str(e)}"
            result.add_error(error_msg)
            self._log.error(error_msg, exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Datetime Features
    # ───────────────────────────────────────────────────────────────────
    
    def _create_date_features(
        self,
        df: pd.DataFrame,
        exclude: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Create datetime-based features.
        
        Features generated:
          • Components: year, month, day, dayofweek, quarter, hour
          • Cyclical: sin/cos for periodic components
          • Time deltas: differences between date pairs
        
        Args:
            df: DataFrame (modified in place)
            exclude: Columns to exclude
        
        Returns:
            List of feature metadata
        """
        created: List[Dict[str, Any]] = []
        
        # Detect datetime columns
        date_cols = self._detect_datetime_columns(df, exclude)
        
        if not date_cols:
            return created
        
        sep = self.config.safe_suffix_sep
        
        # ═══════════════════════════════════════════════════════════════
        # Component Features
        # ═══════════════════════════════════════════════════════════════
        
        for col in date_cols:
            series = df[col]
            
            if series.isna().all():
                continue
            
            # Basic components
            component_map = {
                f"{col}{sep}year": series.dt.year,
                f"{col}{sep}month": series.dt.month,
                f"{col}{sep}day": series.dt.day,
                f"{col}{sep}dayofweek": series.dt.dayofweek,
                f"{col}{sep}quarter": series.dt.quarter,
                f"{col}{sep}is_weekend": (series.dt.dayofweek >= 5).astype("Int8")
            }
            
            # Hour (if available)
            if hasattr(series.dt, "hour"):
                component_map[f"{col}{sep}hour"] = series.dt.hour
            
            # Add components
            for name, values in component_map.items():
                df[name] = values
                created.append(self._make_metadata(name, "date_component", [col]))
            
            # ═══════════════════════════════════════════════════════════
            # Cyclical Features
            # ═══════════════════════════════════════════════════════════
            
            if self.config.add_cyclical_dates:
                cycles = []
                
                if "month" in self.config.date_cycle_cols:
                    cycles.append(("month", 12))
                
                if "dayofweek" in self.config.date_cycle_cols:
                    cycles.append(("dayofweek", 7))
                
                if "hour" in self.config.date_cycle_cols and f"{col}{sep}hour" in df:
                    cycles.append(("hour", 24))
                
                for component_name, period in cycles:
                    base_col = f"{col}{sep}{component_name}"
                    
                    if base_col not in df:
                        continue
                    
                    # Calculate angle
                    angle = 2 * np.pi * (df[base_col] % period) / period
                    
                    # Sin and cos
                    for trig_name, trig_func in [("sin", np.sin), ("cos", np.cos)]:
                        feat_name = f"{base_col}{sep}{trig_name}"
                        df[feat_name] = trig_func(angle)
                        created.append(self._make_metadata(
                            feat_name, "date_cyclical", [base_col]
                        ))
        
        # ═══════════════════════════════════════════════════════════════
        # Time Delta Features
        # ═══════════════════════════════════════════════════════════════
        
        if self.config.add_time_deltas and len(date_cols) >= 2:
            pairs = self._suggest_time_delta_pairs(date_cols)
            
            for col_a, col_b in pairs[:self.config.max_time_delta_pairs]:
                feat_name = f"{col_a}{sep}minus{sep}{col_b}{sep}days"
                
                try:
                    delta = (df[col_a] - df[col_b]).dt.total_seconds() / 86400.0
                    df[feat_name] = delta
                    created.append(self._make_metadata(
                        feat_name, "date_delta_days", [col_a, col_b]
                    ))
                except Exception as e:
                    self._log.debug(f"Time delta failed for {col_a}-{col_b}: {e}")
                    continue
        
        # ═══════════════════════════════════════════════════════════════
        # Drop Original Dates (optional)
        # ═══════════════════════════════════════════════════════════════
        
        if self.config.drop_original_dates:
            cols_to_drop = [c for c in date_cols if c not in exclude]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        return created
    
    def _detect_datetime_columns(
        self,
        df: pd.DataFrame,
        exclude: Set[str]
    ) -> List[str]:
        """Detect datetime columns including string parsing."""
        # Existing datetime columns
        date_cols = [
            col for col in df.select_dtypes(
                include=["datetime64[ns]", "datetime64[ns, UTC]"]
            ).columns
            if col not in exclude
        ]
        
        # Parse object columns
        if self.config.parse_object_dates:
            patterns = set(DATE_FEATURES)
            candidates = []
            
            for col in df.columns:
                if col in date_cols or col in exclude:
                    continue
                
                series = df[col]
                
                if not (pd.api.types.is_object_dtype(series) or 
                        pd.api.types.is_string_dtype(series)):
                    continue
                
                col_lower = str(col).lower()
                
                # Check name patterns
                if any(p.lower() in col_lower for p in patterns) or \
                   any(k in col_lower for k in ("date", "time", "ts", "timestamp", "dt")):
                    candidates.append(col)
            
            # Try parsing candidates
            for col in candidates:
                try:
                    parsed = pd.to_datetime(
                        df[col],
                        errors='coerce',
                        dayfirst=self.config.date_parse_dayfirst or False
                    )
                    
                    # Require at least 50% valid dates
                    if parsed.notna().sum() >= max(5, 0.5 * len(parsed)):
                        df[col] = parsed
                        date_cols.append(col)
                        self._log.debug(f"Parsed date column: {col}")
                
                except Exception as e:
                    self._log.debug(f"Failed to parse {col} as date: {e}")
                    continue
        
        return date_cols
    
    def _suggest_time_delta_pairs(
        self,
        date_cols: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Suggest intelligent date pairs for delta calculation.
        
        Heuristics:
          • end/updated/closed - start/created/open
          • Alphabetical ordering as fallback
        
        Args:
            date_cols: List of datetime column names
        
        Returns:
            List of (col_a, col_b) pairs
        """
        if len(date_cols) < 2:
            return []
        
        pairs: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        
        # Priority patterns
        end_patterns = ("end", "updated", "closed", "finish", "resolved", "completed")
        start_patterns = ("start", "created", "open", "begin", "received", "opened")
        
        # Semantic matching
        for col_a in date_cols:
            for col_b in date_cols:
                if col_a == col_b:
                    continue
                
                col_a_lower = col_a.lower()
                col_b_lower = col_b.lower()
                
                # Check if a is "end" and b is "start"
                if any(p in col_a_lower for p in end_patterns) and \
                   any(p in col_b_lower for p in start_patterns):
                    key = (col_a, col_b)
                    if key not in seen:
                        pairs.append(key)
                        seen.add(key)
        
        # Fallback: alphabetical consecutive pairs
        if not pairs:
            sorted_cols = sorted(date_cols)
            for i in range(1, len(sorted_cols)):
                key = (sorted_cols[i], sorted_cols[i-1])
                if key not in seen:
                    pairs.append(key)
                    seen.add(key)
        
        return pairs
    
    # ───────────────────────────────────────────────────────────────────
    # Text Features
    # ───────────────────────────────────────────────────────────────────
    
    def _create_text_signals(
        self,
        df: pd.DataFrame,
        exclude: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Create text-based statistical features.
        
        Features generated:
          • Length, digit count, space count, letter count
          • Composition ratios (digits/length, spaces/length, etc.)
        
        Args:
            df: DataFrame (modified in place)
            exclude: Columns to exclude
        
        Returns:
            List of feature metadata
        """
        created: List[Dict[str, Any]] = []
        
        # Find text columns
        text_cols = [
            col for col in df.columns
            if col not in exclude and
            (pd.api.types.is_object_dtype(df[col]) or 
             pd.api.types.is_string_dtype(df[col]))
        ]
        
        if not text_cols:
            return created
        
        # Limit processing
        text_cols = text_cols[:self.config.max_text_cols]
        sep = self.config.safe_suffix_sep
        
        for col in text_cols:
            series = df[col].astype(str).fillna("")
            
            # Skip if too uniform
            if series.isna().all():
                continue
            
            n_unique = series.nunique(dropna=True)
            if (n_unique / max(1, len(series))) < self.config.text_min_unique_ratio:
                continue
            
            # ═══════════════════════════════════════════════════════════
            # Basic Counts
            # ═══════════════════════════════════════════════════════════
            
            counts = {
                f"{col}{sep}len": series.str.len().astype("Int32"),
                f"{col}{sep}digits": series.str.count(r"\d").astype("Int32"),
                f"{col}{sep}spaces": series.str.count(r"\s").astype("Int32"),
                f"{col}{sep}letters": series.str.count(r"[A-Za-z]").astype("Int32")
            }
            
            for name, values in counts.items():
                df[name] = values
                created.append(self._make_metadata(name, "text_count", [col]))
            
            # ═══════════════════════════════════════════════════════════
            # Composition Ratios
            # ═══════════════════════════════════════════════════════════
            
            length = counts[f"{col}{sep}len"].replace(0, pd.NA).astype("Float32")
            
            ratios = {
                f"{col}{sep}digit_ratio": counts[f"{col}{sep}digits"] / length,
                f"{col}{sep}space_ratio": counts[f"{col}{sep}spaces"] / length,
                f"{col}{sep}letter_ratio": counts[f"{col}{sep}letters"] / length
            }
            
            for name, values in ratios.items():
                df[name] = values.astype("Float32")
                created.append(self._make_metadata(name, "text_ratio", [col]))
        
        return created
    
    # ───────────────────────────────────────────────────────────────────
    # Numeric Transformations
    # ───────────────────────────────────────────────────────────────────
    
    def _create_numeric_transforms(
        self,
        df: pd.DataFrame,
        exclude: Set[str],
        hard_exclude: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Create numeric transformations.
        
        Features generated:
          • log1p for skewed features (|skew| >= threshold)
          • sqrt for non-negative features
        
        Args:
            df: DataFrame (modified in place)
            exclude: Soft exclude (for skew calculation)
            hard_exclude: Hard exclude (never transform)
        
        Returns:
            List of feature metadata
        """
        created: List[Dict[str, Any]] = []
        
        # Get numeric columns
        num_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in (exclude | hard_exclude)
        ]
        
        if not num_cols:
            return created
        
        # Calculate skewness
        try:
            skew_series = df[num_cols].skew()
        except Exception:
            skew_series = pd.Series(dtype=float)
        
        sep = self.config.safe_suffix_sep
        
        for col in num_cols:
            series = pd.to_numeric(df[col], errors='coerce')
            col_min = series.min(skipna=True)
            
            # ═══════════════════════════════════════════════════════════
            # Log1p Transform (for skewed features)
            # ═══════════════════════════════════════════════════════════
            
            if self.config.enable_log1p_for_skewed:
                try:
                    skew = float(skew_series.get(col, 0.0))
                except Exception:
                    skew = 0.0
                
                if col_min >= 0 and abs(skew) >= self.config.skew_threshold:
                    feat_name = f"{col}{sep}log1p"
                    df[feat_name] = np.log1p(series.astype(float))
                    created.append(self._make_metadata(
                        feat_name, "numeric_log1p", [col]
                    ))
            
            # ═══════════════════════════════════════════════════════════
            # Sqrt Transform (for non-negative)
            # ═══════════════════════════════════════════════════════════
            
            if self.config.enable_sqrt_for_nonneg:
                if col_min >= 0:
                    feat_name = f"{col}{sep}sqrt"
                    df[feat_name] = np.sqrt(series.astype(float))
                    created.append(self._make_metadata(
                        feat_name, "numeric_sqrt", [col]
                    ))
        
        return created
    
    # ───────────────────────────────────────────────────────────────────
    # Interaction Features
    # ───────────────────────────────────────────────────────────────────
    
    def _create_interactions(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str],
        cap_left: int
    ) -> List[Dict[str, Any]]:
        """
        Create interaction features guided by importance.
        
        Args:
            df: DataFrame (modified in place)
            y: Target series
            num_cols: Numeric column names
            problem_type: Problem type
            cap_left: Remaining feature budget
        
        Returns:
            List of feature metadata
        """
        if self.config.max_interactions <= 0 or cap_left <= 0:
            return []
        
        created: List[Dict[str, Any]] = []
        
        # Calculate importance
        importances = self._calculate_feature_importance(
            df, y, num_cols, problem_type
        )
        
        if not importances:
            return created
        
        # Select top candidates
        candidates = [
            col for col, score in importances[:self.config.top_features_for_interactions]
            if score >= self.config.interaction_importance_min
        ]
        
        if len(candidates) < 2:
            return created
        
        sep = self.config.safe_suffix_sep
        count = 0
        max_count = min(self.config.max_interactions, cap_left)
        
        # Generate interactions
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if count >= max_count:
                    break
                
                col_a, col_b = candidates[i], candidates[j]
                
                # Product interaction
                if self.config.interaction_method in ("product", "both"):
                    feat_name = f"{col_a}{sep}x{sep}{col_b}"
                    try:
                        df[feat_name] = df[col_a].astype(float) * df[col_b].astype(float)
                        created.append(self._make_metadata(
                            feat_name, "interaction_product", [col_a, col_b]
                        ))
                        count += 1
                    except Exception:
                        pass
                
                # Division interaction
                if self.config.interaction_method in ("division", "both") and count < max_count:
                    feat_name = f"{col_a}{sep}div{sep}{col_b}"
                    try:
                        divisor = df[col_b].replace(0, np.nan).astype(float)
                        df[feat_name] = df[col_a].astype(float) / divisor
                        created.append(self._make_metadata(
                            feat_name, "interaction_division", [col_a, col_b]
                        ))
                        count += 1
                    except Exception:
                        pass
            
            if count >= max_count:
                break
        
        return created
    
    # ───────────────────────────────────────────────────────────────────
    # Polynomial Features
    # ───────────────────────────────────────────────────────────────────
    
    def _create_polynomials(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str],
        cap_left: int
    ) -> List[Dict[str, Any]]:
        """
        Create polynomial features.
        
        Args:
            df: DataFrame (modified in place)
            y: Target series
            num_cols: Numeric column names
            problem_type: Problem type
            cap_left: Remaining feature budget
        
        Returns:
            List of feature metadata
        """
        if self.config.poly_degree not in (2, 3) or cap_left <= 0:
            return []
        
        created: List[Dict[str, Any]] = []
        
        # Calculate importance
        importances = self._calculate_feature_importance(
            df, y, num_cols, problem_type
        )
        
        if not importances:
            return created
        
        # Select top features
        top_features = [
            col for col, _ in importances[:self.config.poly_top_features]
        ]
        
        sep = self.config.safe_suffix_sep
        
        for col in top_features:
            if len(created) >= cap_left:
                break
            
            try:
                series = df[col].astype(float)
                
                # Squared
                feat_name = f"{col}{sep}squared"
                df[feat_name] = series ** 2
                created.append(self._make_metadata(feat_name, "poly_2", [col]))
                
                # Cubed (if degree 3)
                if self.config.poly_degree == 3 and len(created) < cap_left:
                    feat_name = f"{col}{sep}cubed"
                    df[feat_name] = series ** 3
                    created.append(self._make_metadata(feat_name, "poly_3", [col]))
            
            except Exception as e:
                self._log.debug(f"Polynomial failed for {col}: {e}")
                continue
        
        return created
    
    # ───────────────────────────────────────────────────────────────────
    # Binning
    # ───────────────────────────────────────────────────────────────────
    
    def _create_binned(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        cap_left: int
    ) -> List[Dict[str, Any]]:
        """
        Create binned categorical features.
        
        Args:
            df: DataFrame (modified in place)
            num_cols: Numeric column names
            cap_left: Remaining feature budget
        
        Returns:
            List of feature metadata
        """
        if self.config.bin_top_features <= 0 or self.config.bin_q < 2 or cap_left <= 0:
            return []
        
        created: List[Dict[str, Any]] = []
        
        if not num_cols:
            return created
        
        # Select features by variance
        try:
            variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
            top_features = list(variances.index[:self.config.bin_top_features])
        except Exception:
            top_features = num_cols[:self.config.bin_top_features]
        
        # Bin labels
        labels = ["very_low", "low", "medium", "high", "very_high"]
        n_bins = self.config.bin_q
        bin_labels = labels[:n_bins]
        
        sep = self.config.safe_suffix_sep
        
        for col in top_features:
            if len(created) >= cap_left:
                break
            
            feat_name = f"{col}{sep}binned"
            
            # Try qcut (quantile-based)
            try:
                df[feat_name] = pd.qcut(
                    df[col],
                    q=n_bins,
                    labels=bin_labels,
                    duplicates='drop'
                )
                created.append(self._make_metadata(feat_name, "bin_qcut", [col]))
                continue
            except Exception:
                pass
            
            # Fallback to cut (uniform)
            try:
                df[feat_name] = pd.cut(
                    df[col],
                    bins=n_bins,
                    labels=bin_labels
                )
                created.append(self._make_metadata(feat_name, "bin_cut", [col]))
            except Exception as e:
                self._log.debug(f"Binning failed for {col}: {e}")
                continue
        
        return created
    
    # ───────────────────────────────────────────────────────────────────
    # Feature Importance
    # ───────────────────────────────────────────────────────────────────
    
    def _calculate_feature_importance(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series],
        num_cols: List[str],
        problem_type: Optional[str]
    ) -> List[Tuple[str, float]]:
        """
        Calculate feature importance scores.
        
        Methods:
          • With target: Mutual Information or correlation
          • Without target: Variance
        
        Args:
            df: DataFrame
            y: Target series
            num_cols: Numeric column names
            problem_type: Problem type
        
        Returns:
            List of (column, importance_score) sorted by importance
        """
        if not num_cols:
            return []
        
        # Without target: use variance
        if y is None or y.isna().all():
            try:
                variances = df[num_cols].var(numeric_only=True)
                return sorted(
                    [(col, float(variances[col])) for col in num_cols],
                    key=lambda x: x[1],
                    reverse=True
                )
            except Exception:
                return [(col, 0.0) for col in num_cols]
        
        # With target: use MI or correlation
        try:
            is_classification = (
                problem_type == "classification" or
                (problem_type is None and not pd.api.types.is_numeric_dtype(y))
            )
            
            if is_classification:
                # Classification: MI or correlation with encoded target
                y_encoded = y.astype('category').cat.codes
                
                if _SKLEARN_MI_AVAILABLE:
                    # Mutual Information
                    X = df[num_cols].copy()
                    for col in num_cols:
                        X[col] = X[col].fillna(X[col].median())
                    
                    mi_scores = mutual_info_classif(X.values, y_encoded.values)
                    return sorted(
                        list(zip(num_cols, [float(s) for s in mi_scores])),
                        key=lambda x: x[1],
                        reverse=True
                    )
                else:
                    # Correlation fallback
                    scores = []
                    for col in num_cols:
                        series = df[col].fillna(df[col].median())
                        if series.nunique() < 2:
                            scores.append(0.0)
                        else:
                            corr = np.corrcoef(series.values, y_encoded.values)[0, 1]
                            scores.append(abs(corr) if np.isfinite(corr) else 0.0)
                    
                    return sorted(
                        list(zip(num_cols, scores)),
                        key=lambda x: x[1],
                        reverse=True
                    )
            
            else:
                # Regression: MI or correlation
                if _SKLEARN_MI_AVAILABLE:
                    X = df[num_cols].copy()
                    for col in num_cols:
                        X[col] = X[col].fillna(X[col].median())
                    
                    mi_scores = mutual_info_regression(X.values, y.astype(float).values)
                    return sorted(
                        list(zip(num_cols, [float(s) for s in mi_scores])),
                        key=lambda x: x[1],
                        reverse=True
                    )
                else:
                    # Correlation
                    scores = []
                    for col in num_cols:
                        series = df[col]
                        corr = series.corr(y) if series.notna().sum() > 1 else 0.0
                        scores.append(abs(corr) if np.isfinite(corr) else 0.0)
                    
                    return sorted(
                        list(zip(num_cols, scores)),
                        key=lambda x: x[1],
                        reverse=True
                    )
        
        except Exception as e:
            self._log.debug(f"Importance calculation failed: {e}")
            return [(col, 0.0) for col in num_cols]
    
    # ───────────────────────────────────────────────────────────────────
    # Validation & Sanitization
    # ───────────────────────────────────────────────────────────────────
    
    def _sanitize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> None:
        """Replace infinite values with NaN."""
        for col in feature_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    def _validate_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> List[str]:
        """Validate engineered features."""
        warnings: List[str] = []
        
        for col in feature_cols:
            if col not in df.columns:
                warnings.append(f"Feature '{col}' not found in DataFrame")
                continue
            
            # Check for all-NaN
            if df[col].isna().all():
                warnings.append(f"Feature '{col}' is all-NaN")
            
            # Check for constant
            if df[col].nunique(dropna=True) <= 1:
                warnings.append(f"Feature '{col}' is constant")
        
        return warnings
    
    # ───────────────────────────────────────────────────────────────────
    # Utility Methods
    # ───────────────────────────────────────────────────────────────────
    
    def _get_numeric_cols(
        self,
        df: pd.DataFrame,
        exclude: Set[str]
    ) -> List[str]:
        """Get numeric column names."""
        return [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude
        ]
    
    def _infer_problem_type(
        self,
        y: pd.Series
    ) -> Literal["classification", "regression"]:
        """Infer problem type from target."""
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15:
            return "regression"
        return "classification"
    
    def _make_metadata(
        self,
        name: str,
        feature_type: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """Create feature metadata dictionary."""
        return {
            "name": name,
            "type": feature_type,
            "sources": sources
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    config: Optional[FeatureConfig] = None
) -> pd.DataFrame:
    """
    🚀 **Convenience Function: Engineer Features**
    
    Quick feature engineering with automatic configuration.
    
    Args:
        data: Input DataFrame
        target_column: Target column name
        config: Optional custom configuration
    
    Returns:
        Engineered DataFrame
    
    Example:
```python
        from agents.preprocessing import engineer_features
        
        df_engineered = engineer_features(
            train_df,
            target_column='target'
        )
```
    """
    engineer = FeatureEngineer(config)
    result = engineer.execute(data=data, target_column=target_column)
    
    if not result.is_success():
        raise RuntimeError(f"Feature engineering failed: {result.errors}")
    
    return result.data["engineered_data"]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _module_init():
    """Initialize module on import."""
    logger.info(f"✓ FeatureEngineer v{__version__} loaded")

_module_init()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{'='*80}")
    print(f"FeatureEngineer v{__version__}")
    print(f"{'='*80}")
    
    # Generate synthetic data
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date_created': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'date_updated': pd.date_range('2024-01-02', periods=1000, freq='H'),
        'text_col': ['Sample text ' * np.random.randint(1, 10) for _ in range(1000)],
        'num_skewed': np.random.exponential(2, 1000),
        'num_normal': np.random.randn(1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    print("\n✓ Testing feature engineering...")
    
    engineer = FeatureEngineer()
    result = engineer.execute(
        data=df,
        target_column='target',
        problem_type='classification'
    )
    
    if result.is_success():
        print(f"\n✓ Engineering completed successfully")
        
        n_new = result.data['n_new_features']
        original_shape = result.data['original_shape']
        new_shape = result.data['new_shape']
        
        print(f"\nSummary:")
        print(f"  Original shape: {original_shape}")
        print(f"  New shape: {new_shape}")
        print(f"  Features created: {n_new}")
        
        if result.data.get('telemetry'):
            timing = result.data['telemetry']['timing_s']
            print(f"\nTiming:")
            for stage, seconds in timing.items():
                print(f"  {stage}: {seconds:.4f}s")
        
        print(f"\nSample new features:")
        for feat in result.data['features_created'][:10]:
            print(f"  • {feat}")
    
    else:
        print(f"\n✗ Engineering failed:")
        for error in result.errors:
            print(f"  - {error}")
    
    print(f"\n{'='*80}")
    print("USAGE EXAMPLES:")
    print(f"{'='*80}")
    print("""
from agents.preprocessing import FeatureEngineer, FeatureConfig

# Basic usage
engineer = FeatureEngineer()

result = engineer.execute(
    data=train_df,
    target_column='target',
    problem_type='classification'
)

df_engineered = result.data['engineered_data']
new_features = result.data['features_created']

# Custom configuration
config = FeatureConfig(
    add_cyclical_dates=True,
    max_interactions=10,
    poly_degree=3
)

engineer = FeatureEngineer(config)

# Convenience function
from agents.preprocessing import engineer_features

df_eng = engineer_features(train_df, target_column='target')
    """)