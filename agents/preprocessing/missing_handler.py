# agents/preprocessing/missing_data_handler.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Missing Data Handler v6.0       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ENTERPRISE-GRADE INTELLIGENT MISSING DATA IMPUTATION                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Multi-Strategy Imputation (median, mean, mode, KNN, constant)         ║
║  ✓ Type-Aware Processing (numeric, categorical, datetime)                ║
║  ✓ Missing Value Indicators (automatic feature generation)               ║
║  ✓ Group-wise Imputation (per-category statistics)                       ║
║  ✓ Extreme Missing Handling (column/row dropping)                        ║
║  ✓ KNN with Automatic Scaling                                            ║
║  ✓ Datetime Imputation (multiple strategies)                             ║
║  ✓ Deterministic Apply to New Data                                       ║
║  ✓ Comprehensive Reporting & Telemetry                                   ║
║  ✓ Production-Ready with Safety Guards                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │           MissingDataHandler Core                           │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Missing Data Analysis & Reporting                       │
    │  2. Extreme Missing Detection (drop columns/rows)           │
    │  3. Missing Indicators Generation                           │
    │  4. Type-Specific Imputation                                │
    │     • Numeric: median, mean, KNN, constant                  │
    │     • Categorical: mode, constant, drop                     │
    │     • Datetime: median, mode, forward/backward fill         │
    │  5. Group-wise Statistics (optional)                        │
    │  6. Fitted Imputer Storage                                  │
    │  7. Deterministic Apply to New Data                         │
    └─────────────────────────────────────────────────────────────┘

Imputation Strategies:
    Numeric:
      • auto     → median (robust default)
      • median   → robust to outliers
      • mean     → faster, sensitive to outliers
      • KNN      → multivariate, scaled automatically
      • constant → fill with specific value
    
    Categorical:
      • auto           → most_frequent
      • most_frequent  → mode
      • constant       → fill with token (e.g., '<MISSING>')
      • drop           → drop rows with missing
    
    Datetime:
      • auto          → median
      • median        → middle timestamp
      • most_frequent → mode
      • forward_fill  → propagate forward
      • backward_fill → propagate backward
      • constant      → specific timestamp

Dependencies:
    • Required: pandas, numpy, scikit-learn, loguru

Usage:
```python
    from agents.preprocessing import MissingDataHandler, MissingHandlerConfig
    
    # Basic usage
    handler = MissingDataHandler()
    result = handler.execute(
        data=train_df,
        target_column='target'
    )
    
    df_imputed = result.data['data']
    fitted = result.data['fitted']
    
    # Apply to new data
    test_imputed = MissingDataHandler.apply_to_new(test_df, fitted)
    
    # Custom configuration
    config = MissingHandlerConfig(
        strategy_numeric='knn',
        add_numeric_missing_indicators=True,
        enable_groupwise_imputation=True
    )
    handler = MissingDataHandler(config)
```
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

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
        "logs/missing_data_handler_{time:YYYY-MM-DD}.log",
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
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["MissingHandlerConfig", "MissingDataHandler", "impute_missing"]
__version__ = "6.0.0-enterprise"
__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=False)
class MissingHandlerConfig:
    """
    🎯 **Missing Data Handler Configuration**
    
    Complete configuration for intelligent missing data imputation.
    
    Column Dropping:
        drop_column_missing_pct_threshold: Drop columns with >X% missing (default: 0.80)
        
    Numeric Strategy:
        strategy_numeric: 'auto', 'median', 'mean', 'constant', 'knn' (default: 'auto')
        constant_numeric_fill_value: Value for constant strategy (default: 0.0)
        knn_neighbors: Number of KNN neighbors (default: 5)
        knn_max_features: Max features for KNN (default: 100)
        
    Categorical Strategy:
        strategy_categorical: 'auto', 'most_frequent', 'constant', 'drop' (default: 'auto')
        constant_categorical_fill_value: Token for constant (default: '<MISSING>')
        
    Datetime Strategy:
        strategy_datetime: 'auto', 'median', 'most_frequent', 'forward_fill', 
                          'backward_fill', 'constant' (default: 'auto')
        constant_datetime_fill_value: Timestamp for constant (default: None)
        
    Missing Indicators:
        add_numeric_missing_indicators: Add binary indicators (default: True)
        add_categorical_missing_indicators: Add binary indicators (default: False)
        add_datetime_missing_indicators: Add binary indicators (default: True)
        indicator_suffix: Suffix for indicator columns (default: '__ismissing')
        
    Target Handling:
        drop_rows_if_target_missing: Drop rows with missing target (default: True)
        
    Group-wise Imputation:
        enable_groupwise_imputation: Enable per-group statistics (default: False)
        group_cols: Grouping columns (default: None)
        group_min_size: Minimum group size (default: 3)
        
    Behavior:
        preserve_column_order: Maintain original column order (default: True)
        report_top_n: Number of columns in summary (default: 10)
        random_state: Random seed for reproducibility (default: 42)
    """
    
    # Column dropping
    drop_column_missing_pct_threshold: float = 0.80
    
    # Numeric strategy
    strategy_numeric: Literal["auto", "median", "mean", "constant", "knn"] = "auto"
    constant_numeric_fill_value: float = 0.0
    knn_neighbors: int = 5
    knn_max_features: Optional[int] = 100
    
    # Categorical strategy
    strategy_categorical: Literal["auto", "most_frequent", "constant", "drop"] = "auto"
    constant_categorical_fill_value: str = "<MISSING>"
    
    # Datetime strategy
    strategy_datetime: Literal[
        "auto", "median", "most_frequent", 
        "forward_fill", "backward_fill", "constant"
    ] = "auto"
    constant_datetime_fill_value: Optional[pd.Timestamp] = None
    
    # Missing indicators
    add_numeric_missing_indicators: bool = True
    add_categorical_missing_indicators: bool = False
    add_datetime_missing_indicators: bool = True
    indicator_suffix: str = "__ismissing"
    
    # Target handling
    drop_rows_if_target_missing: bool = True
    
    # Group-wise imputation
    enable_groupwise_imputation: bool = False
    group_cols: Optional[List[str]] = None
    group_min_size: int = 3
    
    # Behavior
    preserve_column_order: bool = True
    report_top_n: int = 10
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.drop_column_missing_pct_threshold <= 1:
            raise ValueError(
                f"drop_column_missing_pct_threshold must be in (0, 1], "
                f"got {self.drop_column_missing_pct_threshold}"
            )
        
        if self.knn_neighbors < 1:
            raise ValueError(f"knn_neighbors must be >= 1, got {self.knn_neighbors}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def create_simple(cls) -> 'MissingHandlerConfig':
        """Create simple configuration (fast, no KNN)."""
        return cls(
            strategy_numeric='median',
            strategy_categorical='most_frequent',
            add_numeric_missing_indicators=False,
            enable_groupwise_imputation=False
        )
    
    @classmethod
    def create_advanced(cls) -> 'MissingHandlerConfig':
        """Create advanced configuration (KNN, indicators, group-wise)."""
        return cls(
            strategy_numeric='knn',
            add_numeric_missing_indicators=True,
            add_categorical_missing_indicators=True,
            add_datetime_missing_indicators=True,
            enable_groupwise_imputation=True
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Missing Data Handler (Main Class)
# ═══════════════════════════════════════════════════════════════════════════

class MissingDataHandler(BaseAgent):
    """
    🚀 **MissingDataHandler PRO Master Enterprise ++++**
    
    Enterprise-grade intelligent missing data imputation system.
    
    Capabilities:
      1. Comprehensive missing data analysis
      2. Extreme missing detection & handling
      3. Type-aware imputation (numeric, categorical, datetime)
      4. Multiple imputation strategies
      5. KNN imputation with automatic scaling
      6. Missing value indicators
      7. Group-wise statistics
      8. Deterministic transformation
      9. Fitted imputer reuse
     10. Production-ready validation
    
    Imputation Flow:
```
        ┌──────────────────────────────────────────┐
        │ Input DataFrame                          │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Missing Data Analysis                    │
        │  • Per-column missing %                  │
        │  • Type detection                        │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Drop Extreme Missing Columns             │
        │  (>80% missing by default)               │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Drop Rows (target missing)               │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Generate Missing Indicators              │
        │  (before imputation)                     │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Type-Specific Imputation                 │
        │  • Numeric (median/mean/KNN/constant)    │
        │  • Categorical (mode/constant/drop)      │
        │  • Datetime (median/mode/ffill/bfill)    │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Store Fitted Imputers                    │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Output: Imputed DataFrame + Metadata     │
        └──────────────────────────────────────────┘
```
    
    Usage:
```python
        # Basic usage
        handler = MissingDataHandler()
        
        result = handler.execute(
            data=train_df,
            target_column='target'
        )
        
        df_imputed = result.data['data']
        fitted = result.data['fitted']
        report = result.data['imputation_report']
        
        # Apply to new data
        test_imputed = MissingDataHandler.apply_to_new(test_df, fitted)
        
        # Custom configuration
        config = MissingHandlerConfig.create_advanced()
        handler = MissingDataHandler(config)
```
    """
    
    version: str = __version__
    
    def __init__(self, config: Optional[MissingHandlerConfig] = None):
        """
        Initialize missing data handler.
        
        Args:
            config: Optional custom configuration
        """
        super().__init__(
            name="MissingDataHandler",
            description="Enterprise intelligent missing data imputation"
        )
        
        self.config = config or MissingHandlerConfig()
        self._log = logger.bind(agent="MissingDataHandler", version=self.version)
        
        self._log.info(f"✓ MissingDataHandler v{self.version} initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        strategy: Optional[str] = None,
        *,
        datetime_cols: Optional[List[str]] = None,
        use_groupwise: Optional[bool] = None,
        group_cols: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        🎯 **Execute Missing Data Imputation**
        
        Intelligently handle missing values with multiple strategies.
        
        Args:
            data: Input DataFrame (features + target)
            target_column: Target column name
            strategy: Legacy strategy override
            datetime_cols: Force datetime parsing for columns
            use_groupwise: Enable group-wise imputation
            group_cols: Grouping columns for group-wise
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with imputed data and fitted imputers
        """
        result = AgentResult(agent_name=self.name)
        t_start = time.perf_counter()
        telemetry: Dict[str, Any] = {
            "timing_s": {},
            "counts": {},
            "notes": []
        }
        
        try:
            self._log.info(
                f"🔧 Starting missing data handling | "
                f"rows={len(data):,} | "
                f"cols={len(data.columns)}"
            )
            
            # Validation
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be non-empty DataFrame")
            
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: Preparation
            # ═══════════════════════════════════════════════════════════
            
            df = data.copy()
            original_order = list(df.columns)
            
            # Force datetime parsing
            if datetime_cols:
                for col in datetime_cols:
                    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: Missing Data Analysis
            # ═══════════════════════════════════════════════════════════
            
            t = time.perf_counter()
            report = self._analyze_missing(df)
            telemetry["timing_s"]["analysis"] = round(time.perf_counter() - t, 4)
            
            self._log_top_missing(report, self.config.report_top_n)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: Drop Extreme Missing Columns
            # ═══════════════════════════════════════════════════════════
            
            cols_to_drop = self._get_columns_to_drop(
                report,
                self.config.drop_column_missing_pct_threshold
            )
            
            if cols_to_drop:
                self._log.warning(
                    f"Dropping {len(cols_to_drop)} columns with >"
                    f"{int(self.config.drop_column_missing_pct_threshold*100)}% missing"
                )
                df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 4: Handle Target Missing
            # ═══════════════════════════════════════════════════════════
            
            rows_dropped_target = 0
            if self.config.drop_rows_if_target_missing:
                if df[target_column].isna().any():
                    rows_dropped_target = int(df[target_column].isna().sum())
                    df = df[~df[target_column].isna()].copy()
                    self._log.warning(
                        f"Dropped {rows_dropped_target} rows with missing target"
                    )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 5: Split X and y
            # ═══════════════════════════════════════════════════════════
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 6: Add Missing Indicators (before imputation)
            # ═══════════════════════════════════════════════════════════
            
            t = time.perf_counter()
            ind_num, ind_cat, ind_dt = self._add_missing_indicators(X)
            telemetry["timing_s"]["indicators"] = round(time.perf_counter() - t, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 7: Resolve Strategies
            # ═══════════════════════════════════════════════════════════
            
            num_strategy, cat_strategy = self._resolve_strategies(strategy)
            dt_strategy = self._resolve_datetime_strategy()
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 8: Group-wise Statistics (optional)
            # ═══════════════════════════════════════════════════════════
            
            use_grp = use_groupwise if use_groupwise is not None else \
                      self.config.enable_groupwise_imputation
            grp_cols = group_cols or self.config.group_cols or []
            
            group_stats = None
            if use_grp and grp_cols:
                t = time.perf_counter()
                group_stats = self._compute_group_stats(X, grp_cols)
                telemetry["timing_s"]["group_stats"] = round(time.perf_counter() - t, 4)
                telemetry["notes"].append(f"Group-wise enabled: {grp_cols}")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 9: Numeric Imputation
            # ═══════════════════════════════════════════════════════════
            
            t = time.perf_counter()
            X, num_imputer, num_cols = self._impute_numeric(
                X,
                num_strategy,
                group_stats=group_stats,
                grp_cols=grp_cols
            )
            telemetry["timing_s"]["numeric"] = round(time.perf_counter() - t, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 10: Categorical Imputation
            # ═══════════════════════════════════════════════════════════
            
            t = time.perf_counter()
            rows_dropped_cat = 0
            
            if cat_strategy == 'drop':
                # Drop rows with missing categorical
                cat_cols_all = list(X.select_dtypes(
                    include=["object", "category"]
                ).columns)
                
                if cat_cols_all:
                    mask = X[cat_cols_all].isna().any(axis=1)
                    rows_dropped_cat = int(mask.sum())
                    
                    if rows_dropped_cat > 0:
                        X = X[~mask].copy()
                        y = y.loc[X.index].copy()
                        self._log.warning(
                            f"Dropped {rows_dropped_cat} rows with missing categorical"
                        )
            
            X, cat_imputer, cat_cols = self._impute_categorical(
                X,
                cat_strategy,
                group_stats=group_stats,
                grp_cols=grp_cols
            )
            telemetry["timing_s"]["categorical"] = round(time.perf_counter() - t, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 11: Datetime Imputation
            # ═══════════════════════════════════════════════════════════
            
            t = time.perf_counter()
            X, dt_imputer, dt_cols = self._impute_datetime(
                X,
                dt_strategy,
                constant_value=self.config.constant_datetime_fill_value
            )
            telemetry["timing_s"]["datetime"] = round(time.perf_counter() - t, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 12: Reconstruct DataFrame
            # ═══════════════════════════════════════════════════════════
            
            X[target_column] = y.values
            
            # Preserve column order
            if self.config.preserve_column_order:
                new_cols = [c for c in original_order if c in X.columns] + \
                          [c for c in X.columns if c not in original_order]
                X = X[new_cols]
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 13: Build Output
            # ═══════════════════════════════════════════════════════════
            
            # Imputation log
            imputation_log = self._build_imputation_log(
                cols_to_drop=cols_to_drop,
                rows_dropped_target=rows_dropped_target,
                rows_dropped_cat=rows_dropped_cat,
                num_strategy=num_strategy,
                num_cols=num_cols,
                cat_strategy=cat_strategy,
                cat_cols=cat_cols,
                dt_strategy=dt_strategy,
                dt_cols=dt_cols,
                ind_num=ind_num,
                ind_cat=ind_cat,
                ind_dt=ind_dt
            )
            
            # Fitted imputers
            fitted = self._build_fitted_dict(
                num_strategy=num_strategy,
                num_imputer=num_imputer,
                num_cols=num_cols,
                cat_strategy=cat_strategy,
                cat_imputer=cat_imputer,
                cat_cols=cat_cols,
                dt_strategy=dt_strategy,
                dt_imputer=dt_imputer,
                dt_cols=dt_cols,
                ind_num=ind_num,
                ind_cat=ind_cat,
                ind_dt=ind_dt,
                cols_to_drop=cols_to_drop,
                target_column=target_column
            )
            
            # Telemetry
            elapsed_s = time.perf_counter() - t_start
            
            telemetry["counts"] = {
                "dropped_cols": len(cols_to_drop),
                "imputed_numeric": len(num_cols),
                "imputed_categorical": len(cat_cols),
                "imputed_datetime": len(dt_cols),
                "rows_dropped_target": rows_dropped_target,
                "rows_dropped_categorical": rows_dropped_cat
            }
            telemetry["timing_s"]["total"] = round(elapsed_s, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 14: Assemble Result
            # ═══════════════════════════════════════════════════════════
            
            result.data = {
                "data": X,
                "fitted": fitted,
                "imputation_report": report,
                "imputation_log": imputation_log,
                "dropped": {
                    "columns": cols_to_drop,
                    "rows_target": rows_dropped_target,
                    "rows_categorical": rows_dropped_cat
                },
                "shapes": {
                    "original": tuple(data.shape),
                    "final": tuple(X.shape)
                },
                "telemetry": telemetry
            }
            
            self._log.success(
                f"✓ Imputation complete | "
                f"dropped_cols={len(cols_to_drop)} | "
                f"numeric={len(num_cols)} | "
                f"categorical={len(cat_cols)} | "
                f"datetime={len(dt_cols)} | "
                f"time={elapsed_s:.2f}s"
            )
        
        except Exception as e:
            error_msg = f"Missing data handling failed: {type(e).__name__}: {str(e)}"
            result.add_error(error_msg)
            self._log.error(error_msg, exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Missing Data Analysis
    # ───────────────────────────────────────────────────────────────────
    
    def _analyze_missing(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze missing data per column.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary mapping column → {n_missing, pct_missing}
        """
        n = len(df)
        report: Dict[str, Dict[str, float]] = {}
        
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            pct_missing = n_missing / max(1, n)
            
            report[col] = {
                "n_missing": n_missing,
                "pct_missing": pct_missing
            }
        
        return report
    
    def _get_columns_to_drop(
        self,
        report: Dict[str, Dict[str, float]],
        threshold: float
    ) -> List[str]:
        """Get columns exceeding missing threshold."""
        return [
            col for col, stats in report.items()
            if stats["pct_missing"] > threshold
        ]
    
    def _log_top_missing(
        self,
        report: Dict[str, Dict[str, float]],
        top_n: int
    ) -> None:
        """Log top columns by missing percentage."""
        if not report:
            return
        
        # Sort by missing percentage
        sorted_cols = sorted(
            report.items(),
            key=lambda x: x[1]["pct_missing"],
            reverse=True
        )
        
        top = sorted_cols[:top_n]
        
        # Format message
        msg_parts = []
        for col, stats in top:
            if stats["pct_missing"] > 0:
                msg_parts.append(f"{col}: {stats['pct_missing']*100:.1f}%")
        
        if msg_parts:
            self._log.info(f"Top missing: {', '.join(msg_parts)}")
    
    # ───────────────────────────────────────────────────────────────────
    # Strategy Resolution
    # ───────────────────────────────────────────────────────────────────

    def _resolve_strategies(
        self,
        strategy: Optional[str]
    ) -> Tuple[str, str]:
        """
        Resolve numeric and categorical strategies.
        
        Legacy mapping:
          • 'mode' → categorical='most_frequent'
          • 'drop' → categorical='drop'
          • 'auto', 'mean', 'median', 'knn', 'constant' → numeric
        
        Args:
            strategy: Legacy strategy override
        
        Returns:
            Tuple of (numeric_strategy, categorical_strategy)
        """
        num_strategy = self.config.strategy_numeric
        cat_strategy = self.config.strategy_categorical
        
        # Legacy override
        if strategy:
            s = strategy.lower().strip()
            
            if s in {"auto", "mean", "median", "knn", "constant"}:
                num_strategy = s
            elif s == "mode":
                cat_strategy = "most_frequent"
            elif s == "drop":
                cat_strategy = "drop"
        
        # Resolve 'auto'
        if num_strategy == "auto":
            num_strategy = "median"  # Robust default
        
        if cat_strategy == "auto":
            cat_strategy = "most_frequent"
        
        return num_strategy, cat_strategy
    
    def _resolve_datetime_strategy(self) -> str:
        """Resolve datetime strategy."""
        dt_strategy = self.config.strategy_datetime
        
        if dt_strategy == "auto":
            return "median"  # Robust default
        
        return dt_strategy
    
    # ───────────────────────────────────────────────────────────────────
    # Missing Indicators
    # ───────────────────────────────────────────────────────────────────
    
    def _add_missing_indicators(
        self,
        X: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Add binary missing indicators.
        
        Args:
            X: DataFrame (modified in place)
        
        Returns:
            Tuple of (numeric_indicators, categorical_indicators, datetime_indicators)
        """
        added_num: List[str] = []
        added_cat: List[str] = []
        added_dt: List[str] = []
        
        suffix = self.config.indicator_suffix
        
        # Numeric indicators
        if self.config.add_numeric_missing_indicators:
            num_cols = list(X.select_dtypes(include=[np.number]).columns)
            
            for col in num_cols:
                if X[col].isna().any():
                    indicator_name = f"{col}{suffix}"
                    X[indicator_name] = X[col].isna().astype("Int8")
                    added_num.append(indicator_name)
        
        # Categorical indicators
        if self.config.add_categorical_missing_indicators:
            cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
            
            for col in cat_cols:
                if X[col].isna().any():
                    indicator_name = f"{col}{suffix}"
                    X[indicator_name] = X[col].isna().astype("Int8")
                    added_cat.append(indicator_name)
        
        # Datetime indicators
        if self.config.add_datetime_missing_indicators:
            dt_cols = list(X.select_dtypes(
                include=["datetime64[ns]", "datetime64[ns, UTC]"]
            ).columns)
            
            for col in dt_cols:
                if X[col].isna().any():
                    indicator_name = f"{col}{suffix}"
                    X[indicator_name] = X[col].isna().astype("Int8")
                    added_dt.append(indicator_name)
        
        return added_num, added_cat, added_dt
    
    # ───────────────────────────────────────────────────────────────────
    # Numeric Imputation
    # ───────────────────────────────────────────────────────────────────
    
    def _impute_numeric(
        self,
        X: pd.DataFrame,
        strategy: str,
        group_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        grp_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[Any], List[str]]:
        """
        Impute numeric columns.
        
        Args:
            X: DataFrame (modified in place)
            strategy: Imputation strategy
            group_stats: Group statistics for group-wise
            grp_cols: Grouping columns
        
        Returns:
            Tuple of (DataFrame, fitted_imputer, imputed_columns)
        """
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cols_with_missing = [col for col in num_cols if X[col].isna().any()]
        
        if not cols_with_missing:
            return X, None, []
        
        X_out = X.copy()
        
        # ═══════════════════════════════════════════════════════════════
        # KNN Strategy (multivariate with scaling)
        # ═══════════════════════════════════════════════════════════════
        
        if strategy == "knn":
            # Limit features for performance
            max_feats = self.config.knn_max_features or len(num_cols)
            used_cols = num_cols[:max_feats]
            
            # Scale → KNN → Inverse scale
            scaler = StandardScaler()
            imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            
            data = X_out[used_cols].values.astype(float)
            data_scaled = scaler.fit_transform(data)
            data_imputed_scaled = imputer.fit_transform(data_scaled)
            data_imputed = scaler.inverse_transform(data_imputed_scaled)
            
            X_out[used_cols] = data_imputed
            
            return X_out, {
                "imputer": imputer,
                "scaler": scaler,
                "columns": used_cols
            }, cols_with_missing
        
        # ═══════════════════════════════════════════════════════════════
        # SimpleImputer Strategies
        # ═══════════════════════════════════════════════════════════════
        
        if strategy == "constant":
            imputer = SimpleImputer(
                strategy="constant",
                fill_value=self.config.constant_numeric_fill_value
            )
        else:
            imputer = SimpleImputer(strategy=strategy)
        
        # Group-wise imputation
        if group_stats and grp_cols:
            for col in cols_with_missing:
                X_out[col] = self._fill_from_groups_numeric(
                    X_out, col, grp_cols, group_stats, imputer
                )
            
            return X_out, imputer, cols_with_missing
        
        # Standard imputation
        X_out[cols_with_missing] = imputer.fit_transform(X_out[cols_with_missing])
        
        return X_out, imputer, cols_with_missing
    
    def _fill_from_groups_numeric(
        self,
        X: pd.DataFrame,
        col: str,
        grp_cols: List[str],
        group_stats: Dict[str, Dict[str, Any]],
        fallback: SimpleImputer
    ) -> pd.Series:
        """Fill using group statistics with global fallback."""
        series = X[col].copy()
        mask = series.isna()
        
        if not mask.any():
            return series
        
        # Try group stats
        key = tuple(grp_cols)
        stats = group_stats.get(key, {}).get(col)
        
        if stats and isinstance(stats, dict):
            if stats.get("count", 0) >= self.config.group_min_size:
                fill_value = stats.get("median")
                series.loc[mask] = fill_value
                mask = series.isna()
        
        # Global fallback
        if mask.any():
            series.loc[mask] = fallback.fit_transform(
                series.to_frame()
            )[mask.values, 0]
        
        return series
    
    def _compute_group_stats(
        self,
        X: pd.DataFrame,
        grp_cols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute group-wise statistics.
        
        Returns:
            Dict mapping (grp_cols) → {col → {median, mean, count}}
        """
        stats: Dict[str, Dict[str, Any]] = {}
        
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        
        if not num_cols or not grp_cols:
            return stats
        
        grouped = X.groupby(grp_cols, dropna=False)
        
        key = tuple(grp_cols)
        stats[key] = {}
        
        for col in num_cols:
            try:
                group_medians = grouped[col].median()
                group_means = grouped[col].mean()
                group_counts = grouped[col].count()
                
                stats[key][col] = {
                    "median": group_medians,
                    "mean": group_means,
                    "count": group_counts
                }
            except Exception:
                continue
        
        return stats
    
    # ───────────────────────────────────────────────────────────────────
    # Categorical Imputation
    # ───────────────────────────────────────────────────────────────────
    
    def _impute_categorical(
        self,
        X: pd.DataFrame,
        strategy: str,
        group_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        grp_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[Any], List[str]]:
        """
        Impute categorical columns.
        
        Args:
            X: DataFrame (modified in place)
            strategy: Imputation strategy
            group_stats: Group statistics (not used for categorical yet)
            grp_cols: Grouping columns
        
        Returns:
            Tuple of (DataFrame, fitted_imputer, imputed_columns)
        """
        cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        cols_with_missing = [col for col in cat_cols if X[col].isna().any()]
        
        if not cols_with_missing:
            return X, None, []
        
        X_out = X.copy()
        
        # Drop strategy handled earlier
        if strategy == "drop":
            return X_out, None, []
        
        # SimpleImputer
        if strategy == "constant":
            imputer = SimpleImputer(
                strategy="constant",
                fill_value=self.config.constant_categorical_fill_value
            )
        else:
            imputer = SimpleImputer(strategy="most_frequent")
        
        # Group-wise mode (optional)
        if group_stats and grp_cols:
            for col in cols_with_missing:
                series = X_out[col].copy()
                mask = series.isna()
                
                if mask.any():
                    try:
                        # Group mode
                        group_modes = X_out.groupby(grp_cols)[col].apply(
                            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
                        )
                        
                        # Merge and fill
                        merged = X_out[grp_cols].merge(
                            group_modes.rename("__mode"),
                            left_on=grp_cols,
                            right_index=True,
                            how="left"
                        )
                        
                        series.loc[mask] = merged.loc[mask, "__mode"].values
                    except Exception:
                        pass
                    
                    # Global fallback
                    mask = series.isna()
                    if mask.any():
                        series.loc[mask] = imputer.fit_transform(
                            series.to_frame()
                        )[mask.values, 0]
                
                X_out[col] = series
            
            return X_out, imputer, cols_with_missing
        
        # Standard imputation
        X_out[cols_with_missing] = imputer.fit_transform(X_out[cols_with_missing])
        
        # Preserve category dtype
        for col in cols_with_missing:
            if pd.api.types.is_categorical_dtype(X[col]):
                X_out[col] = X_out[col].astype("category")
        
        return X_out, imputer, cols_with_missing
    
    # ───────────────────────────────────────────────────────────────────
    # Datetime Imputation
    # ───────────────────────────────────────────────────────────────────
    
    def _impute_datetime(
        self,
        X: pd.DataFrame,
        strategy: str,
        constant_value: Optional[pd.Timestamp] = None
    ) -> Tuple[pd.DataFrame, Optional[Any], List[str]]:
        """
        Impute datetime columns.
        
        Args:
            X: DataFrame (modified in place)
            strategy: Imputation strategy
            constant_value: Constant timestamp for constant strategy
        
        Returns:
            Tuple of (DataFrame, imputer_info, imputed_columns)
        """
        dt_cols = list(X.select_dtypes(
            include=["datetime64[ns]", "datetime64[ns, UTC]"]
        ).columns)
        cols_with_missing = [col for col in dt_cols if X[col].isna().any()]
        
        if not cols_with_missing:
            return X, None, []
        
        X_out = X.copy()
        
        for col in cols_with_missing:
            series = X_out[col]
            
            if strategy == "median":
                # Convert to int64, compute median, convert back
                values_int = series.view("int64")
                median_int = np.nanmedian(values_int.astype(float))
                fill_value = pd.to_datetime(
                    int(median_int),
                    unit="ns",
                    utc="UTC" in str(series.dtype)
                )
                X_out[col] = series.fillna(fill_value)
            
            elif strategy == "most_frequent":
                mode = series.mode(dropna=True)
                fill_value = mode.iloc[0] if len(mode) > 0 else series.dropna().min()
                X_out[col] = series.fillna(fill_value)
            
            elif strategy == "forward_fill":
                X_out[col] = series.ffill().bfill()
            
            elif strategy == "backward_fill":
                X_out[col] = series.bfill().ffill()
            
            elif strategy == "constant":
                const = constant_value if isinstance(constant_value, pd.Timestamp) else \
                        pd.Timestamp("1970-01-01")
                X_out[col] = series.fillna(const)
            
            else:
                # Default to median
                values_int = series.view("int64")
                median_int = np.nanmedian(values_int.astype(float))
                fill_value = pd.to_datetime(
                    int(median_int),
                    unit="ns",
                    utc="UTC" in str(series.dtype)
                )
                X_out[col] = series.fillna(fill_value)
        
        imputer_info = {"strategy": strategy}
        
        return X_out, imputer_info, cols_with_missing
    
    # ───────────────────────────────────────────────────────────────────
    # Output Building
    # ───────────────────────────────────────────────────────────────────
    
    def _build_imputation_log(
        self,
        cols_to_drop: List[str],
        rows_dropped_target: int,
        rows_dropped_cat: int,
        num_strategy: str,
        num_cols: List[str],
        cat_strategy: str,
        cat_cols: List[str],
        dt_strategy: str,
        dt_cols: List[str],
        ind_num: List[str],
        ind_cat: List[str],
        ind_dt: List[str]
    ) -> List[str]:
        """Build human-readable imputation log."""
        log: List[str] = []
        
        if cols_to_drop:
            log.append(f"Dropped {len(cols_to_drop)} columns with extreme missing")
        
        if rows_dropped_target:
            log.append(f"Dropped {rows_dropped_target} rows with missing target")
        
        if rows_dropped_cat:
            log.append(f"Dropped {rows_dropped_cat} rows with missing categorical")
        
        if num_cols:
            log.append(f"Imputed {len(num_cols)} numeric columns ({num_strategy})")
        
        if cat_cols:
            log.append(f"Imputed {len(cat_cols)} categorical columns ({cat_strategy})")
        
        if dt_cols:
            log.append(f"Imputed {len(dt_cols)} datetime columns ({dt_strategy})")
        
        if ind_num:
            log.append(f"Added {len(ind_num)} numeric missing indicators")
        
        if ind_cat:
            log.append(f"Added {len(ind_cat)} categorical missing indicators")
        
        if ind_dt:
            log.append(f"Added {len(ind_dt)} datetime missing indicators")
        
        if not log:
            log.append("No missing data operations required")
        
        return log
    
    def _build_fitted_dict(
        self,
        num_strategy: str,
        num_imputer: Optional[Any],
        num_cols: List[str],
        cat_strategy: str,
        cat_imputer: Optional[Any],
        cat_cols: List[str],
        dt_strategy: str,
        dt_imputer: Optional[Any],
        dt_cols: List[str],
        ind_num: List[str],
        ind_cat: List[str],
        ind_dt: List[str],
        cols_to_drop: List[str],
        target_column: str
    ) -> Dict[str, Any]:
        """Build fitted imputers dictionary."""
        return {
            "numeric": {
                "strategy": num_strategy,
                "imputer": num_imputer,
                "columns": num_cols
            },
            "categorical": {
                "strategy": cat_strategy,
                "imputer": cat_imputer,
                "columns": cat_cols
            },
            "datetime": {
                "strategy": dt_strategy,
                "imputer": dt_imputer,
                "columns": dt_cols,
                "constant_value": self.config.constant_datetime_fill_value.isoformat()
                    if isinstance(self.config.constant_datetime_fill_value, pd.Timestamp)
                    else None
            },
            "indicators": {
                "numeric": ind_num,
                "categorical": ind_cat,
                "datetime": ind_dt,
                "suffix": self.config.indicator_suffix
            },
            "dropped_columns": {
                "threshold": self.config.drop_column_missing_pct_threshold,
                "columns": cols_to_drop
            },
            "target_column": target_column,
            "preserve_column_order": self.config.preserve_column_order
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Apply to New Data (Static Method)
    # ───────────────────────────────────────────────────────────────────
    
    @staticmethod
    def apply_to_new(
        new_data: pd.DataFrame,
        fitted: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        🎯 **Apply Fitted Imputers to New Data**
        
        Deterministically applies same transformations to new data.
        
        Args:
            new_data: New DataFrame to impute
            fitted: Fitted imputers from execute()
        
        Returns:
            Imputed DataFrame
        
        Example:
```python
            # Train
            handler = MissingDataHandler()
            result = handler.execute(train_df, 'target')
            fitted = result.data['fitted']
            
            # Test
            test_imputed = MissingDataHandler.apply_to_new(test_df, fitted)
```
        """
        df = new_data.copy()
        original_order = list(df.columns)
        
        # Extract fitted components
        num_pack = fitted.get("numeric", {})
        cat_pack = fitted.get("categorical", {})
        dt_pack = fitted.get("datetime", {})
        ind_pack = fitted.get("indicators", {})
        
        num_cols = num_pack.get("columns", []) or []
        cat_cols = cat_pack.get("columns", []) or []
        dt_cols = dt_pack.get("columns", []) or []
        
        # Ensure required columns exist
        for col in num_cols + cat_cols + dt_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # ═══════════════════════════════════════════════════════════════
        # Missing Indicators
        # ═══════════════════════════════════════════════════════════════
        
        suffix = ind_pack.get("suffix", "__ismissing")
        
        for indicator in ind_pack.get("numeric", []) or []:
            base_col = indicator.replace(suffix, "")
            if base_col in df.columns:
                df[indicator] = df[base_col].isna().astype("Int8")
        
        for indicator in ind_pack.get("categorical", []) or []:
            base_col = indicator.replace(suffix, "")
            if base_col in df.columns:
                df[indicator] = df[base_col].isna().astype("Int8")
        
        for indicator in ind_pack.get("datetime", []) or []:
            base_col = indicator.replace(suffix, "")
            if base_col in df.columns:
                df[indicator] = df[base_col].isna().astype("Int8")
        
        # ═══════════════════════════════════════════════════════════════
        # Numeric Imputation
        # ═══════════════════════════════════════════════════════════════
        
        num_imputer = num_pack.get("imputer")
        
        if num_imputer and num_cols:
            # KNN variant
            if isinstance(num_imputer, dict) and "scaler" in num_imputer:
                used_cols = num_imputer["columns"]
                scaler = num_imputer["scaler"]
                imputer = num_imputer["imputer"]
                
                data = df[used_cols].values.astype(float)
                data_scaled = scaler.transform(data)
                data_imputed_scaled = imputer.transform(data_scaled)
                data_imputed = scaler.inverse_transform(data_imputed_scaled)
                
                df[used_cols] = data_imputed
            
            # SimpleImputer
            else:
                df[num_cols] = num_imputer.transform(df[num_cols])
        
        # ═══════════════════════════════════════════════════════════════
        # Categorical Imputation
        # ═══════════════════════════════════════════════════════════════
        
        cat_imputer = cat_pack.get("imputer")
        
        if cat_imputer and cat_cols:
            df[cat_cols] = cat_imputer.transform(df[cat_cols])
        
        # ═══════════════════════════════════════════════════════════════
        # Datetime Imputation
        # ═══════════════════════════════════════════════════════════════
        
        dt_strategy = dt_pack.get("strategy", "median")
        
        if dt_cols:
            for col in dt_cols:
                # Ensure datetime type
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                series = df[col]
                
                if series.isna().any():
                    if dt_strategy == "median":
                        values_int = series.view("int64")
                        median_int = np.nanmedian(values_int.astype(float))
                        fill_value = pd.to_datetime(
                            int(median_int),
                            unit="ns",
                            utc="UTC" in str(series.dtype)
                        )
                        df[col] = series.fillna(fill_value)
                    
                    elif dt_strategy == "most_frequent":
                        mode = series.mode(dropna=True)
                        fill_value = mode.iloc[0] if len(mode) > 0 else series.dropna().min()
                        df[col] = series.fillna(fill_value)
                    
                    elif dt_strategy == "forward_fill":
                        df[col] = series.ffill().bfill()
                    
                    elif dt_strategy == "backward_fill":
                        df[col] = series.bfill().ffill()
                    
                    elif dt_strategy == "constant":
                        const_iso = dt_pack.get("constant_value")
                        const = pd.Timestamp(const_iso) if const_iso else pd.Timestamp("1970-01-01")
                        df[col] = series.fillna(const)
        
        # ═══════════════════════════════════════════════════════════════
        # Preserve Column Order
        # ═══════════════════════════════════════════════════════════════
        
        if fitted.get("preserve_column_order", False):
            new_cols = [c for c in original_order if c in df.columns] + \
                      [c for c in df.columns if c not in original_order]
            df = df[new_cols]
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def impute_missing(
    data: pd.DataFrame,
    target_column: str,
    config: Optional[MissingHandlerConfig] = None
) -> pd.DataFrame:
    """
    🚀 **Convenience Function: Impute Missing**
    
    Quick missing data imputation with automatic configuration.
    
    Args:
        data: Input DataFrame
        target_column: Target column name
        config: Optional custom configuration
    
    Returns:
        Imputed DataFrame
    
    Example:
```python
        from agents.preprocessing import impute_missing
        
        df_imputed = impute_missing(train_df, 'target')
```
    """
    handler = MissingDataHandler(config)
    result = handler.execute(data=data, target_column=target_column)
    
    if not result.is_success():
        raise RuntimeError(f"Imputation failed: {result.errors}")
    
    return result.data["data"]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _module_init():
    """Initialize module on import."""
    logger.info(f"✓ MissingDataHandler v{__version__} loaded")

_module_init()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{'='*80}")
    print(f"MissingDataHandler v{__version__}")
    print(f"{'='*80}")
    
    # Generate synthetic data with missing values
    np.random.seed(42)
    
    df = pd.DataFrame({
        'num_1': np.random.randn(1000),
        'num_2': np.random.rand(1000) * 100,
        'cat_1': np.random.choice(['A', 'B', 'C'], 1000),
        'cat_2': np.random.choice(['X', 'Y', 'Z'], 1000),
        'date_1': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Add missing values
    df.loc[np.random.choice(df.index, 100), 'num_1'] = np.nan
    df.loc[np.random.choice(df.index, 50), 'cat_1'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'date_1'] = pd.NaT
    
    print("\n✓ Testing missing data handling...")
    print(f"  Missing before: num_1={df['num_1'].isna().sum()}, "
          f"cat_1={df['cat_1'].isna().sum()}, "
          f"date_1={df['date_1'].isna().sum()}")
    
    handler = MissingDataHandler()
    result = handler.execute(
        data=df,
        target_column='target'
    )
    
    if result.is_success():
        print(f"\n✓ Imputation completed successfully")
        
        df_imputed = result.data['data']
        dropped = result.data['dropped']
        telemetry = result.data['telemetry']
        
        print(f"\nSummary:")
        print(f"  Shape: {result.data['shapes']['original']} → {result.data['shapes']['final']}")
        print(f"  Dropped columns: {dropped['columns']}")
        print(f"  Dropped rows (target): {dropped['rows_target']}")
        print(f"  Dropped rows (categorical): {dropped['rows_categorical']}")
        
        print(f"\nTelemetry:")
        for key, value in telemetry['counts'].items():
            print(f"  {key}: {value}")
        
        print(f"\nImputation Log:")
        for log_entry in result.data['imputation_log']:
            print(f"  • {log_entry}")
    
    else:
        print(f"\n✗ Imputation failed:")
        for error in result.errors:
            print(f"  - {error}")
    
    print(f"\n{'='*80}")
    print("USAGE EXAMPLES:")
    print(f"{'='*80}")
    print("""
from agents.preprocessing import MissingDataHandler, MissingHandlerConfig

# Basic usage
handler = MissingDataHandler()

result = handler.execute(
    data=train_df,
    target_column='target'
)

df_imputed = result.data['data']
fitted = result.data['fitted']

# Apply to new data
test_imputed = MissingDataHandler.apply_to_new(test_df, fitted)

# Custom configuration
config = MissingHandlerConfig(
    strategy_numeric='knn',
    add_numeric_missing_indicators=True,
    enable_groupwise_imputation=True
)

handler = MissingDataHandler(config)

# Convenience function
from agents.preprocessing import impute_missing

df_imputed = impute_missing
(train_df, 'target')
    """)