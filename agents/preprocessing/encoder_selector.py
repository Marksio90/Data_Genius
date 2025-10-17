# agents/preprocessing/encoder_selector.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Encoder Selector v6.0            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ENTERPRISE-GRADE INTELLIGENT CATEGORICAL ENCODING                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Automatic Encoder Selection & Fitting                                 ║
║  ✓ Multiple Encoding Strategies (OHE, Ordinal, Target, Hashing)          ║
║  ✓ Rare Category Handling with Grouping                                  ║
║  ✓ High Cardinality Management                                           ║
║  ✓ Missing Value Imputation                                              ║
║  ✓ Unknown Category Handling                                             ║
║  ✓ Dimension Explosion Protection                                        ║
║  ✓ Target-Based Encoding (Target, LOO, CatBoost)                         ║
║  ✓ Production-Ready ColumnTransformer                                    ║
║  ✓ Comprehensive Telemetry & Recommendations                             ║
║  ✓ Category Encoders Integration (optional)                              ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │              EncoderSelector Core                           │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Categorical Analysis (cardinality, distribution)        │
    │  2. Encoder Selection Logic (per-column strategies)         │
    │  3. Pipeline Construction (imputation → grouping → encode)  │
    │  4. Dimension Management (OHE explosion protection)         │
    │  5. ColumnTransformer Assembly                              │
    │  6. Feature Name Tracking                                   │
    │  7. Recommendations Generation                              │
    └─────────────────────────────────────────────────────────────┘

Encoding Strategies:
    • OneHotEncoder      → Low cardinality (≤20 unique)
    • OrdinalEncoder     → Ordered categories or fallback
    • TargetEncoder      → High cardinality with target
    • LeaveOneOutEncoder → Regression with target
    • CatBoostEncoder    → Classification with target
    • CountEncoder       → Frequency-based encoding
    • HashingEncoder     → Extreme cardinality (100k+)

Rare Category Handling:
    • Automatic detection (< 1% frequency)
    • Grouping into <RARE> token
    • Unknown category handling (<UNK>)

Dependencies:
    • Required: pandas, numpy, scikit-learn, loguru
    • Optional: category_encoders (for advanced encoders)

Usage:
```python
    from agents.preprocessing import EncoderSelector, EncoderPolicy
    
    # Basic usage
    selector = EncoderSelector()
    result = selector.execute(
        data=train_df,
        target_column='target',
        problem_type='classification'
    )
    
    transformer = result.data['transformer']
    X_encoded = transformer.transform(X)
    
    # Custom policy
    policy = EncoderPolicy(
        max_ohe_unique=30,
        rare_min_pct=0.02,
        enable_target_encoder=True
    )
    selector = EncoderSelector(policy)
```
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

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
        "logs/encoder_selector_{time:YYYY-MM-DD}.log",
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

try:
    from config.settings import settings
except ImportError:
    logger.warning("⚠ config.settings not found - using defaults")
    
    class Settings:
        pass
    
    settings = Settings()

# Category Encoders (optional)
try:
    import category_encoders as ce
    _CE_AVAILABLE = True
    logger.info("✓ category_encoders available")
except ImportError:
    ce = None
    _CE_AVAILABLE = False
    logger.warning("⚠ category_encoders not available - using sklearn only")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["EncoderPolicy", "EncoderSelector", "RareCategoryGrouper", "FrequencyEncoder"]
__version__ = "6.0.0-enterprise"
__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=False)
class EncoderPolicy:
    """
    🎯 **Encoder Selection Policy Configuration**
    
    Complete policy for intelligent categorical encoding.
    
    Cardinality Thresholds:
        max_ohe_unique: Maximum unique values for OneHot (default: 20)
        high_cardinality_abs: Absolute threshold for high cardinality (default: 50)
        high_cardinality_ratio: Ratio threshold (unique/rows) (default: 0.30)
        max_ohe_total_features: Max total features after OHE (default: 5000)
        
    Rare Category Handling:
        rare_min_pct: Minimum percentage to avoid <RARE> grouping (default: 0.01)
        rare_token: Token for rare categories (default: '<RARE>')
        enable_rare_grouping: Enable rare category grouping (default: True)
        
    Missing Value Handling:
        impute_strategy_categorical: Strategy for categorical imputation
        impute_strategy_numeric: Strategy for numeric imputation
        add_missing_token: Add explicit missing token (default: True)
        missing_token: Token for missing values (default: '<MISSING>')
        
    Advanced Encoders (requires category_encoders):
        enable_target_encoder: Enable target encoding (default: True)
        enable_catboost_encoder: Enable CatBoost encoding (default: True)
        enable_leave_one_out: Enable LOO encoding (default: True)
        enable_count_encoder: Enable count encoding (default: True)
        enable_hashing_encoder: Enable hashing encoding (default: True)
        
    Behavior:
        handle_unknown_token: Token for unknown categories (default: '<UNK>')
        passthrough_numeric: Pass through numeric features (default: True)
        drop_first: Drop first category in OHE (default: False)
        sparse_output: Use sparse matrices (default: False)
        
    Performance:
        random_state: Random state for reproducibility (default: 42)
        n_jobs: Number of parallel jobs (default: 1)
    """
    
    # Cardinality thresholds
    max_ohe_unique: int = 20
    high_cardinality_abs: int = 50
    high_cardinality_ratio: float = 0.30
    max_ohe_total_features: Optional[int] = 5000
    
    # Rare category handling
    rare_min_pct: float = 0.01
    rare_token: str = "<RARE>"
    enable_rare_grouping: bool = True
    
    # Missing value handling
    impute_strategy_categorical: Literal["most_frequent", "constant"] = "most_frequent"
    impute_strategy_numeric: Literal["median", "mean", "most_frequent", "constant"] = "median"
    add_missing_token: bool = True
    missing_token: str = "<MISSING>"
    
    # Advanced encoders
    enable_target_encoder: bool = True
    enable_catboost_encoder: bool = True
    enable_leave_one_out: bool = True
    enable_count_encoder: bool = True
    enable_hashing_encoder: bool = True
    
    # Behavior
    handle_unknown_token: str = "<UNK>"
    passthrough_numeric: bool = True
    drop_first: bool = False
    sparse_output: bool = False
    
    # Performance
    random_state: int = 42
    n_jobs: int = 1
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_ohe_unique < 2:
            raise ValueError(f"max_ohe_unique must be >= 2, got {self.max_ohe_unique}")
        
        if not 0 < self.rare_min_pct < 1:
            raise ValueError(f"rare_min_pct must be in (0, 1), got {self.rare_min_pct}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def create_fast(cls) -> 'EncoderPolicy':
        """Create fast encoding policy (simple encoders only)."""
        return cls(
            max_ohe_unique=30,
            enable_target_encoder=False,
            enable_catboost_encoder=False,
            enable_leave_one_out=False,
            enable_count_encoder=False
        )
    
    @classmethod
    def create_accurate(cls) -> 'EncoderPolicy':
        """Create accurate encoding policy (prefer target-based)."""
        return cls(
            max_ohe_unique=15,
            rare_min_pct=0.02,
            enable_target_encoder=True,
            enable_catboost_encoder=True,
            enable_leave_one_out=True
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Custom Transformers
# ═══════════════════════════════════════════════════════════════════════════

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    🔧 **Rare Category Grouper Transformer**
    
    Groups rare categories into <RARE> token and handles unknown categories.
    
    Features:
      • Automatic rare category detection
      • Unknown category handling
      • Per-column frequency tracking
      • DataFrame-aware transformation
    
    Args:
        min_pct: Minimum percentage threshold for rare (default: 0.01)
        rare_token: Token for rare categories (default: '<RARE>')
        unk_token: Token for unknown categories (default: '<UNK>')
    
    Example:
```python
        grouper = RareCategoryGrouper(min_pct=0.02)
        grouper.fit(X_train)
        X_transformed = grouper.transform(X_test)
```
    """
    
    def __init__(
        self,
        min_pct: float = 0.01,
        rare_token: str = "<RARE>",
        unk_token: str = "<UNK>"
    ):
        self.min_pct = float(min_pct)
        self.rare_token = rare_token
        self.unk_token = unk_token
        
        # Fitted attributes
        self._seen_: Dict[str, set] = {}
        self._rare_: Dict[str, set] = {}
        self.columns_: List[str] = []
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'RareCategoryGrouper':
        """
        Fit rare category grouper.
        
        Args:
            X: Input data
            y: Target (unused, for sklearn compatibility)
        
        Returns:
            Self
        """
        df = self._to_dataframe(X)
        self.columns_ = list(df.columns)
        n = len(df)
        
        for col in self.columns_:
            # Count values
            value_counts = df[col].astype(str).value_counts(dropna=False)
            
            # Calculate frequencies
            frequencies = value_counts / max(1, n)
            
            # Identify rare categories
            rare_categories = set(
                frequencies[frequencies < self.min_pct].index.astype(str)
            )
            
            # Track all seen categories
            seen_categories = set(value_counts.index.astype(str))
            
            self._rare_[col] = rare_categories
            self._seen_[col] = seen_categories
        
        return self
    
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform data by grouping rare and unknown categories.
        
        Args:
            X: Input data
        
        Returns:
            Transformed DataFrame
        """
        df = self._to_dataframe(X)
        result = pd.DataFrame(index=df.index)
        
        for col in self.columns_:
            series = df[col].astype(str)
            
            # Replace rare categories
            rare_mask = series.isin(self._rare_.get(col, set()))
            series = series.where(~rare_mask, self.rare_token)
            
            # Replace unknown categories
            seen = self._seen_.get(col, set())
            unknown_mask = ~series.isin(seen) & (series != self.rare_token)
            series = series.where(~unknown_mask, self.unk_token)
            
            result[col] = series
        
        return result
    
    def _to_dataframe(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        
        X_array = np.asarray(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        columns = self.columns_ if hasattr(self, 'columns_') else [
            f"col_{i}" for i in range(X_array.shape[1])
        ]
        
        return pd.DataFrame(X_array, columns=columns)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    🔧 **Frequency Encoder Transformer**
    
    Encodes categories as their relative frequencies in training data.
    
    Features:
      • Simple frequency-based encoding
      • Handles unseen categories (frequency = 0)
      • Per-column frequency tracking
      • Fast and memory-efficient
    
    Example:
```python
        encoder = FrequencyEncoder()
        encoder.fit(X_train)
        X_encoded = encoder.transform(X_test)
```
    """
    
    def __init__(self):
        self.freqs_: Dict[str, Dict[str, float]] = {}
        self.columns_: List[str] = []
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'FrequencyEncoder':
        """
        Fit frequency encoder.
        
        Args:
            X: Input data
            y: Target (unused)
        
        Returns:
            Self
        """
        df = self._to_dataframe(X)
        self.columns_ = list(df.columns)
        n = len(df)
        
        for col in self.columns_:
            value_counts = df[col].astype(str).value_counts(dropna=False)
            frequencies = (value_counts / max(1, n)).to_dict()
            self.freqs_[col] = {str(k): float(v) for k, v in frequencies.items()}
        
        return self
    
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform data using frequency encoding.
        
        Args:
            X: Input data
        
        Returns:
            Encoded DataFrame
        """
        df = self._to_dataframe(X)
        result = pd.DataFrame(index=df.index)
        
        for col in self.columns_:
            mapping = self.freqs_.get(col, {})
            encoded = df[col].astype(str).map(mapping).fillna(0.0).astype(float)
            result[col] = encoded
        
        return result
    
    def _to_dataframe(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        
        X_array = np.asarray(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        columns = self.columns_ if hasattr(self, 'columns_') else [
            f"col_{i}" for i in range(X_array.shape[1])
        ]
        
        return pd.DataFrame(X_array, columns=columns)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Encoder Selector (Main Class)
# ═══════════════════════════════════════════════════════════════════════════

class EncoderSelector(BaseAgent):
    """
    🚀 **EncoderSelector PRO Master Enterprise ++++**
    
    Enterprise-grade intelligent categorical encoding system.
    
    Capabilities:
      1. Automatic encoder selection per column
      2. Cardinality-aware strategy selection
      3. Rare category grouping
      4. Missing value handling
      5. Unknown category handling
      6. Dimension explosion protection
      7. Target-based encoding (when available)
      8. Production-ready ColumnTransformer
      9. Comprehensive recommendations
     10. Feature name tracking
    
    Encoding Strategies:
```
        ┌──────────────────────────────────────────┐
        │ Cardinality Analysis                     │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Low (≤20)  │ Medium  │ High (≥50)        │
        ├────────────┼─────────┼───────────────────┤
        │ OneHot     │ Target  │ Hashing/Target    │
        │ Ordinal    │ Count   │ CatBoost/LOO      │
        └────────────┴─────────┴───────────────────┘
```
    
    Usage:
```python
        # Basic usage
        selector = EncoderSelector()
        
        result = selector.execute(
            data=train_df,
            target_column='target',
            problem_type='classification'
        )
        
        transformer = result.data['transformer']
        X_encoded = transformer.transform(X_test)
        
        # Custom policy
        policy = EncoderPolicy.create_accurate()
        selector = EncoderSelector(policy)
```
    """
    
    version: str = __version__
    
    def __init__(self, policy: Optional[EncoderPolicy] = None):
        """
        Initialize encoder selector.
        
        Args:
            policy: Optional custom policy
        """
        super().__init__(
            name="EncoderSelector",
            description="Enterprise intelligent categorical encoding"
        )
        
        self.policy = policy or EncoderPolicy()
        self._log = logger.bind(agent="EncoderSelector", version=self.version)
        self._ce_available = _CE_AVAILABLE
        
        if not self._ce_available:
            self._log.warning(
                "category_encoders not available - using sklearn encoders only"
            )
        
        self._log.info(f"✓ EncoderSelector v{self.version} initialized")
    
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
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        
        data = kwargs["data"]
        
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("'data' must be non-empty DataFrame")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: Optional[Literal["classification", "regression"]] = None,
        ordinal_maps: Optional[Dict[str, List[str]]] = None,
        strategy: Literal["auto", "fast", "accurate"] = "auto",
        **kwargs: Any
    ) -> AgentResult:
        """
        🎯 **Execute Encoder Selection**
        
        Analyzes data and selects optimal encoders per column.
        
        Args:
            data: Training data with features + optional target
            target_column: Target column name
            problem_type: 'classification' or 'regression'
            ordinal_maps: User-defined ordinal mappings {col: [ordered_values]}
            strategy: 'auto' (balanced), 'fast' (simple), 'accurate' (target-based)
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with transformer, plan, and recommendations
        """
        result = AgentResult(agent_name=self.name)
        t_start = time.perf_counter()
        
        try:
            self._log.info(
                f"🔧 Starting encoder selection | "
                f"rows={len(data):,} | "
                f"cols={len(data.columns)}"
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: Data Preparation
            # ═══════════════════════════════════════════════════════════
            
            df = data.copy()
            ordinal_maps = ordinal_maps or {}
            
            # Split X and y
            y = None
            if target_column and target_column in df.columns:
                y = df[target_column]
                X = df.drop(columns=[target_column])
            else:
                X = df
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: Column Type Detection
            # ═══════════════════════════════════════════════════════════
            
            cat_cols = self._get_categorical_columns(X)
            num_cols = self._get_numeric_columns(X)
            
            self._log.info(
                f"📊 Detected: {len(cat_cols)} categorical, "
                f"{len(num_cols)} numeric"
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: Handle Numeric-Only Case
            # ═══════════════════════════════════════════════════════════
            
            if not cat_cols and self.policy.passthrough_numeric:
                transformer = self._build_numeric_only_transformer(num_cols)
                transformer.fit(X)
                
                feature_names = self._get_feature_names(transformer, X.columns)
                
                result.data = {
                    "transformer": transformer,
                    "plan": {},
                    "encoded_feature_names": feature_names,
                    "summary": {
                        "n_categorical": 0,
                        "n_numeric": len(num_cols),
                        "strategy": strategy
                    },
                    "telemetry": {
                        "ce_available": self._ce_available,
                        "elapsed_s": time.perf_counter() - t_start
                    }
                }
                
                self._log.info("No categorical columns - numeric passthrough only")
                return result
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 4: Categorical Analysis
            # ═══════════════════════════════════════════════════════════
            
            stats = self._analyze_categories(X[cat_cols])
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 5: Encoder Selection
            # ═══════════════════════════════════════════════════════════
            
            selection, recommendations, est_ohe_total = self._select_encoders(
                stats=stats,
                strategy=strategy,
                has_target=(y is not None),
                problem_type=problem_type,
                ordinal_maps=ordinal_maps
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 6: Dimension Check
            # ═══════════════════════════════════════════════════════════
            
            dimension_warning = None
            
            if self.policy.max_ohe_total_features and est_ohe_total:
                if est_ohe_total > self.policy.max_ohe_total_features:
                    dimension_warning = (
                        f"Estimated OHE features ({est_ohe_total}) exceeds limit "
                        f"({self.policy.max_ohe_total_features}). "
                        "Consider target-based or hashing encoders."
                    )
                    self._log.warning(dimension_warning)
                    result.add_warning(dimension_warning)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 7: Transformer Construction
            # ═══════════════════════════════════════════════════════════
            
            transformer = self._build_transformer(
                X=X,
                y=y,
                selection=selection,
                num_cols=num_cols
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 8: Fit Transformer
            # ═══════════════════════════════════════════════════════════
            
            if y is not None:
                transformer.fit(X, y)
            else:
                transformer.fit(X)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 9: Feature Name Tracking
            # ═══════════════════════════════════════════════════════════
            
            feature_names = self._get_feature_names(transformer, X.columns)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 10: Telemetry & Summary
            # ═══════════════════════════════════════════════════════════
            
            elapsed_s = time.perf_counter() - t_start
            
            summary = {
                "n_categorical": len(cat_cols),
                "n_numeric": len(num_cols),
                "n_encoded_features": len(feature_names) if feature_names else None,
                "strategy": strategy,
                "ce_available": self._ce_available,
                "estimated_ohe_features": est_ohe_total
            }
            
            telemetry = {
                "elapsed_s": round(elapsed_s, 4),
                "version": self.version,
                "warnings": [dimension_warning] if dimension_warning else []
            }
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 11: Assemble Result
            # ═══════════════════════════════════════════════════════════
            
            result.data = {
                "transformer": transformer,
                "plan": selection,
                "encoded_feature_names": feature_names,
                "recommendations": recommendations if recommendations else None,
                "summary": summary,
                "telemetry": telemetry
            }
            
            self._log.success(
                f"✓ Encoding complete | "
                f"categorical={len(cat_cols)} | "
                f"numeric={len(num_cols)} | "
                f"time={elapsed_s:.2f}s"
            )
        
        except Exception as e:
            error_msg = f"Encoder selection failed: {type(e).__name__}: {str(e)}"
            result.add_error(error_msg)
            self._log.error(error_msg, exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Column Type Detection
    # ───────────────────────────────────────────────────────────────────
    
    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get categorical column names."""
        return [
            col for col in df.columns
            if pd.api.types.is_object_dtype(df[col]) or 
               pd.api.types.is_categorical_dtype(df[col])
        ]
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric column names."""
        return [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]
    
    # ───────────────────────────────────────────────────────────────────
    # Categorical Analysis
    # ───────────────────────────────────────────────────────────────────
    
    def _analyze_categories(
        self,
        df_cat: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze categorical features.
        
        Args:
            df_cat: DataFrame with categorical columns
        
        Returns:
            Dictionary with per-column statistics
        """
        stats: Dict[str, Dict[str, Any]] = {}
        n = len(df_cat)
        
        for col in df_cat.columns:
            series = df_cat[col]
            
            # Basic statistics
            n_unique = int(series.nunique(dropna=True))
            n_missing = int(series.isna().sum())
            missing_pct = (n_missing / max(1, n)) * 100.0
            
            # Value counts
            value_counts = series.astype(str).value_counts(dropna=False)
            
            # Top values
            top_values = value_counts.head(5).to_dict()
            
            # High cardinality detection
            is_high_card = (
                n_unique >= self.policy.high_cardinality_abs or
                (n_unique / max(1, n)) >= self.policy.high_cardinality_ratio
            )
            
            stats[col] = {
                "n_unique": n_unique,
                "n_missing": n_missing,
                "missing_pct": missing_pct,
                "top_values": {str(k): int(v) for k, v in top_values.items()},
                "high_cardinality": is_high_card,
                "value_counts": value_counts
            }
        
        return stats
    
    # ───────────────────────────────────────────────────────────────────
    # Encoder Selection
    # ───────────────────────────────────────────────────────────────────
    
    def _select_encoders(
        self,
        stats: Dict[str, Dict[str, Any]],
        strategy: str,
        has_target: bool,
        problem_type: Optional[str],
        ordinal_maps: Dict[str, List[str]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], Optional[int]]:
        """
        Select optimal encoder for each column.
        
        Args:
            stats: Column statistics
            strategy: Selection strategy
            has_target: Whether target is available
            problem_type: Problem type
            ordinal_maps: User-defined ordinal mappings
        
        Returns:
            Tuple of (selection_dict, recommendations, estimated_ohe_features)
        """
        selection: Dict[str, Dict[str, Any]] = {}
        recommendations: List[str] = []
        total_ohe_features = 0
        
        for col, col_stats in stats.items():
            n_unique = col_stats["n_unique"]
            is_high_card = col_stats["high_cardinality"]
            missing_pct = col_stats["missing_pct"]
            
            # ═══════════════════════════════════════════════════════════
            # Priority 1: User-Defined Ordinal
            # ═══════════════════════════════════════════════════════════
            
            if col in ordinal_maps and ordinal_maps[col]:
                selection[col] = {
                    "encoder": "OrdinalEncoder",
                    "params": {
                        "categories": [ordinal_maps[col]],
                        "handle_unknown": "use_encoded_value",
                        "unknown_value": -1
                    },
                    "with_rare_grouper": self.policy.enable_rare_grouping,
                    "with_missing_token": self.policy.add_missing_token,
                    "reason": "User-defined ordinal mapping"
                }
                continue
            
            # ═══════════════════════════════════════════════════════════
            # Priority 2: Low Cardinality → OneHot
            # ═══════════════════════════════════════════════════════════
            
            if n_unique <= self.policy.max_ohe_unique and not is_high_card:
                # Estimate OHE dimensions
                est_features = self._estimate_ohe_dimensions(
                    col_stats["value_counts"]
                )
                total_ohe_features += est_features
                
                selection[col] = {
                    "encoder": "OneHotEncoder",
                    "params": {
                        "handle_unknown": "ignore",
                        "drop": "first" if self.policy.drop_first else None,
                        "sparse_output": self.policy.sparse_output
                    },
                    "with_rare_grouper": self.policy.enable_rare_grouping,
                    "with_missing_token": self.policy.add_missing_token,
                    "reason": f"Low cardinality ({n_unique} ≤ {self.policy.max_ohe_unique})",
                    "estimated_features": est_features
                }
                continue
            
            # ═══════════════════════════════════════════════════════════
            # Priority 3: High Cardinality with Target
            # ═══════════════════════════════════════════════════════════
            
            if self._ce_available and has_target and strategy in {"auto", "accurate"}:
                
                # Regression: LeaveOneOut
                if problem_type == "regression" and self.policy.enable_leave_one_out:
                    selection[col] = {
                        "encoder": "LeaveOneOutEncoder",
                        "params": {"random_state": self.policy.random_state},
                        "with_rare_grouper": self.policy.enable_rare_grouping,
                        "with_missing_token": self.policy.add_missing_token,
                        "reason": "High cardinality + regression → LeaveOneOut"
                    }
                    continue
                
                # Classification: CatBoost
                elif problem_type == "classification" and self.policy.enable_catboost_encoder:
                    selection[col] = {
                        "encoder": "CatBoostEncoder",
                        "params": {"random_state": self.policy.random_state},
                        "with_rare_grouper": self.policy.enable_rare_grouping,
                        "with_missing_token": self.policy.add_missing_token,
                        "reason": "High cardinality + classification → CatBoost"
                    }
                    continue
                
                # Generic: Target Encoder
                elif self.policy.enable_target_encoder:
                    selection[col] = {
                        "encoder": "TargetEncoder",
                        "params": {"random_state": self.policy.random_state},
                        "with_rare_grouper": self.policy.enable_rare_grouping,
                        "with_missing_token": self.policy.add_missing_token,
                        "reason": "High cardinality + target → TargetEncoder"
                    }
                    continue
            
            # ═══════════════════════════════════════════════════════════
            # Priority 4: High Cardinality without Target
            # ═══════════════════════════════════════════════════════════
            
            if self._ce_available and strategy == "accurate" and self.policy.enable_count_encoder:
                selection[col] = {
                    "encoder": "CountEncoder",
                    "params": {},
                    "with_rare_grouper": self.policy.enable_rare_grouping,
                    "with_missing_token": self.policy.add_missing_token,
                    "reason": "High cardinality without target → CountEncoder"
                }
                continue
            
            # ═══════════════════════════════════════════════════════════
            # Priority 5: Extreme Cardinality → Hashing
            # ═══════════════════════════════════════════════════════════
            
            if self._ce_available and n_unique > 100 and self.policy.enable_hashing_encoder:
                selection[col] = {
                    "encoder": "HashingEncoder",
                    "params": {
                        "n_components": min(16, n_unique // 10),
                        "max_process": 1
                    },
                    "with_rare_grouper": False,  # Hashing is robust
                    "with_missing_token": self.policy.add_missing_token,
                    "reason": f"Extreme cardinality ({n_unique}) → HashingEncoder"
                }
                continue
            
            # ═══════════════════════════════════════════════════════════
            # Fallback: Ordinal
            # ═══════════════════════════════════════════════════════════
            
            selection[col] = {
                "encoder": "OrdinalEncoder",
                "params": {
                    "handle_unknown": "use_encoded_value",
                    "unknown_value": -1
                },
                "with_rare_grouper": self.policy.enable_rare_grouping,
                "with_missing_token": self.policy.add_missing_token,
                "reason": "Fallback: OrdinalEncoder"
            }
            
            # ═══════════════════════════════════════════════════════════
            # Recommendations
            # ═══════════════════════════════════════════════════════════
            
            if is_high_card and not self._ce_available and has_target:
                recommendations.append(
                    f"Install 'category_encoders' for better handling of "
                    f"high-cardinality column '{col}' (currently using Ordinal)"
                )
            
            if missing_pct > 10.0:
                recommendations.append(
                    f"Column '{col}' has {missing_pct:.1f}% missing values - "
                    "consider upstream imputation"
                )
        
        # Remove duplicates
        recommendations = sorted(set(recommendations))
        
        return selection, recommendations, total_ohe_features if total_ohe_features > 0 else None
    
    def _estimate_ohe_dimensions(self, value_counts: pd.Series) -> int:
        """
        Estimate number of features after OneHot encoding.
        
        Args:
            value_counts: Value counts series
        
        Returns:
            Estimated number of features
        """
        n = int(value_counts.sum())
        
        if n == 0:
            return 1
        
        # Account for rare grouping
        if self.policy.enable_rare_grouping and self.policy.rare_min_pct > 0:
            frequencies = value_counts / n
            n_frequent = (frequencies >= self.policy.rare_min_pct).sum()
            has_rare = (frequencies < self.policy.rare_min_pct).any()
            
            # +1 for <RARE> if needed
            return int(n_frequent + (1 if has_rare else 0))
        
        return len(value_counts)
    
    # ───────────────────────────────────────────────────────────────────
    # Transformer Construction
    # ───────────────────────────────────────────────────────────────────
    
    def _build_numeric_only_transformer(
        self,
        num_cols: List[str]
    ) -> ColumnTransformer:
        """Build transformer for numeric columns only."""
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy=self.policy.impute_strategy_numeric
            ))
        ])
        
        return ColumnTransformer(
            transformers=[("num", num_pipeline, num_cols)],
            remainder="drop",
            sparse_threshold=0.0
        )
    
    def _build_transformer(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        selection: Dict[str, Dict[str, Any]],
        num_cols: List[str]
    ) -> ColumnTransformer:
        """
        Build complete ColumnTransformer.
        
        Args:
            X: Feature data
            y: Target data
            selection: Encoder selection per column
            num_cols: Numeric column names
        
        Returns:
            Fitted ColumnTransformer
        """
        transformers: List[Tuple[str, Any, List[str]]] = []
        
        # ═══════════════════════════════════════════════════════════════
        # Numeric Columns
        # ═══════════════════════════════════════════════════════════════
        
        if num_cols and self.policy.passthrough_numeric:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(
                    strategy=self.policy.impute_strategy_numeric
                ))
            ])
            transformers.append(("num", num_pipeline, num_cols))
        
        # ═══════════════════════════════════════════════════════════════
        # Categorical Columns (per-column pipelines)
        # ═══════════════════════════════════════════════════════════════
        
        for col, spec in selection.items():
            pipeline_steps: List[Tuple[str, Any]] = []
            
            # Step 1: Missing value imputation
            if spec.get("with_missing_token") and \
               self.policy.impute_strategy_categorical == "constant":
                pipeline_steps.append((
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=self.policy.missing_token
                    )
                ))
            else:
                pipeline_steps.append((
                    "imputer",
                    SimpleImputer(strategy=self.policy.impute_strategy_categorical)
                ))
            
            # Step 2: Rare category grouping
            if spec.get("with_rare_grouper") and self.policy.enable_rare_grouping:
                pipeline_steps.append((
                    "rare_grouper",
                    RareCategoryGrouper(
                        min_pct=self.policy.rare_min_pct,
                        rare_token=self.policy.rare_token,
                        unk_token=self.policy.handle_unknown_token
                    )
                ))
            
            # Step 3: Encoder
            encoder = self._create_encoder(
                spec["encoder"],
                spec.get("params", {})
            )
            pipeline_steps.append(("encoder", encoder))
            
            # Build pipeline
            pipeline = Pipeline(steps=pipeline_steps)
            transformers.append((f"cat__{col}", pipeline, [col]))
        
        # ═══════════════════════════════════════════════════════════════
        # Assemble ColumnTransformer
        # ═══════════════════════════════════════════════════════════════
        
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0,
            n_jobs=self.policy.n_jobs
        )
    
    def _create_encoder(
        self,
        encoder_name: str,
        params: Dict[str, Any]
    ) -> Any:
        """
        Create encoder instance.
        
        Args:
            encoder_name: Encoder class name
            params: Encoder parameters
        
        Returns:
            Encoder instance
        """
        # ═══════════════════════════════════════════════════════════════
        # Sklearn Encoders
        # ═══════════════════════════════════════════════════════════════
        
        if encoder_name == "OneHotEncoder":
            # Handle sklearn version differences
            try:
                return OneHotEncoder(**params)
            except TypeError:
                # Fallback for older sklearn versions
                params_fixed = params.copy()
                if "sparse_output" in params_fixed:
                    params_fixed["sparse"] = params_fixed.pop("sparse_output")
                return OneHotEncoder(**params_fixed)
        
        if encoder_name == "OrdinalEncoder":
            return OrdinalEncoder(**params)
        
        # ═══════════════════════════════════════════════════════════════
        # Category Encoders (if available)
        # ═══════════════════════════════════════════════════════════════
        
        if not self._ce_available:
            self._log.warning(
                f"category_encoders not available - "
                f"using OrdinalEncoder instead of {encoder_name}"
            )
            return OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        
        if encoder_name == "TargetEncoder":
            return ce.TargetEncoder(**params)
        
        if encoder_name == "LeaveOneOutEncoder":
            return ce.LeaveOneOutEncoder(**params)
        
        if encoder_name == "CatBoostEncoder":
            return ce.CatBoostEncoder(**params)
        
        if encoder_name == "CountEncoder":
            return ce.CountEncoder(**params)
        
        if encoder_name == "HashingEncoder":
            return ce.HashingEncoder(**params)
        
        # Fallback
        self._log.warning(f"Unknown encoder: {encoder_name} - using Ordinal")
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Feature Names
    # ───────────────────────────────────────────────────────────────────
    
    def _get_feature_names(
        self,
        transformer: ColumnTransformer,
        input_features: pd.Index
    ) -> Optional[List[str]]:
        """
        Extract feature names from fitted transformer.
        
        Args:
            transformer: Fitted ColumnTransformer
            input_features: Original feature names
        
        Returns:
            List of output feature names or None
        """
        try:
            return list(transformer.get_feature_names_out(input_features))
        except Exception as e:
            self._log.debug(f"Could not extract feature names: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def encode_categorical(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    strategy: str = "auto",
    policy: Optional[EncoderPolicy] = None
) -> Tuple[ColumnTransformer, List[str]]:
    """
    🚀 **Convenience Function: Encode Categorical**
    
    Quick categorical encoding with automatic selection.
    
    Args:
        data: Input data
        target_column: Target column name
        strategy: Encoding strategy
        policy: Optional custom policy
    
    Returns:
        Tuple of (transformer, feature_names)
    
    Example:
```python
        from agents.preprocessing import encode_categorical
        
        transformer, features = encode_categorical(
            train_df,
            target_column='target',
            strategy='accurate'
        )
        
        X_encoded = transformer.transform(X_test)
```
    """
    selector = EncoderSelector(policy)
    result = selector.execute(
        data=data,
        target_column=target_column,
        strategy=strategy
    )
    
    if not result.is_success():
        raise RuntimeError(f"Encoding failed: {result.errors}")
    
    transformer = result.data["transformer"]
    feature_names = result.data.get("encoded_feature_names", [])
    
    return transformer, feature_names


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _module_init():
    """Initialize module on import."""
    logger.info(f"✓ EncoderSelector v{__version__} loaded")

_module_init()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{'='*80}")
    print(f"EncoderSelector v{__version__}")
    print(f"{'='*80}")
    
    # Generate synthetic data
    np.random.seed(42)
    
    df = pd.DataFrame({
        'cat_low': np.random.choice(['A', 'B', 'C'], 1000),
        'cat_medium': np.random.choice([f'cat_{i}' for i in range(30)], 1000),
        'cat_high': np.random.choice([f'val_{i}' for i in range(200)], 1000),
        'num_1': np.random.randn(1000),
        'num_2': np.random.randint(0, 100, 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    print("\n✓ Testing encoder selection...")
    
    selector = EncoderSelector()
    result = selector.execute(
        data=df,
        target_column='target',
        problem_type='classification',
        strategy='auto'
    )
    
    if result.is_success():
        print(f"\n✓ Encoding completed successfully")
        
        summary = result.data['summary']
        plan = result.data['plan']
        
        print(f"\nSummary:")
        print(f"  Categorical: {summary['n_categorical']}")
        print(f"  Numeric: {summary['n_numeric']}")
        print(f"  Estimated features: {summary.get('estimated_ohe_features', 'N/A')}")
        
        print(f"\nEncoding Plan:")
        for col, spec in plan.items():
            print(f"  {col}:")
            print(f"    Encoder: {spec['encoder']}")
            print(f"    Reason: {spec['reason']}")
        
        if result.data.get('recommendations'):
            print(f"\nRecommendations:")
            for rec in result.data['recommendations']:
                print(f"  • {rec}")
    
    else:
        print(f"\n✗ Encoding failed:")
        for error in result.errors:
            print(f"  - {error}")
    
    print(f"\n{'='*80}")
    print("USAGE EXAMPLES:")
    print(f"{'='*80}")
    print("""
from agents.preprocessing import EncoderSelector, EncoderPolicy

# Basic usage
selector = EncoderSelector()

result = selector.execute(
    data=train_df,
    target_column='target',
    problem_type='classification'
)

transformer = result.data['transformer']
X_encoded = transformer.transform(X_test)

# Custom policy
policy = EncoderPolicy(
    max_ohe_unique=30,
    rare_min_pct=0.02,
    enable_target_encoder=True
)

selector = EncoderSelector(policy)

# Convenience function
from agents.preprocessing import encode_categorical

transformer, features = encode_categorical(
    train_df,
    target_column='target',
    strategy='accurate'
)
    """)