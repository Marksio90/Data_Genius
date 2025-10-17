# agents/preprocessing/scaler_selector.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Scaler Selector v7.0            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE INTELLIGENT FEATURE SCALING WITH RECIPE SYSTEM               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Distribution-Aware Scaler Selection                                   ║
║  ✓ Estimator-Specific Heuristics (tree, SVM, NN, etc.)                   ║
║  ✓ Global vs Per-Feature Strategies                                      ║
║  ✓ Constant/Near-Constant Detection                                      ║
║  ✓ Recipe-Based Reproducibility                                          ║
║  ✓ ColumnTransformer Assembly                                            ║
║  ✓ Feature Name Tracking                                                 ║
║  ✓ Comprehensive Diagnostics                                             ║
║  ✓ Production-Ready with Safety Guards                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Intelligent Heuristics:
    • Tree/Boosting: Prefer 'none' (unless extreme outliers)
    • Very High Skew (|skew| >= 2.0): 'quantile' (normal)
    • High Skew (|skew| >= 1.0): 'power' (Yeo-Johnson)
    • Many Outliers (>5%): 'robust'
    • Bounded [0,1]: 'minmax'
    • Default: 'standard'

Scaling Strategies:
    • none: No scaling (passthrough)
    • standard: StandardScaler (μ=0, σ=1)
    • minmax: MinMaxScaler [0,1]
    • robust: RobustScaler (median, IQR)
    • power: PowerTransformer (Yeo-Johnson)
    • quantile: QuantileTransformer (normal/uniform)

Dependencies:
    • Required: pandas, numpy, scikit-learn, loguru

Usage:
```python
    from agents.preprocessing import ScalerSelector, ScalerSelectorConfig
    
    # Configure
    config = ScalerSelectorConfig(
        prefer_global=True,
        build_transformer=True
    )
    
    selector = ScalerSelector(config)
    
    # Analyze & select
    result = selector.execute(
        data=train_df,
        target_column='target',
        estimator_hint='linear'
    )
    
    transformer = result.data['transformer']
    scaler_map = result.data['scaler_map']
    
    # Transform
    X_scaled = transformer.fit_transform(X_train)
    
    # Apply to test (deterministic)
    X_test_scaled = transformer.transform(X_test)
```
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    MaxAbsScaler
)

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
        "logs/scaler_selector_{time:YYYY-MM-DD}.log",
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

__all__ = ["ScalerSelectorConfig", "ScalerSelector", "select_scaler"]
__version__ = "7.0.0-ultimate"
__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=False)
class ScalerSelectorConfig:
    """
    🎯 **Scaler Selector Configuration**
    
    Complete configuration for intelligent scaler selection.
    
    Thresholds:
        skew_high: High skewness threshold (default: 1.0)
        skew_very_high: Very high skewness threshold (default: 2.0)
        outlier_pct_high: High outlier percentage (default: 0.05)
        zero_inflated_high: High zero percentage (default: 0.5)
        bounded_eps: Epsilon for [0,1] detection (default: 1e-9)
        near_constant_std: Near-constant std threshold (default: 1e-12)
        near_constant_unique_ratio: Near-constant unique ratio (default: 0.01)
        
    Strategy:
        prefer_global: Use global strategy for all features (default: True)
        default_strategy: Default fallback strategy (default: 'standard')
        build_transformer: Build ColumnTransformer (default: True)
        
    Quantile Transformer:
        quantile_output: Output distribution ('normal'/'uniform') (default: 'normal')
        quantile_n_quantiles: Number of quantiles (default: 1000)
        quantile_subsample: Subsample size (default: 100000)
        
    Power Transformer:
        power_method: Transformation method ('yeo-johnson'/'box-cox') (default: 'yeo-johnson')
        power_standardize: Standardize after transform (default: True)
        
    Safety:
        cap_infinite_to_nan: Replace inf with NaN (default: True)
        clip_extreme_quantiles: Clip extremes (e.g., (0.001, 0.999)) (default: None)
        validate_output: Validate scaled output (default: True)
        
    Metadata:
        collect_diagnostics: Collect detailed diagnostics (default: True)
        verbose: Verbose logging (default: True)
    """
    
    # Thresholds
    skew_high: float = 1.0
    skew_very_high: float = 2.0
    outlier_pct_high: float = 0.05
    zero_inflated_high: float = 0.5
    bounded_eps: float = 1e-9
    near_constant_std: float = 1e-12
    near_constant_unique_ratio: float = 0.01
    
    # Strategy
    prefer_global: bool = True
    default_strategy: Literal[
        "none", "standard", "minmax", "robust", "power", "quantile", "maxabs"
    ] = "standard"
    build_transformer: bool = True
    
    # Quantile transformer
    quantile_output: Literal["normal", "uniform"] = "normal"
    quantile_n_quantiles: int = 1000
    quantile_subsample: int = 100000
    
    # Power transformer
    power_method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson"
    power_standardize: bool = True
    
    # Safety
    cap_infinite_to_nan: bool = True
    clip_extreme_quantiles: Optional[Tuple[float, float]] = None
    validate_output: bool = True
    
    # Metadata
    collect_diagnostics: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.skew_high < 0:
            raise ValueError(f"skew_high must be >= 0, got {self.skew_high}")
        
        if self.skew_very_high < self.skew_high:
            raise ValueError(
                f"skew_very_high ({self.skew_very_high}) must be >= "
                f"skew_high ({self.skew_high})"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Scaler Selector
# ═══════════════════════════════════════════════════════════════════════════

class ScalerSelector(BaseAgent):
    """
    🚀 **Ultimate Intelligent Scaler Selector**
    
    Enterprise-grade scaler selection with:
      • Distribution-aware heuristics
      • Estimator-specific recommendations
      • Global vs per-feature strategies
      • Constant column detection
      • Recipe-based reproducibility
      • ColumnTransformer assembly
      • Comprehensive diagnostics
    
    Selection Logic:
```
        ┌──────────────────────────────────────────┐
        │ Analyze Feature Distributions            │
        │  • Skewness, kurtosis                    │
        │  • Outliers (IQR rule)                   │
        │  • Zero-inflation                        │
        │  • Bounded [0,1] detection               │
        │  • Constant/near-constant detection      │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Apply Estimator-Specific Heuristics      │
        │  • Tree/Boosting → 'none' (usually)      │
        │  • SVM/NN/KNN → distribution-aware       │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Select Strategy (Global or Per-Feature)  │
        │  • Very high skew → 'quantile'           │
        │  • High skew → 'power'                   │
        │  • Many outliers → 'robust'              │
        │  • Bounded [0,1] → 'minmax'              │
        │  • Default → 'standard'                  │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Build ColumnTransformer                  │
        │  • Group by strategy                     │
        │  • Handle constants                      │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────┐
        │ Output: Transformer + Recipe + Metadata  │
        └──────────────────────────────────────────┘
```
    
    Usage:
```python
        selector = ScalerSelector(config)
        
        result = selector.execute(
            data=train_df,
            target_column='target',
            estimator_hint='svm'
        )
        
        transformer = result.data['transformer']
        scaler_map = result.data['scaler_map']
        
        # Fit and transform
        X_scaled = transformer.fit_transform(X_train)
        X_test_scaled = transformer.transform(X_test)
```
    """
    
    version: str = __version__
    
    def __init__(self, config: Optional[ScalerSelectorConfig] = None):
        """
        Initialize scaler selector.
        
        Args:
            config: Optional custom configuration
        """
        super().__init__(
            name="ScalerSelector",
            description="Ultimate intelligent scaler selection"
        )
        
        self.config = config or ScalerSelectorConfig()
        self._log = logger.bind(agent="ScalerSelector", version=self.version)
        
        self._log.info(f"✓ ScalerSelector v{self.version} initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        estimator_hint: Optional[Literal[
            "tree", "linear", "svm", "nn", "boosting", "knn", "logistic"
        ]] = None,
        prefer_global: Optional[bool] = None,
        *,
        exclude_columns: Optional[List[str]] = None,
        scaler_map: Optional[Dict[str, Any]] = None,  # For applying fitted scalers
        **kwargs: Any
    ) -> AgentResult:
        """
        🎯 **Execute Scaler Selection**
        
        Analyzes data and selects optimal scaling strategy.
        
        Args:
            data: Input DataFrame
            target_column: Target column to exclude
            estimator_hint: Algorithm hint for strategy
            prefer_global: Override prefer_global config
            exclude_columns: Columns to exclude
            scaler_map: Pre-fitted scalers (for transform mode)
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with transformer, strategies, and metadata
        """
        result = AgentResult(agent_name=self.name)
        t_start = time.perf_counter()
        
        telemetry: Dict[str, Any] = {
            "timing_s": {},
            "counts": {}
        }
        
        try:
            self._log.info(
                f"🔧 Starting scaler selection | "
                f"rows={len(data):,} | "
                f"cols={len(data.columns)}"
            )
            
            # Validation
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("data must be non-empty DataFrame")
            
            df = data.copy()
            
            # Handle infinites
            if self.config.cap_infinite_to_nan:
                df = df.replace([np.inf, -np.inf], np.nan)
            
            # Target validation
            if target_column and target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # Extract features
            X = df.drop(columns=[target_column]) if target_column else df
            
            # Get numeric columns
            num_cols_all = list(X.select_dtypes(include=[np.number]).columns)
            
            # Apply exclusions
            exclude = set(exclude_columns or [])
            num_cols = [c for c in num_cols_all if c not in exclude]
            
            if not num_cols:
                self._log.warning("No numeric features found - scaling skipped")
                
                result.data = {
                    "scaled_data": df,
                    "global_strategy": "none",
                    "per_feature_strategies": {},
                    "transformer": None,
                    "scaler_map": {},
                    "numeric_columns": [],
                    "report": {},
                    "reasoning": ["No numeric features - scaling skipped"],
                    "recipe": {"op": "scale", "strategies": {}, "params": {}},
                    "feature_names_out": None,
                    "telemetry": telemetry,
                    "summary": {
                        "n_numeric": 0,
                        "n_scaled": 0,
                        "strategies_used": {}
                    }
                }
                
                return result
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: Analyze Distributions
            # ═══════════════════════════════════════════════════════════
            
            t = time.perf_counter()
            
            # Optionally clip extremes for stable analysis
            X_analysis = X[num_cols].copy()
            if self.config.clip_extreme_quantiles:
                q_low, q_high = self.config.clip_extreme_quantiles
                lower_bounds = X_analysis.quantile(q_low)
                upper_bounds = X_analysis.quantile(q_high)
                X_analysis = X_analysis.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
            
            report = self._analyze_numeric(X_analysis)
            
            telemetry["timing_s"]["analysis"] = round(time.perf_counter() - t, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: Select Strategies
            # ═══════════════════════════════════════════════════════════
            
            t = time.perf_counter()
            
            # Global strategy
            global_strategy, global_reasons = self._choose_global_strategy(
                report, estimator_hint
            )
            
            # Per-feature strategies
            per_feature = self._choose_per_feature_strategies(report)
            
            # Apply prefer_global
            prefer_global_final = prefer_global if prefer_global is not None else \
                                 self.config.prefer_global
            
            if prefer_global_final:
                # Use global strategy but keep constants as 'none'
                per_feature = {
                    col: "none" if report[col].get("is_constant", False) 
                    else global_strategy
                    for col in num_cols
                }
            
            telemetry["timing_s"]["strategy_selection"] = round(time.perf_counter() - t, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: Build Transformer
            # ═══════════════════════════════════════════════════════════
            
            transformer = None
            feature_names_out = None
            
            if self.config.build_transformer:
                t = time.perf_counter()
                
                transformer = self._build_transformer(
                    num_cols,
                    per_feature,
                    n_rows=len(X)
                )
                
                # Try to get feature names
                try:
                    feature_names_out = list(transformer.get_feature_names_out(
                        input_features=list(X.columns)
                    ))
                except Exception:
                    feature_names_out = None
                
                telemetry["timing_s"]["transformer_build"] = round(time.perf_counter() - t, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 4: Apply Scaling (if scaler_map provided)
            # ═══════════════════════════════════════════════════════════
            
            scaled_data = df.copy()
            
            if scaler_map and transformer:
                # Reuse fitted scalers
                t = time.perf_counter()
                
                X_to_scale = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df
                X_scaled = transformer.transform(X_to_scale)
                
                # Reconstruct DataFrame
                if isinstance(X_scaled, np.ndarray):
                    feat_names = feature_names_out or [f'feature_{i}' for i in range(X_scaled.shape[1])]
                    scaled_data = pd.DataFrame(X_scaled, columns=feat_names, index=df.index)
                else:
                    scaled_data = pd.DataFrame(X_scaled, index=df.index)
                
                # Add target back
                if target_column and target_column in df.columns:
                    scaled_data[target_column] = df[target_column].values
                
                telemetry["timing_s"]["scaling"] = round(time.perf_counter() - t, 4)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 5: Build Output
            # ═══════════════════════════════════════════════════════════
            
            # Reasoning
            reasoning = list(global_reasons)
            reasoning.append(
                f"Strategy mode: {'Global' if prefer_global_final else 'Per-feature'}"
            )
            
            const_cols = [c for c, r in report.items() if r.get("is_constant", False)]
            if const_cols:
                reasoning.append(
                    f"Detected {len(const_cols)} constant columns → 'none'"
                )
            
            # Recipe
            recipe = {
                "op": "scale",
                "strategies": per_feature,
                "params": {
                    "quantile_output": self.config.quantile_output,
                    "quantile_n_quantiles": self.config.quantile_n_quantiles,
                    "power_method": self.config.power_method,
                    "power_standardize": self.config.power_standardize
                }
            }
            
            # Scaler map (for reuse)
            scaler_map_output = self._extract_scaler_map(transformer, per_feature)
            
            # Summary
            strategy_counts = {}
            for strategy in per_feature.values():
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            summary = {
                "n_numeric": len(num_cols),
                "n_scaled": len([s for s in per_feature.values() if s != 'none']),
                "strategies_used": strategy_counts
            }
            
            # Telemetry
            elapsed_s = time.perf_counter() - t_start
            telemetry["timing_s"]["total"] = round(elapsed_s, 4)
            telemetry["counts"] = {
                "n_numeric": len(num_cols),
                "n_constant": len(const_cols)
            }
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 6: Assemble Result
            # ═══════════════════════════════════════════════════════════
            
            result.data = {
                "scaled_data": scaled_data,
                "global_strategy": global_strategy,
                "per_feature_strategies": per_feature,
                "transformer": transformer,
                "scaler_map": scaler_map_output,
                "numeric_columns": num_cols,
                "report": report if self.config.collect_diagnostics else {},
                "reasoning": reasoning,
                "recipe": recipe,
                "feature_names_out": feature_names_out,
                "telemetry": telemetry,
                "summary": summary
            }
            
            self._log.success(
                f"✓ Scaler selection complete | "
                f"strategy={global_strategy} | "
                f"scaled={summary['n_scaled']}/{summary['n_numeric']} | "
                f"time={elapsed_s:.2f}s"
            )
        
        except Exception as e:
            error_msg = f"Scaler selection failed: {type(e).__name__}: {str(e)}"
            result.add_error(error_msg)
            self._log.error(error_msg, exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Distribution Analysis
    # ───────────────────────────────────────────────────────────────────
    
    def _analyze_numeric(
        self,
        X: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze numeric feature distributions.
        
        Args:
            X: Numeric DataFrame
        
        Returns:
            Dictionary mapping column → statistics
        """
        report: Dict[str, Dict[str, Any]] = {}
        n = len(X)
        
        for col in X.columns:
            series = X[col].dropna()
            
            if series.empty:
                report[col] = {
                    "skew": 0.0,
                    "kurtosis": 0.0,
                    "min": np.nan,
                    "max": np.nan,
                    "std": 0.0,
                    "var": 0.0,
                    "iqr": 0.0,
                    "outlier_pct": 0.0,
                    "zero_pct": 0.0,
                    "bounded_01": False,
                    "has_neg": False,
                    "unique_ratio": 0.0,
                    "is_constant": True
                }
                continue
            
            # Basic statistics
            min_val = float(series.min())
            max_val = float(series.max())
            std_val = float(series.std()) if len(series) > 1 else 0.0
            var_val = float(series.var()) if len(series) > 1 else 0.0
            
            # Quantiles and IQR
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1 if not np.isnan(q3) and not np.isnan(q1) else 0.0
            
            # Outliers (IQR rule)
            outlier_pct = 0.0
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_pct = float(((series < lower_bound) | (series > upper_bound)).mean())
            
            # Zero percentage
            zero_pct = float((series == 0).mean())
            
            # Bounded [0,1] detection
            bounded_01 = (
                min_val >= -self.config.bounded_eps and
                max_val <= 1.0 + self.config.bounded_eps
            )
            
            # Has negatives
            has_neg = min_val < 0
            
            # Unique ratio
            unique_ratio = float(series.nunique() / max(1, len(series)))
            
            # Skewness and kurtosis
            try:
                skew = float(series.skew())
            except Exception:
                skew = 0.0
            
            try:
                kurt = float(series.kurtosis())
            except Exception:
                kurt = 0.0
            
            # Constant detection
            is_constant = (
                std_val < self.config.near_constant_std or
                unique_ratio <= self.config.near_constant_unique_ratio
            )
            
            report[col] = {
                "skew": skew,
                "kurtosis": kurt,
                "min": min_val,
                "max": max_val,
                "std": std_val,
                "var": var_val,
                "iqr": iqr,
                "outlier_pct": outlier_pct,
                "zero_pct": zero_pct,
                "bounded_01": bool(bounded_01),
                "has_neg": bool(has_neg),
                "unique_ratio": unique_ratio,
                "is_constant": bool(is_constant)
            }
        
        return report
    
    # ───────────────────────────────────────────────────────────────────
    # Strategy Selection
    # ───────────────────────────────────────────────────────────────────
    
    def _choose_global_strategy(
        self,
        report: Dict[str, Dict[str, Any]],
        estimator_hint: Optional[str]
    ) -> Tuple[str, List[str]]:
        """
        Choose global scaling strategy.
        
        Args:
            report: Feature analysis report
            estimator_hint: Algorithm hint
        
        Returns:
            Tuple of (strategy, reasoning)
        """
        reasoning: List[str] = []
        
        cols = list(report.keys())
        if not cols:
            return "none", ["No columns to scale"]
        
        # Filter out constants
        dynamic_cols = [c for c in cols if not report[c].get("is_constant", False)]
        
        if not dynamic_cols:
            return "none", ["All columns are constant"]
        
        # Calculate aggregate metrics
        skew_high_pct = np.mean([
            abs(report[c]["skew"]) >= self.config.skew_high 
            for c in dynamic_cols
        ])
        
        skew_very_high_pct = np.mean([
            abs(report[c]["skew"]) >= self.config.skew_very_high 
            for c in dynamic_cols
        ])
        
        outlier_pct = np.mean([
            report[c]["outlier_pct"] > self.config.outlier_pct_high 
            for c in dynamic_cols
        ])
        
        bounded_pct = np.mean([
            report[c]["bounded_01"] 
            for c in dynamic_cols
        ])
        
        zero_pct = np.mean([
            report[c]["zero_pct"] > self.config.zero_inflated_high 
            for c in dynamic_cols
        ])
        
        # ═══════════════════════════════════════════════════════════════
        # Estimator-Specific Logic
        # ═══════════════════════════════════════════════════════════════
        
        if estimator_hint in {"tree", "boosting"}:
            if outlier_pct > 0.4 or zero_pct > 0.4:
                reasoning.append(
                    f"Estimator={estimator_hint} with high outliers "
                    f"({outlier_pct:.1%}) or zeros ({zero_pct:.1%}) → 'robust'"
                )
                return "robust", reasoning
                reasoning.append(
                f"Estimator={estimator_hint}: Tree-based models don't require scaling → 'none'"
            )
            return "none", reasoning
        
        if estimator_hint in {"svm", "nn", "knn", "linear", "logistic"}:
            reasoning.append(
                f"Estimator={estimator_hint}: Scale-sensitive algorithm → "
                "distribution-based selection"
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Distribution-Based Logic
        # ═══════════════════════════════════════════════════════════════
        
        # Very high skewness → quantile
        if skew_very_high_pct >= 0.4:
            reasoning.append(
                f"Very high skewness in {skew_very_high_pct:.1%} of features → 'quantile'"
            )
            return "quantile", reasoning
        
        # High skewness → power
        if skew_high_pct >= 0.5:
            reasoning.append(
                f"High skewness in {skew_high_pct:.1%} of features → 'power'"
            )
            return "power", reasoning
        
        # Many outliers or zero-inflated → robust
        if outlier_pct >= 0.3 or zero_pct >= 0.5:
            reasoning.append(
                f"High outliers ({outlier_pct:.1%}) or zeros ({zero_pct:.1%}) → 'robust'"
            )
            return "robust", reasoning
        
        # Most features bounded [0,1] → minmax
        if bounded_pct >= 0.7:
            reasoning.append(
                f"{bounded_pct:.1%} of features bounded [0,1] → 'minmax'"
            )
            return "minmax", reasoning
        
        # Default
        reasoning.append(
            "No strong distribution patterns → 'standard' (z-score normalization)"
        )
        return "standard", reasoning
    
    def _choose_per_feature_strategies(
        self,
        report: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Choose per-feature scaling strategies.
        
        Args:
            report: Feature analysis report
        
        Returns:
            Dictionary mapping column → strategy
        """
        strategies: Dict[str, str] = {}
        
        for col, metrics in report.items():
            # Constant → none
            if metrics.get("is_constant", False):
                strategies[col] = "none"
                continue
            
            skew = abs(metrics["skew"])
            outlier_pct = metrics["outlier_pct"]
            zero_pct = metrics["zero_pct"]
            bounded = metrics["bounded_01"]
            
            # Very high skew → quantile
            if skew >= self.config.skew_very_high:
                strategies[col] = "quantile"
            
            # High skew → power
            elif skew >= self.config.skew_high:
                strategies[col] = "power"
            
            # Many outliers or zeros → robust
            elif outlier_pct > self.config.outlier_pct_high or \
                 zero_pct > self.config.zero_inflated_high:
                strategies[col] = "robust"
            
            # Bounded [0,1] → minmax
            elif bounded:
                strategies[col] = "minmax"
            
            # Default → standard
            else:
                strategies[col] = "standard"
        
        return strategies
    
    # ───────────────────────────────────────────────────────────────────
    # Transformer Construction
    # ───────────────────────────────────────────────────────────────────
    
    def _build_transformer(
        self,
        numeric_columns: List[str],
        per_feature: Dict[str, str],
        n_rows: int
    ) -> ColumnTransformer:
        """
        Build ColumnTransformer for scaling.
        
        Args:
            numeric_columns: Numeric column names
            per_feature: Strategy per column
            n_rows: Number of rows (for quantile transformer)
        
        Returns:
            ColumnTransformer
        """
        # Group columns by strategy
        strategy_groups: Dict[str, List[str]] = {}
        
        for col in numeric_columns:
            strategy = per_feature.get(col, self.config.default_strategy)
            strategy_groups.setdefault(strategy, []).append(col)
        
        # Build transformers
        transformers = []
        
        for strategy, cols in strategy_groups.items():
            if not cols:
                continue
            
            scaler = self._make_scaler(
                strategy,
                n_rows=n_rows,
                n_features=len(cols)
            )
            
            transformer_name = f"{strategy}_scaler"
            transformers.append((transformer_name, scaler, cols))
        
        if not transformers:
            # Passthrough if no scaling needed
            return ColumnTransformer([
                ("passthrough", "passthrough", numeric_columns)
            ])
        
        return ColumnTransformer(
            transformers,
            remainder="passthrough",
            sparse_threshold=0.0
        )
    
    def _make_scaler(
        self,
        strategy: str,
        n_rows: int,
        n_features: int
    ) -> Any:
        """
        Create scaler instance.
        
        Args:
            strategy: Scaler strategy
            n_rows: Number of rows
            n_features: Number of features
        
        Returns:
            Scaler instance
        """
        if strategy == "standard":
            return StandardScaler()
        
        elif strategy == "minmax":
            return MinMaxScaler()
        
        elif strategy == "robust":
            return RobustScaler()
        
        elif strategy == "power":
            return PowerTransformer(
                method=self.config.power_method,
                standardize=self.config.power_standardize
            )
        
        elif strategy == "quantile":
            # Ensure n_quantiles doesn't exceed n_rows
            n_quantiles = min(
                self.config.quantile_n_quantiles,
                max(10, n_rows)
            )
            
            return QuantileTransformer(
                output_distribution=self.config.quantile_output,
                n_quantiles=n_quantiles,
                subsample=self.config.quantile_subsample
            )
        
        elif strategy == "maxabs":
            return MaxAbsScaler()
        
        elif strategy == "none":
            return "passthrough"
        
        else:
            # Default fallback
            self._log.warning(f"Unknown strategy '{strategy}' - using standard")
            return StandardScaler()
    
    def _extract_scaler_map(
        self,
        transformer: Optional[ColumnTransformer],
        per_feature: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract scaler map for reuse.
        
        Args:
            transformer: Fitted ColumnTransformer
            per_feature: Strategy per feature
        
        Returns:
            Scaler map dictionary
        """
        if transformer is None:
            return {}
        
        scaler_map = {
            "transformer": transformer,
            "strategies": per_feature,
            "version": self.version
        }
        
        return scaler_map
    
    # ───────────────────────────────────────────────────────────────────
    # Utility Methods
    # ───────────────────────────────────────────────────────────────────
    
    def get_strategy_description(self, strategy: str) -> str:
        """Get human-readable strategy description."""
        descriptions = {
            "none": "No scaling (passthrough)",
            "standard": "StandardScaler (μ=0, σ=1)",
            "minmax": "MinMaxScaler [0,1]",
            "robust": "RobustScaler (median, IQR)",
            "power": "PowerTransformer (Yeo-Johnson)",
            "quantile": "QuantileTransformer (normal distribution)",
            "maxabs": "MaxAbsScaler [-1,1]"
        }
        return descriptions.get(strategy, f"Unknown: {strategy}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def select_scaler(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    estimator_hint: Optional[str] = None,
    config: Optional[ScalerSelectorConfig] = None
) -> Tuple[ColumnTransformer, Dict[str, str]]:
    """
    🚀 **Convenience Function: Select Scaler**
    
    Quick scaler selection with automatic configuration.
    
    Args:
        data: Input DataFrame
        target_column: Target column name
        estimator_hint: Algorithm hint
        config: Optional custom configuration
    
    Returns:
        Tuple of (transformer, strategies)
    
    Example:
```python
        from agents.preprocessing import select_scaler
        
        transformer, strategies = select_scaler(
            train_df,
            target_column='target',
            estimator_hint='svm'
        )
        
        X_scaled = transformer.fit_transform(X_train)
```
    """
    selector = ScalerSelector(config)
    result = selector.execute(
        data=data,
        target_column=target_column,
        estimator_hint=estimator_hint
    )
    
    if not result.is_success():
        raise RuntimeError(f"Scaler selection failed: {result.errors}")
    
    transformer = result.data['transformer']
    strategies = result.data['per_feature_strategies']
    
    return transformer, strategies


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _module_init():
    """Initialize module on import."""
    logger.info(f"✓ ScalerSelector v{__version__} loaded")

_module_init()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"ScalerSelector v{__version__}")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    
    df = pd.DataFrame({
        'normal': np.random.randn(1000),
        'skewed': np.random.exponential(2, 1000),
        'bounded': np.random.uniform(0, 1, 1000),
        'outliers': np.concatenate([
            np.random.randn(950),
            np.random.uniform(-100, 100, 50)
        ]),
        'constant': np.ones(1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    print("\n✓ Testing scaler selection...")
    print(f"  Data shape: {df.shape}")
    
    selector = ScalerSelector()
    result = selector.execute(
        data=df,
        target_column='target',
        estimator_hint='svm'
    )
    
    if result.is_success():
        print(f"\n✓ Selection completed successfully")
        
        summary = result.data['summary']
        strategies = result.data['per_feature_strategies']
        
        print(f"\nSummary:")
        print(f"  Global strategy: {result.data['global_strategy']}")
        print(f"  Numeric features: {summary['n_numeric']}")
        print(f"  Features scaled: {summary['n_scaled']}")
        
        print(f"\nPer-Feature Strategies:")
        for col, strat in strategies.items():
            desc = selector.get_strategy_description(strat)
            print(f"  {col}: {strat} ({desc})")
        
        print(f"\nReasoning:")
        for reason in result.data['reasoning']:
            print(f"  • {reason}")
        
        if result.data.get('telemetry'):
            print(f"\nTiming:")
            for stage, time_s in result.data['telemetry']['timing_s'].items():
                print(f"  {stage}: {time_s:.4f}s")
    
    else:
        print(f"\n✗ Selection failed:")
        for error in result.errors:
            print(f"  - {error}")
    
    print(f"\n{'='*80}")
    print("USAGE EXAMPLES:")
    print(f"{'='*80}")
    print("""
from agents.preprocessing import ScalerSelector, ScalerSelectorConfig

# Basic usage
selector = ScalerSelector()

result = selector.execute(
    data=train_df,
    target_column='target',
    estimator_hint='svm'
)

transformer = result.data['transformer']
scaler_map = result.data['scaler_map']

# Fit and transform
X = train_df.drop(columns=['target'])
X_scaled = transformer.fit_transform(X)

# Apply to test
X_test_scaled = transformer.transform(X_test)

# Custom configuration
config = ScalerSelectorConfig(
    prefer_global=False,  # Per-feature strategies
    skew_high=0.75,
    build_transformer=True
)

selector = ScalerSelector(config)

# Convenience function
from agents.preprocessing import select_scaler

transformer, strategies = select_scaler(
    train_df,
    target_column='target',
    estimator_hint='linear'
)
    """)
