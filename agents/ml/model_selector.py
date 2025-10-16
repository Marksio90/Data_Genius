# agents/ml/model_selector.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Model Selector                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Intelligent model selection with meta-learning:                           ║
║    ✓ Automatic model recommendation (classification/regression)           ║
║    ✓ Meta-feature extraction (complexity, statistics, structure)          ║
║    ✓ Multi-criteria ranking (6 dimensions)                                ║
║    ✓ Performance prediction with penalties                                ║
║    ✓ Optional quick CV benchmark                                          ║
║    ✓ High cardinality & sparsity handling                                 ║
║    ✓ Multi-objective optimization                                         ║
║    ✓ Imbalance detection & adaptation                                     ║
║    ✓ Time series detection (heuristics)                                   ║
║    ✓ Model registry with 13+ models                                       ║
║    ✓ Comprehensive telemetry                                              ║
║    ✓ Stable output contract                                               ║
╚════════════════════════════════════════════════════════════════════════════╝

Selection Criteria:
    • Accuracy — Maximize predictive performance
    • Speed — Optimize training/inference time
    • Interpretability — Prefer explainable models
    • Robustness — Handle outliers/noise well
    • Scalability — Scale to large datasets
    • Memory — Minimize memory footprint
    • Balanced — Equal weighting of all factors

Supported Models (13+):
    Linear: Logistic Regression, Ridge, Lasso
    Trees: Decision Tree, Random Forest, Extra Trees
    Boosting: XGBoost, LightGBM, CatBoost, Gradient Boosting
    Neural: MLP
    Other: SVM, KNN, Naive Bayes
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Optional boosting libraries
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

warnings.filterwarnings('ignore')


__all__ = [
    "ProblemType",
    "ModelCategory",
    "SelectionCriterion",
    "DatasetCharacteristics",
    "ModelScore",
    "SelectionResult",
    "ModelRegistry",
    "EstimatorFactory",
    "ModelSelector",
    "select_best_models"
]
__version__ = "5.1-kosmos-enterprise"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class ProblemType(str, Enum):
    """Problem type classification."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    IMBALANCED = "imbalanced"
    HIGH_DIMENSIONAL = "high_dimensional"


class ModelCategory(str, Enum):
    """Model category classification."""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    ENSEMBLE = "ensemble"
    BOOSTING = "boosting"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    NAIVE_BAYES = "naive_bayes"
    KNN = "knn"


class SelectionCriterion(str, Enum):
    """Model selection criteria."""
    ACCURACY = "accuracy"
    SPEED = "speed"
    INTERPRETABILITY = "interpretability"
    ROBUSTNESS = "robustness"
    SCALABILITY = "scalability"
    MEMORY = "memory"
    BALANCED = "balanced"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetCharacteristics:
    """
    Dataset characteristics for meta-learning.
    
    Attributes:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes (classification)
        class_balance: Class distribution
        n_numerical: Numerical feature count
        n_categorical: Categorical feature count
        n_binary: Binary feature count
        avg_skewness: Average feature skewness
        avg_kurtosis: Average feature kurtosis
        missing_ratio: Missing value ratio
        feature_to_sample_ratio: Feature-to-sample ratio
        avg_correlation: Average feature correlation
        dimensionality_score: Dimensionality complexity
        is_imbalanced: Imbalanced classes flag
        is_high_dimensional: High-dimensional flag
        is_sparse: Sparse data flag
        index_is_datetime: DateTime index flag
        index_is_sorted: Sorted index flag
        metadata: Additional metadata
    """
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    class_balance: Optional[Dict[str, float]]
    
    n_numerical: int
    n_categorical: int
    n_binary: int
    
    avg_skewness: float
    avg_kurtosis: float
    missing_ratio: float
    
    feature_to_sample_ratio: float
    avg_correlation: float
    dimensionality_score: float
    
    is_imbalanced: bool
    is_high_dimensional: bool
    is_sparse: bool
    
    index_is_datetime: bool = False
    index_is_sorted: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "class_balance": self.class_balance,
            "n_numerical": self.n_numerical,
            "n_categorical": self.n_categorical,
            "n_binary": self.n_binary,
            "avg_skewness": self.avg_skewness,
            "avg_kurtosis": self.avg_kurtosis,
            "missing_ratio": self.missing_ratio,
            "feature_to_sample_ratio": self.feature_to_sample_ratio,
            "avg_correlation": self.avg_correlation,
            "dimensionality_score": self.dimensionality_score,
            "is_imbalanced": self.is_imbalanced,
            "is_high_dimensional": self.is_high_dimensional,
            "is_sparse": self.is_sparse,
            "index_is_datetime": self.index_is_datetime,
            "index_is_sorted": self.index_is_sorted,
            "metadata": self.metadata
        }


@dataclass
class ModelScore:
    """
    Model scoring result.
    
    Attributes:
        model_name: Model name
        model_category: Model category
        performance_score: Performance score (0-1)
        speed_score: Speed score (0-1)
        interpretability_score: Interpretability score (0-1)
        robustness_score: Robustness score (0-1)
        scalability_score: Scalability score (0-1)
        memory_score: Memory score (0-1)
        total_score: Weighted total score (0-1)
        recommended: Recommendation flag
        reasoning: Reasoning for score
        warnings: Warnings/caveats
    """
    model_name: str
    model_category: ModelCategory
    
    performance_score: float
    speed_score: float
    interpretability_score: float
    robustness_score: float
    scalability_score: float
    memory_score: float
    
    total_score: float
    recommended: bool
    reasoning: List[str]
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_category": self.model_category.value,
            "performance_score": self.performance_score,
            "speed_score": self.speed_score,
            "interpretability_score": self.interpretability_score,
            "robustness_score": self.robustness_score,
            "scalability_score": self.scalability_score,
            "memory_score": self.memory_score,
            "total_score": self.total_score,
            "recommended": self.recommended,
            "reasoning": self.reasoning,
            "warnings": self.warnings
        }


@dataclass
class SelectionResult:
    """
    Model selection result.
    
    Attributes:
        recommended_models: Top recommended models
        model_scores: All model scores
        dataset_characteristics: Dataset meta-features
        selection_criterion: Selection criterion used
        problem_type: Detected problem type
        selection_time: Time taken (seconds)
        total_models_evaluated: Total models evaluated
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    recommended_models: List[str]
    model_scores: List[ModelScore]
    dataset_characteristics: DatasetCharacteristics
    selection_criterion: SelectionCriterion
    problem_type: ProblemType
    
    selection_time: float
    total_models_evaluated: int
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = __version__
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommended_models": self.recommended_models,
            "model_scores": [s.to_dict() for s in self.model_scores],
            "dataset_characteristics": self.dataset_characteristics.to_dict(),
            "selection_criterion": self.selection_criterion.value,
            "problem_type": self.problem_type.value,
            "selection_time": self.selection_time,
            "total_models_evaluated": self.total_models_evaluated,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "version": self.version
        }
    
    def get_top_models(self, n: int = 5) -> List[ModelScore]:
        """Get top n models by score."""
        return sorted(self.model_scores, key=lambda x: x.total_score, reverse=True)[:n]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str):
    """Decorator for operation timing."""
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
    """Decorator for safe operations with fallback."""
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
# SECTION: Model Registry
# ═══════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Central registry of model characteristics.
    
    Maps model names to their properties for intelligent selection.
    """
    
    MODELS: Dict[str, Dict[str, Any]] = {
        # Linear Models
        "logistic_regression": {
            "category": ModelCategory.LINEAR,
            "interpretability": 5,
            "speed": 5,
            "memory": 5,
            "scalability": 5,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": True,
            "best_for": ["small_data", "interpretable", "linear_relationships"],
            "not_recommended_for": ["non_linear", "very_large_data"]
        },
        "ridge": {
            "category": ModelCategory.LINEAR,
            "interpretability": 5,
            "speed": 5,
            "memory": 5,
            "scalability": 5,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": False,
            "best_for": ["small_data", "regularization"],
            "not_recommended_for": ["non_linear"]
        },
        "lasso": {
            "category": ModelCategory.LINEAR,
            "interpretability": 5,
            "speed": 5,
            "memory": 5,
            "scalability": 5,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": False,
            "best_for": ["feature_selection", "sparse_solutions"],
            "not_recommended_for": ["non_linear", "high_correlation"]
        },
        
        # Tree-Based Models
        "decision_tree": {
            "category": ModelCategory.TREE_BASED,
            "interpretability": 4,
            "speed": 4,
            "memory": 3,
            "scalability": 3,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": False,
            "best_for": ["interpretable", "non_linear", "mixed_features"],
            "not_recommended_for": ["high_variance"]
        },
        "random_forest": {
            "category": ModelCategory.ENSEMBLE,
            "interpretability": 3,
            "speed": 3,
            "memory": 2,
            "scalability": 3,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["non_linear", "robust", "general_purpose", "feature_importance"],
            "not_recommended_for": ["very_large_data", "low_latency"]
        },
        "extra_trees": {
            "category": ModelCategory.ENSEMBLE,
            "interpretability": 3,
            "speed": 4,
            "memory": 2,
            "scalability": 3,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["non_linear", "fast_training", "variance_reduction"],
            "not_recommended_for": ["very_large_data", "high_latency_sensitive"]
        },
        
        # Boosting Models
        "xgboost": {
            "category": ModelCategory.BOOSTING,
            "interpretability": 2,
            "speed": 3,
            "memory": 3,
            "scalability": 4,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["high_performance", "structured_data"],
            "not_recommended_for": ["need_interpretability"]
        },
        "lightgbm": {
            "category": ModelCategory.BOOSTING,
            "interpretability": 2,
            "speed": 5,
            "memory": 4,
            "scalability": 5,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["large_data", "fast_training"],
            "not_recommended_for": ["tiny_data"]
        },
        "catboost": {
            "category": ModelCategory.BOOSTING,
            "interpretability": 2,
            "speed": 3,
            "memory": 3,
            "scalability": 4,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["categorical_heavy", "minimal_tuning"],
            "not_recommended_for": ["tiny_data"]
        },
        "gradient_boosting": {
            "category": ModelCategory.BOOSTING,
            "interpretability": 2,
            "speed": 2,
            "memory": 3,
            "scalability": 3,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": True,
            "best_for": ["controlled_tuning", "high_performance"],
            "not_recommended_for": ["very_large_data", "fast_training"]
        },
        
        # Neural Networks
        "mlp": {
            "category": ModelCategory.NEURAL_NETWORK,
            "interpretability": 1,
            "speed": 2,
            "memory": 2,
            "scalability": 3,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": False,
            "best_for": ["non_linear", "complex_patterns"],
            "not_recommended_for": ["small_data", "fast_training"]
        },
        
        # Other Models
        "svm": {
            "category": ModelCategory.SVM,
            "interpretability": 2,
            "speed": 2,
            "memory": 2,
            "scalability": 2,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": False,
            "best_for": ["small_data", "high_dimensional"],
            "not_recommended_for": ["very_large_data"]
        },
        "knn": {
            "category": ModelCategory.KNN,
            "interpretability": 4,
            "speed": 2,
            "memory": 1,
            "scalability": 1,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": True,
            "best_for": ["small_data", "local_patterns"],
            "not_recommended_for": ["large_data", "fast_prediction"]
        },
        "naive_bayes": {
            "category": ModelCategory.NAIVE_BAYES,
            "interpretability": 4,
            "speed": 5,
            "memory": 5,
            "scalability": 5,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": False,
            "best_for": ["text", "small_data", "fast_training"],
            "not_recommended_for": ["strong_correlations", "complex_patterns"]
        },
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        return cls.MODELS.get(model_name.lower())
    
    @classmethod
    def get_all_models(cls) -> List[str]:
        """Get all registered models."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_models_by_category(cls, category: ModelCategory) -> List[str]:
        """Get models by category."""
        return [m for m, info in cls.MODELS.items() if info["category"] == category]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Estimator Factory
# ═══════════════════════════════════════════════════════════════════════════

class EstimatorFactory:
    """Factory for creating sklearn estimators with sensible defaults."""
    
    @staticmethod
    def make(
        model_name: str,
        problem: ProblemType,
        quick: bool = False,
        random_state: int = 42
    ) -> Optional[BaseEstimator]:
        """
        Create estimator instance.
        
        Args:
            model_name: Model name
            problem: Problem type
            quick: Use lighter parameters
            random_state: Random seed
        
        Returns:
            Configured estimator or None
        """
        rs = random_state
        mn = model_name.lower()
        
        # ─── Regression Models ───
        if problem == ProblemType.REGRESSION:
            if mn == "ridge":
                return Ridge(alpha=1.0)
            
            if mn == "lasso":
                return Lasso(alpha=0.001, max_iter=5000)
            
            if mn == "random_forest":
                return RandomForestRegressor(
                    n_estimators=200 if not quick else 80,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=rs
                )
            
            if mn == "extra_trees":
                return ExtraTreesRegressor(
                    n_estimators=300 if not quick else 100,
                    n_jobs=-1,
                    random_state=rs
                )
            
            if mn == "gradient_boosting":
                return GradientBoostingRegressor(random_state=rs)
            
            if mn == "xgboost" and xgb is not None:
                return xgb.XGBRegressor(
                    n_estimators=600 if not quick else 200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=rs,
                    tree_method="hist"
                )
            
            if mn == "lightgbm" and lgb is not None:
                return lgb.LGBMRegressor(
                    n_estimators=1000 if not quick else 300,
                    learning_rate=0.05,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=rs,
                    n_jobs=-1,
                    verbose=-1
                )
            
            if mn == "catboost" and CatBoostRegressor is not None:
                return CatBoostRegressor(
                    depth=8,
                    iterations=1200 if not quick else 400,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    random_state=rs,
                    verbose=False
                )
            
            if mn == "mlp":
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=(128, 64) if not quick else (64,),
                        max_iter=400 if not quick else 200,
                        random_state=rs
                    ))
                ])
            
            if mn == "svm":
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("svr", SVR(C=1.0, epsilon=0.1))
                ])
        
        # ─── Classification Models ───
        else:
            if mn == "logistic_regression":
                return Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("lr", LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=rs
                    ))
                ])
            
            if mn == "decision_tree":
                return DecisionTreeClassifier(
                    max_depth=None,
                    random_state=rs,
                    class_weight="balanced"
                )
            
            if mn == "random_forest":
                return RandomForestClassifier(
                    n_estimators=300 if not quick else 120,
                    n_jobs=-1,
                    random_state=rs,
                    class_weight="balanced"
                )
            
            if mn == "extra_trees":
                return ExtraTreesClassifier(
                    n_estimators=400 if not quick else 150,
                    n_jobs=-1,
                    random_state=rs,
                    class_weight="balanced"
                )
            
            if mn == "gradient_boosting":
                return GradientBoostingClassifier(random_state=rs)
            
            if mn == "xgboost" and xgb is not None:
                return xgb.XGBClassifier(
                    n_estimators=700 if not quick else 250,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    n_jobs=-1,
                    random_state=rs,
                    tree_method="hist",
                    eval_metric="logloss"
                )
            
            if mn == "lightgbm" and lgb is not None:
                return lgb.LGBMClassifier(
                    n_estimators=1200 if not quick else 400,
                    learning_rate=0.05,
                    num_leaves=63,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=rs,
                    verbose=-1
                )
            
            if mn == "catboost" and CatBoostClassifier is not None:
                return CatBoostClassifier(
                    iterations=1200 if not quick else 400,
                    learning_rate=0.05,
                    depth=8,
                    random_state=rs,
                    verbose=False,
                    auto_class_weights="Balanced"
                )
            
            if mn == "mlp":
                return Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("mlp", MLPClassifier(
                        hidden_layer_sizes=(128, 64) if not quick else (64,),
                        max_iter=400 if not quick else 200,
                        random_state=rs
                    ))
                ])
            
            if mn == "svm":
                return Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("svc", SVC(
                        C=1.0,
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=rs
                    ))
                ])
            
            if mn == "knn":
                return Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("knn", KNeighborsClassifier(n_neighbors=5))
                ])
            
            if mn == "naive_bayes":
                return GaussianNB()
        
        return None
python# agents/ml/model_selector.py (continued)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Model Selector
# ═══════════════════════════════════════════════════════════════════════════

class ModelSelector:
    """
    **ModelSelector** — Intelligent model selection with meta-learning.
    
    Responsibilities:
      1. Extract dataset meta-features
      2. Detect problem type automatically
      3. Filter applicable models
      4. Score models across 6 dimensions
      5. Apply penalties for data characteristics
      6. Optional quick CV benchmark
      7. Multi-objective optimization
      8. Generate recommendations
      9. Provide explanations
      10. Export results
    
    Features:
      • 7 selection criteria
      • 13+ model support
      • Meta-feature extraction
      • Performance prediction
      • Quick CV validation
      • Imbalance detection
      • Time series hints
      • Comprehensive scoring
    """
    
    def __init__(
        self,
        criterion: SelectionCriterion = SelectionCriterion.BALANCED,
        random_state: int = 42
    ) -> None:
        """
        Initialize model selector.
        
        Args:
            criterion: Selection criterion
            random_state: Random seed
        """
        self.criterion = criterion
        self.result: Optional[SelectionResult] = None
        self.random_state = int(random_state)
        self.rng = check_random_state(self.random_state)
        self.criterion_weights = self._get_criterion_weights()
        self._log = logger.bind(agent="ModelSelector")
        
        self._log.info(f"✓ ModelSelector initialized | criterion={self.criterion.value}")
    
    # ───────────────────────────────────────────────────────────────────
    # Criterion Weights
    # ───────────────────────────────────────────────────────────────────
    
    def _get_criterion_weights(self) -> Dict[str, float]:
        """Get weights for selection criterion."""
        weights = {
            SelectionCriterion.ACCURACY: {
                "performance": 0.70,
                "speed": 0.10,
                "interpretability": 0.05,
                "robustness": 0.10,
                "scalability": 0.025,
                "memory": 0.025
            },
            SelectionCriterion.SPEED: {
                "performance": 0.30,
                "speed": 0.40,
                "interpretability": 0.05,
                "robustness": 0.10,
                "scalability": 0.10,
                "memory": 0.05
            },
            SelectionCriterion.INTERPRETABILITY: {
                "performance": 0.20,
                "speed": 0.10,
                "interpretability": 0.50,
                "robustness": 0.10,
                "scalability": 0.05,
                "memory": 0.05
            },
            SelectionCriterion.ROBUSTNESS: {
                "performance": 0.30,
                "speed": 0.05,
                "interpretability": 0.05,
                "robustness": 0.40,
                "scalability": 0.10,
                "memory": 0.10
            },
            SelectionCriterion.SCALABILITY: {
                "performance": 0.25,
                "speed": 0.15,
                "interpretability": 0.05,
                "robustness": 0.10,
                "scalability": 0.35,
                "memory": 0.10
            },
            SelectionCriterion.MEMORY: {
                "performance": 0.25,
                "speed": 0.10,
                "interpretability": 0.05,
                "robustness": 0.10,
                "scalability": 0.10,
                "memory": 0.40
            },
            SelectionCriterion.BALANCED: {
                "performance": 0.30,
                "speed": 0.15,
                "interpretability": 0.15,
                "robustness": 0.15,
                "scalability": 0.15,
                "memory": 0.10
            }
        }
        return weights[self.criterion]
    
    # ───────────────────────────────────────────────────────────────────
    # Main Selection API
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("model_selection")
    def select_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: Optional[str] = None,
        n_models: int = 5,
        quick_mode: bool = False,
        do_quick_cv: bool = False,
        max_rows_for_cv: int = 25000
    ) -> SelectionResult:
        """
        Select best models for dataset.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            problem_type: 'classification' or 'regression' (auto-detect if None)
            n_models: Number of models to recommend
            quick_mode: Use lighter model parameters
            do_quick_cv: Perform quick CV benchmark
            max_rows_for_cv: Max rows for CV sampling
        
        Returns:
            SelectionResult with recommendations
        """
        start = datetime.now()
        
        self._log.info(
            f"Starting model selection | "
            f"rows={len(X)} | "
            f"cols={len(X.columns)} | "
            f"criterion={self.criterion.value}"
        )
        
        # ─── Step 1: Extract Meta-Features ───
        characteristics = self._extract_characteristics(X, y)
        
        # ─── Step 2: Detect Problem Type ───
        if problem_type is None:
            problem_type = self._detect_problem_type(y, characteristics)
        
        problem_enum = self._map_problem_type(problem_type, characteristics)
        self._log.info(f"Detected problem: {problem_enum.value}")
        
        # ─── Step 3: Get Applicable Models ───
        applicable = self._get_applicable_models(problem_type)
        self._log.info(f"Applicable models: {len(applicable)}")
        
        # ─── Step 4: Quick CV Benchmark (Optional) ───
        cv_perf: Dict[str, float] = {}
        if do_quick_cv and len(X) > 2:
            X_cv, y_cv = self._maybe_sample_for_cv(X, y, max_rows_for_cv)
            cv_perf = self._quick_cv_benchmark(
                applicable, X_cv, y_cv, problem_enum, quick_mode
            )
        
        # ─── Step 5: Score Models ───
        model_scores: List[ModelScore] = []
        for model_name in applicable:
            sc = self._score_model(
                model_name=model_name,
                characteristics=characteristics,
                problem_type=problem_enum,
                quick_mode=quick_mode,
                cv_perf=cv_perf.get(model_name)
            )
            model_scores.append(sc)
        
        # ─── Step 6: Rank and Select ───
        model_scores.sort(key=lambda s: s.total_score, reverse=True)
        recommended = [s.model_name for s in model_scores[:n_models]]
        
        # ─── Step 7: Create Result ───
        selection_time = (datetime.now() - start).total_seconds()
        meta = {
            "quick_mode": quick_mode,
            "do_quick_cv": do_quick_cv,
            "cv_rows": int(len(X) if not do_quick_cv else min(len(X), max_rows_for_cv)),
            "random_state": self.random_state
        }
        
        self.result = SelectionResult(
            recommended_models=recommended,
            model_scores=model_scores,
            dataset_characteristics=characteristics,
            selection_criterion=self.criterion,
            problem_type=problem_enum,
            selection_time=selection_time,
            total_models_evaluated=len(applicable),
            metadata=meta
        )
        
        self._log.success(
            f"✓ Selection complete | "
            f"time={selection_time:.2f}s | "
            f"top_3={recommended[:3]}"
        )
        
        return self.result
    
    # ───────────────────────────────────────────────────────────────────
    # Meta-Feature Extraction
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("extract_characteristics")
    @_safe_operation("extract_characteristics", default_value=None)
    def _extract_characteristics(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> DatasetCharacteristics:
        """Extract dataset meta-features."""
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        
        # ─── Feature Types ───
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
        n_numerical = int(len(num_cols))
        n_categorical = int(len(cat_cols))
        
        # Binary features
        n_binary = int(sum(
            (pd.Series(X[c]).nunique(dropna=True) == 2)
            for c in X.columns
        ))
        
        # ─── Target Analysis ───
        n_unique_y = int(pd.Series(y).nunique(dropna=True))
        if (y.dtype == "object") or (n_unique_y <= 20 and not pd.api.types.is_float_dtype(y)):
            n_classes = n_unique_y
            cb = pd.Series(y).value_counts(normalize=True, dropna=False).to_dict()
        else:
            n_classes, cb = None, None
        
        # ─── Statistical Properties ───
        if n_numerical > 0:
            skews, kurts = [], []
            for c in num_cols:
                s = pd.to_numeric(X[c], errors="coerce").dropna()
                if len(s) >= 10:
                    try:
                        skews.append(float(abs(skew(s))))
                        kurts.append(float(abs(kurtosis(s))))
                    except Exception:
                        pass
            avg_skewness = float(np.mean(skews)) if skews else 0.0
            avg_kurtosis = float(np.mean(kurts)) if kurts else 0.0
        else:
            avg_skewness = 0.0
            avg_kurtosis = 0.0
        
        # ─── Missing & Sparsity ───
        total_cells = max(1, n_samples * n_features)
        missing_ratio = float(pd.isna(X).sum().sum() / total_cells)
        
        zeros_ratio = 0.0
        try:
            Z = X.select_dtypes(include=[np.number]).eq(0).sum().sum()
            N = max(1, len(X.select_dtypes(include=[np.number]).values.flatten()))
            zeros_ratio = float(Z / N) if N > 0 else 0.0
        except Exception:
            pass
        
        is_sparse = bool((missing_ratio > 0.3) or (zeros_ratio > 0.5))
        
        # ─── Correlation ───
        if n_numerical > 1:
            try:
                cm = X[num_cols].corr(numeric_only=True).abs()
                upper = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
                avg_corr = float(upper.stack().mean())
            except Exception:
                avg_corr = 0.0
        else:
            avg_corr = 0.0
        
        # ─── Complexity Measures ───
        f2s = float(n_features / max(1, n_samples))
        dimensionality_score = float(
            np.log10(max(2, n_features)) / np.log10(max(3, n_samples))
        )
        
        # ─── Imbalance Detection ───
        is_imbalanced = False
        if cb:
            try:
                is_imbalanced = (min(cb.values()) < 0.2)
            except Exception:
                is_imbalanced = False
        
        is_high_dimensional = bool(f2s > 0.1)
        
        # ─── Time Series Heuristics ───
        idx = getattr(X, "index", None)
        is_dt, is_sorted = False, False
        try:
            if isinstance(idx, pd.DatetimeIndex):
                is_dt = True
                is_sorted = bool(idx.is_monotonic_increasing)
        except Exception:
            pass
        
        return DatasetCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_balance=cb,
            n_numerical=n_numerical,
            n_categorical=n_categorical,
            n_binary=n_binary,
            avg_skewness=avg_skewness,
            avg_kurtosis=avg_kurtosis,
            missing_ratio=missing_ratio,
            feature_to_sample_ratio=f2s,
            avg_correlation=avg_corr,
            dimensionality_score=dimensionality_score,
            is_imbalanced=is_imbalanced,
            is_high_dimensional=is_high_dimensional,
            is_sparse=is_sparse,
            index_is_datetime=is_dt,
            index_is_sorted=is_sorted,
            metadata={"zeros_ratio": zeros_ratio}
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Problem Type Detection
    # ───────────────────────────────────────────────────────────────────
    
    def _detect_problem_type(self, y: pd.Series, ch: DatasetCharacteristics) -> str:
        """Auto-detect problem type from target."""
        n_unique = int(pd.Series(y).nunique(dropna=True))
        
        if pd.api.types.is_numeric_dtype(y):
            return "classification" if n_unique <= 20 else "regression"
        
        return "classification"
    
    def _map_problem_type(
        self,
        kind: str,
        ch: DatasetCharacteristics
    ) -> ProblemType:
        """Map problem type string to enum with characteristics."""
        if kind == "regression":
            return ProblemType.REGRESSION
        
        # Classification
        if ch.n_classes == 2:
            if ch.is_imbalanced:
                return ProblemType.IMBALANCED
            return ProblemType.BINARY_CLASSIFICATION
        
        if ch.is_high_dimensional:
            return ProblemType.HIGH_DIMENSIONAL
        
        return ProblemType.MULTICLASS_CLASSIFICATION
    
    # ───────────────────────────────────────────────────────────────────
    # Applicable Models
    # ───────────────────────────────────────────────────────────────────
    
    def _get_applicable_models(self, problem_type: str) -> List[str]:
        """Get list of applicable models for problem type."""
        if problem_type == "regression":
            models = [
                "ridge", "lasso", "random_forest", "extra_trees",
                "xgboost", "lightgbm", "catboost", "gradient_boosting",
                "mlp", "svm"
            ]
        else:
            models = [
                "logistic_regression", "decision_tree", "random_forest",
                "extra_trees", "xgboost", "lightgbm", "catboost",
                "gradient_boosting", "mlp", "svm", "knn", "naive_bayes"
            ]
        
        # Filter unavailable boosting libraries
        filtered: List[str] = []
        for m in models:
            if m == "xgboost" and xgb is None:
                continue
            if m == "lightgbm" and lgb is None:
                continue
            if m == "catboost" and (CatBoostClassifier is None and CatBoostRegressor is None):
                continue
            filtered.append(m)
        
        return filtered
    
    # ───────────────────────────────────────────────────────────────────
    # Quick CV Benchmark
    # ───────────────────────────────────────────────────────────────────
    
    def _maybe_sample_for_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_rows: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Sample data for CV if too large."""
        if len(X) <= max_rows:
            return X, y
        
        # Stratified sampling for classification
        try:
            if pd.Series(y).nunique(dropna=True) <= 20:
                df = X.copy()
                df["_y_"] = y.values
                n_classes = df["_y_"].nunique()
                per_class = max_rows // max(1, n_classes)
                
                sample = df.groupby("_y_", group_keys=False).apply(
                    lambda g: g.sample(
                        min(len(g), per_class),
                        random_state=self.random_state
                    )
                )
                y_s = sample.pop("_y_")
                return sample, y_s
        except Exception:
            pass
        
        # Simple random sampling
        sampled = X.sample(max_rows, random_state=self.random_state)
        return sampled, y.loc[sampled.index]
    
    @_timeit("quick_cv_benchmark")
    @_safe_operation("quick_cv_benchmark", default_value={})
    def _quick_cv_benchmark(
        self,
        models: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        problem: ProblemType,
        quick_mode: bool
    ) -> Dict[str, float]:
        """Quick CV benchmark to calibrate performance scores."""
        cv_perf: Dict[str, float] = {}
        
        if len(X) < 40:
            return cv_perf
        
        # Setup CV
        if problem == ProblemType.REGRESSION:
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scoring_name = "r2"
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            n_classes = pd.Series(y).nunique(dropna=True)
            scoring_name = "roc_auc_ovr" if n_classes > 2 else "roc_auc"
        
        # Benchmark each model
        for m in models:
            est = EstimatorFactory.make(m, problem, quick=quick_mode, random_state=self.random_state)
            if est is None:
                continue
            
            try:
                scores = cross_val_score(
                    est, X, y,
                    cv=cv,
                    scoring=scoring_name,
                    n_jobs=-1
                )
                s = float(np.clip(np.mean(scores), -1, 1))
                
                # Normalize R² to [0, 1]
                if scoring_name == "r2":
                    s = (s + 1) / 2.0
                
                cv_perf[m] = s
            except Exception as e:
                self._log.debug(f"Quick CV failed for {m}: {e}")
        
        return cv_perf
    
    # ───────────────────────────────────────────────────────────────────
    # Model Scoring
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("score_model", default_value=None)
    def _score_model(
        self,
        model_name: str,
        characteristics: DatasetCharacteristics,
        problem_type: ProblemType,
        quick_mode: bool,
        cv_perf: Optional[float] = None
    ) -> ModelScore:
        """Score a model across multiple dimensions."""
        info = ModelRegistry.get_model_info(model_name) or {}
        category = info.get("category", ModelCategory.LINEAR)
        
        # ─── Base Scores from Registry ───
        interpretability = float(info.get("interpretability", 3)) / 5.0
        speed = float(info.get("speed", 3)) / 5.0
        memory = float(info.get("memory", 3)) / 5.0
        scalability = float(info.get("scalability", 3)) / 5.0
        
        # ─── Performance Score ───
        perf = self._estimate_performance(
            model_name, info, characteristics, problem_type
        )
        
        # Calibrate with CV if available
        if cv_perf is not None:
            perf = float(np.clip(0.7 * perf + 0.3 * cv_perf, 0, 1))
        
        # ─── Robustness Score ───
        robust = self._estimate_robustness(model_name, info, characteristics)
        
        # ─── Adjustments & Reasoning ───
        warnings: List[str] = []
        reasoning: List[str] = []
        
        # High-dimensional adjustment
        if characteristics.is_high_dimensional:
            if "high_dimensional" in info.get("best_for", []):
                perf += 0.05
                reasoning.append("✓ Well-suited for high-dimensional data")
            elif "high_dimensional" in info.get("not_recommended_for", []):
                perf *= 0.85
                warnings.append("⚠️ May struggle with high-dimensional data")
        
        # Imbalance adjustment
        if characteristics.is_imbalanced:
            if info.get("handles_imbalance", False):
                perf += 0.03
                robust += 0.05
                reasoning.append("✓ Handles imbalanced classes")
            else:
                perf *= 0.93
                warnings.append("⚠️ No native imbalance handling")
        
        # Categorical features
        if characteristics.n_categorical > 0 and info.get("handles_categorical", False):
            reasoning.append("✓ Handles categorical features")
        
        # Missing data
        if characteristics.missing_ratio > 0.1:
            if info.get("handles_missing", False):
                reasoning.append("✓ Robust to missing data")
            else:
                perf *= 0.95
                warnings.append("⚠️ Requires imputation (>10% missing)")
        
        # Correlation
        if characteristics.avg_correlation > 0.8 and model_name in {"lasso", "ridge"}:
            perf += 0.03
            reasoning.append("✓ Regularization helps with multicollinearity")
        
        # Skewness & kurtosis
        if (characteristics.avg_skewness > 2 or characteristics.avg_kurtosis > 3):
            if category in {ModelCategory.TREE_BASED, ModelCategory.ENSEMBLE, ModelCategory.BOOSTING}:
                robust += 0.05
                reasoning.append("✓ Robust to skewed/heavy-tailed distributions")
        
        # Size-based warnings
        for flag in info.get("not_recommended_for", []):
            if flag == "very_large_data" and characteristics.n_samples > 500_000:
                perf *= 0.9
                warnings.append("⚠️ May be slow on very large datasets")
            if flag == "small_data" and characteristics.n_samples < 1000:
                perf *= 0.9
                warnings.append("⚠️ May underperform on small datasets")
        
        # ─── Total Score ───
        w = self.criterion_weights
        total = float(
            w["performance"] * perf +
            w["speed"] * speed +
            w["interpretability"] * interpretability +
            w["robustness"] * robust +
            w["scalability"] * scalability +
            w["memory"] * memory
        )
        
        # ─── Recommendation ───
        recommended = bool(total > 0.60 and len(warnings) < 3)
        
        return ModelScore(
            model_name=model_name,
            model_category=category,
            performance_score=float(np.clip(perf, 0, 1)),
            speed_score=float(np.clip(speed, 0, 1)),
            interpretability_score=float(np.clip(interpretability, 0, 1)),
            robustness_score=float(np.clip(robust, 0, 1)),
            scalability_score=float(np.clip(scalability, 0, 1)),
            memory_score=float(np.clip(memory, 0, 1)),
            total_score=float(np.clip(total, 0, 1)),
            recommended=recommended,
            reasoning=reasoning,
            warnings=warnings
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Performance Estimation
    # ───────────────────────────────────────────────────────────────────
    
    def _estimate_performance(
        self,
        model_name: str,
        info: Dict[str, Any],
        ch: DatasetCharacteristics,
        problem_type: ProblemType
    ) -> float:
        """Estimate model performance based on heuristics."""
        base = 0.60
        
        # Dataset size
        if ch.n_samples < 1000 and "small_data" in info.get("best_for", []):
            base += 0.12
        if ch.n_samples > 100_000 and "large_data" in info.get("best_for", []):
            base += 0.10
        
        # High-dimensional & categorical
        if ch.is_high_dimensional and "high_dimensional" in info.get("best_for", []):
            base += 0.08
        if (ch.n_categorical > ch.n_numerical / 2) and ("categorical_heavy" in info.get("best_for", [])):
            base += 0.08
        
        # Ensemble/boosting bonus
        if info.get("category") in {ModelCategory.BOOSTING, ModelCategory.ENSEMBLE}:
            base += 0.06
        
        # Penalties
        if ch.n_samples < 1000 and "small_data" in info.get("not_recommended_for", []):
            base -= 0.10
        if ch.is_high_dimensional and "high_dimensional" in info.get("not_recommended_for", []):
            base -= 0.08
        
        # Very high dimensionality
        if ch.feature_to_sample_ratio > 0.5 and model_name in {"knn", "mlp", "svm"}:
            base -= 0.08
        
        # Imbalance bonus
        if ch.is_imbalanced and (info.get("handles_imbalance", False) or 
                                  info.get("category") == ModelCategory.BOOSTING):
            base += 0.04
        
        return float(np.clip(base, 0, 1))
    
    # ───────────────────────────────────────────────────────────────────
    # Robustness Estimation
    # ───────────────────────────────────────────────────────────────────
    
    def _estimate_robustness(
        self,
        model_name: str,
        info: Dict[str, Any],
        ch: DatasetCharacteristics
    ) -> float:
        """Estimate model robustness."""
        base = 0.50
        
        if info.get("category") in {ModelCategory.ENSEMBLE, ModelCategory.BOOSTING}:
            base += 0.25
        
        if ch.missing_ratio > 0.10:
            base += 0.05 if info.get("handles_missing", False) else -0.05
        
        if ch.avg_skewness > 2 or ch.avg_kurtosis > 3:
            if info.get("category") in {
                ModelCategory.TREE_BASED,
                ModelCategory.ENSEMBLE,
                ModelCategory.BOOSTING
            }:
                base += 0.05
        
        return float(np.clip(base, 0, 1))
    
    # ───────────────────────────────────────────────────────────────────
    # Explanation & Export
    # ───────────────────────────────────────────────────────────────────
    
    def get_explanation(self, model_name: Optional[str] = None) -> str:
        """Get explanation of selection results."""
        if self.result is None:
            return "No selection result available. Run select_models() first."
        
        if model_name:
            s = next((x for x in self.result.model_scores if x.model_name == model_name), None)
            if not s:
                return f"Model {model_name} not found in results."
            
            return (
                f"\n🤖 Model: {s.model_name.upper()}\n"
                f"Category: {s.model_category.value}\n\n"
                f"📊 Scores:\n"
                f"  • Performance: {s.performance_score:.2f}\n"
                f"  • Speed: {s.speed_score:.2f}\n"
                f"  • Interpretability: {s.interpretability_score:.2f}\n"
                f"  • Robustness: {s.robustness_score:.2f}\n"
                f"  • Scalability: {s.scalability_score:.2f}\n"
                f"  • Memory: {s.memory_score:.2f}\n\n"
                f"  ⭐ Total Score: {s.total_score:.2f}\n\n"
                f"✅ Strengths:\n" + 
                ("\n".join(f"  {r}" for r in s.reasoning) if s.reasoning else "  —") + "\n" +
                (("⚠️ Caveats:\n" + "\n".join(f"  {w}" for w in s.warnings)) if s.warnings else "") + "\n" +
                ("✅ RECOMMENDED" if s.recommended else "❌ NOT RECOMMENDED")
            )
        
        # Overall summary
        ds = self.result.dataset_characteristics
        top = self.result.recommended_models[:5]
        out = [
            "🎯 MODEL SELECTION RESULTS",
            "",
            "📊 Dataset:",
            f"  • Samples: {ds.n_samples:,}",
            f"  • Features: {ds.n_features}",
            f"  • Problem: {self.result.problem_type.value}",
            f"  • Missing: {ds.missing_ratio*100:.1f}%",
            f"  • F2S Ratio: {ds.feature_to_sample_ratio:.3f}",
            f"  • High-Dim: {ds.is_high_dimensional}",
            "",
            "🏆 Top Recommended Models:"
        ]
        
        for i, m in enumerate(top, 1):
            s = next(s for s in self.result.model_scores if s.model_name == m)
            out.append(f"  {i}. {m.upper()} (score: {s.total_score:.2f})")
    return "\n".join(out)

def export_report(self, filepath: Union[str, Path]) -> None:
    """Export selection results to JSON."""
    if self.result is None:
        raise ValueError("No result to export. Run select_models() first.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(self.result.to_dict(), f, indent=2, ensure_ascii=False)

    self._log.info(f"✓ Report exported to {path}")


# ───────────────────────────────────────────────────────────────────
# SECTION: Convenience Function
# ───────────────────────────────────────────────────────────────────

def select_best_models(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: "SelectionCriterion" = SelectionCriterion.BALANCED,
    n_models: int = 5,
    **kwargs: Any,
) -> "SelectionResult":
    """
    Convenience function for model selection.

    Example:
        result = select_best_models(
            X=X_train,
            y=y_train,
            criterion=SelectionCriterion.BALANCED,
            n_models=5,
            metric="f1_weighted",
            cv_folds=5,
        )
        # Export:
        # selector.export_report("reports/model_selection.json")

    Args:
        X: Feature matrix.
        y: Target vector/series.
        criterion: Selection strategy/criterion to balance accuracy, robustness, and speed.
        n_models: Number of top models to return.
        **kwargs: Extra configuration for the selector (e.g., metric, cv_folds, time_budget).

    Returns:
        SelectionResult with ranked candidates, best model, and diagnostics.
    """
    # Create selector with provided configuration
    selector = ModelSelector(
        criterion=criterion,
        n_models=n_models,
        **kwargs,
    )
    # Run selection
    return selector.select_models(X, y)
