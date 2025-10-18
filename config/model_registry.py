# config/model_registry.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Model Registry v7.0              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE ML MODEL REGISTRY & SELECTION SYSTEM                         ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Comprehensive Model Catalog                                           ║
║  ✓ Smart Selection Strategies                                            ║
║  ✓ Dependency Management                                                 ║
║  ✓ Model Metadata & Documentation                                        ║
║  ✓ Validation & Safety Checks                                            ║
║  ✓ Immutable Registries                                                  ║
║  ✓ Caching & Performance                                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Model Registry Structure:
```
    Registry
    ├── Classification Models
    │   ├── Linear (LR, SVM)
    │   ├── Tree-Based (DT, RF, ET)
    │   ├── Boosting (GBC, XGBoost, LightGBM, CatBoost)
    │   └── Others (KNN, NB)
    └── Regression Models
        ├── Linear (LR, Ridge, Lasso, EN)
        ├── Tree-Based (DT, RF, ET)
        └── Boosting (GBR, XGBoost, LightGBM, CatBoost)
    
    Selection Strategies:
    ├── fast          → Quick prototyping
    ├── balanced      → Speed vs accuracy
    ├── accurate      → Maximum quality
    ├── production    → Stable defaults
    ├── interpretable → Explainable models
    ├── all           → All models
    └── all_available → Filter by dependencies
```

Features:
    Model Management:
        • Comprehensive model catalog
        • Detailed metadata (pros, cons, best_for)
        • Category classification
        • Dependency tracking
    
    Selection Strategies:
        • Fast: Quick prototyping
        • Balanced: Speed vs accuracy trade-off
        • Accurate: Maximum quality
        • Production: Stable, robust defaults
        • Interpretable: Explainable models
        • All: Complete catalog
    
    Dependency Management:
        • Automatic dependency detection
        • Cached import checks (LRU)
        • Graceful fallbacks
        • Optional package support
    
    Safety:
        • Immutable registries (MappingProxyType)
        • Validation at import time
        • Type safety
        • Failsafe defaults

Usage:
```python
    from config.model_registry import (
        get_models_for_problem,
        get_model_info,
        ProblemType
    )
    
    # Get models for classification
    models = get_models_for_problem(
        ProblemType.CLASSIFICATION,
        strategy="accurate",
        only_available=True
    )
    
    # Get model information
    info = get_model_info("xgboost", ProblemType.CLASSIFICATION)
    print(info["name"])        # "XGBoost"
    print(info["pros"])        # ["State-of-the-art", ...]
    print(info["best_for"])    # ["Structured data", ...]
    
    # List available strategies
    from config.model_registry import list_strategies
    strategies = list_strategies()
    
    # Check if model supported
    from config.model_registry import is_model_supported
    supported = is_model_supported("xgboost", ProblemType.CLASSIFICATION)
```

Dependencies:
    • None (pure Python)
    • Optional: xgboost, lightgbm, catboost
"""

from __future__ import annotations

import importlib.util
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = [
    "ProblemType",
    "ModelCategory",
    "CLASSIFICATION_MODELS",
    "REGRESSION_MODELS",
    "MODEL_SELECTION_STRATEGIES",
    "get_models_for_problem",
    "get_model_info",
    "get_all_model_ids",
    "list_strategies",
    "is_model_supported",
    "available_models"
]


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class ProblemType(str, Enum):
    """
    🎯 **Problem Type**
    
    Machine learning problem types.
    """
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"        # Reserved for future
    TIME_SERIES = "time_series"      # Reserved for future


class ModelCategory(str, Enum):
    """
    📊 **Model Category**
    
    Model algorithm categories.
    """
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    BAYESIAN = "bayesian"
    BOOSTING = "boosting"


# ═══════════════════════════════════════════════════════════════════════════
# Classification Models Registry
# ═══════════════════════════════════════════════════════════════════════════

_CLASSIFICATION_MODELS: Dict[str, Dict[str, Any]] = {
    "lr": {
        "name": "Logistic Regression",
        "category": ModelCategory.LINEAR,
        "description": "Linear model for binary/multiclass classification",
        "pros": ["Fast training", "Interpretable", "Good baseline", "Probabilistic"],
        "cons": ["Assumes linear separability", "May underfit complex data"],
        "best_for": ["Linearly separable data", "High-dimensional sparse data", "Baseline model"],
        "turbo": True,
        "sklearn_class": "LogisticRegression"
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "category": ModelCategory.LINEAR,
        "description": "Instance-based learning algorithm",
        "pros": ["Simple concept", "No training phase", "Non-parametric"],
        "cons": ["Slow predictions", "Memory intensive", "Sensitive to feature scaling"],
        "best_for": ["Small datasets", "Low-dimensional data", "Non-linear boundaries"],
        "turbo": False,
        "sklearn_class": "KNeighborsClassifier"
    },
    "nb": {
        "name": "Naive Bayes",
        "category": ModelCategory.BAYESIAN,
        "description": "Probabilistic classifier based on Bayes theorem",
        "pros": ["Very fast", "Works well with small data", "Handles high dimensions"],
        "cons": ["Assumes feature independence", "Poor with correlated features"],
        "best_for": ["Text classification", "Small datasets", "Real-time predictions"],
        "turbo": True,
        "sklearn_class": "GaussianNB"
    },
    "dt": {
        "name": "Decision Tree",
        "category": ModelCategory.TREE_BASED,
        "description": "Tree-based classifier with if-then-else rules",
        "pros": ["Highly interpretable", "Handles non-linear", "No feature scaling needed"],
        "cons": ["Prone to overfitting", "Unstable (high variance)"],
        "best_for": ["Rule extraction", "Interpretability", "Mixed data types"],
        "turbo": True,
        "sklearn_class": "DecisionTreeClassifier"
    },
    "svm": {
        "name": "Support Vector Machine",
        "category": ModelCategory.LINEAR,
        "description": "Maximum margin classifier with kernel trick",
        "pros": ["Effective in high dimensions", "Memory efficient", "Versatile (kernels)"],
        "cons": ["Slow on large datasets", "Sensitive to feature scaling", "Hard to interpret"],
        "best_for": ["High-dimensional data", "Clear margin of separation", "Small to medium data"],
        "turbo": False,
        "sklearn_class": "SVC"
    },
    "rf": {
        "name": "Random Forest",
        "category": ModelCategory.ENSEMBLE,
        "description": "Ensemble of decision trees with bagging",
        "pros": ["Robust to overfitting", "Feature importance", "Handles missing values", "Parallel training"],
        "cons": ["Memory intensive", "Slower predictions", "Less interpretable"],
        "best_for": ["Tabular data", "Feature importance", "Robust baseline", "Production"],
        "turbo": True,
        "sklearn_class": "RandomForestClassifier"
    },
    "et": {
        "name": "Extra Trees",
        "category": ModelCategory.ENSEMBLE,
        "description": "Extremely randomized trees",
        "pros": ["Faster than RF", "Less overfitting", "Good generalization"],
        "cons": ["May underfit", "Less accurate than RF on some datasets"],
        "best_for": ["Similar to Random Forest", "When speed matters"],
        "turbo": True,
        "sklearn_class": "ExtraTreesClassifier"
    },
    "gbc": {
        "name": "Gradient Boosting",
        "category": ModelCategory.BOOSTING,
        "description": "Sequential ensemble with gradient descent",
        "pros": ["High accuracy", "Flexible loss functions", "Feature importance"],
        "cons": ["Slow training", "Prone to overfitting", "Sequential (no parallelism)"],
        "best_for": ["Structured data", "When accuracy is critical", "Competitions"],
        "turbo": True,
        "sklearn_class": "GradientBoostingClassifier"
    },
    "xgboost": {
        "name": "XGBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Optimized gradient boosting with regularization",
        "pros": ["State-of-the-art accuracy", "Fast training", "Built-in regularization", "Handles missing values"],
        "cons": ["Complex hyperparameters", "Requires tuning"],
        "best_for": ["Structured/tabular data", "Kaggle competitions", "Production ML"],
        "turbo": True,
        "requires": ["xgboost"],
        "external_package": True
    },
    "lightgbm": {
        "name": "LightGBM",
        "category": ModelCategory.BOOSTING,
        "description": "Fast gradient boosting with leaf-wise growth",
        "pros": ["Very fast training", "Memory efficient", "Handles large datasets", "Categorical features"],
        "cons": ["May overfit small datasets", "Requires careful tuning"],
        "best_for": ["Large datasets", "Fast training", "Production", "Real-time inference"],
        "turbo": True,
        "requires": ["lightgbm"],
        "external_package": True
    },
    "catboost": {
        "name": "CatBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Gradient boosting with native categorical support",
        "pros": ["Handles categorical features natively", "Less hyperparameter tuning", "Robust defaults"],
        "cons": ["Slower than LightGBM", "Large memory footprint"],
        "best_for": ["Categorical features", "Less tuning needed", "Robust performance"],
        "turbo": True,
        "requires": ["catboost"],
        "external_package": True
    },
    "ada": {
        "name": "AdaBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Adaptive boosting algorithm",
        "pros": ["Simple to use", "Less prone to overfitting than GB", "Works with weak learners"],
        "cons": ["Sensitive to outliers", "Slower than modern boosting"],
        "best_for": ["Binary classification", "When using weak learners", "Interpretable boosting"],
        "turbo": True,
        "sklearn_class": "AdaBoostClassifier"
    }
}

CLASSIFICATION_MODELS: Mapping[str, Dict[str, Any]] = MappingProxyType(_CLASSIFICATION_MODELS)


# ═══════════════════════════════════════════════════════════════════════════
# Regression Models Registry
# ═══════════════════════════════════════════════════════════════════════════

_REGRESSION_MODELS: Dict[str, Dict[str, Any]] = {
    "lr": {
        "name": "Linear Regression",
        "category": ModelCategory.LINEAR,
        "description": "Ordinary least squares regression",
        "pros": ["Simple and fast", "Interpretable", "Good baseline"],
        "cons": ["Assumes linearity", "Sensitive to outliers"],
        "best_for": ["Linear relationships", "Baseline model", "Interpretability"],
        "turbo": True,
        "sklearn_class": "LinearRegression"
    },
    "ridge": {
        "name": "Ridge Regression",
        "category": ModelCategory.LINEAR,
        "description": "L2 regularized linear regression",
        "pros": ["Handles multicollinearity", "More stable than OLS", "Prevents overfitting"],
        "cons": ["Doesn't perform feature selection", "May underfit"],
        "best_for": ["Correlated features", "Regularization", "Stable predictions"],
        "turbo": True,
        "sklearn_class": "Ridge"
    },
    "lasso": {
        "name": "Lasso Regression",
        "category": ModelCategory.LINEAR,
        "description": "L1 regularized linear regression",
        "pros": ["Feature selection", "Sparse models", "Interpretable"],
        "cons": ["May underfit", "Unstable with correlated features"],
        "best_for": ["High-dimensional data", "Feature selection", "Sparse solutions"],
        "turbo": True,
        "sklearn_class": "Lasso"
    },
    "en": {
        "name": "Elastic Net",
        "category": ModelCategory.LINEAR,
        "description": "L1 + L2 regularized regression",
        "pros": ["Combines Ridge and Lasso", "Handles correlated features", "Feature selection"],
        "cons": ["More hyperparameters", "Slower than Ridge/Lasso"],
        "best_for": ["Correlated features with sparsity", "Best of both worlds"],
        "turbo": True,
        "sklearn_class": "ElasticNet"
    },
    "dt": {
        "name": "Decision Tree",
        "category": ModelCategory.TREE_BASED,
        "description": "Tree-based regressor",
        "pros": ["Interpretable", "Handles non-linear", "No feature scaling needed"],
        "cons": ["Prone to overfitting", "High variance"],
        "best_for": ["Non-linear relationships", "Interpretability"],
        "turbo": True,
        "sklearn_class": "DecisionTreeRegressor"
    },
    "rf": {
        "name": "Random Forest",
        "category": ModelCategory.ENSEMBLE,
        "description": "Ensemble of decision trees",
        "pros": ["Robust", "Handles outliers", "Feature importance", "Low risk of overfitting"],
        "cons": ["Memory intensive", "Slower predictions"],
        "best_for": ["Tabular data", "Non-linear relationships", "Production"],
        "turbo": True,
        "sklearn_class": "RandomForestRegressor"
    },
    "et": {
        "name": "Extra Trees",
        "category": ModelCategory.ENSEMBLE,
        "description": "Extremely randomized trees",
        "pros": ["Faster than RF", "Less overfitting", "Good generalization"],
        "cons": ["May underfit", "Less accurate on some datasets"],
        "best_for": ["Similar to Random Forest", "When speed matters"],
        "turbo": True,
        "sklearn_class": "ExtraTreesRegressor"
    },
    "gbr": {
        "name": "Gradient Boosting",
        "category": ModelCategory.BOOSTING,
        "description": "Sequential ensemble for regression",
        "pros": ["High accuracy", "Flexible", "Feature importance"],
        "cons": ["Slow training", "Prone to overfitting", "Sequential"],
        "best_for": ["Complex relationships", "When accuracy is critical"],
        "turbo": True,
        "sklearn_class": "GradientBoostingRegressor"
    },
    "xgboost": {
        "name": "XGBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Optimized gradient boosting",
        "pros": ["State-of-the-art", "Fast", "Regularization", "Handles missing values"],
        "cons": ["Complex hyperparameters", "Requires tuning"],
        "best_for": ["Structured data", "Competitions", "Production"],
        "turbo": True,
        "requires": ["xgboost"],
        "external_package": True
    },
    "lightgbm": {
        "name": "LightGBM",
        "category": ModelCategory.BOOSTING,
        "description": "Fast gradient boosting",
        "pros": ["Very fast", "Memory efficient", "Large datasets"],
        "cons": ["May overfit small data", "Requires tuning"],
        "best_for": ["Large datasets", "Fast training", "Production"],
        "turbo": True,
        "requires": ["lightgbm"],
        "external_package": True
    },
    "catboost": {
        "name": "CatBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Gradient boosting with categorical support",
        "pros": ["Handles categorical", "Less tuning", "Robust defaults"],
        "cons": ["Slower than LightGBM", "Large memory"],
        "best_for": ["Categorical features", "Robust performance"],
        "turbo": True,
        "requires": ["catboost"],
        "external_package": True
    }
}

REGRESSION_MODELS: Mapping[str, Dict[str, Any]] = MappingProxyType(_REGRESSION_MODELS)


# ═══════════════════════════════════════════════════════════════════════════
# Model Selection Strategies
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_SELECTION_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "description": "Quick models for rapid prototyping (< 1 minute training)",
        "classification": ["lr", "dt", "rf"],
        "regression": ["lr", "ridge", "rf"]
    },
    "balanced": {
        "description": "Balanced trade-off between speed and accuracy",
        "classification": ["rf", "et", "gbc", "lr"],
        "regression": ["rf", "et", "gbr", "ridge"]
    },
    "accurate": {
        "description": "Focus on maximum accuracy (slower training acceptable)",
        "classification": ["xgboost", "lightgbm", "catboost", "rf", "et", "gbc"],
        "regression": ["xgboost", "lightgbm", "catboost", "rf", "et", "gbr"]
    },
    "production": {
        "description": "Production-friendly: robust, stable, well-tested defaults",
        "classification": ["rf", "et", "lr"],
        "regression": ["rf", "et", "ridge"]
    },
    "interpretable": {
        "description": "Interpretable models for explanations and auditing",
        "classification": ["lr", "dt"],
        "regression": ["lr", "ridge", "dt"]
    },
    "all": {
        "description": "Compare all registered models (ignores dependencies)",
        "classification": list(_CLASSIFICATION_MODELS.keys()),
        "regression": list(_REGRESSION_MODELS.keys())
    },
    "all_available": {
        "description": "All models with installed dependencies",
        "classification": list(_CLASSIFICATION_MODELS.keys()),
        "regression": list(_REGRESSION_MODELS.keys())
    }
}

MODEL_SELECTION_STRATEGIES: Mapping[str, Dict[str, Any]] = MappingProxyType(_MODEL_SELECTION_STRATEGIES)


# ═══════════════════════════════════════════════════════════════════════════
# Dependency Management
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def _has_module(module_name: str) -> bool:
    """
    Check if module is available (cached).
    
    Args:
        module_name: Module name to check
    
    Returns:
        True if module is importable
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _check_dependencies(model_id: str, problem_type: ProblemType) -> bool:
    """
    Check if all required packages for model are available.
    
    Args:
        model_id: Model identifier
        problem_type: Problem type
    
    Returns:
        True if all dependencies available
    """
    registry = (
        CLASSIFICATION_MODELS if problem_type == ProblemType.CLASSIFICATION
        else REGRESSION_MODELS
    )
    
    info = registry.get(model_id, {})
    required = info.get("requires", [])
    
    return all(_has_module(pkg) for pkg in required)


def available_models(models: Iterable[str], problem_type: ProblemType) -> List[str]:
    """
    🔍 **Filter Available Models**
    
    Filters models to only those with available dependencies.
    
    Args:
        models: List of model IDs to filter
        problem_type: Problem type
    
    Returns:
        List of available model IDs
    
    Example:
```python
        all_models = ["rf", "xgboost", "lightgbm"]
        available = available_models(all_models, ProblemType.CLASSIFICATION)
        # Returns only models with installed packages
```
    """
    return [m for m in models if _check_dependencies(m, problem_type)]


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def list_strategies() -> Dict[str, str]:
    """
    📋 **List Selection Strategies**
    
    Returns all available selection strategies with descriptions.
    
    Returns:
        Dictionary mapping strategy name to description
    
    Example:
```python
        strategies = list_strategies()
        for name, desc in strategies.items():
            print(f"{name}: {desc}")
```
    """
    return {k: v["description"] for k, v in MODEL_SELECTION_STRATEGIES.items()}


def get_all_model_ids(problem_type: ProblemType) -> List[str]:
    """
    📝 **Get All Model IDs**
    
    Returns all model IDs for problem type.
    
    Args:
        problem_type: Problem type
    
    Returns:
        List of all model IDs
    
    Example:
```python
        models = get_all_model_ids(ProblemType.CLASSIFICATION)
        print(f"Available: {', '.join(models)}")
```
    """
    if problem_type == ProblemType.CLASSIFICATION:
        return list(CLASSIFICATION_MODELS.keys())
    elif problem_type == ProblemType.REGRESSION:
        return list(REGRESSION_MODELS.keys())
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


def _normalize_strategy_name(strategy: str) -> str:
    """Normalize strategy name (handle aliases and case)."""
    s = (strategy or "").strip().lower()
    
    aliases = {
        "prod": "production",
        "robust": "production",
        "quality": "accurate",
        "accuracy": "accurate",
        "quick": "fast",
        "all_avail": "all_available"
    }
    
    return aliases.get(s, s)


def get_models_for_problem(
    problem_type: ProblemType,
    strategy: str = "accurate",
    *,
    only_available: bool = False
) -> List[str]:
    """
    🎯 **Get Models for Problem Type**
    
    Returns list of model IDs for given problem type and strategy.
    
    Args:
        problem_type: Problem type (classification/regression)
        strategy: Selection strategy (fast/balanced/accurate/production/interpretable/all)
        only_available: If True, filter out models without dependencies
    
    Returns:
        List of model IDs
    
    Raises:
        ValueError: If strategy or problem type invalid
    
    Example:
```python
        # Get accurate models for classification
        models = get_models_for_problem(
            ProblemType.CLASSIFICATION,
            strategy="accurate",
            only_available=True
        )
        # Returns: ["xgboost", "lightgbm", "rf", ...] (only if packages installed)
        
        # Get fast models for regression
        models = get_models_for_problem(
            ProblemType.REGRESSION,
            strategy="fast"
        )
        # Returns: ["lr", "ridge", "rf"]
```
    """
    # Normalize strategy
    strategy_name = _normalize_strategy_name(strategy)
    
    if strategy_name not in MODEL_SELECTION_STRATEGIES:
        available = ', '.join(sorted(MODEL_SELECTION_STRATEGIES.keys()))
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available: {available}"
        )
    
    # Get problem key
    if problem_type == ProblemType.CLASSIFICATION:
        key = "classification"
    elif problem_type == ProblemType.REGRESSION:
        key = "regression"
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
    
    # Get base model list
    base_models = list(MODEL_SELECTION_STRATEGIES[strategy_name][key])
    
    # Handle 'all_available' strategy
    if strategy_name == "all_available":
        base_models = available_models(base_models, problem_type)
    
    # Apply availability filter if requested
    models = base_models
    if only_available:
        models = available_models(models, problem_type)
    
    # Failsafe: ensure at least some models available
    if not models:
        # Fallback to safe defaults
        fallback = ["rf", "et", "dt"]
        if problem_type == ProblemType.CLASSIFICATION:
            fallback.append("lr")
        else:
            fallback.append("ridge")
        
        models = available_models(fallback, problem_type) or fallback
    
    # Remove duplicates while preserving order
    seen = set()
    unique = [m for m in models if not (m in seen or seen.add(m))]
    
    return unique


def get_model_info(model_id: str, problem_type: ProblemType) -> Dict[str, Any]:
    """
    ℹ️ **Get Model Information**
    
    Returns detailed information about a model.
    
    Args:
        model_id: Model identifier
        problem_type: Problem type
    
    Returns:
        Dictionary with model information
    
    Example:
```python
        info = get_model_info("xgboost", ProblemType.CLASSIFICATION)
        
        print(info["name"])         # "XGBoost"
        print(info["category"])     # ModelCategory.BOOSTING
        print(info["pros"])         # ["State-of-the-art", ...]
        print(info["cons"])         # ["Complex hyperparameters", ...]
        print(info["best_for"])     # ["Structured data", ...]
```
    """
    if problem_type == ProblemType.CLASSIFICATION:
        return dict(CLASSIFICATION_MODELS.get(model_id, {}))
    elif problem_type == ProblemType.REGRESSION:
        return dict(REGRESSION_MODELS.get(model_id, {}))
    else:
        return {}


def is_model_supported(model_id: str, problem_type: ProblemType) -> bool:
    """
    ✅ **Check if Model Supported**
    
    Checks if model exists in registry (doesn't check dependencies).
    
    Args:
        model_id: Model identifier
        problem_type: Problem type
    
    Returns:
        True if model in registry
    
    Example:
```python
        if is_model_supported("xgboost", ProblemType.CLASSIFICATION):
            print("XGBoost is supported for classification")
```
    """
    if problem_type == ProblemType.CLASSIFICATION:
        registry = CLASSIFICATION_MODELS
    elif problem_type == ProblemType.REGRESSION:
        registry = REGRESSION_MODELS
    else:
        return False
    
    return model_id in registry


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def _validate_registry() -> None:
    """Validate registry structure at import time."""
    def _check_registry(name: str, registry: Mapping[str, Dict[str, Any]]) -> None:
        assert isinstance(registry, Mapping), f"{name} must be a Mapping"
        
        for model_id, info in registry.items():
            # Required fields
            assert "name" in info and isinstance(info["name"], str), \
                f"{name}.{model_id} missing 'name'"
            assert "category" in info and isinstance(info["category"], ModelCategory), \
                f"{name}.{model_id} invalid 'category'"
            
            # Optional requires field
            if "requires" in info:
                req = info["requires"]
                assert isinstance(req, list) and all(isinstance(x, str) for x in req), \
                    f"{name}.{model_id} 'requires' must be list[str]"
    
    # Check registries
    _check_registry("CLASSIFICATION_MODELS", CLASSIFICATION_MODELS)
    _check_registry("REGRESSION_MODELS", REGRESSION_MODELS)
    
    # Check strategies reference valid models
    for strategy, spec in MODEL_SELECTION_STRATEGIES.items():
        for key in ("classification", "regression"):
            if key in spec:
                registry = (
                    CLASSIFICATION_MODELS if key == "classification"
                    else REGRESSION_MODELS
                )
                
                missing = [m for m in spec[key] if m not in registry]
                assert not missing, \
                    f"Strategy '{strategy}' references unknown models in {key}: {missing}"


# Run validation at import
try:
    _validate_registry()
except AssertionError as e:
    import warnings
    warnings.warn(f"[model_registry] Validation warning: {e}", RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Model Registry v{__version__} - Self Test")
    print("="*80)
    
    # Test strategies
    print("\n1. Testing Strategies...")
    strategies = list_strategies()
    print(f"   Available strategies: {len(strategies)}")
    for name, desc in strategies.items():
        print(f"   • {name}: {desc}")
    
    # Test model retrieval
    print("\n2. Testing Model Retrieval...")
    
    for problem_type in [ProblemType.CLASSIFICATION, ProblemType.REGRESSION]:
        print(f"\n   {problem_type.value.upper()}:")
        
        for strategy in ["fast", "balanced", "accurate"]:
            models = get_models_for_problem(problem_type, strategy)
            available = get_models_for_problem(problem_type, strategy, only_available=True)
            
            print(f"   • {strategy}: {len(models)} models, {len(available)} available")
            print(f"     {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
    
    # Test model info
    print("\n3. Testing Model Information...")
    
    test_models = [
        ("xgboost", ProblemType.CLASSIFICATION),
        ("rf", ProblemType.REGRESSION)
    ]
    
    for model_id, problem_type in test_models:
        info = get_model_info(model_id, problem_type)
        if info:
            print(f"\n   {model_id} ({problem_type.value}):")
            print(f"   • Name: {info['name']}")
            print(f"   • Category: {info['category'].value}")
            print(f"   • Pros: {', '.join(info['pros'][:2])}")
            print(f"   • Best for: {', '.join(info['best_for'][:2])}")
    
    # Test dependency checking
    print("\n4. Testing Dependency Checking...")
    
    optional_packages = ["xgboost", "lightgbm", "catboost"]
    for pkg in optional_packages:
        available = _has_module(pkg)
        print(f"   • {pkg}: {'✓ available' if available else '✗ not installed'}")
    
    # Test model support
    print("\n5. Testing Model Support...")
    
    test_cases = [
        ("rf", ProblemType.CLASSIFICATION, True),
        ("xgboost", ProblemType.REGRESSION, True),
        ("invalid", ProblemType.CLASSIFICATION, False)
    ]
    
    for model_id, problem_type, expected in test_cases:
        result = is_model_supported(model_id, problem_type)
        status = "✓" if result == expected else "✗"
        print(f"   {status} {model_id} in {problem_type.value}: {result}")
    
    # Test all models
    print("\n6. Testing All Models...")
    
    clf_models = get_all_model_ids(ProblemType.CLASSIFICATION)
    reg_models = get_all_model_ids(ProblemType.REGRESSION)
    
    print(f"   Classification models: {len(clf_models)}")
    print(f"   Regression models: {len(reg_models)}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from config.model_registry import (
    get_models_for_problem,
    get_model_info,
    list_strategies,
    ProblemType
)

# === List Available Strategies ===
strategies = list_strategies()
for name, desc in strategies.items():
    print(f"{name}: {desc}")

# === Get Models for Problem ===

# Fast prototyping
fast_models = get_models_for_problem(
    ProblemType.CLASSIFICATION,
    strategy="fast"
)
# Returns: ["lr", "dt", "rf"]

# Maximum accuracy (all models, filter by availability)
accurate_models = get_models_for_problem(
    ProblemType.CLASSIFICATION,
    strategy="accurate",
    only_available=True
)
# Returns: ["xgboost", "lightgbm", "rf", ...] (only installed)

# Production-ready models
prod_models = get_models_for_problem(
    ProblemType.REGRESSION,
    strategy="production"
)
# Returns: ["rf", "et", "ridge"]

# === Get Model Information ===

info = get_model_info("xgboost", ProblemType.CLASSIFICATION)

print(f"Name: {info['name']}")
# Output: "XGBoost"

print(f"Category: {info['category']}")
# Output: ModelCategory.BOOSTING

print(f"Pros: {info['pros']}")
# Output: ["State-of-the-art accuracy", "Fast training", ...]

print(f"Cons: {info['cons']}")
# Output: ["Complex hyperparameters", "Requires tuning"]

print(f"Best for: {info['best_for']}")
# Output: ["Structured/tabular data", "Kaggle competitions", ...]

# === Check Model Support ===

if is_model_supported("xgboost", ProblemType.CLASSIFICATION):
    print("XGBoost is supported")

# === Get All Model IDs ===

all_clf = get_all_model_ids(ProblemType.CLASSIFICATION)
all_reg = get_all_model_ids(ProblemType.REGRESSION)

print(f"Classification: {', '.join(all_clf)}")
print(f"Regression: {', '.join(all_reg)}")

# === Filter by Availability ===

from config.model_registry import available_models

all_models = ["rf", "xgboost", "lightgbm", "catboost"]
available = available_models(all_models, ProblemType.CLASSIFICATION)
print(f"Available: {', '.join(available)}")

# === Integration with ML Pipeline ===

from config.model_registry import get_models_for_problem, ProblemType

class ModelSelector:
    def __init__(self, problem_type: ProblemType, strategy: str = "balanced"):
        self.problem_type = problem_type
        self.strategy = strategy
    
    def get_models(self):
        return get_models_for_problem(
            self.problem_type,
            self.strategy,
            only_available=True
        )
    
    def get_model_details(self, model_id: str):
        return get_model_info(model_id, self.problem_type)

# Usage
selector = ModelSelector(ProblemType.CLASSIFICATION, strategy="accurate")
models = selector.get_models()

for model_id in models:
    details = selector.get_model_details(model_id)
    print(f"{model_id}: {details['name']}")
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)
