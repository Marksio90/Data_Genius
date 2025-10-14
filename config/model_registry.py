# === config/model_registry.py ===
"""
DataGenius PRO - Model Registry Configuration (PRO+++)
Rejestr dostępnych modeli ML oraz strategie doboru per typ problemu.

Zasady:
- Rejestry są niemutowalne (MappingProxyType).
- Zgodność wsteczna z istniejącym kodem (nazwy i signatury).
- Dodane helpery: walidacja strategii, lista strategii, filtrowanie modeli
  zależnych od zewn. pakietów (xgboost/lightgbm/catboost).
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Dict, List, Any, Mapping, Optional
import importlib.util


# ===========================================
# === ENUMY ===
# ===========================================
class ProblemType(str, Enum):
    """ML problem types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"


class ModelCategory(str, Enum):
    """Model categories."""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    BAYESIAN = "bayesian"
    BOOSTING = "boosting"


# ===========================================
# === REJESTR: KLASYFIKACJA ===
# ===========================================
_CLASSIFICATION_MODELS: Dict[str, Dict[str, Any]] = {
    "lr": {
        "name": "Logistic Regression",
        "category": ModelCategory.LINEAR,
        "description": "Linear model for binary/multiclass classification",
        "pros": ["Fast", "Interpretable", "Good baseline"],
        "cons": ["Assumes linear relationship", "May underfit"],
        "best_for": ["Linear separable data", "High-dimensional data"],
        "turbo": True,
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "category": ModelCategory.LINEAR,
        "description": "Instance-based learning algorithm",
        "pros": ["Simple", "No training phase"],
        "cons": ["Slow predictions", "Sensitive to scale"],
        "best_for": ["Small datasets", "Low dimensions"],
        "turbo": False,
    },
    "nb": {
        "name": "Naive Bayes",
        "category": ModelCategory.BAYESIAN,
        "description": "Probabilistic classifier based on Bayes theorem",
        "pros": ["Fast", "Works well with small data"],
        "cons": ["Assumes feature independence"],
        "best_for": ["Text classification", "Small datasets"],
        "turbo": True,
    },
    "dt": {
        "name": "Decision Tree",
        "category": ModelCategory.TREE_BASED,
        "description": "Tree-based classifier",
        "pros": ["Interpretable", "Handles non-linear"],
        "cons": ["Prone to overfitting"],
        "best_for": ["Categorical features", "Rule extraction"],
        "turbo": True,
    },
    "svm": {
        "name": "Support Vector Machine",
        "category": ModelCategory.LINEAR,
        "description": "Maximum margin classifier",
        "pros": ["Effective in high dimensions", "Memory efficient"],
        "cons": ["Slow on large datasets", "Hard to interpret"],
        "best_for": ["High-dimensional data", "Clear margin"],
        "turbo": False,
    },
    "rf": {
        "name": "Random Forest",
        "category": ModelCategory.ENSEMBLE,
        "description": "Ensemble of decision trees",
        "pros": ["Robust", "Handles missing values", "Feature importance"],
        "cons": ["Memory intensive", "Slow inference"],
        "best_for": ["Tabular data", "Feature importance"],
        "turbo": True,
    },
    "et": {
        "name": "Extra Trees",
        "category": ModelCategory.ENSEMBLE,
        "description": "Extremely randomized trees",
        "pros": ["Faster than RF", "Less overfitting"],
        "cons": ["May underfit"],
        "best_for": ["Similar to Random Forest"],
        "turbo": True,
    },
    "gbc": {
        "name": "Gradient Boosting",
        "category": ModelCategory.BOOSTING,
        "description": "Sequential ensemble method",
        "pros": ["High accuracy", "Handles various data types"],
        "cons": ["Slow training", "Prone to overfitting"],
        "best_for": ["Tabular data", "Competitions"],
        "turbo": True,
    },
    "xgboost": {
        "name": "XGBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Optimized gradient boosting",
        "pros": ["State-of-the-art", "Fast", "Regularization"],
        "cons": ["Complex hyperparameters"],
        "best_for": ["Structured data", "Competitions"],
        "turbo": True,
        "requires": ["xgboost"],
    },
    "lightgbm": {
        "name": "LightGBM",
        "category": ModelCategory.BOOSTING,
        "description": "Fast gradient boosting framework",
        "pros": ["Very fast", "Memory efficient", "Handles large data"],
        "cons": ["May overfit small data"],
        "best_for": ["Large datasets", "Fast training"],
        "turbo": True,
        "requires": ["lightgbm"],
    },
    "catboost": {
        "name": "CatBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Gradient boosting with categorical support",
        "pros": ["Handles categorical features", "Less hyperparameters"],
        "cons": ["Slower than LightGBM"],
        "best_for": ["Categorical features", "Less tuning needed"],
        "turbo": True,
        "requires": ["catboost"],
    },
    "ada": {
        "name": "AdaBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Adaptive boosting algorithm",
        "pros": ["Simple", "Less prone to overfitting"],
        "cons": ["Sensitive to outliers"],
        "best_for": ["Binary classification", "Weak learners"],
        "turbo": True,
    },
}
CLASSIFICATION_MODELS: Mapping[str, Dict[str, Any]] = MappingProxyType(_CLASSIFICATION_MODELS)


# ===========================================
# === REJESTR: REGRESJA ===
# ===========================================
_REGRESSION_MODELS: Dict[str, Dict[str, Any]] = {
    "lr": {
        "name": "Linear Regression",
        "category": ModelCategory.LINEAR,
        "description": "Ordinary least squares regression",
        "pros": ["Simple", "Interpretable", "Fast"],
        "cons": ["Assumes linearity"],
        "best_for": ["Linear relationships", "Baseline"],
        "turbo": True,
    },
    "ridge": {
        "name": "Ridge Regression",
        "category": ModelCategory.LINEAR,
        "description": "L2 regularized linear regression",
        "pros": ["Handles multicollinearity", "Stable"],
        "cons": ["May underfit"],
        "best_for": ["Correlated features"],
        "turbo": True,
    },
    "lasso": {
        "name": "Lasso Regression",
        "category": ModelCategory.LINEAR,
        "description": "L1 regularized linear regression",
        "pros": ["Feature selection", "Sparse models"],
        "cons": ["May underfit"],
        "best_for": ["High-dimensional data", "Feature selection"],
        "turbo": True,
    },
    "en": {
        "name": "Elastic Net",
        "category": ModelCategory.LINEAR,
        "description": "L1 + L2 regularized regression",
        "pros": ["Combines Ridge and Lasso", "Robust"],
        "cons": ["More hyperparameters"],
        "best_for": ["Correlated features + sparsity"],
        "turbo": True,
    },
    "dt": {
        "name": "Decision Tree",
        "category": ModelCategory.TREE_BASED,
        "description": "Tree-based regressor",
        "pros": ["Interpretable", "Handles non-linear"],
        "cons": ["Prone to overfitting"],
        "best_for": ["Non-linear relationships"],
        "turbo": True,
    },
    "rf": {
        "name": "Random Forest",
        "category": ModelCategory.ENSEMBLE,
        "description": "Ensemble of decision trees",
        "pros": ["Robust", "Handles outliers", "Feature importance"],
        "cons": ["Memory intensive"],
        "best_for": ["Tabular data", "Non-linear"],
        "turbo": True,
    },
    "et": {
        "name": "Extra Trees",
        "category": ModelCategory.ENSEMBLE,
        "description": "Extremely randomized trees",
        "pros": ["Faster than RF", "Less overfitting"],
        "cons": ["May underfit"],
        "best_for": ["Similar to Random Forest"],
        "turbo": True,
    },
    "gbr": {
        "name": "Gradient Boosting",
        "category": "ModelCategory.BOOSTING",
        "description": "Sequential ensemble for regression",
        "pros": ["High accuracy", "Flexible"],
        "cons": ["Slow training"],
        "best_for": ["Complex relationships"],
        "turbo": True,
    },
    "xgboost": {
        "name": "XGBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Optimized gradient boosting",
        "pros": ["State-of-the-art", "Fast", "Regularization"],
        "cons": ["Complex hyperparameters"],
        "best_for": ["Structured data", "Competitions"],
        "turbo": True,
        "requires": ["xgboost"],
    },
    "lightgbm": {
        "name": "LightGBM",
        "category": ModelCategory.BOOSTING,
        "description": "Fast gradient boosting",
        "pros": ["Very fast", "Memory efficient"],
        "cons": ["May overfit small data"],
        "best_for": ["Large datasets"],
        "turbo": True,
        "requires": ["lightgbm"],
    },
    "catboost": {
        "name": "CatBoost",
        "category": ModelCategory.BOOSTING,
        "description": "Gradient boosting with categorical support",
        "pros": ["Handles categorical", "Less tuning"],
        "cons": ["Slower than LightGBM"],
        "best_for": ["Categorical features"],
        "turbo": True,
        "requires": ["catboost"],
    },
}
REGRESSION_MODELS: Mapping[str, Dict[str, Any]] = MappingProxyType(_REGRESSION_MODELS)


# ===========================================
# === STRATEGIE WYBORU MODELI ===
# ===========================================
_MODEL_SELECTION_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "description": "Quick models for rapid prototyping",
        "classification": ["lr", "dt", "rf"],
        "regression": ["lr", "ridge", "rf"],
    },
    "accurate": {
        "description": "Focus on accuracy (slower training)",
        "classification": ["xgboost", "lightgbm", "catboost", "rf", "et"],
        "regression": ["xgboost", "lightgbm", "catboost", "rf", "et"],
    },
    "interpretable": {
        "description": "Interpretable models for explanations",
        "classification": ["lr", "dt"],
        "regression": ["lr", "ridge", "dt"],
    },
    "all": {
        "description": "Compare all available models",
        "classification": list(_CLASSIFICATION_MODELS.keys()),
        "regression": list(_REGRESSION_MODELS.keys()),
    },
}
MODEL_SELECTION_STRATEGIES: Mapping[str, Dict[str, Any]] = MappingProxyType(_MODEL_SELECTION_STRATEGIES)


# ===========================================
# === HELPERY ===
# ===========================================
def _check_deps(model_id: str, problem_type: ProblemType) -> bool:
    """
    Sprawdza, czy wymagane pakiety modelu są dostępne (importowalne).
    """
    registry = CLASSIFICATION_MODELS if problem_type == ProblemType.CLASSIFICATION else REGRESSION_MODELS
    info = registry.get(model_id, {})
    required = info.get("requires", [])
    if not required:
        return True
    return all(importlib.util.find_spec(pkg) is not None for pkg in required)


def list_strategies() -> Dict[str, str]:
    """Zwraca mapę strategia → opis."""
    return {k: v["description"] for k, v in MODEL_SELECTION_STRATEGIES.items()}


def get_all_model_ids(problem_type: ProblemType) -> List[str]:
    """Zwraca wszystkie identyfikatory modeli dla danego problemu."""
    if problem_type == ProblemType.CLASSIFICATION:
        return list(CLASSIFICATION_MODELS.keys())
    if problem_type == ProblemType.REGRESSION:
        return list(REGRESSION_MODELS.keys())
    raise ValueError(f"Unsupported problem type for registry: {problem_type}")


def get_models_for_problem(
    problem_type: ProblemType,
    strategy: str = "accurate",
    *,
    only_available: bool = False,
) -> List[str]:
    """
    Zwraca listę ID modeli dla danego problemu i strategii.

    Args:
        problem_type: Typ problemu ML.
        strategy: Nazwa strategii (fast/accurate/interpretable/all).
        only_available: Jeżeli True, odfiltruje modele bez zainstalowanych zależności.

    Returns:
        Lista identyfikatorów modeli.
    """
    if strategy not in MODEL_SELECTION_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {', '.join(MODEL_SELECTION_STRATEGIES.keys())}")

    key = "classification" if problem_type == ProblemType.CLASSIFICATION else \
          "regression" if problem_type == ProblemType.REGRESSION else None

    if key is None:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    models = list(MODEL_SELECTION_STRATEGIES[strategy][key])

    if only_available:
        models = [m for m in models if _check_deps(m, problem_type)]
    return models


def get_model_info(model_id: str, problem_type: ProblemType) -> Dict[str, Any]:
    """
    Zwraca szczegóły modelu dla danego ID i typu problemu.
    """
    if problem_type == ProblemType.CLASSIFICATION:
        return dict(CLASSIFICATION_MODELS.get(model_id, {}))
    if problem_type == ProblemType.REGRESSION:
        return dict(REGRESSION_MODELS.get(model_id, {}))
    return {}


def is_model_supported(model_id: str, problem_type: ProblemType) -> bool:
    """
    Czy model istnieje w rejestrze (nie sprawdza zależności)?
    """
    registry = CLASSIFICATION_MODELS if problem_type == ProblemType.CLASSIFICATION else \
               REGRESSION_MODELS if problem_type == ProblemType.REGRESSION else {}
    return model_id in registry


__all__ = [
    "ProblemType",
    "ModelCategory",
    "CLASSIFICATION_MODELS",
    "REGRESSION_MODELS",
    "MODEL_SELECTION_STRATEGIES",
    "get_models_for_problem",
    "get_model_info",
    "list_strategies",
    "get_all_model_ids",
    "is_model_supported",
]
