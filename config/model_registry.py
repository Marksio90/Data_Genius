# === config/model_registry.py ===
"""
DataGenius PRO — Model Registry (PRO++++++)
Rejestr modeli ML + strategie wyboru per typ problemu, z walidacją i
obsługą brakujących zależności (xgboost/lightgbm/catboost).

API (stabilne):
- get_models_for_problem(problem_type, strategy="accurate", only_available=False) -> list[str]
- get_model_info(model_id, problem_type) -> dict
- get_all_model_ids(problem_type) -> list[str]
- list_strategies() -> dict[str,str]
- is_model_supported(model_id, problem_type) -> bool
- available_models(models, problem_type) -> list[str]

Dodatki PRO:
- caching import-checków (szybkie filtrowanie),
- walidacja rejestru przy imporcie (dev-friendly),
- strategie aliasów + „failsafe fallback”,
- strategie „balanced” i „production”,
- MappingProxyType (niemutowalne widoki),
- wersjonowanie modułu.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional
import importlib.util

__version__ = "4.3-kosmos"

# ──────────────────────────────────────────────────────────────────────────────
# Typy / Enums
# ──────────────────────────────────────────────────────────────────────────────

class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"       # zarezerwowane (poza zakresem tego modułu)
    TIME_SERIES = "time_series"     # zarezerwowane (poza zakresem tego modułu)


class ModelCategory(str, Enum):
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    BAYESIAN = "bayesian"
    BOOSTING = "boosting"


# ──────────────────────────────────────────────────────────────────────────────
# Rejestr: CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

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
        "category": ModelCategory.LINEAR,  # w tej taksonomii: klasyczny nieparametryczny
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
        "category": "ModelCategory.BOOSTING",
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

# ──────────────────────────────────────────────────────────────────────────────
# Rejestr: REGRESSION
# ──────────────────────────────────────────────────────────────────────────────

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
        "category": ModelCategory.BOOSTING,
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

# ──────────────────────────────────────────────────────────────────────────────
# Strategie wyboru
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_SELECTION_STRATEGIES: Dict[str, Dict[str, Any]] = {
    # Szybki prototyp
    "fast": {
        "description": "Quick models for rapid prototyping",
        "classification": ["lr", "dt", "rf"],
        "regression": ["lr", "ridge", "rf"],
    },
    # Maksymalna jakość (wolniej)
    "accurate": {
        "description": "Focus on accuracy (slower training)",
        "classification": ["xgboost", "lightgbm", "catboost", "rf", "et"],
        "regression": ["xgboost", "lightgbm", "catboost", "rf", "et", "gbr"],
    },
    # Zbalansowane (czas vs jakość)
    "balanced": {
        "description": "Balanced trade-off between speed and accuracy",
        "classification": ["rf", "et", "gbc", "lr"],
        "regression": ["rf", "et", "gbr", "ridge"],
    },
    # Produkcyjne (stabilne, często dobrze działają bez tuningu)
    "production": {
        "description": "Production-friendly, robust defaults",
        "classification": ["rf", "et", "lr"],
        "regression": ["rf", "et", "ridge"],
    },
    # Interpretowalne
    "interpretable": {
        "description": "Interpretable models for explanations",
        "classification": ["lr", "dt"],
        "regression": ["lr", "ridge", "dt"],
    },
    # Wszystko z rejestru (nie filtruje zależności)
    "all": {
        "description": "Compare all registered models",
        "classification": list(_CLASSIFICATION_MODELS.keys()),
        "regression": list(_REGRESSION_MODELS.keys()),
    },
    # Wszystko co dostępne w środowisku (filtruje zależności)
    "all_available": {
        "description": "All models with installed dependencies",
        "classification": list(_CLASSIFICATION_MODELS.keys()),
        "regression": list(_REGRESSION_MODELS.keys()),
    },
    # Alias kompatybilności
    "accurate_plus": {
        "description": "Alias of 'accurate' (compatibility)",
        "classification": ["xgboost", "lightgbm", "catboost", "rf", "et"],
        "regression": ["xgboost", "lightgbm", "catboost", "rf", "et", "gbr"],
    },
}
MODEL_SELECTION_STRATEGIES: Mapping[str, Dict[str, Any]] = MappingProxyType(_MODEL_SELECTION_STRATEGIES)

# ──────────────────────────────────────────────────────────────────────────────
# Helpres / API
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _has_module(modname: str) -> bool:
    """Cached import check (szybkie filtrowanie zależności)."""
    try:
        return importlib.util.find_spec(modname) is not None
    except Exception:
        return False


def _check_deps(model_id: str, problem_type: ProblemType) -> bool:
    """True, jeśli wszystkie wymagane pakiety danego modelu są importowalne."""
    registry = CLASSIFICATION_MODELS if problem_type == ProblemType.CLASSIFICATION else REGRESSION_MODELS
    info = registry.get(model_id, {})
    required = info.get("requires", [])
    return all(_has_module(pkg) for pkg in required)


def available_models(models: Iterable[str], problem_type: ProblemType) -> List[str]:
    """Filtruje podane modele pozostawiając tylko te z dostępnymi zależnościami."""
    return [m for m in models if _check_deps(m, problem_type)]


def list_strategies() -> Dict[str, str]:
    """Mapa nazwa_strategii → opis."""
    return {k: v["description"] for k, v in MODEL_SELECTION_STRATEGIES.items()}


def get_all_model_ids(problem_type: ProblemType) -> List[str]:
    """Wszystkie identyfikatory modeli dla danego problemu."""
    if problem_type == ProblemType.CLASSIFICATION:
        return list(CLASSIFICATION_MODELS.keys())
    if problem_type == ProblemType.REGRESSION:
        return list(REGRESSION_MODELS.keys())
    raise ValueError(f"Unsupported problem type for registry: {problem_type}")


def _normalize_strategy_name(strategy: str) -> str:
    """Dopuszcza aliasy i różne wielkości liter."""
    s = (strategy or "").strip().lower()
    aliases = {
        "prod": "production",
        "robust": "production",
        "quality": "accurate",
        "accuracy": "accurate",
        "quick": "fast",
        "all_avail": "all_available",
    }
    return aliases.get(s, s)


def get_models_for_problem(
    problem_type: ProblemType,
    strategy: str = "accurate",
    *,
    only_available: bool = False,
) -> List[str]:
    """
    Zwraca listę ID modeli dla danego problemu i strategii.
    - strategy: fast / balanced / accurate / production / interpretable / all / all_available
    - only_available=True odfiltruje modele bez zależności
    - failsafe: jeśli po filtrze nic nie zostanie, automatyczny fallback do bezpiecznych
    """
    sname = _normalize_strategy_name(strategy)
    if sname not in MODEL_SELECTION_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available: {', '.join(sorted(MODEL_SELECTION_STRATEGIES.keys()))}"
        )

    key = "classification" if problem_type == ProblemType.CLASSIFICATION else \
          "regression" if problem_type == ProblemType.REGRESSION else None
    if key is None:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    base = list(MODEL_SELECTION_STRATEGIES[sname][key])

    # 'all_available' – dynamiczny filtr zależności
    if sname == "all_available":
        base = available_models(base, problem_type)

    models = base
    if only_available:
        models = available_models(models, problem_type)

    # failsafe + deduplikacja z zachowaniem kolejności
    if not models:
        models = ["rf", "et", "dt"] + (["lr"] if problem_type == ProblemType.CLASSIFICATION else ["ridge"])

    seen = set()
    unique = [m for m in models if not (m in seen or seen.add(m))]
    return unique


def get_model_info(model_id: str, problem_type: ProblemType) -> Dict[str, Any]:
    """Szczegóły modelu dla danego ID i typu problemu."""
    if problem_type == ProblemType.CLASSIFICATION:
        return dict(CLASSIFICATION_MODELS.get(model_id, {}))
    if problem_type == ProblemType.REGRESSION:
        return dict(REGRESSION_MODELS.get(model_id, {}))
    return {}


def is_model_supported(model_id: str, problem_type: ProblemType) -> bool:
    """Czy model istnieje w rejestrze (nie sprawdza zależności)?"""
    registry = CLASSIFICATION_MODELS if problem_type == ProblemType.CLASSIFICATION else \
               REGRESSION_MODELS if problem_type == ProblemType.REGRESSION else {}
    return model_id in registry


# ──────────────────────────────────────────────────────────────────────────────
# Walidacja przy imporcie (dev-safe)
# ──────────────────────────────────────────────────────────────────────────────

def _validate_registry() -> None:
    def _check_block(name: str, block: Mapping[str, Dict[str, Any]]) -> None:
        assert isinstance(block, Mapping)
        for mid, info in block.items():
            assert "name" in info and isinstance(info["name"], str), f"{name}.{mid} missing 'name'"
            assert "category" in info and isinstance(info["category"], ModelCategory), f"{name}.{mid} invalid 'category'"
            if "requires" in info:
                req = info["requires"]
                assert isinstance(req, list) and all(isinstance(x, str) for x in req), f"{name}.{mid} 'requires' must be list[str]"

    _check_block("CLASSIFICATION_MODELS", CLASSIFICATION_MODELS)
    _check_block("REGRESSION_MODELS", REGRESSION_MODELS)

    # strategie – czy wskazują istniejące modele
    for strat, spec in MODEL_SELECTION_STRATEGIES.items():
        for key in ("classification", "regression"):
            if key in spec:
                registry = CLASSIFICATION_MODELS if key == "classification" else REGRESSION_MODELS
                missing = [m for m in spec[key] if m not in registry]
                assert not missing, f"Strategy '{strat}' references unknown models in {key}: {missing}"

try:  # pragma: no cover
    _validate_registry()
except AssertionError as e:
    import warnings
    warnings.warn(f"[model_registry] validation warning: {e}", RuntimeWarning)


__all__ = [
    "__version__",
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
    "available_models",
]
