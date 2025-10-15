# === OPIS MODUŁU ===
"""
DataGenius PRO++++ - Model Selector (KOSMOS)
Inteligentny wybór modeli z meta-learningiem i charakterystyką danych.

Kluczowe cechy:
- Automatyczna rekomendacja modeli (classification / regression; bin/multi)
- Meta-feature extraction z defensywą i heurystykami złożoności
- Ranking wielokryterialny (performance/speed/interpretability/robustness/scalability/memory)
- Przewidywanie wydajności + penalizacje (imbalance, high-dimensional, korelacje, brak danych)
- Opcjonalna szybka walidacja CV (quick benchmark) na ograniczonej próbce
- Obsługa wysokiej kardynalności i rzadkości (sparsity) – wpływ na punktację
- Multi-objective optimization z wagami dopasowanymi do kryterium użytkownika

Kontrakt wynikowy (SelectionResult) zachowany i rozszerzony w metadata.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, r2_score,
    make_scorer
)
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state

# Klasyfikatory
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    # LightGBM/XGBoost/CatBoost jako opcjonalne – jeśli brak, pomijamy w fabryce
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None

logger = logging.getLogger(__name__)


# === ENUMY ===
class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"            # heurystycznie (opcjonalne)
    IMBALANCED = "imbalanced"
    HIGH_DIMENSIONAL = "high_dimensional"


class ModelCategory(str, Enum):
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    ENSEMBLE = "ensemble"
    BOOSTING = "boosting"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    NAIVE_BAYES = "naive_bayes"
    KNN = "knn"


class SelectionCriterion(str, Enum):
    ACCURACY = "accuracy"
    SPEED = "speed"
    INTERPRETABILITY = "interpretability"
    ROBUSTNESS = "robustness"
    SCALABILITY = "scalability"
    MEMORY = "memory"
    BALANCED = "balanced"


# === DANE WYJŚCIOWE ===
@dataclass
class DatasetCharacteristics:
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    class_balance: Optional[Dict[str, float]]

    # Feature characteristics
    n_numerical: int
    n_categorical: int
    n_binary: int

    # Statistical properties
    avg_skewness: float
    avg_kurtosis: float
    missing_ratio: float

    # Complexity measures
    feature_to_sample_ratio: float
    avg_correlation: float
    dimensionality_score: float

    # Problem characteristics
    is_imbalanced: bool
    is_high_dimensional: bool
    is_sparse: bool

    # Heurystyki TS
    index_is_datetime: bool = False
    index_is_sorted: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    model_name: str
    model_category: ModelCategory

    # składowe punktacji (0..1)
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
    recommended_models: List[str]
    model_scores: List[ModelScore]
    dataset_characteristics: DatasetCharacteristics
    selection_criterion: SelectionCriterion
    problem_type: ProblemType

    selection_time: float
    total_models_evaluated: int

    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_models": self.recommended_models,
            "model_scores": [s.to_dict() for s in self.model_scores],
            "dataset_characteristics": self.dataset_characteristics.to_dict(),
            "selection_criterion": self.selection_criterion.value,
            "problem_type": self.problem_type.value,
            "selection_time": self.selection_time,
            "total_models_evaluated": self.total_models_evaluated,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    def get_top_models(self, n: int = 5) -> List[ModelScore]:
        return sorted(self.model_scores, key=lambda x: x.total_score, reverse=True)[:n]


# === REJESTR MODELI (cechy + przeznaczenie) ===
class ModelRegistry:
    """
    Mapa modeli do kategorii i właściwości – używana do punktacji i filtracji.
    """
    MODELS: Dict[str, Dict[str, Any]] = {
        # Linear
        "logistic_regression": {
            "category": ModelCategory.LINEAR, "interpretability": 5, "speed": 5,
            "memory": 5, "scalability": 5, "handles_categorical": False,
            "handles_missing": False, "handles_imbalance": True,
            "best_for": ["small_data", "interpretable", "linear_relationships"],
            "not_recommended_for": ["non_linear", "very_large_data"]
        },
        "ridge": {
            "category": ModelCategory.LINEAR, "interpretability": 5, "speed": 5,
            "memory": 5, "scalability": 5, "handles_categorical": False,
            "handles_missing": False, "handles_imbalance": False,
            "best_for": ["small_data", "regularization"],
            "not_recommended_for": ["non_linear"]
        },
        "lasso": {
            "category": ModelCategory.LINEAR, "interpretability": 5, "speed": 5,
            "memory": 5, "scalability": 5, "handles_categorical": False,
            "handles_missing": False, "handles_imbalance": False,
            "best_for": ["feature_selection", "sparse_solutions"],
            "not_recommended_for": ["non_linear", "high_correlation"]
        },

        # Trees / Ensembles
        "decision_tree": {
            "category": ModelCategory.TREE_BASED, "interpretability": 4, "speed": 4,
            "memory": 3, "scalability": 3, "handles_categorical": True,
            "handles_missing": True, "handles_imbalance": False,
            "best_for": ["interpretable", "non_linear", "mixed_features"],
            "not_recommended_for": ["high_variance"]
        },
        "random_forest": {
            "category": ModelCategory.ENSEMBLE, "interpretability": 3, "speed": 3,
            "memory": 2, "scalability": 3, "handles_categorical": True,
            "handles_missing": True, "handles_imbalance": True,
            "best_for": ["non_linear", "robust", "general_purpose", "feature_importance"],
            "not_recommended_for": ["very_large_data", "low_latency"]
        },
        "extra_trees": {
            "category": ModelCategory.ENSEMBLE, "interpretability": 3, "speed": 4,
            "memory": 2, "scalability": 3, "handles_categorical": True,
            "handles_missing": True, "handles_imbalance": True,
            "best_for": ["non_linear", "fast_training", "variance_reduction"],
            "not_recommended_for": ["very_large_data", "high_latency_sensitive"]
        },

        # Boosting
        "xgboost": {
            "category": ModelCategory.BOOSTING, "interpretability": 2, "speed": 3,
            "memory": 3, "scalability": 4, "handles_categorical": True,
            "handles_missing": True, "handles_imbalance": True,
            "best_for": ["high_performance", "structured_data"],
            "not_recommended_for": ["need_interpretability"]
        },
        "lightgbm": {
            "category": ModelCategory.BOOSTING, "interpretability": 2, "speed": 5,
            "memory": 4, "scalability": 5, "handles_categorical": True,
            "handles_missing": True, "handles_imbalance": True,
            "best_for": ["large_data", "fast_training"],
            "not_recommended_for": ["tiny_data"]
        },
        "catboost": {
            "category": ModelCategory.BOOSTING, "interpretability": 2, "speed": 3,
            "memory": 3, "scalability": 4, "handles_categorical": True,
            "handles_missing": True, "handles_imbalance": True,
            "best_for": ["categorical_heavy", "minimal_tuning"],
            "not_recommended_for": ["tiny_data"]
        },
        "gradient_boosting": {
            "category": ModelCategory.BOOSTING, "interpretability": 2, "speed": 2,
            "memory": 3, "scalability": 3, "handles_categorical": False,
            "handles_missing": False, "handles_imbalance": True,
            "best_for": ["controlled_tuning", "high_performance"],
            "not_recommended_for": ["very_large_data", "fast_training"]
        },

        # NN
        "mlp": {
            "category": ModelCategory.NEURAL_NETWORK, "interpretability": 1, "speed": 2,
            "memory": 2, "scalability": 3, "handles_categorical": False,
            "handles_missing": False, "handles_imbalance": False,
            "best_for": ["non_linear", "complex_patterns"],
            "not_recommended_for": ["small_data", "fast_training"]
        },

        # SVM / NB / KNN
        "svm": {
            "category": ModelCategory.SVM, "interpretability": 2, "speed": 2,
            "memory": 2, "scalability": 2, "handles_categorical": False,
            "handles_missing": False, "handles_imbalance": False,
            "best_for": ["small_data", "high_dimensional"],
            "not_recommended_for": ["very_large_data"]
        },
        "knn": {
            "category": ModelCategory.KNN, "interpretability": 4, "speed": 2,
            "memory": 1, "scalability": 1, "handles_categorical": False,
            "handles_missing": False, "handles_imbalance": True,
            "best_for": ["small_data", "local_patterns"],
            "not_recommended_for": ["large_data", "fast_prediction"]
        },
        "naive_bayes": {
            "category": ModelCategory.NAIVE_BAYES, "interpretability": 4, "speed": 5,
            "memory": 5, "scalability": 5, "handles_categorical": True,
            "handles_missing": True, "handles_imbalance": False,
            "best_for": ["text", "small_data", "fast_training"],
            "not_recommended_for": ["strong_correlations", "complex_patterns"]
        },
    }

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        return cls.MODELS.get(model_name.lower())

    @classmethod
    def get_all_models(cls) -> List[str]:
        return list(cls.MODELS.keys())

    @classmethod
    def get_models_by_category(cls, category: ModelCategory) -> List[str]:
        return [m for m, info in cls.MODELS.items() if info["category"] == category]


# === FABRYKA ESTYMATORÓW (sklearn) ===
class EstimatorFactory:
    @staticmethod
    def make(model_name: str, problem: ProblemType, quick: bool = False, random_state: int = 42) -> Optional[BaseEstimator]:
        """Tworzy gotowy estimator (opcjonalnie z lekkim tuningiem pod quick)."""
        rs = random_state
        mn = model_name.lower()

        # Regresja
        if problem == ProblemType.REGRESSION:
            if mn == "ridge": return Ridge(alpha=1.0, random_state=None)
            if mn == "lasso": return Lasso(alpha=0.001, max_iter=5000, random_state=None)
            if mn == "random_forest":
                return RandomForestRegressor(n_estimators=200 if not quick else 80, max_depth=None, n_jobs=-1, random_state=rs)
            if mn == "extra_trees":
                return ExtraTreesRegressor(n_estimators=300 if not quick else 100, n_jobs=-1, random_state=rs)
            if mn == "gradient_boosting":
                return GradientBoostingRegressor(random_state=rs)
            if mn == "xgboost" and xgb is not None:
                return xgb.XGBRegressor(
                    n_estimators=600 if not quick else 200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=rs, tree_method="hist"
                )
            if mn == "lightgbm" and lgb is not None:
                return lgb.LGBMRegressor(
                    n_estimators=1000 if not quick else 300, learning_rate=0.05, num_leaves=64, subsample=0.8,
                    colsample_bytree=0.8, random_state=rs, n_jobs=-1
                )
            if mn == "catboost" and CatBoostRegressor is not None:
                return CatBoostRegressor(
                    depth=8, iterations=1200 if not quick else 400, learning_rate=0.05,
                    loss_function="RMSE", random_state=rs, verbose=False
                )
            if mn == "mlp":
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64) if not quick else (64,),
                                         max_iter=400 if not quick else 200, random_state=rs))
                ])
            if mn == "svm":
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("svr", SVR(C=1.0, epsilon=0.1))
                ])

        # Klasyfikacja (bin/multi)
        else:
            if mn == "logistic_regression":
                return Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None, random_state=rs))
                ])
            if mn == "decision_tree":
                # niewielkie ucięcie overfittingu
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier(max_depth=None if quick else None, random_state=rs, class_weight="balanced")
            if mn == "random_forest":
                return RandomForestClassifier(n_estimators=300 if not quick else 120, n_jobs=-1, random_state=rs, class_weight="balanced")
            if mn == "extra_trees":
                return ExtraTreesClassifier(n_estimators=400 if not quick else 150, n_jobs=-1, random_state=rs, class_weight="balanced")
            if mn == "gradient_boosting":
                return GradientBoostingClassifier(random_state=rs)
            if mn == "xgboost" and xgb is not None:
                return xgb.XGBClassifier(
                    n_estimators=700 if not quick else 250, max_depth=6, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=rs, tree_method="hist",
                    eval_metric="logloss"
                )
            if mn == "lightgbm" and lgb is not None:
                return lgb.LGBMClassifier(
                    n_estimators=1200 if not quick else 400, learning_rate=0.05, num_leaves=63,
                    subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
                    n_jobs=-1, random_state=rs
                )
            if mn == "catboost" and CatBoostClassifier is not None:
                return CatBoostClassifier(
                    iterations=1200 if not quick else 400, learning_rate=0.05,
                    depth=8, random_state=rs, verbose=False, auto_class_weights="Balanced"
                )
            if mn == "mlp":
                return Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64) if not quick else (64,),
                                           max_iter=400 if not quick else 200, random_state=rs))
                ])
            if mn == "svm":
                return Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("svc", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=rs))
                ])
            if mn == "knn":
                return Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),
                    ("knn", KNeighborsClassifier(n_neighbors=5))
                ])
            if mn == "naive_bayes":
                return GaussianNB()

        return None


# === GŁÓWNY SELEKTOR ===
class ModelSelector:
    def __init__(self, criterion: SelectionCriterion = SelectionCriterion.BALANCED, random_state: int = 42):
        self.criterion = criterion
        self.result: Optional[SelectionResult] = None
        self.random_state = int(random_state)
        self.rng = check_random_state(self.random_state)
        self.criterion_weights = self._get_criterion_weights()
        logger.info(f"ModelSelector initialized with criterion={self.criterion}")

    # — WAGI —
    def _get_criterion_weights(self) -> Dict[str, float]:
        weights = {
            SelectionCriterion.ACCURACY:       {"performance": 0.70, "speed": 0.10, "interpretability": 0.05, "robustness": 0.10, "scalability": 0.025, "memory": 0.025},
            SelectionCriterion.SPEED:          {"performance": 0.30, "speed": 0.40, "interpretability": 0.05, "robustness": 0.10, "scalability": 0.10,  "memory": 0.05},
            SelectionCriterion.INTERPRETABILITY:{"performance": 0.20, "speed": 0.10, "interpretability": 0.50, "robustness": 0.10, "scalability": 0.05, "memory": 0.05},
            SelectionCriterion.ROBUSTNESS:     {"performance": 0.30, "speed": 0.05, "interpretability": 0.05, "robustness": 0.40, "scalability": 0.10,  "memory": 0.10},
            SelectionCriterion.SCALABILITY:    {"performance": 0.25, "speed": 0.15, "interpretability": 0.05, "robustness": 0.10, "scalability": 0.35, "memory": 0.10},
            SelectionCriterion.MEMORY:         {"performance": 0.25, "speed": 0.10, "interpretability": 0.05, "robustness": 0.10, "scalability": 0.10,  "memory": 0.40},
            SelectionCriterion.BALANCED:       {"performance": 0.30, "speed": 0.15, "interpretability": 0.15, "robustness": 0.15, "scalability": 0.15, "memory": 0.10},
        }
        return weights[self.criterion]

    # — PUBLIC API —
    def select_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: Optional[str] = None,
        n_models: int = 5,
        quick_mode: bool = False,
        do_quick_cv: bool = False,
        max_rows_for_cv: int = 25000,
    ) -> SelectionResult:
        """
        Główna metoda selekcji modeli.

        Args:
            X, y: dane
            problem_type: "classification"|"regression"|None (auto)
            n_models: ile modeli zwrócić
            quick_mode: parametry lżejsze dla fabryk
            do_quick_cv: czy wykonać szybkie CV na próbce dla kalibracji performance_score
            max_rows_for_cv: ile max wierszy użyć w CV (samplowanie losowe)

        Returns:
            SelectionResult
        """
        start = datetime.now()

        # 1) Meta-cechy
        characteristics = self._extract_characteristics(X, y)

        # 2) Detekcja problemu
        if problem_type is None:
            problem_type = self._detect_problem_type(y, characteristics)
        problem_enum = self._map_problem_type(problem_type, characteristics)
        logger.info(f"Detected problem: {problem_enum.value}")

        # 3) Modele kandydujące
        applicable = self._get_applicable_models(problem_type)
        logger.info(f"Applicable models: {len(applicable)}")

        # 4) Punktacja – reguły + (opcjonalnie) szybki benchmark
        # Szybkie CV (sample) – pomaga skalibrować performance_score
        cv_perf: Dict[str, float] = {}
        if do_quick_cv and len(X) > 2:
            X_cv, y_cv = self._maybe_sample_for_cv(X, y, max_rows_for_cv)
            cv_perf = self._quick_cv_benchmark(applicable, X_cv, y_cv, problem_enum, quick_mode)

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

        model_scores.sort(key=lambda s: s.total_score, reverse=True)
        recommended = [s.model_name for s in model_scores[:n_models]]

        # 5) Wynik
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
        logger.info(f"Selection finished in {selection_time:.2f}s. Top-3: {recommended[:3]}")
        return self.result

    # — META-CECHY —
    def _extract_characteristics(self, X: pd.DataFrame, y: pd.Series) -> DatasetCharacteristics:
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])

        # Typy cech
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
        n_numerical = int(len(num_cols))
        n_categorical = int(len(cat_cols))

        # Binaria liczymy niezależnie od dtype
        n_binary = int(sum((pd.Series(X[c]).nunique(dropna=True) == 2) for c in X.columns))

        # Klasy (jeżeli wygląda na kategoryczny target albo mało unikatów)
        n_unique_y = int(pd.Series(y).nunique(dropna=True))
        if (y.dtype == "object") or (n_unique_y <= 20 and not pd.api.types.is_float_dtype(y)):
            n_classes = n_unique_y
            cb = pd.Series(y).value_counts(normalize=True, dropna=False).to_dict()
        else:
            n_classes, cb = None, None

        # Właściwości statystyczne (num tylko)
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

        # Braki i rzadkość
        total_cells = max(1, n_samples * n_features)
        missing_ratio = float(pd.isna(X).sum().sum() / total_cells)
        zeros_ratio = 0.0
        try:
            # bezpiecznie dla nienumerycznych
            Z = X.select_dtypes(include=[np.number]).eq(0).sum().sum()
            N = max(1, len(X.select_dtypes(include=[np.number]).values.flatten()))
            zeros_ratio = float(Z / N) if N > 0 else 0.0
        except Exception:
            pass
        is_sparse = bool((missing_ratio > 0.3) or (zeros_ratio > 0.5))

        # Korelacja średnia (num only)
        if n_numerical > 1:
            try:
                cm = X[num_cols].corr(numeric_only=True).abs()
                upper = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
                avg_corr = float(upper.stack().mean())
            except Exception:
                avg_corr = 0.0
        else:
            avg_corr = 0.0

        f2s = float(n_features / max(1, n_samples))
        dimensionality_score = float(np.log10(max(2, n_features)) / np.log10(max(3, n_samples)))

        is_imbalanced = False
        if cb:
            try:
                is_imbalanced = (min(cb.values()) < 0.2)
            except Exception:
                is_imbalanced = False

        is_high_dimensional = bool(f2s > 0.1)

        # Heurystyka TS (jeśli index datetime i posortowany)
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

    # — DETEKCJA PROBLEMU —
    def _detect_problem_type(self, y: pd.Series, ch: DatasetCharacteristics) -> str:
        n_unique = int(pd.Series(y).nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(y):
            # liczbowe: mało unikatów => klasyfikacja, dużo => regresja
            return "classification" if n_unique <= 20 else "regression"
        return "classification"

    def _map_problem_type(self, kind: str, ch: DatasetCharacteristics) -> ProblemType:
        if kind == "regression":
            return ProblemType.REGRESSION
        # klasyfikacja
        if ch.n_classes == 2:
            if ch.is_imbalanced:
                return ProblemType.IMBALANCED
            return ProblemType.BINARY_CLASSIFICATION
        if ch.is_high_dimensional:
            return ProblemType.HIGH_DIMENSIONAL
        return ProblemType.MULTICLASS_CLASSIFICATION

    # — MODELE KANDYDUJĄCE —
    def _get_applicable_models(self, problem_type: str) -> List[str]:
        if problem_type == "regression":
            models = ["ridge", "lasso", "random_forest", "extra_trees",
                      "xgboost", "lightgbm", "catboost", "gradient_boosting",
                      "mlp", "svm"]
        else:
            models = ["logistic_regression", "decision_tree", "random_forest",
                      "extra_trees", "xgboost", "lightgbm", "catboost",
                      "gradient_boosting", "mlp", "svm", "knn", "naive_bayes"]

        # wytnij boostingi, jeśli brak bibliotek
        filtered: List[str] = []
        for m in models:
            if m == "xgboost" and xgb is None: continue
            if m == "lightgbm" and lgb is None: continue
            if m == "catboost" and (CatBoostClassifier is None and CatBoostRegressor is None): continue
            filtered.append(m)
        return filtered

    # — SZYBKI BENCHMARK CV (opcjonalny) —
    def _maybe_sample_for_cv(self, X: pd.DataFrame, y: pd.Series, max_rows: int) -> Tuple[pd.DataFrame, pd.Series]:
        if len(X) <= max_rows:
            return X, y
        # stratified jeśli to klasyfikacja bin/multi
        try:
            # heurystyka: jeżeli y ma mało klas
            if pd.Series(y).nunique(dropna=True) <= 20:
                # stratyfikowane losowanie – zachowaj proporcje
                df = X.copy()
                df["_y_"] = y.values
                sample = df.groupby("_y_", group_keys=False).apply(lambda g: g.sample(min(len(g), max_rows // max(1, df["_y_"].nunique())), random_state=self.random_state))
                y_s = sample.pop("_y_")
                return sample, y_s
        except Exception:
            pass
        return X.sample(max_rows, random_state=self.random_state), y.loc[X.sample(max_rows, random_state=self.random_state).index]

    def _quick_cv_benchmark(
        self,
        models: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        problem: ProblemType,
        quick_mode: bool
    ) -> Dict[str, float]:
        """Szybkie CV (3-fold) z lekkimi estymatorami — wynik w [0,1] po normalizacji."""
        cv_perf: Dict[str, float] = {}
        if len(X) < 40:
            return cv_perf

        # Folds + scoring
        if problem == ProblemType.REGRESSION:
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scoring_name = "r2"
        else:
            # klasyfikacja: preferuj ROC AUC jeśli >2 klasy => ovr
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scoring_name = "roc_auc_ovr" if pd.Series(y).nunique(dropna=True) > 2 else "roc_auc"

        for m in models:
            est = EstimatorFactory.make(m, problem, quick=quick_mode, random_state=self.random_state)
            if est is None:
                continue
            try:
                scores = cross_val_score(est, X, y, cv=cv, scoring=scoring_name, n_jobs=-1)
                # przemapuj rezultat do [0,1]
                s = float(np.clip(np.mean(scores), -1, 1))
                # r2 może być ujemne — normalizacja do [0,1]
                if scoring_name == "r2":
                    s = (s + 1) / 2.0
                cv_perf[m] = s
            except Exception as e:
                logger.warning(f"Quick CV failed for {m}: {e}")
        return cv_perf

    # — PUNKTACJA MODELU —
    def _score_model(
        self,
        model_name: str,
        characteristics: DatasetCharacteristics,
        problem_type: ProblemType,
        quick_mode: bool,
        cv_perf: Optional[float] = None
    ) -> ModelScore:
        info = ModelRegistry.get_model_info(model_name) or {}
        category = info.get("category", ModelCategory.LINEAR)

        # stałe podstawowe z rejestru (skalujemy do [0,1])
        interpretability = float(info.get("interpretability", 3)) / 5.0
        speed = float(info.get("speed", 3)) / 5.0
        memory = float(info.get("memory", 3)) / 5.0
        scalability = float(info.get("scalability", 3)) / 5.0

        # przewidywana wydajność (rule-based + opcjonalna kalibracja CV)
        perf = self._estimate_performance(model_name, info, characteristics, problem_type)

        # kalibracja perf wynikiem CV (jeśli dostępny)
        if cv_perf is not None:
            # blend: 70% reguł + 30% CV
            perf = float(np.clip(0.7 * perf + 0.3 * cv_perf, 0, 1))

        # odporność (ensembles/trees + brak danych + skośność/ogony)
        robust = self._estimate_robustness(model_name, info, characteristics)

        # Korekty na high-dimensional & imbalance
        warnings: List[str] = []
        reasoning: List[str] = []

        if characteristics.is_high_dimensional:
            if "high_dimensional" in info.get("best_for", []):
                perf += 0.05
                reasoning.append("✓ Dobrze radzi sobie w high-dimensional")
            elif "high_dimensional" in info.get("not_recommended_for", []):
                perf *= 0.85
                warnings.append("⚠️ Może mieć trudności w high-dimensional")

        if characteristics.is_imbalanced:
            if info.get("handles_imbalance", False):
                perf += 0.03
                robust += 0.05
                reasoning.append("✓ Wsparcie dla niezbalansowanych klas")
            else:
                perf *= 0.93
                warnings.append("⚠️ Brak natywnej obsługi niezbalansowania")

        # Kategoryczne + braki
        if characteristics.n_categorical > 0 and info.get("handles_categorical", False):
            reasoning.append("✓ Obsługuje zmienne kategoryczne")
        if characteristics.missing_ratio > 0.1:
            if info.get("handles_missing", False):
                reasoning.append("✓ Odporność na braki danych")
            else:
                perf *= 0.95
                warnings.append("⚠️ Wymaga imputacji (braki danych >10%)")

        # Korelacje i skośności
        if characteristics.avg_correlation > 0.8 and model_name in {"lasso", "ridge"}:
            # L1/L2 pomaga na współliniowość
            perf += 0.03
            reasoning.append("✓ Regularizacja łagodzi współliniowość")
        if (characteristics.avg_skewness > 2 or characteristics.avg_kurtosis > 3) and category in {
            ModelCategory.TREE_BASED, ModelCategory.ENSEMBLE, ModelCategory.BOOSTING
        }:
            robust += 0.05
            reasoning.append("✓ Drzewa/ensembles odporne na skośne/ciężkie ogony")

        # Not-recommended flags
        for flag in info.get("not_recommended_for", []):
            if flag == "very_large_data" and characteristics.n_samples > 500_000:
                perf *= 0.9; warnings.append("⚠️ Może być wolny przy bardzo dużych zbiorach")
            if flag == "small_data" and characteristics.n_samples < 1000:
                perf *= 0.9; warnings.append("⚠️ Może niedomagać na małych zbiorach")

        # złożenie punktacji wielokryterialnej
        w = self.criterion_weights
        total = float(
            w["performance"] * perf +
            w["speed"] * speed +
            w["interpretability"] * interpretability +
            w["robustness"] * robust +
            w["scalability"] * scalability +
            w["memory"] * memory
        )

        # rekomendacja
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

    # — HEURYSTYCZNY PERFORMANCE —
    def _estimate_performance(
        self,
        model_name: str,
        info: Dict[str, Any],
        ch: DatasetCharacteristics,
        problem_type: ProblemType
    ) -> float:
        base = 0.60  # bazowe 60%

        # Skala danych
        if ch.n_samples < 1000 and "small_data" in info.get("best_for", []):
            base += 0.12
        if ch.n_samples > 100_000 and "large_data" in info.get("best_for", []):
            base += 0.10

        # High-dimensional & categorical heavy
        if ch.is_high_dimensional and "high_dimensional" in info.get("best_for", []):
            base += 0.08
        if (ch.n_categorical > ch.n_numerical / 2) and ("categorical_heavy" in info.get("best_for", [])):
            base += 0.08

        # Ensembling & boosting – mały boost
        if info.get("category") in {ModelCategory.BOOSTING, ModelCategory.ENSEMBLE}:
            base += 0.06

        # Penalizacja za brak dopasowania
        if ch.n_samples < 1000 and "small_data" in info.get("not_recommended_for", []):
            base -= 0.10
        if ch.is_high_dimensional and "high_dimensional" in info.get("not_recommended_for", []):
            base -= 0.08

        # Bardzo wysoka wielowymiarowość – kary dla kNN / MLP / SVM (bez kerneli liniowych)
        if ch.feature_to_sample_ratio > 0.5 and model_name in {"knn", "mlp", "svm"}:
            base -= 0.08

        # Imbalance – lekkie zwiększenie dla metod z class_weight lub boostingów
        if ch.is_imbalanced and (info.get("handles_imbalance", False) or info.get("category") == ModelCategory.BOOSTING):
            base += 0.04

        return float(np.clip(base, 0, 1))

    # — HEURYSTYCZNA ODPORNOŚĆ —
    def _estimate_robustness(self, model_name: str, info: Dict[str, Any], ch: DatasetCharacteristics) -> float:
        base = 0.50
        if info.get("category") in {ModelCategory.ENSEMBLE, ModelCategory.BOOSTING}:
            base += 0.25
        if ch.missing_ratio > 0.10:
            base += 0.05 if info.get("handles_missing", False) else -0.05
        if ch.avg_skewness > 2 or ch.avg_kurtosis > 3:
            if info.get("category") in {ModelCategory.TREE_BASED, ModelCategory.ENSEMBLE, ModelCategory.BOOSTING}:
                base += 0.05
        return float(np.clip(base, 0, 1))

    # — RAPORTY/EXPLAIN —
    def get_explanation(self, model_name: Optional[str] = None) -> str:
        if self.result is None:
            return "No selection result available. Run select_models() first."

        if model_name:
            s = next((x for x in self.result.model_scores if x.model_name == model_name), None)
            if not s:
                return f"Model {model_name} not found in results."
            return (
                f"\n🤖 Model: {s.model_name.upper()}\n"
                f"Kategoria: {s.model_category.value}\n\n"
                f"📊 Scores:\n"
                f"  • Performance: {s.performance_score:.2f}\n"
                f"  • Speed: {s.speed_score:.2f}\n"
                f"  • Interpretability: {s.interpretability_score:.2f}\n"
                f"  • Robustness: {s.robustness_score:.2f}\n"
                f"  • Scalability: {s.scalability_score:.2f}\n"
                f"  • Memory: {s.memory_score:.2f}\n\n"
                f"  ⭐ Total Score: {s.total_score:.2f}\n\n"
                f"✅ Plusy:\n" + ("\n".join(f"  {r}" for r in s.reasoning) if s.reasoning else "  —") + "\n" +
                (("⚠️ Uwagi:\n" + "\n".join(f"  {w}" for w in s.warnings)) if s.warnings else "") + "\n" +
                ("✅ RECOMMENDED" if s.recommended else "❌ NOT RECOMMENDED")
            )

        # zbiorczo
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
            f"  • F2S: {ds.feature_to_sample_ratio:.3f}",
            f"  • High-Dim: {ds.is_high_dimensional}",
            "",
            "🏆 Top Recommended Models:"
        ]
        for i, m in enumerate(top, 1):
            s = next(s for s in self.result.model_scores if s.model_name == m)
            out.append(f"  {i}. {m.upper()} (score: {s.total_score:.2f})")
        return "\n".join(out)

    def export_report(self, filepath: Union[str, Path]):
        if self.result is None:
            raise ValueError("No result to export. Run select_models() first.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Report exported to {path}")


# === FUNKCJA WYGODNA ===
def select_best_models(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: SelectionCriterion = SelectionCriterion.BALANCED,
    n_models: int = 5,
    **kwargs
) -> SelectionResult:
    """
    Przykład:
        result = select_best_models(
            X=X_train, y=y_train,
            criterion=SelectionCriterion.ACCURACY,
            n_models=5,
            quick_mode=True,
            do_quick_cv=True
        )
    """
    selector = ModelSelector(criterion)
    return selector.select_models(X, y, n_models=n_models, **kwargs)
