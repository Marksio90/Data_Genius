# === OPIS MODUŁU ===
"""
DataGenius PRO - Problem Classifier (PRO++++)
Klasyfikacja typu problemu ML (classification vs regression) na podstawie kolumny celu.
Zwraca analizę targetu i rekomendacje metryk, preprocessing'u i modeli.

Kontrakt (AgentResult.data):
{
    "problem_type": "classification" | "regression" | None,
    "target_analysis": Dict[str, Any],
    "recommendations": {
        "problem_type": str | None,
        "suggested_metrics": List[str],
        "preprocessing_steps": List[str],
        "model_suggestions": List[str],
        "warnings": List[str]
    }
}
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Callable
import time
import json

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.utils import infer_problem_type
from config.model_registry import ProblemType

# === NAZWA_SEKCJI === MODELE DANYCH / KONFIG ===
@dataclass(frozen=True)
class ProblemClassifierConfig:
    """Parametry i progi heurystyczne dla analizy targetu."""
    imbalance_ratio_threshold: float = 3.0    # >3 uznajemy za istotną nierównowagę klas
    warn_min_samples_per_class: int = 20      # ostrzeżenie dla rzadkich klas
    skew_warn_abs: float = 1.0                # |skew| > 1 ⇒ rozważ transformację
    min_samples_required: int = 1             # minimalna liczba obserwacji po dropna()
    id_like_unique_ratio: float = 0.98        # ~98%+ unikalnych sugeruje ID/sygn. ciągłą
    treat_numeric_str_as_numeric: bool = True # "123" → 123 przy regresji
    truncate_log_chars: int = 400             # cięcie długich struktur w logach

# === NAZWA_SEKCJI === HELPERY ===
def _timeit(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = (time.perf_counter() - t0) * 1000
                logger.debug(f"{name}: {dt:.1f} ms")
        return wrapped
    return deco

def _truncate(obj: Any, limit: int) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s)-limit} chars)"

def _is_numeric_series_like(s: pd.Series) -> bool:
    """Heurystyka: czy seria może być traktowana jako numeryczna po konwersji."""
    if pd.api.types.is_numeric_dtype(s):
        return True
    if s.dtype == "object":
        ss = pd.to_numeric(s.dropna().astype(str), errors="coerce")
        return not ss.dropna().empty and (ss.dropna().shape[0] / max(1, s.dropna().shape[0])) > 0.9
    return False

def _is_id_like(s: pd.Series, ratio_threshold: float) -> bool:
    """Czy kolumna wygląda na ID (prawie same unikalne wartości)."""
    try:
        n_unique = int(s.nunique(dropna=True))
        n = int(len(s))
        return n > 0 and (n_unique / n) >= ratio_threshold
    except Exception:
        return False

# === NAZWA_SEKCJI === KLASA GŁÓWNA AGENDA ===
class ProblemClassifier(BaseAgent):
    """
    Klasyfikuje typ problemu ML na podstawie targetu (classification vs regression),
    wykonuje analizę kolumny celu i zwraca rekomendacje metryk, preprocessing'u i modeli.
    """

    def __init__(self, config: Optional[ProblemClassifierConfig] = None) -> None:
        super().__init__(
            name="ProblemClassifier",
            description="Classifies ML problem as classification or regression"
        )
        self.config = config or ProblemClassifierConfig()
        self._log = logger.bind(agent="ProblemClassifier")

    # === NAZWA_SEKCJI === WALIDACJA WEJŚCIA ===
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        Expected:
            data: pd.DataFrame (required)
            target_column: str (required)
        """
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        if "target_column" not in kwargs:
            raise ValueError("'target_column' parameter is required")

        data = kwargs["data"]
        target_column = kwargs["target_column"]

        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame")
        if not isinstance(target_column, str):
            raise TypeError("'target_column' must be a string")

        return True

    # === NAZWA_SEKCJI === GŁÓWNE WYKONANIE ===
    @_timeit("ProblemClassifier.execute")
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        **kwargs: Any
    ) -> AgentResult:
        """
        Classify problem type and produce analysis + recommendations.

        Args:
            data: Input DataFrame
            target_column: Name of target column

        Returns:
            AgentResult with problem classification payload
        """
        result = AgentResult(agent_name=self.name)

        try:
            # Walidacja obecności kolumny
            if target_column not in data.columns:
                msg = f"Target column '{target_column}' not found"
                result.add_error(msg)
                self._log.error(msg)
                return result

            target = data[target_column]

            # Guard: pusty target lub brak niepustych wartości
            if target is None or len(target) == 0 or target.dropna().empty:
                payload = self._empty_payload_empty_target(target)
                result.data = payload
                self._log.warning("empty target – cannot infer problem type.")
                return result

            # Detekcja typu problemu (Enum/string) + heurystyka awaryjna
            detected = infer_problem_type(target)
            if isinstance(detected, ProblemType):
                problem_type_str = "classification" if detected == ProblemType.CLASSIFICATION else "regression"
            else:
                problem_type_str = str(detected).lower().strip()

            # Heurystyka: jeśli większość wartości jest unikalna lub seria jest numeric-like → regresja
            if problem_type_str not in ("classification", "regression"):
                if _is_id_like(target, self.config.id_like_unique_ratio) or _is_numeric_series_like(target):
                    problem_type_str = "regression"
                else:
                    # fallback do classification (bardziej bezpieczny dla stringowych celów)
                    problem_type_str = "classification"

            # Analiza targetu
            analysis = self._analyze_target(target, problem_type_str)

            # Rekomendacje
            recommendations = self._get_recommendations(target, problem_type_str, analysis)

            # Zbiorczy kontrakt
            result.data = {
                "problem_type": problem_type_str,
                "target_analysis": analysis,
                "recommendations": recommendations,
            }

            self._log.success(f"classified as: {problem_type_str}")

        except Exception as e:
            result.add_error(f"Problem classification failed: {e}")
            self._log.exception(f"Problem classification error: {e}")

        return result

    # === NAZWA_SEKCJI === PAYLOAD DLA PUSTEGO TARGETU ===
    def _empty_payload_empty_target(self, target: Optional[pd.Series]) -> Dict[str, Any]:
        n = int(len(target)) if target is not None else 0
        n_missing = int(target.isna().sum()) if target is not None else 0
        return {
            "problem_type": None,
            "target_analysis": {
                "n_samples": n,
                "n_unique": int(target.nunique(dropna=True)) if target is not None else 0,
                "n_missing": n_missing,
                "missing_pct": float((n_missing / max(1, n)) * 100) if n > 0 else 0.0,
                "error": "Target is empty or all values are missing"
            },
            "recommendations": {
                "problem_type": None,
                "suggested_metrics": [],
                "preprocessing_steps": ["Provide/repair target column values"],
                "model_suggestions": [],
                "warnings": ["Cannot infer problem type from empty target"]
            }
        }

    # === NAZWA_SEKCJI === ANALIZA TARGETU (DYNA.) ===
    def _analyze_target(
        self,
        target: pd.Series,
        problem_type: str
    ) -> Dict[str, Any]:
        """Analyze target column in detail depending on problem type."""
        n = int(len(target))
        n_unique = int(target.nunique(dropna=True))
        n_missing = int(target.isna().sum())
        missing_pct = float((n_missing / max(1, n)) * 100)

        base = {
            "n_samples": n,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "missing_pct": missing_pct,
        }

        if problem_type == "classification":
            base.update(self._analyze_classification_target(target))
        else:
            base.update(self._analyze_regression_target(target))

        return base

    # === NAZWA_SEKCJI === ANALIZA DLA KLASYFIKACJI ===
    def _analyze_classification_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze classification target: liczba klas, rozkład, nierównowaga, rzadkie klasy."""
        cfg = self.config

        # Rozkład (tylko nie-NaN jako „klasy”)
        vc_non_na = target.dropna().value_counts()
        n_classes = int(vc_non_na.shape[0])

        is_binary = (n_classes == 2)
        warnings: List[str] = []

        if n_classes > 1:
            min_class = int(vc_non_na.min())
            max_class = int(vc_non_na.max())
            imbalance_ratio = float(max_class / max(1, min_class))
            is_imbalanced = imbalance_ratio > cfg.imbalance_ratio_threshold
            if is_imbalanced:
                warnings.append(
                    f"Class imbalance detected (ratio={imbalance_ratio:.2f} > {cfg.imbalance_ratio_threshold})"
                )
            # Rzadkie klasy
            rare_classes = {str(k): int(v) for k, v in vc_non_na.items() if v < cfg.warn_min_samples_per_class}
            if rare_classes:
                warnings.append(
                    f"Rare classes detected (<{cfg.warn_min_samples_per_class} samples): {list(rare_classes.keys())}"
                )
        else:
            imbalance_ratio = 1.0
            is_imbalanced = False

        # Majority class
        if n_classes >= 1:
            majority_class = str(vc_non_na.index[0])
            majority_pct = float((vc_non_na.iloc[0] / max(1, len(target.dropna()))) * 100)
        else:
            majority_class = None
            majority_pct = 0.0
            warnings.append("Only one unique non-NA class found — model training may not be feasible.")

        return {
            "classification_type": "binary" if is_binary else "multiclass" if n_classes > 1 else "single_class",
            "n_classes": n_classes,
            "class_distribution": {str(k): int(v) for k, v in vc_non_na.to_dict().items()},
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": float(imbalance_ratio),
            "majority_class": majority_class,
            "majority_class_pct": majority_pct,
            "warnings": warnings,
        }

    # === NAZWA_SEKCJI === ANALIZA DLA REGRESJI ===
    def _analyze_regression_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze regression target: statystyki opisowe z guardami i konwersją numeric-like."""
        cfg = self.config
        target_clean = target.dropna()

        if len(target_clean) < cfg.min_samples_required:
            return {
                "error": "Insufficient non-missing target values for regression analysis",
                "n_non_missing": int(len(target_clean))
            }

        # Konwersja do float (numeric-like z obiektów)
        t = target_clean
        if cfg.treat_numeric_str_as_numeric and not pd.api.types.is_numeric_dtype(t):
            try:
                t = pd.to_numeric(target_clean.astype(str).str.replace(",", ".", regex=False), errors="coerce").dropna()
            except Exception:
                pass

        if not pd.api.types.is_numeric_dtype(t):
            # ostatni bezpiecznik
            try:
                t = pd.to_numeric(t, errors="coerce").dropna()
            except Exception:
                pass

        if isinstance(t, pd.Series) and t.empty:
            return {
                "error": "Target values are non-numeric or cannot be coerced to numeric",
                "n_non_missing": int(len(target_clean))
            }

        # Statystyki
        t = t.astype(float)
        mean = float(t.mean())
        std = float(t.std(ddof=1)) if len(t) > 1 else 0.0
        cv = float(std / mean) if mean != 0 else None
        skew = float(t.skew()) if len(t) > 2 else 0.0
        kurt = float(t.kurtosis()) if len(t) > 3 else 0.0

        return {
            "mean": mean,
            "std": std,
            "min": float(t.min()),
            "max": float(t.max()),
            "median": float(t.median()),
            "q25": float(t.quantile(0.25)),
            "q75": float(t.quantile(0.75)),
            "skewness": skew,
            "kurtosis": kurt,
            "range": float(t.max() - t.min()),
            "cv": cv,
        }

    # === NAZWA_SEKCJI === REKOMENDACJE (DYNA.) ===
    def _get_recommendations(
        self,
        target: pd.Series,
        problem_type: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build recommendations based on problem type and target analysis."""
        rec = {
            "problem_type": problem_type,
            "suggested_metrics": [],
            "preprocessing_steps": [],
            "model_suggestions": [],
            "warnings": [],
        }

        if problem_type == "classification":
            out = self._get_classification_recommendations(target, analysis)
        else:
            out = self._get_regression_recommendations(target, analysis)

        rec.update(out)
        return rec

    # === NAZWA_SEKCJI === REKOMENDACJE: KLASYFIKACJA ===
    def _get_classification_recommendations(
        self,
        target: pd.Series,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        cfg = self.config
        n_classes = int(analysis.get("n_classes", 0))
        is_imbalanced = bool(analysis.get("is_imbalanced", False))
        warnings_local = list(analysis.get("warnings", []))

        # Metrics
        if n_classes <= 1:
            metrics: List[str] = []
            warnings_local.append("Single-class target: classification not feasible until more classes appear.")
        elif n_classes == 2:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        else:
            metrics = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "log_loss"]

        if is_imbalanced and "balanced_accuracy" not in metrics:
            metrics.append("balanced_accuracy")
        # Dodatkowo przy skrajnej nierównowadze:
        if is_imbalanced:
            metrics += ["mcc", "auprc"]

        # Preprocessing
        preprocessing = [
            "Handle missing values in target",
            "Encode categorical target if needed",
            "Use stratified train/validation split",
        ]
        if is_imbalanced:
            preprocessing.append("Apply SMOTE/SMOTEENN or class weighting")

        # Models
        if n_classes <= 1:
            models: List[str] = []
        elif n_classes == 2:
            models = [
                "Logistic Regression (baseline)",
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "CatBoost",
                "Linear/Kernel SVM (z uwagą na skalowanie)"
            ]
        else:
            models = [
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "CatBoost",
                "Linear/Kernel SVM (One-vs-Rest)",
            ]

        return {
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
            "warnings": warnings_local,
        }

    # === NAZWA_SEKCJI === REKOMENDACJE: REGRESJA ===
    def _get_regression_recommendations(
        self,
        target: pd.Series,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        cfg = self.config
        warnings_local: List[str] = []

        # Metrics
        metrics = ["mae", "mse", "rmse", "r2", "mape"]

        # Preprocessing
        preprocessing = [
            "Handle missing values in target",
            "Check/cap/transform outliers in target if needed",
            "Scale features for linear/SVM models",
        ]

        # Skewness hint
        skewness = float(analysis.get("skewness", 0.0))
        if abs(skewness) > cfg.skew_warn_abs:
            preprocessing.append(f"Consider log/Box-Cox/Yeo-Johnson transformation (|skew|={abs(skewness):.2f})")

        # Numeric coercion hint
        if "error" in analysis and "numeric" in str(analysis["error"]).lower():
            warnings_local.append("Target not numeric — ensure numeric dtype or proper coercion.")

        # Models
        models = [
            "Linear Regression (baseline)",
            "Ridge / Lasso / ElasticNet",
            "Random Forest Regressor",
            "XGBoost Regressor",
            "LightGBM Regressor",
            "CatBoost Regressor",
            "SVR (dla mniejszych zbiorów, po skalowaniu)"
        ]

        return {
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
            "warnings": warnings_local,
        }
