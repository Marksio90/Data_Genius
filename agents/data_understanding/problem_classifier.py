# === OPIS MODUŁU ===
"""
DataGenius PRO - Problem Classifier (PRO+++)
Klasyfikacja typu problemu ML (classification vs regression) na podstawie kolumny celu.
Zwraca analizę targetu i rekomendacje metryk, preprocessing'u i modeli.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.utils import infer_problem_type
from config.model_registry import ProblemType


# === MODELE DANYCH / KONFIG ===
@dataclass(frozen=True)
class ProblemClassifierConfig:
    """Parametry i progi heurystyczne dla analizy targetu."""
    imbalance_ratio_threshold: float = 3.0  # >3 uznajemy za istotną nierównowagę klas
    warn_min_samples_per_class: int = 20    # ostrzeżenie dla rzadkich klas
    skew_warn_abs: float = 1.0              # |skew| > 1 ⇒ rozważ transformację
    min_samples_required: int = 1           # minimalna liczba obserwacji po dropna()


# === KLASA GŁÓWNA AGENDA ===
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

    # === WALIDACJA WEJŚCIA ===
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

    # === GŁÓWNE WYKONANIE ===
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
                logger.error(msg)
                return result

            target = data[target_column]

            # Guard: pusty target lub brak niepustych wartości
            if target is None or len(target) == 0 or target.dropna().empty:
                payload = {
                    "problem_type": None,
                    "target_analysis": {
                        "n_samples": int(len(target)) if target is not None else 0,
                        "n_unique": int(target.nunique(dropna=True)) if target is not None else 0,
                        "n_missing": int(target.isna().sum()) if target is not None else 0,
                        "missing_pct": float((target.isna().sum() / max(1, len(target))) * 100) if target is not None else 0.0,
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
                result.data = payload
                logger.warning("ProblemClassifier: empty target – cannot infer problem type.")
                return result

            # Detekcja typu problemu (obsługa Enum/string)
            detected = infer_problem_type(target)
            if isinstance(detected, ProblemType):
                problem_type_str = "classification" if detected == ProblemType.CLASSIFICATION else "regression"
            else:
                problem_type_str = str(detected).lower().strip()

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

            logger.success(f"Problem classified as: {problem_type_str}")

        except Exception as e:
            result.add_error(f"Problem classification failed: {e}")
            logger.exception(f"Problem classification error: {e}")

        return result

    # === ANALIZA TARGETU (DYNA.) ===
    def _analyze_target(
        self,
        target: pd.Series,
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Analyze target column in detail depending on problem type.
        """
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

    # === ANALIZA DLA KLASYFIKACJI ===
    def _analyze_classification_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze classification target: liczba klas, rozkład, nierównowaga."""
        cfg = self.config

        # Wartości i rozkład; dropna=False, aby jasno widzieć potencjalne NaN
        vc = target.value_counts(dropna=False)
        # Statystyki tylko dla nie-NaN (klasy faktyczne)
        vc_non_na = target.dropna().value_counts()
        n_classes = int(vc_non_na.shape[0])

        is_binary = (n_classes == 2)
        warnings: list[str] = []

        # Imbalance
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

        # Majority class (w oparciu o vc_non_na)
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

    # === ANALIZA DLA REGRESJI ===
    def _analyze_regression_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze regression target: statystyki opisowe z guardami."""
        cfg = self.config
        target_clean = target.dropna()

        if len(target_clean) < cfg.min_samples_required:
            return {
                "error": "Insufficient non-missing target values for regression analysis",
                "n_non_missing": int(len(target_clean))
            }

        # Upewniamy się, że da się zrzutować na float (niektóre obiekty mogą być numeryczne w stringach)
        try:
            t = pd.to_numeric(target_clean, errors="coerce").dropna()
        except Exception:
            t = target_clean

        if t.empty:
            return {
                "error": "Target values are non-numeric or cannot be coerced to numeric",
                "n_non_missing": int(len(target_clean))
            }

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

    # === REKOMENDACJE (DYNA.) ===
    def _get_recommendations(
        self,
        target: pd.Series,
        problem_type: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build recommendations based on problem type and target analysis.
        """
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

        # merge
        rec.update(out)
        return rec

    # === REKOMENDACJE DLA KLASYFIKACJI ===
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
            metrics = []
            warnings_local.append("Single-class target: classification not feasible until more classes appear.")
        elif n_classes == 2:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        else:
            metrics = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]

        if is_imbalanced and "balanced_accuracy" not in metrics:
            metrics.append("balanced_accuracy")

        # Preprocessing
        preprocessing = [
            "Handle missing values in target",
            "Encode categorical target if needed",
        ]
        if is_imbalanced:
            preprocessing.append("Apply SMOTE or class weighting for imbalanced data")

        # Models
        if n_classes <= 1:
            models = []
        elif n_classes == 2:
            models = [
                "Logistic Regression (baseline)",
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "CatBoost",
            ]
        else:
            models = [
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "CatBoost",
            ]

        return {
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
            "warnings": warnings_local,
        }

    # === REKOMENDACJE DLA REGRESJI ===
    def _get_regression_recommendations(
        self,
        target: pd.Series,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        cfg = self.config
        warnings_local: list[str] = []

        # Metrics
        metrics = ["mae", "mse", "rmse", "r2", "mape"]

        # Preprocessing
        preprocessing = [
            "Handle missing values in target",
            "Check and cap/transform outliers in target if needed",
        ]

        # Skewness hint
        skewness = float(analysis.get("skewness", 0.0))
        if abs(skewness) > cfg.skew_warn_abs:
            preprocessing.append(f"Consider log/Box-Cox transformation (|skew|={abs(skewness):.2f})")

        # Numeric coercion hint
        if "error" in analysis and "numeric" in str(analysis["error"]).lower():
            warnings_local.append("Target not numeric — ensure numeric dtype or proper coercion.")

        # Models
        models = [
            "Linear Regression (baseline)",
            "Ridge / Lasso",
            "ElasticNet",
            "Random Forest Regressor",
            "XGBoost Regressor",
            "LightGBM Regressor",
            "CatBoost Regressor",
        ]

        return {
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
            "warnings": warnings_local,
        }
