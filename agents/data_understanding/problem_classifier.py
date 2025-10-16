# === OPIS MODUŁU ===
"""
DataGenius PRO++++++++++ - Problem Classifier (Enterprise / KOSMOS)
Klasyfikacja typu problemu ML (classification vs regression) na podstawie kolumny celu.
Wersja ENTERPRISE PRO++++++ ADV: defensywne guardy, heurystyki korekcyjne,
stabilny kontrakt danych, zwięzłe rekomendacje metryk, preprocessing'u i modeli.

⚠️ Kontrakt (AgentResult.data) — ściśle 1:1:
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

from __future__ import annotations

# === IMPORTY ===
import time
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.utils import infer_problem_type
from config.model_registry import ProblemType


# === NAZWA_SEKCJI === KONFIG / PROGI ===
@dataclass(frozen=True)
class ProblemClassifierConfig:
    """Parametry i progi heurystyczne dla analizy targetu (enterprise)."""
    imbalance_ratio_threshold: float = 3.0       # >3 uznajemy za istotną nierównowagę klas
    warn_min_samples_per_class: int = 20         # ostrzeżenie dla rzadkich klas
    rare_class_threshold: int = 5                # bardzo rzadkie klasy
    skew_warn_abs: float = 1.0                   # |skew| > 1 ⇒ transformacje sugerowane
    min_samples_required: int = 1                # minimalna liczba obserwacji po dropna()
    id_like_unique_ratio: float = 0.98           # ~98%+ unikalnych sugeruje ID/sygn. ciągłą
    treat_numeric_str_as_numeric: bool = True    # "123" → 123 przy regresji
    truncate_log_chars: int = 400                # cięcie długich struktur w logach
    small_unique_threshold: int = 15             # <=15 unikalnych (num.) → prefer classification
    allow_none_problem_fallback: bool = False    # jeśli True, zwraca None gdy niepewne (zamiast fallbacku)
    # wskazówki dla binarnych (nie wpływa na kontrakt — tylko wewn. heurystyki)
    binary_positive_hint_keywords: Tuple[str, ...] = ("yes", "true", "1", "success", "positive")


# === NAZWA_SEKCJI === HELPERY ===
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
        try:
            ss = pd.to_numeric(s.dropna().astype(str).str.replace(",", ".", regex=False), errors="coerce")
            ratio = float(ss.notna().mean()) if len(ss) else 0.0
            return ratio > 0.9
        except Exception:
            return False
    return False


def _is_id_like(s: pd.Series, ratio_threshold: float) -> bool:
    """Czy kolumna wygląda na ID (prawie same unikalne wartości)."""
    try:
        n_unique = int(s.nunique(dropna=True))
        n = int(len(s))
        return n > 0 and (n_unique / n) >= ratio_threshold
    except Exception:
        return False


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Bezpieczna próba rzutowania do float (z kropką zamiast przecinka)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    try:
        ss = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
        return ss
    except Exception:
        return s


def _majority_label(vc: pd.Series) -> Tuple[Optional[str], float]:
    """Zwraca (etykieta_max, procent_max) dla value_counts bez NaN."""
    if vc.empty:
        return None, 0.0
    total = int(vc.sum())
    lab = str(vc.index[0])
    pct = float(vc.iloc[0] / max(1, total) * 100.0)
    return lab, pct


# === NAZWA_SEKCJI === KLASA GŁÓWNA ===
class ProblemClassifier(BaseAgent):
    """
    Klasyfikuje typ problemu ML na podstawie targetu (classification vs regression),
    wykonuje analizę kolumny celu i zwraca rekomendacje metryk, preprocessing'u oraz modeli.
    Kontrakt 1:1 zgodny z orkiestratorem.
    """

    def __init__(self, config: Optional[ProblemClassifierConfig] = None) -> None:
        super().__init__(
            name="ProblemClassifier",
            description="Classifies ML problem as classification or regression"
        )
        self.config = config or ProblemClassifierConfig()
        self._log = logger.bind(agent="ProblemClassifier")

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
            AgentResult with problem classification payload (kontrakt powyżej).
        """
        result = AgentResult(agent_name=self.name)
        t0_total = time.perf_counter()

        try:
            # Walidacja obecności kolumny
            if target_column not in data.columns:
                msg = f"Target column '{target_column}' not found"
                result.add_error(msg)
                self._log.error(msg)
                return result

            # NA/Inf safety (nie modyfikujemy wejścia)
            df = data.replace([np.inf, -np.inf], np.nan)
            target = df[target_column]

            # Guard: pusty target lub brak niepustych wartości
            if target is None or len(target) == 0 or target.dropna().empty:
                payload = self._empty_payload_empty_target(target)
                result.data = payload
                self._log.warning("empty target – cannot infer problem type.")
                return result

            # 1) Detekcja typu problemu z pomocą utilsa
            detected = None
            try:
                detected = infer_problem_type(target)
            except Exception as e:
                self._log.debug(f"infer_problem_type failed: {e}")

            if isinstance(detected, ProblemType):
                problem_type_str: Optional[str] = "classification" if detected == ProblemType.CLASSIFICATION else "regression"
            else:
                problem_type_str = str(detected).lower().strip() if detected is not None else None

            # 2) Heurystyczny fallback / korekta decyzji
            if problem_type_str not in ("classification", "regression"):
                problem_type_str = self._fallback_problem_type(target)
            else:
                # korekta: numeric dtype, ale bardzo mało unikalnych → classification (np. {0,1,2})
                if pd.api.types.is_numeric_dtype(target):
                    nunique = int(target.nunique(dropna=True))
                    if nunique <= self.config.small_unique_threshold:
                        problem_type_str = "classification"
                # korekta: object, ale numeric-like/ID-like → regression (chyba że „mało klas”)
                elif _is_numeric_series_like(target) or _is_id_like(target, self.config.id_like_unique_ratio):
                    nunique = int(target.nunique(dropna=True))
                    problem_type_str = "classification" if nunique <= self.config.small_unique_threshold else "regression"

            if problem_type_str is None and not self.config.allow_none_problem_fallback:
                # Bezpieczne domyślne: classification dla tekstowych, regression dla numeric-like
                problem_type_str = "regression" if _is_numeric_series_like(target) else "classification"

            # 3) Analiza targetu
            analysis = self._analyze_target(target, problem_type_str)

            # 4) Rekomendacje
            recommendations = self._get_recommendations(problem_type_str, analysis)

            # 5) Kontrakt (ściśle 1:1)
            result.data = {
                "problem_type": problem_type_str,
                "target_analysis": analysis,
                "recommendations": recommendations,
            }

            self._log.success(f"classified as: {problem_type_str} in {(time.perf_counter()-t0_total)*1000:.1f} ms")

        except Exception as e:
            result.add_error(f"Problem classification failed: {e}")
            self._log.exception(f"Problem classification error: {e}")

        return result

    # === FALLBACK HEURYSTYCZNY ===
    def _fallback_problem_type(self, target: pd.Series) -> Optional[str]:
        """
        Ostrożny fallback:
          - numeric-like oraz/lub id-like → 'regression' (chyba że bardzo mało unikalnych)
          - w przeciwnym razie 'classification'
          - jeżeli allow_none_problem_fallback=True i niepewność wysoka → None
        """
        is_numeric_like = _is_numeric_series_like(target)
        is_idlike = _is_id_like(target, self.config.id_like_unique_ratio)
        nunique = int(target.nunique(dropna=True))

        if is_numeric_like or is_idlike:
            if nunique <= self.config.small_unique_threshold:
                return "classification"
            return "regression"

        if not self.config.allow_none_problem_fallback:
            return "classification"
        return None

    # === PAYLOAD DLA PUSTEGO TARGETU ===
    def _empty_payload_empty_target(self, target: Optional[pd.Series]) -> Dict[str, Any]:
        n = int(len(target)) if target is not None else 0
        n_missing = int(target.isna().sum()) if target is not None else 0
        return {
            "problem_type": None,
            "target_analysis": {
                "n_samples": n,
                "n_unique": int(target.nunique(dropna=True)) if target is not None else 0,
                "n_missing": n_missing,
                "missing_pct": float((n_missing / max(1, n)) * 100.0) if n > 0 else 0.0,
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

    # === ANALIZA TARGETU (DYNA.) ===
    def _analyze_target(self, target: pd.Series, problem_type: str) -> Dict[str, Any]:
        """Analyze target column in detail depending on problem type."""
        n = int(len(target))
        n_unique = int(target.nunique(dropna=True))
        n_missing = int(target.isna().sum())
        missing_pct = float((n_missing / max(1, n)) * 100.0)

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

    # === ANALIZA: KLASYFIKACJA ===
    def _analyze_classification_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze classification target: liczba klas, rozkład, nierównowaga, rzadkie klasy, majority class."""
        cfg = self.config

        vc_non_na = target.dropna().value_counts()
        n_classes = int(vc_non_na.shape[0])

        warnings_local: List[str] = []
        if n_classes <= 1:
            imbalance_ratio = 1.0
            is_imbalanced = False
            warnings_local.append("Only one unique non-NA class found — model training may not be feasible.")
        else:
            min_class = int(vc_non_na.min())
            max_class = int(vc_non_na.max())
            imbalance_ratio = float(max_class / max(1, min_class))
            is_imbalanced = imbalance_ratio > cfg.imbalance_ratio_threshold
            if is_imbalanced:
                warnings_local.append(
                    f"Class imbalance detected (ratio={imbalance_ratio:.2f} > {cfg.imbalance_ratio_threshold})"
                )
            rare = [str(k) for k, v in vc_non_na.items() if v < cfg.warn_min_samples_per_class]
            if rare:
                warnings_local.append(
                    f"Rare classes detected (<{cfg.warn_min_samples_per_class} samples): {rare}"
                )
            very_rare = [str(k) for k, v in vc_non_na.items() if v < cfg.rare_class_threshold]
            if very_rare:
                warnings_local.append(
                    f"Very rare classes (<{cfg.rare_class_threshold} samples): {very_rare}"
                )

        maj_label, maj_pct = _majority_label(vc_non_na)
        classification_type = "binary" if n_classes == 2 else "multiclass" if n_classes > 2 else "single_class"

        return {
            "classification_type": classification_type,
            "n_classes": n_classes,
            "class_distribution": {str(k): int(v) for k, v in vc_non_na.to_dict().items()},
            "is_imbalanced": bool(is_imbalanced),
            "imbalance_ratio": float(imbalance_ratio),
            "majority_class": maj_label,
            "majority_class_pct": float(maj_pct),
            "warnings": warnings_local,
        }

    # === ANALIZA: REGRESJA ===
    def _analyze_regression_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze regression target: statystyki opisowe z guardami i konwersją numeric-like."""
        cfg = self.config
        target_clean = target.dropna()

        if len(target_clean) < cfg.min_samples_required:
            return {
                "error": "Insufficient non-missing target values for regression analysis",
                "n_non_missing": int(len(target_clean))
            }

        t = target_clean.copy()
        if cfg.treat_numeric_str_as_numeric and not pd.api.types.is_numeric_dtype(t):
            t = _coerce_numeric_series(t)

        # jeśli nadal nie numeric → spróbuj jeszcze raz
        if not pd.api.types.is_numeric_dtype(t):
            try:
                t = pd.to_numeric(t, errors="coerce").dropna()
            except Exception:
                pass

        if isinstance(t, pd.Series) and t.dropna().empty:
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

    # === REKOMENDACJE (DYNA.) ===
    def _get_recommendations(self, problem_type: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build recommendations based on problem type and target analysis."""
        if problem_type == "classification":
            return self._get_classification_recommendations(analysis)
        else:
            return self._get_regression_recommendations(analysis)

    # === REKOMENDACJE: KLASYFIKACJA ===
    def _get_classification_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
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

        if is_imbalanced:
            for m in ("balanced_accuracy", "mcc", "auprc"):
                if m not in metrics:
                    metrics.append(m)

        # Preprocessing
        preprocessing = [
            "Handle missing values in target",
            "Encode categorical target if needed",
            "Use stratified train/validation split",
        ]
        if is_imbalanced:
            preprocessing.extend(["Apply SMOTE/SMOTEENN or class weighting", "Tune threshold using PR/ROC curves"])

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
                "Linear/Kernel SVM (po skalowaniu cech)"
            ]
        else:
            models = [
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "CatBoost",
                "Linear/Kernel SVM (One-vs-Rest, po skalowaniu)"
            ]

        return {
            "problem_type": "classification",
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
            "warnings": warnings_local,
        }

    # === REKOMENDACJE: REGRESJA ===
    def _get_regression_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
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
            "problem_type": "regression",
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
            "warnings": warnings_local,
        }
