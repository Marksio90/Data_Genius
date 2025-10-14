# === OPIS MODUŁU ===
"""
DataGenius PRO - Model Evaluator (PRO+++)
Uniwersalny ewaluator modeli ML (PyCaret / sklearn). Bogate metryki klasyfikacji i regresji,
odporny na różne formaty predykcji, z konfiguracją, defensywną walidacją i czytelnym kontraktem.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, mean_squared_log_error
)

from core.base_agent import BaseAgent, AgentResult


# === KONFIG ===
@dataclass(frozen=True)
class EvalConfig:
    """Konfiguracja ewaluatora."""
    return_predictions: bool = True              # czy zwracać pełny DataFrame predykcji
    prefer_pycaret_predict: bool = True          # jeśli dostępny wrapper, użyj predict_model
    classification_roc_strategy: Literal["auto", "binary", "ovr", "ovo"] = "auto"
    primary_metric_default_cls: str = "accuracy" # fallback gdy nie podano primary_metric
    primary_metric_default_reg: str = "r2"       # fallback gdy nie podano primary_metric
    # Dla bezpieczeństwa (duże zbiory) można dodać tu np. capy na sampling predykcji, ale domyślnie bez.


class ModelEvaluator(BaseAgent):
    """
    Evaluates ML models with comprehensive metrics (classification & regression).
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        super().__init__(
            name="ModelEvaluator",
            description="Evaluates trained models"
        )
        self.config = config or EvalConfig()

    # === WYKONANIE GŁÓWNE ===
    def execute(
        self,
        best_model: Any,
        pycaret_wrapper: Optional[Any],
        problem_type: Literal["classification", "regression"],
        data: Optional[pd.DataFrame] = None,
        *,
        y_true: Optional[pd.Series] = None,
        primary_metric: Optional[str] = None,
        positive_label: Optional[Any] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Evaluate model.

        Args:
            best_model: wytrenowany model
            pycaret_wrapper: instancja wrappera PyCaret lub None
            problem_type: "classification" | "regression"
            data: opcjonalny DataFrame do predykcji
            y_true: opcjonalnie prawda (jeśli nie używamy PyCaretowego predict_model)
            primary_metric: preferowana metryka do pola best_score
            positive_label: etykieta pozytywna (dla binary ROC/PR)
        """
        result = AgentResult(agent_name=self.name)

        try:
            # --- Walidacja bazowa ---
            if best_model is None:
                raise ValueError("'best_model' is required")
            if problem_type not in {"classification", "regression"}:
                raise ValueError(f"Unsupported problem_type={problem_type}")

            # --- Predykcje: PyCaret -> sklearn fallback ---
            preds_df, used_cols = self._get_predictions_dataframe(
                model=best_model,
                data=data,
                pycaret_wrapper=pycaret_wrapper
            )

            n_samples = len(preds_df) if preds_df is not None else 0

            # --- Ekstrakcja y_true / y_pred / y_score ---
            y_true_vec, y_pred_vec, y_proba, classes = self._extract_targets_and_scores(
                preds_df=preds_df,
                model=best_model,
                data=data,
                y_true=y_true,
                problem_type=problem_type
            )

            # --- Metryki ---
            if problem_type == "classification":
                metrics = self._evaluate_classification(
                    y_true=y_true_vec,
                    y_pred=y_pred_vec,
                    y_proba=y_proba,
                    classes=classes,
                    positive_label=positive_label
                )
                primary_metric_name = primary_metric or self.config.primary_metric_default_cls
            else:
                metrics = self._evaluate_regression(
                    y_true=y_true_vec,
                    y_pred=y_pred_vec
                )
                primary_metric_name = primary_metric or self.config.primary_metric_default_reg

            # --- Best score (z sensownym fallbackiem) ---
            best_score = self._resolve_best_score(metrics, primary_metric_name)

            # --- Złóż wynik ---
            result.data = {
                "metrics": metrics,
                "predictions": preds_df if self.config.return_predictions else None,
                "best_model_name": type(best_model).__name__,
                "best_score": best_score,
                "problem_type": problem_type,
                "meta": {
                    "n_samples": n_samples,
                    "prediction_columns": used_cols,
                    "used_primary_metric": primary_metric_name
                }
            }

            self.logger.success(
                f"Model evaluation complete: {problem_type} {primary_metric_name} = "
                f"{best_score if best_score is not None else 'n/a'}"
            )

        except Exception as e:
            result.add_error(f"Model evaluation failed: {e}")
            self.logger.error(f"Model evaluation error: {e}", exc_info=True)

        return result

    # === PREDYKCJE: PYCaret -> sklearn fallback ===
    def _get_predictions_dataframe(
        self,
        model: Any,
        data: Optional[pd.DataFrame],
        pycaret_wrapper: Optional[Any]
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Zwraca DataFrame z kolumnami predykcji, jeśli to możliwe.
        Preferuje PyCaret.predict_model, z fallbackiem do model.predict / predict_proba.
        """
        used_cols: List[str] = []
        preds_df: Optional[pd.DataFrame] = None

        # PyCaret path
        if pycaret_wrapper is not None and self.config.prefer_pycaret_predict:
            try:
                preds_df = (pycaret_wrapper.predict_model(model, data=data) if data is not None
                            else pycaret_wrapper.predict_model(model))
                if isinstance(preds_df, pd.DataFrame):
                    used_cols = preds_df.columns.tolist()
                    return preds_df, used_cols
                self.logger.warning("pycaret_wrapper.predict_model returned non-DataFrame; falling back.")
            except Exception as e:
                self.logger.warning(f"pycaret_wrapper.predict_model failed: {e}; falling back to raw model.")

        # sklearn-like fallback
        try:
            if data is None:
                raise ValueError("No 'data' provided for raw model prediction.")

            y_pred = None
            y_proba = None

            if hasattr(model, "predict"):
                y_pred = model.predict(data)

            if hasattr(model, "predict_proba"):
                # predict_proba może zgłosić wyjątek dla regresji – zabezpieczamy
                try:
                    y_proba = model.predict_proba(data)
                except Exception:
                    y_proba = None

            # zbuduj df podobny do PyCaret
            records: Dict[str, Any] = {}
            if y_pred is not None:
                records["Label"] = y_pred
            if y_proba is not None and isinstance(y_proba, np.ndarray):
                # przyjmijmy, że dla binarnej to proba klasy 1; dla multiclass kolumny per klasa
                if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                    records["Score"] = y_proba.ravel()
                else:
                    # stworzymy kolumny proba_<class>
                    # jeśli model ma atrybut classes_ to oznacz kolumny nazwami klas
                    classes = getattr(model, "classes_", np.arange(y_proba.shape[1]))
                    for j, cls in enumerate(classes):
                        records[f"proba_{cls}"] = y_proba[:, j]

            preds_df = pd.DataFrame(records, index=data.index)
            used_cols = preds_df.columns.tolist()
            return preds_df, used_cols

        except Exception as e:
            self.logger.warning(f"Raw model prediction failed or no data: {e}")

        return preds_df, used_cols

    # === EKSTRAKCJA y_true / y_pred / y_proba ===
    def _extract_targets_and_scores(
        self,
        preds_df: Optional[pd.DataFrame],
        model: Any,
        data: Optional[pd.DataFrame],
        y_true: Optional[pd.Series],
        problem_type: str
    ) -> Tuple[pd.Series, pd.Series, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Zwraca: (y_true, y_pred, y_proba (np.ndarray|None), classes (np.ndarray|None))
        Obsługuje zarówno PyCaret-owy format jak i "goły" sklearn.
        """
        # y_true
        if y_true is not None:
            y_true_vec = pd.Series(y_true).reset_index(drop=True)
        elif (preds_df is not None) and (preds_df.shape[1] >= 1):
            # PyCaret zwykle wstawia oryginalny target jako pierwszą kolumnę
            y_true_vec = preds_df.iloc[:, 0].reset_index(drop=True)
        else:
            # brak y_true — metryki będą ograniczone
            y_true_vec = pd.Series(dtype=float)

        # y_pred
        y_pred_vec = pd.Series(dtype=float)
        if preds_df is not None:
            # Próbuj standardowych nazw z PyCaret i fallback
            for col in ["Label", "prediction_label", "pred", "y_pred"]:
                if col in preds_df.columns:
                    y_pred_vec = preds_df[col].reset_index(drop=True)
                    break
            if y_pred_vec.empty and preds_df.shape[1] >= 2:
                # heurystyka: ostatnia kolumna bywa predykcją
                y_pred_vec = preds_df.iloc[:, -1].reset_index(drop=True)

        if y_pred_vec.empty and hasattr(model, "predict") and data is not None:
            try:
                y_pred_vec = pd.Series(model.predict(data)).reset_index(drop=True)
            except Exception:
                pass

        # y_proba (dla klasyfikacji)
        y_proba = None
        classes = getattr(model, "classes_", None) if problem_type == "classification" else None
        if problem_type == "classification":
            # PyCaret często wystawia 'Score' (proba pozytywnej klasy) lub proby per klasa
            if preds_df is not None:
                if "Score" in preds_df.columns:
                    y_proba = preds_df["Score"].to_numpy().reshape(-1, 1)  # jako (n,1) dla binarnej
                else:
                    # zbierz kolumny proba_<class>
                    prob_cols = [c for c in preds_df.columns if c.startswith("proba_")]
                    if prob_cols:
                        y_proba = preds_df[prob_cols].to_numpy()
                        # wyciągnij klasy z sufiksów, jeśli brak model.classes_
                        if classes is None:
                            try:
                                classes = np.array([c.replace("proba_", "") for c in prob_cols])
                            except Exception:
                                pass

            if y_proba is None and hasattr(model, "predict_proba") and data is not None:
                try:
                    y_proba = model.predict_proba(data)
                except Exception:
                    y_proba = None

        # Uporządkuj indexy/kształty
        y_true_vec = y_true_vec.reset_index(drop=True)
        y_pred_vec = y_pred_vec.reset_index(drop=True)
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 1:
                y_proba = y_proba.reshape(-1, 1)

        return y_true_vec, y_pred_vec, y_proba, classes

    # === METRYKI: KLASYFIKACJA ===
    def _evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_proba: Optional[np.ndarray],
        classes: Optional[np.ndarray],
        positive_label: Optional[Any]
    ) -> Dict[str, Any]:
        """Bogaty zestaw metryk klasyfikacji z obroną na braki danych."""
        metrics: Dict[str, Any] = {}
        try:
            if y_true.empty or y_pred.empty:
                self.logger.warning("Classification: missing y_true or y_pred; metrics limited.")
                return metrics

            # Podstawy
            try: metrics["accuracy"] = accuracy_score(y_true, y_pred)
            except Exception: pass

            # Precision/Recall/F1
            for avg in ("weighted", "macro", "micro"):
                try: metrics[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
                except Exception: pass
                try: metrics[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
                except Exception: pass
                try: metrics[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
                except Exception: pass

            # Confusion matrix (bezpiecznie)
            try:
                cm = confusion_matrix(y_true, y_pred)
                metrics["confusion_matrix"] = {
                    "shape": list(cm.shape),
                    "diag_sum": int(np.trace(cm)),
                }
            except Exception:
                pass

            # Raport klas
            try:
                cr = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
                metrics["classification_report"] = cr
            except Exception:
                pass

            # ROC AUC / PR AUC
            if y_proba is not None:
                # detekcja binarnej vs multiclass
                unique_classes = np.unique(y_true.dropna())
                is_binary = len(unique_classes) == 2

                # wybór strategii
                strat = self.config.classification_roc_strategy
                if strat == "auto":
                    strat = "binary" if is_binary else "ovr"

                # przygotuj proby/etykiety
                if strat == "binary":
                    # weź proby klasy pozytywnej
                    # jeśli `positive_label` nie podany, bierz max(y_true) albo 1
                    pos = positive_label
                    if pos is None:
                        try:
                            pos = 1 if 1 in unique_classes else max(unique_classes)
                        except Exception:
                            pos = None
                    # mapowanie do {0,1}
                    y_bin = (y_true == pos).astype(int) if pos is not None else pd.Series(y_true).astype('category').cat.codes
                    proba_pos = None
                    if y_proba.shape[1] == 1:
                        proba_pos = y_proba[:, 0]
                    else:
                        # jeżeli znamy klasy, wybierz kolumnę odpowiadającą pos
                        if classes is not None and pos in classes:
                            j = list(classes).index(pos)
                            proba_pos = y_proba[:, j]
                        else:
                            # weźmy kolumnę o najwyższej średniej (heurystyka)
                            j = int(np.argmax(np.nanmean(y_proba, axis=0)))
                            proba_pos = y_proba[:, j]
                    try:
                        metrics["roc_auc"] = roc_auc_score(y_bin, proba_pos)
                    except Exception:
                        pass
                    try:
                        metrics["average_precision"] = average_precision_score(y_bin, proba_pos)
                    except Exception:
                        pass

                else:
                    # one-vs-rest / one-vs-one
                    multi = "ovr" if strat == "ovr" else "ovo"
                    try:
                        metrics[f"roc_auc_{multi}"] = roc_auc_score(y_true, y_proba, multi_class=multi)
                    except Exception:
                        pass

        except Exception as e:
            self.logger.warning(f"Classification metrics failed: {e}")

        return metrics

    # === METRYKI: REGRESJA ===
    def _evaluate_regression(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, Any]:
        """Bogaty zestaw metryk regresji."""
        metrics: Dict[str, Any] = {}
        try:
            if y_true.empty or y_pred.empty:
                self.logger.warning("Regression: missing y_true or y_pred; metrics limited.")
                return metrics

            try: metrics["mae"] = mean_absolute_error(y_true, y_pred)
            except Exception: pass

            try:
                mse = mean_squared_error(y_true, y_pred)
                metrics["mse"] = mse
                metrics["rmse"] = float(np.sqrt(mse))
            except Exception:
                pass

            try: metrics["r2"] = r2_score(y_true, y_pred)
            except Exception: pass

            try: metrics["median_ae"] = median_absolute_error(y_true, y_pred)
            except Exception: pass

            # MAPE (z maską na zera)
            try:
                mask = (y_true != 0) & (~pd.isna(y_true))
                metrics["mape"] = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            except Exception:
                pass

            # MSLE (tylko gdy wszystkie wartości >= 0)
            try:
                if (y_true.min() >= 0) and (y_pred.min() >= 0):
                    metrics["msle"] = mean_squared_log_error(y_true, y_pred)
            except Exception:
                pass

        except Exception as e:
            self.logger.warning(f"Regression metrics failed: {e}")

        return metrics

    # === BEST SCORE WYBÓR ===
    def _resolve_best_score(self, metrics: Dict[str, Any], primary_metric: str) -> Optional[float]:
        """Zwraca `best_score` zgodny z `primary_metric`, z rozsądnymi aliasami."""
        if not metrics:
            return None

        aliases = {
            # klasyfikacja
            "acc": "accuracy",
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "f1_weighted": "f1_weighted",
            "f1_macro": "f1_macro",
            "f1_micro": "f1_micro",
            "roc_auc": "roc_auc",
            "roc_auc_ovr": "roc_auc_ovr",
            "roc_auc_ovo": "roc_auc_ovo",
            "ap": "average_precision",
            "average_precision": "average_precision",
            # regresja
            "r2": "r2",
            "mae": "mae",
            "mse": "mse",
            "rmse": "rmse",
            "msle": "msle",
            "median_ae": "median_ae",
            "mape": "mape",
        }
        key = aliases.get(primary_metric.lower(), primary_metric) if isinstance(primary_metric, str) else None
        if key and key in metrics:
            val = metrics[key]
            return float(val) if val is not None else None

        # fallback kaskadowy
        for cand in ["accuracy", "f1_weighted", "roc_auc", "roc_auc_ovr", "r2", "rmse", "mae"]:
            if cand in metrics and metrics[cand] is not None:
                v = metrics[cand]
                return float(v) if isinstance(v, (int, float, np.floating)) else None

        return None
