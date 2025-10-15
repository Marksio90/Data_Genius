# === OPIS MODUŁU ===
"""
DataGenius PRO++++ - Model Evaluator (KOSMOS)
Uniwersalny ewaluator modeli ML (PyCaret / sklearn).
- Klasyfikacja: accuracy/precision/recall/f1 (macro/micro/weighted), ROC-AUC (binary/ovr/ovo), PR-AUC,
  confusion matrix (raw/normalized), per-class metrics, threshold tuning (Youden/F1/Custom),
  KS statistic, lift@k, top-k gains.
- Regresja: MAE/MSE/RMSE/R2/MedianAE/MAPE/MSLE + diagnostyka reszt (bias, IQR outliers, quantiles).
- Defensywa: tolerancja różnych formatów predykcji, brak y_true, multiclass/binary rozróżnienie,
  brak predict_proba, brak bibliotek, duże zbiory.
- Ujednolicony kontrakt wyników + metadane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Literal, Union

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, mean_squared_log_error
)
from sklearn.utils.validation import check_is_fitted

from core.base_agent import BaseAgent, AgentResult


# === KONFIG ===
@dataclass(frozen=True)
class EvalConfig:
    """Konfiguracja ewaluatora."""
    return_predictions: bool = True              # czy zwracać DataFrame predykcji
    prefer_pycaret_predict: bool = True          # jeśli dostępny wrapper, użyj predict_model
    classification_roc_strategy: Literal["auto", "binary", "ovr", "ovo"] = "auto"
    primary_metric_default_cls: str = "accuracy" # fallback gdy nie podano primary_metric
    primary_metric_default_reg: str = "r2"       # fallback gdy nie podano primary_metric
    include_per_class: bool = True               # raport per klasa
    include_curves_sample_cap: int = 200_000     # cap na próbkę do krzywych (ROC/PR)
    optimize_thresholds: bool = True             # tunning thresholdu dla binary
    threshold_grid_size: int = 101               # ile punktów siatki dla tuningu
    lift_k_list: Tuple[int, ...] = (1, 3, 5, 10) # lift@k (%) – np. 1,3,5,10
    compute_ks: bool = True                      # Kolmogorov–Smirnov dla binary
    sample_predictions_cap: Optional[int] = None # opcjonalny cap dla zwracanych predykcji


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
        y_true: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
        primary_metric: Optional[str] = None,
        positive_label: Optional[Any] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Evaluate model.

        Args:
            best_model: wytrenowany model
            pycaret_wrapper: instancja wrappera PyCaret lub None
            problem_type: "classification" | "regression"
            data: DataFrame do predykcji (wymagany bez PyCaret)
            y_true: prawda (jeśli nie używamy PyCaretowego predict_model)
            primary_metric: metryka priorytetowa do best_score
            positive_label: etykieta pozytywna (dla binary ROC/PR)
            sample_weight: wagi próbek (opcjonalnie)
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

            n_samples = int(len(preds_df)) if preds_df is not None else 0
            if y_true is None and preds_df is not None and preds_df.shape[1] > 0:
                # jeśli PyCaret zwrócił y_true jako pierwszą kolumnę – nie nadpisujemy
                pass

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
                    positive_label=positive_label,
                    sample_weight=sample_weight
                )
                primary_metric_name = primary_metric or self.config.primary_metric_default_cls
            else:
                metrics = self._evaluate_regression(
                    y_true=y_true_vec,
                    y_pred=y_pred_vec,
                    sample_weight=sample_weight
                )
                primary_metric_name = primary_metric or self.config.primary_metric_default_reg

            # --- Best score (z sensownym fallbackiem) ---
            best_score = self._resolve_best_score(metrics, primary_metric_name)

            # --- Złóż wynik ---
            out_preds = self._maybe_slice_predictions(preds_df)
            result.data = {
                "metrics": metrics,
                "predictions": out_preds if self.config.return_predictions else None,
                "best_model_name": type(best_model).__name__,
                "best_score": best_score,
                "problem_type": problem_type,
                "meta": {
                    "n_samples": n_samples,
                    "prediction_columns": used_cols,
                    "used_primary_metric": primary_metric_name,
                    "classes": (classes.tolist() if isinstance(classes, np.ndarray) else classes),
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

    # === PREDYKCJE: PyCaret -> sklearn fallback ===
    def _get_predictions_dataframe(
        self,
        model: Any,
        data: Optional[pd.DataFrame],
        pycaret_wrapper: Optional[Any]
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Zwraca DataFrame z kolumnami predykcji (zachowując spójny schemat):
        - y_true (jeśli dostępny)
        - Label (predykcja klasa/wartość)
        - Score / proba_<class> (jeśli dostępne)
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

            # próbujemy predict + predict_proba
            y_pred = None
            y_proba = None

            if hasattr(model, "predict"):
                y_pred = model.predict(data)

            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(data)
                except Exception:
                    y_proba = None

            records: Dict[str, Any] = {}
            # y_true nie jest znane na tym etapie (brak kontraktu) – zostawiamy do ekstrakcji
            if y_pred is not None:
                records["Label"] = y_pred

            if y_proba is not None and isinstance(y_proba, np.ndarray):
                if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                    records["Score"] = y_proba.ravel()
                else:
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
        y_true: Optional[Union[pd.Series, np.ndarray, List[Any]]],
        problem_type: str
    ) -> Tuple[pd.Series, pd.Series, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Zwraca: (y_true, y_pred, y_proba (np.ndarray|None), classes (np.ndarray|None))
        Obsługa: PyCaret-owy format i "goły" sklearn.
        """
        # y_true
        if y_true is not None:
            y_true_vec = pd.Series(y_true).reset_index(drop=True)
        elif (preds_df is not None) and (preds_df.shape[1] >= 1):
            # heurystyka: jeśli w PyCaret pierwsza kolumna to target – spróbuj wykryć po nazwie
            first_col = preds_df.columns[0]
            if first_col.lower() in {"target", "y", "y_true"}:
                y_true_vec = preds_df.iloc[:, 0].reset_index(drop=True)
            else:
                # wolimy jawne y_true – jeśli brak, tworzymy pustą serię (ograniczone metryki)
                y_true_vec = pd.Series(dtype=float)
        else:
            y_true_vec = pd.Series(dtype=float)

        # y_pred
        y_pred_vec = pd.Series(dtype=float)
        if preds_df is not None:
            for col in ["Label", "prediction_label", "pred", "y_pred"]:
                if col in preds_df.columns:
                    y_pred_vec = preds_df[col].reset_index(drop=True)
                    break
            if y_pred_vec.empty and preds_df.shape[1] >= 2:
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
            if preds_df is not None:
                if "Score" in preds_df.columns:
                    y_proba = preds_df["Score"].to_numpy().reshape(-1, 1)
                else:
                    prob_cols = [c for c in preds_df.columns if c.startswith("proba_")]
                    if prob_cols:
                        y_proba = preds_df[prob_cols].to_numpy()
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
        positive_label: Optional[Any],
        sample_weight: Optional[Union[pd.Series, np.ndarray, List[float]]] = None
    ) -> Dict[str, Any]:
        """Bogaty zestaw metryk klasyfikacji z obroną na braki danych."""
        metrics: Dict[str, Any] = {}
        try:
            if y_true.empty or y_pred.empty:
                self.logger.warning("Classification: missing y_true or y_pred; metrics limited.")
                return metrics

            sw = np.array(sample_weight) if sample_weight is not None else None

            # Podstawy
            try: metrics["accuracy"] = accuracy_score(y_true, y_pred, sample_weight=sw)
            except Exception: pass

            # Precision/Recall/F1
            for avg in ("weighted", "macro", "micro"):
                try: metrics[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0, sample_weight=sw)
                except Exception: pass
                try: metrics[f"recall_{avg}"]   = recall_score(y_true, y_pred, average=avg, zero_division=0, sample_weight=sw)
                except Exception: pass
                try: metrics[f"f1_{avg}"]       = f1_score(y_true, y_pred, average=avg, zero_division=0, sample_weight=sw)
                except Exception: pass

            # Per-class report
            if self.config.include_per_class:
                try:
                    cr = classification_report(y_true, y_pred, zero_division=0, output_dict=True, sample_weight=sw)
                    metrics["classification_report"] = cr
                except Exception:
                    pass

            # Confusion matrix: raw + normalized
            try:
                cm = confusion_matrix(y_true, y_pred, sample_weight=sw)
                metrics["confusion_matrix"] = {"matrix": cm.tolist()}
                with np.errstate(divide='ignore', invalid='ignore'):
                    cm_norm = cm / cm.sum(axis=1, keepdims=True)
                cm_norm = np.nan_to_num(cm_norm, nan=0.0).tolist()
                metrics["confusion_matrix_normalized"] = {"matrix": cm_norm}
            except Exception:
                pass

            # ROC/PR AUC + krzywe (sampled)
            if y_proba is not None:
                uniq = np.unique(pd.Series(y_true).dropna())
                is_binary = len(uniq) == 2

                # Sampling do krzywych (duże zbiory)
                idx = np.arange(len(y_true))
                cap = self.config.include_curves_sample_cap
                if cap and len(idx) > cap:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(idx, size=cap, replace=False)
                yt = np.array(y_true)[idx]

                # Strategia ROC
                strat = self.config.classification_roc_strategy
                if strat == "auto":
                    strat = "binary" if is_binary else "ovr"

                # PR-AUC / ROC-AUC
                if is_binary:
                    pos = positive_label
                    if pos is None:
                        try:
                            pos = 1 if 1 in uniq else max(uniq)
                        except Exception:
                            pos = uniq[-1]
                    y_bin = (y_true == pos).astype(int)
                    proba_pos = self._extract_positive_proba(y_proba, classes, pos)

                    # AUC
                    try: metrics["roc_auc"] = roc_auc_score(y_bin, proba_pos, sample_weight=sw)
                    except Exception: pass
                    try: metrics["average_precision"] = average_precision_score(y_bin, proba_pos, sample_weight=sw)
                    except Exception: pass

                    # Krzywe (próbka)
                    try:
                        fpr, tpr, roc_thr = roc_curve(yt, proba_pos[idx])
                        pr_p, pr_r, pr_thr = precision_recall_curve(yt, proba_pos[idx])
                        metrics["curves"] = {
                            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thr.tolist()},
                            "pr":  {"precision": pr_p.tolist(), "recall": pr_r.tolist(), "thresholds": pr_thr.tolist()}
                        }
                    except Exception:
                        pass

                    # KS statistic
                    if self.config.compute_ks:
                        try:
                            ks = self._compute_ks_stat(y_bin, proba_pos)
                            metrics["ks_stat"] = float(ks)
                        except Exception:
                            pass

                    # Lift@k
                    try:
                        lifts = self._compute_lift_at_k(y_bin, proba_pos, self.config.lift_k_list)
                        metrics["lift_at_k"] = {f"{k}%": float(v) for k, v in lifts.items()}
                    except Exception:
                        pass

                    # Threshold optimization (Youden/F1)
                    if self.config.optimize_thresholds:
                        try:
                            opt = self._optimize_threshold(y_bin, proba_pos, grid_size=self.config.threshold_grid_size)
                            metrics["threshold_optimization"] = opt
                        except Exception:
                            pass

                else:
                    # multiclass AUC
                    try:
                        metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", sample_weight=sw)
                    except Exception: pass
                    try:
                        metrics["roc_auc_ovo"] = roc_auc_score(y_true, y_proba, multi_class="ovo", sample_weight=sw)
                    except Exception: pass

        except Exception as e:
            self.logger.warning(f"Classification metrics failed: {e}")

        return metrics

    # === METRYKI: REGRESJA ===
    def _evaluate_regression(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sample_weight: Optional[Union[pd.Series, np.ndarray, List[float]]] = None
    ) -> Dict[str, Any]:
        """Bogaty zestaw metryk regresji + diagnostyka reszt."""
        metrics: Dict[str, Any] = {}
        try:
            if y_true.empty or y_pred.empty:
                self.logger.warning("Regression: missing y_true or y_pred; metrics limited.")
                return metrics

            sw = np.array(sample_weight) if sample_weight is not None else None

            try: metrics["mae"] = mean_absolute_error(y_true, y_pred, sample_weight=sw)
            except Exception: pass

            try:
                mse = mean_squared_error(y_true, y_pred, sample_weight=sw)
                metrics["mse"] = mse
                metrics["rmse"] = float(np.sqrt(mse))
            except Exception:
                pass

            try: metrics["r2"] = r2_score(y_true, y_pred, sample_weight=sw)
            except Exception: pass

            try: metrics["median_ae"] = median_absolute_error(y_true, y_pred)
            except Exception: pass

            # MAPE (mask na zera)
            try:
                yt = pd.Series(y_true).astype(float)
                yp = pd.Series(y_pred).astype(float)
                mask = (yt != 0) & (~pd.isna(yt))
                metrics["mape"] = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)
            except Exception:
                pass

            # MSLE (>=0)
            try:
                if (pd.Series(y_true).min() >= 0) and (pd.Series(y_pred).min() >= 0):
                    metrics["msle"] = mean_squared_log_error(y_true, y_pred, sample_weight=sw)
            except Exception:
                pass

            # Diagnostyka reszt
            try:
                resid = pd.Series(y_true).astype(float) - pd.Series(y_pred).astype(float)
                q1, q3 = np.quantile(resid, [0.25, 0.75])
                iqr = q3 - q1
                out_hi = float(q3 + 1.5 * iqr)
                out_lo = float(q1 - 1.5 * iqr)
                metrics["residuals"] = {
                    "mean": float(np.mean(resid)),
                    "median": float(np.median(resid)),
                    "std": float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0,
                    "q05": float(np.quantile(resid, 0.05)),
                    "q50": float(np.quantile(resid, 0.50)),
                    "q95": float(np.quantile(resid, 0.95)),
                    "iqr": float(iqr),
                    "outlier_bounds": {"low": out_lo, "high": out_hi},
                    "outlier_share_pct": float((np.sum((resid < out_lo) | (resid > out_hi)) / max(1, len(resid))) * 100),
                }
            except Exception:
                pass

        except Exception as e:
            self.logger.warning(f"Regression metrics failed: {e}")

        return metrics

    # === POMOC: score→pos proba (binary) ===
    def _extract_positive_proba(
        self, y_proba: np.ndarray, classes: Optional[np.ndarray], positive_label: Any
    ) -> np.ndarray:
        if y_proba.shape[1] == 1:
            return y_proba[:, 0]
        # spróbuj znaleźć kolumnę klasy positive_label
        if classes is not None:
            try:
                j = list(classes).index(positive_label)
                return y_proba[:, j]
            except Exception:
                pass
        # fallback: kolumna o najwyższej średniej (często „pozytywna”)
        j = int(np.argmax(np.nanmean(y_proba, axis=0)))
        return y_proba[:, j]

    # === KS STAT ===
    def _compute_ks_stat(self, y_true_bin: Union[pd.Series, np.ndarray], proba_pos: np.ndarray) -> float:
        y = np.array(y_true_bin).astype(int)
        p = np.array(proba_pos).astype(float)
        order = np.argsort(p)
        y_sorted = y[order]
        p_sorted = p[order]

        cum_pos = np.cumsum(y_sorted) / max(1, y_sorted.sum())
        cum_neg = np.cumsum(1 - y_sorted) / max(1, (1 - y_sorted).sum())
        ks = float(np.max(np.abs(cum_pos - cum_neg)))
        return ks

    # === LIFT @ k (procent populacji) ===
    def _compute_lift_at_k(self, y_true_bin: Union[pd.Series, np.ndarray], proba_pos: np.ndarray, k_list: Tuple[int, ...]) -> Dict[int, float]:
        y = np.array(y_true_bin).astype(int)
        p = np.array(proba_pos).astype(float)
        order = np.argsort(-p)
        lifts: Dict[int, float] = {}
        base_rate = float(np.mean(y)) if len(y) > 0 else 0.0
        for k in k_list:
            k = int(k)
            top_n = max(1, int(np.ceil(len(y) * (k / 100.0))))
            pos_in_top = np.sum(y[order][:top_n])
            rate_top = pos_in_top / top_n
            lifts[k] = (rate_top / base_rate) if base_rate > 0 else np.nan
        return lifts

    # === THRESHOLD TUNING (binary) ===
    def _optimize_threshold(self, y_true_bin: Union[pd.Series, np.ndarray], proba_pos: np.ndarray, grid_size: int = 101) -> Dict[str, Any]:
        y = np.array(y_true_bin).astype(int)
        p = np.array(proba_pos).astype(float)
        grid = np.linspace(0.0, 1.0, num=max(2, grid_size))

        best_f1, thr_f1 = -1.0, 0.5
        best_youden, thr_youden = -1.0, 0.5

        for t in grid:
            pred = (p >= t).astype(int)
            # F1
            try:
                f1 = f1_score(y, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, thr_f1 = f1, t
            except Exception:
                pass
            # Youden's J = TPR - FPR
            try:
                tp = ((pred == 1) & (y == 1)).sum()
                fn = ((pred == 0) & (y == 1)).sum()
                fp = ((pred == 1) & (y == 0)).sum()
                tn = ((pred == 0) & (y == 0)).sum()
                tpr = tp / max(1, (tp + fn))
                fpr = fp / max(1, (fp + tn))
                j = tpr - fpr
                if j > best_youden:
                    best_youden, thr_youden = j, t
            except Exception:
                pass

        return {
            "best_f1": {"threshold": float(thr_f1), "score": float(best_f1)},
            "best_youden": {"threshold": float(thr_youden), "score": float(best_youden)},
        }

    # === BEST SCORE WYBÓR ===
    def _resolve_best_score(self, metrics: Dict[str, Any], primary_metric: str) -> Optional[float]:
        """Zwraca `best_score` zgodny z `primary_metric`, z rozsądnymi aliasami i fallbackami."""
        if not metrics:
            return None

        aliases = {
            # klasyfikacja
            "acc": "accuracy",
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
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

        # wsparcie dla gniazd: curves/threshold_optimization
        def _get_nested(d: Dict[str, Any], path: str) -> Optional[float]:
            try:
                cur = d
                for part in path.split("."):
                    cur = cur[part]
                if isinstance(cur, (int, float, np.floating)):
                    return float(cur)
            except Exception:
                return None
            return None

        if key:
            # np. "threshold_optimization.best_f1.score"
            if "." in key:
                v = _get_nested(metrics, key)
                if v is not None:
                    return v
            if key in metrics and isinstance(metrics[key], (int, float, np.floating)):
                return float(metrics[key])

        # fallback kaskadowy
        for cand in ["accuracy", "f1_weighted", "roc_auc", "roc_auc_ovr", "r2", "rmse", "mae"]:
            if cand in metrics and isinstance(metrics[cand], (int, float, np.floating)):
                return float(metrics[cand])

        return None

    # === OUT: CAP PREDICTIONS ===
    def _maybe_slice_predictions(self, preds_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if preds_df is None or self.config.sample_predictions_cap is None:
            return preds_df
        if len(preds_df) <= self.config.sample_predictions_cap:
            return preds_df
        # stabilny sampling po indeksie (los z seedem dla reprod)
        rng = np.random.default_rng(42)
        idx = rng.choice(preds_df.index.to_numpy(), size=self.config.sample_predictions_cap, replace=False)
        return preds_df.loc[np.sort(idx)]
