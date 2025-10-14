# === performance_tracker.py ===
"""
DataGenius PRO - Performance Tracker (PRO+++)
Śledzenie jakości modeli ML w czasie: liczenie metryk, zapisy historii,
porównanie do baseline i progi SLO, alerty oraz podsumowania trendów.

Zależności: pandas, numpy, scikit-learn (metrics), loguru.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Literal, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error, log_loss,
    brier_score_loss
)

from core.base_agent import BaseAgent, AgentResult
from config.settings import settings


# === KONFIG ===
@dataclass(frozen=True)
class PerformanceConfig:
    """Konfiguracja progów i zachowania trackera."""
    # Plik z historią
    filename: str = "performance_log.csv"
    # Progi SLO (możesz nadpisać w konstruktorze/agencie)
    min_accuracy: float = 0.85
    min_f1: float = 0.85
    min_r2: float = 0.70
    max_rmse_increase_pct: float = 25.0   # maks. dopuszczalny wzrost RMSE vs baseline [%]
    max_mae_increase_pct: float = 25.0
    # Trend / okna
    rolling_window: int = 5
    # Bezpieczeństwo
    allow_overwrite_file: bool = True      # umożliwia tworzenie/aktualizację pliku logu
    # Precyzja zapisu
    float_precision: int = 6


class PerformanceTracker(BaseAgent):
    """
    Śledzi i raportuje jakość modelu: liczenie metryk, zapis historii, porównania, alerty.
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        super().__init__(
            name="PerformanceTracker",
            description="Tracks model performance over time with SLO thresholds and baseline comparisons"
        )
        self.config = config or PerformanceConfig()
        self.metrics_path: Path = Path(getattr(settings, "METRICS_PATH", Path("metrics")))
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.file_path: Path = self.metrics_path / self.config.filename

    # === GŁÓWNY INTERFEJS ===
    def execute(
        self,
        problem_type: Literal["classification", "regression"],
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        *,
        y_proba: Optional[np.ndarray] = None,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        dataset_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compare_to: Literal["last", "best", "none"] = "last",
    ) -> AgentResult:
        """
        Policz i zapisz metryki, porównaj do baseline i zwróć alerty/podsumowania.

        Args:
            problem_type: 'classification' | 'regression'
            y_true, y_pred: etykiety/ciągłe wartości
            y_proba: (opcjonalnie) prawdopodobieństwa/densities (np. (n, n_classes) lub (n,))
            run_id: identyfikator uruchomienia (np. commit SHA, job-id)
            model_name/model_version: metadane modelu
            dataset_name: np. 'production_2025-10-14'
            metadata: dowolne info dodatkowe (słownik)
            compare_to: baseline: 'last' (ostatni wpis), 'best' (najlepszy wg kluczowej metryki), 'none'
        """
        result = AgentResult(agent_name=self.name)
        try:
            # Walidacja
            y_true_arr, y_pred_arr, y_proba_arr = self._validate_inputs(problem_type, y_true, y_pred, y_proba)

            # Liczenie metryk
            metrics = (
                self._compute_classification_metrics(y_true_arr, y_pred_arr, y_proba_arr)
                if problem_type == "classification"
                else self._compute_regression_metrics(y_true_arr, y_pred_arr)
            )

            # Budowa rekordu
            record = self._build_record(
                problem_type=problem_type,
                metrics=metrics,
                run_id=run_id,
                model_name=model_name,
                model_version=model_version,
                dataset_name=dataset_name,
                metadata=metadata
            )

            # Zapis historii
            self._append_record(record)

            # Porównanie do baseline
            baseline, comparison = (None, None)
            if compare_to in {"last", "best"}:
                history = self._read_history()
                baseline = self._choose_baseline(history, problem_type, model_name, model_version, mode=compare_to)
                comparison = self._compare_records(problem_type, record, baseline) if baseline is not None else None

            # Alerty SLO
            alerts = self._evaluate_thresholds(problem_type, record, comparison)

            # Podsumowanie trendów
            history_summary = self._summarize_history(problem_type, model_name, model_version)

            result.data = {
                "record": record,
                "comparison": comparison,
                "baseline": baseline,
                "alerts": alerts,
                "history_summary": history_summary,
                "log_path": str(self.file_path),
            }
            self.logger.success("Performance tracking complete")

        except Exception as e:
            result.add_error(f"Performance tracking failed: {e}")
            self.logger.error(f"Performance tracking error: {e}", exc_info=True)

        return result

    # === METODY POMOCNICZE API ===
    def get_history(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        dataset_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Zwraca historię z opcjonalnymi filtrami (ostatnie `limit` wierszy po czasie)."""
        df = self._read_history()
        if df.empty:
            return df
        if model_name:
            df = df[df["model_name"] == model_name]
        if model_version:
            df = df[df["model_version"] == model_version]
        if dataset_name:
            df = df[df["dataset_name"] == dataset_name]
        df = df.sort_values("timestamp", ascending=False)
        if limit:
            df = df.head(limit)
        return df.reset_index(drop=True)

    # === LICZENIE METRYK ===
    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray]
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # accuracy / precision / recall / f1 (weighted)
        try: metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        except Exception: pass
        try: metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception: pass
        try: metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception: pass
        try: metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception: pass

        # probabilistyczne: log_loss / brier (gdy dostępne proby)
        if y_proba is not None:
            try:
                # Obsługa binarnej tablicy (n,) lub (n,2) i multiclass (n,k)
                if y_proba.ndim == 1:
                    # zakładamy prawdopodobieństwo klasy pozytywnej
                    # do log_loss potrzebne pełne rozkłady -> przemapowanie na [p, 1-p]
                    p1 = y_proba
                    y_proba_ll = np.vstack([1 - p1, p1]).T
                else:
                    y_proba_ll = y_proba
                metrics["log_loss"] = float(log_loss(y_true, y_proba_ll, labels=np.unique(y_true)))
            except Exception:
                pass

            try:
                # Brier sensowny dla binarnej: jeśli multiclass, pomijamy
                if y_proba.ndim == 1:
                    metrics["brier"] = float(brier_score_loss(y_true, y_proba))
                elif y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    metrics["brier"] = float(brier_score_loss(y_true, y_proba[:, 1]))
            except Exception:
                pass

        return metrics

    def _compute_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        try: metrics["r2"] = float(r2_score(y_true, y_pred))
        except Exception: pass
        try:
            mse = mean_squared_error(y_true, y_pred)
            metrics["mse"] = float(mse)
            metrics["rmse"] = float(np.sqrt(mse))
        except Exception: pass
        try: metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        except Exception: pass
        try:
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
                metrics["mape"] = float(mape)
        except Exception: pass

        return metrics

    # === ZAPIS / HISTORIA ===
    def _append_record(self, record: Dict[str, Any]) -> None:
        """Dopisuje rekord do CSV (tworzy nagłówki przy pierwszym zapisie)."""
        if not self.config.allow_overwrite_file:
            raise PermissionError("Overwrites disabled by config.")
        df = pd.DataFrame([record])
        # format liczb
        df = df.applymap(lambda x: round(x, self.config.float_precision) if isinstance(x, float) else x)
        header_needed = not self.file_path.exists()
        df.to_csv(self.file_path, mode="a", header=header_needed, index=False, encoding="utf-8")

    def _read_history(self) -> pd.DataFrame:
        if not self.file_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.file_path, encoding="utf-8")
            # timestamp do datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df
        except Exception as e:
            self.logger.warning(f"Failed to read history: {e}")
            return pd.DataFrame()

    # === BUDOWA REKORDU ===
    def _build_record(
        self,
        *,
        problem_type: str,
        metrics: Dict[str, float],
        run_id: Optional[str],
        model_name: Optional[str],
        model_version: Optional[str],
        dataset_name: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "problem_type": problem_type,
            "run_id": run_id or "",
            "model_name": model_name or "",
            "model_version": model_version or "",
            "dataset_name": dataset_name or "",
        }
        record.update(metrics)
        # Spłaszcz meta (z prefixem meta_)
        if metadata:
            for k, v in metadata.items():
                record[f"meta_{k}"] = v
        return record

    # === WALIDACJA WEJŚCIA ===
    def _validate_inputs(
        self,
        problem_type: str,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_proba: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be 'classification' or 'regression'")
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")
        y_proba_arr = None
        if y_proba is not None:
            y_proba_arr = np.asarray(y_proba)
            if y_proba_arr.shape[0] != y_true_arr.shape[0]:
                raise ValueError("y_proba must have the same number of rows as y_true")
        return y_true_arr, y_pred_arr, y_proba_arr

    # === BASELINE WYBÓR I PORÓWNANIE ===
    def _choose_baseline(
        self,
        history: pd.DataFrame,
        problem_type: str,
        model_name: Optional[str],
        model_version: Optional[str],
        *,
        mode: Literal["last", "best"]
    ) -> Optional[Dict[str, Any]]:
        if history.empty:
            return None

        df = history.copy()
        if model_name:
            df = df[df["model_name"] == model_name]
        if model_version:
            df = df[df["model_version"] == model_version]

        if df.empty:
            return None

        key_metric = "accuracy" if problem_type == "classification" else "r2"

        if mode == "last":
            row = df.sort_values("timestamp").iloc[-1]
        else:  # best
            if key_metric not in df.columns:
                return None
            # max dla accuracy/r2
            row = df.sort_values(key_metric, ascending=False).iloc[0]

        return row.to_dict()

    def _compare_records(
        self,
        problem_type: str,
        current: Dict[str, Any],
        baseline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Porównanie kluczowych metryk current vs baseline (+% delta)."""
        out: Dict[str, Any] = {"against": baseline.get("timestamp", "baseline")}
        if problem_type == "classification":
            keys = ["accuracy", "f1", "precision", "recall"]
        else:
            keys = ["r2", "rmse", "mae", "mse"]

        for k in keys:
            cur = current.get(k)
            base = baseline.get(k)
            if isinstance(cur, (int, float)) and isinstance(base, (int, float)):
                delta = cur - base
                pct = (delta / base * 100.0) if (base not in (None, 0)) else np.nan
                out[k] = {"current": float(cur), "baseline": float(base), "delta": float(delta), "delta_pct": float(pct) if not np.isnan(pct) else None}
        return out

    # === PROGI I ALERTY ===
    def _evaluate_thresholds(
        self,
        problem_type: str,
        record: Dict[str, Any],
        comparison: Optional[Dict[str, Any]]
    ) -> List[str]:
        alerts: List[str] = []
        c = self.config

        if problem_type == "classification":
            # progi bezwzględne
            acc = record.get("accuracy")
            f1 = record.get("f1")
            if isinstance(acc, (int, float)) and acc < c.min_accuracy:
                alerts.append(f"⚠️ accuracy {acc:.3f} poniżej progu {c.min_accuracy:.3f}")
            if isinstance(f1, (int, float)) and f1 < c.min_f1:
                alerts.append(f"⚠️ f1 {f1:.3f} poniżej progu {c.min_f1:.3f}")

        else:
            r2 = record.get("r2")
            if isinstance(r2, (int, float)) and r2 < c.min_r2:
                alerts.append(f"⚠️ r2 {r2:.3f} poniżej progu {c.min_r2:.3f}")

        # względny wzrost błędu vs baseline
        if comparison:
            if problem_type == "regression":
                rmse = comparison.get("rmse", {})
                mae = comparison.get("mae", {})
                rmse_pct = rmse.get("delta_pct") if isinstance(rmse, dict) else None
                mae_pct = mae.get("delta_pct") if isinstance(mae, dict) else None
                if isinstance(rmse_pct, (int, float)) and rmse_pct > c.max_rmse_increase_pct:
                    alerts.append(f"📉 RMSE wzrósł o {rmse_pct:.1f}% (> {c.max_rmse_increase_pct:.0f}%) względem baseline")
                if isinstance(mae_pct, (int, float)) and mae_pct > c.max_mae_increase_pct:
                    alerts.append(f"📉 MAE wzrósł o {mae_pct:.1f}% (> {c.max_mae_increase_pct:.0f}%) względem baseline")

        if not alerts:
            alerts.append("✅ Brak naruszeń progów SLO / baseline.")
        return alerts

    # === TRENDY / PODSUMOWANIE HISTORII ===
    def _summarize_history(
        self,
        problem_type: str,
        model_name: Optional[str],
        model_version: Optional[str]
    ) -> Dict[str, Any]:
        df = self.get_history(model_name=model_name, model_version=model_version)
        if df.empty:
            return {"message": "No history yet."}

        df = df.sort_values("timestamp")
        window = max(2, self.config.rolling_window)

        def slope_of(series: pd.Series) -> Optional[float]:
            try:
                y = series.dropna().values
                x = np.arange(len(y))
                if len(y) < 2:
                    return None
                slope = np.polyfit(x, y, 1)[0]
                return float(slope)
            except Exception:
                return None

        if problem_type == "classification":
            keys = ["accuracy", "f1", "precision", "recall"]
        else:
            keys = ["r2", "rmse", "mae", "mse"]

        summary: Dict[str, Any] = {"rolling_window": window}
        for k in keys:
            if k in df.columns:
                roll = df[k].rolling(window=window, min_periods=2).mean()
                summary[f"{k}_rolling_mean"] = float(roll.iloc[-1]) if not roll.isna().all() else None
                summary[f"{k}_trend_slope"] = slope_of(df[k])

        # najlepszy wynik
        if problem_type == "classification" and "accuracy" in df.columns:
            best_row = df.sort_values("accuracy", ascending=False).iloc[0]
            summary["best_accuracy"] = float(best_row["accuracy"])
            summary["best_ts"] = str(best_row["timestamp"])
        elif problem_type == "regression" and "r2" in df.columns:
            best_row = df.sort_values("r2", ascending=False).iloc[0]
            summary["best_r2"] = float(best_row["r2"])
            summary["best_ts"] = str(best_row["timestamp"])

        return summary
