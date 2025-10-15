# === performance_tracker.py (PRO++++ / KOSMOS) ===
"""
DataGenius PRO++++ — Performance Tracker (KOSMOS)
Śledzenie jakości modeli ML w czasie: liczenie metryk, log historii, porównanie do baseline
('last' / 'best' / 'rolling'), progi SLO, alerty, trendy i telemetria. Stabilny kontrakt + defensywa.

Najważniejsze cechy KOSMOS:
- Klasyfikacja: accuracy, precision/recall/f1 (weighted), log_loss, brier (bin), ROC-AUC (bin/multiclass ovr/ovo),
  Average Precision (bin/multi-ovr), kompaktowy confusion insight.
- Regresja: r2, mae, mse, rmse, mape (bezpieczne dla zer).
- Baseline: 'last' | 'best' | 'rolling' (średnia z okna), porównanie metryk z deltami i %.
- SLO: progi bezwzględne (acc/f1/r2) + względne wzrosty błędu (rmse/mae) vs baseline.
- Historia: CSV (domyślnie) z bezpiecznym dopisywaniem (nagłówki, precyzja float), opcjonalny Parquet.
- Trendy: rolling mean, slope (kierunek zmian), najlepszy wynik i timestamp.
- Telemetria: czasy, rozmiary, ścieżki, liczby rekordów, wersja schematu.
- Defensywa: walidacje, obsługa kształtów y_proba, braków kolumn, brak historii.
- API utylitarny: get_history(), get_latest(), clear_history(), export_history_parquet(), set_slo_thresholds().
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
    brier_score_loss, roc_auc_score, average_precision_score, confusion_matrix
)

from core.base_agent import BaseAgent, AgentResult
from config.settings import settings


# === KONFIG ===
@dataclass(frozen=True)
class PerformanceConfig:
    """Konfiguracja progów, plików i zachowania trackera."""
    # Plik z historią
    filename: str = "performance_log.csv"
    # Progi SLO (możesz nadpisać w konstruktorze/agencie)
    min_accuracy: float = 0.85
    min_f1: float = 0.85
    min_r2: float = 0.70
    max_rmse_increase_pct: float = 25.0   # maks. wzrost RMSE vs baseline [%]
    max_mae_increase_pct: float = 25.0
    # Trend / okna
    rolling_window: int = 5
    # Bezpieczeństwo zapisu
    allow_overwrite_file: bool = True      # umożliwia dopisywanie do logu
    float_precision: int = 6
    # Opcje historii
    write_parquet_also: bool = False
    parquet_filename: str = "performance_log.parquet"
    # Klasyfikacja (strategia AUC)
    roc_multi_strategy: Literal["ovr", "ovo"] = "ovr"  # domyślnie OVR
    ap_multi_strategy: Literal["ovr", "macro"] = "ovr" # AP po OVR lub macro-average
    # Schemat/log
    schema_version: str = "1.2"


class PerformanceTracker(BaseAgent):
    """
    Śledzi i raportuje jakość modelu: metryki → zapis → baseline → alerty → trendy.
    PRO++++: defensywa, bogate metryki, rolling baseline, telemetria.
    """

    version: str = "4.2-kosmos"

    def __init__(self, config: Optional[PerformanceConfig] = None):
        super().__init__(
            name="PerformanceTracker",
            description="Tracks model performance over time with SLO thresholds and baseline comparisons"
        )
        self.config = config or PerformanceConfig()
        self.metrics_path: Path = Path(getattr(settings, "METRICS_PATH", Path("metrics")))
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.file_path: Path = self.metrics_path / self.config.filename
        self.parquet_path: Path = self.metrics_path / self.config.parquet_filename

    # === API GŁÓWNE ===
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
        compare_to: Literal["last", "best", "rolling", "none"] = "last",
    ) -> AgentResult:
        """
        Policz i zapisz metryki, porównaj do baseline i zwróć alerty/podsumowania.

        Args:
            problem_type: 'classification' | 'regression'
            y_true, y_pred: etykiety/ciągłe wartości (tej samej długości)
            y_proba: opcjonalnie prawdopodobieństwa (bin: (n,) lub (n,2); multi: (n,k))
            run_id: identyfikator uruchomienia (commit, job-id)
            model_name/model_version: meta modelu
            dataset_name: np. 'production_2025-10-14'
            metadata: dowolne pola dodatkowe (dict -> z prefiksem meta_)
            compare_to: baseline: 'last' | 'best' | 'rolling' | 'none'
        """
        result = AgentResult(agent_name=self.name)
        t0 = datetime.utcnow()

        try:
            # Walidacja
            y_true_arr, y_pred_arr, y_proba_arr = self._validate_inputs(problem_type, y_true, y_pred, y_proba)

            # Liczenie metryk
            if problem_type == "classification":
                metrics = self._compute_classification_metrics(y_true_arr, y_pred_arr, y_proba_arr)
            else:
                metrics = self._compute_regression_metrics(y_true_arr, y_pred_arr)

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

            # Baseline i porównanie
            baseline, comparison = (None, None)
            if compare_to in {"last", "best", "rolling"}:
                history = self._read_history()
                baseline = self._choose_baseline(
                    history, problem_type, model_name, model_version, mode=compare_to
                )
                comparison = self._compare_records(problem_type, record, baseline) if baseline else None

            # SLO / alerty
            alerts = self._evaluate_thresholds(problem_type, record, comparison)

            # Trendy / podsumowania historii
            history_summary = self._summarize_history(problem_type, model_name, model_version)

            # Telemetria
            elapsed_s = (datetime.utcnow() - t0).total_seconds()
            telemetry = {
                "schema_version": self.config.schema_version,
                "tracker_version": self.version,
                "path_csv": str(self.file_path),
                "path_parquet": str(self.parquet_path) if self.config.write_parquet_also else None,
                "elapsed_s": round(elapsed_s, 4),
                "history_rows": int(self._safe_len_history()),
            }

            result.data = {
                "record": record,
                "comparison": comparison,
                "baseline": baseline,
                "alerts": alerts,
                "history_summary": history_summary,
                "telemetry": telemetry,
                "log_path": str(self.file_path),
            }
            self.logger.success("Performance tracking complete")

        except Exception as e:
            result.add_error(f"Performance tracking failed: {e}")
            self.logger.error(f"Performance tracking error: {e}", exc_info=True)

        return result

    # === UŻYTECZNE AKCJE PUBLICZNE ===
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

    def get_latest(self, model_name: Optional[str] = None, model_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Zwraca najnowszy wpis zgodny z filtrami lub None."""
        df = self.get_history(model_name=model_name, model_version=model_version, limit=1)
        return df.iloc[0].to_dict() if not df.empty else None

    def clear_history(self) -> None:
        """Czyści historię (usuwa pliki)."""
        try:
            if self.file_path.exists():
                self.file_path.unlink()
            if self.config.write_parquet_also and self.parquet_path.exists():
                self.parquet_path.unlink()
            self.logger.warning("Performance history cleared.")
        except Exception as e:
            self.logger.error(f"Failed to clear history: {e}")

    def export_history_parquet(self) -> Optional[str]:
        """Eksportuje aktualną historię do Parquet (zwraca ścieżkę lub None)."""
        try:
            df = self._read_history()
            if df.empty:
                return None
            self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.parquet_path, index=False)
            return str(self.parquet_path)
        except Exception as e:
            self.logger.warning(f"Parquet export failed: {e}")
            return None

    def set_slo_thresholds(
        self,
        *,
        min_accuracy: Optional[float] = None,
        min_f1: Optional[float] = None,
        min_r2: Optional[float] = None,
        max_rmse_increase_pct: Optional[float] = None,
        max_mae_increase_pct: Optional[float] = None,
    ) -> None:
        """Dynamiczna zmiana progów SLO w locie."""
        cfg = self.config
        object.__setattr__(cfg, "min_accuracy", cfg.min_accuracy if min_accuracy is None else float(min_accuracy))
        object.__setattr__(cfg, "min_f1", cfg.min_f1 if min_f1 is None else float(min_f1))
        object.__setattr__(cfg, "min_r2", cfg.min_r2 if min_r2 is None else float(min_r2))
        object.__setattr__(cfg, "max_rmse_increase_pct", cfg.max_rmse_increase_pct if max_rmse_increase_pct is None else float(max_rmse_increase_pct))
        object.__setattr__(cfg, "max_mae_increase_pct", cfg.max_mae_increase_pct if max_mae_increase_pct is None else float(max_mae_increase_pct))
        self.logger.info("SLO thresholds updated.")

    # === LICZENIE METRYK ===
    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray]
    ) -> Dict[str, float | Dict[str, Any]]:
        metrics: Dict[str, float | Dict[str, Any]] = {}

        # accuracy / precision / recall / f1 (weighted)
        try: metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        except Exception: pass
        try: metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception: pass
        try: metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception: pass
        try: metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception: pass

        # Confusion (kompaktowo)
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion"] = {
                "shape": list(cm.shape),
                "diag_sum": int(np.trace(cm)),
                "offdiag_sum": int(cm.sum() - np.trace(cm)),
            }
        except Exception:
            pass

        # probabilistyczne: log_loss / brier / ROC-AUC / Average Precision
        if y_proba is not None:
            # przygotowanie rozkładów dla log_loss
            y_proba_ll: Optional[np.ndarray] = None
            try:
                if y_proba.ndim == 1:
                    p1 = y_proba
                    y_proba_ll = np.vstack([1 - p1, p1]).T
                else:
                    y_proba_ll = y_proba
            except Exception:
                y_proba_ll = None

            # log_loss
            if y_proba_ll is not None:
                try:
                    metrics["log_loss"] = float(log_loss(y_true, y_proba_ll, labels=np.unique(y_true)))
                except Exception:
                    pass

            # Brier (tylko binarny sensownie)
            try:
                if y_proba.ndim == 1:
                    metrics["brier"] = float(brier_score_loss(y_true, y_proba))
                elif y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    metrics["brier"] = float(brier_score_loss(y_true, y_proba[:, 1]))
            except Exception:
                pass

            # ROC-AUC
            try:
                classes = np.unique(y_true)
                if len(classes) == 2:
                    # binarny: spróbuj kolumny pozytywnej
                    proba_pos = y_proba if y_proba.ndim == 1 else (y_proba[:, 1] if y_proba.shape[1] >= 2 else y_proba.ravel())
                    metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
                else:
                    multi = self.config.roc_multi_strategy
                    metrics[f"roc_auc_{multi}"] = float(roc_auc_score(y_true, y_proba, multi_class=multi))
            except Exception:
                pass

            # Average Precision (PR AUC)
            try:
                classes = np.unique(y_true)
                if len(classes) == 2:
                    proba_pos = y_proba if y_proba.ndim == 1 else (y_proba[:, 1] if y_proba.shape[1] >= 2 else y_proba.ravel())
                    metrics["average_precision"] = float(average_precision_score(y_true, proba_pos))
                else:
                    strat = self.config.ap_multi_strategy
                    if strat == "ovr":
                        ap_vals = []
                        for c in classes:
                            y_bin = (y_true == c).astype(int)
                            ap_vals.append(average_precision_score(y_bin, y_proba[:, list(classes).index(c)]))
                        metrics["average_precision_ovr_macro"] = float(np.mean(ap_vals))
                    else:
                        # macro over predicted labels (fallback)
                        metrics["average_precision_macro"] = float(average_precision_score(y_true, y_pred, average="macro"))
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
            mask = (y_true != 0) & (~np.isnan(y_true))
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
                metrics["mape"] = float(mape)
        except Exception:
            pass

        return metrics

    # === ZAPIS / HISTORIA ===
    def _append_record(self, record: Dict[str, Any]) -> None:
        """Dopisuje rekord do CSV (tworzy nagłówki przy pierwszym zapisie) + opcj. Parquet."""
        if not self.config.allow_overwrite_file:
            raise PermissionError("Overwrites disabled by config.")

        df = pd.DataFrame([record])
        # format liczb
        df = df.applymap(lambda x: round(x, self.config.float_precision) if isinstance(x, float) else x)
        header_needed = not self.file_path.exists()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.file_path, mode="a", header=header_needed, index=False, encoding="utf-8")

        if self.config.write_parquet_also:
            try:
                # Parquet: dołączanie — odczytaj, konkatenacja i zapis (bez utraty typów)
                if self.parquet_path.exists():
                    old = pd.read_parquet(self.parquet_path)
                    out = pd.concat([old, df], ignore_index=True)
                else:
                    out = df
                out.to_parquet(self.parquet_path, index=False)
            except Exception as e:
                self.logger.warning(f"Parquet append failed: {e}")

    def _read_history(self) -> pd.DataFrame:
        if not self.file_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.file_path, encoding="utf-8")
            # timestamp do datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            return df
        except Exception as e:
            self.logger.warning(f"Failed to read history: {e}")
            return pd.DataFrame()

    def _safe_len_history(self) -> int:
        try:
            if not self.file_path.exists():
                return 0
            with self.file_path.open("r", encoding="utf-8") as f:
                # szybkie przybliżenie: linie - 1 (nagłówek)
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            return 0

    # === REKORD / WALIDACJE ===
    def _build_record(
        self,
        *,
        problem_type: str,
        metrics: Dict[str, Any],
        run_id: Optional[str],
        model_name: Optional[str],
        model_version: Optional[str],
        dataset_name: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "schema_version": self.config.schema_version,
            "tracker_version": self.version,
            "problem_type": problem_type,
            "run_id": run_id or "",
            "model_name": model_name or "",
            "model_version": model_version or "",
            "dataset_name": dataset_name or "",
        }
        # spłaszcz metryki (dicty do json)
        for k, v in metrics.items():
            record[k] = v if not isinstance(v, dict) else pd.io.json.dumps(v, ensure_ascii=False)
        # meta
        if metadata:
            for k, v in metadata.items():
                record[f"meta_{k}"] = v
        return record

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
        mode: Literal["last", "best", "rolling"]
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
            return row.to_dict()

        if mode == "best":
            if key_metric not in df.columns:
                return None
            row = df.sort_values(key_metric, ascending=False).iloc[0]
            return row.to_dict()

        # rolling baseline: średnia z ostatniego okna
        window = max(2, self.config.rolling_window)
        df = df.sort_values("timestamp")
        if key_metric not in df.columns or len(df) < 2:
            return None
        tail = df.tail(window)
        # baseline jako „syntetyczny” rekord
        base: Dict[str, Any] = tail.mean(numeric_only=True).to_dict()
        base["timestamp"] = str(tail["timestamp"].iloc[-1])
        base["__mode__"] = f"rolling_last_{len(tail)}"
        return base

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
            cur = self._safe_float(current.get(k))
            base = self._safe_float(baseline.get(k))
            if cur is None or base is None:
                continue
            delta = cur - base
            pct = (delta / base * 100.0) if base not in (0.0, None) else None
            out[k] = {
                "current": float(cur),
                "baseline": float(base),
                "delta": float(delta),
                "delta_pct": (float(pct) if pct is not None and np.isfinite(pct) else None),
            }
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
            acc = self._safe_float(record.get("accuracy"))
            f1v = self._safe_float(record.get("f1"))
            if acc is not None and acc < c.min_accuracy:
                alerts.append(f"⚠️ accuracy {acc:.3f} poniżej progu {c.min_accuracy:.3f}")
            if f1v is not None and f1v < c.min_f1:
                alerts.append(f"⚠️ f1 {f1v:.3f} poniżej progu {c.min_f1:.3f}")

        else:
            r2v = self._safe_float(record.get("r2"))
            if r2v is not None and r2v < c.min_r2:
                alerts.append(f"⚠️ r2 {r2v:.3f} poniżej progu {c.min_r2:.3f}")

        # względny wzrost błędu vs baseline
        if comparison and problem_type == "regression":
            rmse = comparison.get("rmse", {})
            mae = comparison.get("mae", {})
            rmse_pct = self._safe_float(rmse.get("delta_pct")) if isinstance(rmse, dict) else None
            mae_pct = self._safe_float(mae.get("delta_pct")) if isinstance(mae, dict) else None
            if rmse_pct is not None and rmse_pct > c.max_rmse_increase_pct:
                alerts.append(f"📉 RMSE wzrósł o {rmse_pct:.1f}% (> {c.max_rmse_increase_pct:.0f}%) względem baseline")
            if mae_pct is not None and mae_pct > c.max_mae_increase_pct:
                alerts.append(f"📉 MAE wzrósł o {mae_pct:.1f}% (> {c.max_mae_increase_pct:.0f}%) względem baseline")

        if not alerts:
            alerts.append("✅ Brak naruszeń progów SLO / baseline.")
        return alerts

    # === TRENDY / PODSUMOWANIA HISTORII ===
    def _summarize_history(
        self,
        problem_type: str,
        model_name: Optional[str],
        model_version: Optional[str]
    ) -> Dict[str, Any]:
        df = self.get_history(model_name=model_name, model_version=model_version)
        if df.empty:
            return {"message": "No history yet.", "rolling_window": self.config.rolling_window}

        df = df.sort_values("timestamp")
        window = max(2, self.config.rolling_window)

        def slope_of(series: pd.Series) -> Optional[float]:
            try:
                y = series.dropna().values
                if len(y) < 2:
                    return None
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                return float(slope)
            except Exception:
                return None

        keys = ["accuracy", "f1", "precision", "recall"] if problem_type == "classification" else ["r2", "rmse", "mae", "mse"]

        summary: Dict[str, Any] = {"rolling_window": window}
        for k in keys:
            if k in df.columns:
                roll = df[k].rolling(window=window, min_periods=2).mean()
                summary[f"{k}_rolling_mean"] = (float(roll.iloc[-1]) if not roll.isna().all() else None)
                summary[f"{k}_trend_slope"] = slope_of(df[k])

        # najlepszy wynik
        try:
            if problem_type == "classification" and "accuracy" in df.columns:
                best_row = df.sort_values("accuracy", ascending=False).iloc[0]
                summary["best_accuracy"] = float(best_row["accuracy"])
                summary["best_ts"] = str(best_row["timestamp"])
            elif problem_type == "regression" and "r2" in df.columns:
                best_row = df.sort_values("r2", ascending=False).iloc[0]
                summary["best_r2"] = float(best_row["r2"])
                summary["best_ts"] = str(best_row["timestamp"])
        except Exception:
            pass

        return summary

    # === POMOCNICZE ===
    @staticmethod
    def _safe_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            if isinstance(v, str) and not v.strip():
                return None
            f = float(v)
            return f if np.isfinite(f) else None
        except Exception:
            return None
