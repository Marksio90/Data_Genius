# === OPIS MODUŁU ===
"""
DataGenius PRO - Outlier Detector (PRO++++)
Wykrywa outliery metodami IQR, Z-score, Robust Z-score (Median+MAD) oraz Isolation Forest (opcjonalnie),
z defensywnymi guardami, telemetry i spójnym kontraktem danych.

Kontrakt (AgentResult.data):
{
  "iqr_method": {
      "method": "IQR (factor=...)",
      "description": str,
      "params": {"iqr_factor": float},
      "columns": {col: {"n_outliers": int, "percentage": float, "lower_bound": float, "upper_bound": float,
                        "outlier_indices": List[Any]}},
      "n_columns_with_outliers": int
  },
  "zscore_method": {
      "method": "Z-Score",
      "description": str,
      "params": {"threshold": float},
      "columns": {col: {"n_outliers": int, "percentage": float, "threshold": float,
                        "outlier_indices": List[Any]}},
      "n_columns_with_outliers": int
  },
  "robust_zscore_method": {
      "method": "Robust Z-Score (Median+MAD)",
      "description": str,
      "params": {"threshold": float},
      "columns": {col: {"n_outliers": int, "percentage": float, "threshold": float,
                        "median": float, "mad": float, "outlier_indices": List[Any]}},
      "n_columns_with_outliers": int
  },
  "isolation_forest": {
      "method": "Isolation Forest",
      "description": str,
      "n_outliers": int,
      "percentage": float,
      "outlier_indices": List[Any],
      "contamination": float,
      "rows_used": int,
      "n_features_used": int,
      "scaled": true
  } | None,
  "summary": {
      "total_outliers_rows_union": int,               # unia indeksów (OR) po wszystkich metodach
      "by_method": {"IQR": int, "Z-Score": int, "RobustZ": int, "Isolation Forest": int},
      "n_columns_with_outliers": int,                 # (IQR ∪ Z ∪ RobustZ)
      "methods_used": List[str],
      "most_outliers": {"column": str, "n_outliers": int} | None,
      "example_outlier_indices": List[Any]
  },
  "recommendations": List[str],
  "telemetry": {
      "elapsed_ms": float,
      "timings_ms": {"iqr": float, "zscore": float, "robust_z": float, "iforest": float},
      "sampled_iforest": bool,
      "iforest_sample_info": {"from_rows": int, "to_rows": int} | None
  }
}
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import IsolationForest

from core.base_agent import BaseAgent, AgentResult


# === NAZWA_SEKCJI === KONFIG / PROGI ===
@dataclass(frozen=True)
class OutlierConfig:
    """Progi i ustawienia wykrywania outlierów."""
    # — IQR —
    iqr_factor: float = 1.5
    # — klasyczny Z-score —
    zscore_threshold: float = 3.0
    # — robust Z-score (Median+MAD) —
    robust_z_threshold: float = 3.5
    # — Isolation Forest —
    enable_isolation_forest: bool = True
    min_rows_for_if: int = 100
    if_random_state: int = 42
    if_n_jobs: int = -1
    if_contam_min: float = 0.01
    if_contam_max: float = 0.20
    if_max_rows: int = 500_000            # sampling safety dla IF
    if_max_features: int = 500            # cap liczby kolumn do IF (wysokowymiarowe macierze)
    # — wynik/telemetria —
    max_indices_return: int = 25           # ile przykładowych indeksów zwracać
    # — bezpieczeństwo —
    clip_inf_to_nan: bool = True           # zamiana inf/-inf → NaN (potem median-impute do IF)


# === NAZWA_SEKCJI === KLASA GŁÓWNA AGENDA ===
class OutlierDetector(BaseAgent):
    """
    Detects outliers using IQR, Z-score, Robust Z-score and Isolation Forest (opcjonalnie).
    Defensywna, szybka i skalowalna implementacja PRO++++.
    """

    def __init__(self, config: Optional[OutlierConfig] = None) -> None:
        super().__init__(name="OutlierDetector", description="Detects outliers in numeric features")
        self.config = config or OutlierConfig()
        self._log = logger.bind(agent="OutlierDetector")

    # === NAZWA_SEKCJI === WYKONANIE GŁÓWNE ===
    def execute(self, data: pd.DataFrame, **kwargs: Any) -> AgentResult:
        """
        Detect outliers across numeric features using multiple robust methods.
        """
        result = AgentResult(agent_name=self.name)
        t0_total = time.perf_counter()

        try:
            # Walidacja
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                msg = "OutlierDetector: 'data' must be a non-empty pandas DataFrame"
                result.add_error(msg)
                self._log.error(msg)
                return result

            # Numeric-only (Inf→NaN opcjonalnie)
            df_num = data.select_dtypes(include=[np.number]).copy()
            if df_num.shape[1] == 0:
                result.add_warning("No numeric columns for outlier detection")
                result.data = self._empty_payload_no_numeric()
                result.data["telemetry"]["elapsed_ms"] = round((time.perf_counter() - t0_total) * 1000, 1)
                return result

            if self.config.clip_inf_to_nan:
                df_num = df_num.replace([np.inf, -np.inf], np.nan)

            # === 1) IQR ===
            t0 = time.perf_counter()
            iqr_dict, mask_iqr = self._detect_iqr_outliers(df_num)
            t_iqr = (time.perf_counter() - t0) * 1000

            # === 2) Z-score ===
            t0 = time.perf_counter()
            z_dict, mask_z = self._detect_zscore_outliers(df_num, threshold=self.config.zscore_threshold)
            t_z = (time.perf_counter() - t0) * 1000

            # === 3) Robust Z-score (Median+MAD) ===
            t0 = time.perf_counter()
            rz_dict, mask_rz = self._detect_robust_zscore_outliers(df_num, threshold=self.config.robust_z_threshold)
            t_rz = (time.perf_counter() - t0) * 1000

            # === 4) Isolation Forest (opcjonalnie, z samplingiem/robust-scaling) ===
            t0 = time.perf_counter()
            if_dict, mask_if, sampled_if, sample_info_if = None, None, False, None
            if self.config.enable_isolation_forest and len(df_num) >= self.config.min_rows_for_if:
                est_contam = self._estimate_contamination_from_iqr(iqr_dict, n_rows=len(df_num))
                if_dict, mask_if, sampled_if, sample_info_if = self._detect_isolation_forest_outliers(
                    df_num, contamination=est_contam
                )
            t_if = (time.perf_counter() - t0) * 1000

            # === 5) Summary (unia wierszy) ===
            summary = self._create_summary(iqr_dict, z_dict, rz_dict, if_dict, mask_iqr, mask_z, mask_rz, mask_if)

            # === 6) Rekomendacje ===
            recommendations = self._get_recommendations(iqr_dict, z_dict, rz_dict, if_dict, summary)

            # === 7) Złóż wynik + telemetry ===
            result.data = {
                "iqr_method": iqr_dict,
                "zscore_method": z_dict,
                "robust_zscore_method": rz_dict,
                "isolation_forest": if_dict,
                "summary": summary,
                "recommendations": recommendations,
                "telemetry": {
                    "elapsed_ms": round((time.perf_counter() - t0_total) * 1000, 1),
                    "timings_ms": {
                        "iqr": round(t_iqr, 1), "zscore": round(t_z, 1),
                        "robust_z": round(t_rz, 1), "iforest": round(t_if, 1)
                    },
                    "sampled_iforest": bool(sampled_if),
                    "iforest_sample_info": sample_info_if
                }
            }
            self._log.success(f"Outlier detection complete: {summary['total_outliers_rows_union']} rows flagged (union)")

        except Exception as e:
            result.add_error(f"Outlier detection failed: {e}")
            self._log.exception(f"Outlier detection error: {e}")

        return result

    # === NAZWA_SEKCJI === IQR ===
    def _detect_iqr_outliers(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.Series]:
        """Detect outliers using IQR method (z guardem IQR=0). Zwraca (payload, row_mask_union)."""
        cfg = self.config
        outliers_by_column: Dict[str, Dict[str, Any]] = {}
        row_mask = pd.Series(False, index=df.index)

        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s_nona = s.dropna()
            if s_nona.empty:
                continue

            q1, q3 = s_nona.quantile(0.25), s_nona.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue

            lower = q1 - cfg.iqr_factor * iqr
            upper = q3 + cfg.iqr_factor * iqr

            mask = (s < lower) | (s > upper)
            n_out = int(mask.sum())
            if n_out > 0:
                row_mask |= mask.fillna(False)
                outliers_by_column[col] = {
                    "n_outliers": n_out,
                    "percentage": float(n_out / len(s) * 100.0),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "outlier_indices": self._head_indices(mask[mask].index, cfg.max_indices_return),
                }

        payload = {
            "method": f"IQR (factor={cfg.iqr_factor})",
            "description": f"Outliers: < Q1-{cfg.iqr_factor}*IQR or > Q3+{cfg.iqr_factor}*IQR",
            "params": {"iqr_factor": float(cfg.iqr_factor)},
            "columns": outliers_by_column,
            "n_columns_with_outliers": len(outliers_by_column),
        }
        return payload, row_mask

    # === NAZWA_SEKCJI === Z-SCORE ===
    def _detect_zscore_outliers(self, df: pd.DataFrame, threshold: float) -> Tuple[Dict[str, Any], pd.Series]:
        """Detect outliers using classic Z-score (guard std=0, NA-safe). Zwraca (payload, row_mask_union)."""
        outliers_by_column: Dict[str, Dict[str, Any]] = {}
        row_mask = pd.Series(False, index=df.index)

        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s_nona = s.dropna()
            if s_nona.empty:
                continue

            std = float(s_nona.std(ddof=1)) if len(s_nona) > 1 else 0.0
            if std == 0.0:
                continue

            z = np.abs((s - s_nona.mean()) / std)  # NaN-safe dzięki s
            mask = z > threshold
            n_out = int(mask.sum(skipna=True))
            if n_out > 0:
                row_mask |= mask.fillna(False)
                outliers_by_column[col] = {
                    "n_outliers": n_out,
                    "percentage": float(n_out / len(s) * 100.0),
                    "threshold": float(threshold),
                    "outlier_indices": self._head_indices(mask[mask].index, self.config.max_indices_return),
                }

        payload = {
            "method": "Z-Score",
            "description": f"Outliers are values with |z-score| > {threshold}",
            "params": {"threshold": float(threshold)},
            "columns": outliers_by_column,
            "n_columns_with_outliers": len(outliers_by_column),
        }
        return payload, row_mask

    # === NAZWA_SEKCJI === ROBUST Z-SCORE (Median+MAD) ===
    def _detect_robust_zscore_outliers(self, df: pd.DataFrame, threshold: float) -> Tuple[Dict[str, Any], pd.Series]:
        """
        Robust Z-score: z = 0.6745 * (x - median) / MAD; flagujemy |z| > threshold.
        Odporny na outliery i ciężkie ogony. Zwraca (payload, row_mask_union).
        """
        outliers_by_column: Dict[str, Dict[str, Any]] = {}
        row_mask = pd.Series(False, index=df.index)
        c = 0.6745

        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s_nona = s.dropna()
            if s_nona.empty:
                continue

            med = float(np.median(s_nona))
            mad = float(np.median(np.abs(s_nona - med)))
            if mad == 0.0:
                continue  # brak rozrzutu robust — pomijamy

            rz = np.abs(c * (s - med) / mad)
            mask = rz > threshold
            n_out = int(mask.sum(skipna=True))
            if n_out > 0:
                row_mask |= mask.fillna(False)
                outliers_by_column[col] = {
                    "n_outliers": n_out,
                    "percentage": float(n_out / len(s) * 100.0),
                    "threshold": float(threshold),
                    "median": med,
                    "mad": mad,
                    "outlier_indices": self._head_indices(mask[mask].index, self.config.max_indices_return),
                }

        payload = {
            "method": "Robust Z-Score (Median+MAD)",
            "description": f"Outliers where |0.6745*(x-median)/MAD| > {threshold}",
            "params": {"threshold": float(threshold)},
            "columns": outliers_by_column,
            "n_columns_with_outliers": len(outliers_by_column),
        }
        return payload, row_mask

    # === NAZWA_SEKCJI === ISOLATION FOREST ===
    def _detect_isolation_forest_outliers(
        self,
        df: pd.DataFrame,
        contamination: float
    ) -> Tuple[Optional[Dict[str, Any]], Optional[pd.Series], bool, Optional[Dict[str, int]]]:
        """
        IF z robust-scalingiem (median/IQR) i samplingiem dla bardzo dużych danych.
        Zwraca (payload|None, row_mask|None, sampled_flag, sample_info|None).
        """
        cfg = self.config
        sampled = False
        sample_info = None
        X = df.copy()

        # Sampling safety
        if len(X) > cfg.if_max_rows:
            sampled = True
            sample_info = {"from_rows": int(len(X)), "to_rows": int(cfg.if_max_rows)}
            self._log.info(f"IsolationForest sampling: {len(X)} → {cfg.if_max_rows} rows")
            X = X.sample(n=cfg.if_max_rows, random_state=cfg.if_random_state)

        # Limit liczby cech (informacyjnie — IF działa, ale może spowalniać)
        if X.shape[1] > cfg.if_max_features:
            self._log.warning(f"IsolationForest features cap: {X.shape[1]} → {cfg.if_max_features} cols")
            X = X.iloc[:, : cfg.if_max_features]

        # Robust scaling (median/IQR) + imputacja medianą (NaN-safe)
        med = X.median(numeric_only=True)
        iqr = X.quantile(0.75, numeric_only=True) - X.quantile(0.25, numeric_only=True)
        scale = iqr.replace(0.0, 1.0)  # guard
        X_scaled = (X - med) / scale
        X_scaled = X_scaled.fillna(0.0)  # po standaryzacji braków — 0 to mediana

        try:
            iso = IsolationForest(
                contamination=float(contamination),
                random_state=cfg.if_random_state,
                n_jobs=cfg.if_n_jobs
            )
            pred = iso.fit_predict(X_scaled)  # -1 → outlier (na próbie X)
            mask = pd.Series(pred == -1, index=X_scaled.index)

            payload = {
                "method": "Isolation Forest",
                "description": "Unsupervised anomaly detection (tree-based) with robust scaling",
                "n_outliers": int(mask.sum()),
                "percentage": float(mask.mean() * 100.0),
                "outlier_indices": self._head_indices(mask[mask].index, cfg.max_indices_return),
                "contamination": float(contamination),
                "rows_used": int(len(X_scaled)),
                "n_features_used": int(X_scaled.shape[1]),
                "scaled": True,
            }

            # Uwaga: mask dotyczy próbkowanego X; w summary liczymy OR po dostępnych maskach
            return payload, mask.reindex(df.index, fill_value=False), sampled, sample_info

        except Exception as e:
            self._log.warning(f"Isolation Forest failed: {e}")
            return None, None, sampled, sample_info

    # === NAZWA_SEKCJI === ESTYMACJA CONTAMINATION ===
    def _estimate_contamination_from_iqr(self, iqr_outliers: Dict[str, Any], n_rows: int) -> float:
        """
        Szacuje udział anomalii na podstawie IQR:
        - średni odsetek outlierów per kolumna → clamp do [if_contam_min, if_contam_max].
        """
        cfg = self.config
        cols = iqr_outliers.get("columns", {})
        if not cols or n_rows <= 0:
            return max(cfg.if_contam_min, 0.01)

        pcts = [float(v.get("percentage", 0.0)) for v in cols.values()]
        mean_pct = float(np.mean(pcts)) if pcts else 0.0
        est = mean_pct / 100.0
        est = float(np.clip(est, cfg.if_contam_min, cfg.if_contam_max))
        return est

    # === NAZWA_SEKCJI === PODSUMOWANIE ===
    def _create_summary(
        self,
        iqr_outliers: Dict[str, Any],
        zscore_outliers: Dict[str, Any],
        robustz_outliers: Dict[str, Any],
        isolation_outliers: Optional[Dict[str, Any]],
        mask_iqr: Optional[pd.Series],
        mask_z: Optional[pd.Series],
        mask_rz: Optional[pd.Series],
        mask_if: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Create summary oparty o **unię wierszy** z masek metod."""
        methods_used = ["IQR", "Z-Score", "RobustZ"] + (["Isolation Forest"] if isolation_outliers else [])

        # n kolumn z outlierami (z metod kolumnowych)
        cols_with = set(iqr_outliers.get("columns", {}).keys()) \
                    | set(zscore_outliers.get("columns", {}).keys()) \
                    | set(robustz_outliers.get("columns", {}).keys())

        # policz unie masek wierszy
        union_mask = None
        for m in (mask_iqr, mask_z, mask_rz, mask_if):
            if m is None:
                continue
            union_mask = m if union_mask is None else (union_mask | m)
        total_union = int(union_mask.sum()) if union_mask is not None else 0

        # per-method „liczba wierszy” (z masek; dla metod kolumnowych to OR po kolumnach)
        by_method = {
            "IQR": int(mask_iqr.sum()) if mask_iqr is not None else 0,
            "Z-Score": int(mask_z.sum()) if mask_z is not None else 0,
            "RobustZ": int(mask_rz.sum()) if mask_rz is not None else 0,
            "Isolation Forest": int(mask_if.sum()) if mask_if is not None else 0
        }

        # przykład indeksów (do podglądu)
        example_indices: List[Any] = []
        for m in (mask_iqr, mask_z, mask_rz, mask_if):
            if m is not None and m.any():
                example_indices.extend(list(m[m].index[: self.config.max_indices_return]))
        example_indices = list(dict.fromkeys(example_indices))[: self.config.max_indices_return]

        most_iqr = self._get_most_outliers_column(iqr_outliers)
        most_z = self._get_most_outliers_column(zscore_outliers)
        most_rz = self._get_most_outliers_column(robustz_outliers)
        most = max([x for x in (most_iqr, most_z, most_rz) if x is not None],
                   key=lambda d: d["n_outliers"], default=None)

        return {
            "total_outliers_rows_union": int(total_union),
            "by_method": by_method,
            "n_columns_with_outliers": int(len(cols_with)),
            "methods_used": methods_used,
            "most_outliers": most,
            "example_outlier_indices": example_indices
        }

    def _get_most_outliers_column(self, method_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get column with most outliers na podstawie payloadu metody (IQR/Z/RobustZ)."""
        cols = (method_payload or {}).get("columns", {})
        if not cols:
            return None
        max_col, meta = max(cols.items(), key=lambda kv: kv[1].get("n_outliers", 0))
        return {"column": str(max_col), "n_outliers": int(meta.get("n_outliers", 0))}

    # === NAZWA_SEKCJI === REKOMENDACJE ===
    def _get_recommendations(
        self,
        iqr_outliers: Dict[str, Any],
        zscore_outliers: Dict[str, Any],
        robustz_outliers: Dict[str, Any],
        isolation_outliers: Optional[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> List[str]:
        """Buduje listę rekomendacji na podstawie wyników."""
        rec: List[str] = []
        n_cols = summary.get("n_columns_with_outliers", 0)
        total_rows = summary.get("total_outliers_rows_union", 0)

        if total_rows == 0:
            rec.append("✅ Brak wykrytych outlierów — możesz pominąć etap ich obsługi.")
            return rec

        if n_cols > 0:
            rec.append(f"🔎 Wykryto outliery w {n_cols} kolumnach numerycznych — rozważ czyszczenie przed treningiem.")

        # IQR → winsoryzacja / klipy
        if iqr_outliers.get("columns"):
            top = self._get_most_outliers_column(iqr_outliers)
            if top:
                rec.append(f"📦 Najwięcej outlierów w '{top['column']}' — rozważ winsoryzację (IQR) lub transformację (log/yeo-johnson).")

        # Z-score → standaryzacja
        if zscore_outliers.get("n_columns_with_outliers", 0) > 0:
            rec.append("📏 Klasyczny Z-score wskazuje odstępstwa — rozważ standaryzację/robust scaling.")

        # Robust Z → ciężkie ogony
        if robustz_outliers.get("n_columns_with_outliers", 0) > 0:
            rec.append("🛡️ Rozkłady o ciężkich ogonach — preferuj robust metody (Median/MAD, Huber, quantile loss).")

        # IF → anomalia wielowymiarowa
        if isolation_outliers and isolation_outliers.get("n_outliers", 0) > 0:
            rec.append("🤖 Isolation Forest wykrył anomalia wielowymiarowe — rozważ filtrowanie lub użycie flagi 'is_anomaly' jako cechy.")

        # Ogólne zalecenia
        rec.append("🧪 Waliduj wpływ usuwania/kapowania outlierów na metryki na zbiorze treningowym (nie na całym DF).")
        return list(dict.fromkeys(rec))  # dedup

    # === NAZWA_SEKCJI === PAYLOADY POMOCNICZE ===
    def _empty_payload_no_numeric(self) -> Dict[str, Any]:
        return {
            "iqr_method": {"method": "IQR", "description": "No numeric columns", "params": {"iqr_factor": self.config.iqr_factor},
                           "columns": {}, "n_columns_with_outliers": 0},
            "zscore_method": {"method": "Z-Score", "description": "No numeric columns", "params": {"threshold": self.config.zscore_threshold},
                              "columns": {}, "n_columns_with_outliers": 0},
            "robust_zscore_method": {"method": "Robust Z-Score (Median+MAD)", "description": "No numeric columns",
                                     "params": {"threshold": self.config.robust_z_threshold},
                                     "columns": {}, "n_columns_with_outliers": 0},
            "isolation_forest": None,
            "summary": {"total_outliers_rows_union": 0, "by_method": {"IQR": 0, "Z-Score": 0, "RobustZ": 0, "Isolation Forest": 0},
                        "n_columns_with_outliers": 0, "methods_used": ["IQR","Z-Score","RobustZ"], "most_outliers": None,
                        "example_outlier_indices": []},
            "recommendations": ["Brak kolumn numerycznych — etap wykrywania outlierów pominięty."],
            "telemetry": {"elapsed_ms": 0.0, "timings_ms": {"iqr": 0.0, "zscore": 0.0, "robust_z": 0.0, "iforest": 0.0},
                          "sampled_iforest": False, "iforest_sample_info": None}
        }

    # === NAZWA_SEKCJI === POMOCNICZE ===
    @staticmethod
    def _head_indices(idx: pd.Index, n: int) -> List[Any]:
        """Zwraca pierwsze n indeksów jako listę (do podglądu)."""
        try:
            return list(idx[:n])
        except Exception:
            return list(idx.to_list()[:n])
