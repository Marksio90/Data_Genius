# === OPIS MODUŁU ===
"""
DataGenius PRO - Outlier Detector (PRO+++)
Wykrywa outliery metodami IQR, Z-score i Isolation Forest (opcjonalnie),
z defensywnymi guardami i zwięzłym, spójnym kontraktem danych.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import IsolationForest

from core.base_agent import BaseAgent, AgentResult


# === KONFIG / PROGI ===
@dataclass(frozen=True)
class OutlierConfig:
    """Progi i ustawienia wykrywania outlierów."""
    iqr_factor: float = 1.5
    zscore_threshold: float = 3.0
    min_rows_for_if: int = 100
    if_random_state: int = 42
    if_n_jobs: int = -1
    max_indices_return: int = 25
    # dolny/górny limit automatycznego contamination dla IF
    if_contam_min: float = 0.01
    if_contam_max: float = 0.20


# === KLASA GŁÓWNA AGENDA ===
class OutlierDetector(BaseAgent):
    """
    Detects outliers using IQR, Z-score, and Isolation Forest methods.
    """

    def __init__(self, config: Optional[OutlierConfig] = None) -> None:
        super().__init__(
            name="OutlierDetector",
            description="Detects outliers in numeric features"
        )
        self.config = config or OutlierConfig()

    # === WYKONANIE GŁÓWNE ===
    def execute(self, data: pd.DataFrame, **kwargs: Any) -> AgentResult:
        """
        Detect outliers.

        Args:
            data: Input DataFrame

        Returns:
            AgentResult with outlier analysis
        """
        result = AgentResult(agent_name=self.name)

        try:
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                msg = "OutlierDetector: 'data' must be a non-empty pandas DataFrame"
                result.add_error(msg)
                logger.error(msg)
                return result

            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                result.add_warning("No numeric columns for outlier detection")
                result.data = {
                    "iqr_method": {"method": "IQR", "columns": {}, "n_columns_with_outliers": 0},
                    "zscore_method": {"method": "Z-Score", "columns": {}, "n_columns_with_outliers": 0},
                    "isolation_forest": None,
                    "summary": {"total_outliers": 0, "n_columns_with_outliers": 0, "methods_used": ["IQR","Z-Score"], "most_outliers": None},
                    "recommendations": ["Brak kolumn numerycznych — etap wykrywania outlierów pominięty."]
                }
                return result

            df_num = data[numeric_cols].copy()

            # 1) IQR
            iqr_outliers = self._detect_iqr_outliers(df_num)

            # 2) Z-score
            zscore_outliers = self._detect_zscore_outliers(df_num, threshold=self.config.zscore_threshold)

            # 3) Isolation Forest (jeśli wystarczająco danych)
            isolation_outliers: Optional[Dict[str, Any]] = None
            if len(df_num) >= self.config.min_rows_for_if:
                # spróbuj oszacować contamination na podstawie IQR
                estimated_contam = self._estimate_contamination_from_iqr(iqr_outliers, n_rows=len(df_num))
                isolation_outliers = self._detect_isolation_forest_outliers(
                    df_num, contamination=estimated_contam
                )

            # 4) Summary (unia indeksów)
            summary = self._create_summary(iqr_outliers, zscore_outliers, isolation_outliers)

            # 5) Rekomendacje
            recommendations = self._get_recommendations(iqr_outliers, zscore_outliers, isolation_outliers, summary)

            result.data = {
                "iqr_method": iqr_outliers,
                "zscore_method": zscore_outliers,
                "isolation_forest": isolation_outliers,
                "summary": summary,
                "recommendations": recommendations,
            }

            logger.success(f"Outlier detection complete: {summary['total_outliers']} outliers found")

        except Exception as e:
            result.add_error(f"Outlier detection failed: {e}")
            logger.exception(f"Outlier detection error: {e}")

        return result

    # === IQR ===
    def _detect_iqr_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method (z guardem IQR=0)."""
        cfg = self.config
        outliers_by_column: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s_nona = s.dropna()
            if s_nona.empty:
                continue

            q1, q3 = s_nona.quantile(0.25), s_nona.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                # brak rozrzutu — nie flagujemy
                continue

            lower = q1 - cfg.iqr_factor * iqr
            upper = q3 + cfg.iqr_factor * iqr

            mask = (s < lower) | (s > upper)
            n_out = int(mask.sum())
            if n_out > 0:
                outliers_by_column[col] = {
                    "n_outliers": n_out,
                    "percentage": float(n_out / len(s) * 100.0),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "outlier_indices": self._head_indices(df[mask].index, cfg.max_indices_return),
                }

        return {
            "method": f"IQR (factor={cfg.iqr_factor})",
            "description": f"Outliers: < Q1-{cfg.iqr_factor}*IQR or > Q3+{cfg.iqr_factor}*IQR",
            "columns": outliers_by_column,
            "n_columns_with_outliers": len(outliers_by_column),
        }

    # === Z-SCORE ===
    def _detect_zscore_outliers(self, df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Detect outliers using Z-score (guard std=0, NA-safe)."""
        outliers_by_column: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue

            std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            if std == 0.0:
                # brak wariancji — nie flagujemy Z-score
                continue

            # scipy stats.zscore zwraca NaN dla wartości NaN — pracujemy na s (bez NaN)
            z = np.abs(stats.zscore(s, nan_policy="omit"))
            mask = z > threshold
            n_out = int(mask.sum())
            if n_out > 0:
                outliers_by_column[col] = {
                    "n_outliers": n_out,
                    "percentage": float(n_out / len(s) * 100.0),
                    "threshold": float(threshold),
                }

        return {
            "method": "Z-Score",
            "description": f"Outliers are values with |z-score| > {threshold}",
            "columns": outliers_by_column,
            "n_columns_with_outliers": len(outliers_by_column),
        }

    # === ISOLATION FOREST ===
    def _detect_isolation_forest_outliers(self, df: pd.DataFrame, contamination: float) -> Optional[Dict[str, Any]]:
        """Detect outliers using Isolation Forest (imputacja medianą, NA-safe)."""
        try:
            df_filled = df.copy()
            medians = df_filled.median(numeric_only=True)
            df_filled = df_filled.fillna(medians)

            iso = IsolationForest(
                contamination=float(contamination),
                random_state=self.config.if_random_state,
                n_jobs=self.config.if_n_jobs
            )
            pred = iso.fit_predict(df_filled)  # -1 → outlier
            mask = (pred == -1)
            n_out = int(mask.sum())
            return {
                "method": "Isolation Forest",
                "description": "ML-based anomaly detection (unsupervised)",
                "n_outliers": n_out,
                "percentage": float(n_out / len(df_filled) * 100.0),
                "outlier_indices": self._head_indices(df_filled[mask].index, self.config.max_indices_return),
                "contamination": float(contamination),
            }
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")
            return None

    # === PODSUMOWANIE ===
    def _create_summary(
        self,
        iqr_outliers: Dict[str, Any],
        zscore_outliers: Dict[str, Any],
        isolation_outliers: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary of outlier detection (unia kolumn/indeksów)."""
        methods_used = ["IQR", "Z-Score"] + (["Isolation Forest"] if isolation_outliers else [])

        # Zbierz kolumny z outlierami
        cols_with = set(iqr_outliers.get("columns", {}).keys()) | set(zscore_outliers.get("columns", {}).keys())
        if isolation_outliers and isolation_outliers.get("n_outliers", 0) > 0:
            # IF jest bezkolumnowy — traktujemy jako sygnał globalny (nie dodaje kolumn)
            pass

        # Zbierz unikalne indeksy outlierów z metod kolumnowych + globalnych
        idx_union: Set[Any] = set()
        for col, meta in iqr_outliers.get("columns", {}).items():
            # uwaga: tu są indeksy tylko pierwszych N do podglądu — do unii policzymy przybliżenie:
            # lepsze oszacowanie: policz pełny n_outliers sumarycznie (już mamy w summary),
            # a unia indeksów do prezentacji jest orientacyjna (top-N z każdej kolumny)
            idx_union.update(meta.get("outlier_indices", []))
        for col, meta in zscore_outliers.get("columns", {}).items():
            # Z-score nie trzyma indeksów — pomijamy w unii
            pass
        if isolation_outliers and isolation_outliers.get("outlier_indices"):
            idx_union.update(isolation_outliers["outlier_indices"])

        # total_outliers — liczmy konserwatywnie: suma IQR per kolumna (tak jak wcześniej),
        # ale nie większa niż liczba wierszy*liczba_kolumn (guard).
        total_by_iqr = int(sum(v.get("n_outliers", 0) for v in iqr_outliers.get("columns", {}).values()))
        total_outliers = max(total_by_iqr, int(isolation_outliers.get("n_outliers", 0)) if isolation_outliers else 0)

        return {
            "total_outliers": int(total_outliers),
            "n_columns_with_outliers": int(len(cols_with)),
            "methods_used": methods_used,
            "most_outliers": self._get_most_outliers_column(iqr_outliers),
            "example_outlier_indices": list(idx_union)[: self.config.max_indices_return] or [],
        }

    def _get_most_outliers_column(self, iqr_outliers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get column with most outliers (na podstawie IQR)."""
        cols = iqr_outliers.get("columns", {})
        if not cols:
            return None
        max_col, meta = max(cols.items(), key=lambda kv: kv[1].get("n_outliers", 0))
        return {"column": str(max_col), "n_outliers": int(meta.get("n_outliers", 0))}

    # === ESTYMACJA CONTAMINATION DLA IF ===
    def _estimate_contamination_from_iqr(self, iqr_outliers: Dict[str, Any], n_rows: int) -> float:
        """
        Szacuje udział anomalii na podstawie IQR:
        - bierzemy średni odsetek outlierów w kolumnach, clamp do [min, max].
        """
        cfg = self.config
        cols = iqr_outliers.get("columns", {})
        if not cols or n_rows <= 0:
            return max(cfg.if_contam_min, 0.01)

        # średni udział procentowy outlierów per kolumna
        pcts = [float(v.get("percentage", 0.0)) for v in cols.values()]
        mean_pct = np.mean(pcts) if pcts else 0.0
        est = float(mean_pct / 100.0)
        est = float(np.clip(est, cfg.if_contam_min, cfg.if_contam_max))
        return est

    # === REKOMENDACJE ===
    def _get_recommendations(
        self,
        iqr_outliers: Dict[str, Any],
        zscore_outliers: Dict[str, Any],
        isolation_outliers: Optional[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> List[str]:
        """Buduje listę rekomendacji na podstawie wyników."""
        rec: List[str] = []
        n_cols = summary.get("n_columns_with_outliers", 0)
        total = summary.get("total_outliers", 0)

        if total == 0:
            rec.append("✅ Brak wykrytych outlierów — możesz pominąć etap ich obsługi.")
            return rec

        if n_cols > 0:
            rec.append(f"🔎 Wykryto outliery w {n_cols} kolumnach numerycznych — rozważ czyszczenie przed treningiem.")

        # Szczegóły IQR (potencjalne kapowanie/winsoryzacja)
        if iqr_outliers.get("columns"):
            top = self._get_most_outliers_column(iqr_outliers)
            if top:
                rec.append(f"📦 Najwięcej outlierów w kolumnie '{top['column']}' — rozważ winsoryzację lub transformację.")

        # Z-score — sugeruj standaryzację
        if zscore_outliers.get("n_columns_with_outliers", 0) > 0:
            rec.append("📏 Z-score wskazuje odstępstwa — rozważ standaryzację/robust scaling.")

        # IF — sugeruj modelowe podejście
        if isolation_outliers and isolation_outliers.get("n_outliers", 0) > 0:
            rec.append("🤖 Isolation Forest wykrył anomalia — rozważ filtrowanie lub oznaczanie anomalii jako cechy.")

        # Ogólne
        rec.append("🧪 Waliduj wpływ usuwania/kapowania outlierów na metryki (rób to wyłącznie na train).")
        return list(dict.fromkeys(rec))  # dedup

    # === POMOCNICZE ===
    @staticmethod
    def _head_indices(idx: pd.Index, n: int) -> List[Any]:
        """Zwraca pierwsze n indeksów jako listę (do podglądu)."""
        try:
            return list(idx[:n])
        except Exception:
            return list(idx.to_list()[:n])
