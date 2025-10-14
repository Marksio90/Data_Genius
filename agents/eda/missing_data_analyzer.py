# === OPIS MODUŁU ===
"""
DataGenius PRO - Missing Data Analyzer (PRO+++)
Analiza wzorców braków danych: poziom zbioru, kolumn i wierszy; korelacje masek braków;
klasyfikacja severity i rekomendacje strategii wypełniania/obsługi.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult


# === KONFIG / PROGI ===
@dataclass(frozen=True)
class MissingConfig:
    """Progi i ustawienia analizy braków."""
    low_threshold_pct: float = 5.0       # <5% => low
    medium_threshold_pct: float = 20.0   # <20% => medium
    high_threshold_pct: float = 50.0     # <50% => high, inaczej critical
    drop_column_over_pct: float = 70.0   # >70% => rekomendacja drop
    row_missing_block_threshold: int = 3 # wiersz z >= tyloma NaN => potencjalny „blok” braków
    corr_mask_threshold: float = 0.70    # |corr| masek braków > 0.7 => korelowane braki
    top_rows_limit: int = 10             # ile wierszy z największą liczbą NaN zwracać


# === KLASA GŁÓWNA AGENDA ===
class MissingDataAnalyzer(BaseAgent):
    """
    Analyzes missing data patterns and suggests handling strategies.
    """

    def __init__(self, config: Optional[MissingConfig] = None) -> None:
        super().__init__(
            name="MissingDataAnalyzer",
            description="Analyzes missing data patterns"
        )
        self.config = config or MissingConfig()

    # === WYKONANIE GŁÓWNE ===
    def execute(self, data: pd.DataFrame, **kwargs: Any) -> AgentResult:
        """
        Analyze missing data.

        Args:
            data: Input DataFrame

        Returns:
            AgentResult with missing data analysis
        """
        result = AgentResult(agent_name=self.name)

        try:
            if data is None or not isinstance(data, pd.DataFrame):
                msg = "MissingDataAnalyzer: 'data' must be a non-empty pandas DataFrame"
                result.add_error(msg)
                logger.error(msg)
                return result
            if data.empty:
                result.add_warning("Empty DataFrame — no missing-data analysis performed.")
                result.data = {
                    "summary": {"total_cells": 0, "total_missing": 0, "missing_percentage": 0.0,
                                "n_columns_with_missing": 0, "n_rows_with_missing": 0, "complete_rows": 0},
                    "columns": [],
                    "rows": {"top_rows_with_many_missing": [], "max_missing_in_row": 0},
                    "patterns": {"correlated": [], "blocks": {"rows_ge_threshold": 0}},
                    "recommendations": ["Dostarcz dane do analizy braków."]
                }
                return result

            # 1) Podsumowanie zbioru
            summary = self._get_missing_summary(data)

            # 2) Kolumny
            column_analysis = self._analyze_missing_by_column(data)

            # 3) Wiersze (bloki braków)
            rows_info = self._analyze_missing_by_row(data)

            # 4) Wzorce (korelacje masek braków + bloki)
            patterns = self._identify_patterns(data, rows_info)

            # 5) Rekomendacje
            recommendations = self._get_recommendations(column_analysis, rows_info, summary)

            result.data = {
                "summary": summary,
                "columns": column_analysis,
                "rows": rows_info,
                "patterns": patterns,
                "recommendations": recommendations,
            }

            if summary["total_missing"] == 0:
                logger.success("No missing data found!")
            else:
                logger.info(
                    "Missing data analysis complete: %d missing values (%.2f%%)",
                    summary["total_missing"], summary["missing_percentage"]
                )

        except Exception as e:
            result.add_error(f"Missing data analysis failed: {e}")
            logger.exception(f"Missing data analysis error: {e}")

        return result

    # === PODSUMOWANIE ZBIORU ===
    def _get_missing_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall missing data summary (odporne na dzielenie przez zero)."""
        total_cells = int(df.shape[0] * df.shape[1])
        total_missing = int(df.isna().sum().sum())
        missing_percentage = float((total_missing / total_cells) * 100) if total_cells > 0 else 0.0
        n_cols_with_missing = int((df.isna().sum() > 0).sum())
        row_missing_mask = df.isna().any(axis=1)
        n_rows_with_missing = int(row_missing_mask.sum())
        complete_rows = int((~row_missing_mask).sum())

        return {
            "total_cells": total_cells,
            "total_missing": total_missing,
            "missing_percentage": missing_percentage,
            "n_columns_with_missing": n_cols_with_missing,
            "n_rows_with_missing": n_rows_with_missing,
            "complete_rows": complete_rows,
        }

    # === ANALIZA KOLUMN ===
    def _analyze_missing_by_column(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze missing data for each column z klasyfikacją severity i strategią."""
        cfg = self.config
        out: List[Dict[str, Any]] = []
        n = max(1, len(df))

        isna = df.isna()
        n_missing_per_col = isna.sum(axis=0)

        for col in df.columns:
            n_missing = int(n_missing_per_col[col])
            if n_missing == 0:
                continue

            missing_pct = float((n_missing / n) * 100)
            dtype_str = str(df[col].dtype)
            severity = self._get_severity(missing_pct, cfg)
            strategy = self._suggest_strategy(df[col], missing_pct, cfg)

            out.append({
                "column": str(col),
                "n_missing": n_missing,
                "missing_percentage": missing_pct,
                "dtype": dtype_str,
                "severity": severity,
                "suggested_strategy": strategy,
            })

        out.sort(key=lambda x: x["missing_percentage"], reverse=True)
        return out

    # === ANALIZA WIERSZY (BLOKI BRAKÓW) ===
    def _analyze_missing_by_row(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Zwraca top wiersze z największą liczbą NaN + max NaN w jednym wierszu."""
        cfg = self.config
        isna = df.isna()
        n_missing_per_row = isna.sum(axis=1).astype(int)

        if n_missing_per_row.max() == 0:
            return {"top_rows_with_many_missing": [], "max_missing_in_row": 0}

        # Top wiersze
        top_idx = n_missing_per_row.sort_values(ascending=False).head(cfg.top_rows_limit).index
        top_rows: List[Dict[str, Any]] = []
        for idx in top_idx:
            missing_cols = df.columns[isna.loc[idx]].tolist()
            top_rows.append({
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "n_missing": int(n_missing_per_row.loc[idx]),
                "missing_cols": [str(c) for c in missing_cols],
            })

        return {
            "top_rows_with_many_missing": top_rows,
            "max_missing_in_row": int(n_missing_per_row.max()),
        }

    # === WZORCE BRAKÓW ===
    def _identify_patterns(self, df: pd.DataFrame, rows_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identyfikuje:
        - "correlated": kolumny, których maski braków są silnie skorelowane (>|thr|),
        - "blocks": liczba wierszy z >= row_missing_block_threshold brakami.
        """
        cfg = self.config
        patterns = {
            "correlated": [],
            "blocks": {
                "rows_ge_threshold": 0,
                "threshold": int(cfg.row_missing_block_threshold),
            },
        }

        # Korelowane maski braków (kolumny)
        isna = df.isna()
        cols_with_missing = [c for c in df.columns if isna[c].any()]
        for i, c1 in enumerate(cols_with_missing):
            s1 = isna[c1].astype(float)
            for c2 in cols_with_missing[i + 1:]:
                s2 = isna[c2].astype(float)
                # Pearsona na maskach (0/1); guard na brak wariancji
                if s1.std(ddof=0) == 0 or s2.std(ddof=0) == 0:
                    continue
                corr = float(s1.corr(s2))
                if np.isfinite(corr) and abs(corr) >= cfg.corr_mask_threshold:
                    patterns["correlated"].append({
                        "column": str(c1),
                        "correlated_with": str(c2),
                        "mask_corr": corr,
                    })

        # Bloki (wiersze z wieloma NaN)
        isna_rows = isna.sum(axis=1)
        patterns["blocks"]["rows_ge_threshold"] = int((isna_rows >= cfg.row_missing_block_threshold).sum())

        return patterns

    # === SEVERITY I STRATEGIE ===
    def _get_severity(self, missing_pct: float, cfg: MissingConfig) -> str:
        """Określa severity na podstawie progów z konfiguracji."""
        if missing_pct < cfg.low_threshold_pct:
            return "low"
        elif missing_pct < cfg.medium_threshold_pct:
            return "medium"
        elif missing_pct < cfg.high_threshold_pct:
            return "high"
        else:
            return "critical"

    def _suggest_strategy(self, series: pd.Series, missing_pct: float, cfg: MissingConfig) -> str:
        """Proponuje strategię w zależności od typu i skali braków."""
        if missing_pct >= cfg.drop_column_over_pct:
            return "consider_dropping_column"

        if pd.api.types.is_numeric_dtype(series):
            return "mean_or_median_imputation" if missing_pct < cfg.low_threshold_pct else "forward_fill_or_interpolation"

        if series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
            return "mode_imputation" if missing_pct < (cfg.low_threshold_pct + 5.0) else "create_missing_indicator"

        # Datetime/other
        return "forward_fill"

    # === REKOMENDACJE ===
    def _get_recommendations(
        self,
        column_analysis: List[Dict[str, Any]],
        rows_info: Dict[str, Any],
        summary: Dict[str, Any]
    ) -> List[str]:
        """Get actionable recommendations based on analysis."""
        rec: List[str] = []

        if summary.get("total_missing", 0) == 0:
            rec.append("✅ Brak brakujących danych — możesz pominąć etap imputacji.")
            return rec

        # Critical / High
        critical = [c for c in column_analysis if c["severity"] == "critical"]
        high = [c for c in column_analysis if c["severity"] == "high"]
        if critical:
            rec.append(f"🚨 {len(critical)} kolumn ma ≥50% braków — rozważ usunięcie lub modelowanie specjalistyczne (np. modele z missing indicator).")
        if high:
            rec.append(f"⚠️ {len(high)} kolumn ma 20–50% braków — użyj zaawansowanej imputacji (KNN/MICE), rozważ 'missingness as feature'.")

        # Numeryczne vs kategoryczne
        numeric_missing = [c for c in column_analysis if ("int" in c["dtype"] or "float" in c["dtype"])]
        categorical_missing = [c for c in column_analysis if c["dtype"] == "object"]
        if numeric_missing:
            rec.append(f"📊 {len(numeric_missing)} kolumn numerycznych z brakami — rozważ medianę/KNN lub interpolację czasową (jeśli sekwencyjne).")
        if categorical_missing:
            rec.append(f"📝 {len(categorical_missing)} kolumn kategorycznych z brakami — rozważ imputację modą albo kategorię 'brak' + indicator.")

        # Bloki w wierszach
        rows_ge_thr = int(rows_info.get("max_missing_in_row", 0) >= self.config.row_missing_block_threshold)
        if rows_ge_thr:
            rec.append(f"🧱 Występują wiersze z ≥{self.config.row_missing_block_threshold} brakami — sprawdź reguły akwizycji/łączenia danych.")

        # Korelowane maski braków
        # (jeśli wiele par → sugeruj wspólną przyczynę/ETL)
        rec.append("🔎 Sprawdź kolumny z silnie skorelowanymi brakami — mogą pochodzić z tego samego źródła ETL lub etapu transformacji.")

        # Dedup
        rec = list(dict.fromkeys(rec))
        return rec
