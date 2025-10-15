# === OPIS MODUŁU ===
"""
DataGenius PRO++++ - Missing Data Analyzer (KOSMOS)
Analiza wzorców braków danych: poziom zbioru, kolumn i wierszy; korelacje masek braków;
sygnały MAR/MNAR; rekomendacje strategii; telemetria i miękkie budżety czasu.

Kontrakt (AgentResult.data):
{
  "summary": {
      "total_cells": int, "total_missing": int, "missing_percentage": float,
      "n_columns_with_missing": int, "n_rows_with_missing": int, "complete_rows": int
  },
  "columns": [
      {"column": str, "n_missing": int, "missing_percentage": float, "dtype": str,
       "severity": "low"|"medium"|"high"|"critical",
       "suggested_strategy": str,
       "is_indicator_like": bool, "n_unique_non_na": int,
       "top_co_missing_with": [{"column": str, "mask_corr": float}]  # <= nowość, max 3
      }
  ],
  "rows": {
      "top_rows_with_many_missing": [{"row_index": int|str, "n_missing": int, "missing_cols": List[str]}],
      "max_missing_in_row": int
  },
  "patterns": {
      "correlated": [{"column": str, "correlated_with": str, "mask_corr": float}],
      "blocks": {"rows_ge_threshold": int, "threshold": int},
      "mar_mnar_signals": [{"column": str, "signal": "MAR_like"|"MNAR_like", "reason": str}]
  },
  "recommendations": List[str],
  "telemetry": {
      "elapsed_ms": float, "mask_cols_analyzed": int, "sampled": bool,
      "sample_info": {"from_rows": int, "to_rows": int}|None,
      "soft_time_budget_ms": int, "soft_time_exceeded": bool
  },
  "version": "4.0-kosmos",
  "extra": {...}  # nieobowiązkowe drobne metadane
}
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult


# === KONFIG / PROGI ===
@dataclass(frozen=True)
class MissingConfig:
    """Progi i ustawienia analizy braków."""
    # klasyfikacja severity
    low_threshold_pct: float = 5.0        # <5% => low
    medium_threshold_pct: float = 20.0    # <20% => medium
    high_threshold_pct: float = 50.0      # <50% => high, inaczej critical
    drop_column_over_pct: float = 70.0    # >70% => rekomendacja drop
    # wzorce wierszowe
    row_missing_block_threshold: int = 3
    # korelacje masek
    corr_mask_threshold: float = 0.70     # |corr| masek braków > 0.7 => korelowane
    max_corr_cols_for_masks: int = 400    # miękki limit kolumn do korelacji masek (O(k^2))
    # top listy
    top_rows_limit: int = 10              # ile wierszy z największą liczbą NaN zwracać
    top_co_missing_with: int = 3          # ile „najbardziej współ-brakujących” kolumn dołączyć per kolumna
    # sampling tabeli
    enable_sampling: bool = True
    sample_rows: int = 500_000
    random_state: int = 42
    # traktowanie pustych stringów jako braków (dla kolumn object/category)
    treat_empty_string_as_na: bool = True
    strip_whitespace_in_object: bool = True
    # miękki budżet czasu (informacyjny)
    soft_time_budget_ms: int = 45_000


class MissingDataAnalyzer(BaseAgent):
    """
    Analyzes missing data patterns and suggests handling strategies.
    PRO++++: szybka, pamięcio-odporna, stabilny kontrakt i bogata telemetria.
    """

    def __init__(self, config: Optional[MissingConfig] = None) -> None:
        super().__init__(name="MissingDataAnalyzer", description="Analyzes missing data patterns")
        self.config = config or MissingConfig()
        self._log = logger.bind(agent="MissingDataAnalyzer")

    # === WYKONANIE GŁÓWNE ===
    def execute(self, data: pd.DataFrame, **kwargs: Any) -> AgentResult:
        """
        Analyze missing data.
        """
        result = AgentResult(agent_name=self.name)
        t0 = time.perf_counter()

        try:
            if data is None or not isinstance(data, pd.DataFrame):
                msg = "MissingDataAnalyzer: 'data' must be a non-empty pandas DataFrame"
                result.add_error(msg)
                self._log.error(msg)
                return result
            if data.empty:
                result.add_warning("Empty DataFrame — no missing-data analysis performed.")
                payload = self._empty_payload()
                payload["telemetry"]["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                result.data = payload
                return result

            cfg = self.config

            # Pre-clean (lekko): opcjonalnie potraktuj puste stringi jako NaN
            df = data.copy()
            if cfg.strip_whitespace_in_object:
                obj_cols = df.select_dtypes(include=["object"]).columns
                if len(obj_cols) > 0:
                    try:
                        df[obj_cols] = df[obj_cols].apply(lambda s: s.astype("object").str.strip())
                    except Exception:
                        pass
            if cfg.treat_empty_string_as_na:
                obj_cols = df.select_dtypes(include=["object"]).columns
                if len(obj_cols) > 0:
                    try:
                        df[obj_cols] = df[obj_cols].replace("", np.nan)
                    except Exception:
                        pass

            # Sampling (dla bardzo dużych tabel)
            sampled = False
            sample_info: Optional[Dict[str, int]] = None
            if cfg.enable_sampling and len(df) > cfg.sample_rows:
                sampled = True
                sample_info = {"from_rows": int(len(df)), "to_rows": int(cfg.sample_rows)}
                self._log.info(f"Sampling for missing analysis: {len(df)} → {cfg.sample_rows}")
                df = df.sample(n=cfg.sample_rows, random_state=cfg.random_state)

            # 1) Podsumowanie zbioru
            summary = self._get_missing_summary(df)

            # 2) Kolumny (severity + strategia + flagi indicator-like)
            column_analysis = self._analyze_missing_by_column(df)

            # 3) Wiersze (bloki braków)
            rows_info = self._analyze_missing_by_row(df)

            # 4) Wzorce (korelacje masek braków + bloki + MAR/MNAR sygnały) – wektorowo i z limitem czasu
            patterns, mask_cols_analyzed = self._identify_patterns_with_time_guard(
                df=df,
                rows_info=rows_info,
                column_analysis=column_analysis,
                started_at=t0
            )

            # 5) Rekomendacje
            recommendations = self._get_recommendations(column_analysis, rows_info, summary, patterns)

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            result.data = {
                "summary": summary,
                "columns": column_analysis,
                "rows": rows_info,
                "patterns": patterns,
                "recommendations": recommendations,
                "telemetry": {
                    "elapsed_ms": elapsed_ms,
                    "mask_cols_analyzed": int(mask_cols_analyzed),
                    "sampled": bool(sampled),
                    "sample_info": sample_info,
                    "soft_time_budget_ms": cfg.soft_time_budget_ms,
                    "soft_time_exceeded": bool(elapsed_ms > cfg.soft_time_budget_ms),
                },
                "version": "4.0-kosmos",
                "extra": {
                    "object_empty_as_na": cfg.treat_empty_string_as_na,
                    "strip_whitespace_in_object": cfg.strip_whitespace_in_object
                }
            }

            if summary["total_missing"] == 0:
                self._log.success("No missing data found!")
            else:
                self._log.info(
                    f"Missing data analysis complete: {summary['total_missing']} missing values "
                    f"({summary['missing_percentage']:.2f}%)"
                )

        except Exception as e:
            result.add_error(f"Missing data analysis failed: {e}")
            self._log.exception(f"Missing data analysis error: {e}")

        return result

    # === PODSUMOWANIE ZBIORU ===
    def _get_missing_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall missing data summary (odporne na dzielenie przez zero)."""
        rows, cols = int(df.shape[0]), int(df.shape[1])
        total_cells = rows * cols
        if total_cells == 0:
            return {
                "total_cells": 0,
                "total_missing": 0,
                "missing_percentage": 0.0,
                "n_columns_with_missing": 0,
                "n_rows_with_missing": 0,
                "complete_rows": rows,
            }

        isna = df.isna()
        total_missing = int(isna.values.sum())
        missing_percentage = float((total_missing / total_cells) * 100.0)
        n_cols_with_missing = int((isna.sum(axis=0) > 0).sum())
        row_missing_mask = isna.any(axis=1)
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
        """Analyze per-column missingness + severity/strategy + indicator-like + co-missing."""
        cfg = self.config
        out: List[Dict[str, Any]] = []
        n = max(1, len(df))

        isna = df.isna()
        n_missing_per_col = isna.sum(axis=0)

        # wstępny cache do „co-missing with” (wybierz tylko kolumny z brakami)
        cols_with_missing = [c for c in df.columns if n_missing_per_col[c] > 0]

        # (opcjonalnie) przygotuj macierz masek w formie float dla szybkiej korelacji — lokalnie wypełnimy potem
        mask_matrix: Optional[pd.DataFrame] = None
        if len(cols_with_missing) > 1:
            try:
                sel = cols_with_missing[: min(len(cols_with_missing), self.config.max_corr_cols_for_masks)]
                mask_matrix = isna[sel].astype(float)
            except Exception:
                mask_matrix = None

        for col in df.columns:
            n_missing = int(n_missing_per_col[col])
            if n_missing == 0:
                # opcjonalnie można zwracać też kolumny bez braków — tu zostajemy przy zwięzłości
                continue

            s = df[col]
            missing_pct = float((n_missing / n) * 100)
            dtype_str = str(s.dtype)
            severity = self._get_severity(missing_pct, cfg)
            strategy = self._suggest_strategy(s, missing_pct, cfg)

            # flaga: wskaźnikowa/niemal wskaźnikowa (dużo NaN + 1–2 wartości nie-NaN)
            non_na_unique = int(s.dropna().nunique())
            is_indicator_like = bool(non_na_unique <= 2 and missing_pct >= cfg.medium_threshold_pct)

            # top co-missing with (max k) — szybki ranking korelacji masek
            top_co: List[Dict[str, Any]] = []
            if (mask_matrix is not None) and (col in mask_matrix.columns):
                try:
                    v = mask_matrix[col]
                    if v.std(ddof=0) > 0:
                        corr = mask_matrix.corrwith(v)  # wektorowa korelacja
                        corr = corr.drop(index=col, errors="ignore").dropna().abs().sort_values(ascending=False)
                        # wybierz najwyższe > próg
                        top_pairs = corr[corr >= cfg.corr_mask_threshold].head(cfg.top_co_missing_with)
                        top_co = [{"column": str(k), "mask_corr": float(val)} for k, val in top_pairs.items()]
                except Exception:
                    top_co = []

            out.append({
                "column": str(col),
                "n_missing": n_missing,
                "missing_percentage": missing_pct,
                "dtype": dtype_str,
                "severity": severity,
                "suggested_strategy": strategy,
                "is_indicator_like": is_indicator_like,
                "n_unique_non_na": non_na_unique,
                "top_co_missing_with": top_co
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
            # uważaj na typ indeksu
            idx_out = int(idx) if isinstance(idx, (int, np.integer)) else str(idx)
            missing_cols = df.columns[isna.loc[idx]].tolist()
            top_rows.append({
                "row_index": idx_out,
                "n_missing": int(n_missing_per_row.loc[idx]),
                "missing_cols": [str(c) for c in missing_cols],
            })

        return {
            "top_rows_with_many_missing": top_rows,
            "max_missing_in_row": int(n_missing_per_row.max()),
        }

    # === WZORCE BRAKÓW z kontrolą czasu ===
    def _identify_patterns_with_time_guard(
        self,
        df: pd.DataFrame,
        rows_info: Dict[str, Any],
        column_analysis: List[Dict[str, Any]],
        started_at: float
    ) -> Tuple[Dict[str, Any], int]:
        """
        Wariant _identify_patterns z miękkim budżetem na cały moduł.
        Jeżeli zbliżamy się do limitu — ograniczamy liczbę kolumn do korelacji.
        """
        cfg = self.config

        # bazowe
        patterns = {
            "correlated": [],
            "blocks": {
                "rows_ge_threshold": 0,
                "threshold": int(cfg.row_missing_block_threshold),
            },
            "mar_mnar_signals": []
        }
        # bloki wierszowe
        isna_rows = df.isna().sum(axis=1)
        patterns["blocks"]["rows_ge_threshold"] = int((isna_rows >= cfg.row_missing_block_threshold).sum())

        # korelacje masek – dobór liczby kolumn zależnie od czasu
        cols_with_missing = [c["column"] for c in column_analysis]  # już posortowane po % braków
        k_max = min(len(cols_with_missing), cfg.max_corr_cols_for_masks)
        # jeśli mało czasu, zmniejsz k
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if elapsed_ms > cfg.soft_time_budget_ms * 0.7:
            k_max = max(2, int(k_max * 0.6))
        if elapsed_ms > cfg.soft_time_budget_ms * 0.9:
            k_max = max(2, int(k_max * 0.4))

        mask_cols = cols_with_missing[:k_max]
        mask_cols_analyzed = len(mask_cols)

        if mask_cols_analyzed >= 2:
            try:
                # wektorowy wariant: liczymy macierz korelacji 0/1 → float
                mask = df[mask_cols].isna().astype(float)
                # pomijamy kolumny bez wariancji (std == 0)
                std = mask.std(axis=0, ddof=0)
                keep = std[std > 0].index
                mask = mask[keep]
                mask_cols_analyzed = int(len(keep))
                if mask_cols_analyzed >= 2:
                    corr = mask.corr()
                    # wyciągnij pary z dolnego trójkąta i filtrem progu
                    cols = list(corr.columns)
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            r = corr.iloc[i, j]
                            if np.isfinite(r) and abs(r) >= cfg.corr_mask_threshold:
                                patterns["correlated"].append({
                                    "column": str(cols[i]),
                                    "correlated_with": str(cols[j]),
                                    "mask_corr": float(r),
                                })
            except Exception as e:
                self._log.warning(f"Mask correlation failed: {e}")

        # MAR/MNAR sygnały
        indicator_like_cols = {c["column"] for c in column_analysis if c.get("is_indicator_like")}
        col2pct = {c["column"]: c["missing_percentage"] for c in column_analysis}
        for c in indicator_like_cols:
            if col2pct.get(c, 0.0) >= self.config.medium_threshold_pct:
                patterns["mar_mnar_signals"].append({
                    "column": c, "signal": "MNAR_like",
                    "reason": "wysoki % braków i wskaźnikowa natura kolumny"
                })
        if patterns["correlated"]:
            involved = set()
            for p in patterns["correlated"]:
                involved.add(p["column"]); involved.add(p["correlated_with"])
            for c in sorted(involved):
                patterns["mar_mnar_signals"].append({
                    "column": c, "signal": "MAR_like",
                    "reason": "maska braków silnie skorelowana z inną kolumną"
                })

        return patterns, mask_cols_analyzed

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

        # Datetime → forward/backward fill jako domyślny kierunek
        if pd.api.types.is_datetime64_any_dtype(series):
            return "time_aware_ffill_bfill"

        if pd.api.types.is_numeric_dtype(series):
            # małe braki: statyczne; większe: KNN/MICE/interpolacja
            return "mean_or_median_imputation" if missing_pct < cfg.low_threshold_pct else "knn_mice_or_interpolation"

        if series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
            return "mode_or_missing_category" if missing_pct < (cfg.low_threshold_pct + 5.0) else "missing_indicator_or_category"

        # fallback
        return "forward_fill"

    # === REKOMENDACJE ===
    def _get_recommendations(
        self,
        column_analysis: List[Dict[str, Any]],
        rows_info: Dict[str, Any],
        summary: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Get actionable recommendations based on analysis."""
        cfg = self.config
        rec: List[str] = []

        if summary.get("total_missing", 0) == 0:
            rec.append("✅ Brak brakujących danych — możesz pominąć etap imputacji.")
            return rec

        # Severity buckets
        critical = [c for c in column_analysis if c["severity"] == "critical"]
        high = [c for c in column_analysis if c["severity"] == "high"]
        if critical:
            rec.append(f"🚨 {len(critical)} kolumn ma ≥{cfg.high_threshold_pct}% braków — rozważ usunięcie lub modele tolerujące braki + missing indicator.")
        if high:
            rec.append(f"⚠️ {len(high)} kolumn ma {cfg.medium_threshold_pct}–{cfg.high_threshold_pct}% braków — użyj zaawansowanej imputacji (KNN/MICE), rozważ 'missingness as feature'.")

        # Typy cech
        numeric_missing = [c for c in column_analysis if ("int" in c["dtype"] or "float" in c["dtype"])]
        categorical_missing = [c for c in column_analysis if (c["dtype"] == "object" or "category" in c["dtype"])]
        datetime_missing = [c for c in column_analysis if "datetime" in c["dtype"]]
        if numeric_missing:
            rec.append(f"📊 {len(numeric_missing)} kolumn numerycznych z brakami — mediana/kwantyl, a dla szeregów: interpolacja.")
        if categorical_missing:
            rec.append(f"📝 {len(categorical_missing)} kolumn kategorycznych z brakami — moda lub kategoria 'brak' + indicator.")
        if datetime_missing:
            rec.append(f"⏱️ {len(datetime_missing)} kolumn czasowych z brakami — preferuj ffill/bfill w obrębie grup/czasu.")

        # Wiersze blokowe
        if int(rows_info.get("max_missing_in_row", 0)) >= cfg.row_missing_block_threshold:
            rec.append(f"🧱 Wiersze z ≥{cfg.row_missing_block_threshold} brakami — sprawdź łączenia źródeł (JOIN), filtry i poprawność kluczy.")

        # Korelowane maski braków
        if patterns.get("correlated"):
            rec.append("🔗 Silnie skorelowane maski braków — rozważ wspólną, wielowymiarową imputację (KNN/MICE) i/lub modelowanie brakowalności.")

        # MAR/MNAR sygnały
        mar_like = any(p.get("signal") == "MAR_like" for p in patterns.get("mar_mnar_signals", []))
        mnar_like = any(p.get("signal") == "MNAR_like" for p in patterns.get("mar_mnar_signals", []))
        if mar_like:
            rec.append("🧪 Wzorzec MAR-like — brak zależny od innych pól; preferuj imputację wielowymiarową i walidację poimputacyjną.")
        if mnar_like:
            rec.append("🕳️ Wzorzec MNAR-like — rozważ modelowanie samej „brakowalności” (indicator), analizę procesu zbierania i ewentualny drop.")

        # Dedup
        rec = list(dict.fromkeys(rec))
        return rec

    # === PAYLOAD DLA PUSTYCH DANYCH ===
    @staticmethod
    def _empty_payload() -> Dict[str, Any]:
        return {
            "summary": {
                "total_cells": 0, "total_missing": 0, "missing_percentage": 0.0,
                "n_columns_with_missing": 0, "n_rows_with_missing": 0, "complete_rows": 0
            },
            "columns": [],
            "rows": {"top_rows_with_many_missing": [], "max_missing_in_row": 0},
            "patterns": {"correlated": [], "blocks": {"rows_ge_threshold": 0, "threshold": 3}, "mar_mnar_signals": []},
            "recommendations": ["Dostarcz dane do analizy braków."],
            "telemetry": {
                "elapsed_ms": 0.0, "mask_cols_analyzed": 0, "sampled": False,
                "sample_info": None, "soft_time_budget_ms": 0, "soft_time_exceeded": False
            },
            "version": "4.0-kosmos",
            "extra": {}
        }
