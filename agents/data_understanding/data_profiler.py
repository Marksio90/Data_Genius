# === OPIS MODUŁU ===
"""
DataGenius PRO - Data Profiler (PRO+++)
Kompleksowe profilowanie danych i ocena jakości datasetu.

Wyjściowy kontrakt (dict):
{
    "quality_score": float,                  # 0..100
    "quality_details": Dict[str, Any],       # szczegóły z DataValidator
    "statistical_profile": Dict[str, Any],   # rozmiary, dtypes, pamięć, braki, duplikaty
    "quality_issues": List[Dict[str, Any]],  # wykryte problemy jakości
    "feature_characteristics": Dict[str, List[str]],  # kategorie cech
    "correlations": Dict[str, Any]           # macierz + lista wysokich korelacji
}
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Zależności domenowe (dostarczane w projekcie)
from core.base_agent import BaseAgent, AgentResult
from core.data_validator import DataValidator


# === KONFIG / STAŁE ===
@dataclass(frozen=True)
class ProfilerConfig:
    """Parametry działania profilera."""
    high_missing_col_threshold: float = 0.50   # >50% braków => issue high
    high_missing_flag_threshold: float = 0.30  # >30% => flaga w characteristics
    high_cardinality_ratio:   float = 0.90     # >90% unikalnych względem N
    outlier_iqr_factor:       float = 3.0      # IQR fence
    outlier_row_ratio_flag:   float = 0.05     # >5% outliers => issue low
    duplicates_row_ratio_mid: float = 0.10     # >10% => severity "medium"
    corr_abs_threshold:       float = 0.80     # wysokie korelacje
    max_corr_rows:            int   = 200_000  # sampling safety dla korelacji
    corr_nan_policy:          str   = "pairwise"  # "pairwise" | "drop" (kolumny z NA)


# === MODELE DANYCH (wynik – opcjonalna struktura wewnętrzna) ===
@dataclass
class StatisticalProfile:
    n_rows: int
    n_columns: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    memory_mb: float
    duplicates: Dict[str, float]
    missing_data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# === KLASA GŁÓWNA AGENDA ===
class DataProfiler(BaseAgent):
    """
    Agent profilujący dane (PRO+++):
    - Jakość danych (score + szczegóły) z DataValidator
    - Profil statystyczny
    - Problemy jakości (braki, stałość, kardynalność, duplikaty, outliery)
    - Charakterystyki cech
    - Korelacje numeryczne (bezpiecznie i wydajnie)
    """

    def __init__(self, config: Optional[ProfilerConfig] = None) -> None:
        super().__init__(
            name="DataProfiler",
            description="Comprehensive data profiling and quality assessment"
        )
        self.config = config or ProfilerConfig()
        self.validator = DataValidator()

    # === WALIDACJA WEJŚCIA ===
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.

        Expected:
            data: pd.DataFrame (required)
        """
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame")
        return True

    # === GŁÓWNE WYKONANIE ===
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Profile data comprehensively.

        Args:
            data: Input DataFrame

        Returns:
            AgentResult with profiling information
        """
        result = AgentResult(agent_name=self.name)

        try:
            # Guard na puste dane — zwracamy strukturalnie poprawny output
            if data is None or data.empty:
                logger.warning("DataProfiler: received empty DataFrame.")
                empty_payload = self._empty_payload()
                result.data = empty_payload
                return result

            # 1) Globalny score jakości
            quality_score, quality_details = self.validator.get_data_quality_score(data)

            # 2) Profil statystyczny
            statistical_profile = self._get_statistical_profile(data)

            # 3) Problemy jakości
            quality_issues = self._identify_quality_issues(data)

            # 4) Charakterystyki cech
            feature_characteristics = self._get_feature_characteristics(data)

            # 5) Korelacje (numeryczne)
            correlations = self._get_correlations(data)

            # Zbiorczy kontrakt
            result.data = {
                "quality_score": float(quality_score),
                "quality_details": quality_details,
                "statistical_profile": statistical_profile,
                "quality_issues": quality_issues,
                "feature_characteristics": feature_characteristics,
                "correlations": correlations,
            }

            logger.success(f"Data profiling complete. Quality score: {quality_score:.1f}/100")

        except Exception as e:
            result.add_error(f"Data profiling failed: {e}")
            logger.exception(f"Data profiling error: {e}")

        return result

    # === FUNKCJE POMOCNICZE: PROFIL STATYSTYCZNY ===
    def _get_statistical_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical profile of dataset (defensywnie i bezpiecznie dla 0 wierszy)."""
        n_rows = int(len(df))
        n_cols = int(len(df.columns))

        # dtypes
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        dt_cols  = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns

        # pamięć (deep=True liczy obiekty)
        try:
            memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
        except Exception:
            memory_mb = float(df.memory_usage().sum() / 1024**2)

        # duplikaty i braki — bez division-by-zero
        n_dup = int(df.duplicated().sum())
        pct_dup = float((n_dup / n_rows) * 100) if n_rows > 0 else 0.0

        total_missing = int(df.isna().sum().sum())
        denom = n_rows * n_cols if (n_rows > 0 and n_cols > 0) else 1
        pct_missing = float((total_missing / denom) * 100)

        cols_with_missing = df.isna().sum()
        cols_with_missing = cols_with_missing[cols_with_missing > 0].astype(int).to_dict()

        profile = StatisticalProfile(
            n_rows=n_rows,
            n_columns=n_cols,
            n_numeric=int(len(num_cols)),
            n_categorical=int(len(cat_cols)),
            n_datetime=int(len(dt_cols)),
            memory_mb=memory_mb,
            duplicates={"n_duplicates": n_dup, "pct_duplicates": pct_dup},
            missing_data={
                "total_missing": total_missing,
                "pct_missing": pct_missing,
                "columns_with_missing": cols_with_missing,
            },
        )
        return profile.to_dict()

    # === FUNKCJE POMOCNICZE: WYKRYWANIE PROBLEMÓW JAKOŚCI ===
    def _identify_quality_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify data quality issues (braki, stałe, kardynalność, duplikaty, outliery)."""
        cfg = self.config
        issues: List[Dict[str, Any]] = []
        n_rows = max(1, len(df))  # guard

        # Missing data issues
        missing_counts = df.isna().sum()
        high_missing_cols = missing_counts[missing_counts > (cfg.high_missing_col_threshold * len(df))].index
        for col in high_missing_cols:
            issues.append({
                "type": "high_missing_data",
                "severity": "high",
                "column": str(col),
                "description": f"Kolumna '{col}' ma >{int(cfg.high_missing_col_threshold*100)}% brakujących wartości",
                "missing_pct": float((missing_counts[col] / n_rows) * 100),
            })

        # Constant columns
        for col in df.columns:
            try:
                nunique = int(df[col].nunique(dropna=False))
            except Exception:
                nunique = 0
            if nunique == 1:
                issues.append({
                    "type": "constant_column",
                    "severity": "medium",
                    "column": str(col),
                    "description": f"Kolumna '{col}' ma tylko jedną unikalną wartość",
                })

        # High cardinality categorical (obejmuje object i category)
        for col in df.select_dtypes(include=["object", "category"]).columns:
            try:
                n_unique = int(df[col].nunique(dropna=True))
            except Exception:
                n_unique = 0
            if len(df) > 0 and (n_unique > cfg.high_cardinality_ratio * len(df)):
                issues.append({
                    "type": "high_cardinality",
                    "severity": "low",
                    "column": str(col),
                    "description": f"Kolumna '{col}' ma bardzo wysoką kardynalność ({n_unique} unikalnych)",
                    "unique": n_unique,
                })

        # Duplicates
        n_duplicates = int(df.duplicated().sum())
        if n_duplicates > 0:
            severity = "medium" if (n_duplicates / n_rows) > self.config.duplicates_row_ratio_mid else "low"
            issues.append({
                "type": "duplicates",
                "severity": severity,
                "description": f"Znaleziono {n_duplicates} zduplikowanych wierszy",
                "n_duplicates": n_duplicates,
            })

        # Outliers (IQR) dla kolumn numerycznych
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower, upper = q1 - cfg.outlier_iqr_factor * iqr, q3 + cfg.outlier_iqr_factor * iqr
            n_outliers = int(((s < lower) | (s > upper)).sum())
            if len(s) > 0 and (n_outliers / len(s)) > cfg.outlier_row_ratio_flag:
                issues.append({
                    "type": "outliers",
                    "severity": "low",
                    "column": str(col),
                    "description": f"Kolumna '{col}' ma {n_outliers} outliers (IQR method)",
                    "n_outliers": n_outliers,
                })

        return issues

    # === FUNKCJE POMOCNICZE: CHARAKTERYSTYKI CECH ===
    def _get_feature_characteristics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Kategoryzuje cechy wg typu, kardynalności i braków."""
        cfg = self.config
        ch: Dict[str, List[str]] = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "high_cardinality": [],
            "binary": [],
            "constant": [],
            "high_missing": [],
        }

        for col in df.columns:
            s = df[col]

            # Typ
            if pd.api.types.is_numeric_dtype(s):
                ch["numeric"].append(str(col))
            elif pd.api.types.is_datetime64_any_dtype(s):
                ch["datetime"].append(str(col))
            else:
                ch["categorical"].append(str(col))

            # Kardynalność
            try:
                n_unique = int(s.nunique(dropna=True))
            except Exception:
                n_unique = 0

            if n_unique == 1:
                ch["constant"].append(str(col))
            elif n_unique == 2:
                ch["binary"].append(str(col))
            elif len(df) > 0 and (n_unique > cfg.high_cardinality_ratio * len(df)):
                ch["high_cardinality"].append(str(col))

            # Braki
            if len(df) > 0 and (int(s.isna().sum()) > cfg.high_missing_flag_threshold * len(df)):
                ch["high_missing"].append(str(col))

        return ch

    # === FUNKCJE POMOCNICZE: KORELACJE ===
    def _get_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza korelacji dla cech numerycznych.
        - Bezpieczne próbkowanie przy bardzo dużych danych.
        - Wysokie korelacje > cfg.corr_abs_threshold.
        """
        cfg = self.config
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            return {"n_numeric_features": len(num_cols), "correlations": None, "high_correlations": []}

        # Sampling safety dla bardzo dużych danych
        if len(df) > cfg.max_corr_rows:
            df_corr = df[num_cols].sample(n=cfg.max_corr_rows, random_state=42).copy()
        else:
            df_corr = df[num_cols].copy()

        # Polityka braków
        if cfg.corr_nan_policy == "drop":
            df_corr = df_corr.dropna()
        # "pairwise" – zostawiamy NA; pandas corr(method='pearson') robi pairwise

        if df_corr.empty or len(df_corr.columns) < 2:
            return {"n_numeric_features": len(num_cols), "correlations": None, "high_correlations": []}

        corr_matrix = df_corr.corr(numeric_only=True)
        # Wyszukiwanie wysokich korelacji
        high_corr: List[Dict[str, Any]] = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = float(corr_matrix.iloc[i, j])
                if abs(val) >= cfg.corr_abs_threshold:
                    high_corr.append({
                        "feature1": str(cols[i]),
                        "feature2": str(cols[j]),
                        "correlation": val,
                    })

        return {
            "n_numeric_features": len(num_cols),
            "correlation_matrix": corr_matrix.round(6).to_dict(),
            "high_correlations": high_corr,
            "n_high_correlations": len(high_corr),
        }

    # === PAYLOAD DLA PUSTYCH DANYCH ===
    def _empty_payload(self) -> Dict[str, Any]:
        return {
            "quality_score": 0.0,
            "quality_details": {"reason": "empty_dataframe"},
            "statistical_profile": {
                "n_rows": 0, "n_columns": 0, "n_numeric": 0, "n_categorical": 0, "n_datetime": 0,
                "memory_mb": 0.0,
                "duplicates": {"n_duplicates": 0, "pct_duplicates": 0.0},
                "missing_data": {"total_missing": 0, "pct_missing": 0.0, "columns_with_missing": {}},
            },
            "quality_issues": [],
            "feature_characteristics": {
                "numeric": [], "categorical": [], "datetime": [],
                "high_cardinality": [], "binary": [], "constant": [], "high_missing": []
            },
            "correlations": {"n_numeric_features": 0, "correlations": None, "high_correlations": []},
        }
