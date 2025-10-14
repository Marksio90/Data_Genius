# === OPIS MODUŁU ===
"""
DataGenius PRO - Schema Analyzer (PRO+++)
Analiza schematu i struktury danych: typy, semantyka, pamięć, statystyki oraz sugestie.
"""

# === IMPORTY ===
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.utils import (
    detect_column_type,
    get_numeric_columns,
    get_categorical_columns,
)


# === KONFIG / PROGI HEURYSTYCZNE ===
@dataclass(frozen=True)
class SchemaAnalyzerConfig:
    """Progi i zachowania analizy schematu."""
    high_missing_ratio_flag: float = 0.30   # >30% braków => flaga
    potential_pk_unique_ratio: float = 0.999  # >=99.9% unikatowości => potencjalny klucz
    potential_pk_max_nulls: int = 0          # dopuszczalne NaN w kol. kluczowej
    high_cardinality_ratio: float = 0.90     # >90% unikatów względem N
    numeric_zero_count_limit: int = 10**9    # bezpieczny limit liczenia zer (ochrona przed złymi dtypes)


# === POMOCNICZE TYPY ===
@dataclass
class ColumnInfo:
    name: str
    dtype: str
    semantic_type: Optional[str]
    n_unique: int
    n_missing: int
    missing_pct: float
    n_zeros: int
    extras: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # extras zostawiamy jak jest (słownik)
        return d


# === KLASA GŁÓWNA AGENDA ===
class SchemaAnalyzer(BaseAgent):
    """
    Analizuje schemat danych i dostarcza szczegółowe metadane o kolumnach, typach i pamięci.
    Zwraca też heurystyczne sugestie (potencjalne klucze, casty, ostrzeżenia).
    """

    def __init__(self, config: Optional[SchemaAnalyzerConfig] = None) -> None:
        super().__init__(
            name="SchemaAnalyzer",
            description="Analyzes data structure and column types"
        )
        self.config = config or SchemaAnalyzerConfig()

    # === WALIDACJA WEJŚCIA ===
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters."""
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")

        df = kwargs["data"]
        if not isinstance(df, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame")

        if df.empty:
            raise ValueError("DataFrame is empty")

        return True

    # === GŁÓWNE WYKONANIE ===
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Analyze data schema.

        Args:
            data: Input DataFrame

        Returns:
            AgentResult with schema information
        """
        result = AgentResult(agent_name=self.name)

        try:
            # 1) Basic info
            basic_info = self._get_basic_info(data)

            # 2) Column analysis
            column_info = self._analyze_columns(data)

            # 3) Dtypes summary
            dtypes_summary = self._get_dtypes_summary(data)

            # 4) Memory usage
            memory_info = self._get_memory_info(data)

            # 5) Suggestions (PK, casty, ostrzeżenia)
            suggestions = self._get_suggestions(data, column_info)

            # 6) Schema fingerprint
            schema_fingerprint = self._schema_fingerprint(data)

            # Store results
            result.data = {
                "basic_info": basic_info,
                "columns": column_info,
                "dtypes_summary": dtypes_summary,
                "memory_info": memory_info,
                "suggestions": suggestions,
                "schema_fingerprint": schema_fingerprint,
            }

            logger.success(f"Schema analysis complete: {len(data.columns)} columns analyzed")

        except Exception as e:
            result.add_error(f"Schema analysis failed: {e}")
            logger.exception(f"Schema analysis error: {e}")

        return result

    # === BASIC INFO ===
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic DataFrame information."""
        return {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns.tolist()],
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "size": int(df.size),
        }

    # === ANALIZA KOLUMN ===
    def _analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze each column in detail."""
        cfg = self.config
        cols: List[Dict[str, Any]] = []

        for col in df.columns:
            s = df[col]
            n = max(1, len(s))  # guard dla procentów
            dtype_str = str(s.dtype)

            # Bezpieczne liczenie zer tylko dla numeric dtype
            n_zeros = int(((s == 0).sum()) if pd.api.types.is_numeric_dtype(s) else 0)

            info = ColumnInfo(
                name=str(col),
                dtype=dtype_str,
                semantic_type=self._safe_detect_semantic(s),
                n_unique=int(s.nunique(dropna=True)),
                n_missing=int(s.isna().sum()),
                missing_pct=float((s.isna().sum() / n) * 100),
                n_zeros=n_zeros if n_zeros < cfg.numeric_zero_count_limit else 0,
                extras={},
            )

            # Rozszerzenia per-typ
            extras: Dict[str, Any] = {}
            if pd.api.types.is_numeric_dtype(s):
                extras.update(self._get_numeric_stats(s))
            elif s.dtype == "object" or pd.api.types.is_categorical_dtype(s):
                extras.update(self._get_categorical_stats(s))
            elif pd.api.types.is_datetime64_any_dtype(s):
                extras.update(self._get_datetime_stats(s))

            info.extras = extras
            cols.append(info.to_dict())

        return cols

    # === STATYSTYKI NUMERYCZNE ===
    def _get_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Stats for numeric column (defensywnie)."""
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() == 0:
            return {
                "mean": None, "std": None, "min": None, "max": None,
                "median": None, "q25": None, "q75": None,
                "skewness": None, "kurtosis": None,
            }

        return {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "skewness": float(s.skew()) if s.count() > 2 else 0.0,
            "kurtosis": float(s.kurtosis()) if s.count() > 3 else 0.0,
        }

    # === STATYSTYKI KATEGORYCZNE ===
    def _get_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Stats for categorical/object column."""
        s = series.astype("string")
        vc = s.value_counts(dropna=True)

        mode_val = None
        try:
            mode_series = s.mode(dropna=True)
            if not mode_series.empty:
                mode_val = str(mode_series.iloc[0])
        except Exception:
            mode_val = None

        return {
            "mode": mode_val,
            "top_values": {str(k): int(v) for k, v in vc.head(5).to_dict().items()},
            "n_categories": int(vc.shape[0]),
            "is_binary": bool(vc.shape[0] == 2),
            "high_cardinality": bool(vc.shape[0] > 0.90 * max(1, len(s))),
        }

    # === STATYSTYKI DATETIME ===
    def _get_datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Stats for datetime-like column."""
        s = pd.to_datetime(series, errors="coerce")
        s = s.dropna()
        if s.empty:
            return {"min_date": None, "max_date": None, "date_range_days": None}

        min_d = s.min()
        max_d = s.max()
        return {
            "min_date": str(min_d),
            "max_date": str(max_d),
            "date_range_days": int((max_d - min_d).days),
        }

    # === PODSUMOWANIE DTYPES ===
    def _get_dtypes_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Summary of pandas dtypes."""
        vc = df.dtypes.value_counts()
        return {str(dtype): int(count) for dtype, count in vc.items()}

    # === PAMIĘĆ ===
    def _get_memory_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Memory usage summary."""
        try:
            mem = df.memory_usage(deep=True)
        except Exception:
            mem = df.memory_usage(deep=False)

        total = int(mem.sum())
        return {
            "total_mb": float(total / 1024**2),
            "per_row_bytes": float(total / max(1, len(df))),
            "by_column_mb": {str(col): float(val / 1024**2) for col, val in mem.items()},
        }

    # === SUGESTIE HEURYSTYCZNE ===
    def _get_suggestions(self, df: pd.DataFrame, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Heurystyki dot. potencjalnych kluczy, castów i ostrzeżeń jakościowych.
        """
        cfg = self.config
        suggestions: Dict[str, Any] = {
            "potential_primary_keys": [],
            "potential_casts": [],   # np. {"column": "colA", "from": "object", "to": "float64"}
            "warnings": [],
        }

        n = max(1, len(df))

        # Potencjalne PK: bardzo wysoka unikatowość, brak/mało NaN
        for c in columns:
            unique_ratio = (c.get("n_unique", 0) / n)
            if (unique_ratio >= cfg.potential_pk_unique_ratio) and (c.get("n_missing", 0) <= cfg.potential_pk_max_nulls):
                suggestions["potential_primary_keys"].append(c["name"])

        # Potencjalne casty: obiektowe kolumny wyglądające na numeryczne/datowe
        cat_cols = get_categorical_columns(df)
        for col in cat_cols:
            s = df[col]
            # numeric-like?
            num_coerced = pd.to_numeric(s, errors="coerce")
            if num_coerced.notna().sum() > 0 and (num_coerced.notna().sum() / n) > 0.9:
                suggestions["potential_casts"].append({"column": col, "from": str(s.dtype), "to": "float64"})
                continue
            # datetime-like?
            dt_coerced = pd.to_datetime(s, errors="coerce", utc=False)
            if dt_coerced.notna().sum() > 0 and (dt_coerced.notna().sum() / n) > 0.9:
                suggestions["potential_casts"].append({"column": col, "from": str(s.dtype), "to": "datetime64[ns]"})

        # Ostrzeżenia: wysokie braki, wysoka kardynalność
        for c in columns:
            if c.get("missing_pct", 0.0) > (cfg.high_missing_ratio_flag * 100):
                suggestions["warnings"].append(
                    f"Column '{c['name']}' has high missing ratio ({c['missing_pct']:.1f}%)."
                )
            # high cardinality dla kategorii
            extras = c.get("extras", {})
            if extras.get("high_cardinality", False):
                suggestions["warnings"].append(
                    f"Column '{c['name']}' has high cardinality."
                )

        return suggestions

    # === FINGERPRINT SCHEMATU ===
    def _schema_fingerprint(self, df: pd.DataFrame) -> str:
        """
        Tworzy odcisk palca (SHA1) schematu na podstawie nazw i dtypes.
        Zmiana kolejności lub dtype zmieni odcisk.
        """
        payload = "|".join([f"{str(c)}::{str(t)}" for c, t in zip(df.columns, df.dtypes)])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    # === BEZPIECZNE WYKRYCIE SEMANTYKI ===
    def _safe_detect_semantic(self, s: pd.Series) -> Optional[str]:
        try:
            return detect_column_type(s)
        except Exception:
            return None
