# === OPIS MODUŁU ===
"""
DataGenius PRO - Schema Analyzer (PRO++++++++++)
Analiza schematu i struktury danych: typy, semantyka, pamięć, statystyki oraz sugestie.

Kontrakt (AgentResult.data):
{
    "basic_info": {...},
    "columns": List[ColumnInfoDict],
    "dtypes_summary": Dict[str, int],
    "memory_info": {...},
    "suggestions": {
        "potential_primary_keys": List[str],
        "potential_casts": List[{"column": str, "from": str, "to": str}],
        "downcast_hints": List[{"column": str, "from": str, "to": str, "est_saving_mb": float}],
        "encoder_recommendations": List[{"column": str, "encoder": str, "reason": str}],
        "warnings": List[str]
    },
    "schema_fingerprint": str
}
"""

# === IMPORTY ===
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.utils import (
    detect_column_type,
    get_numeric_columns,
    get_categorical_columns,
)

# === NAZWA_SEKCJI === KONFIG / PROGI HEURYSTYCZNE ===
@dataclass(frozen=True)
class SchemaAnalyzerConfig:
    """Progi i zachowania analizy schematu."""
    high_missing_ratio_flag: float = 0.30    # >30% braków => flaga
    potential_pk_unique_ratio: float = 0.999 # >=99.9% unikatowości => potencjalny klucz
    potential_pk_max_nulls: int = 0          # dopuszczalne NaN w kol. kluczowej
    high_cardinality_ratio: float = 0.90     # >90% unikatów względem N
    quasi_constant_ratio: float = 0.995      # ≥99.5% dominanta → quasi-constant
    text_heavy_avg_len: int = 64             # średnia długość tekstu powyżej → text_heavy
    id_like_unique_ratio: float = 0.98       # >98% unikatów → id_like
    numeric_zero_count_limit: int = 10**9    # bezpieczny limit liczenia zer
    include_dataset_hash: bool = True        # do porównań/cachu
    truncate_log_chars: int = 500            # skracanie w logach
    cast_object_numeric_ratio: float = 0.90  # ≥90% parsowalnych → cast do float64
    cast_object_datetime_ratio: float = 0.90 # ≥90% parsowalnych → cast do datetime64[ns]
    enable_composite_pk_search: bool = True  # heurystyka 2-kolumnowych PK
    max_cols_for_composite_pk: int = 30      # limit kolumn dla PK-combo
    max_rows_for_composite_pk: int = 100_000 # limit wierszy dla PK-combo

# === NAZWA_SEKCJI === POMOCNICZE TYPY ===
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
        return asdict(self)

# === NAZWA_SEKCJI === HELPERY ===
def _timeit(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Dekorator do logowania czasu wykonania sekcji."""
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = (time.perf_counter() - t0) * 1000
                logger.debug(f"{name}: {dt:.1f} ms")
        return wrapped
    return deco

def _dataset_hash(df: pd.DataFrame, sample_rows: int = 100_000) -> str:
    """Stabilna sygnatura datasetu (kolumny + do 100k wierszy)."""
    try:
        sample = df if len(df) <= sample_rows else df.sample(n=sample_rows, random_state=42)
        h = hashlib.sha1()
        h.update("|".join(map(str, df.columns)).encode("utf-8"))
        h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
        return f"h{h.hexdigest()[:16]}"
    except Exception:
        return f"h{hash((tuple(df.columns), df.shape)) & 0xFFFFFFFF:X}"

def _safe_numeric(series: pd.Series) -> pd.Series:
    """Bezpieczne rzutowanie do float (z zachowaniem NaN)."""
    try:
        s = pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
        return s
    except Exception:
        return pd.to_numeric(series, errors="coerce")

def _truncate(obj: Any, limit: int) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s)-limit} chars)"

def _is_mixed_object(s: pd.Series) -> bool:
    """Heurystyka: mieszane typy w kolumnie object (cyfry + litery)."""
    if s.dtype != "object":
        return False
    ss = s.dropna().astype(str)
    if ss.empty:
        return False
    has_numeric_like = ss.map(lambda x: x.replace(".", "", 1).isdigit()).any()
    has_alpha_like = ss.map(lambda x: any(c.isalpha() for c in x)).any()
    return bool(has_numeric_like and has_alpha_like)

def _is_id_like(s: pd.Series, ratio_threshold: float) -> bool:
    try:
        n_unique = int(s.nunique(dropna=True))
        n = int(len(s))
        return n > 0 and (n_unique / n) >= ratio_threshold
    except Exception:
        return False

def _has_leading_trailing_spaces(s: pd.Series) -> bool:
    """Czy w object występują wiodące/końcowe spacje w ≥1 elemencie."""
    if s.dtype != "object":
        return False
    try:
        ss = s.dropna().astype(str)
        if ss.empty:
            return False
        mask = ss.ne(ss.str.strip())
        return bool(mask.any())
    except Exception:
        return False

def _is_non_normalized_categories(s: pd.Series) -> bool:
    """Czy widać różne warianty tej samej kategorii (np. case/spacing)? Heurystycznie."""
    if s.dtype != "object" and not pd.api.types.is_categorical_dtype(s):
        return False
    try:
        ss = s.dropna().astype(str)
        if ss.empty:
            return False
        norm = ss.str.strip().str.lower()
        return int(norm.nunique()) < int(ss.nunique())  # zlewa warianty → były nienormalizowane
    except Exception:
        return False

def _object_numeric_ratio(s: pd.Series) -> float:
    try:
        ss = pd.to_numeric(s.dropna().astype(str).str.replace(",", ".", regex=False), errors="coerce")
        return float(ss.notna().mean()) if len(ss) else 0.0
    except Exception:
        return 0.0

def _object_datetime_ratio(s: pd.Series) -> float:
    try:
        ss = pd.to_datetime(s.dropna().astype(str), errors="coerce", utc=False)
        return float(ss.notna().mean()) if len(ss) else 0.0
    except Exception:
        return 0.0

# === NAZWA_SEKCJI === KLASA GŁÓWNA AGENDA ===
class SchemaAnalyzer(BaseAgent):
    """
    Analizuje schemat danych i dostarcza szczegółowe metadane o kolumnach, typach i pamięci.
    Zwraca też heurystyczne sugestie (potencjalne klucze, casty, ostrzeżenia, downcasty, rekomendacje enkoderów).
    """

    def __init__(self, config: Optional[SchemaAnalyzerConfig] = None) -> None:
        super().__init__(
            name="SchemaAnalyzer",
            description="Analyzes data structure and column types"
        )
        self.config = config or SchemaAnalyzerConfig()
        self._log = logger.bind(agent="SchemaAnalyzer")

    # === NAZWA_SEKCJI === WALIDACJA WEJŚCIA ===
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters."""
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        df = kwargs["data"]
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame")
        # UWAGA: nie przerywamy na pustym DF — zwrócimy empty payload
        return True

    # === NAZWA_SEKCJI === GŁÓWNE WYKONANIE ===
    @_timeit("SchemaAnalyzer.execute")
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
            if data is None or data.empty:
                self._log.warning("received empty DataFrame.")
                result.data = self._empty_payload()
                return result

            # 1) Basic info
            basic_info = self._get_basic_info(data)

            # 2) Column analysis
            column_info = self._analyze_columns(data)

            # 3) Dtypes summary
            dtypes_summary = self._get_dtypes_summary(data)

            # 4) Memory usage
            memory_info = self._get_memory_info(data)

            # 5) Suggestions (PK, casty, ostrzeżenia, downcasty, rekomendacje enkoderów)
            suggestions = self._get_suggestions(data, column_info)

            # 6) Schema fingerprint (+ opcjonalny dataset hash)
            schema_fingerprint = self._schema_fingerprint(data)
            if self.config.include_dataset_hash:
                basic_info.setdefault("meta", {})["dataset_hash"] = _dataset_hash(data)

            # Store results
            result.data = {
                "basic_info": basic_info,
                "columns": column_info,
                "dtypes_summary": dtypes_summary,
                "memory_info": memory_info,
                "suggestions": suggestions,
                "schema_fingerprint": schema_fingerprint,
            }

            self._log.success(f"schema analysis complete: {len(data.columns)} columns")

        except Exception as e:
            result.add_error(f"Schema analysis failed: {e}")
            self._log.exception(f"Schema analysis error: {e}")

        return result

    # === NAZWA_SEKCJI === BASIC INFO ===
    @_timeit("basic_info")
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic DataFrame information."""
        return {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns.tolist()],
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "size": int(df.size),
        }

    # === NAZWA_SEKCJI === ANALIZA KOLUMN ===
    @_timeit("analyze_columns")
    def _analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze each column in detail."""
        cfg = self.config
        cols: List[Dict[str, Any]] = []

        for col in df.columns:
            s = df[col]
            n = max(1, len(s))

            # pandas "dtype string"
            try:
                dtype_str = str(s.dtype)
            except Exception:
                dtype_str = "unknown"

            # liczba zer tylko dla numeric/bool (odporność na ±inf/NaN)
            n_zeros = 0
            try:
                if pd.api.types.is_bool_dtype(s):
                    n_zeros = int((~s.fillna(False)).sum())  # False jako "zero"
                elif pd.api.types.is_numeric_dtype(s):
                    sn = pd.to_numeric(s, errors="coerce")
                    n_zeros = int((sn == 0).sum())
                if n_zeros > cfg.numeric_zero_count_limit:
                    n_zeros = 0
            except Exception:
                n_zeros = 0

            # semantyka (bezpiecznie)
            try:
                sem = detect_column_type(s)
            except Exception:
                sem = None

            # unikalność / missingi
            try:
                n_unique = int(s.nunique(dropna=True))
            except Exception:
                n_unique = 0
            try:
                n_missing = int(s.isna().sum())
            except Exception:
                n_missing = 0
            missing_pct = float((n_missing / n) * 100.0) if n > 0 else 0.0

            info = ColumnInfo(
                name=str(col),
                dtype=dtype_str,
                semantic_type=sem,
                n_unique=n_unique,
                n_missing=n_missing,
                missing_pct=missing_pct,
                n_zeros=n_zeros,
                extras={},
            )

            extras: Dict[str, Any] = {}

            # numeric / boolean
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                extras.update(self._get_numeric_stats(s))

            # categorical/object
            elif s.dtype == "object" or pd.api.types.is_categorical_dtype(s):
                extras.update(self._get_categorical_stats(s))
                # rozszerzone heurystyki
                try:
                    vc = s.dropna().astype(str).value_counts()
                    if not vc.empty:
                        top_freq = int(vc.iloc[0])
                        denom = max(1, len(s.dropna()))
                        if (top_freq / denom) >= cfg.quasi_constant_ratio and s.nunique(dropna=False) > 1:
                            extras["quasi_constant"] = True
                    if _is_mixed_object(s):
                        extras["mixed_type"] = True
                    # text-heavy
                    ss = s.dropna().astype(str)
                    if not ss.empty and float(ss.str.len().mean()) >= cfg.text_heavy_avg_len:
                        extras["text_heavy"] = True
                    # leading/trailing spaces
                    if _has_leading_trailing_spaces(s):
                        extras["has_leading_trailing_spaces"] = True
                    # non-normalized categories
                    if _is_non_normalized_categories(s):
                        extras["non_normalized_categories"] = True
                except Exception:
                    pass

            # datetime
            if pd.api.types.is_datetime64_any_dtype(s):
                extras.update(self._get_datetime_stats(s))
                # monotoniczność
                try:
                    sd = pd.to_datetime(s, errors="coerce").dropna()
                    if len(sd) > 1 and (sd.is_monotonic_increasing or sd.is_monotonic_decreasing):
                        extras["monotonic"] = True
                except Exception:
                    pass

            # id-like
            try:
                if _is_id_like(s, cfg.id_like_unique_ratio):
                    extras["id_like"] = True
            except Exception:
                pass

            # high cardinality (tylko kategorie/obiekty)
            try:
                if (s.dtype == "object" or pd.api.types.is_categorical_dtype(s)) and len(s) > 0:
                    n_unique_cat = int(s.nunique(dropna=True))
                    if (n_unique_cat > cfg.high_cardinality_ratio * len(s)):
                        extras["high_cardinality"] = True
            except Exception:
                pass

            info.extras = extras
            cols.append(info.to_dict())

        return cols

    # === NAZWA_SEKCJI === STATYSTYKI NUMERYCZNE ===
    def _get_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Stats for numeric/boolean column (defensywnie)."""
        try:
            if pd.api.types.is_bool_dtype(series):
                s = series.astype("Int8").astype(float)
            else:
                s = _safe_numeric(series)
        except Exception:
            s = _safe_numeric(series)

        if s.notna().sum() == 0:
            return {
                "mean": None, "std": None, "min": None, "max": None,
                "median": None, "q25": None, "q75": None,
                "skewness": None, "kurtosis": None,
            }

        s = s.replace([np.inf, -np.inf], np.nan)
        s = s.dropna()
        if s.empty:
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

    # === NAZWA_SEKCJI === STATYSTYKI KATEGORYCZNE ===
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

        n_categories = int(vc.shape[0])
        is_binary = bool(n_categories == 2)

        return {
            "mode": mode_val,
            "top_values": {str(k): int(v) for k, v in vc.head(5).to_dict().items()},
            "n_categories": n_categories,
            "is_binary": is_binary,
            "high_cardinality": bool(n_categories > 0.90 * max(1, len(s))),
        }

    # === NAZWA_SEKCJI === STATYSTYKI DATETIME ===
    def _get_datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Stats for datetime-like column."""
        s = pd.to_datetime(series, errors="coerce")
        s = s.dropna()
        if s.empty:
            return {"min_date": None, "max_date": None, "date_range_days": None, "maybe_mixed_timezones": False}

        min_d = s.min()
        max_d = s.max()
        # Wykrywanie potencjalnie mieszanych stref czasowych (heurystyka: obecność tz-aware + tz-naive)
        tz_mixed = False
        try:
            tz_mixed = bool(getattr(series.dtype, "tz", None)) and any(getattr(x, "tzinfo", None) is None for x in series.dropna().tolist())
        except Exception:
            tz_mixed = False

        return {
            "min_date": str(min_d),
            "max_date": str(max_d),
            "date_range_days": int((max_d - min_d).days),
            "maybe_mixed_timezones": tz_mixed,
        }

    # === NAZWA_SEKCJI === PODSUMOWANIE DTYPES ===
    def _get_dtypes_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Summary of pandas dtypes."""
        try:
            vc = df.dtypes.value_counts()
        except Exception:
            # awaryjnie
            types = [str(t) for t in df.dtypes]
            vc = pd.Series(types).value_counts()
        return {str(dtype): int(count) for dtype, count in vc.items()}

    # === NAZWA_SEKCJI === PAMIĘĆ ===
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

    # === NAZWA_SEKCJI === SUGESTIE HEURYSTYCZNE ===
    @_timeit("suggestions")
    def _get_suggestions(self, df: pd.DataFrame, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Heurystyki dot. potencjalnych kluczy, castów, ostrzeżeń jakościowych,
        downcastów (oszczędność RAM) oraz rekomendacji enkoderów.
        """
        cfg = self.config
        suggestions: Dict[str, Any] = {
            "potential_primary_keys": [],
            "potential_casts": [],
            "downcast_hints": [],
            "encoder_recommendations": [],
            "warnings": [],
        }

        n = max(1, len(df))
        cat_cols = get_categorical_columns(df)
        num_cols = get_numeric_columns(df)

        # Potencjalne PK (pojedyncze)
        for c in columns:
            try:
                unique_ratio = (c.get("n_unique", 0) / n)
                if (unique_ratio >= cfg.potential_pk_unique_ratio) and (c.get("n_missing", 0) <= cfg.potential_pk_max_nulls):
                    suggestions["potential_primary_keys"].append(c["name"])
            except Exception:
                continue

        # Potencjalne PK złożone (2-kolumnowe), tylko dla umiarkowanych rozmiarów
        if cfg.enable_composite_pk_search and len(df.columns) <= cfg.max_cols_for_composite_pk and len(df) <= cfg.max_rows_for_composite_pk:
            try:
                cols = [c["name"] for c in columns if c["name"] not in suggestions["potential_primary_keys"]]
                cols = [c for c in cols if int(df[c].isna().sum()) == 0]
                nunique_map = {c: int(df[c].nunique(dropna=True)) for c in cols}
                cols_sorted = sorted(cols, key=lambda x: nunique_map[x], reverse=True)[:20]  # miękki limit
                for i in range(min(20, len(cols_sorted))):
                    for j in range(i + 1, min(20, len(cols_sorted))):
                        a, b = cols_sorted[i], cols_sorted[j]
                        if nunique_map[a] <= 1 or nunique_map[b] <= 1:
                            continue
                        combo_unique = int(df[[a, b]].drop_duplicates().shape[0])
                        if combo_unique >= int(cfg.potential_pk_unique_ratio * n):
                            suggestions["potential_primary_keys"].append(f"{a} + {b}")
                suggestions["potential_primary_keys"] = list(dict.fromkeys(suggestions["potential_primary_keys"]))
            except Exception:
                pass

        # Potencjalne casty: object → numeric / datetime (progi 90%)
        for col in cat_cols:
            s = df[col]
            try:
                num_ratio = _object_numeric_ratio(s)
                if num_ratio >= cfg.cast_object_numeric_ratio:
                    suggestions["potential_casts"].append({"column": col, "from": str(s.dtype), "to": "float64"})
                    continue
                dt_ratio = _object_datetime_ratio(s)
                if dt_ratio >= cfg.cast_object_datetime_ratio:
                    suggestions["potential_casts"].append({"column": col, "from": str(s.dtype), "to": "datetime64[ns]"})
            except Exception:
                pass

        # Warnings: high missing / high cardinality / mixed / spaces / non-normalized
        for c in columns:
            name = c["name"]
            extras = c.get("extras", {})
            try:
                if c.get("missing_pct", 0.0) > (cfg.high_missing_ratio_flag * 100):
                    suggestions["warnings"].append(
                        f"Column '{name}' has high missing ratio ({c['missing_pct']:.1f}%)."
                    )
                if extras.get("high_cardinality", False):
                    suggestions["warnings"].append(f"Column '{name}' has high cardinality.")
                if extras.get("mixed_type", False):
                    suggestions["warnings"].append(f"Column '{name}' may contain mixed types (numeric/text).")
                if extras.get("has_leading_trailing_spaces", False):
                    suggestions["warnings"].append(f"Column '{name}' may contain leading/trailing spaces.")
                if extras.get("non_normalized_categories", False):
                    suggestions["warnings"].append(
                        f"Column '{name}' may contain non-normalized categories (case/spacing variants)."
                    )
            except Exception:
                continue

        # Downcast hints (num_cols)
        for col in num_cols:
            s = df[col]
            from_dtype = str(s.dtype)
            to_dtype: Optional[str] = None
            try:
                if pd.api.types.is_float_dtype(s):
                    s32 = s.astype(np.float32)
                    if np.isfinite(s32.fillna(0)).all():
                        to_dtype = "float32"
                elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
                    # bool -> uint8 estymacja; ale sugerujemy praktyczniejsze int32/uint32
                    try:
                        smin, smax = s.min(), s.max()
                    except Exception:
                        smin, smax = None, None
                    if smin is not None and smax is not None:
                        if smin >= 0:
                            if smax <= np.iinfo(np.uint32).max:
                                to_dtype = "uint32"
                        else:
                            if smin >= np.iinfo(np.int32).min and smax <= np.iinfo(np.int32).max:
                                to_dtype = "int32"
            except Exception:
                pass

            if to_dtype and len(df) > 0:
                try:
                    est = (s.memory_usage(deep=True) - s.astype(to_dtype).memory_usage(deep=True)) / (1024**2)
                    suggestions["downcast_hints"].append({
                        "column": col, "from": from_dtype, "to": to_dtype, "est_saving_mb": float(max(0.0, est))
                    })
                except Exception:
                    suggestions["downcast_hints"].append({
                        "column": col, "from": from_dtype, "to": to_dtype, "est_saving_mb": None
                    })

        # Encoder recommendations (w tym datetime)
        for c in columns:
            name = c["name"]
            extras = c.get("extras", {})
            dtype = c.get("dtype", "")
            try:
                if pd.api.types.is_datetime64_any_dtype(df[name]):
                    suggestions["encoder_recommendations"].append({
                        "column": name, "encoder": "DateParts", "reason": "datetime features (Y/M/D/DOW/H)"
                    })
                    continue

                if dtype in ("object", "category"):
                    n_categories = int(extras.get("n_categories") or 0)
                    is_binary = bool(extras.get("is_binary", False) or n_categories == 2)
                    if extras.get("high_cardinality"):
                        suggestions["encoder_recommendations"].append({
                            "column": name, "encoder": "Target/LeaveOneOut", "reason": "high_cardinality"
                        })
                    elif is_binary:
                        suggestions["encoder_recommendations"].append({
                            "column": name, "encoder": "Binary/Ordinal", "reason": "binary"
                        })
                    else:
                        suggestions["encoder_recommendations"].append({
                            "column": name, "encoder": "OneHot", "reason": "nominal moderate cardinality"
                        })
            except Exception:
                continue

        return suggestions

    # === NAZWA_SEKCJI === FINGERPRINT SCHEMATU ===
    def _schema_fingerprint(self, df: pd.DataFrame) -> str:
        """
        Tworzy odcisk palca (SHA1) schematu na podstawie nazw i dtypes.
        Zmiana kolejności lub dtype zmieni odcisk.
        """
        payload = "|".join([f"{str(c)}::{str(t)}" for c, t in zip(df.columns, df.dtypes)])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    # === NAZWA_SEKCJI === PAYLOAD DLA PUSTEGO DF ===
    def _empty_payload(self) -> Dict[str, Any]:
        return {
            "basic_info": {"n_rows": 0, "n_columns": 0, "column_names": [], "shape": (0, 0), "size": 0},
            "columns": [],
            "dtypes_summary": {},
            "memory_info": {"total_mb": 0.0, "per_row_bytes": 0.0, "by_column_mb": {}},
            "suggestions": {
                "potential_primary_keys": [],
                "potential_casts": [],
                "downcast_hints": [],
                "encoder_recommendations": [],
                "warnings": ["Empty DataFrame"],
            },
            "schema_fingerprint": hashlib.sha1(b"").hexdigest(),
        }
