# === OPIS MODUŁU ===
"""
DataGenius PRO - Data Profiler (PRO++++++++++)
Kompleksowe profilowanie danych i ocena jakości datasetu.

Wyjściowy kontrakt (dict):
{
    "quality_score": float,                  # 0..100
    "quality_details": Dict[str, Any],       # szczegóły z DataValidator
    "statistical_profile": Dict[str, Any],   # rozmiary, dtypes, pamięć, braki, duplikaty, meta
    "quality_issues": List[Dict[str, Any]],  # wykryte problemy jakości (typed)
    "feature_characteristics": Dict[str, List[str]],  # kategorie cech (w tym id_like, quasi_constant, text_heavy)
    "correlations": Dict[str, Any]           # correlation_matrix + high_correlations + liczby
}
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Literal, Callable
import time
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from loguru import logger

# Zależności domenowe (dostarczane w projekcie)
from core.base_agent import BaseAgent, AgentResult
from core.data_validator import DataValidator


# === NAZWA_SEKCJI === KONFIG / STAŁE ===
@dataclass(frozen=True)
class ProfilerConfig:
    """Parametry działania profilera."""
    high_missing_col_threshold: float = 0.50   # >50% braków => issue high
    high_missing_flag_threshold: float = 0.30  # >30% => flaga w characteristics
    high_cardinality_ratio:   float = 0.90     # >90% unikalnych względem N
    quasi_constant_ratio:     float = 0.995    # >99.5% powtórzeń tej samej wartości → quasi-constant
    outlier_iqr_factor:       float = 3.0      # IQR fence
    outlier_row_ratio_flag:   float = 0.05     # >5% outliers => issue low
    duplicates_row_ratio_mid: float = 0.10     # >10% => severity "medium"
    corr_abs_threshold:       float = 0.80     # wysokie korelacje
    max_corr_rows:            int   = 200_000  # sampling safety dla korelacji
    corr_nan_policy:          Literal["pairwise","drop"] = "pairwise"  # polityka braków
    text_heavy_avg_len:       int   = 64       # średnia długość tekstu powyżej -> text_heavy
    id_like_unique_ratio:     float = 0.98     # >98% unikalnych vs N → id_like
    max_corr_cols:            int   = 200      # miękki limit liczby kolumn do korelacji (O(k^2))
    round_corr:               int   = 6        # precyzja zaokrągleń korelacji
    enable_mixed_type_check:  bool  = True     # wykrywanie „mieszanych” w kolumnie object
    include_dataset_hash:     bool  = True     # sygnatura wejścia w statistical_profile
    corr_method:              Literal["pearson","spearman"] = "pearson"  # metoda korelacji
    object_datetime_parse_ratio: float = 0.95  # jeśli >95% parsowalne → traktuj jako datetime (heurystyka, kopia kolumny)


# === NAZWA_SEKCJI === MODELE DANYCH (profil statystyczny) ===
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
    dtypes: Dict[str, str]
    meta: Dict[str, Any]

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
    """
    Stabilna sygnatura datasetu (kolumny + do 100k wierszy) — przydatne do cache/porównań.
    """
    try:
        sample = df if len(df) <= sample_rows else df.sample(n=sample_rows, random_state=42)
        h = hashlib.sha1()
        h.update("|".join(map(str, df.columns)).encode("utf-8"))
        # Używamy hash_pandas_object dla deterministycznej sygnatury
        h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
        return f"h{h.hexdigest()[:16]}"
    except Exception:
        return f"h{hash((tuple(df.columns), df.shape)) & 0xFFFFFFFF:X}"


def _truncate(obj: Any, limit: int = 400) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s)-limit} chars)"


def _is_object_dtype(s: pd.Series) -> bool:
    """True dla pandas 'object' lub 'string' dtypes."""
    return s.dtype == "object" or s.dtype.name == "string"


def _coerce_object_datetime_like(s: pd.Series, parse_ratio: float) -> Tuple[pd.Series, bool]:
    """
    Heurystycznie sprawdza, czy kolumna object/string jest „datopodobna”.
    Jeśli > parse_ratio wartości parsowalne do datetime → zwraca skopiowaną, sparsowaną serię i True.
    W przeciwnym razie zwraca oryginał i False. Zero side-effectów na oryginalnym df.
    """
    if not _is_object_dtype(s):
        return s, False
    try:
        sample = s.dropna().astype(str)
        if sample.empty:
            return s, False
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        ratio = float(parsed.notna().mean())
        if ratio >= parse_ratio:
            parsed_full = pd.to_datetime(s.astype(str), errors="coerce", utc=False)
            return parsed_full, True
        return s, False
    except Exception:
        return s, False


# === NAZWA_SEKCJI === KLASA GŁÓWNA AGENDA ===
class DataProfiler(BaseAgent):
    """
    Agent profilujący dane (PRO++++++++++):
    - Jakość danych (score + szczegóły) z DataValidator
    - Profil statystyczny + meta (w tym dataset_hash opcjonalnie)
    - Problemy jakości (braki, stałe, kwazi-stałe, kardynalność, duplikaty, outliery, id-like, mixed-type, text-heavy)
    - Charakterystyki cech
    - Korelacje numeryczne (bezpiecznie i wydajnie; pearson/spearman; higiena NaN/inf)
    """

    def __init__(self, config: Optional[ProfilerConfig] = None) -> None:
        super().__init__(
            name="DataProfiler",
            description="Comprehensive data profiling and quality assessment"
        )
        warnings.filterwarnings("ignore")
        self.config = config or ProfilerConfig()
        # DataValidator może rzucić wyjątkiem — opakowujemy konstrukcję w guard
        try:
            self.validator = DataValidator()
        except Exception as e:
            logger.warning(f"DataValidator unavailable, running in degraded mode. Reason: {e}")
            class _NullValidator:
                def get_data_quality_score(self, *_args, **_kwargs):
                    return 0.0, {"error": "validator_unavailable"}
            self.validator = _NullValidator()

    # === NAZWA_SEKCJI === WALIDACJA WEJŚCIA ===
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

    # === NAZWA_SEKCJI === GŁÓWNE WYKONANIE ===
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Profile data comprehensively.

        Args:
            data: Input DataFrame

        Returns:
            AgentResult with profiling information (kontrakt jak w docstringu)
        """
        result = AgentResult(agent_name=self.name)
        try:
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                logger.warning("DataProfiler: received empty or invalid DataFrame.")
                result.data = self._empty_payload()
                return result

            t0 = time.perf_counter()

            # 1) Quality (delegacja do DataValidator) — defensywnie
            try:
                quality_score, quality_details = self.validator.get_data_quality_score(data)
            except Exception as e:
                logger.warning(f"DataValidator failed: {e}")
                quality_score, quality_details = 0.0, {"error": "validator_failed", "message": str(e)}

            # 2) Statistical profile
            statistical_profile = self._get_statistical_profile(data)

            # 3) Quality issues
            quality_issues = self._identify_quality_issues(data)

            # 4) Feature characteristics
            feature_characteristics = self._get_feature_characteristics(data)

            # 5) Correlations (numeric)
            correlations = self._get_correlations(data)

            result.data = {
                "quality_score": float(quality_score),
                "quality_details": quality_details,
                "statistical_profile": statistical_profile,
                "quality_issues": quality_issues,
                "feature_characteristics": feature_characteristics,
                "correlations": correlations,
            }

            dt = (time.perf_counter() - t0) * 1000
            logger.success(f"DataProfiler: profiling complete in {dt:.1f} ms, quality={quality_score:.1f}/100")

        except Exception as e:
            result.add_error(f"Data profiling failed: {e}")
            logger.exception(f"Data profiling error: {e}")

        return result

    # === NAZWA_SEKCJI === PROFIL STATYSTYCZNY ===
    @_timeit("statistical_profile")
    def _get_statistical_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical profile of dataset (defensywnie, odpornie na edge-case’y)."""
        n_rows = int(len(df))
        n_cols = int(len(df.columns))

        # Dtypes
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns

        # Heurystyczna detekcja dat w object/string (na kopiach serii — bez modyfikacji df)
        dt_like_cols: List[str] = []
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                dt_like_cols.append(str(col))
                continue
            if _is_object_dtype(s):
                _, is_dt = _coerce_object_datetime_like(s, self.config.object_datetime_parse_ratio)
                if is_dt:
                    dt_like_cols.append(str(col))

        # Pamięć
        try:
            memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
        except Exception:
            memory_mb = float(df.memory_usage().sum() / 1024**2)

        # Duplikaty
        n_dup = int(df.duplicated().sum())
        pct_dup = float((n_dup / n_rows) * 100) if n_rows > 0 else 0.0

        # Braki
        total_missing = int(df.isna().sum().sum())
        denom = n_rows * n_cols if (n_rows > 0 and n_cols > 0) else 1
        pct_missing = float((total_missing / denom) * 100)
        cols_with_missing = df.isna().sum()
        cols_with_missing = cols_with_missing[cols_with_missing > 0].astype(int).to_dict()

        # Dtypes (per column)
        dtypes_map = {str(c): str(t) for c, t in df.dtypes.items()}

        meta: Dict[str, Any] = {}
        if self.config.include_dataset_hash:
            meta["dataset_hash"] = _dataset_hash(df)

        profile = StatisticalProfile(
            n_rows=n_rows,
            n_columns=n_cols,
            n_numeric=int(len(num_cols)),
            n_categorical=int(len(cat_cols)),
            n_datetime=int(len(dt_like_cols)),
            memory_mb=memory_mb,
            duplicates={"n_duplicates": n_dup, "pct_duplicates": pct_dup},
            missing_data={
                "total_missing": total_missing,
                "pct_missing": pct_missing,
                "columns_with_missing": cols_with_missing,
            },
            dtypes=dtypes_map,
            meta=meta,
        )
        return profile.to_dict()

    # === NAZWA_SEKCJI === WYKRYWANIE PROBLEMÓW JAKOŚCI ===
    @_timeit("quality_issues")
    def _identify_quality_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify data quality issues (braki, stałe, kwazi-stałe, kardynalność, duplikaty, outliery, id-like, mixed-type, text-heavy)."""
        cfg = self.config
        issues: List[Dict[str, Any]] = []
        n_rows = max(1, len(df))  # guard

        # Braki
        missing_counts = df.isna().sum()
        high_missing_cols = missing_counts[missing_counts > (cfg.high_missing_col_threshold * len(df))].index
        for col in high_missing_cols:
            issues.append({
                "type": "high_missing_data",
                "severity": "high",
                "column": str(col),
                "description": f"Kolumna '{col}' ma >{int(cfg.high_missing_col_threshold*100)}% braków",
                "missing_pct": float((missing_counts[col] / n_rows) * 100),
            })

        # Stałe / kwazi-stałe
        for col in df.columns:
            s = df[col]
            try:
                nunique_all = int(s.nunique(dropna=False))
                nunique = int(s.nunique(dropna=True))
            except Exception:
                nunique_all = nunique = 0

            if nunique_all == 1:
                issues.append({
                    "type": "constant_column",
                    "severity": "medium",
                    "column": str(col),
                    "description": f"Kolumna '{col}' ma tylko jedną unikalną wartość",
                })
            else:
                # quasi-constant: dominanta dominuje
                try:
                    top_freq = int(s.value_counts(dropna=False).iloc[0]) if len(s) else 0
                    if len(s) > 0 and (top_freq / len(s)) >= cfg.quasi_constant_ratio:
                        issues.append({
                            "type": "quasi_constant",
                            "severity": "low",
                            "column": str(col),
                            "description": f"Kolumna '{col}' ma dominującą wartość ≈{int(cfg.quasi_constant_ratio*100)}%+",
                            "top_freq_ratio": float(top_freq / len(s)),
                        })
                except Exception:
                    pass

        # Wysoka kardynalność (object/category/string)
        for col in df.select_dtypes(include=["object", "category", "string"]).columns:
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

        # Podejrzane ID (prawie wszystkie wartości unikalne)
        for col in df.columns:
            try:
                n_unique = int(df[col].nunique(dropna=True))
            except Exception:
                n_unique = 0
            if len(df) > 0 and (n_unique / len(df)) >= cfg.id_like_unique_ratio:
                issues.append({
                    "type": "id_like",
                    "severity": "info",
                    "column": str(col),
                    "description": f"Kolumna '{col}' wygląda jak identyfikator (≈{int(cfg.id_like_unique_ratio*100)}%+ unikalnych).",
                    "unique_ratio": float(n_unique / len(df)),
                })

        # Duplikaty wierszy
        n_duplicates = int(df.duplicated().sum())
        if n_duplicates > 0:
            severity = "medium" if (n_duplicates / n_rows) > self.config.duplicates_row_ratio_mid else "low"
            issues.append({
                "type": "duplicates",
                "severity": severity,
                "description": f"Znaleziono {n_duplicates} zduplikowanych wierszy",
                "n_duplicates": n_duplicates,
            })

        # Outliery (IQR) dla numeric
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
                    "description": f"Kolumna '{col}' ma {n_outliers} obserwacji odstających (IQR).",
                    "n_outliers": n_outliers,
                })

        # Mixed-type detection (dla object/string) — np. liczby i teksty w jednej kolumnie
        if self.config.enable_mixed_type_check:
            for col in df.select_dtypes(include=["object", "string"]).columns:
                s = df[col].dropna()
                if s.empty:
                    continue
                def _is_numeric_like(x: Any) -> bool:
                    if isinstance(x, (int, float, np.integer, np.floating)):
                        return True
                    if isinstance(x, str):
                        try:
                            float(x.replace(",", "."))
                            return True
                        except Exception:
                            return False
                    return False
                has_numeric_like = s.map(_is_numeric_like).any()
                has_alpha_like = s.map(lambda x: isinstance(x, str) and any(c.isalpha() for c in x)).any()
                if has_numeric_like and has_alpha_like:
                    issues.append({
                        "type": "mixed_type_heuristic",
                        "severity": "low",
                        "column": str(col),
                        "description": f"Kolumna '{col}' może zawierać mieszane typy (liczbowe i tekstowe).",
                    })

        # Tekst ciężki (średnia długość wysoka)
        for col in df.select_dtypes(include=["object", "string"]).columns:
            s = df[col].dropna().astype(str)
            if s.empty:
                continue
            avg_len = float(s.str.len().mean())
            if avg_len >= self.config.text_heavy_avg_len:
                issues.append({
                    "type": "text_heavy",
                    "severity": "info",
                    "column": str(col),
                    "description": f"Kolumna '{col}' zawiera długie teksty (avg len ≈ {avg_len:.1f}).",
                    "avg_len": avg_len,
                })

        # Datetime monotonicity (series)
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                sdt = s.dropna()
            elif _is_object_dtype(s):
                coerced, is_dt = _coerce_object_datetime_like(s, self.config.object_datetime_parse_ratio)
                if not is_dt:
                    continue
                sdt = coerced.dropna()
            else:
                continue

            if len(sdt) > 1 and (sdt.is_monotonic_increasing or sdt.is_monotonic_decreasing):
                issues.append({
                    "type": "datetime_monotonic",
                    "severity": "info",
                    "column": str(col),
                    "description": f"Kolumna '{col}' jest monotoniczna (czasowo).",
                })

        return issues

    # === NAZWA_SEKCJI === CHARAKTERYSTYKI CECH ===
    @_timeit("feature_characteristics")
    def _get_feature_characteristics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Kategoryzuje cechy wg typu, kardynalności, braków i heurystyk (id_like, quasi_constant, text_heavy)."""
        cfg = self.config
        ch: Dict[str, List[str]] = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "high_cardinality": [],
            "binary": [],
            "constant": [],
            "quasi_constant": [],
            "high_missing": [],
            "id_like": [],
            "text_heavy": [],
        }

        for col in df.columns:
            s = df[col]

            # Typ (uwzględnij object/string → datetime heurystycznie)
            if pd.api.types.is_numeric_dtype(s):
                ch["numeric"].append(str(col))
            elif pd.api.types.is_datetime64_any_dtype(s):
                ch["datetime"].append(str(col))
            elif _is_object_dtype(s):
                _, is_dt = _coerce_object_datetime_like(s, self.config.object_datetime_parse_ratio)
                if is_dt:
                    ch["datetime"].append(str(col))
                else:
                    ch["categorical"].append(str(col))
            else:
                ch["categorical"].append(str(col))

            # Kardynalność i stałość
            try:
                n_unique = int(s.nunique(dropna=True))
                n_unique_all = int(s.nunique(dropna=False))
            except Exception:
                n_unique = n_unique_all = 0

            if n_unique_all == 1:
                ch["constant"].append(str(col))
            elif n_unique == 2:
                ch["binary"].append(str(col))
            elif len(df) > 0 and (n_unique > cfg.high_cardinality_ratio * len(df)):
                ch["high_cardinality"].append(str(col))

            # quasi-constant
            try:
                top_freq = int(s.value_counts(dropna=False).iloc[0]) if len(s) else 0
                if len(s) > 0 and (top_freq / len(s)) >= cfg.quasi_constant_ratio and n_unique_all > 1:
                    ch["quasi_constant"].append(str(col))
            except Exception:
                pass

            # Braki
            try:
                if len(df) > 0 and (int(s.isna().sum()) > cfg.high_missing_flag_threshold * len(df)):
                    ch["high_missing"].append(str(col))
            except Exception:
                pass

            # id_like
            if len(df) > 0 and (n_unique / len(df)) >= cfg.id_like_unique_ratio:
                ch["id_like"].append(str(col))

            # text_heavy
            if _is_object_dtype(s):
                ss = s.dropna().astype(str)
                if not ss.empty and float(ss.str.len().mean()) >= cfg.text_heavy_avg_len:
                    ch["text_heavy"].append(str(col))

        return ch

    # === NAZWA_SEKCJI === KORELACJE NUMERYCZNE ===
    @_timeit("correlations")
    def _get_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza korelacji dla cech numerycznych.
        - Bezpieczne próbkowanie przy bardzo dużych danych.
        - Limit kolumn (O(k^2)) do obliczeń macierzy.
        - Wysokie korelacje > cfg.corr_abs_threshold.
        - Metody: pearson/spearman (konfig).
        - Higiena NaN/inf (zastępowanie na potrzeby corr, bez modyfikacji df).
        """
        cfg = self.config
        num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        k = len(num_cols_all)

        if k < 2:
            return {"n_numeric_features": k, "correlation_matrix": None, "high_correlations": [], "n_high_correlations": 0}

        # Soft limit kolumn
        num_cols = num_cols_all[: cfg.max_corr_cols] if k > cfg.max_corr_cols else num_cols_all
        if k > cfg.max_corr_cols:
            logger.warning(f"correlations: limiting numeric columns from {k} to {len(num_cols)} for performance.")

        # Sampling safety
        if len(df) > cfg.max_corr_rows:
            df_corr = df[num_cols].sample(n=cfg.max_corr_rows, random_state=42).copy()
        else:
            df_corr = df[num_cols].copy()

        # Polityka braków
        if cfg.corr_nan_policy == "drop":
            df_corr = df_corr.dropna()

        # Jeśli pozostał pusto/słabo – przerwij
        if df_corr.empty or len(df_corr.columns) < 2:
            return {"n_numeric_features": len(num_cols), "correlation_matrix": None, "high_correlations": [], "n_high_correlations": 0}

        # Higiena inf/NaN
        df_corr = df_corr.replace([np.inf, -np.inf], np.nan)

        # Obliczanie korelacji
        try:
            corr_matrix = df_corr.corr(method=cfg.corr_method, numeric_only=True)
        except Exception:
            corr_matrix = df_corr.corr(method="pearson", numeric_only=True)

        if cfg.corr_nan_policy == "drop":
            corr_matrix = corr_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

        if corr_matrix.empty or corr_matrix.shape[1] < 2:
            return {"n_numeric_features": len(num_cols), "correlation_matrix": None, "high_correlations": [], "n_high_correlations": 0}

        corr_matrix = corr_matrix.fillna(0.0).clip(-1.0, 1.0).round(cfg.round_corr)

        # Wyszukiwanie wysokich korelacji
        high_corr: List[Dict[str, Any]] = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = float(corr_matrix.iloc[i, j])
                if abs(val) >= cfg.corr_abs_threshold:
                    high_corr.append({"feature1": str(cols[i]), "feature2": str(cols[j]), "correlation": val})

        return {
            "n_numeric_features": len(num_cols),
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": high_corr,
            "n_high_correlations": len(high_corr),
        }

    # === NAZWA_SEKCJI === PAYLOAD DLA PUSTYCH DANYCH ===
    def _empty_payload(self) -> Dict[str, Any]:
        empty_corr = {
            "n_numeric_features": 0,
            "correlation_matrix": None,
            "high_correlations": [],
            "n_high_correlations": 0,
        }
        return {
            "quality_score": 0.0,
            "quality_details": {"reason": "empty_dataframe"},
            "statistical_profile": {
                "n_rows": 0, "n_columns": 0, "n_numeric": 0, "n_categorical": 0, "n_datetime": 0,
                "memory_mb": 0.0,
                "duplicates": {"n_duplicates": 0, "pct_duplicates": 0.0},
                "missing_data": {"total_missing": 0, "pct_missing": 0.0, "columns_with_missing": {}},
                "dtypes": {},
                "meta": {},
            },
            "quality_issues": [],
            "feature_characteristics": {
                "numeric": [], "categorical": [], "datetime": [],
                "high_cardinality": [], "binary": [], "constant": [],
                "quasi_constant": [], "high_missing": [], "id_like": [], "text_heavy": []
            },
            "correlations": empty_corr,
        }
