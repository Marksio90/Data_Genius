# agents/data_understanding/__init__.py
"""
DataGenius PRO — Data Understanding (lazy exports + handy helpers) — Enterprise PRO++++++++++++

Leniwe eksporty (załadują się tylko jeśli moduły istnieją):
- SchemaInspector        (agents.data_understanding.schema: SchemaInspector)
- DataProfiler           (agents.data_understanding.profiler: DataProfiler)
- ConstraintValidator    (agents.data_understanding.constraints: ConstraintValidator)

Wbudowane helpery (działają bez dodatkowych plików):
- profile_summary(df, max_cols=80) -> dict
- target_diagnosis(df, target) -> dict
- potential_leakage(df, target=None, id_like_patterns=('id','uuid','guid')) -> dict
- simple_quality_flags(df, max_card=1000) -> dict

Zasady PRO:
- Pełna walidacja wejść i bezpieczne try/except (częściowe wyniki zamiast twardych awarii)
- Czytelne logowanie (loguru -> fallback na logging)
- Precyzyjne typy i dokumentacja
"""

from __future__ import annotations

# === NAZWA_SEKCJI === Importy bazowe i logowanie ===
import warnings
warnings.filterwarnings("ignore")

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple, Any, Iterable, Optional, List

# Logowanie: preferuj loguru, inaczej stdlib logging.
try:
    from loguru import logger as _LOGGER
    _USE_LOGURU = True
except Exception:  # pragma: no cover
    import logging
    _LOGGER = logging.getLogger(__name__)
    if not _LOGGER.handlers:
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
        )
        _handler.setFormatter(_formatter)
        _LOGGER.addHandler(_handler)
        _LOGGER.setLevel(logging.INFO)
    _USE_LOGURU = False

# === NAZWA_SEKCJI === Lazy-exports (czytelne błędy i cache) ===
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "SchemaInspector": ("agents.data_understanding.schema", "SchemaInspector"),
    "DataProfiler": ("agents.data_understanding.profiler", "DataProfiler"),
    "ConstraintValidator": ("agents.data_understanding.constraints", "ConstraintValidator"),
}

__all__ = tuple(list(_LAZY_EXPORTS.keys()) + [
    "profile_summary",
    "target_diagnosis",
    "potential_leakage",
    "simple_quality_flags",
])

def __getattr__(name: str):
    """
    Leniwe uzyskanie symbolu. Jeśli moduł nie istnieje — rzuci AttributeError
    z czytelną wskazówką co do brakującego zależnego pliku.
    """
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        try:
            module: ModuleType = import_module(mod_name)
        except ModuleNotFoundError as e:  # precyzyjny komunikat
            msg = (f"Nie można załadować modułu '{mod_name}' dla symbolu '{name}'. "
                   f"Upewnij się, że plik i importy istnieją. Oryginalny błąd: {e}")
            _LOGGER.warning(msg)
            raise AttributeError(msg) from e
        obj = getattr(module, symbol, None)
        if obj is None:
            msg = (f"Symbol '{symbol}' nie znaleziony w module '{mod_name}'. "
                   f"Sprawdź wersje plików i eksporty.")
            _LOGGER.warning(msg)
            raise AttributeError(msg)
        globals()[name] = obj  # cache symbolu dla kolejnych odwołań
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))

# === NAZWA_SEKCJI === Lightweight helpers (bez dodatkowych plików) ===
import numpy as np
import pandas as pd

def _validate_df(df: Any) -> Optional[str]:
    """Zwraca komunikat błędu jeśli df jest niepoprawny, inaczej None."""
    if not isinstance(df, pd.DataFrame):
        return "Provide a pandas DataFrame."
    if df.empty:
        return "Provide a non-empty pandas DataFrame."
    return None

def _safe_numeric_describe(num: pd.DataFrame, percentiles: Iterable[float]) -> Dict[str, Dict[str, float]]:
    """Bezpieczne .describe() dla danych numerycznych."""
    try:
        desc = num.describe(percentiles=list(percentiles)).T
        return desc.round(6).to_dict(orient="index")
    except Exception as e:
        _LOGGER.warning(f"numeric describe failed: {e}")
        return {}

def _safe_categorical_top(df: pd.DataFrame, cols: Iterable[str], top_k: int = 5) -> Dict[str, Dict[str, int]]:
    """Top K wartości dla kolumn kategorycznych (bezpiecznie, łącznie z NaN)."""
    out: Dict[str, Dict[str, int]] = {}
    for c in cols:
        try:
            vc = df[c].value_counts(dropna=False).head(top_k)
            out[str(c)] = {str(k): int(v) for k, v in vc.items()}
        except Exception as e:
            _LOGGER.debug(f"value_counts failed for column '{c}': {e}")
    return out

def profile_summary(df: pd.DataFrame, max_cols: int = 80) -> Dict[str, Any]:
    """
    Szybki profil: kształt, dtypes, missing, unikalność, zakres/percentyle, top wartości, pamięć.

    Args:
        df: Dane wejściowe (pandas DataFrame).
        max_cols: Limit liczby analizowanych kolumn (chroni przed OOM i długim czasem).

    Returns:
        dict z kluczowymi metadanymi zbioru.
    """
    err = _validate_df(df)
    if err:
        _LOGGER.warning(f"profile_summary: {err}")
        return {"error": err}

    cols = list(map(str, df.columns[:max_cols]))
    n, m = df.shape

    # Dtypes, missing, nunique (bezpiecznie)
    try:
        dtypes = {c: str(t) for c, t in df[cols].dtypes.items()}
    except Exception as e:
        _LOGGER.warning(f"dtypes failed: {e}")
        dtypes = {}

    try:
        missing = (df[cols].isna().mean() * 100).round(3).to_dict()
    except Exception as e:
        _LOGGER.warning(f"missing failed: {e}")
        missing = {}

    try:
        nunique = df[cols].nunique(dropna=True).to_dict()
    except Exception as e:
        _LOGGER.warning(f"nunique failed: {e}")
        nunique = {}

    # Numeric stats
    num = df[cols].select_dtypes(include=[np.number])
    numeric_stats = _safe_numeric_describe(num, percentiles=[.01, .05, .5, .95, .99]) if not num.empty else {}

    # Categorical top
    cat_cols: List[str] = list(df[cols].select_dtypes(include=["object", "category"]).columns)
    categorical_top_values = _safe_categorical_top(df, cat_cols, top_k=5) if cat_cols else {}

    # Memory footprint
    try:
        mem_mb = float((df.memory_usage(deep=True).sum() / 1024 ** 2))
    except Exception as e:
        _LOGGER.debug(f"memory_usage failed: {e}")
        mem_mb = float("nan")

    result: Dict[str, Any] = {
        "shape": {"rows": int(n), "cols": int(m)},
        "preview_cols": cols,
        "dtypes": dtypes,
        "missing_pct": {str(k): float(v) for k, v in missing.items()},
        "nunique": {str(k): int(v) for k, v in nunique.items()},
        "numeric_stats": numeric_stats,
        "categorical_top_values": categorical_top_values,
        "memory_mb": mem_mb,
    }
    _LOGGER.debug("profile_summary computed")
    return result

def target_diagnosis(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Krótka diagnoza targetu: typ problemu, klasy/rozkład, braki, niezbalansowanie.

    Heurystyka typu problemu:
      - numeric & wiele unikalnych (>15) → regression
      - w pozostałych przypadkach → classification

    Dla classification:
      - zwraca rozkład klas i flagę niezbalansowania (ratio max/min > 10)
    Dla regression:
      - zwraca opis statystyczny percentyli

    Returns:
        dict z 'problem_type' i 'details'
    """
    err = _validate_df(df)
    if err:
        _LOGGER.warning(f"target_diagnosis: {err}")
        return {"error": err}

    if target not in df.columns:
        msg = f"Target '{target}' not in DataFrame"
        _LOGGER.warning(msg)
        return {"error": msg}

    y = df[target]
    try:
        missing = int(y.isna().sum())
    except Exception:
        missing = 0

    try:
        nunique = int(y.nunique(dropna=True))
    except Exception:
        nunique = 0

    is_numeric = pd.api.types.is_numeric_dtype(y)
    problem = "regression" if (is_numeric and nunique > 15) else "classification"

    details: Dict[str, Any] = {"missing": missing, "nunique": nunique, "dtype": str(y.dtype)}
    if problem == "classification":
        try:
            vc = y.value_counts(dropna=True)
            details["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
            if len(vc) >= 2 and vc.min() > 0:
                ratio = float(vc.max() / vc.min())
                details["imbalance_ratio_max_min"] = ratio
                details["imbalance_flag"] = bool(ratio > 10.0)
            else:
                details["imbalance_ratio_max_min"] = float("inf")
                details["imbalance_flag"] = True
        except Exception as e:
            _LOGGER.warning(f"class distribution failed: {e}")
    else:  # regression
        try:
            details["summary"] = {k: float(v) for k, v in y.describe(percentiles=[.01, .05, .5, .95, .99]).to_dict().items()}
        except Exception as e:
            _LOGGER.warning(f"regression describe failed: {e}")

    return {"problem_type": problem, "details": details}

def potential_leakage(
    df: pd.DataFrame,
    target: Optional[str] = None,
    id_like_patterns: Iterable[str] = ("id", "uuid", "guid"),
) -> Dict[str, Any]:
    """
    Proste heurystyki wycieku:
    - kolumny o 1-1 mapowaniu do targetu (dla klasyfikacji: niemal-perfect groupby),
    - ID-like (nunique ~ n_rows lub nazwa zawiera wzorzec),
    - „późniejsze” kolumny (suffixy *_label, *_target, *_outcome, 'groundtruth', 'gt_').

    Uwaga: to heurystyka — wynik należy traktować jako listę „podejrzanych”.
    """
    err = _validate_df(df)
    if err:
        _LOGGER.warning(f"potential_leakage: {err}")
        return {"error": err, "suspected_columns": {}, "n_suspects": 0}

    n = len(df)
    suspects: Dict[str, Dict[str, Any]] = {}
    cols = [c for c in df.columns if c != target]

    # ID-like (wysoka krotność lub nazwa zawiera wzorzec)
    for c in cols:
        try:
            nu = int(df[c].nunique(dropna=True))
            name = str(c).lower()
            if nu >= max(int(0.9 * n), 1000) or any(p in name for p in id_like_patterns):
                suspects[c] = {"reason": "id_like/high_cardinality", "nunique": nu}
        except Exception as e:
            _LOGGER.debug(f"nunique failed for '{c}': {e}")

    # Nazwowe „leakujące”
    for c in cols:
        low = str(c).lower()
        if any(s in low for s in ("target", "label", "outcome", "groundtruth", "gt_")):
            suspects.setdefault(c, {})
            suspects[c]["reason"] = (suspects[c].get("reason", "") + "|name_leaky").strip("|")

    # Prawie perfekcyjne mapowanie do targetu (tylko gdy target o małej krotności)
    if target and target in df.columns:
        y = df[target]
        try:
            small_card_target = (not pd.api.types.is_numeric_dtype(y)) or (int(y.nunique(dropna=True)) <= 50)
        except Exception:
            small_card_target = True

        if small_card_target:
            for c in cols:
                try:
                    s = df[c]
                    if s.isna().all():
                        continue
                    grp = df.groupby(c, dropna=False)[target].nunique()
                    if int(grp.max()) == 1 and int(s.nunique(dropna=True)) <= int(0.8 * n):
                        suspects.setdefault(c, {})
                        suspects[c]["reason"] = (suspects[c].get("reason", "") + "|perfect_map_to_target").strip("|")
                except Exception as e:
                    _LOGGER.debug(f"groupby check failed for '{c}': {e}")

    return {"suspected_columns": suspects, "n_suspects": len(suspects)}

def simple_quality_flags(df: pd.DataFrame, max_card: int = 1000) -> Dict[str, Any]:
    """
    Lekki „data health”:
    - procent braków > 50%
    - stałe kolumny (nunique <= 1)
    - bardzo wysoka krotność dla kategorii (możliwy ID)

    Args:
      df: ramka danych
      max_card: próg dla high-cardinality kategorycznych
    """
    err = _validate_df(df)
    if err:
        _LOGGER.warning(f"simple_quality_flags: {err}")
        return {"error": err, "high_missing": [], "constant_columns": [], "high_cardinality_categoricals": []}

    flags = {"high_missing": [], "constant_columns": [], "high_cardinality_categoricals": []}
    for c in df.columns:
        s = df[c]
        # missing
        try:
            miss = float(s.isna().mean())
            if miss > 0.5:
                flags["high_missing"].append(c)
        except Exception:
            pass
        # constant
        try:
            if int(s.nunique(dropna=True)) <= 1:
                flags["constant_columns"].append(c)
        except Exception:
            pass
        # high-card categorical
        try:
            if (s.dtype == "object" or pd.api.types.is_categorical_dtype(s)) and int(s.nunique(dropna=True)) > max_card:
                flags["high_cardinality_categoricals"].append(c)
        except Exception:
            pass

    return flags
