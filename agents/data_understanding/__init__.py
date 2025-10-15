# agents/data_understanding/__init__.py
"""
DataGenius PRO — Data Understanding (lazy exports + handy helpers)

Leniwe eksporty (załadują się tylko jeśli moduły istnieją):
- SchemaInspector        (agents.data_understanding.schema: SchemaInspector)
- DataProfiler           (agents.data_understanding.profiler: DataProfiler)
- ConstraintValidator    (agents.data_understanding.constraints: ConstraintValidator)

Wbudowane helpery (działają bez dodatkowych plików):
- profile_summary(df, max_cols=80) -> dict
- target_diagnosis(df, target) -> dict
- potential_leakage(df, target=None, id_like_patterns=('id','uuid','guid')) -> dict
- simple_quality_flags(df, max_card=1000) -> dict
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple

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
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        module: ModuleType = import_module(mod_name)
        obj = getattr(module, symbol)
        globals()[name] = obj  # cache
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))

# ==== Lightweight helpers ====
from typing import Any, Iterable, Optional
import numpy as np
import pandas as pd

def profile_summary(df: pd.DataFrame, max_cols: int = 80) -> dict:
    """Szybki profil: kształt, dtypes, missing, unikalność, zakres/percentyle, top wartości."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Provide a non-empty pandas DataFrame."}
    cols = df.columns[:max_cols]
    n, m = df.shape
    dtypes = {c: str(t) for c, t in df[cols].dtypes.items()}
    missing = (df[cols].isna().mean() * 100).round(3).to_dict()
    nunique = df[cols].nunique(dropna=True).to_dict()

    num = df[cols].select_dtypes(include=[np.number])
    num_stats = {}
    if not num.empty:
        num_stats = (
            num.describe(percentiles=[.01, .05, .5, .95, .99])
               .T.round(6)
               .to_dict(orient="index")
        )

    cat = df[cols].select_dtypes(include=["object", "category"])
    cat_top = {}
    for c in cat.columns:
        vc = df[c].value_counts(dropna=False).head(5)
        cat_top[str(c)] = {str(k): int(v) for k, v in vc.items()}

    mem_mb = float((df.memory_usage(deep=True).sum() / 1024**2))
    return {
        "shape": {"rows": int(n), "cols": int(m)},
        "preview_cols": list(map(str, cols)),
        "dtypes": dtypes,
        "missing_pct": missing,
        "nunique": {str(k): int(v) for k, v in nunique.items()},
        "numeric_stats": num_stats,
        "categorical_top_values": cat_top,
        "memory_mb": mem_mb,
    }

def target_diagnosis(df: pd.DataFrame, target: str) -> dict:
    """Krótka diagnoza targetu: typ problemu, klasy/rozklad, braki, niezbalansowanie."""
    if target not in df.columns:
        return {"error": f"Target '{target}' not in DataFrame"}
    y = df[target]
    missing = int(y.isna().sum())
    nunique = int(y.nunique(dropna=True))
    is_numeric = pd.api.types.is_numeric_dtype(y)

    # heurystyka typu problemu
    problem = "regression" if (is_numeric and nunique > 15) else "classification"

    details: dict[str, Any] = {"missing": missing, "nunique": nunique, "dtype": str(y.dtype)}
    if problem == "classification":
        vc = y.value_counts(dropna=True)
        details["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            ratio = float(vc.max() / max(1, vc.min()))
            details["imbalance_ratio_max_min"] = ratio
            details["imbalance_flag"] = bool(ratio > 10)
    else:
        try:
            details["summary"] = y.describe(percentiles=[.01, .05, .5, .95, .99]).to_dict()
        except Exception:
            pass

    return {"problem_type": problem, "details": details}

def potential_leakage(
    df: pd.DataFrame,
    target: Optional[str] = None,
    id_like_patterns: Iterable[str] = ("id", "uuid", "guid"),
) -> dict:
    """
    Proste heurystyki wycieku:
    - kolumny o 1-1 mapowaniu do targetu (dla klasyfikacji: niemal-perfect groupby),
    - ID-like (nunique ~ n_rows),
    - „późniejsze” kolumny (suffixy typu *_label, *_target, *_outcome).
    """
    n = len(df)
    suspects: dict[str, dict[str, Any]] = {}
    cols = [c for c in df.columns if c != target]

    # ID-like
    for c in cols:
        nu = df[c].nunique(dropna=True)
        name = c.lower()
        if nu >= max(0.9 * n, 1000) or any(p in name for p in id_like_patterns):
            suspects[c] = {"reason": "id_like/high_cardinality", "nunique": int(nu)}

    # name-based
    for c in cols:
        low = c.lower()
        if any(s in low for s in ("target", "label", "outcome", "groundtruth", "gt_")):
            suspects.setdefault(c, {"reason": ""})
            suspects[c]["reason"] = (suspects[c].get("reason", "") + "|name_leaky").strip("|")

    # near-perfect mapping to target (only for small cardinality targets)
    if target and target in df.columns:
        y = df[target]
        if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 50:
            for c in cols:
                try:
                    if df[c].isna().all():
                        continue
                    grp = df.groupby(c, dropna=False)[target].nunique()
                    if int(grp.max()) == 1 and df[c].nunique() <= 0.8 * n:
                        suspects.setdefault(c, {"reason": ""})
                        suspects[c]["reason"] = (suspects[c].get("reason", "") + "|perfect_map_to_target").strip("|")
                except Exception:
                    continue

    return {"suspected_columns": suspects, "n_suspects": len(suspects)}

def simple_quality_flags(df: pd.DataFrame, max_card: int = 1000) -> dict:
    """
    Lekki „data health”:
    - procent braków > 50%
    - stałe kolumny
    - bardzo wysoka krotność dla kategorii (możliwy ID)
    """
    flags = {"high_missing": [], "constant_columns": [], "high_cardinality_categoricals": []}
    n = len(df)
    for c in df.columns:
        s = df[c]
        miss = float(s.isna().mean())
        if miss > 0.5:
            flags["high_missing"].append(c)
        try:
            if s.nunique(dropna=True) <= 1:
                flags["constant_columns"].append(c)
        except Exception:
            pass
        if (s.dtype == "object" or pd.api.types.is_categorical_dtype(s)) and s.nunique(dropna=True) > max_card:
            flags["high_cardinality_categoricals"].append(c)
    return flags
