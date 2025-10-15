# agents/eda/__init__.py
"""
DataGenius PRO — EDA package (lazy exports + handy helpers)

Eksporty (leniwe, tylko jeśli moduły istnieją):
- EDAExplorer           (agents.eda.explorer: EDAExplorer)
- EDAReporter           (agents.eda.reporter: EDAReporter)
- DataQualityAgent      (agents.eda.quality: DataQualityAgent)

Dodatkowo dostępne od ręki lekkie helpery:
- quick_overview(df, max_cols=50) -> dict
- memory_usage(df) -> pandas.Series

Użycie:
    from agents.eda import quick_overview, EDAReporter
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Dodaj/zmień ścieżki jeśli Twoje pliki nazywają się inaczej
    "EDAExplorer": ("agents.eda.explorer", "EDAExplorer"),
    "EDAReporter": ("agents.eda.reporter", "EDAReporter"),
    "DataQualityAgent": ("agents.eda.quality", "DataQualityAgent"),
}

__all__ = tuple(list(_LAZY_EXPORTS.keys()) + ["quick_overview", "memory_usage"])


def __getattr__(name: str):
    """Leniwe rozwiązywanie symboli + cache w module globals()."""
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        try:
            module: ModuleType = import_module(mod_name)
        except Exception as e:
            raise AttributeError(
                f"Optional EDA component '{name}' not available "
                f"(failed to import {mod_name}): {e}"
            ) from e
        obj = getattr(module, symbol)
        globals()[name] = obj  # cache
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


# === Light-weight helpers (działają bez dodatkowych modułów) ===
from typing import Any, Optional
import pandas as pd
import numpy as np


def memory_usage(df: pd.DataFrame) -> pd.Series:
    """
    Zwraca zużycie pamięci per kolumna (MB) + total jako ostatni wiersz.
    """
    per_col = df.memory_usage(deep=True) / (1024 ** 2)
    per_col.index = per_col.index.astype(str)
    total = pd.Series({"__TOTAL__": float(per_col.sum())})
    return pd.concat([per_col.astype(float), total])


def quick_overview(df: pd.DataFrame, max_cols: int = 50) -> dict:
    """
    Bardzo szybki przegląd ramki:
      - shape, dtypes, missing%, nunique
      - basic stats (num)
      - top-k najczęstszych wartości (cat)
    Zwraca słownik gotowy do serializacji/wyświetlenia.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Provide a non-empty pandas DataFrame."}

    n_rows, n_cols = df.shape
    sample_cols = df.columns[:max_cols]

    # missing & nunique
    miss_pct = (df[sample_cols].isna().mean() * 100).round(3).to_dict()
    nunique = df[sample_cols].nunique(dropna=True).to_dict()
    dtypes = {c: str(t) for c, t in df[sample_cols].dtypes.items()}

    # numeric stats
    num_cols = df[sample_cols].select_dtypes(include=[np.number]).columns
    num_stats = (
        df[num_cols]
        .describe(percentiles=[0.01, 0.05, 0.95, 0.99])
        .T.round(6)
        .to_dict(orient="index")
        if len(num_cols) else {}
    )

    # categorical top values (do 5)
    cat_cols = df[sample_cols].select_dtypes(include=["object", "category"]).columns
    cat_top = {}
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(5)
        cat_top[str(c)] = {str(k): int(v) for k, v in vc.items()}

    return {
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "preview_cols": list(map(str, sample_cols)),
        "dtypes": dtypes,
        "missing_pct": miss_pct,
        "nunique": {str(k): int(v) for k, v in nunique.items()},
        "numeric_stats": num_stats,
        "categorical_top_values": cat_top,
        "memory_mb": float(memory_usage(df)["__TOTAL__"]),
    }
