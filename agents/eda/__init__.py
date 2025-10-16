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
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd

# ===== Lazy exports =====
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "EDAExplorer": ("agents.eda.explorer", "EDAExplorer"),
    "EDAReporter": ("agents.eda.reporter", "EDAReporter"),
    "DataQualityAgent": ("agents.eda.quality", "DataQualityAgent"),
}

__all__ = tuple(list(_LAZY_EXPORTS.keys()) + ["quick_overview", "memory_usage"])


def __getattr__(name: str):
    """
    Leniwe rozwiązywanie symboli + cache w module globals().

    Pozwala importować np.:
        from agents.eda import EDAReporter
    bez natychmiastowego importu podmodułu, dopóki klasa nie zostanie naprawdę użyta.
    """
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        try:
            module: ModuleType = import_module(mod_name)
        except Exception as e:
            # Spójny i czytelny błąd — nie psuje działania helperów.
            raise AttributeError(
                f"Optional EDA component '{name}' not available "
                f"(failed to import {mod_name}): {e}"
            ) from e
        try:
            obj = getattr(module, symbol)
        except AttributeError as e:
            raise AttributeError(
                f"Optional EDA component '{name}' not available "
                f"(symbol '{symbol}' not found in {mod_name})"
            ) from e
        globals()[name] = obj  # cache w module
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


# === Light-weight helpers (działają bez dodatkowych modułów) ===

def memory_usage(df: pd.DataFrame) -> pd.Series:
    """
    Zwraca zużycie pamięci per kolumna (MB) + ewentualnie 'Index' + '__TOTAL__' jako ostatni wiersz.

    Zasady:
    - deep=True (dokładne liczenie stringów/kategorii),
    - nie modyfikuje wejściowego DF,
    - zawsze zwraca floatowe MB.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("memory_usage expects a pandas DataFrame")
    if df.empty:
        # Zwracamy spójny kształt: pusta seria + TOTAL=0.0
        return pd.Series({"__TOTAL__": 0.0}, dtype="float64")

    # memory_usage(DataFrame) zwraca per-kolumna i dodatkowo wpis 'Index'
    per_col = df.memory_usage(deep=True)
    # Ujednolicamy indeks na stringi (np. kolumny int/nazwa 'Index')
    per_col.index = per_col.index.astype(str)
    per_col_mb = (per_col / (1024 ** 2)).astype(float)

    total_mb = float(per_col_mb.sum())
    total_row = pd.Series({"__TOTAL__": total_mb}, dtype="float64")

    return pd.concat([per_col_mb, total_row])


def quick_overview(df: pd.DataFrame, max_cols: int = 50) -> dict:
    """
    Bardzo szybki przegląd ramki:
      - shape, dtypes, missing%, nunique
      - basic stats (dla kolumn numerycznych)
      - top-5 najczęstszych wartości (dla kolumn kategorycznych/object)
      - memory_mb (łącznie)

    Zwraca słownik gotowy do serializacji/wyświetlenia.
    Defensywnie — nie modyfikuje DF; działa na podzbiorze kolumn (max_cols).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Provide a non-empty pandas DataFrame."}

    # Ograniczenie liczby kolumn do szybkiego wglądu
    n_rows, n_cols = df.shape
    sample_cols = df.columns[: max(0, int(max_cols))]
    if len(sample_cols) == 0:
        return {"error": "No columns to profile (max_cols=0)."}

    # Dtypes, missing, nunique — liczone wyłącznie na sample_cols
    dsub = df[sample_cols]
    # .mean() na booleanach działa, ale traktujemy null-e, więc explicite:
    miss_pct = (dsub.isna().sum() / max(1, len(df)) * 100).round(3).to_dict()
    nunique = dsub.nunique(dropna=True).to_dict()
    dtypes = {str(c): str(t) for c, t in dsub.dtypes.items()}

    # Numeric stats (z defensywną obsługą, bez modyfikacji oryginału)
    num_cols = dsub.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        try:
            # describe jest szybkie; dodajemy percentyle 1/5/95/99
            num_stats_df = (
                dsub[num_cols]
                .describe(percentiles=[0.01, 0.05, 0.95, 0.99], include="all")  # numeric_only od Pandas 2 bywa wymagane
                .T.round(6)
            )
            # Czasem w describe pojawiają się NaN dla count<1 — zamieniamy na None dopiero w serializacji
            num_stats = {
                str(c): {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                for c, row in num_stats_df.to_dict(orient="index").items()
            }
        except Exception:
            # awaryjnie — minimalny zestaw statystyk
            num_stats = {}
            for c in num_cols:
                s = pd.to_numeric(dsub[c], errors="coerce")
                s = s.replace([np.inf, -np.inf], np.nan).dropna()
                if s.empty:
                    num_stats[str(c)] = {"count": 0}
                else:
                    num_stats[str(c)] = {
                        "count": int(s.count()),
                        "mean": float(s.mean()),
                        "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
                        "min": float(s.min()),
                        "25%": float(s.quantile(0.25)),
                        "50%": float(s.median()),
                        "75%": float(s.quantile(0.75)),
                        "max": float(s.max()),
                    }
    else:
        num_stats = {}

    # Categorical / object — top-5
    cat_cols = dsub.select_dtypes(include=["object", "category"]).columns
    cat_top = {}
    for c in cat_cols:
        try:
            vc = dsub[c].value_counts(dropna=False).head(5)
            cat_top[str(c)] = {str(k): int(v) for k, v in vc.items()}
        except Exception:
            cat_top[str(c)] = {}

    # Całkowite zużycie pamięci (MB) – wykorzystaj helpera (unikamy duplikacji logiki)
    try:
        mem_total_mb = float(memory_usage(df).get("__TOTAL__", 0.0))
    except Exception:
        mem_total_mb = float((df.memory_usage(deep=True).sum() / (1024 ** 2)))

    return {
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "preview_cols": list(map(str, sample_cols)),
        "dtypes": dtypes,
        "missing_pct": {str(k): float(v) for k, v in miss_pct.items()},
        "nunique": {str(k): int(v) for k, v in nunique.items()},
        "numeric_stats": num_stats,
        "categorical_top_values": cat_top,
        "memory_mb": mem_total_mb,
    }
