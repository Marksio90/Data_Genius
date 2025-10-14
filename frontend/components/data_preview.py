# src/frontend/data_preview.py
# === OPIS MODUŁU ===
# Moduł podglądu danych PRO+++ dla Streamlit:
# - Upload/źródło z session_state
# - Walidacja i bezpieczne wczytywanie (CSV, Parquet, Excel, JSON/JSONL)
# - Profil "lite": dtypes, missing, unikalność, statystyki, heurystyki ID/datetime/category
# - Prezentacja tabeli i metryk, szybkie wykresy (opcjonalnie)
# - Eksport próbek i słownika danych
# - Cache i defensywna obsługa błędów

from __future__ import annotations

import io
import os
import json
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Logger (zgodny z Twoim ekosystemem) ===
try:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("data_preview")

# === NAZWA_SEKCJI === Konfiguracja domyślna (możesz nadpisać przez argumenty) ===

DEFAULT_MAX_MB: int = 200        # maks. dopuszczalny rozmiar pojedynczego pliku (MB)
DEFAULT_MAX_ROWS: int = 2_000_000  # limit bezpieczeństwa wierszy przy CSV/JSONL
DEFAULT_HEAD_ROWS: int = 1000    # ile wierszy pokazać w widoku
DEFAULT_SAMPLE_ROWS: int = 2000  # ile wierszy w eksporcie próbki
SUPPORTED_EXTS = {".csv", ".tsv", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".jsonl"}

# === NAZWA_SEKCJI === Dataclasses wyników i metryk ===

@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    nulls: int
    null_ratio: float
    nunique: int
    sample_values: List[Any]
    is_id_like: bool
    is_datetime_like: bool
    is_categorical_suggested: bool

@dataclass
class DataPreviewResult:
    n_rows: int
    n_cols: int
    mem_usage_mb: float
    duplicate_columns: List[str]
    empty_name_columns: List[str]
    columns: List[ColumnSummary]
    warnings: List[str]

# === NAZWA_SEKCJI === Walidacja wejścia ===

def _validate_file(name: str, size_bytes: int, max_mb: int) -> None:
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Nieobsługiwane rozszerzenie: {ext}. Obsługiwane: {', '.join(sorted(SUPPORTED_EXTS))}")
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"Plik jest zbyt duży ({size_mb:.1f} MB > {max_mb} MB). Zwiększ limit lub podziel dane.")

def _normalize_colnames(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    duplicate_cols = [c for c in df.columns if list(df.columns).count(c) > 1]
    empty_cols = [c for c in df.columns if (c is None) or (str(c).strip() == "")]
    # Zamień puste nazwy na 'col_i'
    if empty_cols:
        new_cols = []
        auto_id = 0
        for c in df.columns:
            if (c is None) or (str(c).strip() == ""):
                new_cols.append(f"col_{auto_id}")
                auto_id += 1
            else:
                new_cols.append(str(c))
        df.columns = new_cols
    else:
        df.columns = [str(c) for c in df.columns]
    return df, list(sorted(set(duplicate_cols))), empty_cols

# === NAZWA_SEKCJI === Wczytywanie danych (cache + defensywnie) ===

@st.cache_data(show_spinner=False, ttl=3600)
def _read_csv_cached(content: bytes, sep: str, nrows_limit: Optional[int]) -> pd.DataFrame:
    # Próba z pyarrow -> fallback python engine
    try:
        return pd.read_csv(
            io.BytesIO(content),
            sep=sep,
            engine="pyarrow",  # szybciej i stabilniej dla UTF-8/BOM
            nrows=nrows_limit,
            dtype_backend="pyarrow",
        )
    except Exception:
        return pd.read_csv(
            io.BytesIO(content),
            sep=sep,
            engine="python",
            nrows=nrows_limit,
        )

@st.cache_data(show_spinner=False, ttl=3600)
def _read_parquet_cached(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(content))
    except Exception:
        # fallback gdy brak pyarrow/fastparquet skonfigurowanego poprawnie
        return pd.read_parquet(io.BytesIO(content), engine="auto")

@st.cache_data(show_spinner=False, ttl=3600)
def _read_excel_cached(content: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(content), engine="openpyxl")

@st.cache_data(show_spinner=False, ttl=3600)
def _read_json_cached(content: bytes, lines: bool, nrows_limit: Optional[int]) -> pd.DataFrame:
    if lines:
        # JSON Lines — wczytaj ograniczoną liczbę linii
        decoded = io.StringIO(io.BytesIO(content).read().decode("utf-8", errors="ignore"))
        rows = []
        for i, line in enumerate(decoded):
            if nrows_limit and i >= nrows_limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return pd.DataFrame(rows)
    else:
        return pd.read_json(io.BytesIO(content))

def _read_file(name: str, content: bytes, *, max_rows: int) -> pd.DataFrame:
    ext = os.path.splitext(name)[1].lower()
    if ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        return _read_csv_cached(content, sep=sep, nrows_limit=max_rows)
    if ext in {".parquet", ".pq"}:
        return _read_parquet_cached(content)
    if ext in {".xlsx", ".xls"}:
        return _read_excel_cached(content)
    if ext in {".json"}:
        return _read_json_cached(content, lines=False, nrows_limit=max_rows)
    if ext in {".jsonl"}:
        return _read_json_cached(content, lines=True, nrows_limit=max_rows)
    raise ValueError(f"Nieobsługiwany format: {ext}")

# === NAZWA_SEKCJI === Profil "lite" i heurystyki ===

def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    try:
        pd.to_datetime(series.dropna().sample(min(50, len(series.dropna()))), errors="coerce")
        return True
    except Exception:
        return False

def _is_id_like(series: pd.Series) -> bool:
    # Kandydat ID: niski udział nulli, wysoka unikalność i brak wartości ujemnych w int/str bez spacji
    nunique = series.nunique(dropna=True)
    if nunique == 0:
        return False
    ratio = nunique / max(1, len(series))
    if ratio < 0.9:
        return False
    s = series.dropna()
    if s.empty:
        return False
    if pd.api.types.is_integer_dtype(s) and (s.min() >= 0):
        return True
    if pd.api.types.is_string_dtype(s) and (s.str.contains(r"\s", regex=True).mean() < 0.05):
        return True
    return False

def _is_categorical_suggested(series: pd.Series, max_unique: int = 30, max_ratio: float = 0.2) -> bool:
    nunique = series.nunique(dropna=True)
    return (nunique <= max_unique) and (nunique / max(1, len(series)) <= max_ratio)

def _profile_dataframe(df: pd.DataFrame) -> DataPreviewResult:
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))
    duplicate_cols = [c for c in df.columns if list(df.columns).count(c) > 1]
    empty_cols = [c for c in df.columns if (c is None) or (str(c).strip() == "")]
    warnings_list: List[str] = []

    cols: List[ColumnSummary] = []
    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        non_null = int(s.notna().sum())
        nulls = int(s.isna().sum())
        null_ratio = float(nulls / max(1, len(s)))
        nunique = int(s.nunique(dropna=True))
        sample_vals = list(s.dropna().unique()[:5])
        is_dt = _is_datetime_like(s)
        is_id = _is_id_like(s)
        is_cat = _is_categorical_suggested(s)
        cols.append(
            ColumnSummary(
                name=str(col),
                dtype=dtype_str,
                non_null=non_null,
                nulls=nulls,
                null_ratio=round(null_ratio, 4),
                nunique=nunique,
                sample_values=sample_vals,
                is_id_like=is_id,
                is_datetime_like=is_dt,
                is_categorical_suggested=is_cat,
            )
        )

    if any(c.null_ratio > 0.5 for c in cols):
        warnings_list.append("Co najmniej jedna kolumna ma >50% braków.")
    if len(df.columns) != len(set(df.columns)):
        warnings_list.append("Wykryto zduplikowane nazwy kolumn — zalecana normalizacja.")
    if df.empty:
        warnings_list.append("Zbiór jest pusty.")

    return DataPreviewResult(
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        mem_usage_mb=round(mem_mb, 2),
        duplicate_columns=sorted(list(set(duplicate_cols))),
        empty_name_columns=empty_cols,
        columns=cols,
        warnings=warnings_list,
    )

# === NAZWA_SEKCJI === Eksporty: head.csv, schema.json, data_dictionary.csv ===

def _export_head(df: pd.DataFrame, n: int) -> bytes:
    return df.head(n).to_csv(index=False).encode("utf-8")

def _export_schema(result: DataPreviewResult) -> bytes:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2).encode("utf-8")

def _export_data_dictionary(result: DataPreviewResult) -> bytes:
    rows = []
    for c in result.columns:
        rows.append({
            "column": c.name,
            "dtype": c.dtype,
            "non_null": c.non_null,
            "nulls": c.nulls,
            "null_ratio": c.null_ratio,
            "nunique": c.nunique,
            "is_id_like": c.is_id_like,
            "is_datetime_like": c.is_datetime_like,
            "is_categorical_suggested": c.is_categorical_suggested,
            "sample_values": "; ".join(map(str, c.sample_values)),
        })
    dd = pd.DataFrame(rows)
    return dd.to_csv(index=False).encode("utf-8")

# === NAZWA_SEKCJI === UI: główna funkcja renderująca ===

def render_data_preview(
    *,
    title: str = "🧭 Data Preview — PRO+++",
    max_mb: int = DEFAULT_MAX_MB,
    max_rows: int = DEFAULT_MAX_ROWS,
    head_rows: int = DEFAULT_HEAD_ROWS,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> None:
    """
    Główne UI podglądu danych:
    - upload,
    - wczytanie i walidacja,
    - profil 'lite',
    - prezentacja + eksporty.
    Efekty:
      st.session_state["raw_df"] -> pd.DataFrame
      st.session_state["data_meta"] -> dict
    """
    st.header(title)
    st.caption("Bezpieczny podgląd danych z walidacją, heurystykami i eksportami (CSV/Parquet/Excel/JSON/JSONL).")

    # ŹRÓDŁO: istniejący DF w sesji lub upload
    source_tab, upload_tab = st.tabs(["📦 Z sesji", "📤 Upload pliku"])
    df: Optional[pd.DataFrame] = None
    file_meta: Dict[str, Any] = {}

    with source_tab:
        if "raw_df" in st.session_state and isinstance(st.session_state["raw_df"], pd.DataFrame):
            st.success("Znaleziono istniejący DataFrame w sesji.", icon="✅")
            if st.toggle("Użyj danych z sesji", value=True):
                df = st.session_state["raw_df"]
        else:
            st.info("Brak danych w sesji. Przejdź do zakładki 'Upload pliku'.", icon="ℹ️")

    with upload_tab:
        file = st.file_uploader(
            "Wybierz plik danych",
            type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTS)],
            accept_multiple_files=False,
            help=f"Maks. rozmiar pliku: ~{max_mb} MB (CSV/TSV/Parquet/Excel/JSON/JSONL).",
        )
        if file is not None and df is None:
            try:
                _validate_file(file.name, file.size, max_mb=max_mb)
                content = file.read()
                start = time.time()
                df = _read_file(file.name, content, max_rows=max_rows)
                elapsed = time.time() - start
                df, dup_cols, empty_cols = _normalize_colnames(df)
                file_meta = {
                    "filename": file.name,
                    "size_mb": round(file.size / (1024 * 1024), 2),
                    "read_seconds": round(elapsed, 3),
                    "duplicate_cols": dup_cols,
                    "empty_name_cols": empty_cols,
                }
                st.success(f"Wczytano plik {file.name} w {elapsed:.2f}s", icon="✅")
            except Exception as e:
                log.exception("Błąd wczytywania pliku.")
                st.error(f"Nie udało się wczytać pliku: {e}")
                return

    if df is None:
        st.stop()

    # Profil "lite"
    try:
        profile = _profile_dataframe(df)
    except Exception as e:
        log.exception("Błąd profilowania danych.")
        st.error(f"Nie udało się przygotować profilu danych: {e}")
        return

    # Metadane do sesji (dla dalszych kroków)
    st.session_state["raw_df"] = df
    st.session_state["data_meta"] = {
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "mem_usage_mb": profile.mem_usage_mb,
        "file_meta": file_meta,
        "warnings": profile.warnings,
    }

    # === NAZWA_SEKCJI === Podsumowanie ogólne
    st.subheader("📊 Podsumowanie")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wiersze", f"{profile.n_rows:,}")
    c2.metric("Kolumny", f"{profile.n_cols:,}")
    c3.metric("Pamięć", f"{profile.mem_usage_mb:.2f} MB")
    c4.metric("Ostrzeżenia", f"{len(profile.warnings)}")

    if profile.warnings:
        with st.expander("⚠️ Ostrzeżenia i zalecenia", expanded=True):
            for w in profile.warnings:
                st.warning(w)

    if file_meta:
        with st.expander("ℹ️ Metadane pliku"):
            st.json(file_meta)

    # === NAZWA_SEKCJI === Słownik danych (lite)
    st.subheader("📚 Słownik danych (lite)")
    dict_rows = []
    for c in profile.columns:
        dict_rows.append({
            "column": c.name,
            "dtype": c.dtype,
            "non_null": c.non_null,
            "nulls": c.nulls,
            "null_ratio": c.null_ratio,
            "nunique": c.nunique,
            "is_id_like": c.is_id_like,
            "is_datetime_like": c.is_datetime_like,
            "is_categorical_suggested": c.is_categorical_suggested,
            "sample_values": "; ".join(map(str, c.sample_values)),
        })
    dict_df = pd.DataFrame(dict_rows)
    st.dataframe(dict_df, use_container_width=True, height=300)

    # === NAZWA_SEKCJI === Podgląd tabeli
    st.subheader("👀 Podgląd danych")
    max_head = int(st.slider("Ile wierszy pokazać", 10, min(10_000, max(profile.n_rows, 10)), value=min(1000, profile.n_rows)))
    st.dataframe(df.head(max_head), use_container_width=True, height=400)

    # === NAZWA_SEKCJI === Szybkie wykresy (opcjonalnie)
    with st.expander("📈 Szybkie wykresy", expanded=False):
        num_cols = [c.name for c in profile.columns if pd.api.types.is_numeric_dtype(df[c.name])]
        cat_cols = [c.name for c in profile.columns if (not pd.api.types.is_numeric_dtype(df[c.name])) and (df[c.name].nunique() <= 30)]
        left, right = st.columns(2)
        with left:
            if num_cols:
                coln = st.selectbox("Kolumna numeryczna (histogram)", options=num_cols, index=0)
                st.bar_chart(df[coln].dropna().value_counts().head(50))
            else:
                st.info("Brak oczywistych kolumn numerycznych do histogramu.")
        with right:
            if cat_cols:
                colc = st.selectbox("Kolumna kategoryczna (Top 30)", options=cat_cols, index=0)
                st.bar_chart(df[colc].astype(str).value_counts().head(30))
            else:
                st.info("Brak małokardynalnych kolumn kategorycznych.")

    # === NAZWA_SEKCJI === Eksporty
    st.subheader("💾 Eksport")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            label=f"Pobierz head({min(sample_rows, profile.n_rows)}) CSV",
            data=_export_head(df, min(sample_rows, profile.n_rows)),
            file_name="sample_head.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            label="Pobierz schema.json",
            data=_export_schema(profile),
            file_name="schema.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_c:
        st.download_button(
            label="Pobierz data_dictionary.csv",
            data=_export_data_dictionary(profile),
            file_name="data_dictionary.csv",
            mime="text/csv",
            use_container_width=True,
        )

# === NAZWA_SEKCJI === Lokalny punkt wejścia (opcjonalny)
if __name__ == "__main__":
    # Uruchom: streamlit run src/frontend/data_preview.py
    render_data_preview()
