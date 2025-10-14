"""
DataGenius PRO - Data Upload Page
Upload and preview data (PRO+++)
"""

from __future__ import annotations

import io
import os
import sys
import time
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Bootstrapping ścieżek ===
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# === NAZWA_SEKCJI === Importy ekosystemu + fallback UI ===
try:
    from frontend.app_layout import render_header, render_error, render_success
except Exception:
    def render_header(title: str, subtitle: str = "") -> None:
        st.header(title)
        if subtitle:
            st.caption(subtitle)
    def render_error(title: str, detail: Optional[str] = None) -> None:
        st.error(title + (f": {detail}" if detail else ""))
    def render_success(msg: str) -> None:
        st.success(msg)

# Core / Agents
from core.state_manager import get_state_manager
from core.data_loader import get_data_loader
from core.data_validator import DataValidator
from agents.data_understanding.schema_analyzer import SchemaAnalyzer
from agents.data_understanding.target_detector import TargetDetector
from agents.data_understanding.problem_classifier import ProblemClassifier

# Konfiguracja (z bezpiecznym fallbackiem)
try:
    from config.constants import SAMPLE_DATASETS, MAX_PREVIEW_ROWS
except Exception:
    SAMPLE_DATASETS = {
        "titanic": dict(name="Titanic", description="Dane pasażerów", samples=891, features=12, problem_type="classification"),
        "housing": dict(name="Housing", description="Ceny domów", samples=20640, features=9, problem_type="regression"),
        "iris": dict(name="Iris", description="Gatunki irysów", samples=150, features=4, problem_type="classification"),
    }
    MAX_PREVIEW_ROWS = 100

# Opcjonalna integracja z progress trackerem
try:
    from src.frontend.progress_tracker import start_stage, advance, finish_stage, add_warning
    _HAS_PT = True
except Exception:
    _HAS_PT = False


# === NAZWA_SEKCJI === Ustawienia strony ===
try:
    st.set_page_config(page_title="📤 Upload Data — DataGenius PRO+++", page_icon="📤", layout="wide")
except Exception:
    pass


# === NAZWA_SEKCJI === Walidacje i limity ===
_ALLOWED_EXT = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}
_MAX_FILE_MB = int(os.environ.get("DG_UPLOAD_MAX_MB", "250"))  # domyślnie 250MB
_MAX_ROWS_HARD = int(os.environ.get("DG_UPLOAD_MAX_ROWS", "3000000"))  # twardy limit bezpieczeństwa

def _validate_file_input(filename: str, size_bytes: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise ValueError(f"Nieobsługiwany format: {ext}. Obsługiwane: {', '.join(sorted(_ALLOWED_EXT))}")
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > _MAX_FILE_MB:
        raise ValueError(f"Plik jest zbyt duży: {size_mb:.1f} MB > {_MAX_FILE_MB} MB. Zwiększ limit lub podziel plik.")

def _safe_temp_copy(uploaded) -> Path:
    """Bezpieczny zapis uploadu do pliku tymczasowego (z zachowaniem rozszerzenia)."""
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(prefix="dg_upload_", suffix=suffix, delete=False)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# === NAZWA_SEKCJI === Cache: ładowanie próbek ===
@st.cache_data(show_spinner=True, ttl=3600)
def _load_sample_cached(dataset_id: str) -> pd.DataFrame:
    loader = get_data_loader()
    return loader.load_sample(dataset_id)


# === NAZWA_SEKCJI === Podgląd: pomocnicze KPI ===
def _quick_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    rows = int(len(df))
    cols = int(df.shape[1])
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2)) if rows and cols else 0.0
    null_ratio = float(df.isna().sum().sum() / max(1, rows * cols)) if rows and cols else 0.0
    return {"rows": rows, "cols": cols, "mem_mb": mem_mb, "null_ratio": null_ratio}


# === NAZWA_SEKCJI === Główne UI ===
def main() -> None:
    """Main page function"""
    render_header("📊 Załaduj Dane", "Załaduj swoje dane lub użyj przykładowego zbioru danych")

    state_manager = get_state_manager()
    data_loader = get_data_loader()

    # 1) Źródło
    st.subheader("1️⃣ Wybierz źródło danych")
    data_source = st.radio("Źródło danych:", ["📤 Upload pliku", "📚 Przykładowe dane"], horizontal=True)

    df: Optional[pd.DataFrame] = None

    if data_source == "📤 Upload pliku":
        df = handle_file_upload(data_loader)
    else:
        df = handle_sample_data()

    # 2) Podgląd + analiza
    if df is not None and not df.empty:
        st.markdown("---")
        st.subheader("2️⃣ Podgląd danych")

        show_data_preview(df)

        # Zatwierdzenie
        if st.button("✅ Zatwierdź i kontynuuj", type="primary", use_container_width=True):
            # Persist do state_manager
            state_manager.set_data(df)
            render_success("Dane załadowane pomyślnie!")

            # Integracja z modułami współistniejącymi: zapisz także do st.session_state["raw_df"]
            st.session_state["raw_df"] = df

            # Auto analiza
            with st.spinner("Analizuję dane (schema → target → problem)…"):
                analyze_data(df, state_manager)

            st.info("👉 Przejdź do **🔍 EDA** aby zobaczyć analizę danych!")
    else:
        if df is not None and df.empty:
            render_error("Plik został wczytany, ale nie zawiera danych.")


# === NAZWA_SEKCJI === Obsługa uploadu pliku ===
def handle_file_upload(data_loader) -> Optional[pd.DataFrame]:
    """Handle file upload z walidacją i bezpiecznym zapisem do tempfile."""
    uploaded = st.file_uploader(
        "Przeciągnij plik lub kliknij, aby wybrać",
        type=[e.lstrip(".") for e in sorted(_ALLOWED_EXT)],
        help=f"Obsługiwane formaty: {', '.join(sorted(_ALLOWED_EXT))}",
    )

    if not uploaded:
        return None

    try:
        _validate_file_input(uploaded.name, uploaded.size)

        if _HAS_PT:
            start_stage("Upload", total_steps=3)
            advance(note="Walidacja pliku")

        # Zapis tymczasowy + wczytanie przez data_loader (centralny punkt I/O)
        with st.spinner("Ładuję dane…"):
            tmp_path = _safe_temp_copy(uploaded)
            if _HAS_PT:
                advance(note="Kopia do pliku tymczasowego")

            df = data_loader.load(tmp_path)

            # Twardy limit bezpieczeństwa
            if len(df) > _MAX_ROWS_HARD:
                raise ValueError(
                    f"Przekroczono twardy limit wierszy ({len(df):,} > {_MAX_ROWS_HARD:,}). "
                    "Zastosuj próbkowanie lub podział pliku."
                )

            # Sprzątanie tempfile
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        render_success(f"Załadowano {len(df):,} wierszy i {len(df.columns)} kolumn")
        if _HAS_PT:
            finish_stage()

        return df

    except Exception as e:
        if _HAS_PT:
            try:
                add_warning(str(e))
                finish_stage(status="failed")
            except Exception:
                pass
        render_error("Nie udało się załadować pliku", str(e))
        return None


# === NAZWA_SEKCJI === Przykładowe dane ===
def handle_sample_data() -> Optional[pd.DataFrame]:
    """Wybor i wczytanie próbki danych (cache)."""
    st.markdown("### Wybierz przykładowy zbiór danych:")

    cols = st.columns(3)
    chosen_id: Optional[str] = None

    for i, (dataset_id, info) in enumerate(SAMPLE_DATASETS.items()):
        col = cols[i % 3]
        with col:
            st.markdown(f"#### {info['name']}")
            st.caption(info.get("description", ""))
            st.caption(f"📊 {info.get('samples', 'n/d')} próbek")
            st.caption(f"🔢 {info.get('features', 'n/d')} cech")
            st.caption(f"🎯 Problem: {info.get('problem_type', 'n/d')}")
            if st.button(f"Załaduj {info['name']}", key=f"load_{dataset_id}"):
                chosen_id = dataset_id

    # Jeśli nic nie kliknięto, ale wcześniej wybrano
    if not chosen_id and "selected_sample" in st.session_state:
        chosen_id = st.session_state["selected_sample"]

    if not chosen_id:
        return None

    try:
        with st.spinner(f"Ładuję {SAMPLE_DATASETS[chosen_id]['name']}…"):
            df = _load_sample_cached(chosen_id)
            st.session_state["selected_sample"] = chosen_id
            render_success(f"Załadowano {SAMPLE_DATASETS[chosen_id]['name']}")
            return df
    except Exception as e:
        render_error(f"Nie udało się załadować {SAMPLE_DATASETS[chosen_id]['name']}", str(e))
        return None


# === NAZWA_SEKCJI === Podgląd danych ===
def show_data_preview(df: pd.DataFrame) -> None:
    """KPI + head + słownik kolumn (lite)."""
    kpis = _quick_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 Wiersze", f"{kpis['rows']:,}")
    c2.metric("📊 Kolumny", f"{kpis['cols']:,}")
    c3.metric("💾 Rozmiar", f"{kpis['mem_mb']:.2f} MB")
    c4.metric("🔍 Braki", f"{kpis['null_ratio']*100:.1f}%")

    st.markdown("---")
    st.markdown("### 👀 Podgląd danych")

    max_slider = min(MAX_PREVIEW_ROWS, max(5, len(df)))
    n_preview = st.slider(
        "Liczba wierszy do wyświetlenia:",
        min_value=5,
        max_value=max_slider,
        value=min(10, max_slider),
    )
    st.dataframe(df.head(n_preview), use_container_width=True)

    with st.expander("📋 Informacje o kolumnach"):
        # Szybkie statystyki kolumn
        rows: List[Dict[str, Any]] = []
        for col in df.columns:
            s = df[col]
            rows.append({
                "Kolumna": str(col),
                "Typ": str(s.dtype),
                "Unikalne": int(s.nunique(dropna=True)),
                "Braki": int(s.isna().sum()),
                "Braki %": f"{(float(s.isna().mean())*100):.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=min(400, 26 * (len(rows) + 2)))


# === NAZWA_SEKCJI === Auto-analiza danych ===
def analyze_data(df: pd.DataFrame, state_manager) -> None:
    """Schema → Target → Problem (fallback). Zapis do state_manager + session_state."""
    try:
        # 1) Schema
        schema_analyzer = SchemaAnalyzer()
        schema_result = schema_analyzer.run(data=df)
        if not getattr(schema_result, "is_success", lambda: False)():
            st.warning("Nie udało się przeanalizować schematu danych.")
            return

        column_info = schema_result.data.get("columns", [])

        # 2) Target
        target_detector = TargetDetector()
        target_result = target_detector.run(data=df, column_info=column_info)

        target_col = None
        problem_type = None

        if getattr(target_result, "is_success", lambda: False)() and target_result.data.get("target_column"):
            target_col = target_result.data["target_column"]
            problem_type = target_result.data.get("problem_type")
        else:
            # Fallback: ProblemClassifier bez targetu — heurystyka podpowie typ
            try:
                pc = ProblemClassifier()
                pcr = pc.run(data=df, column_info=column_info)
                if getattr(pcr, "is_success", lambda: False)():
                    problem_type = pcr.data.get("problem_type")
            except Exception:
                pass

        # 3) Persist + komunikaty
        if target_col:
            state_manager.set_target_column(target_col)
            st.session_state["target_column"] = target_col
            st.success(f"🎯 Wykryto kolumnę docelową: **{target_col}**")
        else:
            st.info("💡 Nie wykryto jednoznacznego targetu. Wybierz ręcznie w następnym kroku.")

        if problem_type:
            state_manager.set_problem_type(problem_type)
            st.session_state["problem_type"] = problem_type
            st.info(f"📋 Typ problemu: **{problem_type}**")

    except Exception as e:
        st.warning(f"Automatyczna analiza nie powiodła się: {e}")


# === NAZWA_SEKCJI === Wejście modułu ===
if __name__ == "__main__":
    main()
