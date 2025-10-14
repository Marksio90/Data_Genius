"""
DataGenius PRO - EDA Page
Exploratory Data Analysis (PRO+++)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Bootstrapping ścieżek ===
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# === NAZWA_SEKCJI === Importy ekosystemu (UI + Core + Agents) ===
from frontend.app_layout import render_header, render_error, render_success, render_warning
from core.state_manager import get_state_manager
from agents.eda.eda_orchestrator import EDAOrchestrator

# Opcjonalny progress tracker (bez twardej zależności)
try:
    from src.frontend.progress_tracker import start_stage, advance, finish_stage, add_warning
    _HAS_PT = True
except Exception:
    _HAS_PT = False

# === NAZWA_SEKCJI === Page Config (bezpiecznie) ===
try:
    st.set_page_config(page_title="🔍 EDA — DataGenius PRO+++", page_icon="🔍", layout="wide")
except Exception:
    pass


# === NAZWA_SEKCJI === Nawigacja — kompatybilne przejście do stron ===
def _goto(file_hint: str, label: str = "") -> None:
    """
    Próbuj:
      1) st.page_link (link – multipage),
      2) st.switch_page (API jeśli dostępne),
      3) fallback — komunikat.
    """
    try:
        st.page_link(file_hint, label=label or file_hint)
        return
    except Exception:
        pass
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(file_hint)
            return
        except Exception:
            pass
    st.info(f"Przejdź ręcznie do: **{file_hint}** (menu po lewej).")


# === NAZWA_SEKCJI === Ustawienia wejścia i defensywne narzędzia ===
def _data_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    rows = int(len(df))
    cols = int(df.shape[1])
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2)) if rows and cols else 0.0
    null_ratio = float(df.isna().sum().sum() / max(1, rows * cols)) if rows and cols else 0.0
    return {"rows": rows, "cols": cols, "mem_mb": mem_mb, "null_ratio": null_ratio}


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        return d.get(key, default) if isinstance(d, dict) else default
    except Exception:
        return default


def _export_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# === NAZWA_SEKCJI === Główna strona EDA ===
def main() -> None:
    """Main EDA page"""
    render_header("🔍 Eksploracja Danych (EDA)", "Automatyczna analiza eksploracyjna danych")

    state_manager = get_state_manager()

    # 1) Walidacja obecności danych
    if not state_manager.has_data():
        render_warning("Najpierw załaduj dane w sekcji **📊 Data Upload**")
        if st.button("➡️ Przejdź do Upload", use_container_width=True):
            _goto("pages/02_📊_Data_Upload.py", label="📊 Data Upload")
        return

    df: pd.DataFrame = state_manager.get_data()
    target_column: Optional[str] = state_manager.get_target_column()

    # 2) KPI danych
    st.subheader("📋 Informacje o danych")
    k = _data_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 Wiersze", f"{k['rows']:,}")
    c2.metric("📊 Kolumny", f"{k['cols']:,}")
    c3.metric("💾 Rozmiar", f"{k['mem_mb']:.2f} MB")
    c4.metric("🎯 Target", target_column or "Nie wybrano")

    st.markdown("---")

    # 3) Ustawienia EDA (subset i sampling dla wydajności)
    with st.expander("⚙️ Ustawienia analizy (wydajność & zakres)", expanded=True):
        cols = df.columns.tolist()
        selected_cols = st.multiselect(
            "Wybierz kolumny do analizy (puste = wszystkie)",
            options=cols,
            default=[],
            help="Zawężenie kolumn przyspiesza analizę i ułatwia fokus na istotnych polach.",
        )
        sample_max = st.number_input(
            "Maksymalna liczba wierszy do analizy (sampling losowy)",
            min_value=1_000,
            max_value=max(10_000, len(df)),
            step=1_000,
            value=min(len(df), 100_000),
            help="Ograniczenie rozmiaru przyspiesza EDA. Pełny zbiór może znacząco wydłużyć analizę.",
        )
        seed = st.number_input("Losowość (random_state)", min_value=0, max_value=999_999, value=42, step=1)

    # 4) Przyciski akcji
    left, right = st.columns([2, 1])
    with left:
        run_btn = st.button("🚀 Rozpocznij analizę EDA", type="primary", use_container_width=True)
    with right:
        show_prev = st.toggle("Pokaż ostatnie wyniki (jeśli są)", value=True)

    # 5) Uruchom EDA
    if run_btn:
        run_eda_analysis(
            df=df,
            target_column=target_column,
            state_manager=state_manager,
            selected_cols=selected_cols,
            sample_max=int(sample_max),
            random_state=int(seed),
        )

    # 6) Prezentacja wyników (jeśli dostępne)
    if show_prev and state_manager.is_eda_complete():
        display_eda_results(state_manager.get_eda_results())


# === NAZWA_SEKCJI === Uruchomienie EDA (z progress trackerem) ===
def run_eda_analysis(
    df: pd.DataFrame,
    target_column: Optional[str],
    state_manager,
    selected_cols: List[str],
    sample_max: int,
    random_state: int,
) -> None:
    """Run EDA analysis (defensywnie + z opcjonalnym progresem)."""
    # Przedsiewzięcia: subset & sampling
    work_df = df
    if selected_cols:
        keep = [c for c in selected_cols if c in work_df.columns]
        if target_column and target_column not in keep and target_column in work_df.columns:
            keep.append(target_column)
        if not keep:
            render_error("Brak kolumn do analizy", "Zaznacz przynajmniej jedną kolumnę.")
            return
        work_df = work_df[keep]
    if len(work_df) > sample_max:
        work_df = work_df.sample(sample_max, random_state=random_state)

    with st.spinner("⏳ Analizuję dane..."):
        try:
            if _HAS_PT:
                start_stage("EDA", total_steps=3)
                advance(note="Przygotowanie danych")

            # Orkiestracja EDA
            eda = EDAOrchestrator()
            if _HAS_PT:
                advance(note="Uruchomienie EDA")

            result = eda.run(data=work_df, target_column=target_column)

            if hasattr(result, "is_success") and result.is_success():
                state_manager.set_eda_results(result.data)
                render_success("✅ Analiza EDA zakończona pomyślnie!")
                if _HAS_PT:
                    advance(note="Zapis wyników")
                    finish_stage()
                st.rerun()
            else:
                errors = []
                if hasattr(result, "errors"):
                    try:
                        errors = result.errors
                    except Exception:
                        errors = []
                msg = "; ".join(errors) if errors else "Nieznany błąd"
                if _HAS_PT:
                    try:
                        add_warning(msg)
                        finish_stage(status="failed")
                    except Exception:
                        pass
                render_error("EDA nie powiodło się", msg)

        except Exception as e:
            if _HAS_PT:
                try:
                    add_warning(str(e))
                    finish_stage(status="failed")
                except Exception:
                    pass
            render_error("Błąd podczas analizy EDA", str(e))


# === NAZWA_SEKCJI === Render wyników EDA ===
def display_eda_results(eda_results: Dict[str, Any]) -> None:
    """Display EDA results (defensywnie, bez założeń o strukturze)."""
    st.markdown("---")
    st.subheader("📊 Wyniki Analizy EDA")

    # Eksport wyników
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "💾 Pobierz wyniki EDA (JSON)",
            data=_export_json(eda_results),
            file_name="eda_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp2:
        st.caption("Wynik to słownik agregujący moduły EDA: StatisticalAnalyzer, MissingDataAnalyzer, OutlierDetector, CorrelationAnalyzer, VisualizationEngine.")

    # Karty
    tabs = st.tabs(["📈 Statystyki", "🔍 Braki danych", "⚠️ Outliers", "🔗 Korelacje", "📉 Wizualizacje"])

    # Tab 1: Statistik
    with tabs[0]:
        display_statistics(eda_results)

    # Tab 2: Missing
    with tabs[1]:
        display_missing_data(eda_results)

    # Tab 3: Outliers
    with tabs[2]:
        display_outliers(eda_results)

    # Tab 4: Correlations
    with tabs[3]:
        display_correlations(eda_results)

    # Tab 5: Visualizations
    with tabs[4]:
        display_visualizations(eda_results)


# === NAZWA_SEKCJI === Sekcja: Statystyki ===
def display_statistics(eda_results: Dict[str, Any]) -> None:
    st.markdown("### 📊 Analiza Statystyczna")

    eda = eda_results.get("eda_results", {})
    stats = eda.get("StatisticalAnalyzer")
    if not stats:
        st.info("Brak danych statystycznych")
        return

    overall = stats.get("overall")
    if overall:
        st.markdown("#### Ogólne statystyki")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cechy numeryczne", overall.get("n_numeric", 0))
        col2.metric("Cechy kategoryczne", overall.get("n_categorical", 0))
        col3.metric("Rozmiar (MB)", f"{overall.get('memory_mb', 0):.2f}")

    num_features = _safe_get(stats, "numeric_features", {})
    if isinstance(num_features, dict) and num_features.get("features"):
        st.markdown("#### Cechy numeryczne")
        df = pd.DataFrame(num_features["features"]).T
        st.dataframe(df, use_container_width=True, height=min(500, 26 * (len(df) + 2)))
    else:
        st.info("Brak cech numerycznych")


# === NAZWA_SEKCJI === Sekcja: Braki danych ===
def display_missing_data(eda_results: Dict[str, Any]) -> None:
    st.markdown("### 🔍 Analiza Brakujących Danych")

    eda = eda_results.get("eda_results", {})
    miss = eda.get("MissingDataAnalyzer")
    if not miss:
        st.info("Brak analizy brakujących danych")
        return

    summary = miss.get("summary", {})
    if summary.get("total_missing", 0) == 0:
        st.success("🎉 Brak brakujących danych w zbiorze!")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Braki ogółem", f"{summary.get('total_missing', 0):,}")
    c2.metric("Procent braków", f"{summary.get('missing_percentage', 0):.2f}%")
    c3.metric("Kolumny z brakami", summary.get("n_columns_with_missing", 0))

    columns_missing = miss.get("columns", [])
    if columns_missing:
        st.markdown("#### Kolumny z brakującymi danymi")
        df = pd.DataFrame(columns_missing)
        st.dataframe(df, use_container_width=True, height=min(500, 26 * (len(df) + 2)))

    recs = miss.get("recommendations", [])
    if recs:
        st.markdown("#### 💡 Rekomendacje")
        for r in recs:
            st.info(r)


# === NAZWA_SEKCJI === Sekcja: Outliers ===
def display_outliers(eda_results: Dict[str, Any]) -> None:
    st.markdown("### ⚠️ Analiza Outliers")

    eda = eda_results.get("eda_results", {})
    out = eda.get("OutlierDetector")
    if not out:
        st.info("Brak analizy outliers")
        return

    summary = out.get("summary", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Outliers ogółem", summary.get("total_outliers", 0))
    c2.metric("Kolumny z outliers", summary.get("n_columns_with_outliers", 0))
    methods = ", ".join(summary.get("methods_used", []))
    c3.metric("Metody", methods[:20] + "..." if len(methods) > 20 else methods)

    iqr = out.get("iqr_method", {})
    columns_with_outliers = iqr.get("columns", {})
    if columns_with_outliers:
        st.markdown("#### Outliers (metoda IQR)")
        rows: List[Dict[str, Any]] = []
        for col, info in columns_with_outliers.items():
            rows.append({
                "Kolumna": col,
                "Liczba outliers": info.get("n_outliers", 0),
                "Procent": f"{info.get('percentage', 0.0):.2f}%",
                "Dolna granica": f"{info.get('lower_bound', float('nan')):.2f}",
                "Górna granica": f"{info.get('upper_bound', float('nan')):.2f}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=min(500, 26 * (len(df) + 2)))


# === NAZWA_SEKCJI === Sekcja: Korelacje ===
def display_correlations(eda_results: Dict[str, Any]) -> None:
    st.markdown("### 🔗 Analiza Korelacji")

    eda = eda_results.get("eda_results", {})
    corr = eda.get("CorrelationAnalyzer")
    if not corr:
        st.info("Brak analizy korelacji")
        return

    num_corr = corr.get("numeric_correlations", {})
    if num_corr and num_corr.get("correlation_matrix"):
        st.markdown("#### Macierz korelacji")
        try:
            corr_matrix = pd.DataFrame(num_corr["correlation_matrix"])
            st.dataframe(
                corr_matrix.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                use_container_width=True,
                height=min(600, 28 * (len(corr_matrix) + 2)),
            )
        except Exception:
            st.dataframe(pd.DataFrame(num_corr["correlation_matrix"]), use_container_width=True)

    high_corr = corr.get("high_correlations", [])
    if high_corr:
        st.markdown("#### ⚠️ Silne korelacje (|r| > 0.8)")
        for pair in high_corr[:10]:
            try:
                st.warning(f"**{pair['feature1']}** ↔ **{pair['feature2']}**: r = {pair['correlation']:.3f}")
            except Exception:
                continue

    recs = corr.get("recommendations", [])
    if recs:
        st.markdown("#### 💡 Rekomendacje")
        for r in recs:
            st.info(r)


# === NAZWA_SEKCJI === Sekcja: Wizualizacje ===
def display_visualizations(eda_results: Dict[str, Any]) -> None:
    st.markdown("### 📉 Wizualizacje")

    eda = eda_results.get("eda_results", {})
    viz = eda.get("VisualizationEngine")
    if not viz:
        st.info("Brak wizualizacji")
        return

    visualizations = viz.get("visualizations", {})
    if not visualizations:
        st.info("Brak dostępnych wizualizacji")
        return

    if "missing_data" in visualizations:
        st.markdown("#### Brakujące wartości")
        st.plotly_chart(visualizations["missing_data"], use_container_width=True)

    if "correlation_heatmap" in visualizations:
        st.markdown("#### Mapa korelacji")
        st.plotly_chart(visualizations["correlation_heatmap"], use_container_width=True)

    if "distributions" in visualizations:
        st.markdown("#### Rozkłady cech numerycznych")
        dists = visualizations["distributions"]
        if isinstance(dists, list):
            for fig in dists[:5]:
                st.plotly_chart(fig, use_container_width=True)

    st.info("💡 Więcej wizualizacji dostępnych w pełnym raporcie")


# === NAZWA_SEKCJI === Wejście modułu ===
if __name__ == "__main__":
    main()
