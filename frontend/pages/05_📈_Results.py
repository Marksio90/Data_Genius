# 05_📈_Results.py
"""
DataGenius PRO — Results Hub (PRO+++)
Zbiorcze podsumowanie: dane, EDA, model, metryki, FI, eksport snapshotu
"""

from __future__ import annotations

import io
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Bootstrapping ścieżek ===
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# === NAZWA_SEKCJI === Importy ekosystemu (UI + Core) ===
from frontend.app_layout import render_header, render_error, render_success, render_warning
from core.state_manager import get_state_manager

# Opcjonalny progress tracker (bez twardej zależności)
try:
    from src.frontend.progress_tracker import start_stage, advance, finish_stage, add_warning
    _HAS_PT = True
except Exception:
    _HAS_PT = False

# === NAZWA_SEKCJI === Page Config (bezpiecznie) ===
try:
    st.set_page_config(page_title="📈 Results — DataGenius PRO+++", page_icon="📈", layout="wide")
except Exception:
    pass


# === NAZWA_SEKCJI === Utils: bezpieczne gettery i KPI ===

def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        return d.get(key, default) if isinstance(d, dict) else default
    except Exception:
        return default

def _data_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    rows = int(len(df))
    cols = int(df.shape[1])
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2)) if rows and cols else 0.0
    miss = float(df.isna().sum().sum() / max(1, rows * cols)) if rows and cols else 0.0
    return {"rows": rows, "cols": cols, "mem_mb": mem_mb, "miss_ratio": miss}

def _export_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

def _export_markdown(payload: Dict[str, Any]) -> bytes:
    """
    Prosty raport markdown (snapshot).
    """
    lines: List[str] = []
    p = payload
    lines.append(f"# DataGenius — Results Snapshot\n")
    # Overview
    o = p.get("overview", {})
    lines.append("## Overview")
    lines.append(f"- Rows: {o.get('rows','n/d')}")
    lines.append(f"- Columns: {o.get('cols','n/d')}")
    lines.append(f"- Memory MB: {o.get('mem_mb','n/d')}")
    lines.append(f"- Missing %: {o.get('miss_pct','n/d')}")
    lines.append(f"- Target: {o.get('target','n/d')}")
    lines.append(f"- Problem: {o.get('problem','n/d')}")
    # Model
    m = p.get("model", {})
    lines.append("\n## Model")
    lines.append(f"- Best key: {m.get('best_key','n/d')}")
    lines.append(f"- Primary metric: {m.get('primary_metric','n/d')}")
    tm = m.get("test_metrics", {})
    if tm:
        lines.append(f"- Test metrics: " + ", ".join([f"{k}={v}" for k,v in tm.items()]))
    # FI
    fi = p.get("feature_importance", {})
    topn = fi.get("topn", [])
    if topn:
        lines.append("\n## Top Feature Importance")
        for r in topn[:15]:
            lines.append(f"- {r['feature']}: {r.get('importance_norm', r.get('importance'))}")
    # EDA
    e = p.get("eda", {})
    if e:
        lines.append("\n## EDA Summary")
        for k,v in e.items():
            lines.append(f"- {k}: {v}")
    # Pipeline
    pipe = p.get("pipeline", {})
    if pipe:
        lines.append("\n## Pipeline")
        for k,v in pipe.items():
            lines.append(f"- {k}: {v}")
    return "\n".join(lines).encode("utf-8")


# === NAZWA_SEKCJI === Nawigacja — kompatybilne przejście ===
def _goto(file_hint: str, label: str = "") -> None:
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


# === NAZWA_SEKCJI === UI GŁÓWNE ===
def main() -> None:
    render_header("📈 Wyniki & Raport", "Przegląd wyników projektu: dane, EDA, model, metryki i explainability")

    state = get_state_manager()
    if not state.has_data():
        render_warning("Najpierw załaduj dane w sekcji **📊 Data Upload**.")
        if st.button("➡️ Przejdź do Upload", use_container_width=True):
            _goto("pages/02_📊_Data_Upload.py", label="📊 Data Upload")
        return

    df: pd.DataFrame = state.get_data()
    target = state.get_target_column()
    problem = state.get_problem_type()
    trained_model = state.get_trained_model() or st.session_state.get("trained_model")

    eda_results = st.session_state.get("eda_results") or state.get_eda_results() if hasattr(state, "get_eda_results") else None
    ml_training = st.session_state.get("ml_training", {})  # {"table": df, "report": dict}
    model_comp = st.session_state.get("model_comparison", {})  # {"table": df, "meta": dict}
    fi_bundle = st.session_state.get("feature_importance", {})  # {"table": df, "meta": dict}
    pipeline_state = st.session_state.get("pipeline_state", {})

    # === Overview ===
    st.subheader("🔎 Overview")
    k = _data_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 Wiersze", f"{k['rows']:,}")
    c2.metric("📊 Kolumny", f"{k['cols']:,}")
    c3.metric("💾 Pamięć", f"{k['mem_mb']:.2f} MB")
    c4.metric("🔍 Braki", f"{k['miss_ratio']*100:.1f}%")

    c5, c6 = st.columns(2)
    c5.metric("🎯 Target", target or "Nie wybrano")
    c6.metric("🧠 Problem", problem or "n/d")

    # === Sekcje w kartach ===
    tabs = st.tabs([
        "🤖 Model & Metryki",
        "🔍 EDA Summary",
        "🔥 Feature Importance",
        "🧭 Pipeline",
        "💾 Eksport"
    ])

    # === Tab 1: Model & Metryki ===
    with tabs[0]:
        st.markdown("### 🤖 Model & Metryki")

        # źródło wyników: preferuj ML_Training, a jeśli brak — Model Comparison
        res_df = _safe_get(ml_training, "table")
        report = _safe_get(ml_training, "report", {})
        best_key = _safe_get(report, "best_key") or _safe_get(_safe_get(model_comp, "meta", {}), "best_key")
        primary_metric = _safe_get(report, "primary_metric") or _safe_get(_safe_get(model_comp, "meta", {}), "scoring")

        # Ranking (jeśli jest)
        if isinstance(res_df, pd.DataFrame) and not res_df.empty:
            st.markdown("#### 📋 Ranking (walidacja)")
            st.dataframe(res_df, use_container_width=True, height=min(600, 26 * (len(res_df) + 3)))

            st.markdown("#### 📈 Wykres rankingu")
            col = None
            # wykryj kolumnę „główną” (heurystyka)
            for name in ["roc_auc", "accuracy", "f1_weighted", "r2", "rmse", "mae"]:
                if name in res_df.columns:
                    col = name
                    break
            if col:
                fig = px.bar(
                    res_df.sort_values(col, ascending=True),
                    x=col,
                    y="model",
                    orientation="h",
                    text=np.round(res_df[col], 4)
                )
                fig.update_layout(height=max(420, 50 * len(res_df)), margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brak tabeli rankingu z treningu (sekcja ML Training).")

        # Metryki TEST (z raportu ML_Training, jeśli są)
        st.markdown("#### 🩺 Metryki na zbiorze testowym")
        test_metrics = _safe_get(report, "test_metrics", {})
        if test_metrics:
            cols = st.columns(min(4, len(test_metrics) or 1))
            i = 0
            for kname, v in test_metrics.items():
                cols[i % len(cols)].metric(kname, f"{v:.4f}" if isinstance(v, (int,float)) else str(v))
                i += 1
        else:
            st.info("Brak metryk testowych (uruchom stronę ML Training).")

        # Informacja o najlepszym modelu
        st.markdown("#### ✅ Najlepszy model")
        st.write(f"- **Key**: `{best_key or 'n/d'}`")
        st.write(f"- **Primary metric**: `{primary_metric or 'n/d'}`")
        if trained_model is None:
            st.warning("Brak obiektu modelu w stanie aplikacji. Uruchom **ML Training** lub **Model Comparison**.")
        else:
            st.success("Model jest dostępny w pamięci sesji — gotowy do inferencji i raportów.")

    # === Tab 2: EDA Summary ===
    with tabs[1]:
        st.markdown("### 🔍 EDA — Podsumowanie")
        if not eda_results:
            st.info("Brak wyników EDA. Uruchom stronę **EDA**.")
        else:
            eda = eda_results.get("eda_results", {})
            # Ogólne statystyki
            stats = _safe_get(eda.get("StatisticalAnalyzer", {}), "overall", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Num features", stats.get("n_numeric", 0))
            c2.metric("Cat features", stats.get("n_categorical", 0))
            c3.metric("Memory (MB)", f"{stats.get('memory_mb', 0):.2f}")

            # Braki danych
            miss = _safe_get(eda.get("MissingDataAnalyzer", {}), "summary", {})
            st.markdown("#### Braki danych")
            d1, d2, d3 = st.columns(3)
            d1.metric("Ogółem", f"{miss.get('total_missing', 0):,}")
            d2.metric("Procent", f"{miss.get('missing_percentage', 0.0):.2f}%")
            d3.metric("Kolumny", miss.get("n_columns_with_missing", 0))

            # Korelacje
            high_corr = _safe_get(eda.get("CorrelationAnalyzer", {}), "high_correlations", [])
            if high_corr:
                st.markdown("#### Silne korelacje (Top 10)")
                for pair in high_corr[:10]:
                    try:
                        st.warning(f"**{pair['feature1']}** ↔ **{pair['feature2']}**: r = {pair['correlation']:.3f}")
                    except Exception:
                        pass

    # === Tab 3: Feature Importance ===
    with tabs[2]:
        st.markdown("### 🔥 Feature Importance")
        fi_df = _safe_get(fi_bundle, "table")
        if isinstance(fi_df, pd.DataFrame) and not fi_df.empty:
            top_n = st.slider("Pokaż TOP N", min_value=5, max_value=50, value=min(20, len(fi_df)))
            st.dataframe(fi_df.head(top_n), use_container_width=True, height=min(600, 26 * (min(top_n, len(fi_df)) + 3)))

            # Wykres
            plot_df = fi_df.head(top_n).copy()
            # preferuj 'importance_norm' jeśli istnieje
            val_col = "importance_norm" if "importance_norm" in plot_df.columns else "importance"
            if val_col in plot_df.columns and "feature" in plot_df.columns:
                fig = px.bar(
                    plot_df.sort_values(val_col, ascending=True),
                    x=val_col, y="feature", orientation="h",
                    text=np.round(plot_df[val_col], 4),
                    labels={val_col: val_col, "feature": "feature"}
                )
                fig.update_layout(height=max(420, 18 * len(plot_df) + 120), margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brak obliczonej FI. Uruchom **Feature Importance** lub **Model Comparison** (jeśli wspiera FI).")

    # === Tab 4: Pipeline ===
    with tabs[3]:
        st.markdown("### 🧭 Pipeline (snapshot)")
        if not pipeline_state:
            st.info("Brak aktywnego `pipeline_state`.")
        else:
            rows = []
            for s in pipeline_state.get("stages", []):
                started = s.get("started_at")
                finished = s.get("finished_at")
                dur = None
                if started:
                    end = finished or pd.Timestamp.now().timestamp()
                    dur = round(float(end - started), 3)
                rows.append({
                    "Etap": s.get("name"),
                    "Status": s.get("status"),
                    "Procent": round(float(s.get("percent", 0.0)), 1),
                    "Kroki": f"{int(s.get('done_steps', 0))}/{s.get('total_steps') or 'n/d'}",
                    "Czas [s]": dur,
                    "Notatki": " | ".join(s.get("notes", [])[-2:]),
                    "Ostrzeżenia": " | ".join(s.get("warnings", [])[-2:]),
                })
            tdf = pd.DataFrame(rows)
            if not tdf.empty:
                st.dataframe(tdf, use_container_width=True, height=min(500, 26 * (len(tdf) + 3)))
            else:
                st.info("Brak danych timeline.")

    # === Tab 5: Eksport ===
    with tabs[4]:
        st.markdown("### 💾 Eksport Snapshotu")
        snapshot = _build_snapshot_payload(df, target, problem, ml_training, model_comp, fi_bundle, eda_results, pipeline_state)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📦 Pobierz Snapshot (JSON)",
                data=_export_json(snapshot),
                file_name="results_snapshot.json",
                mime="application/json",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "📝 Pobierz Raport (Markdown)",
                data=_export_markdown(snapshot),
                file_name="results_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.caption("Snapshot zawiera KPI danych, meta modelu, metryki testowe, top FI oraz skrót EDA i pipeline’u.")

    # Zakończenie etapu (opcjonalne)
    if _HAS_PT and st.button("✅ Oznacz etap 'Results' jako ukończony", use_container_width=True):
        try:
            start_stage("Results", total_steps=1)
            advance(note="Snapshot viewed/exported")
            finish_stage()
            render_success("Etap 'Results' oznaczony jako completed.")
        except Exception:
            st.info("Progress tracker nieaktywowany — pomijam.")


# === NAZWA_SEKCJI === Snapshot builder ===
def _build_snapshot_payload(
    df: pd.DataFrame,
    target: Optional[str],
    problem: Optional[str],
    ml_training: Dict[str, Any],
    model_comp: Dict[str, Any],
    fi_bundle: Dict[str, Any],
    eda_results: Optional[Dict[str, Any]],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    k = _data_kpis(df)
    overview = {
        "rows": k["rows"],
        "cols": k["cols"],
        "mem_mb": round(k["mem_mb"], 2),
        "miss_pct": round(k["miss_ratio"] * 100, 2),
        "target": target,
        "problem": problem,
    }

    # Model (z ML_Training preferencyjnie)
    report = _safe_get(ml_training, "report", {})
    best_key = _safe_get(report, "best_key") or _safe_get(_safe_get(model_comp, "meta", {}), "best_key")
    primary_metric = _safe_get(report, "primary_metric") or _safe_get(_safe_get(model_comp, "meta", {}), "scoring")
    test_metrics = _safe_get(report, "test_metrics", {})

    model = {
        "best_key": best_key,
        "primary_metric": primary_metric,
        "test_metrics": test_metrics,
        "n_train": _safe_get(report, "n_train"),
        "n_val": _safe_get(report, "n_val"),
        "n_test": _safe_get(report, "n_test"),
    }

    # FI
    fi_df = _safe_get(fi_bundle, "table")
    topn: List[Dict[str, Any]] = []
    if isinstance(fi_df, pd.DataFrame) and not fi_df.empty:
        view = fi_df.head(30).copy()
        for _, r in view.iterrows():
            row = {"feature": str(r.get("feature"))}
            if "importance_norm" in view.columns:
                row["importance_norm"] = float(r.get("importance_norm"))
            if "importance" in view.columns:
                row["importance"] = float(r.get("importance"))
            topn.append(row)
    feature_importance = {"topn": topn}

    # EDA summary (light)
    eda_summary: Dict[str, Any] = {}
    try:
        if eda_results and "eda_results" in eda_results:
            eda = eda_results["eda_results"]
            stats = _safe_get(eda.get("StatisticalAnalyzer", {}), "overall", {})
            miss = _safe_get(eda.get("MissingDataAnalyzer", {}), "summary", {})
            eda_summary = {
                "n_numeric": stats.get("n_numeric"),
                "n_categorical": stats.get("n_categorical"),
                "missing_total": miss.get("total_missing"),
                "missing_pct": miss.get("missing_percentage"),
            }
    except Exception:
        pass

    # Pipeline (light)
    pp = {}
    try:
        if pipeline_state:
            pp["status"] = pipeline_state.get("status")
            pp["stage"] = pipeline_state.get("stage")
            pp["n_stages"] = len(pipeline_state.get("stages", []))
    except Exception:
        pass

    return {
        "overview": overview,
        "model": model,
        "feature_importance": feature_importance,
        "eda": eda_summary,
        "pipeline": pp,
    }


# === NAZWA_SEKCJI === Wejście modułu ===
if __name__ == "__main__":
    main()
