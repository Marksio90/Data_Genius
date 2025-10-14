# 07_📊_Monitoring.py
"""
DataGenius PRO — Monitoring (PRO+++)
Drift danych (PSI/KS/χ²), jakość predykcji, health-check oraz eksport snapshotu
"""

from __future__ import annotations

import io
import os
import sys
import time
import json
import math
import warnings
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
try:
    from frontend.app_layout import render_header, render_error, render_success, render_warning
except Exception:
    def render_header(title: str, subtitle: str = "") -> None:
        st.header(title); st.caption(subtitle or "")
    def render_error(title: str, detail: Optional[str] = None) -> None:
        st.error(title + (f": {detail}" if detail else ""))
    def render_success(msg: str) -> None:
        st.success(msg)
    def render_warning(msg: str) -> None:
        st.warning(msg)

from core.state_manager import get_state_manager  # typowo dostępny

# Opcjonalny progress tracker
try:
    from src.frontend.progress_tracker import start_stage, advance, finish_stage, add_warning
    _HAS_PT = True
except Exception:
    _HAS_PT = False

# Opcjonalne testy statystyczne (delikatnie, bez twardych zależności)
try:
    from scipy.stats import ks_2samp, chi2_contingency  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# === NAZWA_SEKCJI === Page Config (bezpiecznie) ===
try:
    st.set_page_config(page_title="📊 Monitoring — DataGenius PRO+++", page_icon="📊", layout="wide")
except Exception:
    pass


# === NAZWA_SEKCJI === Utils: metryki & eksport ===

def _safe_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)

def _data_kpis(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"rows": 0, "cols": 0, "mem_mb": 0.0, "miss_pct": 0.0}
    rows, cols = int(len(df)), int(df.shape[1])
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024**2))
    miss = float(df.isna().sum().sum() / max(1, rows * cols))
    return {"rows": rows, "cols": cols, "mem_mb": round(mem_mb, 2), "miss_pct": round(100 * miss, 2)}

def _export_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

def _export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# === NAZWA_SEKCJI === PSI / KS / χ² — drift (defensywnie) ===

def _bucketize_numeric(s: pd.Series, bins: int = 10) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.nunique(dropna=True) < 2:
        return pd.Series(["_single"] * len(s))
    try:
        binned = pd.qcut(s, q=min(bins, s.nunique(dropna=True)), duplicates="drop")
        return binned.astype(str)
    except Exception:
        # fallback: stałe przedziały
        try:
            binned = pd.cut(s, bins=min(bins, max(1, s.nunique(dropna=True))), include_lowest=True)
            return binned.astype(str)
        except Exception:
            return pd.Series(["_error"] * len(s))

def _freq_dist(s: pd.Series) -> pd.Series:
    vc = s.value_counts(dropna=True)
    total = vc.sum()
    if total == 0:
        return pd.Series(dtype=float)
    return (vc / total).sort_index()

def _psi(ref: pd.Series, prod: pd.Series) -> float:
    # PSI = Σ (p_i - q_i) * ln(p_i / q_i)
    # 1) kategoryzacja
    if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(prod):
        ref_b = _bucketize_numeric(ref)
        prod_b = _bucketize_numeric(prod)
    else:
        ref_b, prod_b = ref.astype(str), prod.astype(str)

    # 2) wspólna przestrzeń kategorii
    ref_d = _freq_dist(ref_b)
    prod_d = _freq_dist(prod_b)
    cats = sorted(set(ref_d.index).union(set(prod_d.index)))

    psi = 0.0
    for c in cats:
        p = float(ref_d.get(c, 0.0))
        q = float(prod_d.get(c, 0.0))
        # stabilizacja
        p = max(p, 1e-6)
        q = max(q, 1e-6)
        psi += (p - q) * math.log(p / q)
    return float(psi)

def _ks_pvalue(ref: pd.Series, prod: pd.Series) -> Optional[float]:
    if not _HAS_SCIPY:
        return None
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    prod = pd.to_numeric(prod, errors="coerce").dropna()
    if len(ref) < 2 or len(prod) < 2:
        return None
    try:
        stat, pval = ks_2samp(ref, prod)
        return float(pval)
    except Exception:
        return None

def _chi2_pvalue(ref: pd.Series, prod: pd.Series) -> Optional[float]:
    if not _HAS_SCIPY:
        return None
    r = _freq_dist(ref.astype(str))
    p = _freq_dist(prod.astype(str))
    cats = sorted(set(r.index).union(p.index))
    if not cats:
        return None
    obs = np.vstack([ [r.get(c, 0.0), p.get(c, 0.0)] for c in cats ]).T
    # zamiana freq na counts przez skalowanie (umowne 10000)
    obs_counts = np.round(obs * 10000).astype(int)
    try:
        chi2, pval, _, _ = chi2_contingency(obs_counts)
        return float(pval)
    except Exception:
        return None

def _compute_drift(reference: pd.DataFrame, production: pd.DataFrame, features: List[str], psi_bins: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in features:
        if col not in reference.columns or col not in production.columns:
            continue
        ref_col = reference[col]
        prod_col = production[col]
        psi = _psi(ref_col, prod_col)
        pval = None
        test = None
        if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(prod_col):
            pval = _ks_pvalue(ref_col, prod_col)
            test = "KS" if pval is not None else None
        else:
            pval = _chi2_pvalue(ref_col.astype(str), prod_col.astype(str))
            test = "chi2" if pval is not None else None
        sev = "severe" if psi >= 0.3 else ("moderate" if psi >= 0.2 else ("minor" if psi >= 0.1 else "ok"))
        rows.append({
            "feature": col, "psi": float(psi), "test": test, "p_value": (None if pval is None else float(pval)), "severity": sev,
            "ref_missing_pct": float(ref_col.isna().mean() * 100.0),
            "prod_missing_pct": float(prod_col.isna().mean() * 100.0),
        })
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return df


# === NAZWA_SEKCJI === Predykcje i metryki jakości ===

def _detect_problem_type(y: pd.Series) -> str:
    y_non = y.dropna()
    nunique = y_non.nunique()
    if pd.api.types.is_bool_dtype(y_non):
        return "classification"
    if pd.api.types.is_integer_dtype(y_non) and nunique <= min(20, max(2, int(len(y_non) ** 0.5))):
        return "classification"
    if pd.api.types.is_object_dtype(y_non) and nunique <= 50:
        return "classification"
    return "regression"

def _quality_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: Optional[np.ndarray], problem: str) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
    out: Dict[str, Any] = {}
    try:
        if problem == "classification":
            out["accuracy"] = float(accuracy_score(y_true, y_pred))
            out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
            if y_prob is not None:
                # jeżeli multiclass, roc_auc ował się inaczej — dla prostoty pominiemy
                try:
                    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                except Exception:
                    pass
        else:
            out["r2"] = float(r2_score(y_true, y_pred))
            out["rmse"] = float(mean_squared_error(y_true, y_pred) ** 0.5)
            out["mae"] = float(mean_absolute_error(y_true, y_pred))
    except Exception:
        pass
    return out

def _predict_with_pipeline(pipeline: Any, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    y_pred = pipeline.predict(X)
    y_prob = None
    try:
        proba = pipeline.predict_proba(X)
        # binary => druga kolumna
        if proba.ndim == 2 and proba.shape[1] == 2:
            y_prob = proba[:, 1]
    except Exception:
        pass
    return y_pred, y_prob


# === NAZWA_SEKCJI === Health log (latencja/błędy) ===

def _log_health(latency_ms: float, ok: bool, err: Optional[str] = None) -> None:
    st.session_state.setdefault("monitoring_logs", [])
    st.session_state["monitoring_logs"].append({"t": time.time(), "latency_ms": float(latency_ms), "ok": bool(ok), "err": err})
    if len(st.session_state["monitoring_logs"]) > 1000:
        st.session_state["monitoring_logs"] = st.session_state["monitoring_logs"][-1000:]

def _health_summary() -> Dict[str, Any]:
    logs = st.session_state.get("monitoring_logs", [])
    if not logs:
        return {"n": 0, "ok_rate": None, "p95_ms": None}
    lat = [l["latency_ms"] for l in logs if isinstance(l.get("latency_ms"), (int, float))]
    oks = [1 if l.get("ok") else 0 for l in logs]
    p95 = float(np.percentile(lat, 95)) if lat else None
    return {"n": len(logs), "ok_rate": (sum(oks) / len(oks) if oks else None), "p95_ms": p95}


# === NAZWA_SEKCJI === UI: główna strona ===

def main() -> None:
    render_header("📊 Monitoring", "Drift danych, jakość predykcji i health-check Twojego modelu")

    state = get_state_manager()
    if not state.has_data():
        render_warning("Najpierw załaduj dane w sekcji **📊 Data Upload**.")
        return

    df_ref_default: pd.DataFrame = state.get_data()
    trained_model = state.get_trained_model() or st.session_state.get("trained_model")
    target = state.get_target_column()
    problem = state.get_problem_type()

    # === Sekcja: źródła danych ===
    st.subheader("1️⃣ Źródła danych")
    colA, colB = st.columns(2)
    with colA:
        ref_source = st.radio("Reference (do porównania):", ["Aktualny dataset w pamięci (state)"], index=0, horizontal=False)
        st.caption("Możesz tu wpiąć inny snapshot treningowy — utrzymujemy prosto: bierzemy dane ze `state_manager`.")
    with colB:
        st.caption("Production batch (CSV/Parquet) — nowa próbka do monitoringu.")
        uploaded = st.file_uploader("Wgraj batch produkcyjny", type=["csv", "parquet"], help="CSV lub Parquet; nazwy kolumn powinny zgadzać się z referencją.")
        df_prod: Optional[pd.DataFrame] = None
        if uploaded is not None:
            try:
                t0 = time.time()
                if uploaded.name.lower().endswith(".parquet"):
                    df_prod = pd.read_parquet(uploaded)
                else:
                    df_prod = pd.read_csv(uploaded)
                _log_health((time.time() - t0) * 1000.0, ok=True)
                render_success(f"Wczytano batch: {len(df_prod):,} wierszy.")
            except Exception as e:
                _log_health(0.0, ok=False, err=str(e))
                render_error("Nie udało się wczytać batcha", str(e))

    # Jeśli brak batcha, pozwól zasymulować produkcję (inne losowanie)
    if df_prod is None:
        with st.expander("🔁 Brak batcha? Zasymuluj produkcję (losowe próbki)"):
            frac = st.slider("Wielkość próbki produkcyjnej", 0.05, 0.9, 0.3, 0.05)
            seed = st.number_input("Random state", min_value=0, max_value=999_999, value=2024, step=1)
            df_prod = df_ref_default.sample(frac=frac, random_state=int(seed)).reset_index(drop=True)
            st.caption("Uwaga: to tylko symulacja; w realu wgraj batch CSV/Parquet z produkcji.")

    # Konfiguracja listy cech do monitorowania
    with st.expander("⚙️ Konfiguracja monitoringu", expanded=True):
        all_cols = df_ref_default.columns.tolist()
        cols_to_monitor = st.multiselect("Wybierz cechy do monitorowania (puste = wszystkie oprócz targetu)", options=all_cols, default=[])
        if not cols_to_monitor:
            cols_to_monitor = [c for c in all_cols if c != target]
        psi_bins = st.slider("Liczba przedziałów dla PSI (numeryczne)", 5, 20, 10, 1)
        alert_psi = st.slider("Próg alertu PSI (severe)", 0.1, 1.0, 0.3, 0.05)

    # === Sekcja: Drift danych ===
    st.subheader("2️⃣ Drift danych")
    with st.spinner("Liczenie driftu (PSI/KS/χ²)…"):
        try:
            if _HAS_PT:
                start_stage("Monitoring", total_steps=3)
                advance(note="Compute drift")
            drift_df = _compute_drift(df_ref_default, df_prod, cols_to_monitor, psi_bins=psi_bins)
            n_severe = int((drift_df["psi"] >= alert_psi).sum()) if not drift_df.empty else 0
            c1, c2, c3, c4 = st.columns(4)
            k_ref, k_prod = _data_kpis(df_ref_default), _data_kpis(df_prod)
            c1.metric("Ref rows", f"{k_ref['rows']:,}")
            c2.metric("Prod rows", f"{k_prod['rows']:,}")
            c3.metric("Drifted (PSI≥0.1)", int((drift_df["psi"] >= 0.1).sum()) if not drift_df.empty else 0)
            c4.metric("Severe (PSI≥{:.2f})".format(alert_psi), n_severe)

            if drift_df.empty:
                st.info("Brak wyników driftu (pusta lista cech?)")
            else:
                st.markdown("#### Ranking driftu (PSI)")
                st.dataframe(drift_df, use_container_width=True, height=min(600, 26 * (len(drift_df) + 3)))

                # Wykres TOP N
                top_n = st.slider("Pokaż TOP N (PSI)", 5, min(50, len(drift_df)), min(15, len(drift_df)))
                plot_df = drift_df.head(top_n).sort_values("psi", ascending=True)
                fig = px.bar(plot_df, x="psi", y="feature", orientation="h", text=np.round(plot_df["psi"], 4),
                             color=plot_df["psi"] >= alert_psi, labels={"color": f"PSI≥{alert_psi}"})
                fig.update_layout(height=max(420, 20 * len(plot_df) + 120), margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

                # Porównanie rozkładów dla wybranej cechy
                feat = st.selectbox("Porównaj rozkłady dla cechy", options=plot_df["feature"].tolist())
                if feat:
                    if pd.api.types.is_numeric_dtype(df_ref_default[feat]):
                        f1 = px.histogram(df_ref_default, x=feat, nbins=40, opacity=0.6, marginal="box", title=f"{feat} — reference")
                        f2 = px.histogram(df_prod, x=feat, nbins=40, opacity=0.6, marginal="box", title=f"{feat} — production")
                        st.plotly_chart(f1, use_container_width=True)
                        st.plotly_chart(f2, use_container_width=True)
                    else:
                        f1 = px.bar(df_ref_default[feat].value_counts().reset_index().rename(columns={"index":"value", feat:"count"}),
                                    x="count", y="value", orientation="h", title=f"{feat} — reference")
                        f2 = px.bar(df_prod[feat].value_counts().reset_index().rename(columns={"index":"value", feat:"count"}),
                                    x="count", y="value", orientation="h", title=f"{feat} — production")
                        st.plotly_chart(f1, use_container_width=True)
                        st.plotly_chart(f2, use_container_width=True)
        except Exception as e:
            if _HAS_PT:
                try: add_warning(str(e)); finish_stage(status="failed")
                except Exception: pass
            render_error("Błąd liczenia driftu", str(e))

    # === Sekcja: Jakość predykcji (jeśli mamy model) ===
    st.subheader("3️⃣ Jakość predykcji (batch)")
    if trained_model is None:
        st.info("Brak modelu w pamięci — uruchom **ML Training** lub wczytaj pipeline.")
    else:
        try:
            # Dopasuj kolumny: usuń target z X
            X_prod = df_prod.drop(columns=[target]) if target and target in df_prod.columns else df_prod.copy()
            t0 = time.time()
            yhat, yprob = _predict_with_pipeline(trained_model, X_prod)
            _log_health((time.time() - t0) * 1000.0, ok=True)

            st.success(f"Predykcje obliczone dla batcha: {len(X_prod):,} rekordów.")

            # Jeśli batch posiada target — oblicz metryki
            if target and target in df_prod.columns:
                y_true = df_prod[target]
                detected = problem or _detect_problem_type(y_true)
                metrics = _quality_metrics(y_true, _safe_series(yhat), yprob, detected)
                st.markdown("#### Metryki jakości (production batch)")
                if metrics:
                    cols = st.columns(min(4, len(metrics)))
                    i = 0
                    for k, v in metrics.items():
                        cols[i % len(cols)].metric(k, f"{v:.4f}" if isinstance(v, (int,float)) else str(v))
                        i += 1
                else:
                    st.info("Nie udało się policzyć metryk — sprawdź typ problemu / y_true.")
            else:
                st.info("Batch nie zawiera kolumny target — raportuję tylko inferencję.")
        except Exception as e:
            _log_health(0.0, ok=False, err=str(e))
            render_error("Błąd podczas inferencji", str(e))

    # === Sekcja: Health-check (latencja/błędy) ===
    st.subheader("4️⃣ Health-check (lokalny log)")
    hs = _health_summary()
    h1, h2, h3 = st.columns(3)
    h1.metric("Liczba zdarzeń", hs.get("n", 0))
    h2.metric("OK rate", f"{(hs['ok_rate']*100):.1f}%" if hs.get("ok_rate") is not None else "n/d")
    h3.metric("P95 latencji", f"{hs['p95_ms']:.0f} ms" if hs.get("p95_ms") is not None else "n/d")
    with st.expander("Pokaż log"):
        logs = st.session_state.get("monitoring_logs", [])
        if logs:
            df_logs = pd.DataFrame(logs)
            df_logs["ts"] = df_logs["t"].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
            st.dataframe(df_logs[["ts","latency_ms","ok","err"]], use_container_width=True, height=min(400, 26*(len(df_logs)+3)))
        else:
            st.caption("Brak logów.")

    # === Sekcja: Eksport snapshotu ===
    st.subheader("5️⃣ Eksport")
    snapshot_payload: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "reference_kpi": _data_kpis(df_ref_default),
        "production_kpi": _data_kpis(df_prod),
        "drift": (drift_df.to_dict(orient="records") if "drift_df" in locals() and isinstance(drift_df, pd.DataFrame) else []),
        "health": hs,
    }
    c1, c2 = st.columns(2)
    if "drift_df" in locals() and isinstance(drift_df, pd.DataFrame) and not drift_df.empty:
        with c1:
            st.download_button("📄 Eksport drift (CSV)", data=_export_csv(drift_df), file_name="monitoring_drift.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("📦 Snapshot (JSON)", data=_export_json(snapshot_payload), file_name="monitoring_snapshot.json", mime="application/json", use_container_width=True)

    # === Progress tracker: domknięcie etapu ===
    if _HAS_PT:
        try:
            advance(note="Export / finish")
            finish_stage()
        except Exception:
            pass


# === NAZWA_SEKCJI === Wejście modułu ===
if __name__ == "__main__":
    main()
