# src/frontend/metric_cards.py
# === OPIS MODUŁU ===
# Moduł wyświetlania metryk PRO+++:
# - Zbiera metryki z session_state (dane, model, FI, pipeline, system)
# - Renderuje responsywne karty z ikonami i ewentualnym sparkline (Plotly)
# - Eksport metryk do JSON
# - Defensywna obsługa błędów i braków

from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Logger (zgodny z Twoim ekosystemem) ===
try:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("metric_cards")

# === NAZWA_SEKCJI === Dataclasses ===

@dataclass
class Metric:
    title: str
    value: str
    help: Optional[str] = None
    delta: Optional[str] = None
    icon: Optional[str] = None  # emoji lub Tailwind-friendly
    sparkline: Optional[List[float]] = None  # ostatnie punkty do mini-wykresu
    lower_is_better: bool = False  # dla interpretacji delta

@dataclass
class MetricGroup:
    name: str
    items: List[Metric] = field(default_factory=list)

# === NAZWA_SEKCJI === Narzędzia pomocnicze ===

def _fmt_int(n: Optional[int]) -> str:
    if n is None:
        return "n/d"
    return f"{int(n):,}".replace(",", " ")

def _fmt_float(x: Optional[float], ndigits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/d"
    return f"{x:.{ndigits}f}"

def _fmt_pct(x: Optional[float], ndigits: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/d"
    return f"{x*100:.{ndigits}f}%"

def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        return d.get(key, default) if isinstance(d, dict) else default
    except Exception:
        return default

# === NAZWA_SEKCJI === System metrics (opcjonalne psutil) ===

def _system_metrics() -> Dict[str, Optional[float]]:
    cpu = mem = disk = None
    try:
        import psutil  # opcjonalne
        cpu = float(psutil.cpu_percent(interval=0.2))
        mem = float(psutil.virtual_memory().percent)
        disk = float(psutil.disk_usage("/").percent)
    except Exception:
        # fallback: heurystyka lub brak danych
        pass
    return {"cpu": cpu, "mem": mem, "disk": disk}

# === NAZWA_SEKCJI === Cache: lekkie agregaty danych ===

@st.cache_data(show_spinner=False, ttl=600)
def _compute_data_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))
    # Prosty missing ratio
    nulls_total = int(df.isna().sum().sum())
    cells_total = int(n_rows * n_cols) if n_rows and n_cols else 0
    miss_ratio = float(nulls_total / cells_total) if cells_total else 0.0
    # Wybrane heurystyki
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    high_card = [c for c in df.columns if df[c].nunique(dropna=True) > max(50, 0.2 * n_rows)]
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "mem_mb": round(mem_mb, 2),
        "miss_ratio": miss_ratio,
        "num_cols": len(num_cols),
        "cat_cols": len(cat_cols),
        "high_card_cols": len(high_card),
    }

# === NAZWA_SEKCJI === FI: top-cecha i udział ===

def _feature_importance_summary() -> Tuple[Optional[str], Optional[float]]:
    try:
        fi = st.session_state.get("feature_importance")
        if not fi or "table" not in fi:
            return None, None
        df: pd.DataFrame = fi["table"]
        if df.empty:
            return None, None
        row = df.iloc[0]
        feat = str(row["feature"])
        # użyj importance_norm jeśli istnieje
        share = float(row["importance_norm"]) if "importance_norm" in df.columns else float(row["importance"])
        return feat, share
    except Exception:
        return None, None

# === NAZWA_SEKCJI === Model/trening: pseudo-odczyt metryk ===

def _model_summary() -> Dict[str, Any]:
    out: Dict[str, Any] = {"fitted": False, "score": None, "n_features": None, "train_secs": None}
    try:
        model = st.session_state.get("trained_model")
        if model is None:
            return out
        # spróbuj wydobyć metryki, o ile są w run_stats lub atrybutach
        run_stats = st.session_state.get("run_stats", {})
        out["train_secs"] = _safe_get(run_stats, "train_seconds")
        # n_features_
        for attr in ("n_features_in_", "n_features_"):
            try:
                val = getattr(model, attr, None)
                if isinstance(val, (int, float)):
                    out["n_features"] = int(val)
                    break
            except Exception:
                pass
        # score (jeśli zapisany w run_stats)
        out["score"] = _safe_get(run_stats, "best_score")
        out["fitted"] = True
        return out
    except Exception:
        return out

# === NAZWA_SEKCJI === Pipeline status (jeśli podany) ===

def _pipeline_summary() -> Dict[str, Any]:
    d = st.session_state.get("pipeline_state", {})
    return {
        "stage": _safe_get(d, "stage", "n/d"),    # e.g. upload/clean/eda/train/report
        "status": _safe_get(d, "status", "n/d"),  # initialized/active/completed/failed
        "last_sec": _safe_get(d, "last_stage_seconds"),
    }

# === NAZWA_SEKCJI === Budowa grup metryk ===

def _build_metric_groups(df: Optional[pd.DataFrame]) -> List[MetricGroup]:
    groups: List[MetricGroup] = []

    # Dane
    if df is not None and not df.empty:
        k = _compute_data_kpis(df)
        groups.append(
            MetricGroup(
                name="📦 Dane",
                items=[
                    Metric("Wiersze", _fmt_int(k["n_rows"]), help="Liczba rekordów", icon="🧾"),
                    Metric("Kolumny", _fmt_int(k["n_cols"]), help="Liczba pól/kolumn", icon="🧩"),
                    Metric("Pamięć", f"{_fmt_float(k['mem_mb'])} MB", help="Szacowane zużycie RAM", icon="💾"),
                    Metric("Braki danych", _fmt_pct(k["miss_ratio"]), help="Udział braków w całej tabeli", icon="🕳️", lower_is_better=True),
                    Metric("Num / Cat", f"{_fmt_int(k['num_cols'])} / {_fmt_int(k['cat_cols'])}", help="Podział typów kolumn", icon="📐"),
                    Metric("High cardinality", _fmt_int(k["high_card_cols"]), help="Kolumny o bardzo wielu unikalnych wartościach", icon="📈", lower_is_better=True),
                ],
            )
        )
    else:
        groups.append(MetricGroup(name="📦 Dane", items=[Metric("Status", "Brak danych", icon="ℹ️")]))

    # FI
    top_feat, share = _feature_importance_summary()
    if top_feat:
        groups.append(
            MetricGroup(
                name="🔦 Feature Importance",
                items=[
                    Metric("Top cecha", top_feat, help="Najważniejsza cecha wg bieżącej metody FI", icon="🏅"),
                    Metric("Udział top", _fmt_pct(share if share and share <= 1 else None), help="Udział top cechy (znormalizowany)", icon="🥇"),
                ],
            )
        )

    # Model
    ms = _model_summary()
    if ms["fitted"]:
        groups.append(
            MetricGroup(
                name="🤖 Model",
                items=[
                    Metric("Cecha wejściowe", _fmt_int(ms["n_features"]), help="Liczba cech widzianych przez model", icon="🧠"),
                    Metric("Wynik (best)", _fmt_float(ms["score"], 4), help="Najlepsza metryka treningu (jeśli zapisana)", icon="📊"),
                    Metric("Czas treningu", _fmt_float(ms["train_secs"], 2) + " s" if ms["train_secs"] else "n/d", help="Czas treningu (run_stats)", icon="⏱️"),
                ],
            )
        )

    # Pipeline
    ps = _pipeline_summary()
    groups.append(
        MetricGroup(
            name="🛠️ Pipeline",
            items=[
                Metric("Etap", str(ps["stage"]), help="Bieżący etap przetwarzania", icon="🧭"),
                Metric("Status", str(ps["status"]), help="Status przebiegu", icon="📡"),
                Metric("Czas etapu", (_fmt_float(ps["last_sec"]) + " s") if ps.get("last_sec") else "n/d", help="Czas ostatniego etapu", icon="⏳"),
            ],
        )
    )

    # System
    sm = _system_metrics()
    if any(v is not None for v in sm.values()):
        groups.append(
            MetricGroup(
                name="🖥️ System",
                items=[
                    Metric("CPU", _fmt_pct((sm["cpu"] or 0) / 100.0), help="Zużycie CPU", icon="🧮", lower_is_better=True),
                    Metric("RAM", _fmt_pct((sm["mem"] or 0) / 100.0), help="Zużycie pamięci", icon="🧠", lower_is_better=True),
                    Metric("Dysk", _fmt_pct((sm["disk"] or 0) / 100.0), help="Zajętość dysku /", icon="💿", lower_is_better=True),
                ],
            )
        )

    return groups

# === NAZWA_SEKCJI === Sparkline (opcjonalny) ===

def _sparkline_fig(values: List[float]) -> go.Figure:
    v = [float(x) for x in values][-30:]  # ogranicz do 30 pkt
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=v, mode="lines", line=dict(width=2)))
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

# === NAZWA_SEKCJI === CSS dla kart ===

_CARD_CSS = """
<style>
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.metric-card {
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  background: var(--background-color);
  border: 1px solid rgba(128,128,128,0.2);
}
.metric-title {
  font-size: 0.85rem; opacity: 0.85; display:flex; align-items:center; gap:8px;
}
.metric-value {
  font-size: 1.6rem; font-weight: 700; margin-top: 6px;
}
.metric-delta {
  font-size: 0.8rem; opacity: 0.8; margin-top: 4px;
}
.section-title {
  font-size: 1rem; font-weight: 700; margin: 12px 2px 6px 2px;
}
</style>
"""

# === NAZWA_SEKCJI === Eksport metryk ===

def _export_metrics_payload(groups: List[MetricGroup]) -> bytes:
    payload = {
        "exported_at": int(time.time()),
        "sections": [
            {
                "name": g.name,
                "items": [asdict(i) for i in g.items],
            }
            for g in groups
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

# === NAZWA_SEKCJI === Główna funkcja UI ===

def render_metric_cards(
    *,
    title: str = "📈 Metric Cards — PRO+++",
    show_export: bool = True,
    enable_sparklines: bool = False,
) -> None:
    """
    Renderuje karty metryk. Wymaga (opcjonalnie) obiektów w session_state:
      - raw_df (DataFrame), data_meta (dict), feature_importance (dict), trained_model (Any),
        pipeline_state (dict), run_stats (dict)
    """
    st.header(title)
    st.caption("Przegląd najważniejszych metryk projektu: dane, model, FI, pipeline i system.")

    # Wczytaj DF (preferuj raw_df, inaczej próbuj data_meta; jeśli brak — część sekcji się schowa)
    df = None
    if isinstance(st.session_state.get("raw_df"), pd.DataFrame):
        df = st.session_state["raw_df"]

    groups = _build_metric_groups(df)

    # CSS
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    # Render
    for group in groups:
        st.markdown(f"<div class='section-title'>{group.name}</div>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
            cols = st.columns(len(group.items)) if group.items else [st]
            for idx, m in enumerate(group.items):
                with cols[idx]:
                    with st.container():
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        title = f"{m.icon or ''} {m.title}".strip()
                        st.markdown(f"<div class='metric-title'>{title}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-value'>{m.value}</div>", unsafe_allow_html=True)

                        if m.delta:
                            st.markdown(f"<div class='metric-delta'>{m.delta}</div>", unsafe_allow_html=True)

                        if m.help:
                            st.caption(m.help)

                        # opcjonalny sparkline
                        if enable_sparklines and m.sparkline:
                            fig = _sparkline_fig(m.sparkline)
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                        st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Eksport
    if show_export:
        payload = _export_metrics_payload(groups)
        st.download_button(
            "💾 Eksportuj metryki (JSON)",
            data=payload,
            file_name="metric_cards.json",
            mime="application/json",
            use_container_width=True,
        )

# === NAZWA_SEKCJI === Lokalny punkt wejścia (opcjonalny)
if __name__ == "__main__":
    # Uruchom: streamlit run src/frontend/metric_cards.py
    render_metric_cards()
