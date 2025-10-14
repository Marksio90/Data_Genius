# 01_🏠_Home.py
# === OPIS MODUŁU ===
# Strona startowa DataGenius/Universal-Forecasting — PRO+++
# - Landing + skróty nawigacyjne
# - Security Gate dla OPENAI_API_KEY (secrets/session/env)
# - Szybkie KPI: dane/model/pipeline
# - System Health (opcjonalnie psutil)
# - Timeline z pipeline_state i "ostatnia aktywność"
# - Odporna na różne wersje Streamlit nawigacja (page_link / switch_page / fallback)

from __future__ import annotations

import os
import time
import platform
import warnings
from typing import Any, Dict, Optional

import streamlit as st
import pandas as pd

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Logger (zgodny z ekosystemem) ===
try:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("home")

# === NAZWA_SEKCJI === Page Config (bezpiecznie) ===
try:
    st.set_page_config(
        page_title="🏠 Home — DataGenius PRO+++",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    # Już ustawione przez inny plik — zignoruj
    pass

# === NAZWA_SEKCJI === Lekki CSS (chips + nagłówki) ===
st.markdown(
    """
    <style>
      .chip-ok{display:inline-block;padding:2px 10px;border-radius:999px;background:#15a34a1a;border:1px solid #16a34a;color:#16a34a;font-weight:600;font-size:12px}
      .chip-warn{display:inline-block;padding:2px 10px;border-radius:999px;background:#f59e0b1a;border:1px solid #f59e0b;color:#b45309;font-weight:600;font-size:12px}
      .subtle{opacity:.85}
    </style>
    """,
    unsafe_allow_html=True,
)

# === NAZWA_SEKCJI === Pomocnicze: Security Gate i statusy ===

def _get_openai_key() -> Optional[str]:
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
        if key:
            return key
    except Exception:
        pass
    key = st.session_state.get("openai_api_key")
    if key:
        return key
    return os.environ.get("OPENAI_API_KEY")

def _security_banner() -> None:
    key_present = bool(_get_openai_key())
    colA, colB = st.columns([0.75, 0.25])
    with colA:
        if key_present:
            st.success("OPENAI_API_KEY wykryty (secrets/session/env). Czat i AI-funkcje są aktywne.", icon="✅")
        else:
            st.warning(
                "Brak `OPENAI_API_KEY` w **st.secrets**, `st.session_state['openai_api_key']` lub zmiennych środowiskowych. "
                "Tryb AI może działać **OFFLINE**. Uzupełnij klucz w ustawieniach lub `.env`.",
                icon="⚠️",
            )
    with colB:
        status_html = (
            '<span class="chip-ok">AI: ONLINE</span>'
            if key_present
            else '<span class="chip-warn">AI: OFFLINE</span>'
        )
        st.markdown(f"<div style='text-align:right'>{status_html}</div>", unsafe_allow_html=True)


# === NAZWA_SEKCJI === Cache: lekkie KPI danych ===

@st.cache_data(show_spinner=False, ttl=600)
def _data_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    rows = int(len(df))
    cols = int(df.shape[1])
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))
    null_ratio = float(df.isna().sum().sum() / max(1, (rows * cols))) if rows and cols else 0.0
    return {"rows": rows, "cols": cols, "mem_mb": mem_mb, "null_ratio": null_ratio}

# === NAZWA_SEKCJI === System Health (opcjonalny psutil) ===

def _system_health() -> Dict[str, Optional[float]]:
    cpu = mem = disk = None
    try:
        import psutil  # optional
        cpu = float(psutil.cpu_percent(interval=0.2))
        mem = float(psutil.virtual_memory().percent)
        disk = float(psutil.disk_usage("/").percent)
    except Exception:
        pass
    return {"cpu": cpu, "mem": mem, "disk": disk}

# === NAZWA_SEKCJI === Timeline (light) z pipeline_state ===

@st.cache_data(show_spinner=False, ttl=300)
def _timeline_df(payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for s in payload.get("stages", []):
        started = s.get("started_at")
        finished = s.get("finished_at")
        dur = None
        if started:
            end = finished or time.time()
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
    df = pd.DataFrame(rows)
    if not df.empty:
        df.index = range(1, len(df) + 1)
    return df

# === NAZWA_SEKCJI === Ostatnia aktywność (prosty bufor) ===

def _push_activity(msg: str) -> None:
    st.session_state.setdefault("activity_log", [])
    st.session_state["activity_log"].append({"t": time.time(), "msg": msg})
    if len(st.session_state["activity_log"]) > 200:
        st.session_state["activity_log"] = st.session_state["activity_log"][-200:]

def _last_activity(n: int = 8) -> pd.DataFrame:
    log = st.session_state.get("activity_log", [])
    if not log:
        return pd.DataFrame(columns=["Kiedy", "Zdarzenie"])
    rows = []
    for rec in reversed(log[-n:]):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec["t"]))
        rows.append({"Kiedy": ts, "Zdarzenie": rec["msg"]})
    return pd.DataFrame(rows)

# === NAZWA_SEKCJI === Nawigacja: kompatybilne przejście do stron ===

def _goto(label: str, file_hint: str) -> None:
    """
    Próbuj w kolejności:
      1) st.page_link (link bezpośredni – preferowany),
      2) st.switch_page (jeśli API dostępne),
      3) fallback: komunikat z podpowiedzią.
    """
    # 1) Jeśli używasz multipage (folder "pages/"), warto wystawić link:
    try:
        # Streamlit 1.31+: st.page_link może wskazać stronę przez path lub name
        st.page_link(file_hint, label=label, icon=None, help=None, disabled=False)
        return
    except Exception:
        pass

    # 2) Dynamically switch (jeśli dostępne)
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(file_hint)
            return
        except Exception:
            pass

    # 3) Fallback — pokaż informację
    st.info(f"Przejdź ręcznie do strony: **{file_hint}** (menu po lewej).")

# === NAZWA_SEKCJI === Nagłówek i hero ===

st.title("🏠 Home — DataGenius PRO+++")
st.caption("Enterprise-grade Streamlit AI • Modularna architektura • EDA • AutoML • Explainability • Forecasting")

# === NAZWA_SEKCJI === Security Gate + szybkie ustawienie klucza w sesji ===
with st.expander("🔐 Uwierzytelnienie AI (opcjonalnie)", expanded=False):
    st.text_input(
        "OPENAI_API_KEY (tylko dla tej sesji)",
        type="password",
        key="openai_api_key",
        help="Możesz też dodać klucz do st.secrets lub .env (OPENAI_API_KEY=...).",
    )
_security_banner()

# === NAZWA_SEKCJI === Szybkie KPI (Dane / Model / Pipeline) ===
c1, c2, c3, c4 = st.columns(4)
raw_df = st.session_state.get("raw_df")
if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
    k = _data_kpis(raw_df)
    c1.metric("Wiersze", f"{k['rows']:,}")
    c2.metric("Kolumny", f"{k['cols']:,}")
    c3.metric("Pamięć", f"{k['mem_mb']:.2f} MB")
    c4.metric("Braki danych", f"{k['null_ratio']*100:.1f}%")
else:
    c1.metric("Dane", "Brak")
    c2.metric("Kolumny", "n/d")
    c3.metric("Pamięć", "n/d")
    c4.metric("Braki danych", "n/d")

# Model
mc = st.session_state.get("model_comparison", {})
trained_model = st.session_state.get("trained_model")
if trained_model is not None and isinstance(mc, dict) and mc.get("meta"):
    meta = mc["meta"]
    st.success(
        f"Najlepszy model: **{meta.get('best_key', 'n/d')}** • Scoring: **{meta.get('scoring', 'n/d')}** • Problem: **{meta.get('problem', 'n/d')}**",
        icon="🤖",
    )

# Pipeline
pipeline_state = st.session_state.get("pipeline_state", {})
if pipeline_state:
    last_stage = pipeline_state.get("stage") or "n/d"
    last_status = pipeline_state.get("status") or "n/d"
    st.info(f"📦 Pipeline: {last_stage} — {last_status}", icon="📦")

# === NAZWA_SEKCJI === Szybkie akcje / Nawigacja ===
st.subheader("🚀 Szybkie akcje")
a1, a2, a3, a4, a5, a6 = st.columns(6)
with a1:
    if st.button("📤 Upload & Preview", use_container_width=True):
        _push_activity("Przejście: Upload & Preview")
        _goto("Upload & Preview", "src/frontend/data_preview.py")
with a2:
    if st.button("📊 Metric Cards", use_container_width=True):
        _push_activity("Przejście: Metric Cards")
        _goto("Metric Cards", "src/frontend/metric_cards.py")
with a3:
    if st.button("📈 Model Comparison", use_container_width=True):
        _push_activity("Przejście: Model Comparison")
        _goto("Model Comparison", "src/frontend/model_comparison.py")
with a4:
    if st.button("🔥 Feature Importance", use_container_width=True):
        _push_activity("Przejście: Feature Importance")
        _goto("Feature Importance", "src/frontend/feature_importance.py")
with a5:
    if st.button("💬 AI Chat", use_container_width=True):
        _push_activity("Przejście: AI Chat")
        _goto("AI Chat", "src/frontend/chat_interface.py")
with a6:
    if st.button("🧭 Progress", use_container_width=True):
        _push_activity("Przejście: Progress")
        _goto("Progress", "src/frontend/progress_tracker.py")

# === NAZWA_SEKCJI === System Health & Środowisko ===
st.subheader("🖥️ System Health & Środowisko")
sys_col1, sys_col2, sys_col3, sys_col4 = st.columns(4)
h = _system_health()
sys_col1.metric("CPU", f"{(h['cpu'] or 0):.0f}%" if h["cpu"] is not None else "n/d")
sys_col2.metric("RAM", f"{(h['mem'] or 0):.0f}%" if h["mem"] is not None else "n/d")
sys_col3.metric("Dysk /", f"{(h['disk'] or 0):.0f}%" if h["disk"] is not None else "n/d")
sys_col4.metric("Python", platform.python_version())

with st.expander("🔎 Szczegóły środowiska", expanded=False):
    env_info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "session_keys": list(st.session_state.keys()),
    }
    st.json(env_info)

# === NAZWA_SEKCJI === Timeline pipeline’u (snapshot) ===
st.subheader("🗂️ Timeline pipeline’u (snapshot)")
if pipeline_state:
    df_tl = _timeline_df(pipeline_state)
    if df_tl.empty:
        st.info("Brak danych timeline — uruchom etapy w Progress Tracker.")
    else:
        st.dataframe(df_tl, use_container_width=True, height=min(420, 26 * (len(df_tl) + 2)))
else:
    st.info("Brak aktywnego `pipeline_state` w sesji.")

# === NAZWA_SEKCJI === Ostatnia aktywność użytkownika ===
st.subheader("🕘 Ostatnia aktywność")
act_df = _last_activity(8)
if act_df.empty:
    st.caption("Brak wpisów aktywności. Gdy używasz przycisków/stron, log zostanie uzupełniony.")
else:
    st.dataframe(act_df, use_container_width=True, height=min(300, 28 * (len(act_df) + 2)))

# === NAZWA_SEKCJI === Stopka i pomoc ===
st.markdown("---")
st.caption(
    "💡 Tip: Kolejność pracy — **Upload → Metric Cards → Model Comparison → Feature Importance → Reports/Deploy**. "
    "Czat AI działa najlepiej z ustawionym `OPENAI_API_KEY`."
)
