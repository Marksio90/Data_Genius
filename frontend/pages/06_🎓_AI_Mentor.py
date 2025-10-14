# 06_🎓_AI_Mentor.py
"""
DataGenius PRO — AI Mentor (PRO+++)
Interaktywny mentor projektu: Q&A o EDA/ML/metrics/next-steps, z kontekstem aplikacji.
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
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
        st.header(title)
        if subtitle: st.caption(subtitle)
    def render_error(title: str, detail: Optional[str] = None) -> None:
        st.error(title + (f": {detail}" if detail else ""))
    def render_success(msg: str) -> None:
        st.success(msg)
    def render_warning(msg: str) -> None:
        st.warning(msg)

try:
    from core.state_manager import get_state_manager
except Exception:
    get_state_manager = None  # defensywnie


# === NAZWA_SEKCJI === Page Config (bezpiecznie) ===
try:
    st.set_page_config(page_title="🎓 AI Mentor — DataGenius PRO+++", page_icon="🎓", layout="wide")
except Exception:
    pass

# === NAZWA_SEKCJI === Logger (lekki) ===
try:
    from src.utils.logger import get_logger  # opcjonalnie
    log = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ai_mentor")

# === NAZWA_SEKCJI === Security Gate (ONLINE/OFFLINE) ===

def _get_openai_key() -> Optional[str]:
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
        if key: return key
    except Exception:
        pass
    if st.session_state.get("openai_api_key"):  # ustawiane np. na Home
        return st.session_state.get("openai_api_key")
    return os.environ.get("OPENAI_API_KEY")

def _ai_status_chip(online: bool) -> None:
    st.markdown(
        f"""
        <div style="text-align:right;">
          <span style="display:inline-block;padding:2px 10px;border-radius:999px;
                       background:{'#15a34a1a' if online else '#f59e0b1a'};
                       border:1px solid {'#16a34a' if online else '#f59e0b'};
                       color:{'#16a34a' if online else '#b45309'};
                       font-weight:600;font-size:12px">
            AI: {'ONLINE' if online else 'OFFLINE'}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# === NAZWA_SEKCJI === Kontekst projektu (snapshot) ===

def _data_kpis(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"rows": 0, "cols": 0, "mem_mb": 0.0, "miss_pct": 0.0}
    rows = int(len(df)); cols = int(df.shape[1])
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024**2))
    miss_pct = float(df.isna().sum().sum() / max(1, rows * cols) * 100.0)
    return {"rows": rows, "cols": cols, "mem_mb": round(mem_mb, 2), "miss_pct": round(miss_pct, 2)}

def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        return d.get(key, default) if isinstance(d, dict) else default
    except Exception:
        return default

def _build_context_snapshot() -> Dict[str, Any]:
    """
    Tworzy zwięzły snapshot (bez surowych danych) do wstrzyknięcia w system prompt Mentora.
    """
    state = get_state_manager()() if callable(get_state_manager) else None

    df = state.get_data() if state and hasattr(state, "get_data") and state.has_data() else st.session_state.get("raw_df")
    kpi = _data_kpis(df)
    target = state.get_target_column() if state and hasattr(state, "get_target_column") else st.session_state.get("target_column")
    problem = state.get_problem_type() if state and hasattr(state, "get_problem_type") else st.session_state.get("problem_type")

    eda_results = st.session_state.get("eda_results")
    ml_training = st.session_state.get("ml_training", {})          # {"table": df, "report": dict}
    model_comp  = st.session_state.get("model_comparison", {})     # {"table": df, "meta": dict}
    fi_bundle   = st.session_state.get("feature_importance", {})   # {"table": df, "meta": dict}
    pipeline    = st.session_state.get("pipeline_state", {})

    # Skróty EDA
    eda_short: Dict[str, Any] = {}
    try:
        if eda_results and "eda_results" in eda_results:
            eda = eda_results["eda_results"]
            stats = _safe_get(eda.get("StatisticalAnalyzer", {}), "overall", {})
            miss = _safe_get(eda.get("MissingDataAnalyzer", {}), "summary", {})
            eda_short = {
                "n_numeric": stats.get("n_numeric"),
                "n_categorical": stats.get("n_categorical"),
                "missing_pct": miss.get("missing_percentage"),
                "n_columns_missing": miss.get("n_columns_with_missing"),
            }
    except Exception:
        pass

    # Model summary
    report = _safe_get(ml_training, "report", {})
    best_key = _safe_get(report, "best_key") or _safe_get(_safe_get(model_comp, "meta", {}), "best_key")
    primary_metric = _safe_get(report, "primary_metric") or _safe_get(_safe_get(model_comp, "meta", {}), "scoring")
    test_metrics = _safe_get(report, "test_metrics", {})

    # Top FI (do 15)
    fi_df = _safe_get(fi_bundle, "table")
    top_fi: List[Dict[str, Any]] = []
    if isinstance(fi_df, pd.DataFrame) and not fi_df.empty and "feature" in fi_df.columns:
        view = fi_df.copy().head(15)
        score_col = "importance_norm" if "importance_norm" in view.columns else ("importance" if "importance" in view.columns else None)
        for _, r in view.iterrows():
            top_fi.append({"feature": str(r["feature"]), "score": float(r.get(score_col)) if score_col else None})

    # Pipeline quick
    pipe_short = {"status": pipeline.get("status"), "stage": pipeline.get("stage"), "n_stages": len(pipeline.get("stages", []))} if pipeline else {}

    return {
        "project_overview": {
            "kpi": kpi,
            "target": target,
            "problem": problem,
        },
        "eda": eda_short,
        "model": {
            "best_key": best_key,
            "primary_metric": primary_metric,
            "test_metrics": test_metrics,
        },
        "feature_importance": {"top": top_fi},
        "pipeline": pipe_short,
    }

# === NAZWA_SEKCJI === LLM Client (OpenAI kompat.) + Fallback OFFLINE ===

def _get_llm_client():
    """
    Zwraca klienta OpenAI (chat.completions) jeśli dostępny, inaczej None.
    """
    api_key = _get_openai_key()
    if not api_key:
        return None
    try:
        import openai  # OpenAI python SDK v1.x
        client = openai.OpenAI(api_key=api_key)
        return client
    except Exception as e:
        log.warning(f"OpenAI client init failed: {e}")
        return None

def _trim_history(messages: List[Dict[str, str]], max_tokens_est: int = 6000) -> List[Dict[str, str]]:
    """
    Bardzo uproszczone przycinanie historii: limit liczby wiadomości.
    (Estymacja tokenów jest trudna bez tiktoken — przycinamy do N ostatnich tur).
    """
    MAX_TURNS = 16  # 16 tur ~ spokojnie < 6-8k tokenów dla krótkich promptów
    if len(messages) <= MAX_TURNS:
        return messages
    # zachowaj system + pierwszą user + ostatnie 14 wpisów
    head = [messages[0]]
    tail = messages[-(MAX_TURNS-1):]
    return head + tail

def _system_prompt(context: Dict[str, Any]) -> str:
    """
    System prompt mentora — styl PRO+++: konkretnie, defensywnie, bez danych wrażliwych.
    """
    return (
        "Jesteś AI Mentorem klasy PRO+++, ekspertem MLOps/Data Science/Streamlit.\n"
        "Zasady: odpowiadaj zwięźle, konkretnie, w punktach. Wyjaśniaj metryki, proponuj next-steps, "
        "zawsze wskazuj ryzyka (data leakage, overfitting, data quality). Nie wymyślaj danych, "
        "polegaj na przekazanym kontekście. Jeśli czegoś brakuje — zaproponuj jak to zdobyć. "
        "Nie ujawniaj PII/surowych wierszy.\n\n"
        f"Kontekst projektu (JSON):\n{json.dumps(context, ensure_ascii=False)}"
    )

def _call_llm(messages: List[Dict[str, str]], model: str, temperature: float, timeout_s: int = 40, max_retries: int = 2) -> str:
    """
    Wywołanie chat.completions z retry/backoff. Zwraca treść odpowiedzi lub rzuca wyjątek.
    """
    client = _get_llm_client()
    if client is None:
        raise RuntimeError("Brak klienta LLM (OFFLINE).")

    delay = 1.5
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            with st.spinner("🤖 AI myśli…"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout_s,
                )
            return resp.choices[0].message.content or "Brak treści odpowiedzi."
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 2
    raise last_err or RuntimeError("Nieznany błąd LLM")

def _offline_response(user_msg: str, context: Dict[str, Any]) -> str:
    """
    Fallback odpowiedzi OFFLINE — reguły i gotowe szkielety.
    """
    ov = context.get("project_overview", {})
    kpi = ov.get("kpi", {})
    target = ov.get("target") or "nieustalony"
    problem = ov.get("problem") or "n/d"

    hints = []
    # Propozycje kroków na podstawie problemu
    if problem == "classification":
        hints += [
            "Zweryfikuj balans klas (class imbalance) — rozważ stratified split lub wagi klas.",
            "Sprawdź metryki: ROC AUC, F1 weighted; unikaj accuracy przy niezbalansowanych danych.",
            "Zweryfikuj ważne cechy (FI) i możliwy data leakage.",
        ]
    elif problem == "regression":
        hints += [
            "Użyj metryk: R², RMSE, MAE — raportuj wszystkie trzy.",
            "Sprawdź rozkład residuali i heteroskedastyczność.",
            "Przetestuj transformacje cech (log, Box-Cox) oraz interakcje.",
        ]
    else:
        hints += [
            "Zidentyfikuj najpierw typ problemu (classification vs regression).",
            "Zdefiniuj target i walidację.",
        ]

    size_info = f"Dane: {kpi.get('rows','n/d')} wierszy × {kpi.get('cols','n/d')} kolumn; braki ~ {kpi.get('miss_pct','n/d')}%."
    guidance = [
        f"Target: **{target}**, typ problemu: **{problem}**.",
        size_info,
        "Rekomendowane następne kroki:",
        *[f"- {h}" for h in hints[:4]],
        "Skonfiguruj walidację: train/val/test z losowaniem powtarzalnym i raportem metryk.",
    ]

    # Proste reguły odpowiadania na popularne komendy
    msg = user_msg.lower()
    if "kolejne kroki" in msg or "next step" in msg or "co dalej" in msg:
        return "\n".join(guidance)
    if "metryk" in msg or "metrics" in msg:
        return (
            "Dobór metryk:\n"
            "- **Klasyfikacja**: ROC AUC (priorytet), F1 weighted, Accuracy (ostrożnie przy niezbalansowanych).\n"
            "- **Regresja**: RMSE (czuła na outliers), MAE (odporna), R² (globalne dopasowanie).\n"
            "Raportuj na walidacji i teście, z odchyleniem (jeśli CV)."
        )
    if "leak" in msg or "leakage" in msg or "przeciek" in msg:
        return (
            "Data leakage — checklist:\n"
            "1) Czy cechy powstają z docelowej zmiennej lub przyszłej informacji?\n"
            "2) Czy preprocessing/skalowanie był dopasowany na całym zbiorze (zamiast tylko na train)?\n"
            "3) Czy target encoding był robiony w ramach CV/splita?\n"
            "4) Czy identyfikatory czasowe są traktowane poprawnie (time-based split)?"
        )
    return "\n".join(guidance[:3] + [f"- Odpowiadam w trybie OFFLINE — podaj bardziej szczegółowe pytanie, a zasugeruję konkretne działania."])

# === NAZWA_SEKCJI === Utrzymanie historii czatu ===

def _init_chat() -> None:
    st.session_state.setdefault("mentor_chat", [])
    # Inicjalna wiadomość systemowa nie jest wyświetlana, ale trzymamy ją w buforze do LLM
    st.session_state.setdefault("mentor_system", _system_prompt(_build_context_snapshot()))

def _append_chat(role: str, content: str) -> None:
    st.session_state["mentor_chat"].append({"role": role, "content": content, "ts": time.time()})
    # limit historii (20 wpisów user/assistant)
    if len(st.session_state["mentor_chat"]) > 40:
        st.session_state["mentor_chat"] = st.session_state["mentor_chat"][-40:]

def _render_history() -> None:
    for m in st.session_state.get("mentor_chat", []):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(m["content"])

# === NAZWA_SEKCJI === UI GŁÓWNE ===

def main() -> None:
    render_header("🎓 AI Mentor", "Zadawaj pytania o swoje dane, EDA, metryki i modele — mentor wykorzysta kontekst projektu.")

    _init_chat()
    context = _build_context_snapshot()
    online = _get_openai_key() is not None

    # Panel statusu
    cA, cB, cC = st.columns([0.6, 0.2, 0.2])
    with cA:
        st.caption("Mentor działa na streszczeniu projektu (bez surowych rekordów).")
    with cB:
        st.caption("Tryb")
        _ai_status_chip(online)
    with cC:
        st.caption("Kontekst")
        if st.button("🔄 Odśwież snapshot", use_container_width=True):
            st.session_state["mentor_system"] = _system_prompt(_build_context_snapshot())
            render_success("Zaktualizowano kontekst mentora.")

    # Parametry LLM
    with st.expander("⚙️ Ustawienia AI", expanded=False):
        model = st.text_input("Model", value=os.environ.get("DG_LLM_MODEL", "gpt-4o-mini"))
        temperature = st.slider("Temperatura", min_value=0.0, max_value=1.2, value=0.2, step=0.05)
        st.caption("Niższa temperatura = bardziej deterministyczne odpowiedzi.")

    # Szybkie podpowiedzi
    st.markdown("#### 💡 Szybkie podpowiedzi")
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("🧭 Zaproponuj kolejne kroki", use_container_width=True):
        _append_chat("user", "Zaproponuj kolejne kroki w projekcie na podstawie obecnego kontekstu.")
    if p2.button("📏 Wyjaśnij metryki", use_container_width=True):
        _append_chat("user", "Wyjaśnij, które metryki są istotne dla mojego problemu i jak je interpretować.")
    if p3.button("🔥 Omów Feature Importance", use_container_width=True):
        _append_chat("user", "Zinterpretuj top Feature Importance i zasugeruj inżynierię cech.")
    if p4.button("🛡️ Sprawdź ryzyka", use_container_width=True):
        _append_chat("user", "Podaj checklistę ryzyk: data leakage, overfitting, jakość danych, błędy walidacji.")

    # Render historii + input
    _render_history()

    user_msg = st.chat_input("Napisz pytanie do mentora…")
    if user_msg:
        _append_chat("user", user_msg)

    # Odpowiedź mentora dla najnowszego nieobsłużonego pytania
    pending = [m for m in st.session_state["mentor_chat"] if m["role"] == "user" and not m.get("answered")]
    if pending:
        latest = pending[-1]
        messages = [{"role": "system", "content": st.session_state["mentor_system"]}]
        # doklej przyciętą historię (bez system)
        for m in st.session_state["mentor_chat"]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages = _trim_history(messages)

        try:
            if online:
                reply = _call_llm(messages=messages, model=model, temperature=float(temperature))
            else:
                raise RuntimeError("OFFLINE")
        except Exception as e:
            log.warning(f"LLM offline/failure, using fallback: {e}")
            reply = _offline_response(latest["content"], _build_context_snapshot())

        _append_chat("assistant", reply)
        latest["answered"] = True  # oznacz jako obsłużone
        # Odśwież UI
        st.rerun()

    # Notatka compliance
    st.markdown("---")
    st.caption("ℹ️ Mentor AI korzysta wyłącznie z metadanych i streszczeń (bez wysyłania surowych rekordów). Traktuj odpowiedzi jako rekomendacje — weryfikuj w projekcie.")

# === NAZWA_SEKCJI === Wejście modułu ===
if __name__ == "__main__":
    main()
