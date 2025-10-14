"""
DataGenius PRO - App Layout (PRO+++)
Główny layout, nawigacja i standardowe komponenty UI dla Streamlit.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import traceback

import streamlit as st

# === NAZWA_SEKCJI === Importy konfiguracyjne i integracje (defensywnie) ===
try:
    from config.settings import settings
except Exception:
    class _S:  # fallback minimalny
        DEBUG = False
        APP_VERSION = "0.0.0"
    settings = _S()  # type: ignore

try:
    from config.constants import APP_TITLE, APP_SUBTITLE, APP_ICON
except Exception:
    APP_TITLE, APP_SUBTITLE, APP_ICON = "DataGenius PRO", "Next-Gen Auto Data Scientist", "🧠"

# Motyw i CSS
try:
    from theme import ensure_and_apply, render_theme_switcher
except Exception:
    def ensure_and_apply(*args, **kwargs):  # type: ignore
        st.markdown("<!-- theme not available -->", unsafe_allow_html=True)
    def render_theme_switcher(*args, **kwargs):  # type: ignore
        pass

try:
    from custom_css import render_status_chip
except Exception:
    def render_status_chip(online: bool, label_ok: str = "ONLINE", label_off: str = "OFFLINE") -> None:  # type: ignore
        st.caption(f"Status: {'ONLINE' if online else 'OFFLINE'}")

# Ikony (opcjonalnie)
try:
    from icons import get_emoji
except Exception:
    def get_emoji(name: str, default: str = "🔹") -> str:  # type: ignore
        return default

# Stan aplikacji
try:
    from core.state_manager import get_state_manager
except Exception:
    get_state_manager = None  # type: ignore


# === NAZWA_SEKCJI === Page Config & Theme ===

def setup_page_config() -> None:
    """
    Ustawia konfigurację strony i stosuje motyw (runtime).
    Wywołuj NA SAMYM POCZĄTKU każdej strony.
    """
    # Bezpieczny set_page_config
    try:
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon=APP_ICON,
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                "Get Help": "https://github.com/datagenius-pro/help",
                "Report a bug": "https://github.com/datagenius-pro/issues",
                "About": f"{APP_TITLE} — Next-Gen Auto Data Scientist",
            },
        )
    except Exception:
        pass

    # Motyw (runtime) — zapewnia zmienne --dg-* oraz custom CSS
    try:
        ensure_and_apply(default_preset="violet", compact=False)
    except Exception:
        st.markdown("<!-- theme apply failed -->", unsafe_allow_html=True)


# === NAZWA_SEKCJI === SideBar (nawigacja, statusy, szybkie akcje) ===

def render_sidebar(show_theme_switcher: bool = False) -> None:
    """
    Renderuje panel boczny z nawigacją, statusem sesji i szybkimi akcjami.
    Parametry:
        show_theme_switcher: gdy True, pokaż panel zmiany motywu w sidebarze.
    """
    with st.sidebar:
        # Logo i tytuł
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.markdown(f"*{APP_SUBTITLE}*")
        st.markdown("---")

        # Nawigacja — informacja
        st.markdown("### 📍 Nawigacja")
        st.info(
            "Zalecana kolejność:\n"
            "1) 📊 Załaduj dane\n"
            "2) 🔍 Eksploracja (EDA)\n"
            "3) 🤖 Trenowanie modelu\n"
            "4) 📈 Wyniki\n"
            "5) 🎓 AI Mentor"
        )

        st.markdown("---")

        # Status sesji (defensywnie)
        st.markdown("### ℹ️ Status Sesji")

        # State manager
        state = get_state_manager() if callable(get_state_manager) else None

        # Session ID
        try:
            session_id = state.get_session_id() if state and hasattr(state, "get_session_id") else "unknown"
            st.caption(f"ID: `{str(session_id)[:8]}` …")
        except Exception:
            st.caption("ID: `n/d`")

        # Pipeline stage
        try:
            stage = state.get_pipeline_stage() if state and hasattr(state, "get_pipeline_stage") else "initialized"
        except Exception:
            stage = "initialized"

        stage_icons: Dict[str, str] = {
            "initialized": "🟡",
            "data_loaded": "🟢",
            "eda_complete": "🟢",
            "training": "🔄",
            "training_complete": "✅",
            "failed": "🔴",
        }
        st.caption(f"Etap: {stage_icons.get(stage, '⚪')} {stage}")

        # Data / EDA / Model flags
        try:
            if state and hasattr(state, "has_data") and state.has_data():
                st.success("✅ Dane załadowane")
            else:
                st.warning("⚠️ Brak danych")
        except Exception:
            st.info("ℹ️ Status danych: n/d")

        try:
            if state and hasattr(state, "is_eda_complete") and state.is_eda_complete():
                st.success("✅ EDA zakończone")
        except Exception:
            pass

        try:
            if state and hasattr(state, "is_model_trained") and state.is_model_trained():
                st.success("✅ Model wytrenowany")
        except Exception:
            pass

        # AI status chip (ONLINE gdy mamy klucz)
        st.markdown("---")
        st.caption("🧠 AI Status")
        online = _has_openai_key()
        render_status_chip(online, label_ok="AI: ONLINE", label_off="AI: OFFLINE")

        # Theme switcher (opcjonalny)
        if show_theme_switcher:
            st.markdown("---")
            st.caption("🎨 Motyw (runtime)")
            try:
                render_theme_switcher(expanded=False)
            except Exception:
                st.caption("Motyw: panel niedostępny.")

        st.markdown("---")

        # Szybkie akcje
        st.markdown("### ⚡ Szybkie akcje")
        try:
            if st.button(f"{get_emoji('refresh')} Nowa sesja", use_container_width=True):
                if state and hasattr(state, "clear"):
                    state.clear()
                st.rerun()
        except Exception:
            st.button("🔄 Odśwież widok", use_container_width=True, on_click=lambda: st.rerun())

        st.button(f"{get_emoji('save')} Zapisz stan", use_container_width=True, disabled=True)
        st.caption("💡 Persist do pliku/DB wkrótce (adapter storage).")

        st.markdown("---")

        # Settings / Debug
        st.markdown("### ⚙️ Ustawienia")
        if getattr(settings, "DEBUG", False):
            st.caption("🐛 Debug Mode: ON")

        st.markdown("---")

        # Zasoby / linki
        st.markdown("### 📚 Zasoby")
        st.markdown(
            """
            - [📖 Dokumentacja](https://docs.datagenius-pro.com)
            - [💬 Discord](https://discord.gg/datagenius)
            - [🐙 GitHub](https://github.com/datagenius-pro)
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption(f"v{getattr(settings, 'APP_VERSION', '0.0.0')} | Built with ❤️")


# === NAZWA_SEKCJI === Header / Footer ===

def render_header(title: str, subtitle: str = "") -> None:
    """
    Renderuje nagłówek strony z poziomą linią.
    """
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")


def render_footer() -> None:
    """
    Renderuje stopkę strony (lekka, brandowa).
    """
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center;color:#666;padding:16px;'>
            <p>DataGenius PRO — Next-Gen Auto Data Scientist</p>
            <p>Powered by DataGenius Engine & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# === NAZWA_SEKCJI === Komponenty komunikatów i metryk (spójne) ===

def render_error(error_message: str, details: str = "") -> None:
    """
    Wyświetla błąd z możliwością rozwinięcia szczegółów (traceback).
    """
    st.error(f"❌ **Błąd**: {error_message}")
    if details:
        with st.expander("📋 Szczegóły błędu"):
            st.code(details)
    else:
        # jeżeli mamy aktywny wyjątek — pokaż traceback
        tb = traceback.format_exc()
        if "NoneType: None" not in tb:
            with st.expander("📋 Traceback"):
                st.code(tb)

def render_success(message: str) -> None:
    st.success(f"✅ {message}")

def render_warning(message: str) -> None:
    st.warning(f"⚠️ {message}")

def render_info(message: str) -> None:
    st.info(f"ℹ️ {message}")

def render_progress_bar(progress: float, text: str = "") -> None:
    st.progress(progress, text=text)

def render_metric_card(label: str, value: str, delta: Optional[str] = None) -> None:
    st.metric(label=label, value=value, delta=delta)


# === NAZWA_SEKCJI === Pomocnicze: detekcja klucza AI ===

def _has_openai_key() -> bool:
    """
    Zwraca True jeśli klucz OPENAI_API_KEY jest dostępny w secrets/session/env.
    """
    # st.secrets
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
        if key:
            return True
    except Exception:
        pass
    # session_state
    key = st.session_state.get("openai_api_key")
    if key:
        return True
    # env (nie sprawdzamy tu, bo Streamlit cloud i tak mapuje secrets do env)
    import os
    return bool(os.environ.get("OPENAI_API_KEY"))
