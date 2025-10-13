"""
DataGenius PRO - App Layout
Main layout and navigation for Streamlit app
"""

import streamlit as st
from config.settings import settings
from config.constants import APP_TITLE, APP_SUBTITLE, APP_ICON


def setup_page_config():
    """Setup Streamlit page configuration"""
    
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/datagenius-pro/help',
            'Report a bug': 'https://github.com/datagenius-pro/issues',
            'About': f'{APP_TITLE} - Next-Gen Auto Data Scientist'
        }
    )


def render_sidebar():
    """Render sidebar with navigation and info"""
    
    with st.sidebar:
        # Logo and title
        st.markdown(f"# {APP_ICON} {APP_TITLE}")
        st.markdown(f"*{APP_SUBTITLE}*")
        st.markdown("---")
        
        # Navigation info
        st.markdown("### 📍 Nawigacja")
        st.info("""
        Użyj menu po lewej stronie, aby poruszać się między stronami aplikacji.
        
        **Zalecana kolejność:**
        1. 📊 Załaduj dane
        2. 🔍 Eksploracja (EDA)
        3. 🤖 Trenowanie modelu
        4. 📈 Wyniki
        5. 🎓 AI Mentor
        """)
        
        st.markdown("---")
        
        # Session info
        from core.state_manager import get_state_manager
        state_manager = get_state_manager()
        
        st.markdown("### ℹ️ Status Sesji")
        
        # Session ID
        session_id = state_manager.get_session_id()
        st.caption(f"ID: `{session_id[:8]}`...")
        
        # Pipeline stage
        stage = state_manager.get_pipeline_stage()
        stage_icons = {
            "initialized": "🟡",
            "data_loaded": "🟢",
            "eda_complete": "🟢",
            "training": "🔄",
            "training_complete": "✅",
            "failed": "🔴"
        }
        icon = stage_icons.get(stage, "⚪")
        st.caption(f"Etap: {icon} {stage}")
        
        # Data status
        if state_manager.has_data():
            st.success("✅ Dane załadowane")
        else:
            st.warning("⚠️ Brak danych")
        
        # EDA status
        if state_manager.is_eda_complete():
            st.success("✅ EDA zakończone")
        
        # Model status
        if state_manager.is_model_trained():
            st.success("✅ Model wytrenowany")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Szybkie akcje")
        
        if st.button("🔄 Nowa sesja", use_container_width=True):
            state_manager.clear()
            st.rerun()
        
        if st.button("💾 Zapisz stan", use_container_width=True):
            st.info("Funkcja zapisu będzie dostępna wkrótce!")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Ustawienia")
        
        # Theme toggle (info only, actual theme in .streamlit/config.toml)
        st.caption("🎨 Motyw aplikacji")
        st.caption("Zmień w Settings → Theme")
        
        # Debug mode
        if settings.DEBUG:
            st.caption("🐛 Debug Mode: ON")
        
        st.markdown("---")
        
        # Footer
        st.markdown("### 📚 Zasoby")
        st.markdown("""
        - [📖 Dokumentacja](https://docs.datagenius-pro.com)
        - [💬 Discord](https://discord.gg/datagenius)
        - [🐙 GitHub](https://github.com/datagenius-pro)
        """)
        
        st.markdown("---")
        st.caption(f"v{settings.APP_VERSION} | Built with ❤️")


def render_header(title: str, subtitle: str = ""):
    """
    Render page header
    
    Args:
        title: Page title
        subtitle: Page subtitle (optional)
    """
    
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")


def render_footer():
    """Render page footer"""
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>DataGenius PRO - Next-Gen Auto Data Scientist</p>
        <p>Powered by Claude AI, PyCaret, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


def render_error(error_message: str, details: str = ""):
    """
    Render error message
    
    Args:
        error_message: Main error message
        details: Additional details (optional)
    """
    
    st.error(f"❌ **Błąd**: {error_message}")
    
    if details:
        with st.expander("📋 Szczegóły błędu"):
            st.code(details)


def render_success(message: str):
    """
    Render success message
    
    Args:
        message: Success message
    """
    
    st.success(f"✅ {message}")


def render_warning(message: str):
    """
    Render warning message
    
    Args:
        message: Warning message
    """
    
    st.warning(f"⚠️ {message}")


def render_info(message: str):
    """
    Render info message
    
    Args:
        message: Info message
    """
    
    st.info(f"ℹ️ {message}")


def render_progress_bar(progress: float, text: str = ""):
    """
    Render progress bar
    
    Args:
        progress: Progress value (0-1)
        text: Progress text
    """
    
    st.progress(progress, text=text)


def render_metric_card(label: str, value: str, delta: str = None):
    """
    Render metric card
    
    Args:
        label: Metric label
        value: Metric value
        delta: Change indicator (optional)
    """
    
    st.metric(label=label, value=value, delta=delta)