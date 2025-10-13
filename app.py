"""
DataGenius PRO - Next-Gen Auto Data Scientist
Main Application Entry Point

Created by DataGenius Team
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from core.state_manager import StateManager
from frontend.app_layout import setup_page_config, render_sidebar
from frontend.styling.custom_css import load_custom_css
from core.utils import setup_logging

# Initialize logging
logger = setup_logging(__name__)


def main():
    """Main application entry point"""
    
    # Page configuration
    setup_page_config()
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize state manager
    state_manager = StateManager()
    state_manager.initialize_session()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("🚀 DataGenius PRO")
    st.markdown("### Next-Gen Auto Data Scientist z AI Agents")
    
    # Welcome message
    st.info("""
    👋 Witaj w **DataGenius PRO**!
    
    Twój inteligentny asystent do automatycznej analizy danych i Machine Learning.
    Wybierz stronę z menu po lewej stronie, aby rozpocząć.
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Sesje",
            value=state_manager.get_session_count(),
            delta="Wszystkie sesje"
        )
    
    with col2:
        st.metric(
            label="🤖 Modele",
            value=state_manager.get_models_count(),
            delta="Wytrenowane"
        )
    
    with col3:
        st.metric(
            label="📈 Pipeline'y",
            value=state_manager.get_pipelines_count(),
            delta="Wykonane"
        )
    
    with col4:
        st.metric(
            label="✨ Dokładność",
            value="98.5%",
            delta="+2.3%"
        )
    
    # Features overview
    st.markdown("---")
    st.markdown("## 🌟 Kluczowe Funkcje")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 Automatyczne AI Agents
        - **Data Understanding**: Auto-detekcja problemu
        - **EDA Agent**: Pełna analiza eksploracyjna
        - **ML Agent**: Automatyczne trenowanie modeli
        - **AI Mentor**: Asystent tłumaczący w PL
        """)
        
        st.markdown("""
        ### 📊 Continuous Monitoring
        - Wykrywanie model drift
        - Performance tracking
        - Automatyczne alerty
        - Retraining scheduler
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Pipeline Registry
        - Historia wszystkich sesji
        - Wersjonowanie modeli
        - Reprodukowalne eksperymenty
        - Export do MLflow/W&B
        """)
        
        st.markdown("""
        ### 📄 Auto Reports
        - Raporty EDA (PDF/HTML)
        - ML performance reports
        - Monitoring dashboards
        - Scheduled delivery
        """)
    
    # Quick actions
    st.markdown("---")
    st.markdown("## ⚡ Szybki Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Rozpocznij Nową Analizę", use_container_width=True):
            st.switch_page("pages/02_📊_Data_Upload.py")
    
    with col2:
        if st.button("📚 Zobacz Registry", use_container_width=True):
            st.switch_page("pages/08_📚_Registry.py")
    
    with col3:
        if st.button("🎓 Uruchom AI Mentora", use_container_width=True):
            st.switch_page("pages/06_🎓_AI_Mentor.py")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>DataGenius PRO v2.0 | Built with ❤️ by DataGenius Team</p>
        <p>Powered by Claude AI, PyCaret, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"Błąd aplikacji: {e}")
        st.stop()