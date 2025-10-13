"""
DataGenius PRO - Data Upload Page
Upload and preview data
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from frontend.app_layout import render_header, render_error, render_success
from core.state_manager import get_state_manager
from core.data_loader import get_data_loader
from core.data_validator import DataValidator
from agents.data_understanding.schema_analyzer import SchemaAnalyzer
from agents.data_understanding.target_detector import TargetDetector
from agents.data_understanding.problem_classifier import ProblemClassifier
from config.constants import SAMPLE_DATASETS, MAX_PREVIEW_ROWS


def main():
    """Main page function"""
    
    render_header(
        "📊 Załaduj Dane",
        "Załaduj swoje dane lub użyj przykładowego zbioru danych"
    )
    
    state_manager = get_state_manager()
    data_loader = get_data_loader()
    
    # Data source selection
    st.subheader("1️⃣ Wybierz źródło danych")
    
    data_source = st.radio(
        "Źródło danych:",
        ["📤 Upload pliku", "📚 Przykładowe dane"],
        horizontal=True
    )
    
    data = None
    
    if data_source == "📤 Upload pliku":
        data = handle_file_upload(data_loader)
    else:
        data = handle_sample_data(data_loader)
    
    # If data loaded, show preview and analysis
    if data is not None:
        st.markdown("---")
        st.subheader("2️⃣ Podgląd danych")
        
        show_data_preview(data)
        
        # Save to state
        if st.button("✅ Zatwierdź i kontynuuj", type="primary", use_container_width=True):
            state_manager.set_data(data)
            render_success("Dane załadowane pomyślnie!")
            
            # Auto-analyze
            with st.spinner("Analizuję dane..."):
                analyze_data(data, state_manager)
            
            st.info("👉 Przejdź do **🔍 EDA** aby zobaczyć analizę danych!")


def handle_file_upload(data_loader):
    """Handle file upload"""
    
    uploaded_file = st.file_uploader(
        "Przeciągnij plik lub kliknij, aby wybrać",
        type=["csv", "xlsx", "xls", "json"],
        help="Obsługiwane formaty: CSV, Excel (XLSX/XLS), JSON"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Ładuję dane..."):
                # Save temp file
                temp_path = Path(f"/tmp/{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load data
                data = data_loader.load(temp_path)
                
                render_success(f"Załadowano {len(data)} wierszy i {len(data.columns)} kolumn")
                
                return data
        
        except Exception as e:
            render_error("Nie udało się załadować pliku", str(e))
            return None
    
    return None


def handle_sample_data(data_loader):
    """Handle sample dataset selection"""
    
    st.markdown("### Wybierz przykładowy zbiór danych:")
    
    # Display sample datasets
    cols = st.columns(3)
    
    for i, (dataset_id, dataset_info) in enumerate(SAMPLE_DATASETS.items()):
        col = cols[i % 3]
        
        with col:
            st.markdown(f"#### {dataset_info['name']}")
            st.caption(dataset_info['description'])
            st.caption(f"📊 {dataset_info['samples']} próbek")
            st.caption(f"🔢 {dataset_info['features']} cech")
            st.caption(f"🎯 Problem: {dataset_info['problem_type']}")
            
            if st.button(f"Załaduj {dataset_info['name']}", key=f"load_{dataset_id}"):
                try:
                    with st.spinner(f"Ładuję {dataset_info['name']}..."):
                        data = data_loader.load_sample(dataset_id)
                        st.session_state['selected_sample'] = dataset_id
                        render_success(f"Załadowano {dataset_info['name']}")
                        return data
                
                except Exception as e:
                    render_error(f"Nie udało się załadować {dataset_info['name']}", str(e))
                    return None
    
    # If sample was selected in session
    if 'selected_sample' in st.session_state:
        dataset_id = st.session_state['selected_sample']
        try:
            data = data_loader.load_sample(dataset_id)
            return data
        except:
            return None
    
    return None


def show_data_preview(data: pd.DataFrame):
    """Show data preview"""
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Wiersze", f"{len(data):,}")
    
    with col2:
        st.metric("📊 Kolumny", len(data.columns))
    
    with col3:
        memory_mb = data.memory_usage(deep=True).sum() / 1024**2
        st.metric("💾 Rozmiar", f"{memory_mb:.2f} MB")
    
    with col4:
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("🔍 Braki", f"{missing_pct:.1f}%")
    
    st.markdown("---")
    
    # Data preview
    st.markdown("### 👀 Podgląd danych")
    
    n_preview = st.slider(
        "Liczba wierszy do wyświetlenia:",
        min_value=5,
        max_value=min(100, len(data)),
        value=min(10, len(data))
    )
    
    st.dataframe(data.head(n_preview), use_container_width=True)
    
    # Column info
    with st.expander("📋 Informacje o kolumnach"):
        col_info = []
        for col in data.columns:
            col_info.append({
                "Kolumna": col,
                "Typ": str(data[col].dtype),
                "Unikalne": data[col].nunique(),
                "Braki": data[col].isnull().sum(),
                "Braki %": f"{data[col].isnull().sum() / len(data) * 100:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)


def analyze_data(data: pd.DataFrame, state_manager):
    """Auto-analyze data"""
    
    try:
        # Schema analysis
        schema_analyzer = SchemaAnalyzer()
        schema_result = schema_analyzer.run(data=data)
        
        if not schema_result.is_success():
            st.warning("Nie udało się przeanalizować schematu danych")
            return
        
        column_info = schema_result.data["columns"]
        
        # Target detection
        target_detector = TargetDetector()
        target_result = target_detector.run(
            data=data,
            column_info=column_info
        )
        
        if target_result.is_success() and target_result.data["target_column"]:
            target_column = target_result.data["target_column"]
            problem_type = target_result.data["problem_type"]
            
            state_manager.set_target_column(target_column)
            state_manager.set_problem_type(problem_type)
            
            st.success(f"🎯 Wykryto kolumnę docelową: **{target_column}**")
            st.info(f"📋 Typ problemu: **{problem_type}**")
        else:
            st.info("💡 Możesz ręcznie wybrać kolumnę docelową w następnym kroku")
    
    except Exception as e:
        st.warning(f"Automatyczna analiza nie powiodła się: {e}")


if __name__ == "__main__":
    main()