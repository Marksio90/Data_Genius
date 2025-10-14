"""
DataGenius PRO - EDA Page
Exploratory Data Analysis
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from frontend.app_layout import render_header, render_error, render_success, render_warning
from core.state_manager import get_state_manager
from agents.eda.eda_orchestrator import EDAOrchestrator
import plotly.graph_objects as go


def main():
    """Main EDA page"""
    
    render_header(
        "🔍 Eksploracja Danych (EDA)",
        "Automatyczna analiza eksploracyjna danych"
    )
    
    state_manager = get_state_manager()
    
    # Check if data is loaded
    if not state_manager.has_data():
        render_warning("Najpierw załaduj dane w sekcji **📊 Data Upload**")
        if st.button("➡️ Przejdź do Upload"):
            st.switch_page("pages/02_📊_Data_Upload.py")
        return
    
    data = state_manager.get_data()
    target_column = state_manager.get_target_column()
    
    # Display data info
    st.subheader("📋 Informacje o danych")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Wiersze", f"{len(data):,}")
    
    with col2:
        st.metric("📊 Kolumny", len(data.columns))
    
    with col3:
        memory_mb = data.memory_usage(deep=True).sum() / 1024**2
        st.metric("💾 Rozmiar", f"{memory_mb:.2f} MB")
    
    with col4:
        if target_column:
            st.metric("🎯 Target", target_column)
        else:
            st.metric("🎯 Target", "Nie wybrano")
    
    st.markdown("---")
    
    # Run EDA
    if st.button("🚀 Rozpocznij analizę EDA", type="primary", use_container_width=True):
        run_eda_analysis(data, target_column, state_manager)
    
    # Display results if available
    if state_manager.is_eda_complete():
        display_eda_results(state_manager.get_eda_results())


def run_eda_analysis(data, target_column, state_manager):
    """Run EDA analysis"""
    
    with st.spinner("⏳ Analizuję dane... To może potrwać chwilę..."):
        try:
            # Initialize EDA Orchestrator
            eda_orchestrator = EDAOrchestrator()
            
            # Run EDA
            result = eda_orchestrator.run(
                data=data,
                target_column=target_column
            )
            
            if result.is_success():
                # Save results
                state_manager.set_eda_results(result.data)
                render_success("✅ Analiza EDA zakończona pomyślnie!")
                st.rerun()
            else:
                render_error("EDA nie powiodło się", "; ".join(result.errors))
        
        except Exception as e:
            render_error("Błąd podczas analizy EDA", str(e))


def display_eda_results(eda_results):
    """Display EDA results"""
    
    st.markdown("---")
    st.subheader("📊 Wyniki Analizy EDA")
    
    # Tabs for different analyses
    tabs = st.tabs([
        "📈 Statystyki",
        "🔍 Braki danych",
        "⚠️ Outliers",
        "🔗 Korelacje",
        "📉 Wizualizacje"
    ])
    
    # Tab 1: Statistics
    with tabs[0]:
        display_statistics(eda_results)
    
    # Tab 2: Missing data
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


def display_statistics(eda_results):
    """Display statistical analysis"""
    
    st.markdown("### 📊 Analiza Statystyczna")
    
    if "StatisticalAnalyzer" not in eda_results.get("eda_results", {}):
        st.info("Brak danych statystycznych")
        return
    
    stats = eda_results["eda_results"]["StatisticalAnalyzer"]
    
    # Overall stats
    if "overall" in stats:
        st.markdown("#### Ogólne statystyki")
        overall = stats["overall"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cechy numeryczne", overall.get("n_numeric", 0))
        
        with col2:
            st.metric("Cechy kategoryczne", overall.get("n_categorical", 0))
        
        with col3:
            st.metric("Rozmiar (MB)", f"{overall.get('memory_mb', 0):.2f}")
    
    # Numeric features
    if "numeric_features" in stats:
        st.markdown("#### Cechy numeryczne")
        
        numeric_stats = stats["numeric_features"].get("features", {})
        
        if numeric_stats:
            import pandas as pd
            
            # Convert to DataFrame for display
            stats_df = pd.DataFrame(numeric_stats).T
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("Brak cech numerycznych")


def display_missing_data(eda_results):
    """Display missing data analysis"""
    
    st.markdown("### 🔍 Analiza Brakujących Danych")
    
    if "MissingDataAnalyzer" not in eda_results.get("eda_results", {}):
        st.info("Brak analizy brakujących danych")
        return
    
    missing = eda_results["eda_results"]["MissingDataAnalyzer"]
    
    # Summary
    summary = missing.get("summary", {})
    
    if summary.get("total_missing", 0) == 0:
        st.success("🎉 Brak brakujących danych w zbiorze!")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Braki ogółem",
            f"{summary.get('total_missing', 0):,}"
        )
    
    with col2:
        st.metric(
            "Procent braków",
            f"{summary.get('missing_percentage', 0):.2f}%"
        )
    
    with col3:
        st.metric(
            "Kolumny z brakami",
            summary.get("n_columns_with_missing", 0)
        )
    
    # Columns with missing data
    columns_missing = missing.get("columns", [])
    
    if columns_missing:
        st.markdown("#### Kolumny z brakującymi danymi")
        
        import pandas as pd
        missing_df = pd.DataFrame(columns_missing)
        st.dataframe(missing_df, use_container_width=True)
    
    # Recommendations
    recommendations = missing.get("recommendations", [])
    
    if recommendations:
        st.markdown("#### 💡 Rekomendacje")
        for rec in recommendations:
            st.info(rec)


def display_outliers(eda_results):
    """Display outlier analysis"""
    
    st.markdown("### ⚠️ Analiza Outliers")
    
    if "OutlierDetector" not in eda_results.get("eda_results", {}):
        st.info("Brak analizy outliers")
        return
    
    outliers = eda_results["eda_results"]["OutlierDetector"]
    
    # Summary
    summary = outliers.get("summary", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Outliers ogółem",
            summary.get("total_outliers", 0)
        )
    
    with col2:
        st.metric(
            "Kolumny z outliers",
            summary.get("n_columns_with_outliers", 0)
        )
    
    with col3:
        methods = ", ".join(summary.get("methods_used", []))
        st.metric("Metody", methods[:20] + "..." if len(methods) > 20 else methods)
    
    # IQR method results
    if "iqr_method" in outliers:
        iqr = outliers["iqr_method"]
        columns_with_outliers = iqr.get("columns", {})
        
        if columns_with_outliers:
            st.markdown("#### Outliers (metoda IQR)")
            
            import pandas as pd
            outliers_df = pd.DataFrame([
                {
                    "Kolumna": col,
                    "Liczba outliers": info["n_outliers"],
                    "Procent": f"{info['percentage']:.2f}%",
                    "Dolna granica": f"{info['lower_bound']:.2f}",
                    "Górna granica": f"{info['upper_bound']:.2f}"
                }
                for col, info in columns_with_outliers.items()
            ])
            
            st.dataframe(outliers_df, use_container_width=True)


def display_correlations(eda_results):
    """Display correlation analysis"""
    
    st.markdown("### 🔗 Analiza Korelacji")
    
    if "CorrelationAnalyzer" not in eda_results.get("eda_results", {}):
        st.info("Brak analizy korelacji")
        return
    
    corr = eda_results["eda_results"]["CorrelationAnalyzer"]
    
    # Correlation matrix
    if "numeric_correlations" in corr:
        numeric_corr = corr["numeric_correlations"]
        
        if "correlation_matrix" in numeric_corr and numeric_corr["correlation_matrix"]:
            st.markdown("#### Macierz korelacji")
            
            import pandas as pd
            corr_matrix = pd.DataFrame(numeric_corr["correlation_matrix"])
            
            # Display heatmap would be here if we had the visualization
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1), 
                        use_container_width=True)
    
    # High correlations
    high_corr = corr.get("high_correlations", [])
    
    if high_corr:
        st.markdown("#### ⚠️ Silne korelacje (|r| > 0.8)")
        
        for pair in high_corr[:10]:  # Top 10
            st.warning(
                f"**{pair['feature1']}** ↔ **{pair['feature2']}**: "
                f"r = {pair['correlation']:.3f}"
            )
    
    # Recommendations
    recommendations = corr.get("recommendations", [])
    
    if recommendations:
        st.markdown("#### 💡 Rekomendacje")
        for rec in recommendations:
            st.info(rec)


def display_visualizations(eda_results):
    """Display visualizations"""
    
    st.markdown("### 📉 Wizualizacje")
    
    if "VisualizationEngine" not in eda_results.get("eda_results", {}):
        st.info("Brak wizualizacji")
        return
    
    viz = eda_results["eda_results"]["VisualizationEngine"]
    visualizations = viz.get("visualizations", {})
    
    if not visualizations:
        st.info("Brak dostępnych wizualizacji")
        return
    
    # Missing data plot
    if "missing_data" in visualizations:
        st.markdown("#### Brakujące wartości")
        st.plotly_chart(visualizations["missing_data"], use_container_width=True)
    
    # Correlation heatmap
    if "correlation_heatmap" in visualizations:
        st.markdown("#### Mapa korelacji")
        st.plotly_chart(visualizations["correlation_heatmap"], use_container_width=True)
    
    # Distribution plots
    if "distributions" in visualizations:
        st.markdown("#### Rozkłady cech numerycznych")
        distributions = visualizations["distributions"]
        
        if isinstance(distributions, list):
            for fig in distributions[:5]:  # Show first 5
                st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Więcej wizualizacji dostępnych w pełnym raporcie")


if __name__ == "__main__":
    main()