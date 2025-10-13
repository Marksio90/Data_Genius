"""
DataGenius PRO - Visualization Engine
Generates interactive visualizations for EDA
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from config.constants import COLOR_PALETTE_PRIMARY


class VisualizationEngine(BaseAgent):
    """
    Generates interactive Plotly visualizations for EDA
    """
    
    def __init__(self):
        super().__init__(
            name="VisualizationEngine",
            description="Generates interactive visualizations"
        )
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        Generate visualizations
        
        Args:
            data: Input DataFrame
            target_column: Target column (optional)
        
        Returns:
            AgentResult with visualization objects
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            visualizations = {}
            
            # Distribution plots for numeric features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                visualizations["distributions"] = self._create_distribution_plots(
                    data, numeric_cols
                )
            
            # Box plots for outlier detection
            if len(numeric_cols) > 0:
                visualizations["boxplots"] = self._create_boxplots(
                    data, numeric_cols
                )
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                visualizations["correlation_heatmap"] = self._create_correlation_heatmap(
                    data[numeric_cols]
                )
            
            # Categorical feature plots
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                visualizations["categorical_bars"] = self._create_categorical_plots(
                    data, categorical_cols
                )
            
            # Missing data visualization
            visualizations["missing_data"] = self._create_missing_data_plot(data)
            
            # Target analysis (if provided)
            if target_column and target_column in data.columns:
                visualizations["target_analysis"] = self._create_target_analysis(
                    data, target_column
                )
            
            result.data = {
                "visualizations": visualizations,
                "n_visualizations": len(visualizations),
            }
            
            self.logger.success(f"Generated {len(visualizations)} visualizations")
        
        except Exception as e:
            result.add_error(f"Visualization generation failed: {e}")
            self.logger.error(f"Visualization error: {e}", exc_info=True)
        
        return result
    
    def _create_distribution_plots(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_plots: int = 10
    ) -> List[go.Figure]:
        """Create distribution plots (histograms) for numeric features"""
        
        plots = []
        
        for col in columns[:max_plots]:
            fig = px.histogram(
                df,
                x=col,
                title=f"Rozkład: {col}",
                marginal="box",
                color_discrete_sequence=COLOR_PALETTE_PRIMARY,
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
            )
            
            plots.append(fig)
        
        return plots
    
    def _create_boxplots(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_plots: int = 10
    ) -> go.Figure:
        """Create box plots for numeric features"""
        
        n_cols = min(len(columns), max_plots)
        
        fig = make_subplots(
            rows=1,
            cols=n_cols,
            subplot_titles=columns[:n_cols],
        )
        
        for i, col in enumerate(columns[:n_cols], 1):
            fig.add_trace(
                go.Box(
                    y=df[col],
                    name=col,
                    marker_color=COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)],
                ),
                row=1,
                col=i,
            )
        
        fig.update_layout(
            title_text="Box Plots - Wykrywanie Outliers",
            showlegend=False,
            height=400,
        )
        
        return fig
    
    def _create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        
        corr_matrix = df.corr()
        
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu_r",
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Korelacja"),
            )
        )
        
        fig.update_layout(
            title="Macierz Korelacji",
            xaxis_title="Cechy",
            yaxis_title="Cechy",
            height=600,
        )
        
        return fig
    
    def _create_categorical_plots(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_plots: int = 5
    ) -> List[go.Figure]:
        """Create bar plots for categorical features"""
        
        plots = []
        
        for col in columns[:max_plots]:
            value_counts = df[col].value_counts().head(10)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Top 10 wartości: {col}",
                labels={"x": col, "y": "Liczność"},
                color_discrete_sequence=COLOR_PALETTE_PRIMARY,
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
            )
            
            plots.append(fig)
        
        return plots
    
    def _create_missing_data_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create missing data visualization"""
        
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            # No missing data
            fig = go.Figure()
            fig.add_annotation(
                text="Brak brakujących danych! 🎉",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20),
            )
            fig.update_layout(height=300)
            return fig
        
        missing_pct = (missing / len(df) * 100).round(2)
        
        fig = go.Figure(
            data=[
                go.Bar(
                    x=missing.values,
                    y=missing.index,
                    orientation="h",
                    text=[f"{pct}%" for pct in missing_pct],
                    textposition="auto",
                    marker_color=COLOR_PALETTE_PRIMARY[0],
                )
            ]
        )
        
        fig.update_layout(
            title="Brakujące Wartości",
            xaxis_title="Liczba brakujących",
            yaxis_title="Kolumna",
            height=max(300, len(missing) * 30),
        )
        
        return fig
    
    def _create_target_analysis(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, go.Figure]:
        """Create target variable analysis plots"""
        
        plots = {}
        target = df[target_column]
        
        # Distribution of target
        if pd.api.types.is_numeric_dtype(target):
            # Regression target
            fig = px.histogram(
                df,
                x=target_column,
                title=f"Rozkład zmiennej docelowej: {target_column}",
                marginal="box",
                color_discrete_sequence=COLOR_PALETTE_PRIMARY,
            )
            plots["target_distribution"] = fig
        else:
            # Classification target
            value_counts = target.value_counts()
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Rozkład klas: {target_column}",
                labels={"x": target_column, "y": "Liczność"},
                color_discrete_sequence=COLOR_PALETTE_PRIMARY,
            )
            plots["target_distribution"] = fig
        
        return plots