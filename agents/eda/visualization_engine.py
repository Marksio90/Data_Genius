# === OPIS MODUŁU ===
"""
DataGenius PRO - Visualization Engine (PRO+++)
Generates robust, interactive Plotly visualizations for EDA with defensiveness,
downsampling, adaptive layouts, and consistent styling.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from core.base_agent import BaseAgent, AgentResult

# === PALETA KOLORÓW (fallback jeśli config nie istnieje) ===
try:
    from config.constants import COLOR_PALETTE_PRIMARY  # type: ignore
except Exception:
    COLOR_PALETTE_PRIMARY = [
        "#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706",
        "#10B981", "#F43F5E", "#0EA5E9", "#9333EA", "#EF4444"
    ]


# === KONFIG ===
@dataclass(frozen=True)
class VizConfig:
    max_points: int = 100_000              # downsampling threshold for dense plots
    max_plots_numeric: int = 10            # max hist/box plots
    max_plots_categorical: int = 6         # max category plots
    top_k_categories: int = 10             # top-k bars for categorical
    heatmap_max_features: int = 40         # cap features to avoid huge heatmaps
    correlation_method: str = "pearson"    # "pearson" | "spearman"
    annotate_heatmap: bool = True          # show correlation values
    target_rel_top_features: int = 3       # how many top features to relate with target
    random_state: int = 42                 # for sampling reproducibility


class VisualizationEngine(BaseAgent):
    """
    Generates interactive Plotly visualizations for EDA (robust & scalable).
    """

    def __init__(self, config: Optional[VizConfig] = None) -> None:
        super().__init__(
            name="VisualizationEngine",
            description="Generates interactive visualizations"
        )
        self.config = config or VizConfig()

    # === WYKONANIE GŁÓWNE ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Generate visualizations.

        Args:
            data: Input DataFrame
            target_column: Target column (optional)

        Returns:
            AgentResult with visualization objects
        """
        result = AgentResult(agent_name=self.name)

        try:
            # Walidacja wejścia
            if data is None or not isinstance(data, pd.DataFrame):
                msg = "VisualizationEngine: 'data' must be a pandas DataFrame."
                result.add_error(msg)
                self.logger.error(msg)
                return result
            if data.empty:
                result.add_warning("Empty DataFrame — no visualizations produced.")
                result.data = {"visualizations": {}, "n_visualizations": 0}
                return result

            visualizations: Dict[str, Any] = {}
            cfg = self.config

            # --- Selekcje kolumn ---
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

            # --- Downsampling (kopiujemy próbkę do rysowania intensywnych wykresów) ---
            df_plot = self._maybe_sample(data, cfg.max_points)

            # --- Distribution plots (numeric) ---
            if numeric_cols:
                visualizations["distributions"] = self._create_distribution_plots(
                    df_plot, numeric_cols, max_plots=cfg.max_plots_numeric
                )

                visualizations["boxplots"] = self._create_boxplots(
                    df_plot, numeric_cols, max_plots=cfg.max_plots_numeric
                )

            # --- Correlation heatmap (numeric) ---
            if len(numeric_cols) > 1:
                visualizations["correlation_heatmap"] = self._create_correlation_heatmap(
                    data[numeric_cols],  # liczymy na pełnym df (precyzja ważniejsza)
                    method=cfg.correlation_method,
                    max_features=cfg.heatmap_max_features,
                    annotate=cfg.annotate_heatmap
                )

            # --- Categorical bars ---
            if categorical_cols:
                visualizations["categorical_bars"] = self._create_categorical_plots(
                    df_plot, categorical_cols, max_plots=cfg.max_plots_categorical, top_k=cfg.top_k_categories
                )

            # --- Missing data visualization ---
            visualizations["missing_data"] = self._create_missing_data_plot(data)

            # --- Target analysis ---
            if target_column and target_column in data.columns:
                visualizations["target_analysis"] = self._create_target_analysis(
                    df_plot, target_column, numeric_cols, categorical_cols
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

    # === UTIL: SAMPLING ===
    def _maybe_sample(self, df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        """Downsample DataFrame for heavy plots to maintain interactivity."""
        n = len(df)
        if n > max_points:
            self.logger.info(f"Downsampling for plotting: {n} → {max_points} rows")
            return df.sample(n=max_points, random_state=self.config.random_state)
        return df

    # === DISTRIBUTIONS (HIST + BOX MARGINAL) ===
    def _create_distribution_plots(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_plots: int = 10
    ) -> List[go.Figure]:
        """Create distribution plots (histograms) for numeric features."""
        plots: List[go.Figure] = []
        for col in columns[:max_plots]:
            try:
                fig = px.histogram(
                    df,
                    x=col,
                    title=f"Rozkład: {col}",
                    marginal="box",
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY,
                    opacity=0.9,
                )
                fig.update_layout(showlegend=False, height=380, bargap=0.02)
                plots.append(fig)
            except Exception as e:
                logger.warning(f"Histogram failed for '{col}': {e}")
        return plots

    # === BOXPLOTS (ADAPTIVE GRID) ===
    def _create_boxplots(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_plots: int = 10
    ) -> go.Figure:
        """Create box plots for numeric features in an adaptive grid."""
        cols = columns[:max_plots]
        n = len(cols)
        if n == 0:
            return go.Figure()

        # adaptacyjna siatka: max 4 wiersze, 5 kolumn
        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))

        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols, vertical_spacing=0.08, horizontal_spacing=0.04)
        for i, col in enumerate(cols):
            r = (i // ncols) + 1
            c = (i % ncols) + 1
            try:
                fig.add_trace(
                    go.Box(
                        y=pd.to_numeric(df[col], errors="coerce"),
                        name=col,
                        marker_color=COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)],
                        boxpoints="outliers",
                        jitter=0.2
                    ),
                    row=r, col=c
                )
            except Exception as e:
                logger.warning(f"Boxplot failed for '{col}': {e}")

        fig.update_layout(
            title_text="Box Plots — Wykrywanie Outliers",
            showlegend=False,
            height=max(350, nrows * 320),
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    # === HEATMAP KORELACJI ===
    def _create_correlation_heatmap(
        self,
        df_num: pd.DataFrame,
        method: str = "pearson",
        max_features: int = 40,
        annotate: bool = True
    ) -> go.Figure:
        """Create correlation heatmap with optional lower-triangle mask and annotations."""
        # redukcja liczby cech wg wariancji (stabilniej wizualnie)
        if df_num.shape[1] > max_features:
            variances = df_num.var(numeric_only=True).sort_values(ascending=False)
            keep = variances.head(max_features).index
            df_num = df_num[keep]
            logger.info(f"Heatmap features truncated to top-{max_features} by variance.")

        # korelacje
        corr_matrix = df_num.corr(method=method, numeric_only=True).round(3)
        corr_vals = corr_matrix.values

        # maska dolnego trójkąta dla czytelności
        mask = np.triu(np.ones_like(corr_vals, dtype=bool), k=1)
        display_vals = corr_vals.copy()
        display_vals[mask] = np.nan  # ukryj górny trójkąt

        # annotacje (pokazuj tylko gdzie nie NaN)
        text = None
        if annotate:
            text = np.where(np.isnan(display_vals), "", np.vectorize(lambda v: f"{v:.2f}")(display_vals))

        fig = go.Figure(
            data=go.Heatmap(
                z=display_vals,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu_r",
                zmin=-1, zmax=1, zmid=0,
                colorbar=dict(title="Korelacja"),
                text=text,
                texttemplate="%{text}" if annotate else None,
                hovertemplate="(%{y}, %{x}) = %{z}<extra></extra>"
            )
        )
        fig.update_layout(
            title=f"Macierz Korelacji ({method.title()})",
            xaxis_title="Cechy",
            yaxis_title="Cechy",
            height=max(500, 28 * corr_matrix.shape[0]),
            margin=dict(l=60, r=20, t=60, b=60),
        )
        return fig

    # === KATEGORIE (TOP-K) ===
    def _create_categorical_plots(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_plots: int = 5,
        top_k: int = 10
    ) -> List[go.Figure]:
        """Create bar plots for categorical features (top-k)."""
        plots: List[go.Figure] = []
        for i, col in enumerate(columns[:max_plots]):
            try:
                vc = df[col].astype("object").value_counts(dropna=False).head(top_k)
                fig = px.bar(
                    x=[str(x) for x in vc.index],
                    y=vc.values,
                    title=f"Top {top_k} wartości: {col}",
                    labels={"x": col, "y": "Liczność"},
                    color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                )
                fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-30)
                plots.append(fig)
            except Exception as e:
                logger.warning(f"Categorical bar failed for '{col}': {e}")
        return plots

    # === BRAKI DANYCH ===
    def _create_missing_data_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create missing data visualization (horizontal bar or friendly message)."""
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=True)  # rosnąco dla h-bar
        if missing.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Brak brakujących danych! 🎉",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            return fig

        pct = (missing / len(df) * 100).round(2)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=pct.values,
                    y=missing.index.astype(str),
                    orientation="h",
                    text=[f"{v:.2f}%" for v in pct.values],
                    textposition="auto",
                    marker_color=COLOR_PALETTE_PRIMARY[0],
                )
            ]
        )
        fig.update_layout(
            title="Brakujące Wartości (% w kolumnie)",
            xaxis_title="Procent braków",
            yaxis_title="Kolumna",
            height=max(320, 26 * len(missing)),
            margin=dict(l=120, r=30, t=60, b=40),
        )
        return fig

    # === TARGET ANALYSIS (rozkład + szybkie relacje cecha↔target) ===
    def _create_target_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> Dict[str, go.Figure]:
        """Create target variable analysis plots + quick relations with top features."""
        plots: Dict[str, go.Figure] = {}
        target = df[target_column]

        # Rozkład targetu
        try:
            if pd.api.types.is_numeric_dtype(target):
                fig = px.histogram(
                    df, x=target_column, title=f"Rozkład zmiennej docelowej: {target_column}",
                    marginal="box", color_discrete_sequence=COLOR_PALETTE_PRIMARY
                )
                plots["target_distribution"] = fig
            else:
                vc = target.astype("object").value_counts(dropna=False)
                fig = px.bar(
                    x=[str(x) for x in vc.index],
                    y=vc.values,
                    title=f"Rozkład klas: {target_column}",
                    labels={"x": target_column, "y": "Liczność"},
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY
                )
                fig.update_layout(showlegend=False, height=380)
                plots["target_distribution"] = fig
        except Exception as e:
            logger.warning(f"Target distribution failed: {e}")

        # Quick relations (top kilka cech względem korelacji/informacyjności)
        try:
            if pd.api.types.is_numeric_dtype(target) and numeric_cols:
                # Pearson Spearman fallback
                corr_abs = {}
                for col in numeric_cols:
                    if col == target_column:
                        continue
                    x = pd.to_numeric(df[col], errors="coerce")
                    y = pd.to_numeric(target, errors="coerce")
                    valid = x.notna() & y.notna()
                    if valid.sum() < 3:
                        continue
                    r = float(np.corrcoef(x[valid], y[valid])[0, 1]) if np.isfinite(np.corrcoef(x[valid], y[valid])[0, 1]) else 0.0
                    if np.isnan(r):
                        continue
                    corr_abs[col] = abs(r)
                top = [k for k, _ in sorted(corr_abs.items(), key=lambda kv: kv[1], reverse=True)[: self.config.target_rel_top_features]]

                for i, col in enumerate(top):
                    fig = px.scatter(
                        df, x=col, y=target_column, trendline="ols",
                        title=f"{col} vs {target_column}",
                        color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]]
                    )
                    fig.update_layout(height=380)
                    plots[f"rel_{col}_vs_target"] = fig

            elif not pd.api.types.is_numeric_dtype(target) and numeric_cols:
                # violin dla numeric vs kategoryczny target
                for i, col in enumerate(numeric_cols[: self.config.target_rel_top_features]):
                    fig = px.violin(
                        df, x=target_column, y=col, box=True, points="outliers",
                        title=f"{col} vs {target_column}",
                        color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]]
                    )
                    fig.update_layout(height=380)
                    plots[f"rel_{col}_by_target"] = fig

            # dla kategorycznych cech vs kategoryczny target — top-k stacked bars (opcjonalnie)
            if not pd.api.types.is_numeric_dtype(target) and categorical_cols:
                col = categorical_cols[0]
                vc = df[col].astype("object").value_counts().head(6).index
                tmp = df[df[col].isin(vc)].copy()
                fig = px.histogram(
                    tmp, x=col, color=target_column, barmode="group",
                    title=f"{col} vs {target_column} (Top-k)",
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY
                )
                fig.update_layout(height=380)
                plots[f"{col}_vs_{target_column}"] = fig

        except Exception as e:
            logger.warning(f"Target relations failed: {e}")

        return plots
