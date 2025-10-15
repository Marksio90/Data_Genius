# === OPIS MODUŁU ===
"""
DataGenius PRO - Visualization Engine (PRO++++)
Interaktywne wizualizacje EDA (Plotly) ze skalowaniem (downsampling/aggregacje),
adaptacyjnymi layoutami, defensywnymi guardami i spójnym stylem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
import warnings

from core.base_agent import BaseAgent, AgentResult

# === PALETA KOLORÓW (fallback jeśli config nie istnieje) ===
try:
    from config.constants import COLOR_PALETTE_PRIMARY  # type: ignore
except Exception:
    COLOR_PALETTE_PRIMARY = [
        "#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706",
        "#10B981", "#F43F5E", "#0EA5E9", "#9333EA", "#EF4444"
    ]


# === NAZWA_SEKCJI === KONFIG ===
@dataclass(frozen=True)
class VizConfig:
    max_points: int = 120_000              # downsampling dla rysowania gęstych chmur
    max_plots_numeric: int = 12            # maks. liczba histogramów/boxów
    max_plots_categorical: int = 8         # maks. liczba barów
    top_k_categories: int = 12             # top-K dla kategorii
    heatmap_max_features: int = 60         # cap features w heatmapie
    correlation_method: str = "pearson"    # "pearson" | "spearman"
    annotate_heatmap: bool = True          # anotacje w heatmapie
    target_rel_top_features: int = 4       # ile top cech vs target
    random_state: int = 42                 # reproducibility
    hexbin_min_points: int = 10_000        # od ilu wierszy użyć density_heatmap
    datetime_line_max: int = 6             # ile serii do linii czasowej
    ts_rolling_window: int = 7             # okno do rolling median
    max_cat_levels_stack: int = 8          # cap poziomów w stacked barach
    warn_on_truncation: bool = True        # loguj ucinanie listy wykresów/cech
    use_category_aggregation: bool = True  # agreguj długie ogony kategorii do "OTHER"


class VisualizationEngine(BaseAgent):
    """
    Generuje interaktywne wizualizacje dla EDA (skutecznie i bezpiecznie).
    """

    def __init__(self, config: Optional[VizConfig] = None) -> None:
        super().__init__(name="VisualizationEngine", description="Generates interactive visualizations")
        self.config = config or VizConfig()
        warnings.filterwarnings("ignore")

    # === NAZWA_SEKCJI === WYKONANIE GŁÓWNE ===
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Tworzy słownik wykresów Plotly. Klucze: distributions, boxplots, correlation_heatmap,
        categorical_bars, missing_data, time_series (opcjonalnie), target_analysis, density_plots.
        """
        result = AgentResult(agent_name=self.name)

        try:
            # Walidacja
            if data is None or not isinstance(data, pd.DataFrame):
                msg = "VisualizationEngine: 'data' must be a pandas DataFrame."
                result.add_error(msg)
                self.logger.error(msg)
                return result
            if data.empty:
                result.add_warning("Empty DataFrame — no visualizations produced.")
                result.data = {"visualizations": {}, "n_visualizations": 0, "warnings": ["empty_dataframe"]}
                return result

            df = data.copy()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Selekcje
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

            # Downsampling dla ciężkich plotów
            df_plot = self._maybe_sample(df, self.config.max_points)

            visualizations: Dict[str, Any] = {}

            # === Numeric distributions & boxy ===
            if numeric_cols:
                visualizations["distributions"] = self._create_distribution_plots(
                    df_plot, numeric_cols, max_plots=self.config.max_plots_numeric
                )
                visualizations["boxplots"] = self._create_boxplots(
                    df_plot, numeric_cols, max_plots=self.config.max_plots_numeric
                )

            # === Korelacje (na pełnym df numerycznym; precyzja > szybkość) ===
            if len(numeric_cols) > 1:
                visualizations["correlation_heatmap"] = self._create_correlation_heatmap(
                    df[numeric_cols], method=self.config.correlation_method,
                    max_features=self.config.heatmap_max_features, annotate=self.config.annotate_heatmap
                )

            # === Kategorie (top-K, agregacja OTHER) ===
            if categorical_cols:
                visualizations["categorical_bars"] = self._create_categorical_plots(
                    df_plot, categorical_cols, max_plots=self.config.max_plots_categorical, top_k=self.config.top_k_categories
                )

            # === Braki danych ===
            visualizations["missing_data"] = self._create_missing_data_plot(df)

            # === Time Series (dla kolumn datetime + wybranych num) ===
            if datetime_cols and numeric_cols:
                visualizations["time_series"] = self._create_time_series_plots(
                    df_plot, datetime_cols, numeric_cols, max_series=self.config.datetime_line_max
                )

            # === Gęste chmury punktów (hexbin/density) ===
            if numeric_cols and len(df_plot) >= self.config.hexbin_min_points:
                visualizations["density_plots"] = self._create_density_plots(
                    df_plot, numeric_cols, max_plots=min(6, len(numeric_cols))
                )

            # === Target analysis ===
            if target_column and target_column in df.columns:
                visualizations["target_analysis"] = self._create_target_analysis(
                    df_plot, target_column, numeric_cols, categorical_cols
                )

            n_viz = sum(
                (len(v) if isinstance(v, list) else len(v) if isinstance(v, dict) else 1)
                for v in visualizations.values()
            )

            result.data = {"visualizations": visualizations, "n_visualizations": int(n_viz)}
            self.logger.success(f"Generated {int(n_viz)} visualization objects across {len(visualizations)} groups.")

        except Exception as e:
            result.add_error(f"Visualization generation failed: {e}")
            self.logger.error(f"Visualization error: {e}", exc_info=True)

        return result

    # === NAZWA_SEKCJI === UTIL: SAMPLING ===
    def _maybe_sample(self, df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        n = len(df)
        if n > max_points:
            self.logger.info(f"Downsampling for plotting: {n} → {max_points} rows")
            return df.sample(n=max_points, random_state=self.config.random_state)
        return df

    # === NAZWA_SEKCJI === DISTRIBUTIONS (HIST + BOX MARGINAL) ===
    def _create_distribution_plots(self, df: pd.DataFrame, columns: List[str], max_plots: int = 10) -> List[go.Figure]:
        plots: List[go.Figure] = []
        cut = columns[:max_plots]
        if len(columns) > max_plots and self.config.warn_on_truncation:
            logger.info(f"[distributions] Truncated columns {len(columns)} → {max_plots}")
        for i, col in enumerate(cut):
            try:
                fig = px.histogram(
                    df,
                    x=col,
                    title=f"Rozkład: {col}",
                    marginal="box",
                    opacity=0.9,
                    color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                    nbins=min(100, max(10, int(np.sqrt(max(10, df[col].dropna().nunique()))))),
                    histnorm=None
                )
                fig.update_layout(showlegend=False, height=360, bargap=0.02, margin=dict(l=40, r=20, t=60, b=40))
                plots.append(fig)
            except Exception as e:
                logger.warning(f"Histogram failed for '{col}': {e}")
        return plots

    # === NAZWA_SEKCJI === BOXPLOTS (ADAPTIVE GRID) ===
    def _create_boxplots(self, df: pd.DataFrame, columns: List[str], max_plots: int = 10) -> go.Figure:
        cols = columns[:max_plots]
        n = len(cols)
        if n == 0:
            return go.Figure()
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
                        jitter=0.2,
                        whiskerwidth=0.5
                    ),
                    row=r, col=c
                )
            except Exception as e:
                logger.warning(f"Boxplot failed for '{col}': {e}")

        fig.update_layout(
            title_text="Box Plots — wykrywanie outlierów",
            showlegend=False,
            height=max(360, nrows * 300),
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    # === NAZWA_SEKCJI === HEATMAP KORELACJI ===
    def _create_correlation_heatmap(
        self,
        df_num: pd.DataFrame,
        method: str = "pearson",
        max_features: int = 40,
        annotate: bool = True
    ) -> go.Figure:
        # cięcie po wariancji
        data_num = df_num.select_dtypes(include=[np.number])
        if data_num.shape[1] > max_features:
            variances = data_num.var(numeric_only=True).sort_values(ascending=False)
            keep = variances.head(max_features).index
            data_num = data_num[keep]
            logger.info(f"[heatmap] Features truncated to top-{max_features} by variance.")

        corr_matrix = data_num.corr(method=method, numeric_only=True).round(3)
        vals = corr_matrix.values

        # maska górnego trójkąta
        mask = np.triu(np.ones_like(vals, dtype=bool), k=1)
        display_vals = vals.copy()
        display_vals[mask] = np.nan

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
            title=f"Macierz korelacji ({method.title()})",
            xaxis_title="Cechy",
            yaxis_title="Cechy",
            height=max(520, 28 * corr_matrix.shape[0]),
            margin=dict(l=60, r=20, t=60, b=60),
        )
        return fig

    # === NAZWA_SEKCJI === KATEGORIE (TOP-K + OTHER) ===
    def _create_categorical_plots(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_plots: int = 6,
        top_k: int = 10
    ) -> List[go.Figure]:
        plots: List[go.Figure] = []
        cut = columns[:max_plots]
        if len(columns) > max_plots and self.config.warn_on_truncation:
            logger.info(f"[categorical] Truncated columns {len(columns)} → {max_plots}")
        for i, col in enumerate(cut):
            try:
                s = df[col].astype("object")
                vc = s.value_counts(dropna=False)
                if self.config.use_category_aggregation and len(vc) > top_k:
                    top_vals = vc.head(top_k - 1)
                    other = pd.Series({ "OTHER": int(vc.iloc[top_k - 1 :].sum()) })
                    vc = pd.concat([top_vals, other])

                fig = px.bar(
                    x=[str(x) for x in vc.index],
                    y=vc.values,
                    title=f"Top {top_k} wartości: {col}",
                    labels={"x": col, "y": "Liczność"},
                    color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                )
                fig.update_layout(showlegend=False, height=360, xaxis_tickangle=-25, margin=dict(l=40, r=20, t=60, b=60))
                plots.append(fig)
            except Exception as e:
                logger.warning(f"Categorical bar failed for '{col}': {e}")
        return plots

    # === NAZWA_SEKCJI === BRAKI DANYCH ===
    def _create_missing_data_plot(self, df: pd.DataFrame) -> go.Figure:
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        if missing.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Brak brakujących danych 🎉",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            return fig

        pct = (missing / len(df) * 100).round(2)
        fig = go.Figure(
            data=[go.Bar(
                x=pct.values,
                y=missing.index.astype(str),
                orientation="h",
                text=[f"{v:.2f}%" for v in pct.values],
                textposition="auto",
                marker_color=COLOR_PALETTE_PRIMARY[0],
            )]
        )
        fig.update_layout(
            title="Brakujące wartości (% w kolumnie)",
            xaxis_title="Procent braków",
            yaxis_title="Kolumna",
            height=max(320, 26 * len(missing)),
            margin=dict(l=140, r=30, t=60, b=40),
        )
        return fig

    # === NAZWA_SEKCJI === TIME SERIES (DATETIME + NUM) ===
    def _create_time_series_plots(
        self,
        df: pd.DataFrame,
        datetime_cols: List[str],
        numeric_cols: List[str],
        max_series: int = 6
    ) -> Dict[str, go.Figure]:
        plots: Dict[str, go.Figure] = {}
        # wybierz pierwszą kolumnę datetime o najmniejszej liczbie NaN
        dt_col = min(datetime_cols, key=lambda c: df[c].isna().sum())
        try:
            tmp = df[[dt_col] + numeric_cols].copy()
            tmp = tmp.dropna(subset=[dt_col]).sort_values(dt_col)
            # wybór do max_series (największa wariancja)
            var_order = tmp[numeric_cols].var(numeric_only=True).sort_values(ascending=False).index.tolist()
            sel = var_order[:max_series]

            fig = go.Figure()
            for i, col in enumerate(sel):
                y = pd.to_numeric(tmp[col], errors="coerce")
                fig.add_trace(go.Scatter(
                    x=tmp[dt_col], y=y, mode="lines", name=col,
                    line=dict(width=1.4, color=COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)])
                ))
                # rolling median overlay
                roll = y.rolling(self.config.ts_rolling_window, min_periods=max(2, self.config.ts_rolling_window//2)).median()
                fig.add_trace(go.Scatter(
                    x=tmp[dt_col], y=roll, mode="lines", name=f"{col} (roll{self.config.ts_rolling_window})",
                    line=dict(width=2.0, dash="dot", color=COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]),
                    showlegend=False
                ))

            fig.update_layout(
                title=f"Time Series ({dt_col}) — top {len(sel)} zmiennych",
                xaxis_title=dt_col, yaxis_title="Wartość",
                height=max(420, 280 + 20 * len(sel)),
                margin=dict(l=40, r=20, t=60, b=40)
            )
            plots["timeseries_main"] = fig
        except Exception as e:
            logger.warning(f"Time series plot failed: {e}")
        return plots

    # === NAZWA_SEKCJI === DENSITY/HEXBIN DLA GĘSTYCH DANYCH ===
    def _create_density_plots(self, df: pd.DataFrame, numeric_cols: List[str], max_plots: int = 6) -> List[go.Figure]:
        plots: List[go.Figure] = []
        cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
        sel = cols[:max_plots]
        for i, col in enumerate(sel):
            try:
                # density heatmap vs. pierwsza inna numeryczna
                partner = None
                for c2 in cols:
                    if c2 != col:
                        partner = c2
                        break
                if partner is None:
                    break
                fig = px.density_heatmap(
                    df, x=col, y=partner, nbinsx=50, nbinsy=50,
                    title=f"Gęstość: {col} vs {partner}",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(height=420, margin=dict(l=40, r=20, t=60, b=40))
                plots.append(fig)
            except Exception as e:
                logger.warning(f"Density plot failed for '{col}': {e}")
        return plots

    # === NAZWA_SEKCJI === TARGET ANALYSIS (rozkład + relacje) ===
    def _create_target_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> Dict[str, go.Figure]:
        plots: Dict[str, go.Figure] = {}
        target = df[target_column]

        # Rozkład targetu
        try:
            if pd.api.types.is_numeric_dtype(target):
                fig = px.histogram(
                    df, x=target_column, title=f"Rozkład zmiennej docelowej: {target_column}",
                    marginal="box", color_discrete_sequence=COLOR_PALETTE_PRIMARY
                )
                fig.update_layout(showlegend=False, height=360)
                plots["target_distribution"] = fig
            else:
                vc = target.astype("object").value_counts(dropna=False)
                if len(vc) > self.config.max_cat_levels_stack:
                    vc = vc.head(self.config.max_cat_levels_stack)
                fig = px.bar(
                    x=[str(x) for x in vc.index],
                    y=vc.values,
                    title=f"Rozkład klas: {target_column}",
                    labels={"x": target_column, "y": "Liczność"},
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY
                )
                fig.update_layout(showlegend=False, height=360)
                plots["target_distribution"] = fig
        except Exception as e:
            logger.warning(f"Target distribution failed: {e}")

        # Relacje cechy ↔ target
        try:
            if pd.api.types.is_numeric_dtype(target) and numeric_cols:
                # top korelujące (Pearson, NA-safe)
                corr_abs: Dict[str, float] = {}
                y = pd.to_numeric(target, errors="coerce")
                for col in numeric_cols:
                    if col == target_column:
                        continue
                    x = pd.to_numeric(df[col], errors="coerce")
                    valid = x.notna() & y.notna()
                    if valid.sum() < 3:
                        continue
                    r = np.corrcoef(x[valid], y[valid])[0, 1]
                    if np.isfinite(r):
                        corr_abs[col] = float(abs(r))
                top = [k for k, _ in sorted(corr_abs.items(), key=lambda kv: kv[1], reverse=True)[: self.config.target_rel_top_features]]

                for i, col in enumerate(top):
                    trend = "ols" if len(df) <= 5_00_000 else None  # unikaj regresji na ogromnych zbiorach
                    fig = px.scatter(
                        df, x=col, y=target_column, trendline=trend,
                        title=f"{col} vs {target_column}",
                        opacity=0.6,
                        color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]]
                    )
                    fig.update_layout(height=360)
                    plots[f"rel_{col}_vs_target"] = fig

            elif not pd.api.types.is_numeric_dtype(target) and numeric_cols:
                # violin dla numeric vs kategoryczny target (pierwsze N)
                for i, col in enumerate(numeric_cols[: self.config.target_rel_top_features]):
                    fig = px.violin(
                        df, x=target_column, y=col, box=True, points="outliers",
                        title=f"{col} vs {target_column}",
                        color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]]
                    )
                    fig.update_layout(height=360)
                    plots[f"rel_{col}_by_{target_column}"] = fig

            # kategoryczna cecha vs kategoryczny target — grouped/stacked
            if not pd.api.types.is_numeric_dtype(target) and categorical_cols:
                col = categorical_cols[0]
                s = df[col].astype("object")
                vc = s.value_counts().head(self.config.max_cat_levels_stack).index
                tmp = df[df[col].isin(vc)].copy()
                fig = px.histogram(
                    tmp, x=col, color=target_column, barmode="group",
                    title=f"{col} vs {target_column} (Top-k)",
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY
                )
                fig.update_layout(height=360)
                plots[f"{col}_vs_{target_column}"] = fig

        except Exception as e:
            logger.warning(f"Target relations failed: {e}")

        return plots
