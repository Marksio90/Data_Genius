# === OPIS MODUŁU ===
"""
DataGenius PRO++++++++++++ — Visualization Engine (Enterprise / KOSMOS)
Interaktywne wizualizacje EDA (Plotly) ze skalowaniem (downsampling/aggregacje),
adaptacyjnymi layoutami, defensywnymi guardami i spójnym stylem.

Wejście:
    - data: pd.DataFrame
    - target_column: Optional[str]

Wyjście (AgentResult.data):
{
  "visualizations": {
      "distributions": List[go.Figure],
      "boxplots": go.Figure | {},
      "correlation_heatmap": go.Figure | {},
      "categorical_bars": List[go.Figure],
      "missing_data": go.Figure,
      "time_series": Dict[str, go.Figure],
      "density_plots": List[go.Figure],
      "target_analysis": Dict[str, go.Figure]
  },
  "n_visualizations": int,
  "warnings": Optional[List[str]]
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


# === KONFIG ===
@dataclass(frozen=True)
class VizConfig:
    # Sampling i skale
    max_points: int = 120_000
    random_state: int = 42

    # Limity liczby wykresów
    max_plots_numeric: int = 12
    max_plots_categorical: int = 8
    top_k_categories: int = 12
    max_cat_levels_stack: int = 8

    # Korelacje
    heatmap_max_features: int = 60
    correlation_method: str = "pearson"  # "pearson" | "spearman"
    annotate_heatmap: bool = True

    # Target relations
    target_rel_top_features: int = 4

    # Time series
    datetime_line_max: int = 6
    ts_rolling_window: int = 7

    # Gęste chmury
    hexbin_min_points: int = 10_000

    # Zachowanie
    warn_on_truncation: bool = True
    use_category_aggregation: bool = True  # agreguj długie ogony kategorii do "OTHER"


class VisualizationEngine(BaseAgent):
    """
    Generuje interaktywne wizualizacje dla EDA (wydajnie i bezpiecznie).
    """

    def __init__(self, config: Optional[VizConfig] = None) -> None:
        super().__init__(name="VisualizationEngine", description="Generates interactive visualizations")
        self.config = config or VizConfig()
        warnings.filterwarnings("ignore")

    # === API GŁÓWNE ===
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
        warnings_collected: List[str] = []

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

            # Kopia i sanity
            df = data.copy()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Dtypes
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

            # Downsampling na potrzeby ciężkich wykresów
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
            else:
                visualizations["distributions"] = []
                visualizations["boxplots"] = {}

            # === Korelacje (na pełnym df numerycznym; precyzja > szybkość) ===
            if len(numeric_cols) > 1:
                try:
                    visualizations["correlation_heatmap"] = self._create_correlation_heatmap(
                        df[numeric_cols],
                        method=self.config.correlation_method,
                        max_features=self.config.heatmap_max_features,
                        annotate=self.config.annotate_heatmap,
                    )
                except Exception as e:
                    self.logger.warning(f"Correlation heatmap failed: {e}")
                    visualizations["correlation_heatmap"] = {}
                    warnings_collected.append("correlation_heatmap_failed")
            else:
                visualizations["correlation_heatmap"] = {}

            # === Kategorie (top-K, agregacja OTHER) ===
            if categorical_cols:
                visualizations["categorical_bars"] = self._create_categorical_plots(
                    df_plot, categorical_cols, max_plots=self.config.max_plots_categorical, top_k=self.config.top_k_categories
                )
            else:
                visualizations["categorical_bars"] = []

            # === Braki danych ===
            visualizations["missing_data"] = self._create_missing_data_plot(df)

            # === Time Series (dla kolumn datetime + wybranych num) ===
            if datetime_cols and numeric_cols:
                visualizations["time_series"] = self._create_time_series_plots(
                    df_plot, datetime_cols, numeric_cols, max_series=self.config.datetime_line_max
                )
            else:
                visualizations["time_series"] = {}

            # === Gęste chmury punktów (hexbin/density) ===
            if numeric_cols and len(df_plot) >= self.config.hexbin_min_points:
                visualizations["density_plots"] = self._create_density_plots(
                    df_plot, numeric_cols, max_plots=min(6, len(numeric_cols))
                )
            else:
                visualizations["density_plots"] = []

            # === Target analysis ===
            if target_column and target_column in df.columns:
                visualizations["target_analysis"] = self._create_target_analysis(
                    df_plot, target_column, numeric_cols, categorical_cols
                )
            else:
                visualizations["target_analysis"] = {}

            n_viz = self._count_objects(visualizations)

            payload = {"visualizations": visualizations, "n_visualizations": int(n_viz)}
            if warnings_collected:
                payload["warnings"] = warnings_collected

            result.data = payload
            self.logger.success(f"Generated {int(n_viz)} visualization objects across {len(visualizations)} groups.")
            return result

        except Exception as e:
            result.add_error(f"Visualization generation failed: {e}")
            self.logger.error(f"Visualization error: {e}", exc_info=True)
            return result

    # === UTIL: SAMPLING ===
    def _maybe_sample(self, df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        n = len(df)
        if max_points and n > max_points:
            self.logger.info(f"Downsampling for plotting: {n} → {max_points} rows")
            try:
                return df.sample(n=max_points, random_state=self.config.random_state)
            except Exception:
                # fallback do pierwszych N (stabilność)
                return df.head(max_points)
        return df

    # === DISTRIBUTIONS (HIST + BOX MARGINAL) ===
    def _create_distribution_plots(self, df: pd.DataFrame, columns: List[str], max_plots: int = 10) -> List[go.Figure]:
        plots: List[go.Figure] = []
        cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        cut = cols[:max_plots]
        if len(cols) > max_plots and self.config.warn_on_truncation:
            logger.info(f"[distributions] Truncated numeric columns {len(cols)} → {max_plots}")

        for i, col in enumerate(cut):
            try:
                # Heurystyczny nbins (stabilny dla dużej liczby unikatów)
                unique_non_na = df[col].dropna().nunique()
                nbins = int(np.clip(int(np.sqrt(max(10, unique_non_na))), 20, 120))

                fig = px.histogram(
                    df,
                    x=col,
                    title=f"Rozkład: {col}",
                    marginal="box",
                    opacity=0.9,
                    color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                    nbins=nbins,
                    histnorm=None,
                )
                fig.update_layout(
                    showlegend=False, height=360, bargap=0.02,
                    margin=dict(l=40, r=20, t=60, b=40)
                )
                plots.append(fig)
            except Exception as e:
                logger.warning(f"Histogram failed for '{col}': {e}")
        return plots

    # === BOXPLOTS (ADAPTIVE GRID) ===
    def _create_boxplots(self, df: pd.DataFrame, columns: List[str], max_plots: int = 10) -> go.Figure | Dict[str, Any]:
        cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])][:max_plots]
        n = len(cols)
        if n == 0:
            return {}

        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))

        fig = make_subplots(
            rows=nrows, cols=ncols, subplot_titles=cols, vertical_spacing=0.08, horizontal_spacing=0.04
        )
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
                        whiskerwidth=0.5,
                    ),
                    row=r,
                    col=c,
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

    # === HEATMAP KORELACJI ===
    def _create_correlation_heatmap(
        self,
        df_num: pd.DataFrame,
        method: str = "pearson",
        max_features: int = 40,
        annotate: bool = True,
    ) -> go.Figure:
        # Na wypadek kolumn bez wariancji → drop
        data_num = df_num.select_dtypes(include=[np.number])
        variances = data_num.var(numeric_only=True)
        keep_var = variances[variances > 0].index
        data_num = data_num[keep_var]

        if data_num.shape[1] == 0:
            fig = go.Figure()
            fig.add_annotation(text="Brak cech o dodatniej wariancji", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            return fig

        if data_num.shape[1] > max_features:
            # wybierz top-k po wariancji
            keep = variances.loc[keep_var].sort_values(ascending=False).head(max_features).index
            data_num = data_num[keep]
            logger.info(f"[heatmap] Features truncated to top-{max_features} by variance.")

        corr = data_num.corr(method=method, numeric_only=True).round(3).fillna(0.0)
        vals = corr.values

        # maska górnego trójkąta
        mask = np.triu(np.ones_like(vals, dtype=bool), k=1)
        display_vals = vals.copy()
        display_vals[mask] = np.nan

        # ogranicz anotacje dla bardzo dużych macierzy
        do_annotate = annotate and corr.shape[0] <= 40
        text = None
        if do_annotate:
            text = np.where(np.isnan(display_vals), "", np.vectorize(lambda v: f"{v:.2f}")(display_vals))

        fig = go.Figure(
            data=go.Heatmap(
                z=display_vals,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                zmid=0,
                colorbar=dict(title="Korelacja"),
                text=text,
                texttemplate="%{text}" if do_annotate else None,
                hovertemplate="(%{y}, %{x}) = %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Macierz korelacji ({method.title()})",
            xaxis_title="Cechy",
            yaxis_title="Cechy",
            height=max(520, 28 * corr.shape[0]),
            margin=dict(l=60, r=20, t=60, b=60),
        )
        return fig

    # === KATEGORIE (TOP-K + OTHER) ===
    def _create_categorical_plots(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_plots: int = 6,
        top_k: int = 10,
    ) -> List[go.Figure]:
        plots: List[go.Figure] = []
        cols = columns[:max_plots]
        if len(columns) > max_plots and self.config.warn_on_truncation:
            logger.info(f"[categorical] Truncated columns {len(columns)} → {max_plots}")

        for i, col in enumerate(cols):
            try:
                s = df[col].astype("object")
                vc = s.value_counts(dropna=False)

                if self.config.use_category_aggregation and len(vc) > top_k:
                    top_vals = vc.head(top_k - 1)
                    other_sum = int(vc.iloc[top_k - 1 :].sum())
                    vc = pd.concat([top_vals, pd.Series({"OTHER": other_sum})])

                x_labels = [str(x) for x in vc.index]
                fig = px.bar(
                    x=x_labels,
                    y=vc.values,
                    title=f"Top {min(top_k, len(vc))} wartości: {col}",
                    labels={"x": col, "y": "Liczność"},
                    color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                )
                fig.update_layout(
                    showlegend=False, height=360, xaxis_tickangle=-25,
                    margin=dict(l=40, r=20, t=60, b=60)
                )
                plots.append(fig)
            except Exception as e:
                logger.warning(f"Categorical bar failed for '{col}': {e}")
        return plots

    # === BRAKI DANYCH ===
    def _create_missing_data_plot(self, df: pd.DataFrame) -> go.Figure:
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=True)

        if missing.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Brak brakujących danych 🎉",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20),
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
            title="Brakujące wartości (% w kolumnie)",
            xaxis_title="Procent braków",
            yaxis_title="Kolumna",
            height=max(320, 26 * len(missing)),
            margin=dict(l=140, r=30, t=60, b=40),
        )
        return fig

    # === TIME SERIES (DATETIME + NUM) ===
    def _create_time_series_plots(
        self,
        df: pd.DataFrame,
        datetime_cols: List[str],
        numeric_cols: List[str],
        max_series: int = 6,
    ) -> Dict[str, go.Figure]:
        plots: Dict[str, go.Figure] = {}
        # wybierz kolumnę datetime o najmniejszej liczbie NaN
        dt_col = min(datetime_cols, key=lambda c: df[c].isna().sum())

        try:
            tmp = df[[dt_col] + numeric_cols].copy()
            tmp = tmp.dropna(subset=[dt_col]).sort_values(dt_col)

            # wybór do max_series (po wariancji)
            var_order = tmp[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
            sel = [c for c in var_order.index[:max_series] if pd.api.types.is_numeric_dtype(tmp[c])]
            if not sel:
                return plots

            fig = go.Figure()
            for i, col in enumerate(sel):
                y = pd.to_numeric(tmp[col], errors="coerce")
                fig.add_trace(
                    go.Scatter(
                        x=tmp[dt_col],
                        y=y,
                        mode="lines",
                        name=col,
                        line=dict(width=1.4, color=COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]),
                    )
                )
                # rolling median overlay (robust)
                win = max(2, self.config.ts_rolling_window)
                roll = y.rolling(win, min_periods=max(2, win // 2)).median()
                fig.add_trace(
                    go.Scatter(
                        x=tmp[dt_col],
                        y=roll,
                        mode="lines",
                        name=f"{col} (roll{win})",
                        line=dict(width=2.0, dash="dot", color=COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]),
                        showlegend=False,
                    )
                )

            fig.update_layout(
                title=f"Time Series ({dt_col}) — top {len(sel)} zmiennych",
                xaxis_title=dt_col,
                yaxis_title="Wartość",
                height=max(420, 280 + 20 * len(sel)),
                margin=dict(l=40, r=20, t=60, b=40),
            )
            plots["timeseries_main"] = fig
        except Exception as e:
            logger.warning(f"Time series plot failed: {e}")
        return plots

    # === DENSITY/HEXBIN DLA GĘSTYCH DANYCH ===
    def _create_density_plots(self, df: pd.DataFrame, numeric_cols: List[str], max_plots: int = 6) -> List[go.Figure]:
        plots: List[go.Figure] = []
        cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(cols) < 2:
            return plots

        # dobierz pary wg bezwzględnej korelacji (bardziej informacyjne)
        try:
            num_df = df[cols].copy()
            corr = num_df.corr(numeric_only=True).abs()
            np.fill_diagonal(corr.values, 0.0)
            pairs = []
            for i, a in enumerate(corr.columns):
                b = corr.iloc[i].idxmax()
                pairs.append((a, b, float(corr.iloc[i][b])))
            # posortuj po korelacji malejąco i deduplikuj po kolumnach
            seen: set[str] = set()
            uniq_pairs: List[Tuple[str, str]] = []
            for a, b, _ in sorted(pairs, key=lambda x: x[2], reverse=True):
                key = tuple(sorted((a, b)))
                if key[0] in seen or key[1] in seen:
                    continue
                uniq_pairs.append((a, b))
                seen.update(key)
                if len(uniq_pairs) >= max_plots:
                    break
        except Exception:
            # fallback: pierwsze pary
            uniq_pairs = [(cols[i], cols[i + 1]) for i in range(0, min(len(cols) - 1, max_plots))]

        for (x_col, y_col) in uniq_pairs[:max_plots]:
            try:
                fig = px.density_heatmap(
                    df,
                    x=x_col,
                    y=y_col,
                    nbinsx=50,
                    nbinsy=50,
                    title=f"Gęstość: {x_col} vs {y_col}",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(height=420, margin=dict(l=40, r=20, t=60, b=40))
                plots.append(fig)
            except Exception as e:
                logger.warning(f"Density plot failed for '{x_col}' vs '{y_col}': {e}")
        return plots

    # === TARGET ANALYSIS (rozkład + relacje) ===
    def _create_target_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> Dict[str, go.Figure]:
        plots: Dict[str, go.Figure] = {}
        if target_column not in df.columns:
            return plots

        target = df[target_column]

        # Rozkład targetu
        try:
            if pd.api.types.is_numeric_dtype(target):
                fig = px.histogram(
                    df,
                    x=target_column,
                    title=f"Rozkład zmiennej docelowej: {target_column}",
                    marginal="box",
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY,
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
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY,
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
                    if valid.sum() < 8:
                        continue
                    r = np.corrcoef(x[valid], y[valid])[0, 1]
                    if np.isfinite(r):
                        corr_abs[col] = float(abs(r))
                top = [k for k, _ in sorted(corr_abs.items(), key=lambda kv: kv[1], reverse=True)[: self.config.target_rel_top_features]]

                for i, col in enumerate(top):
                    trend = "ols" if len(df) <= 500_000 else None  # unikaj regresji na ogromnych zbiorach
                    fig = px.scatter(
                        df, x=col, y=target_column, trendline=trend,
                        title=f"{col} vs {target_column}",
                        opacity=0.6,
                        color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                    )
                    fig.update_layout(height=360)
                    plots[f"rel_{col}_vs_target"] = fig

            elif not pd.api.types.is_numeric_dtype(target) and numeric_cols:
                # violin dla numeric vs kategoryczny target (pierwsze N)
                for i, col in enumerate(numeric_cols[: self.config.target_rel_top_features]):
                    fig = px.violin(
                        df,
                        x=target_column,
                        y=col,
                        box=True,
                        points="outliers",
                        title=f"{col} vs {target_column}",
                        color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
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
                    tmp,
                    x=col,
                    color=target_column,
                    barmode="group",
                    title=f"{col} vs {target_column} (Top-k)",
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY,
                )
                fig.update_layout(height=360)
                plots[f"{col}_vs_{target_column}"] = fig

        except Exception as e:
            logger.warning(f"Target relations failed: {e}")

        return plots

    # === UTIL: LICZENIE OBIEKTÓW ===
    def _count_objects(self, viz: Dict[str, Any]) -> int:
        total = 0
        for v in viz.values():
            if isinstance(v, list):
                total += len(v)
            elif isinstance(v, dict):
                total += len(v)
            elif isinstance(v, go.Figure):
                total += 1
            elif v in (None, {}):
                total += 0
            else:
                total += 1
        return total
