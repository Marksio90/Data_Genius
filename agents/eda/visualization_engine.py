# agents/visualization/visualization_engine.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Visualization Engine              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade interactive EDA visualizations (Plotly):                 ║
║    ✓ Distribution plots (histograms with marginal boxplots)               ║
║    ✓ Adaptive boxplot grids (outlier detection)                           ║
║    ✓ Correlation heatmaps (Pearson/Spearman with truncation)              ║
║    ✓ Categorical bar charts (top-K with OTHER aggregation)                ║
║    ✓ Missing data visualizations                                          ║
║    ✓ Time series plots (with rolling median overlay)                      ║
║    ✓ Density heatmaps (hexbin for large datasets)                         ║
║    ✓ Target analysis (distribution + feature relationships)               ║
║    ✓ Intelligent downsampling (120k point limit)                          ║
║    ✓ Defensive guards (variance checks, NA handling)                      ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "visualizations": {
        "distributions": List[go.Figure],
        "boxplots": go.Figure | {},
        "correlation_heatmap": go.Figure | {},
        "categorical_bars": List[go.Figure],
        "missing_data": go.Figure,
        "time_series": Dict[str, go.Figure],
        "density_plots": List[go.Figure],
        "target_analysis": Dict[str, go.Figure],
    },
    "n_visualizations": int,
    "metadata": {
        "sampled": bool,
        "sample_info": {"from_rows": int, "to_rows": int} | None,
        "plots_truncated": Dict[str, bool],
    },
    "warnings": List[str],
    "telemetry": {
        "elapsed_ms": float,
        "timings_ms": {
            "distributions": float,
            "boxplots": float,
            "correlation": float,
            "categorical": float,
            "missing": float,
            "timeseries": float,
            "density": float,
            "target": float,
        },
    },
    "version": "5.0-kosmos-enterprise",
}
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from functools import wraps

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("⚠ plotly not available — visualization engine disabled")
    go = None  # type: ignore
    px = None  # type: ignore

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
except ImportError:
    # Fallback for testing
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data = None
            self.errors = []
            self.warnings = []
        
        def add_error(self, msg: str):
            self.errors.append(msg)
        
        def add_warning(self, msg: str):
            self.warnings.append(msg)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

# Color palette (enterprise-grade)
COLOR_PALETTE_PRIMARY = [
    "#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706",
    "#10B981", "#F43F5E", "#0EA5E9", "#9333EA", "#EF4444"
]

@dataclass(frozen=True)
class VisualizationEngineConfig:
    """Enterprise configuration for visualization generation."""
    
    # Sampling & performance
    max_points: int = 120_000              # Maximum points for plotting
    random_state: int = 42
    
    # Plot limits
    max_plots_numeric: int = 12            # Max numeric distribution plots
    max_plots_categorical: int = 8         # Max categorical bar charts
    max_plots_density: int = 6             # Max density heatmaps
    top_k_categories: int = 12             # Top K categories to display
    max_cat_levels_stack: int = 8          # Max levels for stacked charts
    
    # Correlation heatmap
    heatmap_max_features: int = 60         # Max features in heatmap
    correlation_method: str = "pearson"    # "pearson" | "spearman"
    annotate_heatmap: bool = True          # Show correlation values
    
    # Target analysis
    target_rel_top_features: int = 4       # Top features for target relationships
    
    # Time series
    datetime_line_max: int = 6             # Max time series to plot
    ts_rolling_window: int = 7             # Rolling window for smoothing
    
    # Density plots
    hexbin_min_points: int = 10_000        # Minimum points for hexbin
    density_nbins: int = 50                # Number of bins for density
    
    # Behavior
    warn_on_truncation: bool = True        # Warn when truncating plots
    use_category_aggregation: bool = True  # Aggregate tail categories to OTHER
    
    # Layout
    default_height: int = 360              # Default plot height
    margin_dict: Dict[str, int] = None     # Default margins
    
    def __post_init__(self):
        # Set default margins if not provided
        if self.margin_dict is None:
            object.__setattr__(self, 'margin_dict', {"l": 40, "r": 20, "t": 60, "b": 40})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str):
    """Decorator for operation timing with intelligent logging."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(f"⏱ {operation_name}: {elapsed_ms:.2f}ms")
        return wrapper
    return decorator


def _safe_operation(operation_name: str, default_value: Any = None):
    """Decorator for defensive operations with fallback values."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠ {operation_name} failed: {type(e).__name__}: {str(e)[:80]}")
                return default_value
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Visualization Engine Agent
# ═══════════════════════════════════════════════════════════════════════════

class VisualizationEngine(BaseAgent):
    """
    **VisualizationEngine** — Enterprise interactive visualization generation.
    
    Responsibilities:
      1. Distribution plots (histograms with marginals)
      2. Boxplot grids for outlier detection
      3. Correlation heatmaps with smart truncation
      4. Categorical bar charts with aggregation
      5. Missing data visualizations
      6. Time series plots with rolling trends
      7. Density heatmaps for large datasets
      8. Target variable analysis
      9. Intelligent downsampling for performance
      10. Zero side-effects on input DataFrame
    
    Features:
      • Adaptive grid layouts
      • Top-K with OTHER aggregation
      • Variance-based feature selection
      • Rolling median overlays
      • Defensive NA handling
      • Color palette consistency
    """
    
    def __init__(self, config: Optional[VisualizationEngineConfig] = None) -> None:
        """Initialize visualization engine with optional custom configuration."""
        super().__init__(
            name="VisualizationEngine",
            description="Generates interactive Plotly visualizations for EDA"
        )
        self.config = config or VisualizationEngineConfig()
        self._log = logger.bind(agent="VisualizationEngine")
        warnings.filterwarnings("ignore")
        
        # Check plotly availability
        if not PLOTLY_AVAILABLE:
            self._log.warning("⚠ Plotly unavailable — visualization engine disabled")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            data: pd.DataFrame
        
        Optional:
            target_column: str — target variable for analysis
        """
        if "data" not in kwargs:
            raise ValueError("Required parameter 'data' not provided")
        
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError(f"'data' must be pd.DataFrame, got {type(kwargs['data']).__name__}")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("VisualizationEngine.execute")
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Generate comprehensive EDA visualizations.
        
        Args:
            data: Input DataFrame (not modified)
            target_column: Optional target variable name
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with visualization objects (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t0_total = time.perf_counter()
        
        try:
            # Input validation
            if data is None or not isinstance(data, pd.DataFrame):
                msg = "Invalid input: expected pandas DataFrame"
                result.add_error(msg)
                self._log.error(msg)
                result.data = self._empty_payload()
                return result
            
            if data.empty:
                self._log.warning("⚠ Empty DataFrame provided")
                result.add_warning("Empty DataFrame — no visualizations produced")
                payload = self._empty_payload()
                payload["warnings"] = ["empty_dataframe"]
                payload["telemetry"]["elapsed_ms"] = round((time.perf_counter() - t0_total) * 1000, 1)
                result.data = payload
                return result
            
            if not PLOTLY_AVAILABLE:
                msg = "Plotly not available — cannot generate visualizations"
                result.add_error(msg)
                self._log.error(msg)
                result.data = self._empty_payload()
                return result
            
            cfg = self.config
            warnings_list: List[str] = []
            plots_truncated: Dict[str, bool] = {}
            
            # ─── Data Preparation
            df = self._prepare_dataframe(data)
            
            # Detect column types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            datetime_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
            
            # Sampling for performance
            df_plot, sampled, sample_info = self._maybe_sample(df)
            
            visualizations: Dict[str, Any] = {}
            timings: Dict[str, float] = {}
            
            # ─── 1. Distribution Plots
            t0 = time.perf_counter()
            if numeric_cols:
                visualizations["distributions"], trunc = self._create_distribution_plots(
                    df_plot, numeric_cols
                )
                plots_truncated["distributions"] = trunc
            else:
                visualizations["distributions"] = []
                plots_truncated["distributions"] = False
            timings["distributions"] = (time.perf_counter() - t0) * 1000
            
            # ─── 2. Boxplots
            t0 = time.perf_counter()
            if numeric_cols:
                visualizations["boxplots"], trunc = self._create_boxplots(
                    df_plot, numeric_cols
                )
                plots_truncated["boxplots"] = trunc
            else:
                visualizations["boxplots"] = {}
                plots_truncated["boxplots"] = False
            timings["boxplots"] = (time.perf_counter() - t0) * 1000
            
            # ─── 3. Correlation Heatmap (use full df for precision)
            t0 = time.perf_counter()
            if len(numeric_cols) > 1:
                visualizations["correlation_heatmap"], trunc = self._create_correlation_heatmap(
                    df[numeric_cols]
                )
                plots_truncated["correlation"] = trunc
                if visualizations["correlation_heatmap"] == {}:
                    warnings_list.append("correlation_heatmap_failed")
            else:
                visualizations["correlation_heatmap"] = {}
                plots_truncated["correlation"] = False
            timings["correlation"] = (time.perf_counter() - t0) * 1000
            
            # ─── 4. Categorical Bar Charts
            t0 = time.perf_counter()
            if categorical_cols:
                visualizations["categorical_bars"], trunc = self._create_categorical_plots(
                    df_plot, categorical_cols
                )
                plots_truncated["categorical"] = trunc
            else:
                visualizations["categorical_bars"] = []
                plots_truncated["categorical"] = False
            timings["categorical"] = (time.perf_counter() - t0) * 1000
            
            # ─── 5. Missing Data
            t0 = time.perf_counter()
            visualizations["missing_data"] = self._create_missing_data_plot(df)
            timings["missing"] = (time.perf_counter() - t0) * 1000
            
            # ─── 6. Time Series
            t0 = time.perf_counter()
            if datetime_cols and numeric_cols:
                visualizations["time_series"] = self._create_time_series_plots(
                    df_plot, datetime_cols, numeric_cols
                )
            else:
                visualizations["time_series"] = {}
            timings["timeseries"] = (time.perf_counter() - t0) * 1000
            
            # ─── 7. Density Plots
            t0 = time.perf_counter()
            if numeric_cols and len(df_plot) >= cfg.hexbin_min_points:
                visualizations["density_plots"] = self._create_density_plots(
                    df_plot, numeric_cols
                )
            else:
                visualizations["density_plots"] = []
            timings["density"] = (time.perf_counter() - t0) * 1000
            
            # ─── 8. Target Analysis
            t0 = time.perf_counter()
            if target_column and target_column in df.columns:
                visualizations["target_analysis"] = self._create_target_analysis(
                    df_plot, target_column, numeric_cols, categorical_cols
                )
            else:
                visualizations["target_analysis"] = {}
            timings["target"] = (time.perf_counter() - t0) * 1000
            
            # ─── Compile Results
            n_viz = self._count_visualizations(visualizations)
            elapsed_ms = round((time.perf_counter() - t0_total) * 1000, 1)
            
            result.data = {
                "visualizations": visualizations,
                "n_visualizations": int(n_viz),
                "metadata": {
                    "sampled": sampled,
                    "sample_info": sample_info,
                    "plots_truncated": plots_truncated,
                },
                "warnings": warnings_list,
                "telemetry": {
                    "elapsed_ms": elapsed_ms,
                    "timings_ms": {k: round(v, 1) for k, v in timings.items()},
                },
                "version": "5.0-kosmos-enterprise",
            }
            
            self._log.success(
                f"✓ Generated {n_viz} visualizations | "
                f"elapsed={elapsed_ms:.1f}ms | "
                f"sampled={sampled}"
            )
        
        except Exception as e:
            msg = f"Visualization generation failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            result.data = self._empty_payload()
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Data Preparation
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("data_preparation")
    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for visualization.
        
        Operations:
          • Deep copy (no side-effects)
          • Replace ±Inf with NaN
        
        Returns:
            Prepared DataFrame copy
        """
        df = data.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        return df
    
    @_timeit("sampling")
    def _maybe_sample(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, bool, Optional[Dict[str, int]]]:
        """
        Sample DataFrame if too large for performant visualization.
        
        Returns:
            Tuple of (sampled_df, was_sampled, sample_info)
        """
        cfg = self.config
        n = len(df)
        
        if n > cfg.max_points:
            self._log.info(
                f"Downsampling for plotting: {n:,} → {cfg.max_points:,} rows"
            )
            
            try:
                sampled_df = df.sample(n=cfg.max_points, random_state=cfg.random_state)
            except Exception:
                # Fallback to head (deterministic)
                sampled_df = df.head(cfg.max_points)
            
            sample_info = {
                "from_rows": int(n),
                "to_rows": int(cfg.max_points),
            }
            
            return sampled_df, True, sample_info
        
        return df, False, None
    
    # ───────────────────────────────────────────────────────────────────
    # Distribution Plots (Histograms with Marginal Boxplots)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("distribution_plots")
    @_safe_operation("distribution_plots", default_value=([], False))
    def _create_distribution_plots(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[List[go.Figure], bool]:
        """
        Create distribution plots with marginal boxplots.
        
        Returns:
            Tuple of (list_of_figures, was_truncated)
        """
        cfg = self.config
        plots: List[go.Figure] = []
        
        # Filter numeric columns
        cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        truncated = len(cols) > cfg.max_plots_numeric
        cols = cols[:cfg.max_plots_numeric]
        
        if truncated and cfg.warn_on_truncation:
            self._log.info(
                f"⚠ Truncating numeric columns for distributions: "
                f"{len(columns)} → {cfg.max_plots_numeric}"
            )
        
        for i, col in enumerate(cols):
            try:
                # Heuristic nbins
                unique_count = df[col].dropna().nunique()
                nbins = int(np.clip(int(np.sqrt(max(10, unique_count))), 20, 120))
                
                fig = px.histogram(
                    df,
                    x=col,
                    title=f"Distribution: {col}",
                    marginal="box",
                    opacity=0.9,
                    color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                    nbins=nbins,
                )
                
                fig.update_layout(
                    showlegend=False,
                    height=cfg.default_height,
                    bargap=0.02,
                    margin=cfg.margin_dict,
                )
                
                plots.append(fig)
            
            except Exception as e:
                self._log.warning(f"⚠ Histogram failed for '{col}': {e}")
        
        return plots, truncated
    
    # ───────────────────────────────────────────────────────────────────
    # Boxplot Grid
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("boxplots")
    @_safe_operation("boxplots", default_value=({}, False))
    def _create_boxplots(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[go.Figure | Dict, bool]:
        """
        Create adaptive grid of boxplots for outlier detection.
        
        Returns:
            Tuple of (figure_or_empty_dict, was_truncated)
        """
        cfg = self.config
        
        # Filter numeric columns
        cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        truncated = len(cols) > cfg.max_plots_numeric
        cols = cols[:cfg.max_plots_numeric]
        
        if len(cols) == 0:
            return {}, False
        
        if truncated and cfg.warn_on_truncation:
            self._log.info(
                f"⚠ Truncating numeric columns for boxplots: "
                f"{len(columns)} → {cfg.max_plots_numeric}"
            )
        
        # Adaptive grid
        n = len(cols)
        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))
        
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=cols,
            vertical_spacing=0.08,
            horizontal_spacing=0.04,
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
                self._log.warning(f"⚠ Boxplot trace failed for '{col}': {e}")
        
        fig.update_layout(
            title_text="Boxplots — Outlier Detection",
            showlegend=False,
            height=max(360, nrows * 300),
            margin=cfg.margin_dict,
        )
        
        return fig, truncated
    
    # ───────────────────────────────────────────────────────────────────
    # Correlation Heatmap
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("correlation_heatmap")
    @_safe_operation("correlation_heatmap", default_value=({}, False))
    def _create_correlation_heatmap(
        self,
        df_num: pd.DataFrame
    ) -> Tuple[go.Figure | Dict, bool]:
        """
        Create correlation heatmap with smart truncation.
        
        Returns:
            Tuple of (figure_or_empty_dict, was_truncated)
        """
        cfg = self.config
        
        # Remove zero-variance features
        variances = df_num.var(numeric_only=True)
        keep_var = variances[variances > 0].index
        data_num = df_num[keep_var]
        
        if data_num.shape[1] == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No features with positive variance",
                x=0.5, y=0.5,
                showarrow=False,
                xref="paper", yref="paper",
            )
            fig.update_layout(height=280, margin={"l": 20, "r": 20, "t": 40, "b": 20})
            return fig, False
        
        # Truncate to top-K by variance
        truncated = data_num.shape[1] > cfg.heatmap_max_features
        if truncated:
            top_features = variances.loc[keep_var].sort_values(ascending=False).head(cfg.heatmap_max_features).index
            data_num = data_num[top_features]
            self._log.info(
                f"⚠ Truncating heatmap features: "
                f"{len(keep_var)} → {cfg.heatmap_max_features}"
            )
        
        # Compute correlation
        corr = data_num.corr(method=cfg.correlation_method, numeric_only=True).round(3).fillna(0.0)
        vals = corr.values
        
        # Upper triangle mask
        mask = np.triu(np.ones_like(vals, dtype=bool), k=1)
        display_vals = vals.copy()
        display_vals[mask] = np.nan
        
        # Annotations (disable for large matrices)
        do_annotate = cfg.annotate_heatmap and corr.shape[0] <= 40
        text = None
        if do_annotate:
            text = np.where(
                np.isnan(display_vals),
                "",
                np.vectorize(lambda v: f"{v:.2f}")(display_vals)
            )
        
        fig = go.Figure(
            data=go.Heatmap(
                z=display_vals,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                zmid=0,
                colorbar=dict(title="Correlation"),
                text=text,
                texttemplate="%{text}" if do_annotate else None,
                hovertemplate="(%{y}, %{x}) = %{z}<extra></extra>",
            )
        )
        
        fig.update_layout(
            title=f"Correlation Matrix ({cfg.correlation_method.title()})",
            xaxis_title="Features",
            yaxis_title="Features",
            height=max(520, 28 * corr.shape[0]),
            margin={"l": 60, "r": 20, "t": 60, "b": 60},
        )
        
        return fig, truncated
    
    # ───────────────────────────────────────────────────────────────────
    # Categorical Bar Charts
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("categorical_plots")
    @_safe_operation("categorical_plots", default_value=([], False))
    def _create_categorical_plots(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[List[go.Figure], bool]:
        """
        Create bar charts for categorical features with top-K + OTHER.
        
        Returns:
            Tuple of (list_of_figures, was_truncated)
        """
        cfg = self.config
        plots: List[go.Figure] = []
        
        truncated = len(columns) > cfg.max_plots_categorical
        cols = columns[:cfg.max_plots_categorical]
        
        if truncated and cfg.warn_on_truncation:
            self._log.info(
                f"⚠ Truncating categorical columns: "
                f"{len(columns)} → {cfg.max_plots_categorical}"
            )
        
        for i, col in enumerate(cols):
            try:
                s = df[col].astype("object")
                vc = s.value_counts(dropna=False)
                
                # Aggregate tail to OTHER
                if cfg.use_category_aggregation and len(vc) > cfg.top_k_categories:
                    top_vals = vc.head(cfg.top_k_categories - 1)
                    other_sum = int(vc.iloc[cfg.top_k_categories - 1:].sum())
                    vc = pd.concat([top_vals, pd.Series({"OTHER": other_sum})])
                
                x_labels = [str(x) for x in vc.index]
                
                fig = px.bar(
                    x=x_labels,
                    y=vc.values,
                    title=f"Top {min(cfg.top_k_categories, len(vc))} Values: {col}",
                    labels={"x": col, "y": "Count"},
                    color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                )
                
                fig.update_layout(
                    showlegend=False,
                    height=cfg.default_height,
                    xaxis_tickangle=-25,
                    margin={"l": 40, "r": 20, "t": 60, "b": 60},
                )
                
                plots.append(fig)
            
            except Exception as e:
                self._log.warning(f"⚠ Categorical bar failed for '{col}': {e}")
        
        return plots, truncated
    
    # ───────────────────────────────────────────────────────────────────
    # Missing Data Visualization
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("missing_data_plot")
    @_safe_operation("missing_data_plot", default_value=go.Figure())
    def _create_missing_data_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create missing data visualization (horizontal bar chart).

        Returns:
            Plotly Figure showing missing data percentages.
        """
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=True)

        if missing.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No Missing Data 🎉",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color=COLOR_PALETTE_PRIMARY[2]),
            )
            fig.update_layout(
                height=280,
                margin={"l": 20, "r": 20, "t": 40, "b": 20},
            )
            return fig

        # Calculate percentages
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
            title="Missing Values (% per Column)",
            xaxis_title="Missing Percentage",
            yaxis_title="Column",
            height=max(320, 26 * len(missing)),
            margin={"l": 140, "r": 30, "t": 60, "b": 40},
        )

        return fig

    # ───────────────────────────────────────────────────────────────────
    # Time Series Plots
    # ───────────────────────────────────────────────────────────────────

    @_timeit("time_series_plots")
    @_safe_operation("time_series_plots", default_value={})
    def _create_time_series_plots(
        self,
        df: pd.DataFrame,
        datetime_cols: List[str],
        numeric_cols: List[str]
    ) -> Dict[str, go.Figure]:
        """
        Create time series plots with rolling median overlay.

        Returns:
            Dictionary of time series figures.
        """
        cfg = self.config
        plots: Dict[str, go.Figure] = {}

        # Select datetime column with fewest NaN
        dt_col = min(datetime_cols, key=lambda c: df[c].isna().sum())

        # Prepare data
        tmp = df[[dt_col] + numeric_cols].copy()
        tmp = tmp.dropna(subset=[dt_col]).sort_values(dt_col)

        if len(tmp) == 0:
            return plots

        # Select top-K numeric by variance
        variances = tmp[numeric_cols].var(numeric_only=True)
        top_cols = variances.sort_values(ascending=False).head(cfg.datetime_line_max).index.tolist()
        sel = [c for c in top_cols if pd.api.types.is_numeric_dtype(tmp[c])]

        if not sel:
            return plots

        # Create plot
        fig = go.Figure()

        for i, col in enumerate(sel):
            y = pd.to_numeric(tmp[col], errors="coerce")
            color = COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]

            # Main line
            fig.add_trace(
                go.Scatter(
                    x=tmp[dt_col],
                    y=y,
                    mode="lines",
                    name=col,
                    line=dict(width=1.4, color=color),
                )
            )

            # Rolling median overlay
            window = max(2, cfg.ts_rolling_window)
            rolling_median = y.rolling(window, min_periods=max(2, window // 2)).median()

            fig.add_trace(
                go.Scatter(
                    x=tmp[dt_col],
                    y=rolling_median,
                    mode="lines",
                    name=f"{col} (rolling {window})",
                    line=dict(width=2.0, dash="dot", color=color),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f"Time Series ({dt_col}) — Top {len(sel)} Variables",
            xaxis_title=dt_col,
            yaxis_title="Value",
            height=max(420, 280 + 20 * len(sel)),
            margin=cfg.margin_dict,
            hovermode="x unified",
        )

        plots["timeseries_main"] = fig
        return plots

    # ───────────────────────────────────────────────────────────────────
    # Density Plots (Hexbin/Heatmap)
    # ───────────────────────────────────────────────────────────────────

    @_timeit("density_plots")
    @_safe_operation("density_plots", default_value=[])
    def _create_density_plots(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> List[go.Figure]:
        """
        Create density heatmaps for pairs of numeric features.

        Returns:
            List of density heatmap figures.
        """
        cfg = self.config
        plots: List[go.Figure] = []

        # Filter numeric columns
        cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]

        if len(cols) < 2:
            return plots

        # Select pairs by absolute correlation
        try:
            num_df = df[cols].copy()
            corr = num_df.corr(numeric_only=True).abs()
            np.fill_diagonal(corr.values, 0.0)

            # Find high-correlation pairs
            pairs: List[Tuple[str, str, float]] = []
            for i, col_a in enumerate(corr.columns):
                col_b = corr.iloc[i].idxmax()
                corr_val = float(corr.iloc[i][col_b])
                pairs.append((col_a, col_b, corr_val))

            # Sort by correlation and deduplicate
            seen: Set[str] = set()
            unique_pairs: List[Tuple[str, str]] = []

            for col_a, col_b, _ in sorted(pairs, key=lambda x: x[2], reverse=True):
                key = tuple(sorted((col_a, col_b)))
                if key[0] in seen or key[1] in seen:
                    continue
                unique_pairs.append((col_a, col_b))
                seen.update(key)
                if len(unique_pairs) >= cfg.max_plots_density:
                    break

        except Exception:
            # Fallback: sequential pairs
            unique_pairs = [
                (cols[i], cols[i + 1])
                for i in range(0, min(len(cols) - 1, cfg.max_plots_density))
            ]

        # Generate density plots
        for x_col, y_col in unique_pairs[:cfg.max_plots_density]:
            try:
                fig = px.density_heatmap(
                    df,
                    x=x_col,
                    y=y_col,
                    nbinsx=cfg.density_nbins,
                    nbinsy=cfg.density_nbins,
                    title=f"Density: {x_col} vs {y_col}",
                    color_continuous_scale="Viridis",
                )

                fig.update_layout(
                    height=420,
                    margin=cfg.margin_dict,
                )

                plots.append(fig)

            except Exception as e:
                self._log.warning(f"⚠ Density plot failed for '{x_col}' vs '{y_col}': {e}")

        return plots

    # ───────────────────────────────────────────────────────────────────
    # Target Analysis
    # ───────────────────────────────────────────────────────────────────

    @_timeit("target_analysis")
    @_safe_operation("target_analysis", default_value={})
    def _create_target_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> Dict[str, go.Figure]:
        """
        Create target variable analysis plots.

        Returns:
            Dictionary of target-related figures.
        """
        cfg = self.config
        plots: Dict[str, go.Figure] = {}

        if target_column not in df.columns:
            return plots

        target = df[target_column]

        # 1) Target Distribution
        try:
            if pd.api.types.is_numeric_dtype(target):
                # Numeric target: histogram with boxplot
                fig = px.histogram(
                    df,
                    x=target_column,
                    title=f"Target Distribution: {target_column}",
                    marginal="box",
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY,
                )
                fig.update_layout(
                    showlegend=False,
                    height=cfg.default_height,
                    margin=cfg.margin_dict,
                )
                plots["target_distribution"] = fig

            else:
                # Categorical target: bar chart
                vc = target.astype("object").value_counts(dropna=False)
                if len(vc) > cfg.max_cat_levels_stack:
                    vc = vc.head(cfg.max_cat_levels_stack)

                fig = px.bar(
                    x=[str(x) for x in vc.index],
                    y=vc.values,
                    title=f"Target Class Distribution: {target_column}",
                    labels={"x": target_column, "y": "Count"},
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY,
                )
                fig.update_layout(
                    showlegend=False,
                    height=cfg.default_height,
                    margin=cfg.margin_dict,
                )
                plots["target_distribution"] = fig

        except Exception as e:
            self._log.warning(f"⚠ Target distribution failed: {e}")

        # 2) Feature-Target Relationships
        try:
            if pd.api.types.is_numeric_dtype(target) and numeric_cols:
                # Numeric target: scatter plots with top correlated features
                y = pd.to_numeric(target, errors="coerce")
                correlations: Dict[str, float] = {}

                for col in numeric_cols:
                    if col == target_column:
                        continue

                    x = pd.to_numeric(df[col], errors="coerce")
                    valid = x.notna() & y.notna()

                    if valid.sum() < 8:
                        continue

                    try:
                        r = np.corrcoef(x[valid], y[valid])[0, 1]
                        if np.isfinite(r):
                            correlations[col] = float(abs(r))
                    except Exception:
                        continue

                # Top features
                top_features = [
                    k for k, _ in sorted(
                        correlations.items(),
                        key=lambda kv: kv[1],
                        reverse=True
                    )[:cfg.target_rel_top_features]
                ]

                for i, col in enumerate(top_features):
                    # Use trendline only for smaller datasets
                    trendline = "ols" if len(df) <= 500_000 else None

                    fig = px.scatter(
                        df,
                        x=col,
                        y=target_column,
                        trendline=trendline,
                        title=f"{col} vs {target_column}",
                        opacity=0.6,
                        color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                    )
                    fig.update_layout(
                        height=cfg.default_height,
                        margin=cfg.margin_dict,
                    )
                    plots[f"rel_{col}_vs_target"] = fig

            elif not pd.api.types.is_numeric_dtype(target) and numeric_cols:
                # Categorical target: violin plots
                for i, col in enumerate(numeric_cols[:cfg.target_rel_top_features]):
                    fig = px.violin(
                        df,
                        x=target_column,
                        y=col,
                        box=True,
                        points="outliers",
                        title=f"{col} by {target_column}",
                        color_discrete_sequence=[COLOR_PALETTE_PRIMARY[i % len(COLOR_PALETTE_PRIMARY)]],
                    )
                    fig.update_layout(
                        height=cfg.default_height,
                        margin=cfg.margin_dict,
                    )
                    plots[f"rel_{col}_by_{target_column}"] = fig

            # Categorical feature vs categorical target
            if not pd.api.types.is_numeric_dtype(target) and categorical_cols:
                col = categorical_cols[0]
                s = df[col].astype("object")
                top_values = s.value_counts().head(cfg.max_cat_levels_stack).index
                tmp = df[df[col].isin(top_values)].copy()

                fig = px.histogram(
                    tmp,
                    x=col,
                    color=target_column,
                    barmode="group",
                    title=f"{col} vs {target_column} (Top-K)",
                    color_discrete_sequence=COLOR_PALETTE_PRIMARY,
                )
                fig.update_layout(
                    height=cfg.default_height,
                    margin=cfg.margin_dict,
                )
                plots[f"{col}_vs_{target_column}"] = fig

        except Exception as e:
            self._log.warning(f"⚠ Target relationships failed: {e}")

        return plots

    # ───────────────────────────────────────────────────────────────────
    # Helper Functions
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _count_visualizations(viz: Dict[str, Any]) -> int:
        """
        Count total number of visualization objects.

        Returns:
            Total count of plots.
        """
        total = 0
        for v in viz.values():
            if isinstance(v, list):
                total += len(v)
            elif isinstance(v, dict):
                total += len(v)
            elif v is not None and v != {}:
                total += 1
        return total

    # ───────────────────────────────────────────────────────────────────
    # Empty Payload (Fallback)
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_payload() -> Dict[str, Any]:
        """Generate empty payload for failed/invalid input."""
        return {
            "visualizations": {
                "distributions": [],
                "boxplots": {},
                "correlation_heatmap": {},
                "categorical_bars": [],
                "missing_data": go.Figure() if PLOTLY_AVAILABLE else {},
                "time_series": {},
                "density_plots": [],
                "target_analysis": {},
            },
            "n_visualizations": 0,
            "metadata": {
                "sampled": False,
                "sample_info": None,
                "plots_truncated": {},
            },
            "warnings": [],
            "telemetry": {
                "elapsed_ms": 0.0,
                "timings_ms": {
                    "distributions": 0.0,
                    "boxplots": 0.0,
                    "correlation": 0.0,
                    "categorical": 0.0,
                    "missing": 0.0,
                    "timeseries": 0.0,
                    "density": 0.0,
                    "target": 0.0,
                },
            },
            "version": "5.0-kosmos-enterprise",
        }
