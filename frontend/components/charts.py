"""
DataGenius PRO - Charts Utility
Zestaw spójnych, wielokrotnego użytku wykresów Plotly dla EDA i ML.

Użycie:
    from viz.charts import ChartFactory as CF

    fig = CF.histogram(df, "age", by="sex", bins=30, title="Rozkład wieku")
    CF.save_figure(fig, "reports/exports/figs/age_hist.html")

Wszystkie funkcje zwracają `plotly.graph_objects.Figure`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots
from sklearn import metrics

from config.settings import settings
from config.constants import (
    COLOR_PALETTE_PRIMARY,
    COLOR_PALETTE_CATEGORICAL,
)

# =========================
# Konfiguracja / Template
# =========================


def _get_base_layout() -> Dict[str, Any]:
    """Domyślny layout dla wykresów Plotly (ciemniejszy tekst, białe tło)."""
    return dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color="#2c3e50"),
        margin=dict(l=60, r=30, t=60, b=60),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )


def _apply_layout(fig: go.Figure, title: Optional[str] = None, height: Optional[int] = None) -> go.Figure:
    """Zastosuj domyślny layout + opcjonalny tytuł i wysokość."""
    fig.update_layout(**_get_base_layout())
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left"))
    if height:
        fig.update_layout(height=height)
    return fig


# =========================
# Utils
# =========================

def _ensure_series(y: Union[pd.Series, Sequence, np.ndarray]) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(y)


def _top_n_categories(series: pd.Series, top_n: int) -> pd.Series:
    vc = series.value_counts(dropna=False)
    if len(vc) <= top_n:
        return series
    top = vc.index[:top_n]
    return series.where(series.isin(top), other="__OTHER__")


def _as_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# =========================
# Fabryka wykresów
# =========================

class ChartFactory:
    """Zbiór statycznych metod do tworzenia spójnych wykresów Plotly."""

    # ---------- Ogólne ----------

    @staticmethod
    def histogram(
        df: pd.DataFrame,
        column: str,
        by: Optional[str] = None,
        bins: Optional[int] = 30,
        title: Optional[str] = None,
        height: int = 420,
    ) -> go.Figure:
        data = df.copy()
        if by and data[by].dtype == "O":
            data[by] = data[by].astype(str)

        fig = px.histogram(
            data,
            x=column,
            color=by,
            nbins=bins,
            marginal="box",
            barmode="overlay" if by else "relative",
            color_discrete_sequence=COLOR_PALETTE_PRIMARY,
        )
        fig.update_traces(opacity=0.85)
        fig.update_xaxes(title_text=column)
        fig.update_yaxes(title_text="Liczność")
        return _apply_layout(fig, title or f"Rozkład: {column}", height)

    @staticmethod
    def boxplot(
        df: pd.DataFrame,
        column: str,
        by: Optional[str] = None,
        title: Optional[str] = None,
        height: int = 420,
    ) -> go.Figure:
        data = df.copy()
        if by:
            data[by] = data[by].astype(str)
        fig = px.box(
            data,
            x=by,
            y=column,
            color=by,
            points="outliers",
            color_discrete_sequence=COLOR_PALETTE_CATEGORICAL,
        )
        fig.update_xaxes(title_text=by or "")
        fig.update_yaxes(title_text=column)
        return _apply_layout(fig, title or f"Boxplot: {column}", height)

    @staticmethod
    def bar_count(
        df: pd.DataFrame,
        column: str,
        top_n: int = 20,
        title: Optional[str] = None,
        height: int = 460,
    ) -> go.Figure:
        ser = df[column].astype(str)
        vc = ser.value_counts().head(top_n)
        fig = px.bar(
            x=vc.index,
            y=vc.values,
            labels={"x": column, "y": "Liczność"},
            color_discrete_sequence=COLOR_PALETTE_PRIMARY,
        )
        fig.update_xaxes(tickangle=45)
        return _apply_layout(fig, title or f"Top {top_n} wartości: {column}", height)

    @staticmethod
    def scatter(
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        trendline: bool = True,
        title: Optional[str] = None,
        height: int = 460,
        opacity: float = 0.8,
    ) -> go.Figure:
        data = df.copy()
        if color:
            data[color] = data[color].astype(str)
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            opacity=opacity,
            color_discrete_sequence=COLOR_PALETTE_CATEGORICAL,
        )
        if trendline:
            try:
                tl = px.scatter(
                    data, x=x, y=y, trendline="ols"
                )  # dodaje trace trendline
                # wyciągnij linię trendu i dodaj do fig
                for tr in tl.data:
                    if getattr(tr, "mode", "") == "lines":
                        fig.add_trace(tr)
            except Exception as e:
                logger.warning(f"Trendline OLS niedostępny: {e}")
        fig.update_xaxes(title_text=x)
        fig.update_yaxes(title_text=y)
        return _apply_layout(fig, title or f"{y} względem {x}", height)

    @staticmethod
    def correlation_heatmap(
        df: pd.DataFrame,
        method: str = "pearson",
        title: Optional[str] = "Macierz korelacji",
        height: int = 660,
        annotate: bool = True,
    ) -> go.Figure:
        corr = df.corr(method=method)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu_r",
                zmid=0,
                colorbar=dict(title="r"),
                text=np.round(corr.values, 2) if annotate else None,
                texttemplate="%{text}" if annotate else None,
                textfont={"size": 10},
            )
        )
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(autorange="reversed")
        return _apply_layout(fig, title, height)

    @staticmethod
    def missing_data(
        df: pd.DataFrame,
        title: str = "Brakujące wartości",
        height: Optional[int] = None,
    ) -> go.Figure:
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=True)
        if miss.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Brak brakujących danych! 🎉",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return _apply_layout(fig, title, 320)

        pct = (miss / len(df) * 100).round(2)
        fig = go.Figure(
            data=go.Bar(
                x=miss.values,
                y=miss.index,
                orientation="h",
                text=[f"{p}%" for p in pct],
                textposition="auto",
                marker_color=COLOR_PALETTE_PRIMARY[0],
            )
        )
        fig.update_xaxes(title_text="Liczba braków")
        fig.update_yaxes(title_text="Kolumna")
        h = height or max(320, 26 * len(miss))
        return _apply_layout(fig, title, h)

    @staticmethod
    def feature_importance(
        importance_df: pd.DataFrame,
        feature_col: str = "feature",
        importance_col: str = "importance",
        top_n: int = 20,
        title: str = "Ważność cech",
        height: Optional[int] = None,
    ) -> go.Figure:
        if importance_df is None or importance_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Brak danych o ważności cech.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return _apply_layout(fig, title, 320)

        imp = importance_df[[feature_col, importance_col]].copy()
        imp = imp.head(top_n)
        imp = imp.sort_values(importance_col, ascending=True)

        fig = go.Figure(
            data=go.Bar(
                x=imp[importance_col].values,
                y=imp[feature_col].values,
                orientation="h",
                marker_color=COLOR_PALETTE_PRIMARY[0],
                text=np.round(imp[importance_col].values, 4),
                textposition="auto",
            )
        )
        fig.update_xaxes(title_text=importance_col)
        fig.update_yaxes(title_text="Cecha")
        h = height or max(360, 24 * len(imp))
        return _apply_layout(fig, title, h)

    # ---------- Klasyfikacja ----------

    @staticmethod
    def confusion_matrix(
        y_true: Union[pd.Series, Sequence, np.ndarray],
        y_pred: Union[pd.Series, Sequence, np.ndarray],
        labels: Optional[Sequence] = None,
        normalize: Optional[str] = None,  # {'true', 'pred', 'all'}
        title: str = "Macierz pomyłek",
        height: int = 480,
    ) -> go.Figure:
        yt = _ensure_series(y_true)
        yp = _ensure_series(y_pred)
        cm = metrics.confusion_matrix(yt, yp, labels=labels, normalize=normalize)
        labels = labels or np.unique(np.concatenate([yt.unique(), yp.unique()]))

        z = cm
        z_text = np.round(cm, 3) if normalize else cm.astype(int)

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=z_text,
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(title="Udział" if normalize else "Liczba"),
            )
        )
        fig.update_xaxes(title_text="Predykcja")
        fig.update_yaxes(title_text="Prawda", autorange="reversed")
        return _apply_layout(fig, title, height)

    @staticmethod
    def roc_curve(
        y_true: Union[pd.Series, Sequence, np.ndarray],
        y_score: Union[pd.Series, Sequence, np.ndarray],
        title: str = "Krzywa ROC",
        height: int = 420,
    ) -> go.Figure:
        yt = _ensure_series(y_true)
        ys = _ensure_series(y_score)

        # Weryfikacja binarności
        classes = np.unique(yt.dropna())
        if len(classes) != 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Krzywa ROC wymaga klasyfikacji binarnej.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return _apply_layout(fig, title, 320)

        fpr, tpr, _ = metrics.roc_curve(yt, ys)
        auc = metrics.roc_auc_score(yt, ys)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Losowy", line=dict(dash="dash")))
        fig.update_xaxes(title_text="FPR")
        fig.update_yaxes(title_text="TPR")
        return _apply_layout(fig, title, height)

    @staticmethod
    def pr_curve(
        y_true: Union[pd.Series, Sequence, np.ndarray],
        y_score: Union[pd.Series, Sequence, np.ndarray],
        title: str = "Precision-Recall",
        height: int = 420,
    ) -> go.Figure:
        yt = _ensure_series(y_true)
        ys = _ensure_series(y_score)

        classes = np.unique(yt.dropna())
        if len(classes) != 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Krzywa PR wymaga klasyfikacji binarnej.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return _apply_layout(fig, title, 320)

        precision, recall, _ = metrics.precision_recall_curve(yt, ys)
        ap = metrics.average_precision_score(yt, ys)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"AP={ap:.3f}"))
        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text="Precision")
        return _apply_layout(fig, title, height)

    @staticmethod
    def lift_gain(
        y_true: Union[pd.Series, Sequence, np.ndarray],
        y_score: Union[pd.Series, Sequence, np.ndarray],
        bins: int = 10,
        title: str = "Lift & Gain (Decyle)",
        height: int = 460,
    ) -> go.Figure:
        yt = _ensure_series(y_true).astype(int)
        ys = _ensure_series(y_score).astype(float)

        df = pd.DataFrame({"y": yt, "score": ys}).sort_values("score", ascending=False)
        df["decile"] = pd.qcut(df["score"], q=bins, labels=False, duplicates="drop")
        # '0' to najwyższe score → odwróć
        df["decile"] = df["decile"].max() - df["decile"]

        grouped = df.groupby("decile").agg(events=("y", "sum"), total=("y", "count")).sort_index()
        grouped["event_rate"] = grouped["events"] / grouped["total"]
        grouped["cum_events"] = grouped["events"].cumsum()
        grouped["cum_total"] = grouped["total"].cumsum()
        grouped["cum_rate"] = grouped["cum_events"] / grouped["cum_total"]

        overall_rate = df["y"].mean()
        grouped["lift"] = grouped["event_rate"] / overall_rate
        grouped["cum_gain"] = grouped["cum_events"] / df["y"].sum()

        deciles = (grouped.index.astype(int) + 1).astype(str)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Lift", "Cumulative Gain"))
        # Lift
        fig.add_trace(
            go.Bar(x=deciles, y=grouped["lift"], marker_color=COLOR_PALETTE_PRIMARY[0], name="Lift"), row=1, col=1
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="#7f8c8d", row=1, col=1)
        fig.update_xaxes(title_text="Decyl (1=Najwyższe score)", row=1, col=1)
        fig.update_yaxes(title_text="Lift", row=1, col=1)

        # Gain
        fig.add_trace(
            go.Scatter(x=deciles, y=grouped["cum_gain"], mode="lines+markers", name="Gain"),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Decyl", row=1, col=2)
        fig.update_yaxes(title_text="Skumulowany Gain", row=1, col=2)

        return _apply_layout(fig, title, height)

    # ---------- Regresja ----------

    @staticmethod
    def residuals(
        y_true: Union[pd.Series, Sequence, np.ndarray],
        y_pred: Union[pd.Series, Sequence, np.ndarray],
        title: str = "Wykres reszt",
        height: int = 420,
    ) -> go.Figure:
        yt = _ensure_series(y_true).astype(float)
        yp = _ensure_series(y_pred).astype(float)
        res = yt - yp
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yp, y=res, mode="markers", name="Reszty", opacity=0.75))
        fig.add_hline(y=0, line_dash="dash", line_color="#7f8c8d")
        fig.update_xaxes(title_text="Predykcja")
        fig.update_yaxes(title_text="Reszta (y_true - y_pred)")
        return _apply_layout(fig, title, height)

    @staticmethod
    def y_true_vs_pred(
        y_true: Union[pd.Series, Sequence, np.ndarray],
        y_pred: Union[pd.Series, Sequence, np.ndarray],
        title: str = "y_true vs y_pred",
        height: int = 420,
    ) -> go.Figure:
        yt = _ensure_series(y_true).astype(float)
        yp = _ensure_series(y_pred).astype(float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yt, y=yp, mode="markers", name="Punkty", opacity=0.8))
        mn, mx = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Idealna linia", line=dict(dash="dash")))
        fig.update_xaxes(title_text="y_true")
        fig.update_yaxes(title_text="y_pred")
        return _apply_layout(fig, title, height)

    # ---------- SHAP / Importances ----------

    @staticmethod
    def shap_importance_from_dict(
        shap_payload: Optional[Dict[str, Any]],
        key: str = "shap_importance",
        feature_field: str = "feature",
        value_field: str = "shap_importance",
        top_n: int = 20,
        title: str = "SHAP – średnia bezwzględna ważność",
        height: Optional[int] = None,
    ) -> go.Figure:
        """
        Tworzy wykres ważności SHAP z payloadu zwracanego przez ModelExplainer._get_shap_explanations().
        Oczekuje struktury:
            shap_payload = {
                "shap_importance": {"feature": [...], "shap_importance": [...]},
                "shap_values": "Available" | ...
            }
        """
        if not shap_payload or key not in shap_payload or not shap_payload[key]:
            fig = go.Figure()
            fig.add_annotation(
                text="Brak danych SHAP.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return _apply_layout(fig, title, 320)

        d = shap_payload[key]
        # d może być dict-of-lists; spróbuj skonwertować do DataFrame
        try:
            imp = pd.DataFrame(d)
        except Exception:
            imp = pd.DataFrame.from_dict(d)

        imp = imp[[feature_field, value_field]].sort_values(value_field, ascending=False).head(top_n)
        imp = imp.sort_values(value_field, ascending=True)

        fig = go.Figure(
            data=go.Bar(
                x=imp[value_field].values,
                y=imp[feature_field].values,
                orientation="h",
                marker_color=COLOR_PALETTE_PRIMARY[1],
                text=np.round(imp[value_field].values, 4),
                textposition="auto",
            )
        )
        fig.update_xaxes(title_text=value_field)
        fig.update_yaxes(title_text="Cecha")
        h = height or max(360, 24 * len(imp))
        return _apply_layout(fig, title, h)

    # ---------- Kompozycje ----------

    @staticmethod
    def grid(
        figs: List[go.Figure],
        cols: int = 2,
        titles: Optional[List[str]] = None,
        shared_xaxes: bool = False,
        shared_yaxes: bool = False,
        height_per_row: int = 380,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Łączy kilka figur w jedną siatkę (kopiuje traces).
        Uwaga: skale kolorów/legendy mogą się dublować.
        """
        n = len(figs)
        rows = math.ceil(n / cols)
        subplot_titles = titles or [f.layout.title.text if f.layout.title.text else f"Wykres {i+1}" for i, f in enumerate(figs)]

        grid_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            horizontal_spacing=0.06,
            vertical_spacing=0.12,
        )

        for i, f in enumerate(figs):
            r = i // cols + 1
            c = i % cols + 1
            for tr in f.data:
                grid_fig.add_trace(tr, row=r, col=c)
            # Osie z podpisów oryginału
            x_title = f.layout.xaxis.title.text if "xaxis" in f.layout and f.layout.xaxis.title.text else ""
            y_title = f.layout.yaxis.title.text if "yaxis" in f.layout and f.layout.yaxis.title.text else ""
            grid_fig.update_xaxes(title_text=x_title, row=r, col=c)
            grid_fig.update_yaxes(title_text=y_title, row=r, col=c)

        total_height = rows * height_per_row + 80
        _apply_layout(grid_fig, title or "", total_height)
        return grid_fig

    # ---------- Zapisywanie ----------

    @staticmethod
    def save_figure(
        fig: go.Figure,
        path: Union[str, Path],
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: float = 2.0,
    ) -> Path:
        """
        Zapisuje figurę do HTML (zalecane) lub PNG/SVG (wymaga kaleido).
        """
        path = _as_path(path)
        suffix = path.suffix.lower()

        # Wymiar
        if width:
            fig.update_layout(width=width)
        if height:
            fig.update_layout(height=height)

        if suffix in (".html", ""):
            # domyślnie HTML
            out = path.with_suffix(".html")
            fig.write_html(out, include_plotlyjs="cdn", full_html=True)
        elif suffix in (".png", ".svg", ".pdf"):
            try:
                fig.write_image(path, scale=scale)
                out = path
            except Exception as e:
                logger.error(f"Zapisywanie do {suffix} wymaga pakietu 'kaleido'. Błąd: {e}")
                # fallback do HTML
                out = path.with_suffix(".html")
                fig.write_html(out, include_plotlyjs="cdn", full_html=True)
        else:
            # Nieznane — fallback na HTML
            out = path.with_suffix(".html")
            fig.write_html(out, include_plotlyjs="cdn", full_html=True)

        logger.success(f"Wykres zapisany: {out}")
        return out

    @staticmethod
    def save_batch(figs: Dict[str, go.Figure], directory: Optional[Union[str, Path]] = None, fmt: str = "html") -> List[Path]:
        """
        Zapisuje wiele figur pod nazwami kluczy słownika.
        """
        out_dir = Path(directory) if directory else (settings.REPORTS_PATH / "figs")
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for name, fig in figs.items():
            p = out_dir / f"{name}.{fmt}"
            saved.append(ChartFactory.save_figure(fig, p))
        return saved


# Syntactic sugar alias
CF = ChartFactory
