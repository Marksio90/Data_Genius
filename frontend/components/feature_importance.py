# src/frontend/feature_importance.py
# === OPIS MODUŁU ===
# Interaktywny panel Feature Importance (FI) PRO+++:
# - Metody: Native / Permutation / SHAP (opcjonalnie) / Linear Coeffs
# - Auto-trening baseline jeśli brak modelu w session_state
# - Stabilizacja: cache, walidacje, defensywne fallbacki
# - Eksport wyników: CSV/JSON
# - Wizualizacja: poziomy bar chart (Plotly)

from __future__ import annotations

import io
import json
import math
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Logger (zgodny z Twoim ekosystemem) ===
try:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("feature_importance")

# === NAZWA_SEKCJI === Dataclasses konfiguracji i wyników ===

@dataclass
class FIConfig:
    method: str = "auto"            # auto | native | permutation | shap | linear
    top_n: int = 30
    n_repeats: int = 5              # permutation
    sample_size: int = 10000
    random_state: int = 42
    scoring: Optional[str] = None   # np. "r2", "neg_mean_squared_error", "roc_auc", "f1"

@dataclass
class FIResult:
    feature: str
    importance: float
    std: Optional[float] = None
    method: str = "auto"
    meta: Dict[str, Any] = None  # np. {"problem_type": "...", "n_features": 123, "n_rows": 456}

# === NAZWA_SEKCJI === Pomocnicze: detekcja problemu i przygotowanie danych ===

def _detect_problem_type(y: pd.Series) -> str:
    """Heurystyka: klasyfikacja gdy <= 20 klas i dtype nie-float; w innym razie regresja."""
    y_non_null = y.dropna()
    nunique = y_non_null.nunique()
    if pd.api.types.is_bool_dtype(y_non_null):
        return "classification"
    if pd.api.types.is_integer_dtype(y_non_null) and nunique <= min(20, max(2, int(len(y_non_null) ** 0.5))):
        return "classification"
    if pd.api.types.is_object_dtype(y_non_null) and nunique <= 50:
        return "classification"
    return "regression"

def _prepare_xy(
    df: pd.DataFrame,
    target: str,
    drop_cols: Iterable[str],
    sample_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, Pipeline, List[str]]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' nie istnieje w danych.")
    use_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = use_df[target]
    X = use_df.drop(columns=[target])

    if X.empty:
        raise ValueError("Brak cech po odjęciu targetu/wykluczeń.")

    # Downsample dla szybkości/stabilności
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=random_state)
        y = y.loc[X.index]

    # Kolumny kat./num
    cat_cols = [c for c in X.columns if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocesor: imputacja + OneHot dla kateg.
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Zbuduj pipeline „goły” (bez modelu; model podpinamy później)
    pipe = Pipeline(steps=[("pre", pre)])
    Xt = pipe.fit_transform(X)
    # Nazwy kolumn po one-hot
    feat_names = []
    if num_cols:
        feat_names.extend(num_cols)
    if cat_cols:
        try:
            onehot: OneHotEncoder = pipe.named_steps["pre"].transformers_[1][1].named_steps["onehot"]  # type: ignore
            oh_names = list(onehot.get_feature_names_out(cat_cols))
        except Exception:
            oh_names = [f"{c}__{i}" for c in cat_cols for i in range(1000)]  # awaryjnie
        feat_names.extend(oh_names)

    # Ujednolicenie typów
    Xt = np.asarray(Xt)
    y = pd.Series(y, index=X.index, name=target)

    return pd.DataFrame(Xt, index=X.index, columns=feat_names), y, pipe, feat_names

# === NAZWA_SEKCJI === Trenowanie baseline, jeśli brak modelu ===

def _fit_baseline_model(problem_type: str, random_state: int) -> Any:
    if problem_type == "classification":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )

# === NAZWA_SEKCJI === Obliczanie Feature Importance (różne tryby) ===

def _fi_native(model: Any, X: pd.DataFrame, y: pd.Series, feat_names: List[str]) -> pd.DataFrame:
    """Obsługuje: feature_importances_ (drzewa), współczynniki (Linear/Logistic)."""
    # feature_importances_
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            vals = np.asarray(imp, dtype=float)
            return pd.DataFrame({"feature": feat_names, "importance": vals})
    except Exception:
        pass

    # Współczynniki
    for attr in ("coef_", "coefs_"):
        try:
            coef = getattr(model, attr, None)
            if coef is None:
                continue
            coef = np.asarray(coef, dtype=float)
            if coef.ndim == 2 and coef.shape[0] > 1:  # one-vs-rest itp.
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef).ravel()
            return pd.DataFrame({"feature": feat_names, "importance": coef})
        except Exception:
            continue

    raise ValueError("Model nie udostępnia natywnej ważności cech (brak feature_importances_/coef_).")

def _fi_permutation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: Optional[str],
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    try:
        scorer = get_scorer(scoring) if scoring else None
    except Exception:
        scorer = None
    r = permutation_importance(
        model, X, y,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    return pd.DataFrame(
        {"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std}
    )

def _fi_shap(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap  # type: ignore
    except Exception as e:
        raise RuntimeError("Brak pakietu 'shap' — zainstaluj, aby korzystać z tej metody.") from e

    try:
        # Próba: TreeExplainer → KernelExplainer (fallback)
        try:
            explainer = shap.TreeExplainer(model)
            sh_vals = explainer.shap_values(X, check_additivity=False)
            if isinstance(sh_vals, list):  # klasyfikacja wieloklasowa
                mean_abs = np.mean([np.abs(v).mean(axis=0) for v in sh_vals], axis=0)
            else:
                mean_abs = np.abs(sh_vals).mean(axis=0)
        except Exception:
            explainer = shap.Explainer(model, X)
            sh_vals = explainer(X)
            mean_abs = np.abs(sh_vals.values).mean(axis=0)
        return pd.DataFrame({"feature": X.columns, "importance": mean_abs})
    except Exception as e:
        raise RuntimeError(f"SHAP nie powiódł się: {e}") from e

# === NAZWA_SEKCJI === Cache: obliczenia FI ===

@st.cache_data(show_spinner=True, ttl=3600, max_entries=8)
def _compute_fi_cached(
    df: pd.DataFrame,
    target: str,
    method: str,
    scoring: Optional[str],
    top_n: int,
    n_repeats: int,
    sample_size: int,
    random_state: int,
    columns_exclude: Tuple[str, ...],
    use_existing_model: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Przygotowanie danych
    X, y, pre, feat_names = _prepare_xy(
        df=df,
        target=target,
        drop_cols=list(columns_exclude),
        sample_size=sample_size,
        random_state=random_state,
    )
    problem_type = _detect_problem_type(y)

    # Model: użyj istniejącego (jeśli podano i fitted), w innym razie baseline
    model = None
    if use_existing_model and "trained_model" in st.session_state:
        model = st.session_state.get("trained_model")  # model ma już widzieć cechy w postaci 'feat_names'
        # jeśli istniejący model to Pipeline, spróbuj go użyć bez naszego preprocesu
        if isinstance(model, Pipeline):
            try:
                model.predict(X.iloc[:2])
            except Exception:
                # dopasuj do X (ostrożnie)
                try:
                    model.fit(X, y)
                except Exception:
                    model = None
    # Jeśli brak modelu lub nie nadaje się → baseline
    if model is None:
        base = _fit_baseline_model(problem_type, random_state)
        model = base.fit(X, y)

    # Oblicz FI wg metody
    used_method = method
    fi_df: pd.DataFrame
    if method == "auto":
        # preferencja: native → permutation → linear → shap (shap kosztowny)
        try:
            fi_df = _fi_native(model, X, y, feat_names)
            used_method = "native"
        except Exception:
            try:
                fi_df = _fi_permutation(model, X, y, scoring, n_repeats, random_state)
                used_method = "permutation"
            except Exception:
                try:
                    fi_df = _fi_shap(model, X)
                    used_method = "shap"
                except Exception:
                    # Ostatnia próba: jeśli ma coef_
                    fi_df = _fi_native(model, X, y, feat_names)  # podniesie błąd jeśli się nie uda
    else:
        if method == "native":
            fi_df = _fi_native(model, X, y, feat_names)
        elif method == "permutation":
            fi_df = _fi_permutation(model, X, y, scoring, n_repeats, random_state)
        elif method == "shap":
            fi_df = _fi_shap(model, X)
        elif method == "linear":
            # Wymuś linearny baseline (dla spójności)
            from sklearn.linear_model import LogisticRegression, Ridge
            if problem_type == "classification":
                lin = LogisticRegression(max_iter=200, n_jobs=None, random_state=random_state)
            else:
                lin = Ridge(random_state=random_state)
            lin.fit(X, y)
            fi_df = _fi_native(lin, X, y, feat_names)
            used_method = "native"
        else:
            raise ValueError(f"Nieznana metoda FI: {method}")

    # Normalizacja i Top-N
    fi_df = fi_df.copy()
    fi_df["importance"] = fi_df["importance"].astype(float)
    fi_df = fi_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["importance"])
    # Ustandaryzuj do sumy = 1 (jeśli możliwe)
    total = fi_df["importance"].sum()
    if total > 0:
        fi_df["importance_norm"] = fi_df["importance"] / total
    else:
        fi_df["importance_norm"] = fi_df["importance"]

    fi_df = fi_df.sort_values("importance_norm", ascending=False).head(top_n)
    meta = {
        "target": target,
        "method": used_method,
        "scoring": scoring,
        "problem_type": problem_type,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "sample_size": sample_size,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "used_existing_model": bool(use_existing_model and "trained_model" in st.session_state),
    }
    return fi_df.reset_index(drop=True), meta

# === NAZWA_SEKCJI === Eksporty wyników ===

def _export_csv(df: pd.DataFrame) -> bytes:
    cols = [c for c in ["feature", "importance", "std", "importance_norm"] if c in df.columns]
    return df[cols].to_csv(index=False).encode("utf-8")

def _export_json(df: pd.DataFrame, meta: Dict[str, Any]) -> bytes:
    payload = {"meta": meta, "features": df.to_dict(orient="records")}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

# === NAZWA_SEKCJI === Wizualizacja (Plotly) ===

def _plot_bar(df: pd.DataFrame, title: str = "Feature Importance (Top)") -> "plotly.graph_objs._figure.Figure":
    plot_df = df.copy()
    if "importance_norm" in plot_df:
        plot_df["importance_%"] = 100.0 * plot_df["importance_norm"]
        color_col = "importance_%"
        x_col = "importance_%"
    else:
        color_col = "importance"
        x_col = "importance"
    fig = px.bar(
        plot_df.sort_values(x_col, ascending=True),
        x=x_col,
        y="feature",
        orientation="h",
        title=title,
        text=np.round(plot_df[x_col], 3),
        labels={x_col: x_col, "feature": "feature"},
    )
    fig.update_layout(
        height=max(400, 20 * len(plot_df)),
        xaxis_title=x_col,
        yaxis_title="feature",
        bargap=0.2,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig

# === NAZWA_SEKCJI === UI główne ===

def render_feature_importance(
    *,
    title: str = "🔥 Feature Importance — PRO+++",
    default_method: str = "auto",
    default_top_n: int = 30,
    default_n_repeats: int = 5,
    default_sample_size: int = 10000,
    default_scoring: Optional[str] = None,
) -> None:
    """
    Wpięcie: `from src/frontend.feature_importance import render_feature_importance; render_feature_importance()`
    Wymaga `st.session_state["raw_df"]` (DataFrame). Opcjonalnie: `st.session_state["trained_model"]`.
    """
    st.header(title)
    st.caption("Oblicz i zwizualizuj ważność cech (Native / Permutation / SHAP / Linear).")

    # Dane wejściowe
    if "raw_df" not in st.session_state or not isinstance(st.session_state["raw_df"], pd.DataFrame):
        st.error("Brak danych wejściowych. Najpierw wgraj i przygotuj dane (np. w Data Preview).")
        st.stop()
    df: pd.DataFrame = st.session_state["raw_df"]

    # Panel wyboru
    with st.expander("⚙️ Ustawienia FI", expanded=True):
        cols = df.columns.tolist()
        target = st.selectbox("Wybierz kolumnę docelową (target)", options=cols, index=len(cols) - 1 if cols else 0)
        exclude = st.multiselect(
            "Kolumny do wykluczenia (oprócz targetu)",
            options=[c for c in cols if c != target],
            default=[],
        )
        method = st.selectbox(
            "Metoda",
            options=["auto", "native", "permutation", "shap", "linear"],
            index=["auto", "native", "permutation", "shap", "linear"].index(default_method),
            help="auto → spróbuje kolejno: native → permutation → shap → linear.",
        )
        left, right, right2 = st.columns([1, 1, 1])
        with left:
            top_n = int(st.number_input("Top N", min_value=5, max_value=200, value=default_top_n, step=5))
            sample_size = int(
                st.number_input("Sample size", min_value=1000, max_value=200000, value=default_sample_size, step=1000)
            )
        with right:
            n_repeats = int(st.number_input("n_repeats (Permutation)", min_value=3, max_value=30, value=default_n_repeats))
            scoring = st.text_input("scoring (opcjonalnie)", value=default_scoring or "")
            scoring = scoring or None
        with right2:
            use_existing_model = st.toggle("Użyj istniejącego modelu z sesji (jeśli jest)", value=True)
            st.caption("Jeśli brak modelu — trenowany jest szybki baseline dla FI.")

    # Przycisk i obliczenia
    if st.button("🚀 Policz Feature Importance", use_container_width=True):
        try:
            fi_df, meta = _compute_fi_cached(
                df=df,
                target=target,
                method=method,
                scoring=scoring,
                top_n=top_n,
                n_repeats=n_repeats,
                sample_size=sample_size,
                random_state=42,
                columns_exclude=tuple(set(exclude) | {target}),
                use_existing_model=use_existing_model,
            )
        except Exception as e:
            log.exception("Błąd podczas obliczania FI.")
            st.error(f"Nie udało się obliczyć ważności cech: {e}")
            st.stop()

        # Wyniki → sesja
        st.session_state["feature_importance"] = {
            "table": fi_df.copy(),
            "meta": meta,
        }

        # Podsumowanie
        st.success(
            f"Gotowe: metoda = **{meta['method']}**, problem = **{meta['problem_type']}**, "
            f"n_rows = {meta['n_rows']}, n_features = {meta['n_features']}.",
            icon="✅",
        )

        # Tabela
        st.subheader("📋 Tabela Feature Importance")
        view_cols = [c for c in ["feature", "importance", "importance_norm", "std"] if c in fi_df.columns]
        st.dataframe(fi_df[view_cols], use_container_width=True, height=min(700, 24 * len(fi_df) + 100))

        # Wykres
        st.subheader("📈 Wykres (Top)")
        fig = _plot_bar(fi_df, title=f"Feature Importance (Top {len(fi_df)})")
        st.plotly_chart(fig, use_container_width=True)

        # Eksporty
        st.subheader("💾 Eksport")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Pobierz CSV",
                data=_export_csv(fi_df),
                file_name="feature_importance.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Pobierz JSON",
                data=_export_json(fi_df, meta),
                file_name="feature_importance.json",
                mime="application/json",
                use_container_width=True,
            )

# === NAZWA_SEKCJI === Lokalny punkt wejścia (opcjonalny)
if __name__ == "__main__":
    # Uruchom lokalnie: streamlit run src/frontend/feature_importance.py
    render_feature_importance()
