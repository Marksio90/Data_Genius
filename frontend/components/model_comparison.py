# src/frontend/model_comparison.py
# === OPIS MODUŁU ===
# Porównanie modeli (klasyfikacja/regresja) PRO+++:
# - Pipeline: ColumnTransformer (imputacja + OneHot) → Model
# - Modele: Linear/Ridge, RandomForest, (opcjonalnie) XGBoost/LightGBM
# - CV (KFold/StratifiedKFold), metryki, czasy, ranking
# - Retrain best on full data + zapis do session_state
# - Wizualizacja wyników (Plotly) + diagnostyka (CM/residuals)
# - Eksport CSV/JSON

from __future__ import annotations

import io
import json
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Logger (zgodny z Twoim ekosystemem) ===
try:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("model_comparison")

# === NAZWA_SEKCJI === Próby importu XGBoost/LightGBM (opcjonalne) ===
_HAS_XGB = False
_HAS_LGBM = False
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
    _HAS_LGBM = True
except Exception:
    pass

# === NAZWA_SEKCJI === Dataclasses konfiguracji i wyników ===

@dataclass
class CompareConfig:
    cv_folds: int = 5
    random_state: int = 42
    sample_size: int = 100_000
    scoring_cls: str = "roc_auc"
    scoring_reg: str = "r2"
    include_xgb: bool = True
    include_lgbm: bool = True
    test_size: float = 0.2  # na diagnostykę (CM/residuals)

@dataclass
class ModelResult:
    key: str
    mean_score: float
    std_score: float
    fit_time: float
    score_time: float
    problem_type: str
    scoring: str
    n_features_after_ohe: int
    params: Dict[str, Any]

# === NAZWA_SEKCJI === Pomocnicze: detekcja problemu, przygotowanie danych ===

def _detect_problem_type(y: pd.Series) -> str:
    y_non_null = y.dropna()
    nunique = y_non_null.nunique()
    if pd.api.types.is_bool_dtype(y_non_null):
        return "classification"
    if pd.api.types.is_integer_dtype(y_non_null) and nunique <= min(20, max(2, int(len(y_non_null) ** 0.5))):
        return "classification"
    if pd.api.types.is_object_dtype(y_non_null) and nunique <= 50:
        return "classification"
    return "regression"

def _split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' nie istnieje w danych.")
    y = df[target]
    X = df.drop(columns=[target])
    if X.empty:
        raise ValueError("Brak cech po odjęciu targetu.")
    return X, y

def _preprocessor_for(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # safe dla sparse
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre

def _get_cv(problem_type: str, n_splits: int, rs: int):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    return KFold(n_splits=n_splits, shuffle=True, random_state=rs)

# === NAZWA_SEKCJI === Zestaw modeli kandydatów ===

def _candidate_models(problem_type: str, rs: int, include_xgb: bool, include_lgbm: bool) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    if problem_type == "classification":
        models["logreg"] = LogisticRegression(
            max_iter=500, n_jobs=None, random_state=rs
        )
        models["rf_cls"] = RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=rs, class_weight="balanced_subsample"
        )
        if include_xgb and _HAS_XGB:
            models["xgb_cls"] = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, objective="binary:logistic", tree_method="hist", random_state=rs, n_jobs=-1
            )
        if include_lgbm and _HAS_LGBM:
            models["lgbm_cls"] = LGBMClassifier(
                n_estimators=500, max_depth=-1, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=rs, n_jobs=-1
            )
    else:
        models["linreg"] = LinearRegression()
        models["ridge"] = Ridge(random_state=rs)
        models["rf_reg"] = RandomForestRegressor(
            n_estimators=500, max_depth=None, n_jobs=-1, random_state=rs
        )
        if include_xgb and _HAS_XGB:
            models["xgb_reg"] = XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, tree_method="hist", random_state=rs, n_jobs=-1
            )
        if include_lgbm and _HAS_LGBM:
            models["lgbm_reg"] = LGBMRegressor(
                n_estimators=600, max_depth=-1, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=rs, n_jobs=-1
            )
    return models

# === NAZWA_SEKCJI === Cache: porównanie modeli ===

@st.cache_data(show_spinner=True, ttl=3600, max_entries=6)
def _compare_models_cached(
    df: pd.DataFrame,
    target: str,
    cfg: CompareConfig,
    drop_cols: Tuple[str, ...],
    scoring_override: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Subsample dla stabilności UI
    X, y = _split_xy(df, target)
    if len(X) > cfg.sample_size:
        X = X.sample(cfg.sample_size, random_state=cfg.random_state)
        y = y.loc[X.index]

    problem = _detect_problem_type(y)
    pre = _preprocessor_for(X)

    models = _candidate_models(
        problem_type=problem,
        rs=cfg.random_state,
        include_xgb=(cfg.include_xgb and _HAS_XGB),
        include_lgbm=(cfg.include_lgbm and _HAS_LGBM),
    )

    # Pipeline + CV
    cv = _get_cv(problem, cfg.cv_folds, cfg.random_state)
    scoring = scoring_override or (cfg.scoring_cls if problem == "classification" else cfg.scoring_reg)

    rows: List[ModelResult] = []
    raw_scores: Dict[str, List[float]] = {}
    n_feat_after: Dict[str, int] = {}

    for key, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        try:
            t0 = time.time()
            cvres = cross_validate(
                pipe, X, y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False,
                return_estimator=False,
            )
            t1 = time.time()
            mean_score = float(np.mean(cvres["test_score"]))
            std_score = float(np.std(cvres["test_score"]))
            fit_time = float(np.mean(cvres["fit_time"]))
            score_time = float(np.mean(cvres["score_time"]))
            rows.append(ModelResult(
                key=key,
                mean_score=mean_score,
                std_score=std_score,
                fit_time=fit_time,
                score_time=score_time,
                problem_type=problem,
                scoring=scoring,
                n_features_after_ohe=-1,  # uzupełnimy za chwilę
                params=getattr(model, "get_params", lambda: {})(),
            ))
            raw_scores[key] = list(map(float, cvres["test_score"]))
            # Fit na fragmencie aby policzyć liczbę cech po OHE (bez kosztu CV)
            try:
                pipe.fit(X.head(min(2000, len(X))), y.head(min(2000, len(y))))
                try:
                    ohe = pipe.named_steps["pre"].transformers_[1][1].named_steps["onehot"]  # type: ignore
                    n_ohe = int(len(ohe.get_feature_names_out()))
                except Exception:
                    n_ohe = 0
                num_cols = int(pipe.named_steps["pre"].transformers_[0][2].__len__()) if pipe.named_steps["pre"].transformers_[0][2] else 0  # type: ignore
                n_feat_after[key] = int(n_ohe + num_cols)
            except Exception:
                n_feat_after[key] = -1
            log.info(f"CV done: {key} | score={mean_score:.4f}±{std_score:.4f} | time={t1-t0:.2f}s")
        except Exception as e:
            log.warning(f"Model {key} pominiety (błąd): {e}")
            continue

    if not rows:
        raise RuntimeError("Żaden model nie został poprawnie oceniony w CV.")

    df_rows = []
    for r in rows:
        r.n_features_after_ohe = n_feat_after.get(r.key, -1)
        df_rows.append(asdict(r))
    res_df = pd.DataFrame(df_rows).sort_values("mean_score", ascending=False).reset_index(drop=True)

    # Best model: retrain na pełnym X,y
    best_key = res_df.iloc[0]["key"]
    best_model = _candidate_models(problem, cfg.random_state, cfg.include_xgb and _HAS_XGB, cfg.include_lgbm and _HAS_LGBM)[best_key]
    best_pipe = Pipeline(steps=[("pre", pre), ("model", best_model)])
    best_pipe.fit(X, y)

    # Diagnostyka na holdoucie
    diag: Dict[str, Any] = {"problem": problem, "scoring": scoring}
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if problem == "classification" else None
        )
        best_pipe.fit(X_tr, y_tr)
        y_pred = best_pipe.predict(X_te)
        if problem == "classification":
            # progi: gdy predict_proba dostępne → roc_auc
            try:
                y_prob = best_pipe.predict_proba(X_te)[:, 1]
                diag["roc_auc_holdout"] = float(roc_auc_score(y_te, y_prob))
            except Exception:
                diag["roc_auc_holdout"] = None
            diag["acc_holdout"] = float(accuracy_score(y_te, y_pred))
            try:
                diag["f1_holdout"] = float(f1_score(y_te, y_pred, average="weighted"))
            except Exception:
                diag["f1_holdout"] = None
            cm = confusion_matrix(y_te, y_pred)
            diag["confusion_matrix"] = cm.tolist()
        else:
            diag["r2_holdout"] = float(r2_score(y_te, y_pred))
            diag["mae_holdout"] = float(mean_absolute_error(y_te, y_pred))
            diag["rmse_holdout"] = float(mean_squared_error(y_te, y_pred) ** 0.5)
            diag["residuals_sample"] = (y_te - y_pred).tolist()[:500]
    except Exception as e:
        log.warning(f"Diagnostyka holdout nie powiodła się: {e}")

    meta = {
        "best_key": best_key,
        "problem": problem,
        "scoring": scoring,
        "cv_folds": cfg.cv_folds,
        "random_state": cfg.random_state,
        "sample_size": min(len(df), cfg.sample_size),
        "raw_scores": raw_scores,
        "diag": diag,
        "has_xgb": _HAS_XGB,
        "has_lgbm": _HAS_LGBM,
    }

    return res_df, {"meta": meta, "best_model": best_pipe}

# === NAZWA_SEKCJI === Eksporty ===

def _export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _export_json(df: pd.DataFrame, meta: Dict[str, Any]) -> bytes:
    payload = {"meta": meta, "results": df.to_dict(orient="records")}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

# === NAZWA_SEKCJI === Wykres rankingowy ===

def _plot_ranking(df: pd.DataFrame, title: str) -> "plotly.graph_objs._figure.Figure":
    plot_df = df.copy()
    plot_df["label"] = plot_df["key"] + " (" + plot_df["scoring"] + ")"
    fig = px.bar(
        plot_df.sort_values("mean_score", ascending=True),
        x="mean_score",
        y="key",
        error_x="std_score",
        orientation="h",
        title=title,
        text=np.round(plot_df["mean_score"], 4),
        labels={"mean_score": "CV mean score", "key": "model"},
    )
    fig.update_layout(
        height=max(420, 60 * len(plot_df)),
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.2,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig

# === NAZWA_SEKCJI === UI GŁÓWNE ===

def render_model_comparison(
    *,
    title: str = "🏁 Model Comparison — PRO+++",
) -> None:
    """
    Wpięcie: `from src.frontend.model_comparison import render_model_comparison; render_model_comparison()`
    Wymaga `st.session_state["raw_df"]` (DataFrame).
    Po wykonaniu zapisuje:
      - st.session_state["trained_model"]  -> najlepszy Pipeline (pre + model)
      - st.session_state["model_comparison"] -> {"table": DataFrame, "meta": dict}
    """
    st.header(title)
    st.caption("Porównanie modeli w zunifikowanym pipeline z CV i wyborem najlepszego modelu.")

    # Dane wejściowe
    if "raw_df" not in st.session_state or not isinstance(st.session_state["raw_df"], pd.DataFrame):
        st.error("Brak danych wejściowych. Najpierw wgraj dane w Data Preview.")
        st.stop()

    df: pd.DataFrame = st.session_state["raw_df"]
    cols = df.columns.tolist()
    if not cols:
        st.error("Zbiór danych nie posiada kolumn.")
        st.stop()

    # Panel ustawień
    with st.expander("⚙️ Ustawienia porównania", expanded=True):
        target = st.selectbox("Target", options=cols, index=len(cols) - 1 if cols else 0)
        exclude = st.multiselect("Kolumny do wykluczenia (oprócz targetu)", options=[c for c in cols if c != target], default=[])
        left, mid, right = st.columns(3)
        with left:
            cv_folds = int(st.number_input("CV folds", min_value=3, max_value=15, value=5))
            sample_size = int(st.number_input("Sample size (max)", min_value=1000, max_value=300000, value=100000, step=1000))
        with mid:
            test_size = float(st.slider("Holdout test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05))
            rs = int(st.number_input("Random state", min_value=0, max_value=99999, value=42))
        with right:
            include_xgb = st.toggle("XGBoost (jeśli dostępny)", value=True)
            include_lgbm = st.toggle("LightGBM (jeśli dostępny)", value=True)

        # Metryka/wynik
        problem_guess = _detect_problem_type(df[target])
        scoring_default = "roc_auc" if problem_guess == "classification" else "r2"
        scoring = st.text_input("Scoring (sklearn)", value=scoring_default, help="Np. acc/roc_auc dla klasyfikacji; r2/neg_mean_squared_error dla regresji.")

    if st.button("🚀 Uruchom porównanie modeli", use_container_width=True):
        cfg = CompareConfig(
            cv_folds=cv_folds,
            random_state=rs,
            sample_size=sample_size,
            include_xgb=include_xgb,
            include_lgbm=include_lgbm,
            test_size=test_size,
        )
        try:
            res_df, bundle = _compare_models_cached(
                df=df.drop(columns=[c for c in exclude if c in df.columns]),
                target=target,
                cfg=cfg,
                drop_cols=tuple(exclude),
                scoring_override=scoring,
            )
        except Exception as e:
            log.exception("Błąd porównania modeli.")
            st.error(f"Nie udało się wykonać porównania modeli: {e}")
            st.stop()

        # Zapis wyników do sesji
        st.session_state["trained_model"] = bundle["best_model"]
        st.session_state["model_comparison"] = {"table": res_df.copy(), "meta": bundle["meta"]}

        # Raport
        meta = bundle["meta"]
        best_key = meta["best_key"]
        st.success(
            f"Gotowe! Najlepszy model: **{best_key}** (scoring: **{meta['scoring']}**). "
            f"Problem: **{meta['problem']}**, CV folds: {meta['cv_folds']}.",
            icon="✅",
        )

        st.subheader("📋 Wyniki CV (ranking)")
        st.dataframe(
            res_df[["key", "mean_score", "std_score", "fit_time", "score_time", "n_features_after_ohe", "scoring"]],
            use_container_width=True,
            height=min(600, 26 * len(res_df) + 110),
        )

        st.subheader("📈 Wykres rankingowy")
        fig = _plot_ranking(res_df, title=f"Model ranking ({meta['scoring']})")
        st.plotly_chart(fig, use_container_width=True)

        # Diagnostyka
        st.subheader("🩺 Diagnostyka holdout")
        diag = meta.get("diag", {})
        if meta["problem"] == "classification":
            c1, c2, c3 = st.columns(3)
            c1.metric("ROC AUC (holdout)", f"{diag.get('roc_auc_holdout'):.4f}" if diag.get("roc_auc_holdout") is not None else "n/d")
            c2.metric("Accuracy (holdout)", f"{diag.get('acc_holdout'):.4f}" if diag.get("acc_holdout") is not None else "n/d")
            c3.metric("F1 (holdout)", f"{diag.get('f1_holdout'):.4f}" if diag.get("f1_holdout") is not None else "n/d")
            cm = diag.get("confusion_matrix")
            if cm:
                import plotly.figure_factory as ff
                z = np.array(cm)
                x = [f"pred_{i}" for i in range(z.shape[0])]
                y = [f"true_{i}" for i in range(z.shape[0])]
                figcm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale="Blues", showscale=True)
                figcm.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(figcm, use_container_width=True)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("R² (holdout)", f"{diag.get('r2_holdout'):.4f}" if diag.get("r2_holdout") is not None else "n/d")
            c2.metric("MAE (holdout)", f"{diag.get('mae_holdout'):.4f}" if diag.get("mae_holdout") is not None else "n/d")
            c3.metric("RMSE (holdout)", f"{diag.get('rmse_holdout'):.4f}" if diag.get("rmse_holdout") is not None else "n/d")
            # prosty wykres residuali
            res = diag.get("residuals_sample")
            if res:
                import plotly.graph_objects as go
                figres = go.Figure()
                figres.add_trace(go.Histogram(x=res, nbinsx=40))
                figres.update_layout(title="Residuals (sample)", height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(figres, use_container_width=True)

        # Eksporty
        st.subheader("💾 Eksport wyników")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Pobierz CSV (ranking)",
                data=_export_csv(res_df),
                file_name="model_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Pobierz JSON (pełny raport)",
                data=_export_json(res_df, meta),
                file_name="model_comparison.json",
                mime="application/json",
                use_container_width=True,
            )

# === NAZWA_SEKCJI === Lokalny punkt wejścia (opcjonalny)
if __name__ == "__main__":
    # Uruchom: streamlit run src/frontend/model_comparison.py
    render_model_comparison()
