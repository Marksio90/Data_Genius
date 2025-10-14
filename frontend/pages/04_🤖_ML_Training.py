# 04_🤖_ML_Training.py
"""
DataGenius PRO — ML Training Page (PRO+++)
Trenuj i oceniaj modele w zunifikowanym Pipeline (ColumnTransformer → Model)
"""

from __future__ import annotations

import io
import json
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Bootstrapping ścieżek ===
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# === NAZWA_SEKCJI === Importy ekosystemu (UI + Core) ===
from frontend.app_layout import render_header, render_error, render_success, render_warning
from core.state_manager import get_state_manager

# Opcjonalny progress tracker
try:
    from src.frontend.progress_tracker import start_stage, advance, finish_stage, add_warning
    _HAS_PT = True
except Exception:
    _HAS_PT = False

# Opcjonalne biblioteki boostingowe
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

# === NAZWA_SEKCJI === Page Config (bezpiecznie) ===
try:
    st.set_page_config(page_title="🤖 ML Training — DataGenius PRO+++", page_icon="🤖", layout="wide")
except Exception:
    pass

# === NAZWA_SEKCJI === Utils: problem detect, preprocessor, metrics ===

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
        ("scaler", StandardScaler(with_mean=False)),
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

def _cls_metrics(y_true, y_pred, y_prob=None) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    try: out["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception: out["accuracy"] = None
    try: out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    except Exception: out["f1_weighted"] = None
    try:
        if y_prob is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            out["roc_auc"] = None
    except Exception:
        out["roc_auc"] = None
    return out

def _reg_metrics(y_true, y_pred) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    try: out["r2"] = float(r2_score(y_true, y_pred))
    except Exception: out["r2"] = None
    try: out["mae"] = float(mean_absolute_error(y_true, y_pred))
    except Exception: out["mae"] = None
    try: out["rmse"] = float(mean_squared_error(y_true, y_pred) ** 0.5)
    except Exception: out["rmse"] = None
    return out

# === NAZWA_SEKCJI === Katalog algorytmów i siatki HPO (lite) ===

def _algorithms(problem: str, use_xgb: bool, use_lgbm: bool) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    if problem == "classification":
        models["logreg"] = LogisticRegression(max_iter=500, n_jobs=None)
        models["rf_cls"] = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced_subsample")
        if use_xgb and _HAS_XGB:
            models["xgb_cls"] = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, objective="binary:logistic", tree_method="hist", n_jobs=-1
            )
        if use_lgbm and _HAS_LGBM:
            models["lgbm_cls"] = LGBMClassifier(
                n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, n_jobs=-1
            )
    else:
        models["linreg"] = LinearRegression()
        models["ridge"] = Ridge()
        models["rf_reg"] = RandomForestRegressor(n_estimators=400, max_depth=None, n_jobs=-1)
        if use_xgb and _HAS_XGB:
            models["xgb_reg"] = XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, tree_method="hist", n_jobs=-1
            )
        if use_lgbm and _HAS_LGBM:
            models["lgbm_reg"] = LGBMRegressor(
                n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, n_jobs=-1
            )
    return models

def _grid_for(key: str, problem: str) -> Dict[str, List[Any]]:
    # Lekka, bezpieczna siatka. Utrzymujemy w ryzach czas treningu.
    grids: Dict[str, Dict[str, List[Any]]] = {
        "logreg": {"model__C": [0.5, 1.0, 2.0]},
        "rf_cls": {"model__n_estimators": [200, 400], "model__max_depth": [None, 12]},
        "rf_reg": {"model__n_estimators": [300, 600], "model__max_depth": [None, 14]},
        "ridge": {"model__alpha": [0.5, 1.0, 2.0]},
        "xgb_cls": {"model__n_estimators": [300, 500], "model__max_depth": [4, 6]},
        "xgb_reg": {"model__n_estimators": [400, 700], "model__max_depth": [4, 6]},
        "lgbm_cls": {"model__n_estimators": [400, 700], "model__num_leaves": [31, 63]},
        "lgbm_reg": {"model__n_estimators": [500, 800], "model__num_leaves": [31, 63]},
    }
    return grids.get(key, {})

# === NAZWA_SEKCJI === Eksporty ===

def _export_json(report: Dict[str, Any]) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")

def _export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _export_model_bytes(pipeline: Any) -> bytes:
    import pickle
    return pickle.dumps(pipeline)

# === NAZWA_SEKCJI === Nawigacja (kompatybilna) ===

def _goto(file_hint: str, label: str = "") -> None:
    try:
        st.page_link(file_hint, label=label or file_hint)
        return
    except Exception:
        pass
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(file_hint)
            return
        except Exception:
            pass
    st.info(f"Przejdź ręcznie do: **{file_hint}** (menu po lewej).")

# === NAZWA_SEKCJI === UI GŁÓWNE ===

def main() -> None:
    render_header("🤖 Trening Modeli", "Trenuj i porównuj modele w zunifikowanym pipeline")

    state = get_state_manager()
    if not state.has_data():
        render_warning("Najpierw załaduj dane w sekcji **📊 Data Upload**.")
        if st.button("➡️ Przejdź do Upload", use_container_width=True):
            _goto("pages/02_📊_Data_Upload.py", label="📊 Data Upload")
        return

    df: pd.DataFrame = state.get_data()
    cols = df.columns.tolist()

    # Target
    target_auto = state.get_target_column()
    target = st.selectbox("🎯 Wybierz kolumnę docelową (target)", options=cols, index=cols.index(target_auto) if target_auto in cols else len(cols)-1)

    # Ustawienia cech i sampling
    with st.expander("⚙️ Zakres danych i losowanie", expanded=True):
        selected_features = st.multiselect("Wybierz cechy (puste = wszystkie oprócz targetu)", options=[c for c in cols if c != target], default=[])
        sample_max = st.number_input("Maks. liczba wierszy (sample)", min_value=1_000, max_value=max(10_000, len(df)), value=min(len(df), 200_000), step=1_000)
        random_state = st.number_input("Random state", min_value=0, max_value=999_999, value=42, step=1)

    # Dobór problemu i algorytmów
    y = df[target]
    problem_guess = _detect_problem_type(y)
    problem = st.selectbox("🧠 Typ problemu", options=["classification", "regression"], index=["classification","regression"].index(problem_guess))

    with st.expander("🧪 Ustawienia trenowania", expanded=True):
        test_size = st.slider("Udział zb. testowego", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        val_size = st.slider("Udział walidacyjny (z TRAIN)", min_value=0.0, max_value=0.3, value=0.1, step=0.05, help="Wydzielimy walidację z części treningowej.")
        use_xgb = st.toggle("Dołącz XGBoost (jeśli dostępny)", value=True)
        use_lgbm = st.toggle("Dołącz LightGBM (jeśli dostępny)", value=True)
        do_hpo = st.toggle("Grid Search (lite)", value=True, help="Krótkie HPO dla wybranych modeli.")
        primary_metric = st.selectbox("Główna metryka", options=(["roc_auc","accuracy","f1_weighted"] if problem=="classification" else ["r2","neg_rmse","neg_mae"]), index=0)

    algo_catalog = _algorithms(problem, use_xgb, use_lgbm)
    chosen_keys = st.multiselect("📦 Wybierz algorytmy do trenowania", options=list(algo_catalog.keys()), default=list(algo_catalog.keys())[:3])

    # Start treningu
    if st.button("🚀 Trenuj modele", type="primary", use_container_width=True):
        try:
            if _HAS_PT:
                start_stage("ML_Training", total_steps=4)
                advance(note="Przygotowanie danych")

            # Subset + sampling
            work_df = df.copy()
            keep = selected_features[:] if selected_features else [c for c in cols if c != target]
            if target not in keep:
                keep = keep + [target]
            work_df = work_df[keep]

            if len(work_df) > sample_max:
                work_df = work_df.sample(sample_max, random_state=int(random_state))

            X, y = _split_xy(work_df, target)

            # Split train/test (+ val z train)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=int(random_state),
                stratify=y if problem == "classification" else None
            )
            if val_size > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=float(val_size), random_state=int(random_state),
                    stratify=y_train if problem == "classification" else None
                )
            else:
                X_val = X_train
                y_val = y_train

            pre = _preprocessor_for(X_train)

            if _HAS_PT:
                advance(note="Trening modeli")

            rows: List[Dict[str, Any]] = []
            best_key: Optional[str] = None
            best_score: float = -1e18 if problem=="regression" else -1e18  # będziemy interpretować metryki znakami
            best_pipe: Optional[Pipeline] = None
            details: Dict[str, Any] = {}

            for key in chosen_keys:
                if key not in algo_catalog:
                    continue
                model = algo_catalog[key]
                pipe = Pipeline(steps=[("pre", pre), ("model", model)])

                # GridSearch (lite)
                if do_hpo:
                    grid = _grid_for(key, problem)
                    if grid:
                        gs = GridSearchCV(pipe, param_grid=grid, scoring=_to_sklearn_scoring(primary_metric), cv=3, n_jobs=-1, refit=True)
                        gs.fit(X_train, y_train)
                        pipe = gs.best_estimator_
                        details[key] = {"best_params": gs.best_params_, "cv_best_score": float(gs.best_score_)}
                    else:
                        pipe.fit(X_train, y_train)
                        details[key] = {"best_params": None, "cv_best_score": None}
                else:
                    pipe.fit(X_train, y_train)
                    details[key] = {"best_params": None, "cv_best_score": None}

                # Ewaluacja na walidacji
                y_pred_val = pipe.predict(X_val)
                if problem == "classification":
                    y_prob_val = None
                    try:
                        y_prob_val = pipe.predict_proba(X_val)[:, 1]
                    except Exception:
                        pass
                    m = _cls_metrics(y_val, y_pred_val, y_prob_val)
                    score = _primary_score_from_metrics(m, primary_metric)
                else:
                    m = _reg_metrics(y_val, y_pred_val)
                    score = _primary_score_from_metrics(m, primary_metric)

                rows.append({"model": key, **m})

                # aktualizacja najlepszego
                if score is not None and (best_key is None or score > best_score):
                    best_score = float(score)
                    best_key = key
                    best_pipe = pipe

            if not rows or best_pipe is None or best_key is None:
                raise RuntimeError("Żaden model nie został poprawnie wytrenowany.")

            if _HAS_PT:
                advance(note=f"Najlepszy model: {best_key}")

            # Finalna ocena na TEST
            y_pred_test = best_pipe.predict(X_test)
            if problem == "classification":
                y_prob_test = None
                try:
                    y_prob_test = best_pipe.predict_proba(X_test)[:, 1]
                except Exception:
                    pass
                test_metrics = _cls_metrics(y_test, y_pred_test, y_prob_test)
            else:
                test_metrics = _reg_metrics(y_test, y_pred_test)

            # Raport i zapis do stanu
            res_df = pd.DataFrame(rows).sort_values(by=_primary_col_name(primary_metric), ascending=False, na_position="last").reset_index(drop=True)
            report = {
                "problem": problem,
                "target": target,
                "primary_metric": primary_metric,
                "validation_results": rows,
                "best_key": best_key,
                "test_metrics": test_metrics,
                "grid_search": details,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
            }

            # Persist
            state.set_target_column(target)
            state.set_problem_type(problem)
            state.set_trained_model(best_pipe)
            st.session_state["trained_model"] = best_pipe
            st.session_state["ml_training"] = {"table": res_df.copy(), "report": report}

            if _HAS_PT:
                advance(note="Zapis wyników")
                finish_stage()

            # UI: podsumowanie
            st.success(f"✅ Najlepszy model: **{best_key}** (primary: **{primary_metric}**)")

            st.subheader("📋 Walidacja (ranking)")
            st.dataframe(res_df, use_container_width=True, height=min(600, 26 * (len(res_df) + 3)))

            # Wykres rankingowy
            st.subheader("📈 Ranking modeli")
            col_name = _primary_col_name(primary_metric)
            fig = px.bar(res_df.sort_values(col_name, ascending=True), x=col_name, y="model", orientation="h", text=np.round(res_df[col_name], 4))
            fig.update_layout(height=max(420, 50 * len(res_df)), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # TEST – diagnostyka
            st.subheader("🩺 Diagnostyka na zbiorze testowym")
            _render_test_diagnostics(problem, y_test, y_pred_test)

            # Eksporty
            st.subheader("💾 Eksport")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("Pobierz ranking (CSV)", data=_export_csv(res_df), file_name="ml_training_ranking.csv", mime="text/csv", use_container_width=True)
            with c2:
                st.download_button("Pobierz raport (JSON)", data=_export_json(report), file_name="ml_training_report.json", mime="application/json", use_container_width=True)
            with c3:
                st.download_button("Pobierz model (pickle)", data=_export_model_bytes(best_pipe), file_name=f"best_model_{best_key}.pkl", mime="application/octet-stream", use_container_width=True)

        except Exception as e:
            if _HAS_PT:
                try:
                    add_warning(str(e))
                    finish_stage(status="failed")
                except Exception:
                    pass
            render_error("Błąd treningu", str(e))


# === NAZWA_SEKCJI === Metryki i diagnostyka pomocnicza ===

def _to_sklearn_scoring(metric: str) -> str:
    # mapujemy nazwy UI → scoring sklearn / nasze interpretacje
    if metric == "neg_rmse":
        return "neg_root_mean_squared_error"
    if metric == "neg_mae":
        return "neg_mean_absolute_error"
    # reszta ma nazwy zgodne lub standardowe
    return metric

def _primary_col_name(metric: str) -> str:
    # kolumna w tabeli wyników odpowiadająca metryce
    mapping = {
        "roc_auc": "roc_auc", "accuracy": "accuracy", "f1_weighted": "f1_weighted",
        "r2": "r2", "neg_rmse": "rmse", "neg_mae": "mae"
    }
    return mapping.get(metric, metric)

def _primary_score_from_metrics(m: Dict[str, Optional[float]], metric: str) -> Optional[float]:
    # interpretacja znaku metryk „neg_*”
    if metric == "neg_rmse":
        return -m.get("rmse") if m.get("rmse") is not None else None
    if metric == "neg_mae":
        return -m.get("mae") if m.get("mae") is not None else None
    return m.get(metric)

def _render_test_diagnostics(problem: str, y_true, y_pred) -> None:
    if problem == "classification":
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            import plotly.figure_factory as ff
            labels = [str(c) for c in sorted(pd.Series(y_true).dropna().unique())]
            figcm = ff.create_annotated_heatmap(z=cm, x=[f"pred_{l}" for l in labels], y=[f"true_{l}" for l in labels], colorscale="Blues", showscale=True)
            figcm.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(figcm, use_container_width=True)
        except Exception:
            st.info("Nie udało się narysować macierzy pomyłek.")
    else:
        # Histogram residuals
        try:
            resid = (pd.Series(y_true).reset_index(drop=True) - pd.Series(y_pred).reset_index(drop=True)).values
            fig = px.histogram(resid, nbins=40, title="Residuals (test)")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Nie udało się narysować rozkładu residuali.")

# === NAZWA_SEKCJI === Wejście modułu ===
if __name__ == "__main__":
    main()
