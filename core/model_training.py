# model_training.py
# ==============================================
#  Moduł: Automatyczny trening modelu
#            (regresja/klasyfikacja) + ważność cech
# ==============================================

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

# ---------- Przygotowanie danych ----------

def przygotuj_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    # Usuń wiersze z brakami danych w zmiennej docelowej
    df_clean = df.dropna(subset=[target])
    
    y = df_clean[target]
    X = df_clean.drop(columns=[target])
    kat = [c for c in X.columns if (not pd.api.types.is_numeric_dtype(X[c]))]
    num = [c for c in X.columns if c not in kat]
    return X, y, num, kat

def zbuduj_transformer(num: List[str], kat: List[str]) -> ColumnTransformer:
    # Pipeline dla kolumn numerycznych: imputacja -> skalowanie
    num_tr = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # Uzupełnij braki medianą
        ("scaler", StandardScaler(with_mean=False))
    ])
    
    # Pipeline dla kolumn kategorycznych: imputacja -> one-hot encoding
    kat_tr = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Uzupełnij braki najczęstszą wartością
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num", num_tr, num),
            ("kat", kat_tr, kat)
        ],
        remainder="drop"
    )

# ---------- Trening modeli ----------

def trenuj_regresje(X, y, transformer, random_state=42, test_size=0.2) -> Tuple[Pipeline, Dict[str, float]]:
    modele = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=random_state)
    }
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    best_pipe = None
    best_score = -np.inf
    best_metrics: Dict[str, float] = {}
    for name, model in modele.items():
        pipe = Pipeline([("prep", transformer), ("model", model)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        r2 = r2_score(y_te, pred)
        mae = mean_absolute_error(y_te, pred)
        rmse = root_mean_squared_error(y_te, pred)
        
        if r2 > best_score:
            best_score = r2
            best_pipe = pipe
            best_metrics = {"model": name, "R2": r2, "MAE": mae, "RMSE": rmse}
    
    return best_pipe, best_metrics


def trenuj_klasyfikacje(X, y, transformer, random_state=42, test_size=0.2) -> Tuple[Pipeline, Dict[str, float]]:
    y = y.astype("category")
    modele = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced"
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=random_state)
    }
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    best_pipe = None
    best_score = -np.inf
    best_metrics: Dict[str, float] = {}
    for name, model in modele.items():
        pipe = Pipeline([("prep", transformer), ("model", model)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, pred)
        bal_acc = balanced_accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        if bal_acc > best_score:
            best_score = bal_acc
            best_pipe = pipe
            best_metrics = {"model": name, "balanced_accuracy": bal_acc, "accuracy": acc, "f1_macro": f1}
    return best_pipe, best_metrics

# ---------- Ważność cech ----------

def policz_waznosci(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, random_state=42, n_repeats=5) -> pd.DataFrame:
    """
    Ważność permutacyjna działa niezależnie od typu modelu i po przetwarzaniu cech.
    """
    if pipe is None:
        # Jeśli nie ma pipeline, zwróć pusty DataFrame
        return pd.DataFrame(columns=["cecha", "waznosc_srednia", "waznosc_std"])
    
    prep = pipe.named_steps["prep"]
    pipe.fit(X, y)

    # Nazwy cech po transformacji
    num_names = prep.transformers_[0][2]
    kat_enc = prep.transformers_[1][1]
    kat_names = prep.transformers_[1][2]
    try:
        kat_ohe_names = list(kat_enc.get_feature_names_out(kat_names))
    except Exception:
        kat_ohe_names = [f"{kat_names[i]}_{j}" for i in range(len(kat_names)) for j in range(1)]

    feature_names = list(num_names) + list(kat_ohe_names)

    result = permutation_importance(
        pipe, X, y, n_repeats=n_repeats, random_state=random_state, scoring=None
    )

    importances = pd.DataFrame({
        "cecha": feature_names[:len(result.importances_mean)],
        "waznosc_srednia": result.importances_mean,
        "waznosc_std": result.importances_std
    }).sort_values("waznosc_srednia", ascending=False).reset_index(drop=True)

    return importances


# ---------- Funkcja główna ----------

def trenowanie_automatyczne(df: pd.DataFrame, target: str, typ: str, random_state=42, test_size=0.2, permutation_repeats=5) -> Dict[str, Any]:
    X, y, num, kat = przygotuj_X_y(df, target)
    transformer = zbuduj_transformer(num, kat)

    if typ == "regresja":
        pipe, metrics = trenuj_regresje(X, y, transformer, random_state=random_state, test_size=test_size)
    else:
        pipe, metrics = trenuj_klasyfikacje(X, y, transformer, random_state=random_state, test_size=test_size)

    waznosci = policz_waznosci(pipe, X, y, random_state=random_state, n_repeats=permutation_repeats)
    return {
        "pipeline": pipe,
        "metrics": metrics,
        "waznosci": waznosci,
        "num_features": num,
        "cat_features": kat
    }