# === OPIS MODUŁU ===
"""
DataGenius PRO++++ - Model Explainer (KOSMOS)
Interpretowalność modeli: natywne importance, permutation (ze scorerem) oraz SHAP
(Tree/Linear/Auto). Pełne wsparcie dla Pipeline/ColumnTransformer (passthrough/OHE),
próbkowanie stratyfikowane, defensywne guardy i spójny kontrakt wyników.

Kontrakt (result.data):
{
    "feature_importance": pd.DataFrame | None,   # kolumny: feature, importance
    "shap_values": Dict[str, Any] | None,        # {"mean_abs_shap": {feat: val, ...}, "method": str, "n_samples": int}
    "top_features": List[str],                   # top-k cech wg FI lub SHAP
    "insights": List[str],                       # krótkie wnioski tekstowe
}
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Literal, Union

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult

# sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.utils import check_random_state
from scipy import sparse


# === KONFIG ===
@dataclass(frozen=True)
class ExplainerConfig:
    """Ustawienia działania ModelExplainer (PRO++++)."""
    top_n_features: int = 5
    permutation_repeats: int = 8
    permutation_n_jobs: int = -1
    shap_sample_size: int = 1000                 # ile próbek do liczenia SHAP (X_sample)
    background_sample_size: int = 200            # tło dla Kernel/Linear
    shap_method_preference: Tuple[str, ...] = ("tree", "linear", "auto")  # kolejność prób
    return_raw_shap_values: bool = False         # opcjonalny surowy zwrot (uwaga na rozmiar)
    random_state: int = 42
    # Scorer dla permutation; None => auto: 'roc_auc_ovr' (clf, multiclass), 'roc_auc' (bin), 'neg_mean_squared_error' (reg)
    permutation_scorer: Optional[str] = None
    # Max liczba surowych indeksów out-of-shape logów do debug
    debug_max_logs: int = 3


class ModelExplainer(BaseAgent):
    """
    Explains model predictions using various interpretability methods (FI + SHAP).
    Wspiera Pipelines/ColumnTransformer, OHE i passthrough.
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        super().__init__(
            name="ModelExplainer",
            description="Provides model interpretability (PRO++++)"
        )
        self.config = config or ExplainerConfig()
        self._rng = check_random_state(self.config.random_state)

    # === WYKONANIE GŁÓWNE ===
    def execute(
        self,
        best_model: Any,
        pycaret_wrapper: Any,            # utrzymanie kontraktu — niewykorzystywany
        data: pd.DataFrame,
        target_column: str,
        **kwargs: Any
    ) -> AgentResult:
        """
        Explain model.

        Args:
            best_model: Trained model (może być Pipeline)
            pycaret_wrapper: PyCaret wrapper (opcjonalny; nieużywany)
            data: DataFrame z cechami + target
            target_column: Nazwa targetu
        """
        result = AgentResult(agent_name=self.name)

        try:
            # === Walidacja wejścia
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            if best_model is None:
                raise ValueError("'best_model' is required")

            # === Przygotuj X, y
            X_raw, y = self._split_xy(data, target_column)

            # === Pipeline-aware transform i nazwy cech (po preprocesingu)
            tx = self._transform_if_pipeline(best_model, X_raw)
            X_for_model = tx.X  # już po preprocesingu (spójne z estymatorem)
            feature_names = tx.feature_names
            estimator = self._extract_estimator(best_model)

            # === Feature Importance (drzewa / liniowe / permutation)
            scorer_name = kwargs.get("permutation_scorer", self.config.permutation_scorer)
            # auto-scorer jeśli nie podano
            if scorer_name is None:
                scorer_name = self._infer_default_scorer(y)

            feature_importance = self._get_feature_importance(
                estimator, X_for_model, y, feature_names, scorer_name=scorer_name
            )

            # === SHAP (z samplowaniem stratyfikowanym) na datach po preprocesingu
            shap_summary = self._get_shap_explanations(
                estimator, X_for_model, feature_names, y=y
            )

            # === Top features + insights
            top_features = self._resolve_top_features(feature_importance, shap_summary, self.config.top_n_features)
            insights = self._generate_insights(feature_importance, shap_summary, top_features)

            # === Result
            result.data = {
                "feature_importance": feature_importance,
                "shap_values": shap_summary,
                "top_features": top_features,
                "insights": insights,
            }
            self.logger.success("Model explanation generated")

        except Exception as e:
            result.add_error(f"Model explanation failed: {e}")
            self.logger.error(f"Model explanation error: {e}", exc_info=True)

        return result

    # === UTIL: split X/y ===
    def _split_xy(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    # === PIPELINE HELPERS ===
    def _extract_estimator(self, model: Any) -> Any:
        """Dla Pipeline zwróć finalny estimator; w innym wypadku zwróć model."""
        try:
            if isinstance(model, Pipeline) and model.steps:
                return model.steps[-1][1]
        except Exception:
            pass
        return model

    def _extract_preprocessor(self, model: Any) -> Optional[Any]:
        """Zwraca preprocessor (np. ColumnTransformer) z Pipeline lub None."""
        try:
            if isinstance(model, Pipeline) and model.steps:
                # bierzemy wszystkie poza finalnym estymatorem i szukamy transformera
                for name, step in model.steps[:-1]:
                    if isinstance(step, ColumnTransformer) or hasattr(step, "transform"):
                        return step
        except Exception:
            pass
        return None

    def _transform_if_pipeline(self, model: Any, X: pd.DataFrame) -> Bunch:
        """
        Jeśli model to Pipeline/ma preprocessor, zwróć X_trans i nazwy cech po preprocesingu.
        W innym przypadku zwróć surowe X i X.columns.
        """
        pre = self._extract_preprocessor(model)
        if pre is None:
            return Bunch(X=X, feature_names=list(X.columns), meta=None)

        # transform lub fit_transform (gdy pre nie jest dopasowany)
        try:
            X_trans = pre.transform(X)
        except Exception:
            X_trans = pre.fit_transform(X)

        feature_names = self._safe_feature_names_out(pre, X_trans, X.columns)
        return Bunch(X=X_trans, feature_names=feature_names, meta={"preprocessor": pre})

    def _safe_feature_names_out(
        self,
        pre: Any,
        X_trans: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
        raw_cols: List[str]
    ) -> List[str]:
        """Pobierz nazwy cech po ColumnTransformer (passthrough/OHE)."""
        # 1) Proste API
        try:
            names = list(pre.get_feature_names_out())
            if names:
                return names
        except Exception:
            pass

        # 2) Ręczna rekonstrukcja z transformers_
        names: List[str] = []
        try:
            if hasattr(pre, "transformers_"):
                for name, trans, cols in pre.transformers_:
                    if name == "remainder" and trans == "drop":
                        continue
                    if trans == "passthrough":
                        # kolumny przepuszczone
                        names += [str(c) for c in (cols if isinstance(cols, list) else raw_cols)]
                    elif hasattr(trans, "get_feature_names_out"):
                        # wiele transformerów wspiera to API
                        try:
                            out = trans.get_feature_names_out(cols)
                        except Exception:
                            out = trans.get_feature_names_out()
                        names += [f"{name}__{f}" for f in out]
                    else:
                        # fallback: po prostu nazwy wejściowe z prefiksem transformera
                        base_cols = cols if isinstance(cols, list) else raw_cols
                        names += [f"{name}__{c}" for c in base_cols]
        except Exception as e:
            self.logger.warning(f"Manual feature name reconstruction failed: {e}")
            # awaryjnie: indeksy
            n = X_trans.shape[1] if hasattr(X_trans, "shape") else len(raw_cols)
            names = [f"f{i}" for i in range(n)]

        # 3) Jeżeli X_trans jest DataFrame z kolumnami — zaufaj kolumnom
        try:
            if isinstance(X_trans, pd.DataFrame) and len(X_trans.columns) == len(names):
                return [str(c) for c in X_trans.columns]
        except Exception:
            pass
        return names

    # === FEATURE IMPORTANCE ===
    def _get_feature_importance(
        self,
        estimator: Any,
        X: Union[pd.DataFrame, np.ndarray, sparse.spmatrix],
        y: pd.Series,
        feature_names: List[str],
        scorer_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Zwraca DataFrame: feature, importance. Kolejność: feature_importances_ → coef_ → permutation."""
        # 1) feature_importances_
        try:
            if hasattr(estimator, "feature_importances_"):
                imp = np.asarray(estimator.feature_importances_).ravel()
                imp = self._align_length(imp, len(feature_names))
                df_imp = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
                    "importance", ascending=False
                )
                return df_imp
        except Exception as e:
            self.logger.warning(f"Direct feature_importances_ failed: {e}")

        # 2) coef_
        try:
            if hasattr(estimator, "coef_"):
                coef = np.asarray(estimator.coef_)
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    coef = np.abs(coef)
                coef = self._align_length(coef, len(feature_names))
                df_imp = pd.DataFrame({"feature": feature_names, "importance": coef}).sort_values(
                    "importance", ascending=False
                )
                return df_imp
        except Exception as e:
            self.logger.warning(f"Linear coef_ importance failed: {e}")

        # 3) permutation importance — z właściwym scorerem (auto lub podany)
        try:
            scorer = get_scorer(scorer_name) if scorer_name else None
            res = permutation_importance(
                estimator, X, y,
                scoring=scorer,
                n_repeats=self.config.permutation_repeats,
                random_state=self.config.random_state,
                n_jobs=self.config.permutation_n_jobs
            )
            imp = self._align_length(res.importances_mean, len(feature_names))
            df_imp = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
                "importance", ascending=False
            )
            return df_imp
        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
            return None

    def _align_length(self, arr: np.ndarray, n: int) -> np.ndarray:
        """Dopasuj długość wektora importance do liczby cech."""
        arr = np.asarray(arr).ravel()
        if len(arr) == n:
            return arr
        if len(arr) > n:
            self.logger.warning(f"Importance vector longer than features ({len(arr)}>{n}) — truncating.")
            return arr[:n]
        # len(arr) < n
        self.logger.warning(f"Importance vector shorter than features ({len(arr)}<{n}) — padding zeros.")
        pad = np.zeros(n - len(arr))
        return np.concatenate([arr, pad])

    # === SHAP ===
    def _sample_for_shap(
        self,
        X: Union[pd.DataFrame, np.ndarray, sparse.spmatrix],
        y: Optional[pd.Series]
    ) -> Union[pd.DataFrame, np.ndarray, sparse.spmatrix]:
        """Próbkowanie do SHAP. Dla klasyfikacji — stratyfikacja na ile to możliwe (dla DataFrame)."""
        n = X.shape[0]
        size = min(self.config.shap_sample_size, n)
        if size >= n:
            return X

        # jeśli mamy DataFrame i y — spróbuj zbalansować
        if isinstance(X, pd.DataFrame) and y is not None:
            try:
                y_non_na = y.dropna()
                if y_non_na.nunique() > 1 and len(y_non_na) == len(y):
                    # prosty stratified-ish sampling: porówno z każdej klasy
                    classes = y.unique()
                    per_class = max(1, size // max(1, len(classes)))
                    idxs: List[int] = []
                    for cls in classes:
                        cls_idx = np.where(y.values == cls)[0]
                        take = min(len(cls_idx), per_class)
                        if take > 0:
                            idxs.extend(self._rng.choice(cls_idx, size=take, replace=False))
                    if len(idxs) > 0:
                        idxs = self._rng.permutation(list(set(idxs)))[:size]
                        return X.iloc[idxs]
            except Exception:
                pass

        # fallback: zwykłe losowanie
        if isinstance(X, pd.DataFrame):
            return X.sample(size, random_state=self.config.random_state)
        else:
            idx = self._rng.choice(n, size=size, replace=False)
            return X[idx]

    def _dense_if_needed(self, A: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """Konwersja do dense dla explainerów liniowych/auto gdy wymagane (ostrożnie z pamięcią)."""
        if sparse.issparse(A):
            return A.toarray()
        return np.asarray(A)

    def _get_shap_explanations(
        self,
        estimator: Any,
        X: Union[pd.DataFrame, np.ndarray, sparse.spmatrix],
        feature_names: List[str],
        y: Optional[pd.Series] = None
    ) -> Optional[Dict[str, Any]]:
        """Liczy SHAP i zwraca agregaty (mean |SHAP| per cecha) + metadata."""
        try:
            import shap  # noqa: F401
        except Exception:
            self.logger.warning("SHAP is not installed; skipping SHAP explanations.")
            return None

        # próbka + background (na X już po preprocesingu)
        X_sample = self._sample_for_shap(X, y)
        background = self._sample_for_shap(X, y)
        # SHAP lepiej działa na dense dla wielu estymatorów liniowych
        use_dense = False

        method_used = "auto"
        mean_abs_shap: Optional[np.ndarray] = None
        raw_payload: Optional[Any] = None

        for method in self.config.shap_method_preference:
            try:
                # wybór Explainer'a
                explainer, method_used = self._build_shap_explainer(
                    estimator, X_sample, background, preferred=method
                )
                sv = explainer(X_sample)  # unified API; może być ndarray-like
                mean_abs_shap = self._aggregate_mean_abs_shap(sv)
                raw_payload = sv if self.config.return_raw_shap_values else None
                if mean_abs_shap is not None:
                    break
            except Exception as e:
                # awaryjnie spróbuj dense
                if not use_dense and (sparse.issparse(X_sample) or sparse.issparse(background)):
                    try:
                        X_sample = self._dense_if_needed(X_sample)
                        background = self._dense_if_needed(background)
                        use_dense = True
                        self.logger.warning(f"Retry SHAP with dense matrices due to: {e}")
                        continue
                    except Exception as e2:
                        self.logger.warning(f"Dense conversion for SHAP failed: {e2}")
                self.logger.warning(f"SHAP {method} explainer failed: {e}")
                continue

        if mean_abs_shap is None:
            self.logger.warning("SHAP computation failed for all methods.")
            return None

        # Dopasuj długość i zbuduj wynik
        mean_abs_shap = self._align_length(mean_abs_shap, len(feature_names))
        shap_dict = {fname: float(val) for fname, val in zip(feature_names, mean_abs_shap)}

        out: Dict[str, Any] = {
            "mean_abs_shap": shap_dict,
            "method": method_used,
            "n_samples": int(X_sample.shape[0] if hasattr(X_sample, "shape") else len(X_sample)),
        }
        if self.config.return_raw_shap_values:
            # Uwaga: surowe wartości mogą być bardzo duże
            try:
                out["raw"] = getattr(raw_payload, "values", None)
            except Exception:
                out["raw"] = None
        return out

    def _build_shap_explainer(
        self,
        estimator: Any,
        X_sample: Union[pd.DataFrame, np.ndarray],
        background: Union[pd.DataFrame, np.ndarray],
        preferred: Literal["tree", "linear", "auto"]
    ):
        """Dobiera i tworzy SHAP Explainer (tree → linear → auto)."""
        import shap

        method_used = preferred
        if preferred == "tree":
            # TreeExplainer dla drzew / boostingów
            return shap.TreeExplainer(estimator, feature_perturbation="interventional"), "tree"
        if preferred == "linear":
            # LinearExplainer dla modeli liniowych; wymaga background
            bg = background
            return shap.LinearExplainer(estimator, bg, feature_perturbation="interventional"), "linear"

        # auto fallback (unified Explainer)
        masker = shap.maskers.Independent(background)
        return shap.Explainer(estimator, masker), "auto"

    def _aggregate_mean_abs_shap(self, shap_values_obj: Any) -> Optional[np.ndarray]:
        """Zwraca średnią bezwzględną SHAP per cecha (binarna/regresja: (n,m), multiclass: (k,n,m))."""
        try:
            import numpy as np  # noqa: F401
            sv = shap_values_obj
            vals = getattr(sv, "values", None)

            # sv.values bywa ndarray; w nowych wersjach czasem bezpośrednio sv to ndarray-like
            if vals is None and hasattr(sv, "__array__"):
                vals = np.array(sv)

            if vals is None:
                # czasem lista macierzy per klasa
                if isinstance(sv, list) and len(sv) > 0:
                    mats = []
                    for v in sv:
                        vv = getattr(v, "values", None)
                        vv = np.array(vv if vv is not None else v)
                        mats.append(np.abs(vv).mean(axis=0))
                    return np.mean(np.vstack(mats), axis=0)
                return None

            arr = np.asarray(vals)
            if arr.ndim == 2:
                # (n, m)
                return np.abs(arr).mean(axis=0)
            if arr.ndim == 3:
                # (k, n, m) multiclass → średnia po klasach i próbkach
                return np.abs(arr).mean(axis=(0, 1))
            if arr.ndim > 3:
                return np.abs(arr).reshape((-1, arr.shape[-1])).mean(axis=0)
        except Exception as e:
            self.logger.warning(f"SHAP mean-abs aggregation failed: {e}")
        return None

    # === TOP FEATURES ===
    def _resolve_top_features(
        self,
        feature_importance: Optional[pd.DataFrame],
        shap_summary: Optional[Dict[str, Any]],
        k: int
    ) -> List[str]:
        if feature_importance is not None and not feature_importance.empty:
            return feature_importance["feature"].head(k).tolist()
        if shap_summary and "mean_abs_shap" in shap_summary:
            sorted_feats = sorted(shap_summary["mean_abs_shap"].items(), key=lambda kv: kv[1], reverse=True)
            return [f for f, _ in sorted_feats[:k]]
        return []

    # === INSIGHTS ===
    def _generate_insights(
        self,
        feature_importance: Optional[pd.DataFrame],
        shap_summary: Optional[Dict[str, Any]],
        top_features: List[str]
    ) -> List[str]:
        insights: List[str] = []

        # FI-based
        if feature_importance is not None and not feature_importance.empty:
            top_feature = feature_importance.iloc[0]["feature"]
            top_importance = float(feature_importance.iloc[0]["importance"])
            insights.append(f"Najważniejsza cecha (FI): {top_feature} (ważność: {top_importance:.4f})")

            try:
                total = float(feature_importance["importance"].sum())
                if total > 0:
                    top3 = float(feature_importance.head(3)["importance"].sum()) / total * 100.0
                    if top3 > 70:
                        insights.append(
                            f"Top 3 cechy odpowiadają za {top3:.1f}% całkowitej ważności — model mocno polega na nielicznych cechach."
                        )
            except Exception:
                pass

        # SHAP-based
        if shap_summary and "mean_abs_shap" in shap_summary:
            method = shap_summary.get("method", "auto")
            insights.append(f"SHAP ({method}) wskazuje globalny wpływ cech (mean |SHAP|).")
            if (not feature_importance or feature_importance.empty) and top_features:
                insights.append(f"Top cechy (SHAP): {', '.join(top_features[:3])}")

        if not insights:
            insights.append("Brak wystarczających danych do wygenerowania wniosków o ważności cech.")

        return insights

    # === INNE POMOCNICZE ===
    def _infer_default_scorer(self, y: pd.Series) -> str:
        """Dobierz sensowny domyślny scorer dla permutation importance."""
        try:
            if pd.api.types.is_numeric_dtype(y):
                # regresja
                return "neg_mean_squared_error"
            # klasyfikacja: binarna vs multiclass
            n_uni = y.dropna().nunique()
            if n_uni <= 2:
                return "roc_auc"
            return "roc_auc_ovr"
        except Exception:
            return "neg_mean_squared_error"
