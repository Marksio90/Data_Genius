# === OPIS MODUŁU ===
"""
DataGenius PRO - Model Explainer (PRO+++)
Interpretowalność modeli: natywne importance, permutation fallback oraz SHAP
(Tree/Linear/Kernel). Wsparcie dla Pipeline/ColumnTransformer, sampling, defensywa.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Literal

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult


# === KONFIG ===
@dataclass(frozen=True)
class ExplainerConfig:
    """Ustawienia działania ModelExplainer."""
    top_n_features: int = 5
    permutation_repeats: int = 8
    permutation_n_jobs: int = -1
    shap_sample_size: int = 1000                 # ile próbek do liczenia SHAP (X_sample)
    background_sample_size: int = 200            # tło dla Kernel/Linear
    shap_method_preference: Tuple[str, ...] = ("tree", "linear", "auto")  # kolejnosc prob
    return_raw_shap_values: bool = False         # nie zwracamy surowych macierzy SHAP (domyślnie False)
    random_state: int = 42


class ModelExplainer(BaseAgent):
    """
    Explains model predictions using various interpretability methods (FI + SHAP).
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        super().__init__(
            name="ModelExplainer",
            description="Provides model interpretability"
        )
        self.config = config or ExplainerConfig()

    # === WYKONANIE GŁÓWNE ===
    def execute(
        self,
        best_model: Any,
        pycaret_wrapper: Any,            # utrzymanie kontraktu — nie jest wymagany
        data: pd.DataFrame,
        target_column: str,
        **kwargs: Any
    ) -> AgentResult:
        """
        Explain model.

        Args:
            best_model: Trained model (może być Pipeline)
            pycaret_wrapper: PyCaret wrapper (opcjonalny; niewykorzystywany tu)
            data: Training (lub finalny) DataFrame z kolumnami cech + target
            target_column: Nazwa targetu
        """
        result = AgentResult(agent_name=self.name)

        try:
            # Walidacja wejścia
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            if best_model is None:
                raise ValueError("'best_model' is required")

            # Przygotuj X, y oraz nazwy cech (uwzględnij pipeline)
            X, y = self._split_xy(data, target_column)
            feature_names = self._infer_feature_names(best_model, X)

            # === Feature Importance ===
            feature_importance = self._get_feature_importance(best_model, X, y, feature_names)

            # === SHAP (z samplowaniem) ===
            shap_summary = self._get_shap_explanations(best_model, X, feature_names)

            # === Top features + insights ===
            top_features = self._resolve_top_features(feature_importance, shap_summary, self.config.top_n_features)
            insights = self._generate_insights(feature_importance, shap_summary, top_features)

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

    # === UTIL: nazwy cech (Pipeline/ColumnTransformer) ===
    def _infer_feature_names(self, model: Any, X: pd.DataFrame) -> List[str]:
        """Spróbuj pozyskać nazwy cech po przetwarzaniu (get_feature_names_out), w przeciwnym razie użyj X.columns."""
        # Jeśli model ma transformację z get_feature_names_out (np. Pipeline->ColumnTransformer->OneHot)
        try:
            if hasattr(model, "named_steps"):
                # możliwe, że końcowy estymator: model.named_steps['preprocessor'].get_feature_names_out()
                for name, step in getattr(model, "named_steps", {}).items():
                    if hasattr(step, "get_feature_names_out"):
                        return list(step.get_feature_names_out())
            if hasattr(model, "get_feature_names_out"):
                return list(model.get_feature_names_out())
        except Exception as e:
            self.logger.warning(f"Feature names inference with get_feature_names_out failed: {e}")

        # Fallback: surowe kolumny X
        return list(X.columns)

    # === FEATURE IMPORTANCE ===
    def _get_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> Optional[pd.DataFrame]:
        """Zwraca DataFrame z kolumnami: feature, importance. Próbuje w kolejności:
        1) feature_importances_ (drzewa),
        2) coef_ (modele liniowe — abs),
        3) permutation importance (fallback)."""

        # 1) feature_importances_
        try:
            est = self._extract_estimator(model)
            if hasattr(est, "feature_importances_"):
                imp = np.asarray(est.feature_importances_).ravel()
                imp = self._align_length(imp, len(feature_names))
                df_imp = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
                    "importance", ascending=False
                )
                return df_imp
        except Exception as e:
            self.logger.warning(f"Direct feature_importances_ failed: {e}")

        # 2) coef_
        try:
            est = self._extract_estimator(model)
            if hasattr(est, "coef_"):
                coef = np.asarray(est.coef_)
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

        # 3) permutation importance
        try:
            from sklearn.inspection import permutation_importance
            res = permutation_importance(
                self._extract_estimator(model),
                X, y,
                n_repeats=self.config.permutation_repeats,
                random_state=self.config.random_state,
                n_jobs=self.config.permutation_n_jobs
            )
            imp = res.importances_mean
            imp = self._align_length(imp, len(feature_names))
            df_imp = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
                "importance", ascending=False
            )
            return df_imp
        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
            return None

    def _extract_estimator(self, model: Any) -> Any:
        """Dla Pipeline zwróć finalny estimator; w innym wypadku zwróć model."""
        try:
            # sklearn Pipeline: .steps[-1][1] lub .named_steps[last]
            if hasattr(model, "steps") and isinstance(model.steps, list) and len(model.steps) > 0:
                return model.steps[-1][1]
        except Exception:
            pass
        return model

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
    def _get_shap_explanations(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Liczy SHAP i zwraca tylko **agregaty** (średnia bezwzględna wartość SHAP per cecha),
        aby nie przerzucać ogromnych macierzy. Dobiera Explainer (tree→linear→kernel/auto)."""
        try:
            import shap  # noqa: F401
        except Exception:
            self.logger.warning("SHAP is not installed; skipping SHAP explanations.")
            return None

        # Próbkowanie danych
        X_sample = X
        try:
            if len(X) > self.config.shap_sample_size:
                X_sample = X.sample(self.config.shap_sample_size, random_state=self.config.random_state)
        except Exception:
            pass

        # Tło (masker / background)
        background = None
        try:
            if len(X) > self.config.background_sample_size:
                background = X.sample(self.config.background_sample_size, random_state=self.config.random_state)
            else:
                background = X
        except Exception:
            background = X_sample

        # Wybór explainer'a
        method_used = "auto"
        shap_values = None
        mean_abs_shap: Optional[np.ndarray] = None

        for method in self.config.shap_method_preference:
            try:
                explainer, method_used = self._build_shap_explainer(model, X_sample, background, preferred=method)
                sv = explainer(X_sample)  # może zwrócić obiekt z .values (ndarray) lub listę ndarrays (multiclass)
                shap_values = sv
                mean_abs_shap = self._aggregate_mean_abs_shap(sv)
                if mean_abs_shap is not None:
                    break
            except Exception as e:
                self.logger.warning(f"SHAP {method} explainer failed: {e}")
                continue

        if mean_abs_shap is None:
            self.logger.warning("SHAP computation failed for all methods.")
            return None

        # Dopasuj długość i zbuduj wynik
        mean_abs_shap = self._align_length(mean_abs_shap, len(feature_names))
        shap_dict = {fname: float(val) for fname, val in zip(feature_names, mean_abs_shap)}

        return {
            "mean_abs_shap": shap_dict,
            "method": method_used,
            "n_samples": int(len(X_sample)),
        }

    def _build_shap_explainer(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        background: pd.DataFrame,
        preferred: Literal["tree", "linear", "auto"]
    ):
        """Dobiera i tworzy SHAP Explainer."""
        import shap

        est = self._extract_estimator(model)
        method_used = preferred

        if preferred == "tree":
            # TreeExplainer dla drzew / boostingów
            return shap.TreeExplainer(est, feature_perturbation="interventional"), "tree"

        if preferred == "linear":
            # LinearExplainer dla modeli liniowych; wymaga background
            return shap.LinearExplainer(est, background, feature_perturbation="interventional"), "linear"

        # auto fallback (kernel/Unified Explainer)
        # Nowe API shap.Explainer sam dobiera metodę na bazie modelu/maskera
        masker = shap.maskers.Independent(background)
        return shap.Explainer(est, masker), "auto"

    def _aggregate_mean_abs_shap(self, shap_values_obj: Any) -> Optional[np.ndarray]:
        """Zwraca średnią bezwzględną SHAP per cecha.
        Obsługa:
          - binarna/regresja: sv.values shape (n, m)
          - multiclass: sv.values -> array o shape (k, n, m) lub lista [ (n,m) x k ]
        """
        try:
            import numpy as np  # noqa: F401
            sv = shap_values_obj
            vals = getattr(sv, "values", None)

            # shap>=0.43 unified API: .values lub .values[i] per class
            if vals is None:
                # próbuj traktować sv jak listę
                if isinstance(sv, list) and len(sv) > 0:
                    # lista macierzy (n, m) dla każdej klasy
                    mats = [np.abs(v.values if hasattr(v, "values") else np.array(v)).mean(axis=0) for v in sv]
                    return np.mean(np.vstack(mats), axis=0)
                return None

            arr = np.asarray(vals)
            if arr.ndim == 2:
                # (n, m)
                return np.abs(arr).mean(axis=0)
            if arr.ndim == 3:
                # (k, n, m) multiclass -> średnia po klasach i próbkach
                return np.abs(arr).mean(axis=(0, 1))
            # inne kształty — spróbuj spłaszczyć po pierwszym wymiarze (próbki)
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

        # FI-based insight
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
                            f"Top 3 cechy odpowiadają za {top3:.1f}% całkowitej ważności — model silnie zależy od nielicznych cech."
                        )
            except Exception:
                pass

        # SHAP-based insight
        if shap_summary and "mean_abs_shap" in shap_summary:
            method = shap_summary.get("method", "auto")
            insights.append(f"Wartości SHAP policzono ({method}); dostępna agregacja ważności per cecha.")
            if not feature_importance and top_features:
                insights.append(f"Top cechy (SHAP): {', '.join(top_features[:3])}")

        if not insights:
            insights.append("Brak wystarczających danych do wygenerowania wniosków o ważności cech.")

        return insights
