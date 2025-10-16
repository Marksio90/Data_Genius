# agents/ml/model_explainer.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Model Explainer                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Advanced model interpretability with enterprise safeguards:               ║
║    ✓ Feature importance (native, permutation, coefficients)              ║
║    ✓ SHAP analysis (Tree/Linear/Auto explainers)                         ║
║    ✓ Pipeline support (ColumnTransformer, OneHotEncoder)                 ║
║    ✓ Stratified sampling for SHAP                                        ║
║    ✓ Automatic feature name resolution                                   ║
║    ✓ Sparse matrix handling                                              ║
║    ✓ Multiple explainer fallbacks                                        ║
║    ✓ Length alignment guards                                             ║
║    ✓ Defensive error handling                                            ║
║    ✓ Comprehensive insights generation                                   ║
║    ✓ Stable output contract                                              ║
║    ✓ Full telemetry tracking                                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Interpretability Methods:
    1. Native Feature Importance:
        • tree.feature_importances_
        • linear.coef_
    
    2. Permutation Importance:
        • Model-agnostic
        • Custom scorer support
        • Parallel execution
    
    3. SHAP Analysis:
        • TreeExplainer (tree models)
        • LinearExplainer (linear models)
        • Explainer (auto fallback)
        • Stratified sampling
        • Mean |SHAP| aggregation

Pipeline Support:
    • ColumnTransformer compatibility
    • OneHotEncoder tracking
    • Passthrough column preservation
    • Feature name reconstruction
    • Sparse matrix support
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch, check_random_state

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
except ImportError:
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
            self.logger = logger
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data: Dict[str, Any] = {}
            self.errors: List[str] = []
        
        def add_error(self, error: str):
            self.errors.append(error)
        
        def is_success(self) -> bool:
            return len(self.errors) == 0

warnings.filterwarnings('ignore')


__all__ = ["ExplainerConfig", "ModelExplainer", "explain_model"]
__version__ = "5.1-kosmos-enterprise"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExplainerConfig:
    """
    Configuration for model explainer.
    
    Attributes:
        top_n_features: Number of top features to return
        permutation_repeats: Permutation importance repeats
        permutation_n_jobs: Parallel jobs for permutation
        shap_sample_size: Samples for SHAP computation
        background_sample_size: Background samples for SHAP
        shap_method_preference: SHAP explainer preference order
        return_raw_shap_values: Include raw SHAP values
        random_state: Random seed
        permutation_scorer: Scorer for permutation importance
        debug_max_logs: Max debug log entries
    """
    top_n_features: int = 5
    permutation_repeats: int = 8
    permutation_n_jobs: int = -1
    shap_sample_size: int = 1000
    background_sample_size: int = 200
    shap_method_preference: Tuple[str, ...] = ("tree", "linear", "auto")
    return_raw_shap_values: bool = False
    random_state: int = 42
    permutation_scorer: Optional[str] = None
    debug_max_logs: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "top_n_features": self.top_n_features,
            "permutation_repeats": self.permutation_repeats,
            "shap_sample_size": self.shap_sample_size,
            "background_sample_size": self.background_sample_size,
            "shap_method_preference": list(self.shap_method_preference),
            "return_raw_shap_values": self.return_raw_shap_values,
            "random_state": self.random_state
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str):
    """Decorator for operation timing."""
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
    """Decorator for safe operations with fallback."""
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
# SECTION: Main Model Explainer
# ═══════════════════════════════════════════════════════════════════════════

class ModelExplainer(BaseAgent):
    """
    **ModelExplainer** — Advanced model interpretability.
    
    Responsibilities:
      1. Extract native feature importance
      2. Compute permutation importance
      3. Calculate SHAP values
      4. Handle Pipeline transformations
      5. Resolve feature names post-preprocessing
      6. Sample data for SHAP (stratified)
      7. Aggregate SHAP values
      8. Generate top features list
      9. Create interpretability insights
      10. Maintain stable output contract
    
    Features:
      • 3 importance methods
      • 3 SHAP explainers
      • Pipeline support
      • Stratified sampling
      • Sparse matrix handling
      • Feature name tracking
      • Defensive fallbacks
    """
    
    def __init__(self, config: Optional[ExplainerConfig] = None) -> None:
        """
        Initialize model explainer.
        
        Args:
            config: Optional custom configuration
        """
        super().__init__(
            name="ModelExplainer",
            description="Advanced model interpretability with SHAP and feature importance"
        )
        self.config = config or ExplainerConfig()
        self._log = logger.bind(agent="ModelExplainer")
        self._rng = check_random_state(self.config.random_state)
        
        self._log.info("✓ ModelExplainer initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("model_explanation")
    def execute(
        self,
        best_model: Any,
        pycaret_wrapper: Any,
        data: pd.DataFrame,
        target_column: str,
        **kwargs: Any
    ) -> AgentResult:
        """
        Explain model predictions.
        
        Args:
            best_model: Trained model (may be Pipeline)
            pycaret_wrapper: PyCaret wrapper (unused, for contract)
            data: DataFrame with features + target
            target_column: Target column name
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with feature importance and SHAP analysis
        """
        result = AgentResult(agent_name=self.name)
        
        try:
            # ─── Validation ───
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("'data' must be a non-empty pandas DataFrame")
            
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
            if best_model is None:
                raise ValueError("'best_model' is required")
            
            self._log.info(f"Explaining model | rows={len(data)} | cols={len(data.columns)}")
            
            # ─── Split X/y ───
            X_raw, y = self._split_xy(data, target_column)
            
            # ─── Pipeline-Aware Transform ───
            tx = self._transform_if_pipeline(best_model, X_raw)
            X_for_model = tx.X
            feature_names = tx.feature_names
            estimator = self._extract_estimator(best_model)
            
            self._log.debug(f"Features after preprocessing: {len(feature_names)}")
            
            # ─── Feature Importance ───
            scorer_name = kwargs.get("permutation_scorer", self.config.permutation_scorer)
            if scorer_name is None:
                scorer_name = self._infer_default_scorer(y)
            
            feature_importance = self._get_feature_importance(
                estimator, X_for_model, y, feature_names, scorer_name=scorer_name
            )
            
            # ─── SHAP Analysis ───
            shap_summary = self._get_shap_explanations(
                estimator, X_for_model, feature_names, y=y
            )
            
            # ─── Top Features & Insights ───
            top_features = self._resolve_top_features(
                feature_importance, shap_summary, self.config.top_n_features
            )
            insights = self._generate_insights(
                feature_importance, shap_summary, top_features
            )
            
            # ─── Result ───
            result.data = {
                "feature_importance": feature_importance,
                "shap_values": shap_summary,
                "top_features": top_features,
                "insights": insights,
            }
            
            self._log.success(
                f"✓ Model explanation complete | "
                f"top_features={len(top_features)} | "
                f"has_shap={shap_summary is not None}"
            )
        
        except Exception as e:
            result.add_error(f"Model explanation failed: {e}")
            self._log.error(f"Explanation error: {e}", exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Data Preparation
    # ───────────────────────────────────────────────────────────────────
    
    def _split_xy(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split features and target."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    
    # ───────────────────────────────────────────────────────────────────
    # Pipeline Handling
    # ───────────────────────────────────────────────────────────────────
    
    def _extract_estimator(self, model: Any) -> Any:
        """Extract final estimator from Pipeline."""
        try:
            if isinstance(model, Pipeline) and model.steps:
                return model.steps[-1][1]
        except Exception:
            pass
        return model
    
    def _extract_preprocessor(self, model: Any) -> Optional[Any]:
        """Extract preprocessor from Pipeline."""
        try:
            if isinstance(model, Pipeline) and model.steps:
                for name, step in model.steps[:-1]:
                    if isinstance(step, ColumnTransformer) or hasattr(step, "transform"):
                        return step
        except Exception:
            pass
        return None
    
    @_safe_operation("transform_pipeline", default_value=None)
    def _transform_if_pipeline(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> Bunch:
        """Transform data through Pipeline and track feature names."""
        pre = self._extract_preprocessor(model)
        if pre is None:
            return Bunch(X=X, feature_names=list(X.columns), meta=None)
        
        # Transform
        try:
            X_trans = pre.transform(X)
        except Exception:
            X_trans = pre.fit_transform(X)
        
        # Get feature names
        feature_names = self._safe_feature_names_out(pre, X_trans, X.columns)
        
        return Bunch(X=X_trans, feature_names=feature_names, meta={"preprocessor": pre})
    
    @_safe_operation("get_feature_names", default_value=[])
    def _safe_feature_names_out(
        self,
        pre: Any,
        X_trans: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
        raw_cols: List[str]
    ) -> List[str]:
        """Get feature names after ColumnTransformer."""
        # Try sklearn API
        try:
            names = list(pre.get_feature_names_out())
            if names:
                return names
        except Exception:
            pass
        
        # Manual reconstruction
        names: List[str] = []
        try:
            if hasattr(pre, "transformers_"):
                for name, trans, cols in pre.transformers_:
                    if name == "remainder" and trans == "drop":
                        continue
                    
                    if trans == "passthrough":
                        names += [str(c) for c in (cols if isinstance(cols, list) else raw_cols)]
                    
                    elif hasattr(trans, "get_feature_names_out"):
                        try:
                            out = trans.get_feature_names_out(cols)
                        except Exception:
                            out = trans.get_feature_names_out()
                        names += [f"{name}__{f}" for f in out]
                    
                    else:
                        base_cols = cols if isinstance(cols, list) else raw_cols
                        names += [f"{name}__{c}" for c in base_cols]
        
        except Exception as e:
            self._log.warning(f"Feature name reconstruction failed: {e}")
            n = X_trans.shape[1] if hasattr(X_trans, "shape") else len(raw_cols)
            names = [f"f{i}" for i in range(n)]
        
        # Use DataFrame columns if available
        try:
            if isinstance(X_trans, pd.DataFrame) and len(X_trans.columns) == len(names):
                return [str(c) for c in X_trans.columns]
        except Exception:
            pass
        
        return names
    
    # ───────────────────────────────────────────────────────────────────
    # Feature Importance
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("feature_importance")
    @_safe_operation("feature_importance", default_value=None)
    def _get_feature_importance(
        self,
        estimator: Any,
        X: Union[pd.DataFrame, np.ndarray, sparse.spmatrix],
        y: pd.Series,
        feature_names: List[str],
        scorer_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Get feature importance using multiple methods."""
        # Try native feature_importances_
        try:
            if hasattr(estimator, "feature_importances_"):
                imp = np.asarray(estimator.feature_importances_).ravel()
                imp = self._align_length(imp, len(feature_names))
                df_imp = pd.DataFrame({
                    "feature": feature_names,
                    "importance": imp
                }).sort_values("importance", ascending=False)
                return df_imp
        except Exception as e:
            self._log.debug(f"Native importance failed: {e}")
        
        # Try coefficients
        try:
            if hasattr(estimator, "coef_"):
                coef = np.asarray(estimator.coef_)
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    coef = np.abs(coef)
                coef = self._align_length(coef, len(feature_names))
                df_imp = pd.DataFrame({
                    "feature": feature_names,
                    "importance": coef
                }).sort_values("importance", ascending=False)
                return df_imp
        except Exception as e:
            self._log.debug(f"Coefficient importance failed: {e}")
        
        # Try permutation importance
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
            df_imp = pd.DataFrame({
                "feature": feature_names,
                "importance": imp
            }).sort_values("importance", ascending=False)
            return df_imp
        except Exception as e:
            self._log.warning(f"Permutation importance failed: {e}")
            return None
    
    def _align_length(self, arr: np.ndarray, n: int) -> np.ndarray:
        """Align importance vector length to feature count."""
        arr = np.asarray(arr).ravel()
        
        if len(arr) == n:
            return arr
        
        if len(arr) > n:
            self._log.warning(f"Importance vector too long ({len(arr)} > {n}); truncating")
            return arr[:n]
        
        # Pad with zeros
        self._log.warning(f"Importance vector too short ({len(arr)} < {n}); padding")
        pad = np.zeros(n - len(arr))
        return np.concatenate([arr, pad])
    
    # ───────────────────────────────────────────────────────────────────
    # SHAP Analysis
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("shap_analysis")
    @_safe_operation("shap_analysis", default_value=None)
    def _get_shap_explanations(
        self,
        estimator: Any,
        X: Union[pd.DataFrame, np.ndarray, sparse.spmatrix],
        feature_names: List[str],
        y: Optional[pd.Series] = None
    ) -> Optional[Dict[str, Any]]:
        """Compute SHAP values and aggregate."""
        try:
            import shap
        except ImportError:
            self._log.warning("SHAP not installed; skipping SHAP analysis")
            return None
        
        # Sample data
        X_sample = self._sample_for_shap(X, y)
        background = self._sample_for_shap(X, y)
        
        use_dense = False
        method_used = "auto"
        mean_abs_shap: Optional[np.ndarray] = None
        raw_payload: Optional[Any] = None
        
        # Try explainers in preference order
        for method in self.config.shap_method_preference:
            try:
                explainer, method_used = self._build_shap_explainer(
                    estimator, X_sample, background, preferred=method
                )
                sv = explainer(X_sample)
                mean_abs_shap = self._aggregate_mean_abs_shap(sv)
                raw_payload = sv if self.config.return_raw_shap_values else None
                
                if mean_abs_shap is not None:
                    break
            
            except Exception as e:
                # Try dense conversion
                if not use_dense and (sparse.issparse(X_sample) or sparse.issparse(background)):
                    try:
                        X_sample = self._dense_if_needed(X_sample)
                        background = self._dense_if_needed(background)
                        use_dense = True
                        self._log.debug(f"Retrying SHAP with dense matrices")
                        continue
                    except Exception:
                        pass
                
                self._log.debug(f"SHAP {method} explainer failed: {e}")
                continue
        
        if mean_abs_shap is None:
            self._log.warning("SHAP computation failed for all methods")
            return None
        
        # Align and build result
        mean_abs_shap = self._align_length(mean_abs_shap, len(feature_names))
        shap_dict = {fname: float(val) for fname, val in zip(feature_names, mean_abs_shap)}
        
        out: Dict[str, Any] = {
            "mean_abs_shap": shap_dict,
            "method": method_used,
            "n_samples": int(X_sample.shape[0] if hasattr(X_sample, "shape") else len(X_sample)),
        }
        
        if self.config.return_raw_shap_values:
            try:
                out["raw"] = getattr(raw_payload, "values", None)
            except Exception:
                out["raw"] = None
        
        return out
    
    def _sample_for_shap(
        self,
        X: Union[pd.DataFrame, np.ndarray, sparse.spmatrix],
        y: Optional[pd.Series]
    ) -> Union[pd.DataFrame, np.ndarray, sparse.spmatrix]:
        """Sample data for SHAP with stratification."""
        n = X.shape[0]
        size = min(self.config.shap_sample_size, n)
        
        if size >= n:
            return X
        
        # Stratified sampling for classification
        if isinstance(X, pd.DataFrame) and y is not None:
            try:
                y_non_na = y.dropna()
                if y_non_na.nunique() > 1 and len(y_non_na) == len(y):
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
        
        # Fallback: simple random sampling
        if isinstance(X, pd.DataFrame):
            return X.sample(size, random_state=self.config.random_state)
        else:
            idx = self._rng.choice(n, size=size, replace=False)
            return X[idx]
    
    def _dense_if_needed(
        self,
        A: Union[np.ndarray, sparse.spmatrix]
    ) -> np.ndarray:
        """Convert sparse to dense if needed."""
        if sparse.issparse(A):
            return A.toarray()
        return np.asarray(A)
    
    def _build_shap_explainer(
        self,
        estimator: Any,
        X_sample: Union[pd.DataFrame, np.ndarray],
        background: Union[pd.DataFrame, np.ndarray],
        preferred: Literal["tree", "linear", "auto"]
    ):
        """Build SHAP explainer."""
        import shap
        
        if preferred == "tree":
            return shap.TreeExplainer(estimator, feature_perturbation="interventional"), "tree"
        
        if preferred == "linear":
            return shap.LinearExplainer(estimator, background, feature_perturbation="interventional"), "linear"
        
        # Auto fallback
        masker = shap.maskers.Independent(background)
        return shap.Explainer(estimator, masker), "auto"
    
    def _aggregate_mean_abs_shap(self, shap_values_obj: Any) -> Optional[np.ndarray]:
        """Aggregate SHAP values to mean |SHAP| per feature."""
        try:
            sv = shap_values_obj
            vals = getattr(sv, "values", None)
            
            if vals is None and hasattr(sv, "__array__"):
                vals = np.array(sv)
            
            if vals is None:
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
                return np.abs(arr).mean(axis=0)
            
            if arr.ndim == 3:
                return np.abs(arr).mean(axis=(0, 1))
            
            if arr.ndim > 3:
                return np.abs(arr).reshape((-1, arr.shape[-1])).mean(axis=0)
        
        except Exception as e:
            self._log.warning(f"SHAP aggregation failed: {e}")
        
        return None
    
    # ───────────────────────────────────────────────────────────────────
    # Top Features & Insights
    # ───────────────────────────────────────────────────────────────────
    
    def _resolve_top_features(
        self,
        feature_importance: Optional[pd.DataFrame],
        shap_summary: Optional[Dict[str, Any]],
        k: int
    ) -> List[str]:
        """Resolve top k features."""
        if feature_importance is not None and not feature_importance.empty:
            return feature_importance["feature"].head(k).tolist()
        
        if shap_summary and "mean_abs_shap" in shap_summary:
            sorted_feats = sorted(
                shap_summary["mean_abs_shap"].items(),
                key=lambda kv: kv[1],
                reverse=True
            )
            return [f for f, _ in sorted_feats[:k]]
        
        return []
    
    def _generate_insights(
        self,
        feature_importance: Optional[pd.DataFrame],
        shap_summary: Optional[Dict[str, Any]],
        top_features: List[str]
    ) -> List[str]:
        """Generate interpretability insights."""
        insights: List[str] = []
        
        # Feature importance insights
        if feature_importance is not None and not feature_importance.empty:
            top_feature = feature_importance.iloc[0]["feature"]
            top_importance = float(feature_importance.iloc[0]["importance"])
            insights.append(
                f"Most important feature: {top_feature} (importance: {top_importance:.4f})"
            )
            
            try:
                total = float(feature_importance["importance"].sum())
                if total > 0:
                    top3 = float(feature_importance.head(3)["importance"].sum()) / total * 100.0
                    if top3 > 70:
                        insights.append(
                            f"Top 3 features account for {top3:.1f}% of total importance — "
                            f"model heavily relies on few features"
                        )
            except Exception:
                pass
        
        # SHAP insights
        if shap_summary and "mean_abs_shap" in shap_summary:
            method = shap_summary.get("method", "auto")
            insights.append(f"SHAP ({method}) analysis shows global feature impact")
            
            if (not feature_importance or feature_importance.empty) and top_features:
                insights.append(f"Top features (SHAP): {', '.join(top_features[:3])}")
        
        if not insights:
            insights.append("Insufficient data to generate feature importance insights")
        
        return insights
    
    # ───────────────────────────────────────────────────────────────────
    # Helper Methods
    # ───────────────────────────────────────────────────────────────────
    
    def _infer_default_scorer(self, y: pd.Series) -> str:
        """Infer appropriate scorer for permutation importance."""
        try:
            if pd.api.types.is_numeric_dtype(y):
                return "neg_mean_squared_error"
            
            n_uni = y.dropna().nunique()
            if n_uni <= 2:
                return "roc_auc"
            return "roc_auc_ovr"
        
        except Exception:
            return "neg_mean_squared_error"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Function
# ═══════════════════════════════════════════════════════════════════════════

def explain_model(
    best_model: Any,
    data: pd.DataFrame,
    target_column: str,
    pycaret_wrapper: Optional[Any] = None,
    config: Optional[ExplainerConfig] = None,
    **kwargs
) -> AgentResult:
    """
    Convenience function to explain a model.
    
    Usage:
```python
        from agents.ml import explain_model
        
        # Basic usage
        result = explain_model(
            best_model=model,
            data=df,
            target_column='target'
        )
        
        # Access results
        feature_importance = result.data['feature_importance']
        shap_values = result.data['shap_values']
        top_features = result.data['top_features']
        insights = result.data['insights']
        
        print(f"Top Features: {top_features}")
        print(f"\nFeature Importance:")
        print(feature_importance.head())
        
        # With custom config
        config = ExplainerConfig(top_n_features=10,
shap_sample_size=2000,
permutation_repeats=10
)
    result = explain_model(
        best_model=model,
        data=df,
        target_column='target',
        config=config
    )
    
    Args:
        best_model: Trained model to explain
        data: DataFrame with features + target
        target_column: Target column name
        pycaret_wrapper: Optional PyCaret wrapper
        config: Optional custom configuration
        **kwargs: Additional parameters
    
    Returns:
        AgentResult with feature importance and SHAP analysis
    """
    explainer = ModelExplainer(config)
    return explainer.execute(
        best_model=best_model,
        pycaret_wrapper=pycaret_wrapper,
        data=data,
        target_column=target_column,
        **kwargs
    )