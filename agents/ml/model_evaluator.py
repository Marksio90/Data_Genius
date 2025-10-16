# agents/ml/model_evaluator.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Model Evaluator                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Universal ML model evaluation with enterprise safeguards:                 ║
║    ✓ Classification metrics (20+ metrics)                                 ║
║      • Accuracy, Precision, Recall, F1 (macro/micro/weighted)            ║
║      • ROC-AUC (binary/OvR/OvO), PR-AUC                                  ║
║      • Confusion matrix (raw + normalized)                               ║
║      • Per-class metrics, Classification report                          ║
║      • Threshold optimization (Youden/F1)                                ║
║      • KS statistic, Lift@k, Top-k gains                                 ║
║      • ROC/PR curves (sampled for large datasets)                        ║
║    ✓ Regression metrics (10+ metrics)                                     ║
║      • MAE, MSE, RMSE, R², Median AE                                     ║
║      • MAPE, MSLE                                                        ║
║      • Residual diagnostics (bias, IQR outliers, quantiles)             ║
║    ✓ Defensive programming                                                ║
║      • Handles missing predictions/targets                               ║
║      • Multi-format prediction support                                   ║
║      • Binary/multiclass auto-detection                                  ║
║      • Large dataset sampling                                            ║
║    ✓ PyCaret + sklearn compatibility                                      ║
║    ✓ Stable output contract                                               ║
║    ✓ Comprehensive telemetry                                              ║
╚════════════════════════════════════════════════════════════════════════════╝

Supported Metrics:
    Classification:
        • Basic: accuracy, precision, recall, f1
        • Advanced: ROC-AUC, PR-AUC, confusion matrix
        • Threshold: Youden's J, F1-optimal
        • Business: KS statistic, Lift@k
    
    Regression:
        • Error: MAE, MSE, RMSE, Median AE
        • Variance: R², MAPE, MSLE
        • Diagnostics: Residual analysis
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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


__all__ = ["EvalConfig", "ModelEvaluator", "evaluate_model"]
__version__ = "5.1-kosmos-enterprise"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EvalConfig:
    """
    Configuration for model evaluation.
    
    Attributes:
        return_predictions: Include predictions in output
        prefer_pycaret_predict: Use PyCaret predict_model if available
        classification_roc_strategy: ROC strategy (auto/binary/ovr/ovo)
        primary_metric_default_cls: Default classification metric
        primary_metric_default_reg: Default regression metric
        include_per_class: Include per-class metrics
        include_curves_sample_cap: Max samples for ROC/PR curves
        optimize_thresholds: Enable threshold optimization
        threshold_grid_size: Threshold grid resolution
        lift_k_list: Lift@k percentiles
        compute_ks: Compute KS statistic
        sample_predictions_cap: Max predictions to return
    """
    return_predictions: bool = True
    prefer_pycaret_predict: bool = True
    classification_roc_strategy: Literal["auto", "binary", "ovr", "ovo"] = "auto"
    primary_metric_default_cls: str = "accuracy"
    primary_metric_default_reg: str = "r2"
    include_per_class: bool = True
    include_curves_sample_cap: int = 200_000
    optimize_thresholds: bool = True
    threshold_grid_size: int = 101
    lift_k_list: Tuple[int, ...] = (1, 3, 5, 10)
    compute_ks: bool = True
    sample_predictions_cap: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "return_predictions": self.return_predictions,
            "prefer_pycaret_predict": self.prefer_pycaret_predict,
            "classification_roc_strategy": self.classification_roc_strategy,
            "primary_metric_default_cls": self.primary_metric_default_cls,
            "primary_metric_default_reg": self.primary_metric_default_reg,
            "include_per_class": self.include_per_class,
            "optimize_thresholds": self.optimize_thresholds,
            "compute_ks": self.compute_ks
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
# SECTION: Main Model Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class ModelEvaluator(BaseAgent):
    """
    **ModelEvaluator** — Universal ML model evaluation.
    
    Responsibilities:
      1. Generate predictions (PyCaret/sklearn)
      2. Compute classification metrics (20+)
      3. Compute regression metrics (10+)
      4. Optimize decision thresholds
      5. Calculate business metrics (KS, Lift)
      6. Generate diagnostic plots data
      7. Handle missing data gracefully
      8. Support multi-format predictions
      9. Maintain stable output contract
      10. Track comprehensive telemetry
    
    Features:
      • 30+ metrics total
      • Threshold optimization
      • Business metrics (KS, Lift)
      • Residual diagnostics
      • ROC/PR curves
      • Confusion matrices
      • Per-class reports
      • Large dataset sampling
    """
    
    def __init__(self, config: Optional[EvalConfig] = None) -> None:
        """
        Initialize model evaluator.
        
        Args:
            config: Optional custom configuration
        """
        super().__init__(
            name="ModelEvaluator",
            description="Universal ML model evaluation with comprehensive metrics"
        )
        self.config = config or EvalConfig()
        self._log = logger.bind(agent="ModelEvaluator")
        
        self._log.info("✓ ModelEvaluator initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("model_evaluation")
    def execute(
        self,
        best_model: Any,
        pycaret_wrapper: Optional[Any],
        problem_type: Literal["classification", "regression"],
        data: Optional[pd.DataFrame] = None,
        *,
        y_true: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
        primary_metric: Optional[str] = None,
        positive_label: Optional[Any] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Evaluate trained model.
        
        Args:
            best_model: Trained model to evaluate
            pycaret_wrapper: PyCaret wrapper instance (if available)
            problem_type: 'classification' or 'regression'
            data: DataFrame for predictions
            y_true: True labels (if not using PyCaret)
            primary_metric: Primary metric for best_score
            positive_label: Positive class label (binary classification)
            sample_weight: Sample weights
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with comprehensive evaluation metrics
        """
        result = AgentResult(agent_name=self.name)
        
        try:
            # ─── Validation ───
            if best_model is None:
                raise ValueError("'best_model' is required")
            
            if problem_type not in {"classification", "regression"}:
                raise ValueError(f"Unsupported problem_type: {problem_type}")
            
            self._log.info(f"Evaluating model | type={problem_type}")
            
            # ─── Step 1: Generate Predictions ───
            preds_df, used_cols = self._get_predictions_dataframe(
                model=best_model,
                data=data,
                pycaret_wrapper=pycaret_wrapper
            )
            
            n_samples = int(len(preds_df)) if preds_df is not None else 0
            self._log.debug(f"Generated predictions | n_samples={n_samples}")
            
            # ─── Step 2: Extract Targets and Scores ───
            y_true_vec, y_pred_vec, y_proba, classes = self._extract_targets_and_scores(
                preds_df=preds_df,
                model=best_model,
                data=data,
                y_true=y_true,
                problem_type=problem_type
            )
            
            # ─── Step 3: Compute Metrics ───
            if problem_type == "classification":
                metrics = self._evaluate_classification(
                    y_true=y_true_vec,
                    y_pred=y_pred_vec,
                    y_proba=y_proba,
                    classes=classes,
                    positive_label=positive_label,
                    sample_weight=sample_weight
                )
                primary_metric_name = primary_metric or self.config.primary_metric_default_cls
            else:
                metrics = self._evaluate_regression(
                    y_true=y_true_vec,
                    y_pred=y_pred_vec,
                    sample_weight=sample_weight
                )
                primary_metric_name = primary_metric or self.config.primary_metric_default_reg
            
            # ─── Step 4: Resolve Best Score ───
            best_score = self._resolve_best_score(metrics, primary_metric_name)
            
            # ─── Step 5: Prepare Output ───
            out_preds = self._maybe_slice_predictions(preds_df)
            
            result.data = {
                "metrics": metrics,
                "predictions": out_preds if self.config.return_predictions else None,
                "best_model_name": type(best_model).__name__,
                "best_score": best_score,
                "problem_type": problem_type,
                "meta": {
                    "n_samples": n_samples,
                    "prediction_columns": used_cols,
                    "used_primary_metric": primary_metric_name,
                    "classes": (classes.tolist() if isinstance(classes, np.ndarray) else classes),
                }
            }
            
            self._log.success(
                f"✓ Evaluation complete | "
                f"type={problem_type} | "
                f"metric={primary_metric_name} | "
                f"score={best_score if best_score is not None else 'n/a'}"
            )
        
        except Exception as e:
            result.add_error(f"Model evaluation failed: {e}")
            self._log.error(f"Evaluation error: {e}", exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Prediction Generation
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("generate_predictions")
    @_safe_operation("generate_predictions", default_value=(None, []))
    def _get_predictions_dataframe(
        self,
        model: Any,
        data: Optional[pd.DataFrame],
        pycaret_wrapper: Optional[Any]
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Generate predictions DataFrame with consistent schema.
        
        Returns:
            Tuple of (predictions_df, column_names)
        """
        used_cols: List[str] = []
        preds_df: Optional[pd.DataFrame] = None
        
        # ─── Try PyCaret First ───
        if pycaret_wrapper is not None and self.config.prefer_pycaret_predict:
            try:
                preds_df = (
                    pycaret_wrapper.predict_model(model, data=data)
                    if data is not None
                    else pycaret_wrapper.predict_model(model)
                )
                
                if isinstance(preds_df, pd.DataFrame):
                    used_cols = preds_df.columns.tolist()
                    return preds_df, used_cols
                
                self._log.warning("PyCaret predict_model returned non-DataFrame; falling back")
            
            except Exception as e:
                self._log.warning(f"PyCaret predict_model failed: {e}; falling back")
        
        # ─── Sklearn Fallback ───
        try:
            if data is None:
                raise ValueError("No 'data' provided for raw model prediction")
            
            y_pred = None
            y_proba = None
            
            # Get predictions
            if hasattr(model, "predict"):
                y_pred = model.predict(data)
            
            # Get probabilities
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(data)
                except Exception:
                    y_proba = None
            
            # Build DataFrame
            records: Dict[str, Any] = {}
            
            if y_pred is not None:
                records["Label"] = y_pred
            
            if y_proba is not None and isinstance(y_proba, np.ndarray):
                if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                    records["Score"] = y_proba.ravel()
                else:
                    classes = getattr(model, "classes_", np.arange(y_proba.shape[1]))
                    for j, cls in enumerate(classes):
                        records[f"proba_{cls}"] = y_proba[:, j]
            
            preds_df = pd.DataFrame(records, index=data.index)
            used_cols = preds_df.columns.tolist()
            return preds_df, used_cols
        
        except Exception as e:
            self._log.warning(f"Raw model prediction failed: {e}")
        
        return preds_df, used_cols
    
    # ───────────────────────────────────────────────────────────────────
    # Target & Score Extraction
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("extract_targets", default_value=(pd.Series(), pd.Series(), None, None))
    def _extract_targets_and_scores(
        self,
        preds_df: Optional[pd.DataFrame],
        model: Any,
        data: Optional[pd.DataFrame],
        y_true: Optional[Union[pd.Series, np.ndarray, List[Any]]],
        problem_type: str
    ) -> Tuple[pd.Series, pd.Series, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract y_true, y_pred, y_proba, and classes.
        
        Returns:
            Tuple of (y_true, y_pred, y_proba, classes)
        """
        # ─── Extract y_true ───
        if y_true is not None:
            y_true_vec = pd.Series(y_true).reset_index(drop=True)
        elif (preds_df is not None) and (preds_df.shape[1] >= 1):
            first_col = preds_df.columns[0]
            if first_col.lower() in {"target", "y", "y_true"}:
                y_true_vec = preds_df.iloc[:, 0].reset_index(drop=True)
            else:
                y_true_vec = pd.Series(dtype=float)
        else:
            y_true_vec = pd.Series(dtype=float)
        
        # ─── Extract y_pred ───
        y_pred_vec = pd.Series(dtype=float)
        if preds_df is not None:
            for col in ["Label", "prediction_label", "pred", "y_pred"]:
                if col in preds_df.columns:
                    y_pred_vec = preds_df[col].reset_index(drop=True)
                    break
            
            if y_pred_vec.empty and preds_df.shape[1] >= 2:
                y_pred_vec = preds_df.iloc[:, -1].reset_index(drop=True)
        
        if y_pred_vec.empty and hasattr(model, "predict") and data is not None:
            try:
                y_pred_vec = pd.Series(model.predict(data)).reset_index(drop=True)
            except Exception:
                pass
        
        # ─── Extract y_proba (Classification) ───
        y_proba = None
        classes = getattr(model, "classes_", None) if problem_type == "classification" else None
        
        if problem_type == "classification":
            if preds_df is not None:
                if "Score" in preds_df.columns:
                    y_proba = preds_df["Score"].to_numpy().reshape(-1, 1)
                else:
                    prob_cols = [c for c in preds_df.columns if c.startswith("proba_")]
                    if prob_cols:
                        y_proba = preds_df[prob_cols].to_numpy()
                        if classes is None:
                            try:
                                classes = np.array([c.replace("proba_", "") for c in prob_cols])
                            except Exception:
                                pass
            
            if y_proba is None and hasattr(model, "predict_proba") and data is not None:
                try:
                    y_proba = model.predict_proba(data)
                except Exception:
                    y_proba = None
        
        # ─── Normalize Arrays ───
        y_true_vec = y_true_vec.reset_index(drop=True)
        y_pred_vec = y_pred_vec.reset_index(drop=True)
        
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 1:
                y_proba = y_proba.reshape(-1, 1)
        
        return y_true_vec, y_pred_vec, y_proba, classes
    
    # ───────────────────────────────────────────────────────────────────
    # Classification Metrics
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("evaluate_classification")
    @_safe_operation("classification_metrics", default_value={})
    def _evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_proba: Optional[np.ndarray],
        classes: Optional[np.ndarray],
        positive_label: Optional[Any],
        sample_weight: Optional[Union[pd.Series, np.ndarray, List[float]]] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive classification metrics."""
        metrics: Dict[str, Any] = {}
        
        if y_true.empty or y_pred.empty:
            self._log.warning("Classification: missing y_true or y_pred; metrics limited")
            return metrics
        
        sw = np.array(sample_weight) if sample_weight is not None else None
        
        # ─── Basic Metrics ───
        try:
            metrics["accuracy"] = accuracy_score(y_true, y_pred, sample_weight=sw)
        except Exception:
            pass
        
        # ─── Precision/Recall/F1 (Multiple Averages) ───
        for avg in ("weighted", "macro", "micro"):
            try:
                metrics[f"precision_{avg}"] = precision_score(
                    y_true, y_pred, average=avg, zero_division=0, sample_weight=sw
                )
            except Exception:
                pass
            
            try:
                metrics[f"recall_{avg}"] = recall_score(
                    y_true, y_pred, average=avg, zero_division=0, sample_weight=sw
                )
            except Exception:
                pass
            
            try:
                metrics[f"f1_{avg}"] = f1_score(
                    y_true, y_pred, average=avg, zero_division=0, sample_weight=sw
                )
            except Exception:
                pass
        
        # ─── Per-Class Report ───
        if self.config.include_per_class:
            try:
                cr = classification_report(
                    y_true, y_pred, zero_division=0, output_dict=True, sample_weight=sw
                )
                metrics["classification_report"] = cr
            except Exception:
                pass
        
        # ─── Confusion Matrix ───
        try:
            cm = confusion_matrix(y_true, y_pred, sample_weight=sw)
            metrics["confusion_matrix"] = {"matrix": cm.tolist()}
            
            # Normalized
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = cm / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm, nan=0.0).tolist()
            metrics["confusion_matrix_normalized"] = {"matrix": cm_norm}
        except Exception:
            pass
        
        # ─── ROC/PR Analysis ───
        if y_proba is not None:
            uniq = np.unique(pd.Series(y_true).dropna())
            is_binary = len(uniq) == 2
            
            # Sample for curves (large datasets)
            idx = np.arange(len(y_true))
            cap = self.config.include_curves_sample_cap
            if cap and len(idx) > cap:
                rng = np.random.default_rng(42)
                idx = rng.choice(idx, size=cap, replace=False)
            yt = np.array(y_true)[idx]
            
            # ROC strategy
            strat = self.config.classification_roc_strategy
            if strat == "auto":
                strat = "binary" if is_binary else "ovr"
            
            if is_binary:
                # ─── Binary Classification ───
                pos = positive_label
                if pos is None:
                    try:
                        pos = 1 if 1 in uniq else max(uniq)
                    except Exception:
                        pos = uniq[-1]
                
                y_bin = (y_true == pos).astype(int)
                proba_pos = self._extract_positive_proba(y_proba, classes, pos)
                
                # AUC scores
                try:
                    metrics["roc_auc"] = roc_auc_score(y_bin, proba_pos, sample_weight=sw)
                except Exception:
                    pass
                
                try:
                    metrics["average_precision"] = average_precision_score(
                        y_bin, proba_pos, sample_weight=sw
                    )
                except Exception:
                    pass
                
                # Curves
                try:
                    fpr, tpr, roc_thr = roc_curve(yt, proba_pos[idx])
                    pr_p, pr_r, pr_thr = precision_recall_curve(yt, proba_pos[idx])
                    metrics["curves"] = {
                        "roc": {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "thresholds": roc_thr.tolist()
                        },
                        "pr": {
                            "precision": pr_p.tolist(),
                            "recall": pr_r.tolist(),
                            "thresholds": pr_thr.tolist()
                        }
                    }
                except Exception:
                    pass
                
                # KS Statistic
                if self.config.compute_ks:
                    try:
                        ks = self._compute_ks_stat(y_bin, proba_pos)
                        metrics["ks_stat"] = float(ks)
                    except Exception:
                        pass
                
                # Lift@k
                try:
                    lifts = self._compute_lift_at_k(y_bin, proba_pos, self.config.lift_k_list)
                    metrics["lift_at_k"] = {f"{k}%": float(v) for k, v in lifts.items()}
                except Exception:
                    pass
                
                # Threshold Optimization
                if self.config.optimize_thresholds:
                    try:
                        opt = self._optimize_threshold(
                            y_bin, proba_pos, grid_size=self.config.threshold_grid_size
                        )
                        metrics["threshold_optimization"] = opt
                    except Exception:
                        pass
            
            else:
                # ─── Multiclass Classification ───
                try:
                    metrics["roc_auc_ovr"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", sample_weight=sw
                    )
                except Exception:
                    pass
                
                try:
                    metrics["roc_auc_ovo"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovo", sample_weight=sw
                    )
                except Exception:
                    pass
        
        return metrics
    
        # ───────────────────────────────────────────────────────────────────
    # Regression Metrics
    # ───────────────────────────────────────────────────────────────────

    @_timeit("evaluate_regression")
    @_safe_operation("regression_metrics", default_value={})
    def _evaluate_regression(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sample_weight: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive regression metrics."""
        metrics: Dict[str, Any] = {}

        if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
            self._log.warning("Regression: missing y_true or y_pred; metrics limited")
            return metrics

        sw = np.asarray(sample_weight) if sample_weight is not None else None

        # ─── Basic Metrics ───
        try:
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred, sample_weight=sw))
        except Exception:
            pass

        try:
            mse = float(mean_squared_error(y_true, y_pred, sample_weight=sw))
            metrics["mse"] = mse
            metrics["rmse"] = float(np.sqrt(mse))
        except Exception:
            pass

        try:
            metrics["r2"] = float(r2_score(y_true, y_pred, sample_weight=sw))
        except Exception:
            pass

        try:
            metrics["median_ae"] = float(median_absolute_error(y_true, y_pred))
        except Exception:
            pass

        # ─── MAPE (Mask Zero Values) ───
        try:
            yt = pd.Series(y_true, dtype="float64")
            yp = pd.Series(y_pred, dtype="float64")
            mask = (yt != 0) & (~pd.isna(yt)) & (~pd.isna(yp))
            metrics["mape"] = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)
        except Exception:
            pass

        # ─── MSLE (Non-negative Only) ───
        try:
            if pd.Series(y_true).min() >= 0 and pd.Series(y_pred).min() >= 0:
                metrics["msle"] = float(mean_squared_log_error(y_true, y_pred, sample_weight=sw))
        except Exception:
            pass

        # ─── Residual Diagnostics ───
        try:
            resid = pd.Series(y_true, dtype="float64") - pd.Series(y_pred, dtype="float64")
            q1, q3 = np.quantile(resid, [0.25, 0.75])
            iqr = float(q3 - q1)
            out_hi = float(q3 + 1.5 * iqr)
            out_lo = float(q1 - 1.5 * iqr)
            out_share = float((np.sum((resid < out_lo) | (resid > out_hi)) / max(1, len(resid))) * 100.0)

            metrics["residuals"] = {
                "mean": float(np.mean(resid)),
                "median": float(np.median(resid)),
                "std": float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0,
                "q05": float(np.quantile(resid, 0.05)),
                "q50": float(np.quantile(resid, 0.50)),
                "q95": float(np.quantile(resid, 0.95)),
                "iqr": iqr,
                "outlier_bounds": {"low": out_lo, "high": out_hi},
                "outlier_share_pct": out_share,
            }
        except Exception:
            pass

        return metrics

# ───────────────────────────────────────────────────────────────────
# Helper Methods
# ───────────────────────────────────────────────────────────────────

def _extract_positive_proba(
    self,
    y_proba: np.ndarray,
    classes: Optional[np.ndarray],
    positive_label: Any
) -> np.ndarray:
    """Extract probability for positive class."""
    if y_proba.shape[1] == 1:
        return y_proba[:, 0]
    
    # Find positive class column
    if classes is not None:
        try:
            j = list(classes).index(positive_label)
            return y_proba[:, j]
        except Exception:
            pass
    
    # Fallback: column with highest mean
    j = int(np.argmax(np.nanmean(y_proba, axis=0)))
    return y_proba[:, j]

@_safe_operation("compute_ks_stat", default_value=0.0)
def _compute_ks_stat(
    self,
    y_true_bin: Union[pd.Series, np.ndarray],
    proba_pos: np.ndarray
) -> float:
    """Compute Kolmogorov-Smirnov statistic."""
    y = np.array(y_true_bin).astype(int)
    p = np.array(proba_pos).astype(float)
    order = np.argsort(p)
    y_sorted = y[order]
    
    cum_pos = np.cumsum(y_sorted) / max(1, y_sorted.sum())
    cum_neg = np.cumsum(1 - y_sorted) / max(1, (1 - y_sorted).sum())
    ks = float(np.max(np.abs(cum_pos - cum_neg)))
    
    return ks

@_safe_operation("compute_lift", default_value={})
def _compute_lift_at_k(
    self,
    y_true_bin: Union[pd.Series, np.ndarray],
    proba_pos: np.ndarray,
    k_list: Tuple[int, ...]
) -> Dict[int, float]:
    """Compute Lift@k for different percentiles."""
    y = np.array(y_true_bin).astype(int)
    p = np.array(proba_pos).astype(float)
    order = np.argsort(-p)
    
    lifts: Dict[int, float] = {}
    base_rate = float(np.mean(y)) if len(y) > 0 else 0.0
    
    for k in k_list:
        k = int(k)
        top_n = max(1, int(np.ceil(len(y) * (k / 100.0))))
        pos_in_top = np.sum(y[order][:top_n])
        rate_top = pos_in_top / top_n
        lifts[k] = (rate_top / base_rate) if base_rate > 0 else np.nan
    
    return lifts

@_safe_operation("optimize_threshold", default_value={})
def _optimize_threshold(
    self,
    y_true_bin: Union[pd.Series, np.ndarray],
    proba_pos: np.ndarray,
    grid_size: int = 101
) -> Dict[str, Any]:
    """Optimize decision threshold using multiple criteria."""
    y = np.array(y_true_bin).astype(int)
    p = np.array(proba_pos).astype(float)
    grid = np.linspace(0.0, 1.0, num=max(2, grid_size))
    
    best_f1, thr_f1 = -1.0, 0.5
    best_youden, thr_youden = -1.0, 0.5
    
    for t in grid:
        pred = (p >= t).astype(int)
        
        # F1 Score
        try:
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, thr_f1 = f1, t
        except Exception:
            pass
        
        # Youden's J Index (TPR - FPR)
        try:
            tp = ((pred == 1) & (y == 1)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            tn = ((pred == 0) & (y == 0)).sum()
            tpr = tp / max(1, (tp + fn))
            fpr = fp / max(1, (fp + tn))
            j = tpr - fpr
            if j > best_youden:
                best_youden, thr_youden = j, t
        except Exception:
            pass
    
    return {
        "best_f1": {"threshold": float(thr_f1), "score": float(best_f1)},
        "best_youden": {"threshold": float(thr_youden), "score": float(best_youden)},
    }

@_safe_operation("resolve_best_score", default_value=None)
def _resolve_best_score(
    self,
    metrics: Dict[str, Any],
    primary_metric: str
) -> Optional[float]:
    """Resolve best_score from primary metric."""
    if not metrics:
        return None
    
    # Metric aliases
    aliases = {
        # Classification
        "acc": "accuracy",
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "roc_auc": "roc_auc",
        "roc_auc_ovr": "roc_auc_ovr",
        "roc_auc_ovo": "roc_auc_ovo",
        "ap": "average_precision",
        "average_precision": "average_precision",
        # Regression
        "r2": "r2",
        "mae": "mae",
        "mse": "mse",
        "rmse": "rmse",
        "msle": "msle",
        "median_ae": "median_ae",
        "mape": "mape",
    }
    
    key = aliases.get(primary_metric.lower(), primary_metric) if isinstance(primary_metric, str) else None
    
    # Helper for nested paths
    def _get_nested(d: Dict[str, Any], path: str) -> Optional[float]:
        try:
            cur = d
            for part in path.split("."):
                cur = cur[part]
            if isinstance(cur, (int, float, np.floating)):
                return float(cur)
        except Exception:
            return None
        return None
    
    # Try direct or nested access
    if key:
        if "." in key:
            v = _get_nested(metrics, key)
            if v is not None:
                return v
        
        if key in metrics and isinstance(metrics[key], (int, float, np.floating)):
            return float(metrics[key])
    
    # Fallback cascade
    for cand in ["accuracy", "f1_weighted", "roc_auc", "roc_auc_ovr", "r2", "rmse", "mae"]:
        if cand in metrics and isinstance(metrics[cand], (int, float, np.floating)):
            return float(metrics[cand])
    
    return None

@_safe_operation("slice_predictions", default_value=None)
def _maybe_slice_predictions(
    self,
    preds_df: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Sample predictions if too large (stable sampling with fixed seed)."""
    cap = getattr(self.config, "sample_predictions_cap", None)
    if preds_df is None or cap is None:
        return preds_df

    if len(preds_df) <= cap:
        return preds_df

    # Stable sampling with seed
    rng = np.random.default_rng(42)
    idx = rng.choice(
        preds_df.index.to_numpy(),
        size=int(cap),
        replace=False
    )
    return preds_df.loc[np.sort(idx)]


# ───────────────────────────────────────────────────────────────────
# SECTION: Convenience Function
# ───────────────────────────────────────────────────────────────────

def evaluate_model(
    best_model: Any,
    problem_type: Literal["classification", "regression"],
    data: Optional[pd.DataFrame] = None,
    y_true: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
    pycaret_wrapper: Optional[Any] = None,
    config: Optional["EvalConfig"] = None,
    **kwargs: Any,
) -> "AgentResult":
    """
    Convenience function to evaluate a model.

    Example:
        from agents.ml import evaluate_model

        # Basic usage
        result = evaluate_model(
            best_model=model,
            problem_type="classification",
            data=X_test,
            y_true=y_test,
        )

        # Access metrics
        metrics = result.data["metrics"]
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_weighted']:.4f}")
        print(f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

        # With custom config
        config = EvalConfig(
            optimize_thresholds=True,
            compute_ks=True,
            include_per_class=True,
        )
        result = evaluate_model(
            best_model=model,
            problem_type="classification",
            data=X_test,
            y_true=y_test,
            config=config,
        )

    Args:
        best_model: Trained model to evaluate.
        problem_type: 'classification' or 'regression'.
        data: Test features.
        y_true: True labels/targets.
        pycaret_wrapper: Optional PyCaret wrapper.
        config: Optional custom configuration.
        **kwargs: Additional parameters passed to evaluator.

    Returns:
        AgentResult with comprehensive evaluation metrics.
    """
    evaluator = ModelEvaluator(config)
    return evaluator.execute(
        best_model=best_model,
        pycaret_wrapper=pycaret_wrapper,
        problem_type=problem_type,
        data=data,
        y_true=y_true,
        **kwargs,
    )
