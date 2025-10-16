# agents/data_understanding/problem_classifier.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Problem Classifier               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade ML problem type classification:                          ║
║    ✓ Automatic detection: classification vs regression                    ║
║    ✓ Defensive heuristics with correction logic                           ║
║    ✓ Target analysis (numeric, categorical, imbalance detection)          ║
║    ✓ Smart metrics & preprocessing recommendations                        ║
║    ✓ Model suggestions (baseline to advanced)                             ║
║    ✓ Stable 1:1 contract with orchestrator                                ║
║    ✓ Zero side-effects on input DataFrame                                 ║
║    ✓ Graceful error handling & comprehensive logging                      ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract (1:1 Orchestrator Alignment):
{
    "problem_type": "classification" | "regression" | None,
    "target_analysis": Dict[str, Any],  # Dynamic based on problem type
    "recommendations": {
        "problem_type": str | None,
        "suggested_metrics": List[str],
        "preprocessing_steps": List[str],
        "model_suggestions": List[str],
        "warnings": List[str],
    }
}
"""

from __future__ import annotations

import warnings
import time
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Literal
from functools import wraps

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
    from core.utils import infer_problem_type
    from config.model_registry import ProblemType
except ImportError:
    # Fallback for testing/standalone usage
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data = None
            self.errors = []
        
        def add_error(self, msg: str):
            self.errors.append(msg)
    
    def infer_problem_type(target):
        """Fallback heuristic."""
        nunique = target.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(target) and nunique > 15:
            return "regression"
        return "classification"
    
    class ProblemType:
        CLASSIFICATION = "classification"
        REGRESSION = "regression"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProblemClassifierConfig:
    """Enterprise configuration for problem classification heuristics."""
    
    # Imbalance & class rarity thresholds
    imbalance_ratio_threshold: float = 3.0
    warn_min_samples_per_class: int = 20
    rare_class_threshold: int = 5
    
    # Numeric & distribution analysis
    skew_warn_abs: float = 1.0
    small_unique_threshold: int = 15
    id_like_unique_ratio: float = 0.98
    
    # Data quality thresholds
    min_samples_required: int = 1
    treat_numeric_str_as_numeric: bool = True
    
    # Logging & formatting
    truncate_log_chars: int = 400
    
    # Fallback strategy
    allow_none_problem_fallback: bool = False
    
    # Binary classification hints (internal heuristics)
    binary_positive_hint_keywords: Tuple[str, ...] = field(
        default_factory=lambda: ("yes", "true", "1", "success", "positive", "win")
    )


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


def _safe_json_str(obj: Any, limit: int = 400) -> str:
    """Safely convert object to truncated JSON string for logging."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    
    if len(s) <= limit:
        return s
    return s[:limit] + f"...(+{len(s)-limit} chars)"


def _is_numeric_series_like(s: pd.Series, sample_size: int = 1000) -> bool:
    """
    Heuristic: Can series be treated as numeric after coercion?
    
    Checks:
      • bool dtype → numeric
      • numeric dtype → numeric
      • object dtype with >90% numeric-like values after parsing
    """
    try:
        if pd.api.types.is_bool_dtype(s):
            return True
        
        if pd.api.types.is_numeric_dtype(s):
            return True
        
        if s.dtype == "object":
            sample = s.dropna().astype(str).head(sample_size)
            if sample.empty:
                return False
            
            # Try numeric coercion on sample
            parsed = pd.to_numeric(
                sample.str.replace(",", ".", regex=False),
                errors="coerce"
            )
            success_ratio = parsed.notna().mean()
            return float(success_ratio) > 0.9
        
        return False
    
    except Exception as e:
        logger.debug(f"numeric_series_like check failed: {e}")
        return False


def _is_id_like(s: pd.Series, ratio_threshold: float = 0.98) -> bool:
    """Detect ID-like columns (almost all unique values)."""
    try:
        n_unique = int(s.nunique(dropna=True))
        n_total = int(len(s))
        
        if n_total <= 0:
            return False
        
        return (n_unique / n_total) >= ratio_threshold
    
    except Exception:
        return False


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Safely coerce series to numeric (handles bool, decimals with commas)."""
    try:
        if pd.api.types.is_bool_dtype(s):
            return s.astype(int).astype(float)
        
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        
        # Try string coercion with comma handling
        return pd.to_numeric(
            s.astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
    
    except Exception as e:
        logger.debug(f"numeric coercion failed: {e}")
        return s


def _extract_majority_label(vc: pd.Series) -> Tuple[Optional[str], float]:
    """
    Extract majority class label and percentage.
    
    Returns:
        Tuple[label_str, percentage_float] or (None, 0.0) if empty
    """
    if vc.empty:
        return None, 0.0
    
    total = int(vc.sum())
    majority_label = str(vc.index[0])
    majority_pct = float(vc.iloc[0] / max(1, total) * 100.0)
    
    return majority_label, majority_pct


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Classifier Agent
# ═══════════════════════════════════════════════════════════════════════════

class ProblemClassifier(BaseAgent):
    """
    **ProblemClassifier** — Enterprise ML problem type classification.
    
    Responsibilities:
      1. Detect problem type (classification vs regression) from target
      2. Perform defensive heuristic analysis with correction logic
      3. Analyze target characteristics (imbalance, distribution, cardinality)
      4. Generate actionable recommendations (metrics, preprocessing, models)
      5. Maintain 1:1 contract with orchestrator
      6. Zero side-effects on input DataFrame
      7. Comprehensive error handling & logging
    
    Output format is stable and deterministic.
    """
    
    def __init__(self, config: Optional[ProblemClassifierConfig] = None) -> None:
        """Initialize classifier with optional custom configuration."""
        super().__init__(
            name="ProblemClassifier",
            description="Classifies ML problem type and provides recommendations"
        )
        self.config = config or ProblemClassifierConfig()
        self._log = logger.bind(agent="ProblemClassifier")
        warnings.filterwarnings("ignore")
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            data: pd.DataFrame
            target_column: str
        
        Raises:
            ValueError, TypeError on validation failure
        """
        if "data" not in kwargs:
            raise ValueError("Required parameter 'data' not provided")
        
        if "target_column" not in kwargs:
            raise ValueError("Required parameter 'target_column' not provided")
        
        if not isinstance(kwargs["data"], pd.DataFrame):
            raise TypeError(f"'data' must be pd.DataFrame, got {type(kwargs['data']).__name__}")
        
        if not isinstance(kwargs["target_column"], str):
            raise TypeError(f"'target_column' must be str, got {type(kwargs['target_column']).__name__}")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Public Interface)
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        **kwargs: Any
    ) -> AgentResult:
        """
        Classify problem type and generate recommendations.
        
        Args:
            data: Input DataFrame (not modified)
            target_column: Name of target column
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with classification payload (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t_start = time.perf_counter()
        
        try:
            # Input validation
            if target_column not in data.columns:
                msg = f"Target column '{target_column}' not found in DataFrame"
                result.add_error(msg)
                self._log.error(msg)
                return result
            
            # Prep data (replace inf with NaN, don't modify original)
            df_safe = data.replace([np.inf, -np.inf], np.nan).copy(deep=False)
            target = df_safe[target_column]
            
            # Guard: empty target
            if target is None or len(target) == 0 or target.dropna().empty:
                self._log.warning("⚠ Empty or all-null target column")
                result.data = self._empty_payload_for_null_target(target)
                return result
            
            # 1. Detect problem type (with infer_problem_type + fallback)
            problem_type_str = self._detect_problem_type(target)
            
            # 2. Analyze target
            target_analysis = self._analyze_target(target, problem_type_str)
            
            # 3. Generate recommendations
            recommendations = self._get_recommendations(problem_type_str, target_analysis)
            
            # 4. Build result (strict 1:1 contract)
            result.data = {
                "problem_type": problem_type_str,
                "target_analysis": target_analysis,
                "recommendations": recommendations,
            }
            
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            self._log.success(
                f"✓ Classified as '{problem_type_str}' in {elapsed_ms:.1f}ms | "
                f"n_samples={len(target)} n_unique={target.nunique(dropna=True)}"
            )
        
        except Exception as e:
            msg = f"Problem classification failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            result.data = self._empty_payload_for_null_target(None)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Problem Type Detection (Multi-Stage Heuristic)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("problem_type_detection")
    def _detect_problem_type(self, target: pd.Series) -> Optional[str]:
        """
        Multi-stage problem type detection with defensive fallback.
        
        Process:
          1. Try utility function infer_problem_type()
          2. Apply heuristic corrections (cardinality, dtype)
          3. Fallback strategies if unclear
        
        Returns:
            "classification" | "regression" | None
        """
        cfg = self.config
        
        # Stage 1: Utility function
        detected = None
        try:
            detected = infer_problem_type(target)
        except Exception as e:
            self._log.debug(f"infer_problem_type() failed: {e}")
        
        # Convert to string
        if isinstance(detected, ProblemType):
            problem_type = "classification" if detected == ProblemType.CLASSIFICATION else "regression"
        elif detected is not None:
            problem_type = str(detected).lower().strip()
        else:
            problem_type = None
        
        # Stage 2: Heuristic corrections
        if problem_type not in ("classification", "regression"):
            # Correction: numeric/bool with few unique values → classification
            if pd.api.types.is_bool_dtype(target) or pd.api.types.is_numeric_dtype(target):
                nunique = int(target.nunique(dropna=True))
                if nunique <= cfg.small_unique_threshold:
                    problem_type = "classification"
                    self._log.debug(f"heuristic: numeric with {nunique} unique values → classification")
            
            # Correction: object/numeric-like/ID-like → decide by cardinality
            elif _is_numeric_series_like(target) or _is_id_like(target, cfg.id_like_unique_ratio):
                nunique = int(target.nunique(dropna=True))
                problem_type = (
                    "classification" if nunique <= cfg.small_unique_threshold
                    else "regression"
                )
                self._log.debug(f"heuristic: numeric-like with {nunique} unique → {problem_type}")
        
        # Stage 3: Final fallback
        if problem_type not in ("classification", "regression"):
            if not cfg.allow_none_problem_fallback:
                # Safe default: regression for numeric-like, else classification
                problem_type = "regression" if _is_numeric_series_like(target) else "classification"
                self._log.debug(f"fallback: defaulting to '{problem_type}'")
            else:
                problem_type = None
                self._log.warning("⚠ Unable to determine problem type; returning None")
        
        return problem_type
    
    # ───────────────────────────────────────────────────────────────────
    # Target Analysis (Dynamic Based on Problem Type)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("target_analysis")
    def _analyze_target(self, target: pd.Series, problem_type: Optional[str]) -> Dict[str, Any]:
        """
        Analyze target column characteristics.
        
        Returns structure depends on problem_type:
          • classification: class_distribution, imbalance, rare classes
          • regression: mean, std, skewness, kurtosis, quantiles
        """
        n_samples = int(len(target))
        n_unique = int(target.nunique(dropna=True))
        n_missing = int(target.isna().sum())
        missing_pct = float((n_missing / max(1, n_samples)) * 100.0)
        
        base_analysis = {
            "n_samples": n_samples,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "missing_pct": missing_pct,
            "dtype": str(target.dtype),
        }
        
        # Type-specific analysis
        if problem_type == "classification":
            base_analysis.update(self._analyze_classification_target(target))
        elif problem_type == "regression":
            base_analysis.update(self._analyze_regression_target(target))
        else:
            base_analysis["note"] = "Unable to classify problem type"
        
        return base_analysis
    
    def _analyze_classification_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze classification target: classes, distribution, imbalance."""
        cfg = self.config
        
        vc = target.dropna().value_counts()
        n_classes = int(len(vc))
        warnings_list: List[str] = []
        
        # Basic classification type
        if n_classes <= 1:
            classification_type = "single_class"
            imbalance_ratio = 1.0
            is_imbalanced = False
            warnings_list.append("Only 1 unique class after removing NaN — model training not feasible")
        
        elif n_classes == 2:
            classification_type = "binary"
            imbalance_ratio = float(vc.iloc[0] / vc.iloc[1]) if vc.iloc[1] > 0 else float("inf")
            is_imbalanced = imbalance_ratio > cfg.imbalance_ratio_threshold
        
        else:
            classification_type = "multiclass"
            imbalance_ratio = float(vc.max() / vc.min()) if vc.min() > 0 else float("inf")
            is_imbalanced = imbalance_ratio > cfg.imbalance_ratio_threshold
        
        # Imbalance warnings
        if is_imbalanced:
            warnings_list.append(
                f"Class imbalance: ratio {imbalance_ratio:.2f} (threshold: {cfg.imbalance_ratio_threshold})"
            )
        
        # Rare classes
        rare_classes = [str(k) for k, v in vc.items() if v < cfg.warn_min_samples_per_class]
        if rare_classes:
            warnings_list.append(f"Rare classes: {rare_classes} (<{cfg.warn_min_samples_per_class} samples)")
        
        very_rare = [str(k) for k, v in vc.items() if v < cfg.rare_class_threshold]
        if very_rare:
            warnings_list.append(f"Very rare classes: {very_rare} (<{cfg.rare_class_threshold} samples)")
        
        # Majority class
        maj_label, maj_pct = _extract_majority_label(vc)
        
        return {
            "classification_type": classification_type,
            "n_classes": n_classes,
            "class_distribution": {str(k): int(v) for k, v in vc.to_dict().items()},
            "is_imbalanced": bool(is_imbalanced),
            "imbalance_ratio": round(imbalance_ratio, 3) if imbalance_ratio != float("inf") else None,
            "majority_class": maj_label,
            "majority_class_pct": round(maj_pct, 2),
            "warnings": warnings_list,
        }
    
    def _analyze_regression_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze regression target: statistics, distribution."""
        cfg = self.config
        target_clean = target.dropna()
        
        if len(target_clean) < cfg.min_samples_required:
            return {
                "error": "Insufficient non-missing values for regression analysis",
                "n_valid_samples": int(len(target_clean)),
            }
        
        # Coerce to numeric if configured
        t = target_clean.copy()
        if cfg.treat_numeric_str_as_numeric and not (
            pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t)
        ):
            t = _coerce_numeric_series(t)
        
        # Final numeric check
        if not (pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t)):
            try:
                t = pd.to_numeric(t, errors="coerce").dropna()
            except Exception:
                pass
        
        if t.empty:
            return {
                "error": "Target not numeric and cannot be coerced",
                "n_valid_samples": int(len(target_clean)),
            }
        
        # Convert bool to int for stats
        if pd.api.types.is_bool_dtype(t):
            t = t.astype(int)
        
        t = t.astype(float)
        
        # Compute statistics
        mean = float(t.mean())
        std = float(t.std(ddof=1)) if len(t) > 1 else 0.0
        cv = float(std / mean) if mean != 0 else None
        skewness = float(t.skew()) if len(t) > 2 else 0.0
        kurtosis = float(t.kurtosis()) if len(t) > 3 else 0.0
        
        return {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(float(t.min()), 6),
            "max": round(float(t.max()), 6),
            "median": round(float(t.median()), 6),
            "q25": round(float(t.quantile(0.25)), 6),
            "q75": round(float(t.quantile(0.75)), 6),
            "iqr": round(float(t.quantile(0.75) - t.quantile(0.25)), 6),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "range": round(float(t.max() - t.min()), 6),
            "cv": round(cv, 3) if cv is not None else None,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Recommendations Generation (1:1 Contract)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("recommendations_generation")
    def _get_recommendations(
        self,
        problem_type: Optional[str],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate recommendations based on problem type and analysis.
        
        Returns:
            Dict with metrics, preprocessing, models, warnings (strict contract)
        """
        if problem_type == "classification":
            return self._recommendations_classification(analysis)
        elif problem_type == "regression":
            return self._recommendations_regression(analysis)
        else:
            return {
                "problem_type": None,
                "suggested_metrics": [],
                "preprocessing_steps": [],
                "model_suggestions": [],
                "warnings": ["Unable to classify problem type"],
            }
    
    def _recommendations_classification(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate classification recommendations."""
        cfg = self.config
        n_classes = int(analysis.get("n_classes", 0))
        is_imbalanced = bool(analysis.get("is_imbalanced", False))
        warnings_local = list(analysis.get("warnings", []))
        
        # Metrics
        if n_classes <= 1:
            metrics = []
            warnings_local.append("Single-class target: classification not feasible")
        
        elif n_classes == 2:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        
        else:  # multiclass
            metrics = ["accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted"]
        
        # Add imbalance metrics
        if is_imbalanced and metrics:
            for m in ["balanced_accuracy", "mcc"]:
                if m not in metrics:
                    metrics.append(m)
        
        # Preprocessing steps
        preprocessing = [
            "Handle missing values in target (if any)",
            "Encode categorical target if needed",
            "Apply stratified train/validation split",
        ]
        
        if is_imbalanced:
            preprocessing.extend([
                "Apply SMOTE/ADASYN for minority class oversampling",
                "Consider class weights or sample weights",
                "Tune decision threshold using PR/ROC curves",
            ])
        
        # Model suggestions
        if n_classes <= 1:
            models = []
        elif n_classes == 2:
            models = [
                "Logistic Regression (baseline)",
                "Random Forest Classifier",
                "XGBoost Classifier",
                "LightGBM Classifier",
                "CatBoost Classifier",
                "SVM (linear/RBF after feature scaling)",
                "Neural Network (shallow → deep)",
            ]
        else:  # multiclass
            models = [
                "Random Forest Classifier",
                "XGBoost Classifier (multi:softmax)",
                "LightGBM Classifier",
                "CatBoost Classifier",
                "SVM One-vs-Rest (after feature scaling)",
                "Neural Network",
            ]
        
        return {
            "problem_type": "classification",
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
            "warnings": warnings_local,
        }
    
    def _recommendations_regression(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate regression recommendations."""
        cfg = self.config
        warnings_local: List[str] = []
        
        # Check for errors
        if "error" in analysis:
            warnings_local.append(f"Target analysis error: {analysis['error']}")
            return {
                "problem_type": "regression",
                "suggested_metrics": [],
                "preprocessing_steps": ["Inspect and repair target column"],
                "model_suggestions": [],
                "warnings": warnings_local,
            }
        
        # Metrics
        metrics = ["mae", "mse", "rmse", "r2", "mape"]
        
        # Preprocessing
        preprocessing = [
            "Handle missing values in target",
            "Check for and handle outliers (IQR/isolation forest)",
            "Scale features for linear/SVM/NN models",
        ]
        
        # Skewness-based transformation hint
        skewness = float(analysis.get("skewness", 0.0))
        if abs(skewness) > cfg.skew_warn_abs:
            preprocessing.append(
                f"Consider log/Box-Cox/Yeo-Johnson transformation (|skew|={abs(skewness):.2f})"
            )
            warnings_local.append(
                f"Target skewness {skewness:.2f}: consider target transformation"
            )
        
        # Models
        models = [
            "Linear Regression (baseline)",
            "Ridge / Lasso / ElasticNet (regularization)",
            "Random Forest Regressor",
            "XGBoost Regressor",
            "LightGBM Regressor",
            "CatBoost Regressor",
            "SVR (for smaller datasets, after scaling)",
            "Neural Network (shallow → deep)",
        ]
        
        return {
            "problem_type": "regression",
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
            "warnings": warnings_local,
        }
    
    # ───────────────────────────────────────────────────────────────────
    # Fallback Payloads
    # ───────────────────────────────────────────────────────────────────
    
    def _empty_payload_for_null_target(self, target: Optional[pd.Series]) -> Dict[str, Any]:
        """Generate empty payload for null/empty target."""
        n_samples = int(len(target)) if target is not None else 0
        n_missing = int(target.isna().sum()) if target is not None else n_samples
        n_unique = int(target.nunique(dropna=True)) if target is not None else 0
        
        return {
            "problem_type": None,
            "target_analysis": {
                "n_samples": n_samples,
                "n_unique": n_unique,
                "n_missing": n_missing,
                "missing_pct": float(n_missing / max(1, n_samples) * 100),
                "error": "Target is empty or all values are missing",
            },
            "recommendations": {
                "problem_type": None,
                "suggested_metrics": [],
                "preprocessing_steps": ["Provide/repair target column with valid values"],
                "model_suggestions": [],
                "warnings": ["Cannot infer problem type from empty target"],
            },
        }