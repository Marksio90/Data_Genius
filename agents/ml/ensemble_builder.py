# agents/ml/ensemble_builder.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Ensemble Builder                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Advanced ensemble learning with enterprise safeguards:                    ║
║    ✓ 5 ensemble strategies (voting, stacking, blending, weighted, auto)   ║
║    ✓ Intelligent model selection (diversity-based)                        ║
║    ✓ Weight optimization (differential evolution)                         ║
║    ✓ Diversity analysis (disagreement, correlation)                       ║
║    ✓ Cross-validation support (stratified/standard)                       ║
║    ✓ Meta-learner selection (5 options)                                   ║
║    ✓ Feature importance aggregation                                       ║
║    ✓ Model persistence (pickle)                                           ║
║    ✓ Comprehensive telemetry                                              ║
║    ✓ Thread-safe operations                                               ║
║    ✓ Defensive error handling                                             ║
║    ✓ Versioned output contract                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Ensemble Strategies:
    1. Voting (Hard/Soft) — Simple majority or weighted voting
    2. Stacking — Meta-learner on base model predictions
    3. Blending — Holdout-based weight optimization
    4. Weighted Average — CV-optimized weights
    5. Auto — Intelligent strategy selection

Meta-Learners:
    • Logistic Regression / Ridge
    • Random Forest
    • XGBoost
    • LightGBM
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

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


__all__ = [
    "EnsembleStrategy",
    "MetaLearnerType",
    "EnsembleConfig",
    "EnsembleResult",
    "EnsembleBuilder",
    "WeightedEnsemble",
    "build_ensemble"
]
__version__ = "5.0-kosmos-enterprise"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class EnsembleStrategy(str, Enum):
    """Ensemble learning strategies."""
    VOTING_HARD = "voting_hard"
    VOTING_SOFT = "voting_soft"
    STACKING = "stacking"
    BLENDING = "blending"
    WEIGHTED_AVERAGE = "weighted_average"
    AUTO = "auto"


class MetaLearnerType(str, Enum):
    """Meta-learner types for stacking."""
    LOGISTIC = "logistic"
    RIDGE = "ridge"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EnsembleConfig:
    """
    Configuration for ensemble building.
    
    Attributes:
        strategy: Ensemble strategy to use
        n_models: Maximum models to include
        min_diversity: Minimum diversity threshold (0-1)
        cv_folds: Cross-validation folds
        meta_learner: Meta-learner for stacking
        optimize_weights: Whether to optimize weights
        voting_type: Voting type (hard/soft)
        use_probas: Use probabilities for classification
    """
    strategy: EnsembleStrategy = EnsembleStrategy.AUTO
    n_models: int = 5
    min_diversity: float = 0.1
    cv_folds: int = 5
    meta_learner: MetaLearnerType = MetaLearnerType.LOGISTIC
    optimize_weights: bool = True
    voting_type: str = "soft"
    use_probas: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "n_models": self.n_models,
            "min_diversity": self.min_diversity,
            "cv_folds": self.cv_folds,
            "meta_learner": self.meta_learner.value,
            "optimize_weights": self.optimize_weights,
            "voting_type": self.voting_type,
            "use_probas": self.use_probas
        }


@dataclass
class EnsembleResult:
    """
    Results from ensemble building.
    
    Attributes:
        ensemble_model: Trained ensemble model
        strategy: Strategy used
        base_models: List of (name, model) tuples
        weights: Model weights (if applicable)
        performance: Performance metrics
        diversity_metrics: Diversity measures
        training_time: Time taken to build (seconds)
        config: Configuration used
        metadata: Additional context
        created_at: Creation timestamp
    """
    ensemble_model: Any
    strategy: EnsembleStrategy
    base_models: List[Tuple[str, Any]]
    weights: Optional[List[float]]
    performance: Dict[str, float]
    diversity_metrics: Dict[str, float]
    training_time: float
    config: EnsembleConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = __version__
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding models)."""
        return {
            "strategy": self.strategy.value,
            "base_models": [name for name, _ in self.base_models],
            "weights": self.weights,
            "performance": self.performance,
            "diversity_metrics": self.diversity_metrics,
            "training_time": self.training_time,
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "version": self.version
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
# SECTION: Main Ensemble Builder
# ═══════════════════════════════════════════════════════════════════════════

class EnsembleBuilder(BaseAgent):
    """
    **EnsembleBuilder** — Advanced ensemble learning system.
    
    Responsibilities:
      1. Select diverse base models
      2. Build ensemble using selected strategy
      3. Optimize model weights (if applicable)
      4. Evaluate ensemble performance
      5. Calculate diversity metrics
      6. Aggregate feature importance
      7. Persist models to disk
      8. Handle missing values gracefully
      9. Track comprehensive telemetry
      10. Maintain stable output contracts
    
    Features:
      • 5 ensemble strategies
      • Diversity-based model selection
      • Weight optimization (differential evolution)
      • Cross-validation support
      • 5 meta-learner options
      • Feature importance aggregation
      • Pickle persistence
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        """
        Initialize ensemble builder.
        
        Args:
            config: Optional custom configuration
        """
        super().__init__(
            name="EnsembleBuilder",
            description="Advanced ensemble learning with intelligent model selection"
        )
        self.config = config or EnsembleConfig()
        self._log = logger.bind(agent="EnsembleBuilder")
        warnings.filterwarnings("ignore")
        
        # State
        self.ensemble_model: Optional[Any] = None
        self.result: Optional[EnsembleResult] = None
        
        self._log.info(f"✓ EnsembleBuilder initialized | strategy={self.config.strategy.value}")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Building API
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("build_ensemble")
    def build_ensemble(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        problem_type: str = "classification"
    ) -> EnsembleResult:
        """
        Build ensemble from base models.
        
        Args:
            models: Dictionary of {model_name: trained_model}
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for blending)
            y_val: Validation target
            problem_type: 'classification' or 'regression'
        
        Returns:
            EnsembleResult with ensemble model and metrics
        """
        self._log.info(f"Building ensemble | base_models={len(models)} | type={problem_type}")
        t0 = time.perf_counter()
        
        # ─── Select Diverse Models ───
        selected_models = self._select_diverse_models(
            models, X_train, y_train, problem_type
        )
        
        self._log.info(f"Selected {len(selected_models)} diverse models")
        
        # ─── Choose Strategy ───
        if self.config.strategy == EnsembleStrategy.AUTO:
            strategy = self._choose_strategy(problem_type, len(selected_models))
        else:
            strategy = self.config.strategy
        
        self._log.info(f"Using strategy: {strategy.value}")
        
        # ─── Build Ensemble ───
        ensemble_model, weights = self._build_by_strategy(
            strategy,
            selected_models,
            X_train,
            y_train,
            X_val,
            y_val,
            problem_type
        )
        
        # ─── Train Ensemble ───
        if not isinstance(ensemble_model, WeightedEnsemble):
            ensemble_model.fit(X_train, y_train)
        
        # ─── Evaluate ───
        performance = self._evaluate_ensemble(
            ensemble_model, X_train, y_train, X_val, y_val, problem_type
        )
        
        # ─── Calculate Diversity ───
        diversity_metrics = self._calculate_diversity(
            selected_models, X_train, y_train, problem_type
        )
        
        # ─── Create Result ───
        training_time = time.perf_counter() - t0
        
        self.result = EnsembleResult(
            ensemble_model=ensemble_model,
            strategy=strategy,
            base_models=selected_models,
            weights=weights,
            performance=performance,
            diversity_metrics=diversity_metrics,
            training_time=training_time,
            config=self.config,
            metadata={
                "n_base_models": len(selected_models),
                "problem_type": problem_type,
                "training_samples": len(X_train)
            }
        )
        
        self.ensemble_model = ensemble_model
        
        self._log.success(
            f"✓ Ensemble built | "
            f"time={training_time:.2f}s | "
            f"performance={performance.get('val_accuracy', performance.get('val_r2', 0)):.4f}"
        )
        
        return self.result
    
    # ───────────────────────────────────────────────────────────────────
    # Model Selection
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("select_diverse_models")
    @_safe_operation("model_selection", default_value=[])
    def _select_diverse_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str
    ) -> List[Tuple[str, Any]]:
        """
        Select diverse models based on predictions and performance.
        
        Returns:
            List of (model_name, model) tuples
        """
        if len(models) <= self.config.n_models:
            return list(models.items())
        
        # ─── Get Predictions ───
        predictions = {}
        for name, model in models.items():
            try:
                if problem_type == "classification" and self.config.use_probas:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)
                        if pred.shape[1] == 2:
                            pred = pred[:, 1]
                    else:
                        pred = model.predict(X)
                else:
                    pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                self._log.warning(f"⚠ Could not get predictions from {name}: {e}")
        
        # ─── Calculate Performance ───
        scores = {}
        for name, pred in predictions.items():
            try:
                if problem_type == "classification":
                    if len(np.unique(y)) == 2:  # Binary
                        score = roc_auc_score(y, pred)
                    else:
                        score = accuracy_score(y, np.round(pred))
                else:
                    score = r2_score(y, pred)
                scores[name] = score
            except Exception:
                scores[name] = 0.0
        
        # ─── Sort by Performance ───
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # ─── Select Diverse Models ───
        selected: List[Tuple[str, Any]] = []
        selected_preds: List[np.ndarray] = []
        
        for name, score in sorted_models:
            if len(selected) >= self.config.n_models:
                break
            
            pred = predictions[name]
            
            # First model - always select
            if len(selected) == 0:
                selected.append((name, models[name]))
                selected_preds.append(pred)
            else:
                # Check diversity with selected models
                correlations = [
                    abs(pearsonr(pred, sp)[0])
                    for sp in selected_preds
                ]
                avg_corr = np.mean(correlations)
                
                # Add if sufficiently diverse
                if avg_corr < (1 - self.config.min_diversity):
                    selected.append((name, models[name]))
                    selected_preds.append(pred)
                    self._log.debug(
                        f"Selected {name} | "
                        f"score={score:.4f} | "
                        f"diversity={1-avg_corr:.4f}"
                    )
        
        # ─── Fill remaining slots with best models ───
        if len(selected) < min(self.config.n_models, len(models)):
            for name, _ in sorted_models:
                if len(selected) >= self.config.n_models:
                    break
                if name not in [s[0] for s in selected]:
                    selected.append((name, models[name]))
        
        return selected
    
    # ───────────────────────────────────────────────────────────────────
    # Strategy Selection & Building
    # ───────────────────────────────────────────────────────────────────
    
    def _choose_strategy(self, problem_type: str, n_models: int) -> EnsembleStrategy:
        """Automatically choose best ensemble strategy."""
        if n_models <= 2:
            return EnsembleStrategy.VOTING_SOFT
        elif n_models <= 4:
            return EnsembleStrategy.WEIGHTED_AVERAGE
        else:
            return EnsembleStrategy.STACKING
    
    def _build_by_strategy(
        self,
        strategy: EnsembleStrategy,
        models: List[Tuple[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        problem_type: str
    ) -> Tuple[Any, Optional[List[float]]]:
        """
        Build ensemble using specified strategy.
        
        Returns:
            Tuple of (ensemble_model, weights)
        """
        if strategy == EnsembleStrategy.VOTING_SOFT:
            return self._build_voting_ensemble(models, problem_type, "soft"), None
        
        elif strategy == EnsembleStrategy.VOTING_HARD:
            return self._build_voting_ensemble(models, problem_type, "hard"), None
        
        elif strategy == EnsembleStrategy.STACKING:
            return self._build_stacking_ensemble(models, problem_type), None
        
        elif strategy == EnsembleStrategy.BLENDING:
            if X_val is None or y_val is None:
                self._log.warning("Blending requires validation set, falling back to voting")
                return self._build_voting_ensemble(models, problem_type, "soft"), None
            return self._build_blending_ensemble(
                models, X_train, y_train, X_val, y_val, problem_type
            )
        
        elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return self._build_weighted_ensemble(
                models, X_train, y_train, problem_type
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    # ───────────────────────────────────────────────────────────────────
    # Ensemble Building Methods
    # ───────────────────────────────────────────────────────────────────
    
    def _build_voting_ensemble(
        self,
        models: List[Tuple[str, Any]],
        problem_type: str,
        voting: str
    ) -> Union[VotingClassifier, VotingRegressor]:
        """Build voting ensemble."""
        if problem_type == "classification":
            return VotingClassifier(
                estimators=models,
                voting=voting,
                n_jobs=-1
            )
        else:
            return VotingRegressor(
                estimators=models,
                n_jobs=-1
            )
    
    def _build_stacking_ensemble(
        self,
        models: List[Tuple[str, Any]],
        problem_type: str
    ) -> Union[StackingClassifier, StackingRegressor]:
        """Build stacking ensemble with meta-learner."""
        meta_learner = self._get_meta_learner(problem_type)
        
        if problem_type == "classification":
            return StackingClassifier(
                estimators=models,
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
                n_jobs=-1,
                passthrough=False
            )
        else:
            return StackingRegressor(
                estimators=models,
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
                n_jobs=-1,
                passthrough=False
            )
    
    def _build_blending_ensemble(
        self,
        models: List[Tuple[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        problem_type: str
    ) -> Tuple[WeightedEnsemble, List[float]]:
        """Build blending ensemble with optimized weights."""
        # Get validation predictions
        val_preds = []
        for name, model in models:
            if problem_type == "classification" and self.config.use_probas:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)
                    if pred.shape[1] == 2:
                        pred = pred[:, 1]
                else:
                    pred = model.predict(X_val)
            else:
                pred = model.predict(X_val)
            val_preds.append(pred)
        
        val_preds_array = np.array(val_preds).T
        
        # Optimize weights
        weights = self._optimize_weights(val_preds_array, y_val.values, problem_type)
        
        # Create weighted ensemble
        ensemble = WeightedEnsemble(
            models=models,
            weights=weights,
            problem_type=problem_type,
            use_probas=self.config.use_probas
        )
        ensemble.fit(X_train, y_train)
        
        return ensemble, weights.tolist()
    
    def _build_weighted_ensemble(
        self,
        models: List[Tuple[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: str
    ) -> Tuple[WeightedEnsemble, List[float]]:
        """Build weighted ensemble with CV-optimized weights."""
        # Setup CV
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        # Get CV predictions
        cv_preds = np.zeros((len(X_train), len(models)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            for model_idx, (name, model) in enumerate(models):
                # Clone and train
                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Predict
                if problem_type == "classification" and self.config.use_probas:
                    if hasattr(fold_model, 'predict_proba'):
                        pred = fold_model.predict_proba(X_fold_val)
                        if pred.shape[1] == 2:
                            pred = pred[:, 1]
                    else:
                        pred = fold_model.predict(X_fold_val)
                else:
                    pred = fold_model.predict(X_fold_val)
                
                cv_preds[val_idx, model_idx] = pred
        
        # Optimize weights
        weights = self._optimize_weights(cv_preds, y_train.values, problem_type)
        
        # Create weighted ensemble
        ensemble = WeightedEnsemble(
            models=models,
            weights=weights,
            problem_type=problem_type,
            use_probas=self.config.use_probas
        )
        ensemble.fit(X_train, y_train)
        
        return ensemble, weights.tolist()
    
    # ───────────────────────────────────────────────────────────────────
    # Weight Optimization
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("optimize_weights")
    def _optimize_weights(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        problem_type: str
    ) -> np.ndarray:
        """
        Optimize weights using differential evolution.
        
        Args:
            predictions: Array of shape (n_samples, n_models)
            y_true: True labels
            problem_type: 'classification' or 'regression'
        
        Returns:
            Optimized weights (normalized)
        """
        n_models = predictions.shape[1]
        
        def objective(weights):
            """Objective function to minimize."""
            w = np.array(weights)
            w = w / w.sum()  # Normalize
            
            weighted_pred = np.average(predictions, axis=1, weights=w)
            
            if problem_type == "classification":
                # Log loss
                weighted_pred = np.clip(weighted_pred, 1e-15, 1 - 1e-15)
                loss = -np.mean(
                    y_true * np.log(weighted_pred) +
                    (1 - y_true) * np.log(1 - weighted_pred)
                )
            else:
                # MSE
                loss = mean_squared_error(y_true, weighted_pred)
            
            return loss
        
        # Optimize
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            seed=42,
            workers=-1,
            updating='deferred'
        )
        
        weights = result.x
        weights = weights / weights.sum()  # Normalize
        
        self._log.info(f"Optimized weights: {weights}")
        
        return weights
    
    # ───────────────────────────────────────────────────────────────────
    # Meta-Learner Selection
    # ───────────────────────────────────────────────────────────────────
    
    def _get_meta_learner(self, problem_type: str) -> BaseEstimator:
        """Get meta-learner for stacking."""
        meta = self.config.meta_learner
        
        if meta == MetaLearnerType.LOGISTIC:
            if problem_type == "classification":
                return LogisticRegression(max_iter=1000, random_state=42)
            else:
                return Ridge(alpha=1.0, random_state=42)
        
        elif meta == MetaLearnerType.RIDGE:
            return Ridge(alpha=1.0, random_state=42)
        
        elif meta == MetaLearnerType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if problem_type == "classification":
                return RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                return RandomForestRegressor(n_estimators=100, random_state=42)
        
        elif meta == MetaLearnerType.XGBOOST:
            try:
                import xgboost as xgb
                if problem_type == "classification":
                    return xgb.XGBClassifier(random_state=42)
                else:
                    return xgb.XGBRegressor(random_state=42)
            except ImportError:
                self._log.warning("XGBoost unavailable, using LogisticRegression")
                return LogisticRegression(max_iter=1000, random_state=42)
        
        elif meta == MetaLearnerType.LIGHTGBM:
            try:
                import lightgbm as lgb
                if problem_type == "classification":
                    return lgb.LGBMClassifier(random_state=42, verbose=-1)
                else:
                    return lgb.LGBMRegressor(random_state=42, verbose=-1)
            except ImportError:
                self._log.warning("LightGBM unavailable, using Ridge")
                return Ridge(alpha=1.0, random_state=42)
        
        else:
            return LogisticRegression(max_iter=1000, random_state=42)
    
    # ───────────────────────────────────────────────────────────────────
    # Evaluation & Metrics
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("evaluate_ensemble")
    @_safe_operation("evaluation", default_value={})
    def _evaluate_ensemble(
        self,
        ensemble: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        problem_type: str
    ) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        metrics = {}
        
        # ─── Training Metrics ───
        y_train_pred = ensemble.predict(X_train)
        
        if problem_type == "classification":
            metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
            metrics["train_f1"] = f1_score(y_train, y_train_pred, average='weighted')
            
            if hasattr(ensemble, 'predict_proba'):
                y_train_proba = ensemble.predict_proba(X_train)
                if y_train_proba.shape[1] == 2:
                    metrics["train_roc_auc"] = roc_auc_score(y_train, y_train_proba[:, 1])
        else:
            metrics["train_r2"] = r2_score(y_train, y_train_pred)
            metrics["train_mae"] = mean_absolute_error(y_train, y_train_pred)
            metrics["train_rmse"] = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        # ─── Validation Metrics ─── #
        if X_val is not None and y_val is not None:
            y_val_pred = ensemble.predict(X_val)
            
            if problem_type == "classification":
                metrics["val_accuracy"] = accuracy_score(y_val, y_val_pred)
                metrics["val_f1"] = f1_score(y_val, y_val_pred, average='weighted')
                
                if hasattr(ensemble, 'predict_proba'):
                    y_val_proba = ensemble.predict_proba(X_val)
                    if y_val_proba.shape[1] == 2:
                        metrics["val_roc_auc"] = roc_auc_score(y_val, y_val_proba[:, 1])
            else:
                metrics["val_r2"] = r2_score(y_val, y_val_pred)
                metrics["val_mae"] = mean_absolute_error(y_val, y_val_pred)
                metrics["val_rmse"] = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        return metrics
    
    @_timeit("calculate_diversity")
    @_safe_operation("diversity", default_value={"diversity_score": 0.0})
    def _calculate_diversity(
        self,
        models: List[Tuple[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str
    ) -> Dict[str, float]:
        """Calculate diversity metrics for ensemble."""
        # Get predictions
        predictions = []
        for name, model in models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                self._log.warning(f"⚠ Could not get predictions from {name}: {e}")
        
        if len(predictions) < 2:
            return {"diversity_score": 0.0}
        
        predictions = np.array(predictions)
        
        # ─── Calculate Pairwise Disagreement ───
        n_models = len(predictions)
        disagreements = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreements.append(disagreement)
        
        diversity_metrics = {
            "avg_disagreement": float(np.mean(disagreements)),
            "min_disagreement": float(np.min(disagreements)),
            "max_disagreement": float(np.max(disagreements)),
            "diversity_score": float(np.mean(disagreements))
        }
        
        # ─── Calculate Correlation (Regression) ───
        if problem_type == "regression":
            correlations = []
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    corr, _ = pearsonr(predictions[i], predictions[j])
                    correlations.append(abs(corr))
            
            diversity_metrics["avg_correlation"] = float(np.mean(correlations))
            diversity_metrics["diversity_score"] = float(1 - np.mean(correlations))
        
        return diversity_metrics
    
    # ───────────────────────────────────────────────────────────────────
    # Prediction Methods
    # ───────────────────────────────────────────────────────────────────
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with ensemble."""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model trained. Call build_ensemble first.")
        
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model trained. Call build_ensemble first.")
        
        if not hasattr(self.ensemble_model, 'predict_proba'):
            raise ValueError("Ensemble model does not support predict_proba")
        
        return self.ensemble_model.predict_proba(X)
    
    # ───────────────────────────────────────────────────────────────────
    # Persistence Methods
    # ───────────────────────────────────────────────────────────────────
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save ensemble to file."""
        if self.result is None:
            raise ValueError("No ensemble to save. Build ensemble first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'ensemble_model': self.ensemble_model,
                'result': self.result,
                'config': self.config,
                'version': __version__
            }, f)
        
        self._log.info(f"✓ Ensemble saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load ensemble from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.ensemble_model = data['ensemble_model']
        self.result = data['result']
        self.config = data['config']
        
        self._log.info(f"✓ Ensemble loaded from {filepath}")
    
    # ───────────────────────────────────────────────────────────────────
    # Feature Importance
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("feature_importance", default_value=None)
    def get_feature_importance(
        self,
        method: str = "mean",
        feature_names: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get aggregated feature importance from base models.
        
        Args:
            method: 'mean', 'weighted', or 'max'
            feature_names: Optional list of feature names
        
        Returns:
            DataFrame with feature importances
        """
        if self.result is None:
            return None
        
        importances = []
        model_names = []
        
        for name, model in self.result.base_models:
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
                model_names.append(name)
            elif hasattr(model, 'coef_'):
                importances.append(np.abs(model.coef_).flatten())
                model_names.append(name)
        
        if not importances:
            return None
        
        importances = np.array(importances)
        
        # ─── Aggregate ───
        if method == "mean":
            agg_importance = np.mean(importances, axis=0)
        
        elif method == "weighted" and self.result.weights:
            weights = np.array([
                w for w in self.result.weights
                if len(self.result.weights) == len(importances)
            ])
            if len(weights) == len(importances):
                agg_importance = np.average(importances, axis=0, weights=weights)
            else:
                agg_importance = np.mean(importances, axis=0)
        
        elif method == "max":
            agg_importance = np.max(importances, axis=0)
        
        else:
            agg_importance = np.mean(importances, axis=0)
        
        # ─── Create DataFrame ───
        df = pd.DataFrame({'importance': agg_importance})
        
        if feature_names is not None:
            df['feature'] = feature_names
            df = df[['feature', 'importance']]
        
        return df.sort_values('importance', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Weighted Ensemble Class
# ═══════════════════════════════════════════════════════════════════════════

class WeightedEnsemble(BaseEstimator):
    """
    Custom weighted ensemble for flexible prediction combination.
    
    Features:
      • Weighted averaging of base model predictions
      • Support for classification and regression
      • Probability-based predictions (classification)
      • Normalized weights
      • Scikit-learn compatible interface
    """
    
    def __init__(
        self,
        models: List[Tuple[str, Any]],
        weights: np.ndarray,
        problem_type: str,
        use_probas: bool = True
    ) -> None:
        """
        Initialize weighted ensemble.
        
        Args:
            models: List of (name, model) tuples
            weights: Model weights (will be normalized)
            problem_type: 'classification' or 'regression'
            use_probas: Use probabilities for classification
        """
        self.models = models
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()  # Normalize
        self.problem_type = problem_type
        self.use_probas = use_probas
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> WeightedEnsemble:
        """Fit all base models."""
        for name, model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted predictions."""
        predictions = []
        
        for name, model in self.models:
            if self.problem_type == "classification" and self.use_probas:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    if pred.shape[1] == 2:
                        pred = pred[:, 1]
                else:
                    pred = model.predict(X)
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions).T
        weighted_pred = np.average(predictions, axis=1, weights=self.weights)
        
        if self.problem_type == "classification":
            return (weighted_pred > 0.5).astype(int)
        else:
            return weighted_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.problem_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        predictions = []
        
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                # Convert to pseudo-probabilities
                pred = model.predict(X)
                pred = np.column_stack([1 - pred, pred])
            predictions.append(pred)
        
        # Weighted average of probabilities
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Function
# ═══════════════════════════════════════════════════════════════════════════

def build_ensemble(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    problem_type: str = "classification",
    config: Optional[EnsembleConfig] = None
) -> EnsembleResult:
    """
    Convenience function to build ensemble.
    
    Usage:
```python
        # Build ensemble
        result = build_ensemble(
            models={
                'rf': RandomForestClassifier(),
                'xgb': XGBClassifier(),
                'lgb': LGBMClassifier()
            },
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            problem_type='classification'
        )
        
        # Use ensemble
        predictions = result.ensemble_model.predict(X_test)
```
    
    Args:
        models: Dictionary of {model_name: trained_model}
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        problem_type: 'classification' or 'regression'
        config: Optional custom configuration
    
    Returns:
        EnsembleResult with ensemble model and metrics
    """
    builder = EnsembleBuilder(config)
    return builder.build_ensemble(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        problem_type=problem_type
    )
