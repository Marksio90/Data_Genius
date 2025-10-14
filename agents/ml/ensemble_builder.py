"""
DataGenius PRO - Ensemble Builder
Advanced ensemble learning system for combining multiple models.

Supports:
- Voting (hard/soft)
- Stacking with meta-learners
- Blending
- Weighted averaging with optimization
- Automated model selection
- Diversity analysis
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import logging
import pickle
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class EnsembleStrategy(str, Enum):
    """Ensemble strategies."""
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


@dataclass
class EnsembleConfig:
    """Configuration for ensemble building."""
    
    strategy: EnsembleStrategy = EnsembleStrategy.AUTO
    n_models: int = 5  # Max number of models to include
    min_diversity: float = 0.1  # Minimum diversity between models
    cv_folds: int = 5
    meta_learner: MetaLearnerType = MetaLearnerType.LOGISTIC
    optimize_weights: bool = True
    voting_type: str = "soft"  # "hard" or "soft"
    use_probas: bool = True  # Use probabilities for classification
    
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
    """Results from ensemble building."""
    
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
            "created_at": self.created_at.isoformat()
        }


class EnsembleBuilder:
    """
    Advanced ensemble builder for combining multiple models.
    
    Features:
    - Multiple ensemble strategies
    - Automatic model selection based on diversity
    - Weight optimization
    - Cross-validation
    - Performance tracking
    - Model persistence
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize ensemble builder."""
        self.config = config or EnsembleConfig()
        self.ensemble_model = None
        self.result: Optional[EnsembleResult] = None
        
        logger.info(f"EnsembleBuilder initialized with strategy: {self.config.strategy}")
    
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
        logger.info(f"Building ensemble with {len(models)} base models")
        start_time = datetime.now()
        
        # Select best models based on diversity and performance
        selected_models = self._select_diverse_models(
            models, X_train, y_train, problem_type
        )
        
        logger.info(f"Selected {len(selected_models)} models for ensemble")
        
        # Choose strategy
        if self.config.strategy == EnsembleStrategy.AUTO:
            strategy = self._choose_strategy(problem_type, len(selected_models))
        else:
            strategy = self.config.strategy
        
        logger.info(f"Using strategy: {strategy}")
        
        # Build ensemble based on strategy
        if strategy == EnsembleStrategy.VOTING_SOFT:
            ensemble_model = self._build_voting_ensemble(
                selected_models, problem_type, "soft"
            )
            weights = None
            
        elif strategy == EnsembleStrategy.VOTING_HARD:
            ensemble_model = self._build_voting_ensemble(
                selected_models, problem_type, "hard"
            )
            weights = None
            
        elif strategy == EnsembleStrategy.STACKING:
            ensemble_model = self._build_stacking_ensemble(
                selected_models, problem_type
            )
            weights = None
            
        elif strategy == EnsembleStrategy.BLENDING:
            if X_val is None or y_val is None:
                logger.warning("Blending requires validation set, falling back to voting")
                ensemble_model = self._build_voting_ensemble(
                    selected_models, problem_type, "soft"
                )
                weights = None
            else:
                ensemble_model, weights = self._build_blending_ensemble(
                    selected_models, X_train, y_train, X_val, y_val, problem_type
                )
        
        elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            ensemble_model, weights = self._build_weighted_ensemble(
                selected_models, X_train, y_train, problem_type
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Train ensemble
        if not isinstance(ensemble_model, WeightedEnsemble):
            ensemble_model.fit(X_train, y_train)
        
        # Evaluate ensemble
        performance = self._evaluate_ensemble(
            ensemble_model, X_train, y_train, X_val, y_val, problem_type
        )
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity(
            selected_models, X_train, y_train, problem_type
        )
        
        # Create result
        training_time = (datetime.now() - start_time).total_seconds()
        
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
        
        logger.info(f"Ensemble built successfully in {training_time:.2f}s")
        logger.info(f"Performance: {performance}")
        
        return self.result
    
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
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            try:
                if problem_type == "classification" and self.config.use_probas:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1] if pred.shape[1] == 2 else pred
                    else:
                        pred = model.predict(X)
                else:
                    pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Could not get predictions from {name}: {e}")
        
        # Calculate performance scores
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
            except:
                scores[name] = 0.0
        
        # Sort by performance
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select diverse models
        selected = []
        selected_preds = []
        
        for name, score in sorted_models:
            if len(selected) >= self.config.n_models:
                break
            
            pred = predictions[name]
            
            # Check diversity with already selected models
            if len(selected) == 0:
                selected.append((name, models[name]))
                selected_preds.append(pred)
            else:
                # Calculate correlation with selected models
                correlations = [pearsonr(pred, sp)[0] for sp in selected_preds]
                avg_corr = np.mean(np.abs(correlations))
                
                # Add if sufficiently diverse
                if avg_corr < (1 - self.config.min_diversity):
                    selected.append((name, models[name]))
                    selected_preds.append(pred)
                    logger.info(f"Selected {name} (score: {score:.4f}, diversity: {1-avg_corr:.4f})")
        
        # If we don't have enough diverse models, add best remaining
        if len(selected) < min(self.config.n_models, len(models)):
            for name, _ in sorted_models:
                if len(selected) >= self.config.n_models:
                    break
                if name not in [s[0] for s in selected]:
                    selected.append((name, models[name]))
        
        return selected
    
    def _choose_strategy(self, problem_type: str, n_models: int) -> EnsembleStrategy:
        """Automatically choose best ensemble strategy."""
        if n_models <= 2:
            return EnsembleStrategy.VOTING_SOFT
        elif n_models <= 4:
            return EnsembleStrategy.WEIGHTED_AVERAGE
        else:
            return EnsembleStrategy.STACKING
    
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
        
        # Get meta-learner
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
    
    def _get_meta_learner(self, problem_type: str) -> BaseEstimator:
        """Get meta-learner for stacking."""
        if self.config.meta_learner == MetaLearnerType.LOGISTIC:
            if problem_type == "classification":
                return LogisticRegression(max_iter=1000, random_state=42)
            else:
                return Ridge(alpha=1.0, random_state=42)
        
        elif self.config.meta_learner == MetaLearnerType.RIDGE:
            return Ridge(alpha=1.0, random_state=42)
        
        elif self.config.meta_learner == MetaLearnerType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if problem_type == "classification":
                return RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                return RandomForestRegressor(n_estimators=100, random_state=42)
        
        elif self.config.meta_learner == MetaLearnerType.XGBOOST:
            try:
                import xgboost as xgb
                if problem_type == "classification":
                    return xgb.XGBClassifier(random_state=42)
                else:
                    return xgb.XGBRegressor(random_state=42)
            except ImportError:
                logger.warning("XGBoost not available, using LogisticRegression")
                return LogisticRegression(max_iter=1000, random_state=42)
        
        elif self.config.meta_learner == MetaLearnerType.LIGHTGBM:
            try:
                import lightgbm as lgb
                if problem_type == "classification":
                    return lgb.LGBMClassifier(random_state=42, verbose=-1)
                else:
                    return lgb.LGBMRegressor(random_state=42, verbose=-1)
            except ImportError:
                logger.warning("LightGBM not available, using Ridge")
                return Ridge(alpha=1.0, random_state=42)
        
        else:
            return LogisticRegression(max_iter=1000, random_state=42)
    
    def _build_blending_ensemble(
        self,
        models: List[Tuple[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        problem_type: str
    ) -> Tuple[Any, List[float]]:
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
        
        val_preds = np.array(val_preds).T
        
        # Optimize weights
        weights = self._optimize_weights(val_preds, y_val.values, problem_type)
        
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
    ) -> Tuple[Any, List[float]]:
        """Build weighted ensemble with CV-optimized weights."""
        
        # Get CV predictions
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        cv_preds = np.zeros((len(X_train), len(models)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            for model_idx, (name, model) in enumerate(models):
                # Clone and train model
                from sklearn.base import clone
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
            Optimized weights
        """
        n_models = predictions.shape[1]
        
        def objective(weights):
            """Objective function to minimize."""
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            weighted_pred = np.average(predictions, axis=1, weights=weights)
            
            if problem_type == "classification":
                # Use log loss
                weighted_pred = np.clip(weighted_pred, 1e-15, 1 - 1e-15)
                loss = -np.mean(y_true * np.log(weighted_pred) + 
                               (1 - y_true) * np.log(1 - weighted_pred))
            else:
                # Use MSE
                loss = mean_squared_error(y_true, weighted_pred)
            
            return loss
        
        # Optimize using differential evolution
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
        
        logger.info(f"Optimized weights: {weights}")
        
        return weights
    
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
        
        # Training metrics
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
        
        # Validation metrics
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
                logger.warning(f"Could not get predictions from {name}: {e}")
        
        if len(predictions) < 2:
            return {"diversity_score": 0.0}
        
        predictions = np.array(predictions)
        
        # Calculate pairwise disagreement
        n_models = len(predictions)
        disagreements = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreements.append(disagreement)
        
        diversity_metrics = {
            "avg_disagreement": np.mean(disagreements),
            "min_disagreement": np.min(disagreements),
            "max_disagreement": np.max(disagreements),
            "diversity_score": np.mean(disagreements)  # Main metric
        }
        
        # Calculate correlation for continuous predictions
        if problem_type == "regression":
            correlations = []
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    corr = pearsonr(predictions[i], predictions[j])[0]
                    correlations.append(abs(corr))
            
            diversity_metrics["avg_correlation"] = np.mean(correlations)
            diversity_metrics["diversity_score"] = 1 - np.mean(correlations)
        
        return diversity_metrics
    
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
    
    def save(self, filepath: Union[str, Path]):
        """Save ensemble to file."""
        if self.result is None:
            raise ValueError("No ensemble to save. Build ensemble first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble
        with open(filepath, 'wb') as f:
            pickle.dump({
                'ensemble_model': self.ensemble_model,
                'result': self.result,
                'config': self.config
            }, f)
        
        logger.info(f"Ensemble saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """Load ensemble from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.ensemble_model = data['ensemble_model']
        self.result = data['result']
        self.config = data['config']
        
        logger.info(f"Ensemble loaded from {filepath}")
    
    def get_feature_importance(self, method: str = "mean") -> Optional[pd.DataFrame]:
        """
        Get aggregated feature importance from base models.
        
        Args:
            method: 'mean', 'weighted', or 'max'
        
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
        
        # Aggregate
        if method == "mean":
            agg_importance = np.mean(importances, axis=0)
        elif method == "weighted" and self.result.weights:
            weights = np.array([w for w in self.result.weights 
                              if len(self.result.weights) == len(importances)])
            if len(weights) == len(importances):
                agg_importance = np.average(importances, axis=0, weights=weights)
            else:
                agg_importance = np.mean(importances, axis=0)
        elif method == "max":
            agg_importance = np.max(importances, axis=0)
        else:
            agg_importance = np.mean(importances, axis=0)
        
        # Create DataFrame
        # Note: We need feature names - assuming they're in the model or we can extract them
        return pd.DataFrame({
            'importance': agg_importance
        })


class WeightedEnsemble(BaseEstimator):
    """
    Custom weighted ensemble for flexible prediction combination.
    """
    
    def __init__(
        self,
        models: List[Tuple[str, Any]],
        weights: np.ndarray,
        problem_type: str,
        use_probas: bool = True
    ):
        """Initialize weighted ensemble."""
        self.models = models
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()  # Normalize
        self.problem_type = problem_type
        self.use_probas = use_probas
    
    def fit(self, X, y):
        """Fit all base models."""
        for name, model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
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
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.problem_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        predictions = []
        
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                # Convert predictions to pseudo-probabilities
                pred = model.predict(X)
                pred = np.column_stack([1 - pred, pred])
            predictions.append(pred)
        
        # Weighted average of probabilities
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred


# Convenience function
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
        result = build_ensemble(
            models={'rf': rf_model, 'xgb': xgb_model},
            X_train=X_train,
            y_train=y_train,
            problem_type='classification'
        )
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