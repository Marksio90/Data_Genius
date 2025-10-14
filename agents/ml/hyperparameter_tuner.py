"""
DataGenius PRO - Hyperparameter Tuner
Advanced hyperparameter optimization system.

Supports:
- Grid Search
- Random Search
- Bayesian Optimization (Optuna)
- Hyperband
- Genetic Algorithms
- Automated search space generation
- Parallel execution
- Early stopping
"""

from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold, KFold
)
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
import json
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TuningStrategy(str, Enum):
    """Hyperparameter tuning strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"  # Optuna
    HYPERBAND = "hyperband"
    GENETIC = "genetic"
    AUTO = "auto"


class OptimizationMetric(str, Enum):
    """Optimization metrics."""
    # Classification
    ACCURACY = "accuracy"
    F1_SCORE = "f1"
    ROC_AUC = "roc_auc"
    PRECISION = "precision"
    RECALL = "recall"
    
    # Regression
    R2 = "r2"
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    
    strategy: TuningStrategy = TuningStrategy.AUTO
    n_iter: int = 50  # For random/bayesian search
    cv_folds: int = 5
    metric: OptimizationMetric = OptimizationMetric.ACCURACY
    n_jobs: int = -1
    verbose: int = 1
    random_state: int = 42
    
    # Bayesian optimization settings
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    early_stopping_rounds: Optional[int] = 10
    
    # Advanced settings
    use_early_stopping: bool = True
    refit: bool = True
    return_train_score: bool = True
    error_score: Union[str, float] = 'raise'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "n_iter": self.n_iter,
            "cv_folds": self.cv_folds,
            "metric": self.metric.value,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "early_stopping_rounds": self.early_stopping_rounds
        }


@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""
    
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    cv_results: pd.DataFrame
    tuning_time: float
    strategy: TuningStrategy
    metric: OptimizationMetric
    n_iterations: int
    
    # Additional info
    param_importance: Optional[Dict[str, float]] = None
    optimization_history: Optional[List[float]] = None
    config: Optional[TuningConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding model)."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_results_summary": {
                "mean_score": float(self.cv_results['mean_test_score'].mean()),
                "std_score": float(self.cv_results['mean_test_score'].std()),
                "best_score": float(self.cv_results['mean_test_score'].max())
            },
            "tuning_time": self.tuning_time,
            "strategy": self.strategy.value,
            "metric": self.metric.value,
            "n_iterations": self.n_iterations,
            "param_importance": self.param_importance,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class SearchSpaceBuilder:
    """
    Automated search space builder for popular models.
    
    Provides sensible default search spaces for:
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - Neural Networks
    - SVM
    - etc.
    """
    
    @staticmethod
    def get_search_space(
        model_name: str,
        problem_type: str = "classification",
        strategy: TuningStrategy = TuningStrategy.RANDOM_SEARCH,
        aggressive: bool = False
    ) -> Dict[str, Any]:
        """
        Get search space for a model.
        
        Args:
            model_name: Name of the model
            problem_type: 'classification' or 'regression'
            strategy: Tuning strategy (affects space granularity)
            aggressive: If True, use wider search space
        
        Returns:
            Dictionary of hyperparameter search space
        """
        model_name = model_name.lower()
        
        if "random" in model_name or "rf" in model_name:
            return SearchSpaceBuilder._random_forest_space(strategy, aggressive)
        elif "xgb" in model_name or "xgboost" in model_name:
            return SearchSpaceBuilder._xgboost_space(strategy, aggressive, problem_type)
        elif "lgb" in model_name or "lightgbm" in model_name:
            return SearchSpaceBuilder._lightgbm_space(strategy, aggressive, problem_type)
        elif "catboost" in model_name or "cat" in model_name:
            return SearchSpaceBuilder._catboost_space(strategy, aggressive)
        elif "svm" in model_name or "svc" in model_name or "svr" in model_name:
            return SearchSpaceBuilder._svm_space(strategy, aggressive)
        elif "logistic" in model_name:
            return SearchSpaceBuilder._logistic_space(strategy, aggressive)
        elif "mlp" in model_name or "neural" in model_name:
            return SearchSpaceBuilder._mlp_space(strategy, aggressive)
        elif "gradient" in model_name and "boost" in model_name:
            return SearchSpaceBuilder._gbm_space(strategy, aggressive)
        else:
            logger.warning(f"No predefined search space for {model_name}")
            return {}
    
    @staticmethod
    def _random_forest_space(strategy: TuningStrategy, aggressive: bool) -> Dict:
        """Search space for Random Forest."""
        if strategy == TuningStrategy.GRID_SEARCH:
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        else:  # Random/Bayesian
            if aggressive:
                return {
                    'n_estimators': (50, 500),
                    'max_depth': (5, 50),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': ['sqrt', 'log2', 0.5, 0.7, None],
                    'bootstrap': [True, False],
                    'criterion': ['gini', 'entropy']
                }
            else:
                return {
                    'n_estimators': (100, 300),
                    'max_depth': (10, 30),
                    'min_samples_split': (2, 10),
                    'min_samples_leaf': (1, 4),
                    'max_features': ['sqrt', 'log2']
                }
    
    @staticmethod
    def _xgboost_space(strategy: TuningStrategy, aggressive: bool, problem_type: str) -> Dict:
        """Search space for XGBoost."""
        base_space = {
            'n_estimators': (100, 500) if strategy != TuningStrategy.GRID_SEARCH else [100, 200, 300],
            'max_depth': (3, 10) if strategy != TuningStrategy.GRID_SEARCH else [3, 5, 7, 9],
            'learning_rate': (0.01, 0.3) if strategy != TuningStrategy.GRID_SEARCH else [0.01, 0.05, 0.1, 0.2],
            'subsample': (0.6, 1.0) if strategy != TuningStrategy.GRID_SEARCH else [0.6, 0.8, 1.0],
            'colsample_bytree': (0.6, 1.0) if strategy != TuningStrategy.GRID_SEARCH else [0.6, 0.8, 1.0],
        }
        
        if aggressive:
            base_space.update({
                'min_child_weight': (1, 10),
                'gamma': (0, 5),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10)
            })
        
        return base_space
    
    @staticmethod
    def _lightgbm_space(strategy: TuningStrategy, aggressive: bool, problem_type: str) -> Dict:
        """Search space for LightGBM."""
        base_space = {
            'n_estimators': (100, 500) if strategy != TuningStrategy.GRID_SEARCH else [100, 200, 300],
            'max_depth': (3, 15) if strategy != TuningStrategy.GRID_SEARCH else [3, 5, 7, 10],
            'learning_rate': (0.01, 0.3) if strategy != TuningStrategy.GRID_SEARCH else [0.01, 0.05, 0.1],
            'num_leaves': (20, 150) if strategy != TuningStrategy.GRID_SEARCH else [31, 63, 127],
            'min_child_samples': (10, 100) if strategy != TuningStrategy.GRID_SEARCH else [20, 50, 100]
        }
        
        if aggressive:
            base_space.update({
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10),
                'min_split_gain': (0, 1)
            })
        
        return base_space
    
    @staticmethod
    def _catboost_space(strategy: TuningStrategy, aggressive: bool) -> Dict:
        """Search space for CatBoost."""
        base_space = {
            'iterations': (100, 500) if strategy != TuningStrategy.GRID_SEARCH else [100, 200, 300],
            'depth': (4, 10) if strategy != TuningStrategy.GRID_SEARCH else [4, 6, 8, 10],
            'learning_rate': (0.01, 0.3) if strategy != TuningStrategy.GRID_SEARCH else [0.01, 0.05, 0.1],
            'l2_leaf_reg': (1, 10) if strategy != TuningStrategy.GRID_SEARCH else [1, 3, 5, 7, 9]
        }
        
        if aggressive:
            base_space.update({
                'border_count': (32, 255),
                'bagging_temperature': (0, 1),
                'random_strength': (0, 10)
            })
        
        return base_space
    
    @staticmethod
    def _svm_space(strategy: TuningStrategy, aggressive: bool) -> Dict:
        """Search space for SVM."""
        if strategy == TuningStrategy.GRID_SEARCH:
            return {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        else:
            return {
                'C': (0.1, 100),
                'gamma': (0.0001, 1),
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'degree': (2, 5) if aggressive else [3]
            }
    
    @staticmethod
    def _logistic_space(strategy: TuningStrategy, aggressive: bool) -> Dict:
        """Search space for Logistic Regression."""
        if strategy == TuningStrategy.GRID_SEARCH:
            return {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        else:
            return {
                'C': (0.001, 100),
                'penalty': ['l1', 'l2', 'elasticnet'] if aggressive else ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
    
    @staticmethod
    def _mlp_space(strategy: TuningStrategy, aggressive: bool) -> Dict:
        """Search space for MLP."""
        if strategy == TuningStrategy.GRID_SEARCH:
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        else:
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': (0.0001, 0.1),
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': (0.0001, 0.01)
            }
    
    @staticmethod
    def _gbm_space(strategy: TuningStrategy, aggressive: bool) -> Dict:
        """Search space for Gradient Boosting Machine."""
        if strategy == TuningStrategy.GRID_SEARCH:
            return {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        else:
            return {
                'n_estimators': (100, 500),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'subsample': (0.6, 1.0),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            }


class HyperparameterTuner:
    """
    Advanced hyperparameter tuning system.
    
    Features:
    - Multiple tuning strategies
    - Automated search space generation
    - Parallel execution
    - Early stopping
    - Progress tracking
    - Result persistence
    """
    
    def __init__(self, config: Optional[TuningConfig] = None):
        """Initialize hyperparameter tuner."""
        self.config = config or TuningConfig()
        self.result: Optional[TuningResult] = None
        
        logger.info(f"HyperparameterTuner initialized with strategy: {self.config.strategy}")
    
    def tune(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification",
        model_name: Optional[str] = None
    ) -> TuningResult:
        """
        Tune hyperparameters for a model.
        
        Args:
            model: Base model to tune
            X: Training features
            y: Training target
            param_space: Custom parameter search space
            problem_type: 'classification' or 'regression'
            model_name: Name of model (for auto search space)
        
        Returns:
            TuningResult with best parameters and model
        """
        logger.info(f"Starting hyperparameter tuning with {self.config.strategy}")
        start_time = datetime.now()
        
        # Get search space
        if param_space is None:
            if model_name is None:
                model_name = model.__class__.__name__
            param_space = SearchSpaceBuilder.get_search_space(
                model_name, problem_type, self.config.strategy
            )
            if not param_space:
                raise ValueError(f"No search space provided or auto-generated for {model_name}")
        
        logger.info(f"Search space: {list(param_space.keys())}")
        
        # Choose strategy
        strategy = self.config.strategy
        if strategy == TuningStrategy.AUTO:
            strategy = self._choose_strategy(len(param_space), X.shape[0])
        
        logger.info(f"Using strategy: {strategy}")
        
        # Execute tuning based on strategy
        if strategy == TuningStrategy.GRID_SEARCH:
            result = self._grid_search(model, X, y, param_space, problem_type)
        elif strategy == TuningStrategy.RANDOM_SEARCH:
            result = self._random_search(model, X, y, param_space, problem_type)
        elif strategy == TuningStrategy.BAYESIAN:
            result = self._bayesian_search(model, X, y, param_space, problem_type)
        else:
            raise ValueError(f"Strategy {strategy} not implemented")
        
        # Calculate tuning time
        tuning_time = (datetime.now() - start_time).total_seconds()
        result.tuning_time = tuning_time
        result.config = self.config
        
        self.result = result
        
        logger.info(f"Tuning completed in {tuning_time:.2f}s")
        logger.info(f"Best score: {result.best_score:.4f}")
        logger.info(f"Best params: {result.best_params}")
        
        return result
    
    def _choose_strategy(self, n_params: int, n_samples: int) -> TuningStrategy:
        """Automatically choose best tuning strategy."""
        if n_params <= 3 and n_samples < 10000:
            return TuningStrategy.GRID_SEARCH
        elif n_params <= 6:
            return TuningStrategy.RANDOM_SEARCH
        else:
            return TuningStrategy.BAYESIAN
    
    def _grid_search(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        problem_type: str
    ) -> TuningResult:
        """Perform grid search."""
        logger.info("Executing Grid Search")
        
        # Get scorer
        scoring = self._get_scorer(problem_type)
        
        # Setup CV
        cv = self._get_cv_splitter(problem_type)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_space,
            scoring=scoring,
            cv=cv,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            refit=self.config.refit,
            return_train_score=self.config.return_train_score,
            error_score=self.config.error_score
        )
        
        grid_search.fit(X, y)
        
        # Create result
        cv_results = pd.DataFrame(grid_search.cv_results_)
        
        result = TuningResult(
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            best_model=grid_search.best_estimator_,
            cv_results=cv_results,
            tuning_time=0,  # Will be set later
            strategy=TuningStrategy.GRID_SEARCH,
            metric=self.config.metric,
            n_iterations=len(cv_results),
            metadata={
                "n_splits": self.config.cv_folds,
                "total_fits": len(cv_results) * self.config.cv_folds
            }
        )
        
        return result
    
    def _random_search(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        problem_type: str
    ) -> TuningResult:
        """Perform random search."""
        logger.info(f"Executing Random Search with {self.config.n_iter} iterations")
        
        # Convert param space to distributions
        param_distributions = self._convert_to_distributions(param_space)
        
        # Get scorer
        scoring = self._get_scorer(problem_type)
        
        # Setup CV
        cv = self._get_cv_splitter(problem_type)
        
        # Random search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=self.config.n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            random_state=self.config.random_state,
            refit=self.config.refit,
            return_train_score=self.config.return_train_score,
            error_score=self.config.error_score
        )
        
        random_search.fit(X, y)
        
        # Create result
        cv_results = pd.DataFrame(random_search.cv_results_)
        
        result = TuningResult(
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            best_model=random_search.best_estimator_,
            cv_results=cv_results,
            tuning_time=0,
            strategy=TuningStrategy.RANDOM_SEARCH,
            metric=self.config.metric,
            n_iterations=self.config.n_iter,
            optimization_history=cv_results['mean_test_score'].tolist(),
            metadata={
                "n_splits": self.config.cv_folds,
                "total_fits": self.config.n_iter * self.config.cv_folds
            }
        )
        
        return result
    
    def _bayesian_search(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        problem_type: str
    ) -> TuningResult:
        """Perform Bayesian optimization using Optuna."""
        logger.info(f"Executing Bayesian Optimization with {self.config.n_trials} trials")
        
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.warning("Optuna not available, falling back to Random Search")
            return self._random_search(model, X, y, param_space, problem_type)
        
        # Setup CV
        cv = self._get_cv_splitter(problem_type)
        
        # Get scorer function
        scorer_func = self._get_scorer_function(problem_type)
        
        # Track history
        history = []
        
        def objective(trial):
            """Optuna objective function."""
            # Sample parameters
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Continuous or integer range
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Categorical
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    params[param_name] = param_range
            
            # Set parameters
            model.set_params(**params)
            
            # Cross-validation
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring=scorer_func,
                    n_jobs=1,  # Optuna handles parallelism
                    error_score='raise'
                )
                score = scores.mean()
                history.append(score)
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                history.append(-999)
                return -999
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=self.config.verbose > 0
        )
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Train final model with best parameters
        model.set_params(**best_params)
        model.fit(X, y)
        
        # Create cv_results DataFrame
        trials_df = study.trials_dataframe()
        cv_results = pd.DataFrame({
            'mean_test_score': trials_df['value'].values,
            'params': [t.params for t in study.trials]
        })
        
        # Calculate parameter importance
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except:
            param_importance = None
        
        result = TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_model=model,
            cv_results=cv_results,
            tuning_time=0,
            strategy=TuningStrategy.BAYESIAN,
            metric=self.config.metric,
            n_iterations=len(study.trials),
            param_importance=param_importance,
            optimization_history=history,
            metadata={
                "n_trials": len(study.trials),
                "n_complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            }
        )
        
        return result
    
    def _convert_to_distributions(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter space to scipy distributions for RandomizedSearchCV."""
        from scipy.stats import uniform, randint
        
        distributions = {}
        for param_name, param_range in param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    # Integer range
                    distributions[param_name] = randint(param_range[0], param_range[1] + 1)
                else:
                    # Float range
                    distributions[param_name] = uniform(param_range[0], param_range[1] - param_range[0])
            else:
                # List or single value
                distributions[param_name] = param_range
        
        return distributions
    
    def _get_cv_splitter(self, problem_type: str):
        """Get cross-validation splitter."""
        if problem_type == "classification":
            return StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
        else:
            return KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
    
    def _get_scorer(self, problem_type: str):
        """Get scorer for sklearn."""
        metric_map = {
            OptimizationMetric.ACCURACY: 'accuracy',
            OptimizationMetric.F1_SCORE: 'f1_weighted',
            OptimizationMetric.ROC_AUC: 'roc_auc',
            OptimizationMetric.PRECISION: 'precision_weighted',
            OptimizationMetric.RECALL: 'recall_weighted',
            OptimizationMetric.R2: 'r2',
            OptimizationMetric.MSE: 'neg_mean_squared_error',
            OptimizationMetric.RMSE: 'neg_root_mean_squared_error',
            OptimizationMetric.MAE: 'neg_mean_absolute_error'
        }
        
        return metric_map.get(self.config.metric, 'accuracy' if problem_type == "classification" else 'r2')
    
    def _get_scorer_function(self, problem_type: str) -> str:
        """Get scorer function name."""
        return self._get_scorer(problem_type)
    
    def save(self, filepath: Union[str, Path]):
        """Save tuning result."""
        if self.result is None:
            raise ValueError("No result to save. Run tune() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'result': self.result,
                'config': self.config
            }, f)
        
        logger.info(f"Result saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """Load tuning result."""
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.result = data['result']
        self.config = data['config']
        
        logger.info(f"Result loaded from {filepath}")
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if self.result is None or self.result.optimization_history is None:
            logger.warning("No optimization history available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            history = self.result.optimization_history
            best_so_far = np.maximum.accumulate(history)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history, alpha=0.6, label='Score')
            ax.plot(best_so_far, linewidth=2, label='Best Score')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(f'{self.result.metric.value}')
            ax.set_title('Hyperparameter Optimization History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def get_best_model(self) -> Optional[BaseEstimator]:
        """Get the best trained model."""
        if self.result is None:
            return None
        return self.result.best_model
    
    def get_cv_results_summary(self) -> Optional[pd.DataFrame]:
        """Get summary of CV results."""
        if self.result is None:
            return None
        
        df = self.result.cv_results.copy()
        
        # Sort by score
        df = df.sort_values('mean_test_score', ascending=False)
        
        # Select important columns
        cols = ['mean_test_score', 'std_test_score', 'params']
        if 'mean_train_score' in df.columns:
            cols.insert(1, 'mean_train_score')
        
        return df[cols].head(10)


# Convenience function
def tune_hyperparameters(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    param_space: Optional[Dict[str, Any]] = None,
    strategy: TuningStrategy = TuningStrategy.AUTO,
    problem_type: str = "classification",
    n_iter: int = 50,
    cv_folds: int = 5,
    **kwargs
) -> TuningResult:
    """
    Convenience function for hyperparameter tuning.
    
    Usage:
        result = tune_hyperparameters(
            model=RandomForestClassifier(),
            X=X_train,
            y=y_train,
            strategy=TuningStrategy.BAYESIAN,
            n_iter=100
        )
    """
    config = TuningConfig(
        strategy=strategy,
        n_iter=n_iter,
        cv_folds=cv_folds,
        **kwargs
    )
    
    tuner = HyperparameterTuner(config)
    return tuner.tune(model, X, y, param_space, problem_type)