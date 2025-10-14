"""
DataGenius PRO - Model Selector
Intelligent model selection system using meta-learning and data characteristics.

Features:
- Automatic model recommendation
- Meta-feature extraction
- Model ranking based on multiple criteria
- Performance prediction
- Dataset complexity analysis
- Multi-objective optimization
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from scipy.stats import skew, kurtosis
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ProblemType(str, Enum):
    """Types of ML problems."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    IMBALANCED = "imbalanced"
    HIGH_DIMENSIONAL = "high_dimensional"


class ModelCategory(str, Enum):
    """Categories of ML models."""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    ENSEMBLE = "ensemble"
    BOOSTING = "boosting"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    NAIVE_BAYES = "naive_bayes"
    KNN = "knn"


class SelectionCriterion(str, Enum):
    """Criteria for model selection."""
    ACCURACY = "accuracy"
    SPEED = "speed"
    INTERPRETABILITY = "interpretability"
    ROBUSTNESS = "robustness"
    SCALABILITY = "scalability"
    MEMORY = "memory"
    BALANCED = "balanced"


@dataclass
class DatasetCharacteristics:
    """Characteristics of the dataset for meta-learning."""
    
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    class_balance: Optional[Dict[str, float]]
    
    # Feature characteristics
    n_numerical: int
    n_categorical: int
    n_binary: int
    
    # Statistical properties
    avg_skewness: float
    avg_kurtosis: float
    missing_ratio: float
    
    # Complexity measures
    feature_to_sample_ratio: float
    avg_correlation: float
    dimensionality_score: float
    
    # Problem characteristics
    is_imbalanced: bool
    is_high_dimensional: bool
    is_sparse: bool
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "class_balance": self.class_balance,
            "n_numerical": self.n_numerical,
            "n_categorical": self.n_categorical,
            "n_binary": self.n_binary,
            "avg_skewness": self.avg_skewness,
            "avg_kurtosis": self.avg_kurtosis,
            "missing_ratio": self.missing_ratio,
            "feature_to_sample_ratio": self.feature_to_sample_ratio,
            "avg_correlation": self.avg_correlation,
            "dimensionality_score": self.dimensionality_score,
            "is_imbalanced": self.is_imbalanced,
            "is_high_dimensional": self.is_high_dimensional,
            "is_sparse": self.is_sparse,
            "metadata": self.metadata
        }


@dataclass
class ModelScore:
    """Score for a model candidate."""
    
    model_name: str
    model_category: ModelCategory
    
    # Performance scores
    performance_score: float
    speed_score: float
    interpretability_score: float
    robustness_score: float
    scalability_score: float
    memory_score: float
    
    # Overall score
    total_score: float
    
    # Additional info
    recommended: bool
    reasoning: List[str]
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_category": self.model_category.value,
            "performance_score": self.performance_score,
            "speed_score": self.speed_score,
            "interpretability_score": self.interpretability_score,
            "robustness_score": self.robustness_score,
            "scalability_score": self.scalability_score,
            "memory_score": self.memory_score,
            "total_score": self.total_score,
            "recommended": self.recommended,
            "reasoning": self.reasoning,
            "warnings": self.warnings
        }


@dataclass
class SelectionResult:
    """Result of model selection."""
    
    recommended_models: List[str]
    model_scores: List[ModelScore]
    dataset_characteristics: DatasetCharacteristics
    selection_criterion: SelectionCriterion
    problem_type: ProblemType
    
    # Meta information
    selection_time: float
    total_models_evaluated: int
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommended_models": self.recommended_models,
            "model_scores": [score.to_dict() for score in self.model_scores],
            "dataset_characteristics": self.dataset_characteristics.to_dict(),
            "selection_criterion": self.selection_criterion.value,
            "problem_type": self.problem_type.value,
            "selection_time": self.selection_time,
            "total_models_evaluated": self.total_models_evaluated,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    def get_top_models(self, n: int = 5) -> List[ModelScore]:
        """Get top N models by total score."""
        sorted_scores = sorted(self.model_scores, key=lambda x: x.total_score, reverse=True)
        return sorted_scores[:n]


class ModelRegistry:
    """
    Registry of available models with their characteristics.
    
    Each model has:
    - Category
    - Strengths/weaknesses
    - Recommended use cases
    - Computational complexity
    - Interpretability
    """
    
    MODELS = {
        # Linear Models
        "logistic_regression": {
            "category": ModelCategory.LINEAR,
            "interpretability": 5,
            "speed": 5,
            "memory": 5,
            "scalability": 5,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": False,
            "best_for": ["small_data", "interpretable", "linear_relationships"],
            "not_recommended_for": ["non_linear", "large_data", "high_dimensional"]
        },
        "ridge": {
            "category": ModelCategory.LINEAR,
            "interpretability": 5,
            "speed": 5,
            "memory": 5,
            "scalability": 5,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": False,
            "best_for": ["small_data", "interpretable", "regularization"],
            "not_recommended_for": ["non_linear", "categorical_heavy"]
        },
        "lasso": {
            "category": ModelCategory.LINEAR,
            "interpretability": 5,
            "speed": 5,
            "memory": 5,
            "scalability": 5,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": False,
            "best_for": ["feature_selection", "sparse_solutions", "interpretable"],
            "not_recommended_for": ["non_linear", "correlated_features"]
        },
        
        # Tree-based Models
        "decision_tree": {
            "category": ModelCategory.TREE_BASED,
            "interpretability": 4,
            "speed": 4,
            "memory": 3,
            "scalability": 3,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": False,
            "best_for": ["interpretable", "non_linear", "mixed_features"],
            "not_recommended_for": ["small_data", "high_variance"]
        },
        "random_forest": {
            "category": ModelCategory.ENSEMBLE,
            "interpretability": 3,
            "speed": 3,
            "memory": 2,
            "scalability": 3,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["non_linear", "robust", "feature_importance", "general_purpose"],
            "not_recommended_for": ["very_large_data", "real_time"]
        },
        "extra_trees": {
            "category": ModelCategory.ENSEMBLE,
            "interpretability": 3,
            "speed": 4,
            "memory": 2,
            "scalability": 3,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["non_linear", "fast_training", "variance_reduction"],
            "not_recommended_for": ["very_large_data", "interpretable"]
        },
        
        # Boosting Models
        "xgboost": {
            "category": ModelCategory.BOOSTING,
            "interpretability": 2,
            "speed": 3,
            "memory": 3,
            "scalability": 4,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["high_performance", "competitions", "structured_data", "general_purpose"],
            "not_recommended_for": ["simple_problems", "need_interpretability"]
        },
        "lightgbm": {
            "category": ModelCategory.BOOSTING,
            "interpretability": 2,
            "speed": 5,
            "memory": 4,
            "scalability": 5,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["large_data", "fast_training", "high_performance", "memory_efficient"],
            "not_recommended_for": ["small_data", "overfitting_prone"]
        },
        "catboost": {
            "category": ModelCategory.BOOSTING,
            "interpretability": 2,
            "speed": 3,
            "memory": 3,
            "scalability": 4,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": True,
            "best_for": ["categorical_heavy", "high_performance", "minimal_tuning"],
            "not_recommended_for": ["small_data", "simple_problems"]
        },
        "gradient_boosting": {
            "category": ModelCategory.BOOSTING,
            "interpretability": 2,
            "speed": 2,
            "memory": 3,
            "scalability": 3,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": True,
            "best_for": ["high_performance", "careful_tuning"],
            "not_recommended_for": ["large_data", "fast_training"]
        },
        
        # Neural Networks
        "mlp": {
            "category": ModelCategory.NEURAL_NETWORK,
            "interpretability": 1,
            "speed": 2,
            "memory": 2,
            "scalability": 3,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": False,
            "best_for": ["complex_patterns", "large_data", "non_linear"],
            "not_recommended_for": ["small_data", "interpretable", "fast_training"]
        },
        
        # SVM
        "svm": {
            "category": ModelCategory.SVM,
            "interpretability": 2,
            "speed": 2,
            "memory": 2,
            "scalability": 2,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": False,
            "best_for": ["small_data", "high_dimensional", "kernel_tricks"],
            "not_recommended_for": ["large_data", "many_features"]
        },
        
        # Others
        "knn": {
            "category": ModelCategory.KNN,
            "interpretability": 4,
            "speed": 2,
            "memory": 1,
            "scalability": 1,
            "handles_categorical": False,
            "handles_missing": False,
            "handles_imbalance": True,
            "best_for": ["small_data", "simple_patterns", "local_structures"],
            "not_recommended_for": ["large_data", "high_dimensional", "fast_prediction"]
        },
        "naive_bayes": {
            "category": ModelCategory.NAIVE_BAYES,
            "interpretability": 4,
            "speed": 5,
            "memory": 5,
            "scalability": 5,
            "handles_categorical": True,
            "handles_missing": True,
            "handles_imbalance": False,
            "best_for": ["text_classification", "fast_training", "small_data"],
            "not_recommended_for": ["correlated_features", "complex_patterns"]
        }
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model."""
        return cls.MODELS.get(model_name.lower())
    
    @classmethod
    def get_all_models(cls) -> List[str]:
        """Get list of all available models."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_models_by_category(cls, category: ModelCategory) -> List[str]:
        """Get models in a specific category."""
        return [name for name, info in cls.MODELS.items() 
                if info["category"] == category]


class ModelSelector:
    """
    Intelligent model selection system.
    
    Analyzes dataset characteristics and recommends suitable models
    based on multiple criteria.
    """
    
    def __init__(self, criterion: SelectionCriterion = SelectionCriterion.BALANCED):
        """Initialize model selector."""
        self.criterion = criterion
        self.result: Optional[SelectionResult] = None
        
        # Weights for different criteria
        self.criterion_weights = self._get_criterion_weights()
        
        logger.info(f"ModelSelector initialized with criterion: {criterion}")
    
    def _get_criterion_weights(self) -> Dict[str, float]:
        """Get weights for different criteria."""
        weights = {
            SelectionCriterion.ACCURACY: {
                "performance": 0.7, "speed": 0.1, "interpretability": 0.05,
                "robustness": 0.1, "scalability": 0.025, "memory": 0.025
            },
            SelectionCriterion.SPEED: {
                "performance": 0.3, "speed": 0.4, "interpretability": 0.05,
                "robustness": 0.1, "scalability": 0.1, "memory": 0.05
            },
            SelectionCriterion.INTERPRETABILITY: {
                "performance": 0.2, "speed": 0.1, "interpretability": 0.5,
                "robustness": 0.1, "scalability": 0.05, "memory": 0.05
            },
            SelectionCriterion.ROBUSTNESS: {
                "performance": 0.3, "speed": 0.05, "interpretability": 0.05,
                "robustness": 0.4, "scalability": 0.1, "memory": 0.1
            },
            SelectionCriterion.SCALABILITY: {
                "performance": 0.25, "speed": 0.15, "interpretability": 0.05,
                "robustness": 0.1, "scalability": 0.35, "memory": 0.1
            },
            SelectionCriterion.MEMORY: {
                "performance": 0.25, "speed": 0.1, "interpretability": 0.05,
                "robustness": 0.1, "scalability": 0.1, "memory": 0.4
            },
            SelectionCriterion.BALANCED: {
                "performance": 0.3, "speed": 0.15, "interpretability": 0.15,
                "robustness": 0.15, "scalability": 0.15, "memory": 0.1
            }
        }
        return weights[self.criterion]
    
    def select_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: Optional[str] = None,
        n_models: int = 5,
        quick_mode: bool = False
    ) -> SelectionResult:
        """
        Select best models for the dataset.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            problem_type: 'classification' or 'regression' (auto-detected if None)
            n_models: Number of models to recommend
            quick_mode: If True, skip performance evaluation
        
        Returns:
            SelectionResult with recommended models
        """
        logger.info("Starting model selection...")
        start_time = datetime.now()
        
        # Extract dataset characteristics
        characteristics = self._extract_characteristics(X, y)
        
        # Detect problem type
        if problem_type is None:
            problem_type = self._detect_problem_type(y, characteristics)
        
        problem_enum = self._map_problem_type(problem_type, characteristics)
        
        logger.info(f"Problem type: {problem_enum}")
        logger.info(f"Dataset: {characteristics.n_samples} samples, {characteristics.n_features} features")
        
        # Get applicable models
        applicable_models = self._get_applicable_models(problem_type)
        
        logger.info(f"Evaluating {len(applicable_models)} models...")
        
        # Score each model
        model_scores = []
        for model_name in applicable_models:
            score = self._score_model(
                model_name,
                characteristics,
                problem_enum,
                quick_mode
            )
            model_scores.append(score)
        
        # Sort by total score
        model_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Get top N recommended models
        recommended = [score.model_name for score in model_scores[:n_models]]
        
        # Create result
        selection_time = (datetime.now() - start_time).total_seconds()
        
        self.result = SelectionResult(
            recommended_models=recommended,
            model_scores=model_scores,
            dataset_characteristics=characteristics,
            selection_criterion=self.criterion,
            problem_type=problem_enum,
            selection_time=selection_time,
            total_models_evaluated=len(applicable_models),
            metadata={
                "quick_mode": quick_mode,
                "n_requested": n_models
            }
        )
        
        logger.info(f"Selection completed in {selection_time:.2f}s")
        logger.info(f"Top 3 models: {recommended[:3]}")
        
        return self.result
    
    def _extract_characteristics(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> DatasetCharacteristics:
        """Extract meta-features from dataset."""
        
        n_samples, n_features = X.shape
        
        # Feature types
        n_numerical = len(X.select_dtypes(include=[np.number]).columns)
        n_categorical = len(X.select_dtypes(include=['object', 'category']).columns)
        
        # Binary features
        n_binary = sum((X[col].nunique() == 2) for col in X.columns)
        
        # Class information
        n_classes = None
        class_balance = None
        if y.dtype == 'object' or len(y.unique()) < 20:  # Categorical
            n_classes = y.nunique()
            class_counts = y.value_counts(normalize=True)
            class_balance = class_counts.to_dict()
        
        # Statistical properties (numerical features only)
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            avg_skewness = np.mean([abs(skew(X[col].dropna())) for col in numerical_cols])
            avg_kurtosis = np.mean([abs(kurtosis(X[col].dropna())) for col in numerical_cols])
        else:
            avg_skewness = 0
            avg_kurtosis = 0
        
        # Missing values
        missing_ratio = X.isnull().sum().sum() / (n_samples * n_features)
        
        # Complexity measures
        feature_to_sample_ratio = n_features / n_samples
        
        # Correlation (numerical only)
        if len(numerical_cols) > 1:
            corr_matrix = X[numerical_cols].corr().abs()
            # Get upper triangle
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            avg_correlation = upper_triangle.stack().mean()
        else:
            avg_correlation = 0
        
        # Dimensionality score
        dimensionality_score = np.log10(n_features) / np.log10(n_samples) if n_samples > 0 else 0
        
        # Problem characteristics
        is_imbalanced = False
        if class_balance:
            min_class_ratio = min(class_balance.values())
            is_imbalanced = min_class_ratio < 0.2
        
        is_high_dimensional = feature_to_sample_ratio > 0.1
        is_sparse = missing_ratio > 0.3 or (X == 0).sum().sum() / (n_samples * n_features) > 0.5
        
        return DatasetCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_balance=class_balance,
            n_numerical=n_numerical,
            n_categorical=n_categorical,
            n_binary=n_binary,
            avg_skewness=float(avg_skewness),
            avg_kurtosis=float(avg_kurtosis),
            missing_ratio=float(missing_ratio),
            feature_to_sample_ratio=float(feature_to_sample_ratio),
            avg_correlation=float(avg_correlation),
            dimensionality_score=float(dimensionality_score),
            is_imbalanced=is_imbalanced,
            is_high_dimensional=is_high_dimensional,
            is_sparse=is_sparse
        )
    
    def _detect_problem_type(
        self,
        y: pd.Series,
        characteristics: DatasetCharacteristics
    ) -> str:
        """Detect problem type from target variable."""
        n_unique = y.nunique()
        
        # Check if continuous
        if y.dtype in ['float64', 'float32'] and n_unique > 20:
            return "regression"
        
        # Check if classification
        if y.dtype == 'object' or n_unique < 20:
            return "classification"
        
        # Default to regression for numeric with many unique values
        return "regression"
    
    def _map_problem_type(
        self,
        problem_type: str,
        characteristics: DatasetCharacteristics
    ) -> ProblemType:
        """Map problem type string to enum."""
        if problem_type == "regression":
            return ProblemType.REGRESSION
        
        if characteristics.n_classes == 2:
            if characteristics.is_imbalanced:
                return ProblemType.IMBALANCED
            return ProblemType.BINARY_CLASSIFICATION
        
        if characteristics.is_high_dimensional:
            return ProblemType.HIGH_DIMENSIONAL
        
        return ProblemType.MULTICLASS_CLASSIFICATION
    
    def _get_applicable_models(self, problem_type: str) -> List[str]:
        """Get models applicable for the problem type."""
        if problem_type == "regression":
            return [
                "ridge", "lasso", "random_forest", "extra_trees",
                "xgboost", "lightgbm", "catboost", "gradient_boosting",
                "mlp", "svm"
            ]
        else:  # classification
            return [
                "logistic_regression", "decision_tree", "random_forest",
                "extra_trees", "xgboost", "lightgbm", "catboost",
                "gradient_boosting", "mlp", "svm", "knn", "naive_bayes"
            ]
    
    def _score_model(
        self,
        model_name: str,
        characteristics: DatasetCharacteristics,
        problem_type: ProblemType,
        quick_mode: bool
    ) -> ModelScore:
        """Score a model based on dataset characteristics."""
        
        model_info = ModelRegistry.get_model_info(model_name)
        if not model_info:
            logger.warning(f"No info for model: {model_name}")
            return ModelScore(
                model_name=model_name,
                model_category=ModelCategory.LINEAR,
                performance_score=0,
                speed_score=0,
                interpretability_score=0,
                robustness_score=0,
                scalability_score=0,
                memory_score=0,
                total_score=0,
                recommended=False,
                reasoning=["Unknown model"]
            )
        
        # Base scores from registry
        interpretability_score = model_info["interpretability"] / 5.0
        speed_score = model_info["speed"] / 5.0
        memory_score = model_info["memory"] / 5.0
        scalability_score = model_info["scalability"] / 5.0
        
        # Adjust scores based on dataset characteristics
        performance_score = self._estimate_performance(model_name, model_info, characteristics)
        robustness_score = self._estimate_robustness(model_name, model_info, characteristics)
        
        # Reasoning
        reasoning = []
        warnings = []
        
        # Check strengths
        if characteristics.n_categorical > 0 and model_info["handles_categorical"]:
            reasoning.append("✓ Handles categorical features well")
        
        if characteristics.missing_ratio > 0.1 and model_info["handles_missing"]:
            reasoning.append("✓ Robust to missing values")
        
        if characteristics.is_imbalanced and model_info["handles_imbalance"]:
            reasoning.append("✓ Good for imbalanced data")
        
        if characteristics.n_samples > 100000 and scalability_score > 0.7:
            reasoning.append("✓ Scalable to large datasets")
        
        # Check weaknesses
        if characteristics.n_samples < 1000 and "small_data" in model_info.get("not_recommended_for", []):
            warnings.append("⚠ May underperform on small datasets")
            performance_score *= 0.8
        
        if characteristics.is_high_dimensional and "high_dimensional" in model_info.get("not_recommended_for", []):
            warnings.append("⚠ May struggle with high dimensionality")
            performance_score *= 0.7
        
        if characteristics.n_categorical > characteristics.n_numerical and not model_info["handles_categorical"]:
            warnings.append("⚠ Requires categorical encoding")
            performance_score *= 0.9
        
        # Calculate total score
        weights = self.criterion_weights
        total_score = (
            weights["performance"] * performance_score +
            weights["speed"] * speed_score +
            weights["interpretability"] * interpretability_score +
            weights["robustness"] * robustness_score +
            weights["scalability"] * scalability_score +
            weights["memory"] * memory_score
        )
        
        # Recommended if score > threshold
        recommended = total_score > 0.6 and len(warnings) < 2
        
        return ModelScore(
            model_name=model_name,
            model_category=model_info["category"],
            performance_score=performance_score,
            speed_score=speed_score,
            interpretability_score=interpretability_score,
            robustness_score=robustness_score,
            scalability_score=scalability_score,
            memory_score=memory_score,
            total_score=total_score,
            recommended=recommended,
            reasoning=reasoning,
            warnings=warnings
        )
    
    def _estimate_performance(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        characteristics: DatasetCharacteristics
    ) -> float:
        """Estimate expected performance based on characteristics."""
        
        base_score = 0.6  # Start with 60%
        
        # Adjust based on best_for and not_recommended_for
        if characteristics.n_samples < 1000:
            if "small_data" in model_info.get("best_for", []):
                base_score += 0.2
            if "small_data" in model_info.get("not_recommended_for", []):
                base_score -= 0.2
        
        if characteristics.n_samples > 100000:
            if "large_data" in model_info.get("best_for", []):
                base_score += 0.2
            if "large_data" in model_info.get("not_recommended_for", []):
                base_score -= 0.2
        
        if characteristics.is_high_dimensional:
            if "high_dimensional" in model_info.get("best_for", []):
                base_score += 0.15
            if "high_dimensional" in model_info.get("not_recommended_for", []):
                base_score -= 0.15
        
        if characteristics.n_categorical > characteristics.n_numerical / 2:
            if "categorical_heavy" in model_info.get("best_for", []):
                base_score += 0.15
        
        # Tree-based and boosting models generally perform well
        if model_info["category"] in [ModelCategory.BOOSTING, ModelCategory.ENSEMBLE]:
            base_score += 0.1
        
        return np.clip(base_score, 0, 1)
    
    def _estimate_robustness(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        characteristics: DatasetCharacteristics
    ) -> float:
        """Estimate model robustness."""
        
        base_score = 0.5
        
        # Ensemble models are generally more robust
        if model_info["category"] in [ModelCategory.ENSEMBLE, ModelCategory.BOOSTING]:
            base_score += 0.3
        
        # Missing values handling
        if characteristics.missing_ratio > 0.1:
            if model_info["handles_missing"]:
                base_score += 0.1
            else:
                base_score -= 0.1
        
        # Outliers and noise
        if characteristics.avg_skewness > 2 or characteristics.avg_kurtosis > 3:
            if model_info["category"] in [ModelCategory.TREE_BASED, ModelCategory.ENSEMBLE]:
                base_score += 0.1
        
        return np.clip(base_score, 0, 1)
    
    def get_explanation(self, model_name: Optional[str] = None) -> str:
        """Get detailed explanation of recommendations."""
        if self.result is None:
            return "No selection result available. Run select_models() first."
        
        if model_name:
            # Explanation for specific model
            score = next((s for s in self.result.model_scores if s.model_name == model_name), None)
            if not score:
                return f"Model {model_name} not found in results."
            
            explanation = f"""
🤖 Model: {score.model_name.upper()}
Category: {score.category.value}

📊 Scores:
  • Performance: {score.performance_score:.2f}
  • Speed: {score.speed_score:.2f}
  • Interpretability: {score.interpretability_score:.2f}
  • Robustness: {score.robustness_score:.2f}
  • Scalability: {score.scalability_score:.2f}
  • Memory: {score.memory_score:.2f}
  
  ⭐ Total Score: {score.total_score:.2f}

✅ Strengths:
{chr(10).join(f"  {r}" for r in score.reasoning)}

{"⚠️ Considerations:" if score.warnings else ""}
{chr(10).join(f"  {w}" for w in score.warnings)}

{"✅ RECOMMENDED" if score.recommended else "❌ NOT RECOMMENDED"}
"""
            return explanation
        
        else:
            # Overall explanation
            explanation = f"""
🎯 MODEL SELECTION RESULTS

📊 Dataset Characteristics:
  • Samples: {self.result.dataset_characteristics.n_samples:,}
  • Features: {self.result.dataset_characteristics.n_features}
  • Problem: {self.result.problem_type.value}
  • Selection Time: {self.result.selection_time:.2f}s

🏆 Top Recommended Models:
"""
            for i, model_name in enumerate(self.result.recommended_models[:5], 1):
                score = next(s for s in self.result.model_scores if s.model_name == model_name)
                explanation += f"  {i}. {model_name.upper()} (score: {score.total_score:.2f})\n"
            
            return explanation
    
    def export_report(self, filepath: Union[str, Path]):
        """Export detailed selection report."""
        if self.result is None:
            raise ValueError("No result to export. Run select_models() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.result.to_dict(), f, indent=2)
        
        logger.info(f"Report exported to {filepath}")


# Convenience function
def select_best_models(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: SelectionCriterion = SelectionCriterion.BALANCED,
    n_models: int = 5,
    **kwargs
) -> SelectionResult:
    """
    Convenience function for model selection.
    
    Usage:
        result = select_best_models(
            X=X_train,
            y=y_train,
            criterion=SelectionCriterion.ACCURACY,
            n_models=5
        )
    """
    selector = ModelSelector(criterion)
    return selector.select_models(X, y, n_models=n_models, **kwargs)