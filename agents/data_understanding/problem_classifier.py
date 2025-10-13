"""
DataGenius PRO - Problem Classifier
Classifies ML problem type (classification vs regression)
"""

import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from core.utils import infer_problem_type
from config.model_registry import ProblemType


class ProblemClassifier(BaseAgent):
    """
    Classifies the ML problem type based on target column
    """
    
    def __init__(self):
        super().__init__(
            name="ProblemClassifier",
            description="Classifies ML problem as classification or regression"
        )
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters"""
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        
        if "target_column" not in kwargs:
            raise ValueError("'target_column' parameter is required")
        
        return True
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> AgentResult:
        """
        Classify problem type
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
        
        Returns:
            AgentResult with problem classification
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            if target_column not in data.columns:
                result.add_error(f"Target column '{target_column}' not found")
                return result
            
            target = data[target_column]
            
            # Detect problem type
            problem_type = infer_problem_type(target)
            
            # Get detailed analysis
            analysis = self._analyze_target(target, problem_type)
            
            # Get recommendations
            recommendations = self._get_recommendations(target, problem_type)
            
            result.data = {
                "problem_type": problem_type,
                "target_analysis": analysis,
                "recommendations": recommendations,
            }
            
            self.logger.success(f"Problem classified as: {problem_type}")
        
        except Exception as e:
            result.add_error(f"Problem classification failed: {e}")
            self.logger.error(f"Problem classification error: {e}", exc_info=True)
        
        return result
    
    def _analyze_target(
        self,
        target: pd.Series,
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Analyze target column in detail
        
        Args:
            target: Target series
            problem_type: Detected problem type
        
        Returns:
            Analysis dictionary
        """
        
        analysis = {
            "n_samples": len(target),
            "n_unique": target.nunique(),
            "n_missing": target.isnull().sum(),
            "missing_pct": target.isnull().sum() / len(target) * 100,
        }
        
        if problem_type == "classification":
            analysis.update(self._analyze_classification_target(target))
        else:
            analysis.update(self._analyze_regression_target(target))
        
        return analysis
    
    def _analyze_classification_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze classification target"""
        
        value_counts = target.value_counts()
        n_classes = len(value_counts)
        
        # Check for binary classification
        is_binary = n_classes == 2
        
        # Check for class imbalance
        if n_classes > 1:
            min_class = value_counts.min()
            max_class = value_counts.max()
            imbalance_ratio = max_class / min_class
            is_imbalanced = imbalance_ratio > 3
        else:
            imbalance_ratio = 1.0
            is_imbalanced = False
        
        return {
            "classification_type": "binary" if is_binary else "multiclass",
            "n_classes": n_classes,
            "class_distribution": value_counts.to_dict(),
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": float(imbalance_ratio),
            "majority_class": str(value_counts.index[0]),
            "majority_class_pct": float(value_counts.iloc[0] / len(target) * 100),
        }
    
    def _analyze_regression_target(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze regression target"""
        
        # Clean target (remove NaN)
        target_clean = target.dropna()
        
        if len(target_clean) == 0:
            return {
                "error": "All target values are missing"
            }
        
        return {
            "mean": float(target_clean.mean()),
            "std": float(target_clean.std()),
            "min": float(target_clean.min()),
            "max": float(target_clean.max()),
            "median": float(target_clean.median()),
            "q25": float(target_clean.quantile(0.25)),
            "q75": float(target_clean.quantile(0.75)),
            "skewness": float(target_clean.skew()),
            "kurtosis": float(target_clean.kurtosis()),
            "range": float(target_clean.max() - target_clean.min()),
            "cv": float(target_clean.std() / target_clean.mean()) if target_clean.mean() != 0 else None,
        }
    
    def _get_recommendations(
        self,
        target: pd.Series,
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Get recommendations based on problem type and target analysis
        
        Args:
            target: Target series
            problem_type: Problem type
        
        Returns:
            Recommendations dictionary
        """
        
        recommendations = {
            "problem_type": problem_type,
            "suggested_metrics": [],
            "preprocessing_steps": [],
            "model_suggestions": [],
            "warnings": [],
        }
        
        if problem_type == "classification":
            recommendations.update(
                self._get_classification_recommendations(target)
            )
        else:
            recommendations.update(
                self._get_regression_recommendations(target)
            )
        
        return recommendations
    
    def _get_classification_recommendations(
        self,
        target: pd.Series
    ) -> Dict[str, Any]:
        """Get recommendations for classification"""
        
        value_counts = target.value_counts()
        n_classes = len(value_counts)
        is_binary = n_classes == 2
        
        # Metrics
        if is_binary:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        else:
            metrics = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
        
        # Preprocessing
        preprocessing = [
            "Handle missing values in target",
            "Encode categorical target if needed",
        ]
        
        # Check for imbalance
        if n_classes > 1:
            min_class = value_counts.min()
            max_class = value_counts.max()
            imbalance_ratio = max_class / min_class
            
            if imbalance_ratio > 3:
                preprocessing.append("Apply SMOTE or class weighting for imbalanced data")
                metrics.append("balanced_accuracy")
        
        # Model suggestions
        if is_binary:
            models = [
                "Logistic Regression (baseline)",
                "Random Forest",
                "XGBoost",
                "LightGBM",
            ]
        else:
            models = [
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "CatBoost",
            ]
        
        return {
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
        }
    
    def _get_regression_recommendations(
        self,
        target: pd.Series
    ) -> Dict[str, Any]:
        """Get recommendations for regression"""
        
        target_clean = target.dropna()
        
        # Metrics
        metrics = ["mae", "mse", "rmse", "r2"]
        
        # Preprocessing
        preprocessing = [
            "Handle missing values in target",
            "Check for outliers in target",
        ]
        
        # Check skewness
        skewness = abs(target_clean.skew())
        if skewness > 1:
            preprocessing.append(
                f"Consider log transformation (skewness: {skewness:.2f})"
            )
        
        # Model suggestions
        models = [
            "Linear Regression (baseline)",
            "Ridge/Lasso Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM",
        ]
        
        return {
            "suggested_metrics": metrics,
            "preprocessing_steps": preprocessing,
            "model_suggestions": models,
        }