"""
DataGenius PRO - Model Evaluator
Evaluates trained models with comprehensive metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
from loguru import logger
from core.base_agent import BaseAgent, AgentResult


class ModelEvaluator(BaseAgent):
    """
    Evaluates ML models with comprehensive metrics
    """
    
    def __init__(self):
        super().__init__(
            name="ModelEvaluator",
            description="Evaluates trained models"
        )
    
    def execute(
        self,
        best_model: Any,
        pycaret_wrapper: Any,
        problem_type: str,
        data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> AgentResult:
        """
        Evaluate model
        
        Args:
            best_model: Trained model
            pycaret_wrapper: PyCaret wrapper instance
            problem_type: classification or regression
            data: Test data (optional)
        
        Returns:
            AgentResult with evaluation metrics
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Get predictions
            if data is not None:
                predictions = pycaret_wrapper.predict_model(best_model, data=data)
            else:
                predictions = pycaret_wrapper.predict_model(best_model)
            
            # Calculate metrics based on problem type
            if problem_type == "classification":
                metrics = self._evaluate_classification(predictions)
            else:
                metrics = self._evaluate_regression(predictions)
            
            # Get best score
            if problem_type == "classification":
                best_score = metrics.get("accuracy", 0)
            else:
                best_score = metrics.get("r2", 0)
            
            result.data = {
                "metrics": metrics,
                "predictions": predictions,
                "best_model_name": str(type(best_model).__name__),
                "best_score": best_score,
                "problem_type": problem_type,
            }
            
            self.logger.success(
                f"Model evaluation complete: {problem_type} score = {best_score:.4f}"
            )
        
        except Exception as e:
            result.add_error(f"Model evaluation failed: {e}")
            self.logger.error(f"Model evaluation error: {e}", exc_info=True)
        
        return result
    
    def _evaluate_classification(self, predictions: pd.DataFrame) -> Dict[str, float]:
        """Evaluate classification model"""
        
        # Assuming PyCaret format with 'Label' column
        if 'Label' not in predictions.columns:
            self.logger.warning("Could not find prediction columns")
            return {}
        
        y_true = predictions.iloc[:, 0]  # First column is usually true label
        y_pred = predictions['Label']
        
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        except:
            pass
        
        try:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            pass
        
        try:
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            pass
        
        try:
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            pass
        
        # Try ROC AUC (for binary/probability predictions)
        try:
            if 'Score' in predictions.columns:
                metrics['roc_auc'] = roc_auc_score(y_true, predictions['Score'])
        except:
            pass
        
        return metrics
    
    def _evaluate_regression(self, predictions: pd.DataFrame) -> Dict[str, float]:
        """Evaluate regression model"""
        
        # Assuming PyCaret format with 'Label' column
        if 'Label' not in predictions.columns:
            self.logger.warning("Could not find prediction columns")
            return {}
        
        y_true = predictions.iloc[:, 0]  # First column is usually true label
        y_pred = predictions['Label']
        
        metrics = {}
        
        try:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        except:
            pass
        
        try:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
        except:
            pass
        
        try:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        except:
            pass
        
        try:
            metrics['r2'] = r2_score(y_true, y_pred)
        except:
            pass
        
        try:
            # MAPE
            mask = y_true != 0
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        except:
            pass
        
        return metrics