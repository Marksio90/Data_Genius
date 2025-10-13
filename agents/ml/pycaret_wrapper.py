"""
DataGenius PRO - PyCaret Wrapper
Wrapper for PyCaret AutoML functionality
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from loguru import logger
from config.settings import settings
from config.model_registry import get_models_for_problem, ProblemType


class PyCaretWrapper:
    """
    Wrapper for PyCaret with unified interface
    """
    
    def __init__(self, problem_type: str):
        """
        Initialize PyCaret wrapper
        
        Args:
            problem_type: 'classification' or 'regression'
        """
        self.problem_type = problem_type
        self.logger = logger.bind(component="PyCaretWrapper")
        
        # Import appropriate module
        if problem_type == "classification":
            from pycaret.classification import (
                setup, compare_models, tune_model, finalize_model,
                predict_model, plot_model, save_model, load_model
            )
        else:
            from pycaret.regression import (
                setup, compare_models, tune_model, finalize_model,
                predict_model, plot_model, save_model, load_model
            )
        
        self.setup = setup
        self.compare_models = compare_models
        self.tune_model = tune_model
        self.finalize_model = finalize_model
        self.predict_model = predict_model
        self.plot_model = plot_model
        self.save_model = save_model
        self.load_model = load_model
        
        self.experiment = None
    
    def initialize_experiment(
        self,
        data: pd.DataFrame,
        target: str,
        **kwargs
    ) -> Any:
        """
        Initialize PyCaret experiment
        
        Args:
            data: Training data
            target: Target column name
            **kwargs: Additional setup parameters
        
        Returns:
            PyCaret experiment object
        """
        
        try:
            self.logger.info("Initializing PyCaret experiment")
            
            # Setup parameters
            setup_params = {
                'data': data,
                'target': target,
                'session_id': settings.PYCARET_SESSION_ID,
                'verbose': settings.PYCARET_VERBOSE,
                'n_jobs': settings.PYCARET_N_JOBS,
                'use_gpu': False,  # Set to True if GPU available
            }
            
            # Problem-specific parameters
            if self.problem_type == "classification":
                setup_params.update({
                    'fix_imbalance': True,  # Handle imbalanced data
                    'remove_multicollinearity': True,
                    'multicollinearity_threshold': 0.9,
                })
            else:
                setup_params.update({
                    'normalize': True,
                    'remove_multicollinearity': True,
                    'multicollinearity_threshold': 0.9,
                })
            
            # Override with user params
            setup_params.update(kwargs)
            
            # Initialize
            self.experiment = self.setup(**setup_params)
            
            self.logger.success("PyCaret experiment initialized")
            return self.experiment
        
        except Exception as e:
            self.logger.error(f"PyCaret setup failed: {e}")
            raise
    
    def compare_all_models(
        self,
        n_select: int = 5,
        sort: str = 'auto',
        **kwargs
    ) -> tuple:
        """
        Compare all available models
        
        Args:
            n_select: Number of top models to return
            sort: Metric to sort by
            **kwargs: Additional compare_models parameters
        
        Returns:
            Tuple of (best_models_list, comparison_results)
        """
        
        try:
            self.logger.info(f"Comparing models (top {n_select})")
            
            # Get model list
            if self.problem_type == "classification":
                include_models = get_models_for_problem(
                    ProblemType.CLASSIFICATION,
                    strategy="accurate"
                )
            else:
                include_models = get_models_for_problem(
                    ProblemType.REGRESSION,
                    strategy="accurate"
                )
            
            # Compare
            best_models = self.compare_models(
                n_select=n_select,
                sort=sort,
                include=include_models,
                **kwargs
            )
            
            self.logger.success(f"Model comparison complete")
            return best_models, None  # comparison_results would be from pull()
        
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            raise
    
    def tune_best_model(
        self,
        model: Any,
        n_iter: int = 10,
        optimize: str = 'auto',
        **kwargs
    ) -> Any:
        """
        Tune hyperparameters of best model
        
        Args:
            model: Model to tune
            n_iter: Number of iterations
            optimize: Metric to optimize
            **kwargs: Additional tune_model parameters
        
        Returns:
            Tuned model
        """
        
        try:
            self.logger.info("Tuning model hyperparameters")
            
            tuned_model = self.tune_model(
                model,
                n_iter=n_iter,
                optimize=optimize,
                **kwargs
            )
            
            self.logger.success("Model tuning complete")
            return tuned_model
        
        except Exception as e:
            self.logger.error(f"Model tuning failed: {e}")
            raise
    
    def finalize_and_save(
        self,
        model: Any,
        model_path: str
    ) -> Any:
        """
        Finalize model and save
        
        Args:
            model: Model to finalize
            model_path: Path to save model
        
        Returns:
            Finalized model
        """
        
        try:
            self.logger.info("Finalizing model")
            
            # Finalize (train on full dataset)
            final_model = self.finalize_model(model)
            
            # Save
            self.save_model(final_model, model_path)
            
            self.logger.success(f"Model saved to {model_path}")
            return final_model
        
        except Exception as e:
            self.logger.error(f"Model finalization failed: {e}")
            raise
    
    def evaluate_model(
        self,
        model: Any,
        test_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model
        
        Args:
            model: Model to evaluate
            test_data: Test data (optional)
        
        Returns:
            Evaluation metrics
        """
        
        try:
            self.logger.info("Evaluating model")
            
            if test_data is not None:
                predictions = self.predict_model(model, data=test_data)
            else:
                predictions = self.predict_model(model)
            
            # Extract metrics
            # Note: Actual metric extraction depends on PyCaret version
            metrics = self._extract_metrics(predictions)
            
            self.logger.success("Model evaluation complete")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _extract_metrics(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Extract metrics from predictions"""
        
        # This is a simplified version
        # Actual implementation depends on PyCaret's output format
        
        return {
            "predictions": predictions,
            "shape": predictions.shape,
        }
    
    def get_feature_importance(self, model: Any) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            model: Trained model
        
        Returns:
            Feature importance DataFrame
        """
        
        try:
            # Try to plot feature importance
            self.plot_model(model, plot='feature', save=True)
            
            # Try to extract feature importance directly
            if hasattr(model, 'feature_importances_'):
                return pd.DataFrame({
                    'feature': self.experiment[0].feature_names_in_,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            return None
        
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None