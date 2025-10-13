"""
DataGenius PRO - Model Trainer
Trains and tunes ML models using PyCaret
"""

import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from agents.ml.pycaret_wrapper import PyCaretWrapper
from config.settings import settings


class ModelTrainer(BaseAgent):
    """
    Trains ML models using PyCaret
    """
    
    def __init__(self):
        super().__init__(
            name="ModelTrainer",
            description="Trains and tunes ML models"
        )
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: str,
        **kwargs
    ) -> AgentResult:
        """
        Train models
        
        Args:
            data: Training data
            target_column: Target column name
            problem_type: classification or regression
        
        Returns:
            AgentResult with trained models
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Initialize PyCaret
            pycaret = PyCaretWrapper(problem_type)
            
            # Setup experiment
            self.logger.info("Setting up PyCaret experiment")
            pycaret.initialize_experiment(data, target_column)
            
            # Compare models
            self.logger.info("Comparing models...")
            best_models = pycaret.compare_all_models(n_select=3)
            
            # Get best model
            if isinstance(best_models, list):
                best_model = best_models[0]
            else:
                best_model = best_models
            
            # Tune best model (if enabled)
            if settings.ENABLE_HYPERPARAMETER_TUNING:
                self.logger.info("Tuning best model...")
                tuned_model = pycaret.tune_best_model(
                    best_model,
                    n_iter=settings.DEFAULT_TUNING_ITERATIONS
                )
            else:
                tuned_model = best_model
            
            # Finalize model
            model_path = settings.MODELS_PATH / f"model_{problem_type}"
            final_model = pycaret.finalize_and_save(tuned_model, str(model_path))
            
            result.data = {
                "best_model": final_model,
                "model_path": str(model_path),
                "pycaret_wrapper": pycaret,
                "models_comparison": best_models,
            }
            
            self.logger.success("Model training completed")
        
        except Exception as e:
            result.add_error(f"Model training failed: {e}")
            self.logger.error(f"Model training error: {e}", exc_info=True)
        
        return result


class ModelSelector(BaseAgent):
    """
    Selects appropriate models based on problem type
    """
    
    def __init__(self):
        super().__init__(
            name="ModelSelector",
            description="Selects appropriate ML models"
        )
    
    def execute(
        self,
        problem_type: str,
        data: pd.DataFrame,
        **kwargs
    ) -> AgentResult:
        """
        Select models
        
        Args:
            problem_type: classification or regression
            data: Input data (for context)
        
        Returns:
            AgentResult with model selection
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            from config.model_registry import (
                get_models_for_problem,
                ProblemType,
                CLASSIFICATION_MODELS,
                REGRESSION_MODELS
            )
            
            # Get recommended models
            if problem_type == "classification":
                model_ids = get_models_for_problem(
                    ProblemType.CLASSIFICATION,
                    strategy="accurate"
                )
                model_registry = CLASSIFICATION_MODELS
            else:
                model_ids = get_models_for_problem(
                    ProblemType.REGRESSION,
                    strategy="accurate"
                )
                model_registry = REGRESSION_MODELS
            
            # Get model details
            selected_models = {
                model_id: model_registry[model_id]
                for model_id in model_ids
            }
            
            result.data = {
                "selected_models": selected_models,
                "model_ids": model_ids,
                "n_models": len(model_ids),
            }
            
            self.logger.success(f"Selected {len(model_ids)} models for {problem_type}")
        
        except Exception as e:
            result.add_error(f"Model selection failed: {e}")
            self.logger.error(f"Model selection error: {e}", exc_info=True)
        
        return result