"""
DataGenius PRO - ML Orchestrator
Orchestrates machine learning pipeline
"""

import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger
from core.base_agent import PipelineAgent, AgentResult


class MLOrchestrator(PipelineAgent):
    """
    Orchestrates complete ML training pipeline
    """
    
    def __init__(self):
        # Import ML agents
        from agents.ml.model_selector import ModelSelector
        from agents.ml.model_trainer import ModelTrainer
        from agents.ml.model_evaluator import ModelEvaluator
        from agents.ml.model_explainer import ModelExplainer
        
        # Create agent pipeline
        agents = [
            ModelSelector(),
            ModelTrainer(),
            ModelEvaluator(),
            ModelExplainer(),
        ]
        
        super().__init__(
            name="MLOrchestrator",
            agents=agents,
            description="Complete ML training pipeline"
        )
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: str,
        **kwargs
    ) -> AgentResult:
        """
        Execute complete ML pipeline
        
        Args:
            data: Preprocessed DataFrame
            target_column: Target column name
            problem_type: classification or regression
        
        Returns:
            AgentResult with ML results
        """
        
        self.logger.info(f"Starting ML pipeline for {problem_type}")
        
        # Run pipeline
        result = super().execute(
            data=data,
            target_column=target_column,
            problem_type=problem_type,
            **kwargs
        )
        
        if result.is_success():
            # Aggregate results
            ml_results = self._aggregate_ml_results(result.data["pipeline_results"])
            
            # Generate summary
            summary = self._generate_ml_summary(ml_results)
            
            result.data = {
                "ml_results": ml_results,
                "summary": summary,
            }
            
            self.logger.success("ML pipeline completed successfully")
        
        return result
    
    def _aggregate_ml_results(self, pipeline_results: list) -> Dict[str, Any]:
        """Aggregate results from all ML agents"""
        
        aggregated = {}
        
        for agent_result in pipeline_results:
            agent_name = agent_result.agent_name
            aggregated[agent_name] = agent_result.data
        
        return aggregated
    
    def _generate_ml_summary(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of ML results"""
        
        summary = {
            "best_model": None,
            "best_score": None,
            "models_trained": 0,
            "key_insights": [],
        }
        
        # Extract best model info
        if "ModelEvaluator" in ml_results:
            eval_results = ml_results["ModelEvaluator"]
            summary["best_model"] = eval_results.get("best_model_name")
            summary["best_score"] = eval_results.get("best_score")
        
        # Count models
        if "ModelTrainer" in ml_results:
            trainer_results = ml_results["ModelTrainer"]
            summary["models_trained"] = len(trainer_results.get("models_comparison", []))
        
        # Extract insights from explainer
        if "ModelExplainer" in ml_results:
            explainer_results = ml_results["ModelExplainer"]
            top_features = explainer_results.get("top_features", [])
            if top_features:
                summary["key_insights"].append(
                    f"Najważniejsze cechy: {', '.join(top_features[:3])}"
                )
        
        return summary