"""
DataGenius PRO - EDA Orchestrator
Orchestrates all EDA agents and generates comprehensive analysis
"""

import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger
from core.base_agent import PipelineAgent, AgentResult


class EDAOrchestrator(PipelineAgent):
    """
    Orchestrates all EDA agents to provide comprehensive exploratory analysis
    """
    
    def __init__(self):
        # Import EDA agents
        from agents.eda.statistical_analysis import StatisticalAnalyzer
        from agents.eda.visualization_engine import VisualizationEngine
        from agents.eda.missing_data_analyzer import MissingDataAnalyzer
        from agents.eda.outlier_detector import OutlierDetector
        from agents.eda.correlation_analyzer import CorrelationAnalyzer
        
        # Create agent pipeline
        agents = [
            StatisticalAnalyzer(),
            MissingDataAnalyzer(),
            OutlierDetector(),
            CorrelationAnalyzer(),
            VisualizationEngine(),
        ]
        
        super().__init__(
            name="EDAOrchestrator",
            agents=agents,
            description="Comprehensive exploratory data analysis pipeline"
        )
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute complete EDA pipeline
        
        Args:
            data: Input DataFrame
            target_column: Target column name (optional)
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with comprehensive EDA results
        """
        
        self.logger.info("Starting comprehensive EDA analysis")
        
        # Run pipeline
        result = super().execute(
            data=data,
            target_column=target_column,
            **kwargs
        )
        
        if result.is_success():
            # Aggregate results from all agents
            eda_results = self._aggregate_results(result.data["pipeline_results"])
            
            # Generate summary
            summary = self._generate_summary(eda_results, data)
            
            result.data = {
                "eda_results": eda_results,
                "summary": summary,
            }
            
            self.logger.success("EDA analysis completed successfully")
        
        return result
    
    def _aggregate_results(self, pipeline_results: list) -> Dict[str, Any]:
        """Aggregate results from all EDA agents"""
        
        aggregated = {}
        
        for agent_result in pipeline_results:
            agent_name = agent_result.agent_name
            aggregated[agent_name] = agent_result.data
        
        return aggregated
    
    def _generate_summary(
        self,
        eda_results: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate summary of EDA findings"""
        
        summary = {
            "dataset_shape": data.shape,
            "key_findings": [],
            "data_quality": "good",  # Will be calculated
            "recommendations": [],
        }
        
        # Extract key findings from each agent
        
        # Statistical findings
        if "StatisticalAnalyzer" in eda_results:
            stats = eda_results["StatisticalAnalyzer"]
            summary["key_findings"].append(
                f"Dataset ma {len(data)} wierszy i {len(data.columns)} kolumn"
            )
        
        # Missing data findings
        if "MissingDataAnalyzer" in eda_results:
            missing = eda_results["MissingDataAnalyzer"]
            total_missing = missing.get("summary", {}).get("total_missing", 0)
            if total_missing > 0:
                summary["key_findings"].append(
                    f"Znaleziono {total_missing} brakujących wartości"
                )
                summary["recommendations"].append(
                    "Rozważ imputację lub usunięcie brakujących wartości"
                )
        
        # Outlier findings
        if "OutlierDetector" in eda_results:
            outliers = eda_results["OutlierDetector"]
            n_outliers = outliers.get("summary", {}).get("total_outliers", 0)
            if n_outliers > 0:
                summary["key_findings"].append(
                    f"Wykryto {n_outliers} outliers w danych numerycznych"
                )
        
        # Correlation findings
        if "CorrelationAnalyzer" in eda_results:
            corr = eda_results["CorrelationAnalyzer"]
            high_corr = corr.get("high_correlations", [])
            if high_corr:
                summary["key_findings"].append(
                    f"Znaleziono {len(high_corr)} par silnie skorelowanych cech"
                )
                summary["recommendations"].append(
                    "Rozważ usunięcie silnie skorelowanych cech"
                )
        
        return summary