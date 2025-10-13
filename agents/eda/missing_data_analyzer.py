"""
DataGenius PRO - Missing Data Analyzer
Analyzes missing data patterns and provides recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from loguru import logger
from core.base_agent import BaseAgent, AgentResult


class MissingDataAnalyzer(BaseAgent):
    """
    Analyzes missing data patterns and suggests handling strategies
    """
    
    def __init__(self):
        super().__init__(
            name="MissingDataAnalyzer",
            description="Analyzes missing data patterns"
        )
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Analyze missing data
        
        Args:
            data: Input DataFrame
        
        Returns:
            AgentResult with missing data analysis
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Overall missing data summary
            summary = self._get_missing_summary(data)
            
            # Column-wise analysis
            column_analysis = self._analyze_missing_by_column(data)
            
            # Missing data patterns
            patterns = self._identify_patterns(data)
            
            # Recommendations
            recommendations = self._get_recommendations(column_analysis)
            
            result.data = {
                "summary": summary,
                "columns": column_analysis,
                "patterns": patterns,
                "recommendations": recommendations,
            }
            
            if summary["total_missing"] == 0:
                self.logger.success("No missing data found!")
            else:
                self.logger.info(
                    f"Missing data analysis complete: "
                    f"{summary['total_missing']} missing values "
                    f"({summary['missing_percentage']:.2f}%)"
                )
        
        except Exception as e:
            result.add_error(f"Missing data analysis failed: {e}")
            self.logger.error(f"Missing data analysis error: {e}", exc_info=True)
        
        return result
    
    def _get_missing_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall missing data summary"""
        
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        
        return {
            "total_cells": int(total_cells),
            "total_missing": int(total_missing),
            "missing_percentage": float(total_missing / total_cells * 100),
            "n_columns_with_missing": int((df.isnull().sum() > 0).sum()),
            "n_rows_with_missing": int(df.isnull().any(axis=1).sum()),
            "complete_rows": int((~df.isnull().any(axis=1)).sum()),
        }
    
    def _analyze_missing_by_column(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze missing data for each column"""
        
        column_analysis = []
        
        for col in df.columns:
            n_missing = df[col].isnull().sum()
            
            if n_missing > 0:
                missing_pct = n_missing / len(df) * 100
                
                column_analysis.append({
                    "column": col,
                    "n_missing": int(n_missing),
                    "missing_percentage": float(missing_pct),
                    "dtype": str(df[col].dtype),
                    "severity": self._get_severity(missing_pct),
                    "suggested_strategy": self._suggest_strategy(df[col], missing_pct),
                })
        
        # Sort by missing percentage
        column_analysis.sort(key=lambda x: x["missing_percentage"], reverse=True)
        
        return column_analysis
    
    def _get_severity(self, missing_pct: float) -> str:
        """Determine severity of missing data"""
        if missing_pct < 5:
            return "low"
        elif missing_pct < 20:
            return "medium"
        elif missing_pct < 50:
            return "high"
        else:
            return "critical"
    
    def _suggest_strategy(self, series: pd.Series, missing_pct: float) -> str:
        """Suggest strategy for handling missing data"""
        
        if missing_pct > 70:
            return "consider_dropping_column"
        
        if pd.api.types.is_numeric_dtype(series):
            if missing_pct < 5:
                return "mean_or_median_imputation"
            else:
                return "forward_fill_or_interpolation"
        
        elif series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
            if missing_pct < 10:
                return "mode_imputation"
            else:
                return "create_missing_indicator"
        
        else:
            return "forward_fill"
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify missing data patterns"""
        
        patterns = {
            "random": [],
            "systematic": [],
            "correlated": [],
        }
        
        # Check for columns with systematic missingness
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                # Check if missingness is correlated with other columns
                correlations = []
                for other_col in df.columns:
                    if other_col != col and df[other_col].isnull().sum() > 0:
                        # Calculate correlation of missingness patterns
                        corr = df[col].isnull().corr(df[other_col].isnull())
                        if abs(corr) > 0.7:
                            correlations.append(other_col)
                
                if correlations:
                    patterns["correlated"].append({
                        "column": col,
                        "correlated_with": correlations,
                    })
        
        return patterns
    
    def _get_recommendations(
        self,
        column_analysis: List[Dict[str, Any]]
    ) -> List[str]:
        """Get recommendations for handling missing data"""
        
        recommendations = []
        
        if not column_analysis:
            recommendations.append("✅ Brak brakujących danych - dane są kompletne!")
            return recommendations
        
        # Critical columns
        critical_cols = [c for c in column_analysis if c["severity"] == "critical"]
        if critical_cols:
            recommendations.append(
                f"🚨 {len(critical_cols)} kolumn ma >50% brakujących danych - "
                "rozważ usunięcie tych kolumn"
            )
        
        # High severity
        high_severity = [c for c in column_analysis if c["severity"] == "high"]
        if high_severity:
            recommendations.append(
                f"⚠️ {len(high_severity)} kolumn ma 20-50% brakujących danych - "
                "użyj zaawansowanych metod imputacji"
            )
        
        # Numeric columns
        numeric_missing = [
            c for c in column_analysis
            if "int" in c["dtype"] or "float" in c["dtype"]
        ]
        if numeric_missing:
            recommendations.append(
                f"📊 {len(numeric_missing)} kolumn numerycznych z brakami - "
                "rozważ imputację średnią/medianą lub interpolację"
            )
        
        # Categorical columns
        categorical_missing = [
            c for c in column_analysis
            if c["dtype"] == "object"
        ]
        if categorical_missing:
            recommendations.append(
                f"📝 {len(categorical_missing)} kolumn kategorycznych z brakami - "
                "rozważ imputację modą lub stwórz kategorię 'brak'"
            )
        
        return recommendations