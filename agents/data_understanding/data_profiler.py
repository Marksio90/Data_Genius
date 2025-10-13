"""
DataGenius PRO - Data Profiler
Comprehensive data profiling and quality assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from core.data_validator import DataValidator


class DataProfiler(BaseAgent):
    """
    Comprehensive data profiling agent
    Provides detailed statistics and quality assessment
    """
    
    def __init__(self):
        super().__init__(
            name="DataProfiler",
            description="Comprehensive data profiling and quality assessment"
        )
        self.validator = DataValidator()
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters"""
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        
        return True
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Profile data comprehensively
        
        Args:
            data: Input DataFrame
        
        Returns:
            AgentResult with profiling information
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Data quality score
            quality_score, quality_details = self.validator.get_data_quality_score(data)
            
            # Statistical profile
            statistical_profile = self._get_statistical_profile(data)
            
            # Data quality issues
            quality_issues = self._identify_quality_issues(data)
            
            # Feature characteristics
            feature_characteristics = self._get_feature_characteristics(data)
            
            # Correlations (for numeric features)
            correlations = self._get_correlations(data)
            
            result.data = {
                "quality_score": quality_score,
                "quality_details": quality_details,
                "statistical_profile": statistical_profile,
                "quality_issues": quality_issues,
                "feature_characteristics": feature_characteristics,
                "correlations": correlations,
            }
            
            self.logger.success(
                f"Data profiling complete. Quality score: {quality_score:.1f}/100"
            )
        
        except Exception as e:
            result.add_error(f"Data profiling failed: {e}")
            self.logger.error(f"Data profiling error: {e}", exc_info=True)
        
        return result
    
    def _get_statistical_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical profile of dataset"""
        
        profile = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
            "n_categorical": len(df.select_dtypes(include=['object', 'category']).columns),
            "n_datetime": len(df.select_dtypes(include=['datetime64']).columns),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "duplicates": {
                "n_duplicates": df.duplicated().sum(),
                "pct_duplicates": df.duplicated().sum() / len(df) * 100,
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "pct_missing": df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
                "columns_with_missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            },
        }
        
        return profile
    
    def _identify_quality_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify data quality issues"""
        
        issues = []
        
        # Missing data issues
        missing = df.isnull().sum()
        for col in missing[missing > len(df) * 0.5].index:
            issues.append({
                "type": "high_missing_data",
                "severity": "high",
                "column": col,
                "description": f"Kolumna '{col}' ma >50% brakujących wartości",
                "missing_pct": float(missing[col] / len(df) * 100),
            })
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append({
                    "type": "constant_column",
                    "severity": "medium",
                    "column": col,
                    "description": f"Kolumna '{col}' ma tylko jedną unikalną wartość",
                })
        
        # High cardinality categorical
        for col in df.select_dtypes(include=['object']).columns:
            n_unique = df[col].nunique()
            if n_unique > len(df) * 0.9:
                issues.append({
                    "type": "high_cardinality",
                    "severity": "low",
                    "column": col,
                    "description": f"Kolumna '{col}' ma bardzo wysoką kardynalność ({n_unique} unikalnych)",
                })
        
        # Duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            issues.append({
                "type": "duplicates",
                "severity": "medium" if n_duplicates > len(df) * 0.1 else "low",
                "description": f"Znaleziono {n_duplicates} zduplikowanych wierszy",
                "n_duplicates": int(n_duplicates),
            })
        
        # Outliers in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
            
            if outliers > len(df) * 0.05:
                issues.append({
                    "type": "outliers",
                    "severity": "low",
                    "column": col,
                    "description": f"Kolumna '{col}' ma {outliers} outliers (IQR method)",
                    "n_outliers": int(outliers),
                })
        
        return issues
    
    def _get_feature_characteristics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize features by characteristics"""
        
        characteristics = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "high_cardinality": [],
            "binary": [],
            "constant": [],
            "high_missing": [],
        }
        
        for col in df.columns:
            col_data = df[col]
            
            # Type
            if pd.api.types.is_numeric_dtype(col_data):
                characteristics["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                characteristics["datetime"].append(col)
            else:
                characteristics["categorical"].append(col)
            
            # Cardinality
            n_unique = col_data.nunique()
            if n_unique == 1:
                characteristics["constant"].append(col)
            elif n_unique == 2:
                characteristics["binary"].append(col)
            elif n_unique > len(df) * 0.9:
                characteristics["high_cardinality"].append(col)
            
            # Missing data
            if col_data.isnull().sum() > len(df) * 0.3:
                characteristics["high_missing"].append(col)
        
        return characteristics
    
    def _get_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get correlation analysis for numeric features"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                "n_numeric_features": len(numeric_cols),
                "correlations": None,
                "high_correlations": [],
            }
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Find high correlations (>0.8)
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val),
                    })
        
        return {
            "n_numeric_features": len(numeric_cols),
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": high_corr,
            "n_high_correlations": len(high_corr),
        }