"""
DataGenius PRO - Statistical Analyzer
Performs comprehensive statistical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from loguru import logger
from core.base_agent import BaseAgent, AgentResult


class StatisticalAnalyzer(BaseAgent):
    """
    Comprehensive statistical analysis agent
    """
    
    def __init__(self):
        super().__init__(
            name="StatisticalAnalyzer",
            description="Comprehensive statistical analysis of dataset"
        )
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Perform statistical analysis
        
        Args:
            data: Input DataFrame
        
        Returns:
            AgentResult with statistical analysis
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Overall statistics
            overall_stats = self._get_overall_statistics(data)
            
            # Numeric features statistics
            numeric_stats = self._analyze_numeric_features(data)
            
            # Categorical features statistics
            categorical_stats = self._analyze_categorical_features(data)
            
            # Distribution analysis
            distributions = self._analyze_distributions(data)
            
            result.data = {
                "overall": overall_stats,
                "numeric_features": numeric_stats,
                "categorical_features": categorical_stats,
                "distributions": distributions,
            }
            
            self.logger.success("Statistical analysis completed")
        
        except Exception as e:
            result.add_error(f"Statistical analysis failed: {e}")
            self.logger.error(f"Statistical analysis error: {e}", exc_info=True)
        
        return result
    
    def _get_overall_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall dataset statistics"""
        
        return {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
            "n_categorical": len(df.select_dtypes(include=['object', 'category']).columns),
            "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            "sparsity": float(df.isnull().sum().sum() / (len(df) * len(df.columns))),
        }
    
    def _analyze_numeric_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numeric features"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "No numeric features found"}
        
        features_stats = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            features_stats[col] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "variance": float(series.var()),
                "range": float(series.max() - series.min()),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                "cv": float(series.std() / series.mean()) if series.mean() != 0 else None,
            }
        
        return {
            "n_features": len(numeric_cols),
            "features": features_stats,
            "summary": self._get_numeric_summary(features_stats),
        }
    
    def _get_numeric_summary(self, features_stats: Dict) -> Dict[str, Any]:
        """Get summary of numeric features"""
        
        if not features_stats:
            return {}
        
        # Find features with high/low variance
        variances = {k: v["variance"] for k, v in features_stats.items()}
        
        return {
            "highest_variance": max(variances, key=variances.get) if variances else None,
            "lowest_variance": min(variances, key=variances.get) if variances else None,
            "avg_skewness": float(np.mean([v["skewness"] for v in features_stats.values()])),
        }
    
    def _analyze_categorical_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical features"""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {"message": "No categorical features found"}
        
        features_stats = {}
        
        for col in categorical_cols:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            value_counts = series.value_counts()
            
            features_stats[col] = {
                "count": int(series.count()),
                "n_unique": int(series.nunique()),
                "mode": str(series.mode()[0]) if not series.mode().empty else None,
                "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "mode_percentage": float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0,
                "top_5_values": value_counts.head(5).to_dict(),
                "is_binary": len(value_counts) == 2,
                "cardinality": "high" if series.nunique() > 50 else "low",
            }
        
        return {
            "n_features": len(categorical_cols),
            "features": features_stats,
        }
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric features"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "No numeric features for distribution analysis"}
        
        distributions = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 10:  # Need enough data for distribution tests
                continue
            
            # Normality test (Shapiro-Wilk for sample, Anderson-Darling for larger)
            if len(series) <= 5000:
                try:
                    stat, p_value = stats.shapiro(series)
                    is_normal = p_value > 0.05
                except:
                    is_normal = None
            else:
                is_normal = None
            
            # Distribution shape
            skew = series.skew()
            kurt = series.kurtosis()
            
            if is_normal:
                dist_type = "normal"
            elif abs(skew) < 0.5:
                dist_type = "symmetric"
            elif skew > 0.5:
                dist_type = "right_skewed"
            else:
                dist_type = "left_skewed"
            
            distributions[col] = {
                "distribution_type": dist_type,
                "is_normal": is_normal,
                "skewness": float(skew),
                "kurtosis": float(kurt),
                "has_outliers": self._check_outliers_iqr(series),
            }
        
        return distributions
    
    def _check_outliers_iqr(self, series: pd.Series) -> bool:
        """Check if series has outliers using IQR method"""
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
        
        return bool(outliers > 0)