"""
DataGenius PRO - Outlier Detector
Detects outliers using multiple methods
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from sklearn.ensemble import IsolationForest
from loguru import logger
from core.base_agent import BaseAgent, AgentResult


class OutlierDetector(BaseAgent):
    """
    Detects outliers using IQR, Z-score, and Isolation Forest methods
    """
    
    def __init__(self):
        super().__init__(
            name="OutlierDetector",
            description="Detects outliers in numeric features"
        )
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Detect outliers
        
        Args:
            data: Input DataFrame
        
        Returns:
            AgentResult with outlier analysis
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                result.add_warning("No numeric columns for outlier detection")
                return result
            
            # Detect outliers using multiple methods
            iqr_outliers = self._detect_iqr_outliers(data[numeric_cols])
            zscore_outliers = self._detect_zscore_outliers(data[numeric_cols])
            
            # Isolation Forest (if enough data)
            if len(data) >= 100:
                isolation_outliers = self._detect_isolation_forest_outliers(
                    data[numeric_cols]
                )
            else:
                isolation_outliers = None
            
            # Summary
            summary = self._create_summary(
                iqr_outliers,
                zscore_outliers,
                isolation_outliers
            )
            
            result.data = {
                "iqr_method": iqr_outliers,
                "zscore_method": zscore_outliers,
                "isolation_forest": isolation_outliers,
                "summary": summary,
            }
            
            self.logger.success(
                f"Outlier detection complete: {summary['total_outliers']} outliers found"
            )
        
        except Exception as e:
            result.add_error(f"Outlier detection failed: {e}")
            self.logger.error(f"Outlier detection error: {e}", exc_info=True)
        
        return result
    
    def _detect_iqr_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        
        outliers_by_column = {}
        
        for col in df.columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            n_outliers = outliers_mask.sum()
            
            if n_outliers > 0:
                outliers_by_column[col] = {
                    "n_outliers": int(n_outliers),
                    "percentage": float(n_outliers / len(df) * 100),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_indices": df[outliers_mask].index.tolist()[:10],  # First 10
                }
        
        return {
            "method": "IQR (Interquartile Range)",
            "description": "Outliers are values below Q1-1.5*IQR or above Q3+1.5*IQR",
            "columns": outliers_by_column,
            "n_columns_with_outliers": len(outliers_by_column),
        }
    
    def _detect_zscore_outliers(
        self,
        df: pd.DataFrame,
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """Detect outliers using Z-score method"""
        
        outliers_by_column = {}
        
        for col in df.columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            z_scores = np.abs(stats.zscore(series))
            outliers_mask = z_scores > threshold
            n_outliers = outliers_mask.sum()
            
            if n_outliers > 0:
                outliers_by_column[col] = {
                    "n_outliers": int(n_outliers),
                    "percentage": float(n_outliers / len(series) * 100),
                    "threshold": threshold,
                }
        
        return {
            "method": "Z-Score",
            "description": f"Outliers are values with |z-score| > {threshold}",
            "columns": outliers_by_column,
            "n_columns_with_outliers": len(outliers_by_column),
        }
    
    def _detect_isolation_forest_outliers(
        self,
        df: pd.DataFrame,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest"""
        
        try:
            # Fill NaN with column mean for Isolation Forest
            df_filled = df.fillna(df.mean())
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
            predictions = iso_forest.fit_predict(df_filled)
            outliers_mask = predictions == -1
            n_outliers = outliers_mask.sum()
            
            return {
                "method": "Isolation Forest",
                "description": "ML-based anomaly detection",
                "n_outliers": int(n_outliers),
                "percentage": float(n_outliers / len(df) * 100),
                "outlier_indices": df[outliers_mask].index.tolist()[:10],
            }
        
        except Exception as e:
            self.logger.warning(f"Isolation Forest failed: {e}")
            return None
    
    def _create_summary(
        self,
        iqr_outliers: Dict,
        zscore_outliers: Dict,
        isolation_outliers: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create summary of outlier detection"""
        
        # Count unique columns with outliers
        columns_with_outliers = set()
        columns_with_outliers.update(iqr_outliers["columns"].keys())
        columns_with_outliers.update(zscore_outliers["columns"].keys())
        
        # Total outliers (using IQR as reference)
        total_outliers = sum(
            col_info["n_outliers"]
            for col_info in iqr_outliers["columns"].values()
        )
        
        return {
            "total_outliers": int(total_outliers),
            "n_columns_with_outliers": len(columns_with_outliers),
            "methods_used": ["IQR", "Z-Score"] + (["Isolation Forest"] if isolation_outliers else []),
            "most_outliers": self._get_most_outliers_column(iqr_outliers),
        }
    
    def _get_most_outliers_column(self, iqr_outliers: Dict) -> Dict[str, Any]:
        """Get column with most outliers"""
        
        if not iqr_outliers["columns"]:
            return None
        
        max_col = max(
            iqr_outliers["columns"].items(),
            key=lambda x: x[1]["n_outliers"]
        )
        
        return {
            "column": max_col[0],
            "n_outliers": max_col[1]["n_outliers"],
        }