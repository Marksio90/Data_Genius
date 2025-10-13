"""
DataGenius PRO - Schema Analyzer
Analyzes data schema and structure
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from core.utils import detect_column_type, get_numeric_columns, get_categorical_columns


class SchemaAnalyzer(BaseAgent):
    """
    Analyzes data schema and provides detailed column information
    """
    
    def __init__(self):
        super().__init__(
            name="SchemaAnalyzer",
            description="Analyzes data structure and column types"
        )
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters"""
        if "data" not in kwargs:
            raise ValueError("'data' parameter is required")
        
        df = kwargs["data"]
        if not isinstance(df, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        return True
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AgentResult:
        """
        Analyze data schema
        
        Args:
            data: Input DataFrame
        
        Returns:
            AgentResult with schema information
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Basic info
            basic_info = self._get_basic_info(data)
            
            # Column analysis
            column_info = self._analyze_columns(data)
            
            # Data types summary
            dtypes_summary = self._get_dtypes_summary(data)
            
            # Memory usage
            memory_info = self._get_memory_info(data)
            
            # Store results
            result.data = {
                "basic_info": basic_info,
                "columns": column_info,
                "dtypes_summary": dtypes_summary,
                "memory_info": memory_info,
            }
            
            self.logger.success(f"Schema analysis complete: {len(data.columns)} columns analyzed")
            
        except Exception as e:
            result.add_error(f"Schema analysis failed: {e}")
            self.logger.error(f"Schema analysis error: {e}", exc_info=True)
        
        return result
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic DataFrame information"""
        return {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "shape": df.shape,
            "size": df.size,
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze each column in detail"""
        
        columns_info = []
        
        for col in df.columns:
            col_data = df[col]
            
            # Basic stats
            info = {
                "name": col,
                "dtype": str(col_data.dtype),
                "semantic_type": detect_column_type(col_data),
                "n_unique": int(col_data.nunique()),
                "n_missing": int(col_data.isnull().sum()),
                "missing_pct": float(col_data.isnull().sum() / len(df) * 100),
                "n_zeros": int((col_data == 0).sum()) if pd.api.types.is_numeric_dtype(col_data) else 0,
            }
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                info.update(self._get_numeric_stats(col_data))
            
            # Categorical columns
            elif col_data.dtype == "object" or pd.api.types.is_categorical_dtype(col_data):
                info.update(self._get_categorical_stats(col_data))
            
            # Datetime columns
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                info.update(self._get_datetime_stats(col_data))
            
            columns_info.append(info)
        
        return columns_info
    
    def _get_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Get statistics for numeric column"""
        
        return {
            "mean": float(series.mean()) if not series.isnull().all() else None,
            "std": float(series.std()) if not series.isnull().all() else None,
            "min": float(series.min()) if not series.isnull().all() else None,
            "max": float(series.max()) if not series.isnull().all() else None,
            "median": float(series.median()) if not series.isnull().all() else None,
            "q25": float(series.quantile(0.25)) if not series.isnull().all() else None,
            "q75": float(series.quantile(0.75)) if not series.isnull().all() else None,
            "skewness": float(series.skew()) if not series.isnull().all() else None,
            "kurtosis": float(series.kurtosis()) if not series.isnull().all() else None,
        }
    
    def _get_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Get statistics for categorical column"""
        
        value_counts = series.value_counts()
        
        return {
            "mode": str(series.mode()[0]) if not series.mode().empty else None,
            "top_values": value_counts.head(5).to_dict(),
            "n_categories": len(value_counts),
            "is_binary": len(value_counts) == 2,
        }
    
    def _get_datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Get statistics for datetime column"""
        
        return {
            "min_date": str(series.min()) if not series.isnull().all() else None,
            "max_date": str(series.max()) if not series.isnull().all() else None,
            "date_range_days": (series.max() - series.min()).days if not series.isnull().all() else None,
        }
    
    def _get_dtypes_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get summary of data types"""
        
        dtypes_count = df.dtypes.value_counts()
        return {str(dtype): int(count) for dtype, count in dtypes_count.items()}
    
    def _get_memory_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get memory usage information"""
        
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        return {
            "total_mb": float(total_memory / 1024**2),
            "per_row_bytes": float(total_memory / len(df)) if len(df) > 0 else 0,
            "by_column_mb": {
                col: float(mem / 1024**2)
                for col, mem in memory_usage.items()
            },
        }