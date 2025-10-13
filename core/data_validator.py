"""
DataGenius PRO - Data Validator
Data quality checks and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from pydantic import BaseModel, Field
from config.constants import (
    MIN_ROWS_FOR_ML,
    MAX_CATEGORICAL_UNIQUE_VALUES,
    MISSING_DATA_THRESHOLD,
)


class ValidationResult(BaseModel):
    """Validation result model"""
    
    is_valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add error message"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add warning message"""
        self.warnings.append(message)
    
    def add_info(self, key: str, value: Any) -> None:
        """Add info"""
        self.info[key] = value


class DataValidator:
    """
    Data quality validator
    Performs various checks on data quality and suitability for ML
    """
    
    def __init__(self):
        self.logger = logger.bind(component="DataValidator")
    
    def validate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        check_ml_readiness: bool = True
    ) -> ValidationResult:
        """
        Perform comprehensive data validation
        
        Args:
            df: DataFrame to validate
            target_column: Target column name (optional)
            check_ml_readiness: Check if data is ready for ML
        
        Returns:
            ValidationResult object
        """
        
        result = ValidationResult()
        
        self.logger.info("Starting data validation")
        
        # Basic checks
        self._check_not_empty(df, result)
        self._check_columns(df, result)
        self._check_data_types(df, result)
        
        # Data quality checks
        self._check_missing_data(df, result)
        self._check_duplicates(df, result)
        self._check_constants(df, result)
        
        # ML readiness checks
        if check_ml_readiness:
            self._check_ml_readiness(df, result, target_column)
        
        # Target column checks
        if target_column:
            self._check_target_column(df, target_column, result)
        
        # Add summary info
        result.add_info("n_rows", len(df))
        result.add_info("n_columns", len(df.columns))
        result.add_info("memory_mb", df.memory_usage(deep=True).sum() / 1024**2)
        
        if result.is_valid:
            self.logger.success("Data validation passed")
        else:
            self.logger.warning(
                f"Data validation failed with {len(result.errors)} errors"
            )
        
        return result
    
    def _check_not_empty(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check if DataFrame is not empty"""
        
        if df.empty:
            result.add_error("DataFrame jest pusty (0 wierszy)")
        
        if len(df.columns) == 0:
            result.add_error("DataFrame nie ma kolumn")
    
    def _check_columns(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check column names"""
        
        # Check for duplicate column names
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            result.add_error(f"Zduplikowane nazwy kolumn: {duplicates}")
        
        # Check for unnamed columns
        unnamed = [col for col in df.columns if "Unnamed" in str(col)]
        if unnamed:
            result.add_warning(f"Kolumny bez nazwy: {unnamed}")
        
        # Check for special characters in column names
        special_chars = [col for col in df.columns if not str(col).replace("_", "").isalnum()]
        if special_chars:
            result.add_warning(
                f"Kolumny ze znakami specjalnymi (może powodować problemy): {special_chars[:5]}"
            )
    
    def _check_data_types(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check data types"""
        
        type_counts = df.dtypes.value_counts().to_dict()
        result.add_info("data_types", {str(k): v for k, v in type_counts.items()})
        
        # Check for object columns that might be numeric
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_numeric(df[col])
                result.add_warning(
                    f"Kolumna '{col}' jest typu object, ale może być liczbowa"
                )
            except (ValueError, TypeError):
                pass
    
    def _check_missing_data(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for missing data"""
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        # Get columns with missing data
        missing_cols = missing[missing > 0].to_dict()
        
        if missing_cols:
            result.add_info("missing_data", {
                col: {"count": int(count), "percentage": float(missing_pct[col])}
                for col, count in missing_cols.items()
            })
            
            # Warning for high missing data
            high_missing = missing_pct[missing_pct > MISSING_DATA_THRESHOLD * 100]
            if not high_missing.empty:
                result.add_warning(
                    f"Kolumny z >50% brakujących danych: {high_missing.index.tolist()}"
                )
    
    def _check_duplicates(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for duplicate rows"""
        
        n_duplicates = df.duplicated().sum()
        
        if n_duplicates > 0:
            pct = (n_duplicates / len(df) * 100).round(2)
            result.add_warning(
                f"Znaleziono {n_duplicates} zduplikowanych wierszy ({pct}%)"
            )
            result.add_info("n_duplicates", int(n_duplicates))
    
    def _check_constants(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for constant columns (no variance)"""
        
        constant_cols = []
        
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            result.add_warning(
                f"Kolumny ze stałą wartością (bez wariancji): {constant_cols}"
            )
            result.add_info("constant_columns", constant_cols)
    
    def _check_ml_readiness(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        target_column: Optional[str]
    ) -> None:
        """Check if data is ready for ML training"""
        
        # Minimum rows check
        if len(df) < MIN_ROWS_FOR_ML:
            result.add_error(
                f"Za mało danych do trenowania modelu. "
                f"Minimum: {MIN_ROWS_FOR_ML}, aktualnie: {len(df)}"
            )
        
        # Check if there are any numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            result.add_warning("Brak kolumn numerycznych w danych")
        
        # Check for high cardinality in categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        high_cardinality = []
        for col in categorical_cols:
            if col != target_column:
                n_unique = df[col].nunique()
                if n_unique > MAX_CATEGORICAL_UNIQUE_VALUES:
                    high_cardinality.append((col, n_unique))
        
        if high_cardinality:
            result.add_warning(
                f"Kolumny kategoryczne z dużą liczbą unikalnych wartości: "
                f"{[(col, n) for col, n in high_cardinality[:5]]}"
            )
    
    def _check_target_column(
        self,
        df: pd.DataFrame,
        target_column: str,
        result: ValidationResult
    ) -> None:
        """Check target column"""
        
        # Check if column exists
        if target_column not in df.columns:
            result.add_error(f"Kolumna docelowa '{target_column}' nie istnieje w danych")
            return
        
        target = df[target_column]
        
        # Check for missing values in target
        if target.isnull().any():
            n_missing = target.isnull().sum()
            result.add_error(
                f"Kolumna docelowa ma {n_missing} brakujących wartości"
            )
        
        # Check target distribution
        n_unique = target.nunique()
        result.add_info("target_unique_values", int(n_unique))
        
        # Classification vs Regression detection
        if n_unique <= 20:  # Likely classification
            value_counts = target.value_counts()
            result.add_info("target_distribution", value_counts.to_dict())
            
            # Check for imbalanced classes
            if len(value_counts) > 1:
                min_class = value_counts.min()
                max_class = value_counts.max()
                imbalance_ratio = max_class / min_class
                
                if imbalance_ratio > 10:
                    result.add_warning(
                        f"Niezbalansowane klasy w target (ratio: {imbalance_ratio:.1f}:1)"
                    )
        
        else:  # Likely regression
            result.add_info("target_stats", {
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
            })
    
    def get_data_quality_score(self, df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate overall data quality score (0-100)
        
        Args:
            df: DataFrame to score
        
        Returns:
            Tuple of (score, details_dict)
        """
        
        score = 100.0
        details = {}
        
        # Completeness (30 points)
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        completeness_score = max(0, 30 - missing_pct * 0.3)
        score -= (30 - completeness_score)
        details["completeness"] = completeness_score
        
        # Uniqueness (20 points)
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        uniqueness_score = max(0, 20 - dup_pct * 0.2)
        score -= (20 - uniqueness_score)
        details["uniqueness"] = uniqueness_score
        
        # Consistency (20 points)
        # Check for constant columns
        n_constant = sum(df[col].nunique() == 1 for col in df.columns)
        constant_pct = (n_constant / len(df.columns)) * 100
        consistency_score = max(0, 20 - constant_pct * 0.5)
        score -= (20 - consistency_score)
        details["consistency"] = consistency_score
        
        # Validity (30 points)
        # Simple check: no invalid types, no extreme outliers, etc.
        validity_score = 30.0  # Simplified
        details["validity"] = validity_score
        
        score = max(0, min(100, score))
        
        return score, details