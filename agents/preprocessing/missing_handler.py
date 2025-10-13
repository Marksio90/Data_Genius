"""
DataGenius PRO - Missing Data Handler
Intelligent missing data imputation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.impute import SimpleImputer, KNNImputer
from loguru import logger
from core.base_agent import BaseAgent, AgentResult


class MissingDataHandler(BaseAgent):
    """
    Handles missing data using various imputation strategies
    """
    
    def __init__(self):
        super().__init__(
            name="MissingDataHandler",
            description="Intelligent missing data imputation"
        )
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        strategy: str = "auto",
        **kwargs
    ) -> AgentResult:
        """
        Handle missing data
        
        Args:
            data: Input DataFrame
            target_column: Target column name
            strategy: Imputation strategy (auto, mean, median, mode, knn, drop)
        
        Returns:
            AgentResult with imputed data
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            df = data.copy()
            imputation_log = []
            
            # Check if there's missing data
            if not df.isnull().any().any():
                result.data = {
                    "data": df,
                    "imputation_log": ["No missing data found"],
                    "n_imputed": 0,
                }
                self.logger.info("No missing data to handle")
                return result
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle missing in features
            if X.isnull().any().any():
                X_imputed, feat_log = self._impute_features(X, strategy)
                imputation_log.extend(feat_log)
            else:
                X_imputed = X
            
            # Handle missing in target (critical - usually drop)
            if y.isnull().any():
                n_missing_target = y.isnull().sum()
                self.logger.warning(
                    f"Target has {n_missing_target} missing values - dropping rows"
                )
                
                # Keep only rows where target is not null
                valid_indices = ~y.isnull()
                X_imputed = X_imputed[valid_indices]
                y = y[valid_indices]
                
                imputation_log.append(
                    f"Dropped {n_missing_target} rows with missing target values"
                )
            
            # Combine back
            df_imputed = X_imputed.copy()
            df_imputed[target_column] = y.values
            
            result.data = {
                "data": df_imputed,
                "imputation_log": imputation_log,
                "n_imputed": len(imputation_log),
                "original_shape": data.shape,
                "final_shape": df_imputed.shape,
            }
            
            self.logger.success(
                f"Missing data handled: {len(imputation_log)} operations"
            )
        
        except Exception as e:
            result.add_error(f"Missing data handling failed: {e}")
            self.logger.error(f"Missing data handling error: {e}", exc_info=True)
        
        return result
    
    def _impute_features(
        self,
        X: pd.DataFrame,
        strategy: str
    ) -> tuple:
        """Impute missing values in features"""
        
        X_imputed = X.copy()
        imputation_log = []
        
        # Numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        numeric_missing = [col for col in numeric_cols if X[col].isnull().any()]
        
        if numeric_missing:
            if strategy == "auto" or strategy == "median":
                imputer = SimpleImputer(strategy="median")
                X_imputed[numeric_missing] = imputer.fit_transform(X[numeric_missing])
                imputation_log.append(
                    f"Imputed {len(numeric_missing)} numeric columns with median"
                )
            
            elif strategy == "mean":
                imputer = SimpleImputer(strategy="mean")
                X_imputed[numeric_missing] = imputer.fit_transform(X[numeric_missing])
                imputation_log.append(
                    f"Imputed {len(numeric_missing)} numeric columns with mean"
                )
            
            elif strategy == "knn":
                imputer = KNNImputer(n_neighbors=5)
                X_imputed[numeric_missing] = imputer.fit_transform(X[numeric_missing])
                imputation_log.append(
                    f"Imputed {len(numeric_missing)} numeric columns with KNN"
                )
        
        # Categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        categorical_missing = [col for col in categorical_cols if X[col].isnull().any()]
        
        if categorical_missing:
            if strategy == "drop":
                # Drop rows with missing categorical
                X_imputed = X_imputed.dropna(subset=categorical_missing)
                imputation_log.append(
                    f"Dropped rows with missing categorical values in {len(categorical_missing)} columns"
                )
            else:
                # Mode imputation
                imputer = SimpleImputer(strategy="most_frequent")
                X_imputed[categorical_missing] = imputer.fit_transform(X[categorical_missing])
                imputation_log.append(
                    f"Imputed {len(categorical_missing)} categorical columns with mode"
                )
        
        return X_imputed, imputation_log