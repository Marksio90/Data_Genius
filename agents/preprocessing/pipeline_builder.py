"""
DataGenius PRO - Pipeline Builder
Builds complete preprocessing pipeline
"""

import pandas as pd
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from loguru import logger
from core.base_agent import BaseAgent, AgentResult


class PipelineBuilder(BaseAgent):
    """
    Builds complete preprocessing pipeline for ML
    """
    
    def __init__(self):
        super().__init__(
            name="PipelineBuilder",
            description="Builds preprocessing pipeline"
        )
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: str,
        **kwargs
    ) -> AgentResult:
        """
        Build preprocessing pipeline
        
        Args:
            data: Input DataFrame
            target_column: Target column name
            problem_type: classification or regression
        
        Returns:
            AgentResult with pipeline and processed data
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Build feature preprocessing pipeline
            feature_pipeline = self._build_feature_pipeline(X)
            
            # Prepare target
            y_processed, target_encoder = self._prepare_target(y, problem_type)
            
            # Apply transformations
            X_transformed = feature_pipeline.fit_transform(X)
            
            # Get feature names after transformation
            feature_names = self._get_feature_names(feature_pipeline, X)
            
            # Create DataFrame
            X_df = pd.DataFrame(X_transformed, columns=feature_names)
            
            result.data = {
                "X": X_df,
                "y": y_processed,
                "feature_pipeline": feature_pipeline,
                "target_encoder": target_encoder,
                "feature_names": feature_names,
                "original_shape": data.shape,
                "transformed_shape": (len(X_df), len(feature_names) + 1),
            }
            
            self.logger.success(
                f"Pipeline built: {len(feature_names)} features"
            )
        
        except Exception as e:
            result.add_error(f"Pipeline building failed: {e}")
            self.logger.error(f"Pipeline building error: {e}", exc_info=True)
        
        return result
    
    def _build_feature_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build feature preprocessing pipeline"""
        
        import numpy as np
        
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def _prepare_target(
        self,
        y: pd.Series,
        problem_type: str
    ) -> tuple:
        """Prepare target variable"""
        
        if problem_type == "regression":
            # Regression - keep as is
            return y.values, None
        
        else:
            # Classification - encode if needed
            if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(y)
                return y_encoded, encoder
            else:
                return y.values, None
    
    def _get_feature_names(
        self,
        pipeline: ColumnTransformer,
        X: pd.DataFrame
    ) -> list:
        """Get feature names after transformation"""
        
        import numpy as np
        
        feature_names = []
        
        # Numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        feature_names.extend(numeric_features)
        
        # Categorical features (one-hot encoded)
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for transformer_name, transformer, columns in pipeline.transformers_:
            if transformer_name == 'cat' and hasattr(transformer, 'named_steps'):
                if 'onehot' in transformer.named_steps:
                    onehot = transformer.named_steps['onehot']
                    if hasattr(onehot, 'get_feature_names_out'):
                        cat_features = onehot.get_feature_names_out(columns).tolist()
                        feature_names.extend(cat_features)
                    else:
                        # Fallback
                        for col in columns:
                            n_categories = X[col].nunique()
                            feature_names.extend([f"{col}_{i}" for i in range(n_categories)])
        
        return feature_names