"""
DataGenius PRO - Feature Engineer
Automated feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from config.constants import DATE_FEATURES


class FeatureEngineer(BaseAgent):
    """
    Automated feature engineering agent
    """
    
    def __init__(self):
        super().__init__(
            name="FeatureEngineer",
            description="Automated feature engineering"
        )
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> AgentResult:
        """
        Perform feature engineering
        
        Args:
            data: Input DataFrame
            target_column: Target column name
        
        Returns:
            AgentResult with engineered features
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            df = data.copy()
            features_created = []
            
            # Date features
            date_features = self._create_date_features(df)
            if date_features:
                features_created.extend(date_features)
            
            # Interaction features
            interaction_features = self._create_interaction_features(
                df,
                target_column
            )
            if interaction_features:
                features_created.extend(interaction_features)
            
            # Polynomial features (for selected numeric)
            poly_features = self._create_polynomial_features(df, target_column)
            if poly_features:
                features_created.extend(poly_features)
            
            # Binning features
            binned_features = self._create_binned_features(df)
            if binned_features:
                features_created.extend(binned_features)
            
            result.data = {
                "engineered_data": df,
                "features_created": features_created,
                "n_new_features": len(features_created),
                "original_shape": data.shape,
                "new_shape": df.shape,
            }
            
            self.logger.success(
                f"Feature engineering complete: {len(features_created)} new features"
            )
        
        except Exception as e:
            result.add_error(f"Feature engineering failed: {e}")
            self.logger.error(f"Feature engineering error: {e}", exc_info=True)
        
        return result
    
    def _create_date_features(self, df: pd.DataFrame) -> List[str]:
        """Create features from datetime columns"""
        
        features_created = []
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in date_cols:
            # Extract date features
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)
            
            features_created.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek",
                f"{col}_quarter",
                f"{col}_is_weekend",
            ])
            
            # Drop original datetime column
            df.drop(columns=[col], inplace=True)
        
        return features_created
    
    def _create_interaction_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        max_interactions: int = 5
    ) -> List[str]:
        """Create interaction features for top correlated features"""
        
        features_created = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != target_column]
        
        if len(numeric_cols) < 2:
            return features_created
        
        # Calculate correlations with target
        if target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
            correlations = df[numeric_cols].corrwith(df[target_column]).abs()
            top_features = correlations.nlargest(min(5, len(correlations))).index.tolist()
        else:
            # Just use first few features
            top_features = numeric_cols[:5].tolist()
        
        # Create interactions
        created_count = 0
        for i, feat1 in enumerate(top_features):
            if created_count >= max_interactions:
                break
            
            for feat2 in top_features[i+1:]:
                if created_count >= max_interactions:
                    break
                
                # Multiplication
                new_feat_name = f"{feat1}_x_{feat2}"
                df[new_feat_name] = df[feat1] * df[feat2]
                features_created.append(new_feat_name)
                created_count += 1
        
        return features_created
    
    def _create_polynomial_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        degree: int = 2,
        max_features: int = 3
    ) -> List[str]:
        """Create polynomial features for highly correlated numeric features"""
        
        features_created = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != target_column]
        
        if len(numeric_cols) == 0:
            return features_created
        
        # Select top correlated features
        if target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
            correlations = df[numeric_cols].corrwith(df[target_column]).abs()
            top_features = correlations.nlargest(min(max_features, len(correlations))).index.tolist()
        else:
            top_features = numeric_cols[:max_features]
        
        # Create polynomial features
        for feat in top_features:
            # Square
            new_feat_name = f"{feat}_squared"
            df[new_feat_name] = df[feat] ** 2
            features_created.append(new_feat_name)
            
            # Square root (if all values >= 0)
            if (df[feat] >= 0).all():
                new_feat_name = f"{feat}_sqrt"
                df[new_feat_name] = np.sqrt(df[feat])
                features_created.append(new_feat_name)
        
        return features_created
    
    def _create_binned_features(
        self,
        df: pd.DataFrame,
        max_features: int = 3
    ) -> List[str]:
        """Create binned versions of numeric features"""
        
        features_created = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return features_created
        
        # Select features with high variance
        variances = df[numeric_cols].var()
        top_features = variances.nlargest(min(max_features, len(variances))).index.tolist()
        
        for feat in top_features:
            try:
                # Create quintiles
                new_feat_name = f"{feat}_binned"
                df[new_feat_name] = pd.qcut(
                    df[feat],
                    q=5,
                    labels=["very_low", "low", "medium", "high", "very_high"],
                    duplicates="drop"
                )
                features_created.append(new_feat_name)
            except:
                # If qcut fails, use cut
                try:
                    df[new_feat_name] = pd.cut(
                        df[feat],
                        bins=5,
                        labels=["very_low", "low", "medium", "high", "very_high"]
                    )
                    features_created.append(new_feat_name)
                except:
                    pass
        
        return features_created