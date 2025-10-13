"""
DataGenius PRO - Model Explainer
Provides model interpretability using SHAP and feature importance
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from loguru import logger
from core.base_agent import BaseAgent, AgentResult


class ModelExplainer(BaseAgent):
    """
    Explains model predictions using various interpretability methods
    """
    
    def __init__(self):
        super().__init__(
            name="ModelExplainer",
            description="Provides model interpretability"
        )
    
    def execute(
        self,
        best_model: Any,
        pycaret_wrapper: Any,
        data: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> AgentResult:
        """
        Explain model
        
        Args:
            best_model: Trained model
            pycaret_wrapper: PyCaret wrapper
            data: Training data
            target_column: Target column
        
        Returns:
            AgentResult with explanations
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Get feature importance
            feature_importance = self._get_feature_importance(
                best_model,
                data,
                target_column
            )
            
            # Get SHAP values (if possible)
            shap_values = self._get_shap_explanations(
                best_model,
                data,
                target_column
            )
            
            # Generate insights
            insights = self._generate_insights(feature_importance, shap_values)
            
            result.data = {
                "feature_importance": feature_importance,
                "shap_values": shap_values,
                "top_features": feature_importance['feature'].head(5).tolist() if feature_importance is not None else [],
                "insights": insights,
            }
            
            self.logger.success("Model explanation generated")
        
        except Exception as e:
            result.add_error(f"Model explanation failed: {e}")
            self.logger.error(f"Model explanation error: {e}", exc_info=True)
        
        return result
    
    def _get_feature_importance(
        self,
        model: Any,
        data: pd.DataFrame,
        target_column: str
    ) -> Optional[pd.DataFrame]:
        """Get feature importance"""
        
        try:
            X = data.drop(columns=[target_column])
            
            # Try direct feature importance
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance
            
            # Try coefficients (for linear models)
            elif hasattr(model, 'coef_'):
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(model.coef_).flatten()
                }).sort_values('importance', ascending=False)
                
                return importance
            
            # Fallback: permutation importance
            else:
                return self._permutation_importance(model, X, data[target_column])
        
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    def _permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """Calculate permutation importance"""
        
        try:
            from sklearn.inspection import permutation_importance
            
            result = permutation_importance(
                model,
                X,
                y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False)
            
            return importance
        
        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
            return None
    
    def _get_shap_explanations(
        self,
        model: Any,
        data: pd.DataFrame,
        target_column: str
    ) -> Optional[Dict[str, Any]]:
        """Get SHAP explanations"""
        
        try:
            import shap
            
            X = data.drop(columns=[target_column])
            
            # Sample data if too large
            if len(X) > 1000:
                X_sample = X.sample(n=1000, random_state=42)
            else:
                X_sample = X
            
            # Create SHAP explainer
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            
            # Get mean absolute SHAP values
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            
            shap_importance = pd.DataFrame({
                'feature': X.columns,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)
            
            return {
                "shap_importance": shap_importance.to_dict(),
                "shap_values": "Available",  # Actual values too large to store
            }
        
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed: {e}")
            return None
    
    def _generate_insights(
        self,
        feature_importance: Optional[pd.DataFrame],
        shap_values: Optional[Dict]
    ) -> List[str]:
        """Generate insights from explanations"""
        
        insights = []
        
        if feature_importance is not None and not feature_importance.empty:
            top_feature = feature_importance.iloc[0]['feature']
            top_importance = feature_importance.iloc[0]['importance']
            
            insights.append(
                f"Najważniejsza cecha: {top_feature} (ważność: {top_importance:.4f})"
            )
            
            # Check if importance is concentrated
            top_3_pct = feature_importance.head(3)['importance'].sum() / feature_importance['importance'].sum() * 100
            
            if top_3_pct > 70:
                insights.append(
                    f"Top 3 cechy stanowią {top_3_pct:.1f}% całkowitej ważności - model mocno zależy od kilku cech"
                )
        
        if shap_values is not None:
            insights.append(
                "Wartości SHAP zostały policzone - dostępna szczegółowa interpretacja"
            )
        
        return insights