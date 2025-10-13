"""
DataGenius PRO - Correlation Analyzer
Analyzes correlations between features
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import chi2_contingency
from loguru import logger
from core.base_agent import BaseAgent, AgentResult


class CorrelationAnalyzer(BaseAgent):
    """
    Analyzes correlations between features (numeric and categorical)
    """
    
    def __init__(self):
        super().__init__(
            name="CorrelationAnalyzer",
            description="Analyzes feature correlations"
        )
    
    def execute(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        Analyze correlations
        
        Args:
            data: Input DataFrame
            target_column: Target column (optional)
        
        Returns:
            AgentResult with correlation analysis
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Numeric correlations
            numeric_corr = self._analyze_numeric_correlations(data)
            
            # Categorical associations (Chi-square)
            categorical_assoc = self._analyze_categorical_associations(data)
            
            # Feature-target correlations
            target_corr = None
            if target_column and target_column in data.columns:
                target_corr = self._analyze_target_correlations(
                    data,
                    target_column
                )
            
            # High correlations (potential multicollinearity)
            high_corr = self._identify_high_correlations(numeric_corr)
            
            # Recommendations
            recommendations = self._get_recommendations(high_corr, target_corr)
            
            result.data = {
                "numeric_correlations": numeric_corr,
                "categorical_associations": categorical_assoc,
                "target_correlations": target_corr,
                "high_correlations": high_corr,
                "recommendations": recommendations,
            }
            
            self.logger.success("Correlation analysis complete")
        
        except Exception as e:
            result.add_error(f"Correlation analysis failed: {e}")
            self.logger.error(f"Correlation analysis error: {e}", exc_info=True)
        
        return result
    
    def _analyze_numeric_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric features"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                "message": "Less than 2 numeric features",
                "correlation_matrix": None,
            }
        
        # Correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        return {
            "n_features": len(numeric_cols),
            "correlation_matrix": corr_matrix.to_dict(),
            "features": numeric_cols.tolist(),
        }
    
    def _analyze_categorical_associations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze associations between categorical features using Chi-square"""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) < 2:
            return {
                "message": "Less than 2 categorical features",
                "associations": [],
            }
        
        associations = []
        
        # Calculate pairwise Chi-square tests
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                try:
                    # Create contingency table
                    contingency = pd.crosstab(df[col1], df[col2])
                    
                    # Chi-square test
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    # Cramér's V (effect size)
                    n = contingency.sum().sum()
                    min_dim = min(contingency.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                    
                    associations.append({
                        "feature1": col1,
                        "feature2": col2,
                        "chi2": float(chi2),
                        "p_value": float(p_value),
                        "cramers_v": float(cramers_v),
                        "is_significant": p_value < 0.05,
                    })
                
                except Exception as e:
                    self.logger.warning(
                        f"Chi-square test failed for {col1} vs {col2}: {e}"
                    )
        
        return {
            "n_features": len(categorical_cols),
            "associations": associations,
            "n_significant": sum(1 for a in associations if a["is_significant"]),
        }
    
    def _analyze_target_correlations(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Analyze correlations with target variable"""
        
        target = df[target_column]
        
        # Separate features from target
        features = df.drop(columns=[target_column])
        
        correlations = {}
        
        # Numeric target
        if pd.api.types.is_numeric_dtype(target):
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                corr = features[col].corr(target)
                if not np.isnan(corr):
                    correlations[col] = {
                        "correlation": float(corr),
                        "abs_correlation": float(abs(corr)),
                    }
        
        # Categorical target
        else:
            # For categorical target, calculate point-biserial or ANOVA
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Simple approach: calculate correlation with encoded target
                try:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    target_encoded = le.fit_transform(target.dropna())
                    feature_clean = features[col].loc[target.dropna().index]
                    
                    corr = feature_clean.corr(pd.Series(target_encoded))
                    if not np.isnan(corr):
                        correlations[col] = {
                            "correlation": float(corr),
                            "abs_correlation": float(abs(corr)),
                        }
                except:
                    pass
        
        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(),
            key=lambda x: x[1]["abs_correlation"],
            reverse=True
        )
        
        return {
            "target_column": target_column,
            "correlations": dict(sorted_corr),
            "top_5_features": [feat for feat, _ in sorted_corr[:5]],
        }
    
    def _identify_high_correlations(
        self,
        numeric_corr: Dict[str, Any],
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Identify highly correlated feature pairs"""
        
        if "correlation_matrix" not in numeric_corr or numeric_corr["correlation_matrix"] is None:
            return []
        
        corr_matrix = pd.DataFrame(numeric_corr["correlation_matrix"])
        high_corr = []
        
        # Find pairs with high correlation
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) > threshold:
                    high_corr.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val),
                        "abs_correlation": float(abs(corr_val)),
                    })
        
        # Sort by absolute correlation
        high_corr.sort(key=lambda x: x["abs_correlation"], reverse=True)
        
        return high_corr
    
    def _get_recommendations(
        self,
        high_corr: List[Dict],
        target_corr: Optional[Dict]
    ) -> List[str]:
        """Get recommendations based on correlation analysis"""
        
        recommendations = []
        
        # High correlations (multicollinearity)
        if high_corr:
            recommendations.append(
                f"🔍 Znaleziono {len(high_corr)} par silnie skorelowanych cech (|r| > 0.8). "
                "Rozważ usunięcie jednej z każdej pary, aby uniknąć multicolinearności."
            )
            
            # List top 3 pairs
            for i, pair in enumerate(high_corr[:3], 1):
                recommendations.append(
                    f"  {i}. {pair['feature1']} ↔ {pair['feature2']}: "
                    f"r = {pair['correlation']:.3f}"
                )
        
        # Target correlations
        if target_corr and "top_5_features" in target_corr:
            top_features = target_corr["top_5_features"]
            if top_features:
                recommendations.append(
                    f"📊 Najbardziej skorelowane cechy z targetem: {', '.join(top_features[:3])}"
                )
        
        if not recommendations:
            recommendations.append(
                "✅ Brak silnych korelacji między cechami - dobra różnorodność!"
            )
        
        return recommendations