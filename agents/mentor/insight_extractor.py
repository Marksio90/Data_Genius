"""
DataGenius PRO - Insight Extractor
Automatically extracts actionable insights from data and ML results
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client


class InsightExtractor(BaseAgent):
    """
    Extracts actionable insights from data analysis and ML results
    Identifies patterns, anomalies, and opportunities
    """
    
    def __init__(self):
        super().__init__(
            name="InsightExtractor",
            description="Extracts actionable insights from data and results"
        )
        self.llm_client = get_llm_client()
    
    def execute(
        self,
        data: Optional[pd.DataFrame] = None,
        eda_results: Optional[Dict[str, Any]] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """
        Extract insights
        
        Args:
            data: Raw data (optional)
            eda_results: EDA results (optional)
            ml_results: ML results (optional)
        
        Returns:
            AgentResult with extracted insights
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            insights = []
            
            # Extract insights from different sources
            if data is not None:
                insights.extend(self._extract_data_insights(data))
            
            if eda_results is not None:
                insights.extend(self._extract_eda_insights(eda_results))
            
            if ml_results is not None:
                insights.extend(self._extract_ml_insights(ml_results))
            
            # Prioritize insights
            prioritized_insights = self._prioritize_insights(insights)
            
            # Generate summary using LLM
            summary = self._generate_insights_summary(prioritized_insights)
            
            result.data = {
                "insights": prioritized_insights,
                "summary": summary,
                "n_insights": len(prioritized_insights),
                "categories": self._categorize_insights(prioritized_insights),
            }
            
            self.logger.success(f"Extracted {len(prioritized_insights)} insights")
        
        except Exception as e:
            result.add_error(f"Insight extraction failed: {e}")
            self.logger.error(f"Insight extraction error: {e}", exc_info=True)
        
        return result
    
    # ==================== DATA INSIGHTS ====================
    
    def _extract_data_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract insights directly from raw data"""
        
        insights = []
        
        # Size insights
        if len(data) < 100:
            insights.append({
                "type": "data_size",
                "category": "warning",
                "priority": "high",
                "title": "Mały zbiór danych",
                "description": f"Dataset ma tylko {len(data)} wierszy. To może być za mało dla niektórych modeli ML.",
                "recommendation": "Rozważ zbieranie więcej danych lub użycie technik augmentacji.",
                "impact": "medium"
            })
        
        # Imbalance detection
        insights.extend(self._detect_class_imbalance(data))
        
        # Feature distribution insights
        insights.extend(self._analyze_feature_distributions(data))
        
        # Relationship insights
        insights.extend(self._find_interesting_relationships(data))
        
        return insights
    
    def _detect_class_imbalance(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect class imbalance in categorical columns"""
        
        insights = []
        
        for col in data.select_dtypes(include=['object', 'category']).columns:
            value_counts = data[col].value_counts()
            
            if len(value_counts) > 1:
                # Calculate imbalance ratio
                min_class = value_counts.min()
                max_class = value_counts.max()
                imbalance_ratio = max_class / min_class
                
                if imbalance_ratio > 10:
                    insights.append({
                        "type": "class_imbalance",
                        "category": "warning",
                        "priority": "high",
                        "title": f"Niezbalansowane klasy w kolumnie '{col}'",
                        "description": f"Największa klasa ma {imbalance_ratio:.1f}x więcej próbek niż najmniejsza.",
                        "recommendation": "Użyj SMOTE, class weights lub undersampling.",
                        "impact": "high",
                        "details": {
                            "column": col,
                            "imbalance_ratio": float(imbalance_ratio),
                            "distribution": value_counts.to_dict()
                        }
                    })
        
        return insights
    
    def _analyze_feature_distributions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze feature distributions for insights"""
        
        insights = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].dropna()
            
            if len(series) == 0:
                continue
            
            # Check for high skewness
            skewness = series.skew()
            if abs(skewness) > 2:
                insights.append({
                    "type": "distribution",
                    "category": "info",
                    "priority": "medium",
                    "title": f"Silnie skośny rozkład: {col}",
                    "description": f"Kolumna '{col}' ma skośność {skewness:.2f}. Rozkład jest mocno asymetryczny.",
                    "recommendation": "Rozważ transformację logarytmiczną lub Box-Cox.",
                    "impact": "medium",
                    "details": {
                        "column": col,
                        "skewness": float(skewness)
                    }
                })
            
            # Check for constant or near-constant values
            if series.nunique() == 1:
                insights.append({
                    "type": "constant_feature",
                    "category": "warning",
                    "priority": "high",
                    "title": f"Stała wartość: {col}",
                    "description": f"Kolumna '{col}' ma tylko jedną wartość - nie wnosi informacji.",
                    "recommendation": "Usuń tę kolumnę przed trenowaniem modelu.",
                    "impact": "low",
                    "details": {"column": col}
                })
            
            # Check for outliers
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < (Q1 - 3 * IQR)) | (series > (Q3 + 3 * IQR))).sum()
            
            if outliers > len(series) * 0.05:  # More than 5% outliers
                insights.append({
                    "type": "outliers",
                    "category": "info",
                    "priority": "medium",
                    "title": f"Dużo outliers w {col}",
                    "description": f"Wykryto {outliers} outliers ({outliers/len(series)*100:.1f}% danych).",
                    "recommendation": "Sprawdź czy to błędy pomiarowe czy prawdziwe wartości ekstremalne.",
                    "impact": "medium",
                    "details": {
                        "column": col,
                        "n_outliers": int(outliers)
                    }
                })
        
        return insights
    
    def _find_interesting_relationships(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find interesting relationships between features"""
        
        insights = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return insights
        
        # Calculate correlations
        corr_matrix = data[numeric_cols].corr()
        
        # Find strong correlations (but not perfect)
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) > 0.8 and abs(corr_val) < 0.99:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    
                    insights.append({
                        "type": "correlation",
                        "category": "info",
                        "priority": "medium",
                        "title": f"Silna korelacja: {col1} ↔ {col2}",
                        "description": f"Cechy są silnie skorelowane (r={corr_val:.3f}).",
                        "recommendation": "Rozważ usunięcie jednej z cech, aby uniknąć multicolinearności.",
                        "impact": "medium",
                        "details": {
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": float(corr_val)
                        }
                    })
        
        return insights
    
    # ==================== EDA INSIGHTS ====================
    
    def _extract_eda_insights(self, eda_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from EDA results"""
        
        insights = []
        
        # Missing data insights
        if "MissingDataAnalyzer" in eda_results.get("eda_results", {}):
            insights.extend(self._missing_data_insights(
                eda_results["eda_results"]["MissingDataAnalyzer"]
            ))
        
        # Outlier insights
        if "OutlierDetector" in eda_results.get("eda_results", {}):
            insights.extend(self._outlier_insights(
                eda_results["eda_results"]["OutlierDetector"]
            ))
        
        # Correlation insights
        if "CorrelationAnalyzer" in eda_results.get("eda_results", {}):
            insights.extend(self._correlation_insights(
                eda_results["eda_results"]["CorrelationAnalyzer"]
            ))
        
        return insights
    
    def _missing_data_insights(self, missing_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from missing data analysis"""
        
        insights = []
        summary = missing_analysis.get("summary", {})
        
        total_missing = summary.get("total_missing", 0)
        missing_pct = summary.get("missing_percentage", 0)
        
        if total_missing > 0:
            if missing_pct > 20:
                priority = "high"
                category = "warning"
                description = f"Ponad 20% danych jest brakujących! To może poważnie wpłynąć na model."
            elif missing_pct > 5:
                priority = "medium"
                category = "warning"
                description = f"{missing_pct:.1f}% danych jest brakujących."
            else:
                priority = "low"
                category = "info"
                description = f"Tylko {missing_pct:.1f}% danych jest brakujących - niewielki problem."
            
            insights.append({
                "type": "missing_data",
                "category": category,
                "priority": priority,
                "title": "Brakujące dane w zbiorze",
                "description": description,
                "recommendation": "Użyj odpowiedniej strategii imputacji przed trenowaniem.",
                "impact": "high" if missing_pct > 20 else "medium",
                "details": {
                    "total_missing": total_missing,
                    "percentage": missing_pct
                }
            })
        
        return insights
    
    def _outlier_insights(self, outlier_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from outlier detection"""
        
        insights = []
        summary = outlier_analysis.get("summary", {})
        
        total_outliers = summary.get("total_outliers", 0)
        
        if total_outliers > 0:
            insights.append({
                "type": "outliers",
                "category": "info",
                "priority": "medium",
                "title": f"Wykryto {total_outliers} wartości odstających",
                "description": "Outliers mogą być błędami pomiarowymi lub prawdziwymi wartościami ekstremalnymi.",
                "recommendation": "Przeanalizuj outliers manualnie i zdecyduj czy usunąć czy zachować.",
                "impact": "medium",
                "details": {
                    "total_outliers": total_outliers,
                    "n_columns": summary.get("n_columns_with_outliers", 0)
                }
            })
        
        return insights
    
    def _correlation_insights(self, corr_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from correlation analysis"""
        
        insights = []
        high_corr = corr_analysis.get("high_correlations", [])
        
        if high_corr:
            insights.append({
                "type": "multicollinearity",
                "category": "warning",
                "priority": "high",
                "title": "Wykryto multicolinearność",
                "description": f"Znaleziono {len(high_corr)} par silnie skorelowanych cech.",
                "recommendation": "Usuń jedną cechę z każdej pary lub użyj PCA.",
                "impact": "high",
                "details": {
                    "n_pairs": len(high_corr),
                    "top_pairs": high_corr[:3]
                }
            })
        
        return insights
    
    # ==================== ML INSIGHTS ====================
    
    def _extract_ml_insights(self, ml_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from ML results"""
        
        insights = []
        
        # Performance insights
        if "summary" in ml_results:
            insights.extend(self._performance_insights(ml_results["summary"]))
        
        # Feature importance insights
        if "ml_results" in ml_results and "ModelExplainer" in ml_results["ml_results"]:
            insights.extend(self._feature_importance_insights(
                ml_results["ml_results"]["ModelExplainer"]
            ))
        
        return insights
    
    def _performance_insights(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from model performance"""
        
        insights = []
        
        best_score = summary.get("best_score")
        best_model = summary.get("best_model")
        
        if best_score is not None:
            # Determine performance level
            if best_score > 0.9:
                category = "success"
                priority = "low"
                description = f"Świetny wynik! Model osiąga {best_score:.1%} dokładności."
                recommendation = "Model jest gotowy do użycia. Monitoruj performance w production."
            elif best_score > 0.75:
                category = "info"
                priority = "medium"
                description = f"Dobry wynik: {best_score:.1%}. Można jeszcze poprawić."
                recommendation = "Spróbuj feature engineering lub ensemble methods."
            else:
                category = "warning"
                priority = "high"
                description = f"Słaby wynik: {best_score:.1%}. Model wymaga poprawy."
                recommendation = "Sprawdź jakość danych, spróbuj innych algorytmów lub zbierz więcej danych."
            
            insights.append({
                "type": "model_performance",
                "category": category,
                "priority": priority,
                "title": f"Wydajność modelu: {best_model}",
                "description": description,
                "recommendation": recommendation,
                "impact": "high",
                "details": {
                    "score": float(best_score),
                    "model": best_model
                }
            })
        
        return insights
    
    def _feature_importance_insights(self, explainer_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from feature importance"""
        
        insights = []
        
        top_features = explainer_results.get("top_features", [])
        
        if top_features:
            insights.append({
                "type": "feature_importance",
                "category": "success",
                "priority": "medium",
                "title": "Kluczowe cechy zidentyfikowane",
                "description": f"Top 3 cechy: {', '.join(top_features[:3])}",
                "recommendation": "Skup się na jakości tych cech - mają największy wpływ na predykcje.",
                "impact": "high",
                "details": {
                    "top_features": top_features[:5]
                }
            })
            
            # Check if importance is concentrated
            if len(top_features) >= 3:
                insights.append({
                    "type": "feature_concentration",
                    "category": "info",
                    "priority": "low",
                    "title": "Koncentracja ważności cech",
                    "description": "Kilka cech ma dominujący wpływ na model.",
                    "recommendation": "To dobrze - model jest stabilny i interpretowalny.",
                    "impact": "low"
                })
        
        return insights
    
    # ==================== PRIORITIZATION ====================
    
    def _prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize insights by importance"""
        
        # Priority scores
        priority_scores = {
            "high": 3,
            "medium": 2,
            "low": 1
        }
        
        # Sort by priority, then by category
        sorted_insights = sorted(
            insights,
            key=lambda x: (
                -priority_scores.get(x.get("priority", "low"), 0),
                x.get("category", "info")
            )
        )
        
        return sorted_insights
    
    def _categorize_insights(self, insights: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize insights by type"""
        
        categories = {}
        
        for insight in insights:
            category = insight.get("category", "info")
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    # ==================== LLM SUMMARY ====================
    
    def _generate_insights_summary(self, insights: List[Dict[str, Any]]) -> str:
        """Generate natural language summary of insights using LLM"""
        
        if not insights:
            return "Nie znaleziono znaczących insightów."
        
        # Prepare insights for LLM
        insights_text = "\n".join([
            f"- [{i.get('priority', 'medium').upper()}] {i.get('title', '')}: {i.get('description', '')}"
            for i in insights[:10]  # Top 10 insights
        ])
        
        prompt = f"""
Przeanalizuj poniższe insighty z analizy danych i stwórz zwięzłe podsumowanie po polsku.

Insighty:
{insights_text}

Stwórz podsumowanie, które:
1. Rozpoczyna się od najważniejszego odkrycia
2. Grupuje podobne insighty
3. Jest napisane prostym językiem
4. Ma maksymalnie 3-4 zdania

Format: Krótki paragraf (bez bullet points).
"""
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=300
            )
            return response.content.strip()
        
        except Exception as e:
            self.logger.error(f"LLM summary generation failed: {e}")
            return self._fallback_summary(insights)
    
    def _fallback_summary(self, insights: List[Dict[str, Any]]) -> str:
        """Fallback summary when LLM fails"""
        
        n_high = sum(1 for i in insights if i.get("priority") == "high")
        n_medium = sum(1 for i in insights if i.get("priority") == "medium")
        n_low = sum(1 for i in insights if i.get("priority") == "low")
        
        summary = f"Znaleziono {len(insights)} insightów: "
        
        parts = []
        if n_high > 0:
            parts.append(f"{n_high} wysokiego priorytetu")
        if n_medium > 0:
            parts.append(f"{n_medium} średniego")
        if n_low > 0:
            parts.append(f"{n_low} niskiego")
        
        summary += ", ".join(parts) + "."
        
        # Add top insight
        if insights:
            top = insights[0]
            summary += f" Najważniejsze: {top.get('title', '')}."
        
        return summary