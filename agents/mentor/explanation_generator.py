"""
DataGenius PRO - Explanation Generator
Generates user-friendly explanations of data science concepts and results
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from agents.mentor.prompt_templates import (
    EDA_EXPLANATION_TEMPLATE,
    ML_RESULTS_TEMPLATE,
    MODEL_COMPARISON_TEMPLATE
)


class ExplanationGenerator(BaseAgent):
    """
    Generates clear, user-friendly explanations in Polish
    Translates technical ML/DS concepts into simple language
    """
    
    def __init__(self):
        super().__init__(
            name="ExplanationGenerator",
            description="Generates user-friendly explanations"
        )
        self.llm_client = get_llm_client()
    
    def execute(
        self,
        content_type: str,
        content: Dict[str, Any],
        user_level: str = "beginner",
        **kwargs
    ) -> AgentResult:
        """
        Generate explanation
        
        Args:
            content_type: Type of content (eda, ml_results, model_comparison, feature_importance)
            content: Content to explain
            user_level: User expertise level (beginner, intermediate, advanced)
        
        Returns:
            AgentResult with explanation
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Generate explanation based on content type
            if content_type == "eda":
                explanation = self.explain_eda_results(content, user_level)
            
            elif content_type == "ml_results":
                explanation = self.explain_ml_results(content, user_level)
            
            elif content_type == "model_comparison":
                explanation = self.explain_model_comparison(content, user_level)
            
            elif content_type == "feature_importance":
                explanation = self.explain_feature_importance(content, user_level)
            
            elif content_type == "metrics":
                explanation = self.explain_metrics(content, user_level)
            
            elif content_type == "data_quality":
                explanation = self.explain_data_quality(content, user_level)
            
            else:
                explanation = self.explain_generic(content, user_level)
            
            result.data = {
                "explanation": explanation,
                "content_type": content_type,
                "user_level": user_level,
            }
            
            self.logger.success(f"Explanation generated for {content_type}")
        
        except Exception as e:
            result.add_error(f"Explanation generation failed: {e}")
            self.logger.error(f"Explanation error: {e}", exc_info=True)
        
        return result
    
    def explain_eda_results(
        self,
        eda_results: Dict[str, Any],
        user_level: str = "beginner"
    ) -> str:
        """
        Explain EDA results in user-friendly way
        
        Args:
            eda_results: EDA results from EDAOrchestrator
            user_level: User expertise level
        
        Returns:
            Explanation text in Polish
        """
        
        try:
            # Prepare summary of key findings
            summary = self._summarize_eda_results(eda_results)
            
            # Create prompt
            prompt = EDA_EXPLANATION_TEMPLATE.format(
                eda_results=summary,
                user_level=user_level
            )
            
            # Generate explanation using LLM
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.content
        
        except Exception as e:
            self.logger.error(f"EDA explanation failed: {e}")
            return self._fallback_eda_explanation(eda_results)
    
    def explain_ml_results(
        self,
        ml_results: Dict[str, Any],
        user_level: str = "beginner"
    ) -> str:
        """
        Explain ML training results
        
        Args:
            ml_results: ML results from MLOrchestrator
            user_level: User expertise level
        
        Returns:
            Explanation text in Polish
        """
        
        try:
            # Prepare summary
            summary = self._summarize_ml_results(ml_results)
            
            # Create prompt
            prompt = ML_RESULTS_TEMPLATE.format(
                ml_results=summary,
                user_level=user_level
            )
            
            # Generate explanation
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.content
        
        except Exception as e:
            self.logger.error(f"ML explanation failed: {e}")
            return self._fallback_ml_explanation(ml_results)
    
    def explain_model_comparison(
        self,
        models_comparison: List[Dict[str, Any]],
        user_level: str = "beginner"
    ) -> str:
        """
        Explain comparison between different models
        
        Args:
            models_comparison: List of model results
            user_level: User expertise level
        
        Returns:
            Explanation text in Polish
        """
        
        try:
            prompt = MODEL_COMPARISON_TEMPLATE.format(
                models_comparison=str(models_comparison)
            )
            
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.content
        
        except Exception as e:
            self.logger.error(f"Model comparison explanation failed: {e}")
            return self._fallback_model_comparison(models_comparison)
    
    def explain_feature_importance(
        self,
        feature_importance: Dict[str, Any],
        user_level: str = "beginner"
    ) -> str:
        """
        Explain feature importance in simple terms
        
        Args:
            feature_importance: Feature importance data
            user_level: User expertise level
        
        Returns:
            Explanation text in Polish
        """
        
        prompt = f"""
Wyjaśnij ważność cech (feature importance) w sposób zrozumiały dla użytkownika na poziomie: {user_level}

Dane o ważności cech:
{feature_importance}

Stwórz wyjaśnienie, które:
1. Tłumaczy co oznacza "ważność cechy"
2. Wskazuje 3-5 najważniejszych cech
3. Wyjaśnia dlaczego te cechy są ważne
4. Podaje praktyczne wnioski

Format odpowiedzi:
- Rozpocznij od prostego wyjaśnienia czym jest feature importance
- Lista najważniejszych cech z wyjaśnieniem
- Praktyczne wnioski co to oznacza dla modelu

Poziomy wyjaśnienia:
- beginner: analogie z życia codziennego, zero terminów technicznych
- intermediate: podstawowe terminy ML z wyjaśnieniem
- advanced: techniczna precyzja, metryki, algorytmy
"""
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            return response.content
        
        except Exception as e:
            self.logger.error(f"Feature importance explanation failed: {e}")
            return self._fallback_feature_importance(feature_importance)
    
    def explain_metrics(
        self,
        metrics: Dict[str, float],
        problem_type: str,
        user_level: str = "beginner"
    ) -> str:
        """
        Explain ML metrics in simple terms
        
        Args:
            metrics: Dictionary of metric names and values
            problem_type: classification or regression
            user_level: User expertise level
        
        Returns:
            Explanation text in Polish
        """
        
        prompt = f"""
Wyjaśnij metryki modelu {problem_type} w sposób zrozumiały dla użytkownika na poziomie: {user_level}

Metryki:
{json.dumps(metrics, indent=2)}

Dla każdej metryki wyjaśnij:
1. Co ona oznacza (w prostych słowach)
2. Jaka jest jej wartość i czy to dobrze czy źle
3. Co to oznacza w praktyce

Format:
```
🎯 **[Nazwa metryki]**
- **Wartość**: [wartość] 
- **Co to znaczy**: [proste wyjaśnienie]
- **Ocena**: [dobra/średnia/słaba z uzasadnieniem]
```

Poziomy:
- beginner: analogie, przykłady z życia, bez wzorów
- intermediate: podstawowe wzory, interpretacja
- advanced: precyzja matematyczna, szczegóły
"""
        
        try:
            import json
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1200
            )
            return response.content
        
        except Exception as e:
            self.logger.error(f"Metrics explanation failed: {e}")
            return self._fallback_metrics_explanation(metrics, problem_type)
    
    def explain_data_quality(
        self,
        quality_assessment: Dict[str, Any],
        user_level: str = "beginner"
    ) -> str:
        """
        Explain data quality assessment
        
        Args:
            quality_assessment: Data quality metrics and issues
            user_level: User expertise level
        
        Returns:
            Explanation text in Polish
        """
        
        prompt = f"""
Wyjaśnij ocenę jakości danych dla użytkownika na poziomie: {user_level}

Ocena jakości:
{quality_assessment}

Wyjaśnij:
1. Ogólną ocenę jakości danych (0-100)
2. Najważniejsze problemy z danymi
3. Wpływ tych problemów na model ML
4. Co należy poprawić

Format:
- Rozpocznij od oceny ogólnej z emoji (😊/😐/😞)
- Lista problemów od najważniejszych
- Konkretne rekomendacje

Ton: {user_level == 'beginner' and 'przyjazny, bez żargonu' or 'techniczny, precyzyjny'}
"""
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            return response.content
        
        except Exception as e:
            self.logger.error(f"Data quality explanation failed: {e}")
            return self._fallback_quality_explanation(quality_assessment)
    
    def explain_generic(
        self,
        content: Dict[str, Any],
        user_level: str = "beginner"
    ) -> str:
        """
        Generic explanation for any content
        
        Args:
            content: Content to explain
            user_level: User expertise level
        
        Returns:
            Explanation text
        """
        
        prompt = f"""
Wyjaśnij poniższe wyniki analizy w prosty sposób po polsku dla użytkownika na poziomie: {user_level}

Dane:
{content}

Wyjaśnij kluczowe punkty w sposób zrozumiały i praktyczny.
"""
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            return response.content
        
        except Exception as e:
            self.logger.error(f"Generic explanation failed: {e}")
            return "Nie udało się wygenerować wyjaśnienia."
    
    # ==================== SUMMARY HELPERS ====================
    
    def _summarize_eda_results(self, eda_results: Dict[str, Any]) -> str:
        """Summarize EDA results for LLM prompt"""
        
        summary_parts = []
        
        # Data overview
        if "summary" in eda_results:
            summary = eda_results["summary"]
            summary_parts.append(f"Dataset: {summary.get('dataset_shape', 'N/A')}")
        
        # Key findings
        if "summary" in eda_results and "key_findings" in eda_results["summary"]:
            findings = eda_results["summary"]["key_findings"]
            summary_parts.append(f"Kluczowe odkrycia: {', '.join(findings[:3])}")
        
        # Missing data
        if "eda_results" in eda_results and "MissingDataAnalyzer" in eda_results["eda_results"]:
            missing = eda_results["eda_results"]["MissingDataAnalyzer"]
            total_missing = missing.get("summary", {}).get("total_missing", 0)
            summary_parts.append(f"Braki danych: {total_missing}")
        
        # Outliers
        if "eda_results" in eda_results and "OutlierDetector" in eda_results["eda_results"]:
            outliers = eda_results["eda_results"]["OutlierDetector"]
            total_outliers = outliers.get("summary", {}).get("total_outliers", 0)
            summary_parts.append(f"Outliers: {total_outliers}")
        
        return "\n".join(summary_parts)
    
    def _summarize_ml_results(self, ml_results: Dict[str, Any]) -> str:
        """Summarize ML results for LLM prompt"""
        
        summary_parts = []
        
        if "summary" in ml_results:
            summary = ml_results["summary"]
            
            if "best_model" in summary:
                summary_parts.append(f"Najlepszy model: {summary['best_model']}")
            
            if "best_score" in summary:
                summary_parts.append(f"Wynik: {summary['best_score']:.4f}")
            
            if "key_insights" in summary:
                insights = summary["key_insights"]
                summary_parts.append(f"Insights: {', '.join(insights[:2])}")
        
        return "\n".join(summary_parts)
    
    # ==================== FALLBACK EXPLANATIONS ====================
    
    def _fallback_eda_explanation(self, eda_results: Dict[str, Any]) -> str:
        """Fallback explanation when LLM fails"""
        
        explanation = "🔍 **Podsumowanie Analizy EDA**\n\n"
        
        if "summary" in eda_results:
            summary = eda_results["summary"]
            
            # Dataset info
            if "dataset_shape" in summary:
                shape = summary["dataset_shape"]
                explanation += f"Dataset ma {shape[0]} wierszy i {shape[1]} kolumn.\n\n"
            
            # Key findings
            if "key_findings" in summary:
                explanation += "**Kluczowe odkrycia:**\n"
                for finding in summary["key_findings"]:
                    explanation += f"- {finding}\n"
                explanation += "\n"
            
            # Recommendations
            if "recommendations" in summary:
                explanation += "**Rekomendacje:**\n"
                for rec in summary["recommendations"]:
                    explanation += f"- {rec}\n"
        
        return explanation
    
    def _fallback_ml_explanation(self, ml_results: Dict[str, Any]) -> str:
        """Fallback ML explanation"""
        
        explanation = "🤖 **Wyniki Trenowania Modelu**\n\n"
        
        if "summary" in ml_results:
            summary = ml_results["summary"]
            
            if "best_model" in summary:
                explanation += f"Najlepszy model: **{summary['best_model']}**\n"
            
            if "best_score" in summary:
                score = summary["best_score"]
                explanation += f"Dokładność: **{score:.2%}**\n\n"
                
                if score > 0.9:
                    explanation += "✅ Bardzo dobry wynik!\n"
                elif score > 0.75:
                    explanation += "👍 Dobry wynik.\n"
                else:
                    explanation += "⚠️ Wynik wymaga poprawy.\n"
        
        return explanation
    
    def _fallback_model_comparison(self, models: List[Dict]) -> str:
        """Fallback model comparison"""
        
        explanation = "📊 **Porównanie Modeli**\n\n"
        
        for i, model in enumerate(models[:3], 1):
            model_name = model.get("name", f"Model {i}")
            score = model.get("score", 0)
            explanation += f"{i}. **{model_name}**: {score:.2%}\n"
        
        return explanation
    
    def _fallback_feature_importance(self, importance: Dict) -> str:
        """Fallback feature importance"""
        
        explanation = "🔑 **Ważność Cech**\n\n"
        explanation += "Najważniejsze cechy wpływające na predykcje:\n\n"
        
        # Try to extract top features
        if isinstance(importance, dict):
            if "top_features" in importance:
                for i, feature in enumerate(importance["top_features"][:5], 1):
                    explanation += f"{i}. {feature}\n"
        
        return explanation
    
    def _fallback_metrics_explanation(
        self,
        metrics: Dict[str, float],
        problem_type: str
    ) -> str:
        """Fallback metrics explanation"""
        
        explanation = "📈 **Metryki Modelu**\n\n"
        
        for metric_name, value in metrics.items():
            explanation += f"**{metric_name}**: {value:.4f}\n"
        
        return explanation
    
    def _fallback_quality_explanation(self, quality: Dict) -> str:
        """Fallback quality explanation"""
        
        explanation = "📊 **Ocena Jakości Danych**\n\n"
        
        if "quality_score" in quality:
            score = quality["quality_score"]
            explanation += f"Ogólna ocena: **{score:.1f}/100**\n\n"
            
            if score > 80:
                explanation += "✅ Bardzo dobra jakość danych!\n"
            elif score > 60:
                explanation += "👍 Dobra jakość, drobne poprawki.\n"
            else:
                explanation += "⚠️ Dane wymagają poprawy.\n"
        
        return explanation