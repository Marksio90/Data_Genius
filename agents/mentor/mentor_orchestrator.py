"""
DataGenius PRO - AI Mentor Orchestrator
Main AI Mentor agent with LLM integration
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger
from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from agents.mentor.prompt_templates import (
    MENTOR_SYSTEM_PROMPT,
    EDA_EXPLANATION_TEMPLATE,
    ML_RESULTS_TEMPLATE,
    RECOMMENDATION_TEMPLATE,
)


class MentorOrchestrator(BaseAgent):
    """
    AI Mentor - provides explanations and recommendations in Polish
    """
    
    def __init__(self):
        super().__init__(
            name="MentorOrchestrator",
            description="AI Mentor for data science guidance"
        )
        self.llm_client = get_llm_client()
    
    def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """
        Process user query with AI Mentor
        
        Args:
            query: User question
            context: Additional context (EDA results, ML results, etc.)
        
        Returns:
            AgentResult with AI Mentor response
        """
        
        result = AgentResult(agent_name=self.name)
        
        try:
            # Build context for LLM
            context_str = self._build_context(context) if context else ""
            
            # Create prompt
            full_prompt = self._create_prompt(query, context_str)
            
            # Get LLM response
            response = self.llm_client.generate(
                prompt=full_prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=2000,
            )
            
            result.data = {
                "response": response.content,
                "query": query,
                "tokens_used": response.tokens_used,
            }
            
            self.logger.success("AI Mentor response generated")
        
        except Exception as e:
            result.add_error(f"AI Mentor failed: {e}")
            self.logger.error(f"AI Mentor error: {e}", exc_info=True)
        
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
            user_level: beginner, intermediate, advanced
        
        Returns:
            Explanation text
        """
        
        try:
            prompt = EDA_EXPLANATION_TEMPLATE.format(
                eda_results=str(eda_results),
                user_level=user_level
            )
            
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=0.7,
            )
            
            return response.content
        
        except Exception as e:
            self.logger.error(f"EDA explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."
    
    def explain_ml_results(
        self,
        ml_results: Dict[str, Any],
        user_level: str = "beginner"
    ) -> str:
        """
        Explain ML results
        
        Args:
            ml_results: ML results from MLOrchestrator
            user_level: User expertise level
        
        Returns:
            Explanation text
        """
        
        try:
            prompt = ML_RESULTS_TEMPLATE.format(
                ml_results=str(ml_results),
                user_level=user_level
            )
            
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=0.7,
            )
            
            return response.content
        
        except Exception as e:
            self.logger.error(f"ML explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."
    
    def generate_recommendations(
        self,
        eda_results: Optional[Dict] = None,
        ml_results: Optional[Dict] = None,
        data_quality: Optional[Dict] = None,
    ) -> List[str]:
        """
        Generate actionable recommendations
        
        Args:
            eda_results: EDA results
            ml_results: ML results
            data_quality: Data quality info
        
        Returns:
            List of recommendations
        """
        
        try:
            context = {
                "eda": eda_results,
                "ml": ml_results,
                "quality": data_quality,
            }
            
            prompt = RECOMMENDATION_TEMPLATE.format(
                context=str(context)
            )
            
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
            )
            
            recommendations = response.get("recommendations", [])
            return recommendations
        
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return ["Nie udało się wygenerować rekomendacji."]
    
    def _build_context(self, context: Dict[str, Any]) -> str:
        """Build context string for LLM"""
        
        context_parts = []
        
        if "eda_results" in context:
            context_parts.append(f"**Wyniki EDA:**\n{context['eda_results']}")
        
        if "ml_results" in context:
            context_parts.append(f"**Wyniki ML:**\n{context['ml_results']}")
        
        if "data_info" in context:
            context_parts.append(f"**Informacje o danych:**\n{context['data_info']}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create full prompt for LLM"""
        
        if context:
            return f"""
Kontekst analizy:
{context}

Pytanie użytkownika:
{query}

Odpowiedz na pytanie użytkownika w kontekście powyższej analizy. Bądź konkretny i praktyczny.
"""
        else:
            return query