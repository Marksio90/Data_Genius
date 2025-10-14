# === OPIS MODUŁU ===
"""
DataGenius PRO - AI Mentor Orchestrator (PRO+++)
Główny agent AI Mentor z integracją LLM: wyjaśnienia EDA/ML, rekomendacje, Q&A.
Defensywna walidacja, kontrola kontekstu, retry/backoff i spójny kontrakt wyników.
"""

# === IMPORTY ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Literal

import math
import time
from loguru import logger
import pandas as pd  # zgodnie z zależnością modułu (może nie być używany bezpośrednio)

from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from agents.mentor.prompt_templates import (
    MENTOR_SYSTEM_PROMPT,
    EDA_EXPLANATION_TEMPLATE,
    ML_RESULTS_TEMPLATE,
    RECOMMENDATION_TEMPLATE,
)


# === KONFIG / PARAMETRY ===
@dataclass(frozen=True)
class MentorConfig:
    """Ustawienia działania Mentora."""
    temperature: float = 0.7
    max_tokens: int = 2000                # dla odpowiedzi tekstowych
    json_max_tokens: int = 1200           # dla odpowiedzi JSON
    retries: int = 2                      # dodatkowe próby po błędach tymczasowych
    initial_backoff_s: float = 0.6        # początkowy backoff
    context_max_chars: int = 60_000       # twardy limit znaków kontekstu do LLM
    trim_section_head: int = 8_000        # ile znaków pobrać z początku sekcji przy trimie
    trim_section_tail: int = 2_000        # ile z końca (np. rekomendacje/podsumowania)
    # hint: jeśli klient LLM wspiera timeout — warto użyć (tu tylko semantycznie)
    request_timeout_s: Optional[float] = None


# === KLASA GŁÓWNA ===
class MentorOrchestrator(BaseAgent):
    """
    AI Mentor - provides explanations and recommendations in Polish.
    """

    def __init__(self, config: Optional[MentorConfig] = None) -> None:
        super().__init__(
            name="MentorOrchestrator",
            description="AI Mentor for data science guidance"
        )
        self.llm_client = get_llm_client()
        self.config = config or MentorConfig()

    # === WYKONANIE GŁÓWNE ===
    def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Process user query with AI Mentor.

        Args:
            query: User question (required)
            context: Additional context (EDA results, ML results, data_info)

        Returns:
            AgentResult with AI Mentor response
        """
        result = AgentResult(agent_name=self.name)

        try:
            # Walidacja wejścia
            if not isinstance(query, str) or not query.strip():
                msg = "MentorOrchestrator: 'query' must be a non-empty string."
                result.add_error(msg)
                self.logger.error(msg)
                return result
            if context is not None and not isinstance(context, dict):
                msg = "MentorOrchestrator: 'context' must be a dict if provided."
                result.add_error(msg)
                self.logger.error(msg)
                return result

            # Zbuduj/obtnij kontekst
            context_str = self._build_context(context or {})
            context_str = self._trim_context_if_needed(context_str)

            # Stwórz prompt
            full_prompt = self._create_prompt(query, context_str)

            # Wywołanie LLM z retry/backoff
            response = self._call_llm_with_retry(
                prompt=full_prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # Uporządkuj wynik
            content = getattr(response, "content", None) or str(response)
            tokens_used = getattr(response, "tokens_used", None)

            result.data = {
                "response": content,
                "query": query,
                "tokens_used": tokens_used,
                "meta": {
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "context_chars": len(context_str),
                }
            }
            self.logger.success("AI Mentor response generated")

        except Exception as e:
            result.add_error(f"AI Mentor failed: {e}")
            self.logger.error(f"AI Mentor error: {e}", exc_info=True)

        return result

    # === WYJAŚNIENIA EDA ===
    def explain_eda_results(
        self,
        eda_results: Dict[str, Any],
        user_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    ) -> str:
        """Explain EDA results in user-friendly way."""
        try:
            if not isinstance(eda_results, dict):
                raise ValueError("'eda_results' must be a dict")

            eda_str = self._trim_context_if_needed(self._safe_stringify(eda_results))
            prompt = EDA_EXPLANATION_TEMPLATE.format(
                eda_results=eda_str,
                user_level=user_level
            )

            response = self._call_llm_with_retry(
                prompt=prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return getattr(response, "content", "") or "—"
        except Exception as e:
            self.logger.error(f"EDA explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."

    # === WYJAŚNIENIA ML ===
    def explain_ml_results(
        self,
        ml_results: Dict[str, Any],
        user_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    ) -> str:
        """Explain ML results."""
        try:
            if not isinstance(ml_results, dict):
                raise ValueError("'ml_results' must be a dict")

            ml_str = self._trim_context_if_needed(self._safe_stringify(ml_results))
            prompt = ML_RESULTS_TEMPLATE.format(
                ml_results=ml_str,
                user_level=user_level
            )

            response = self._call_llm_with_retry(
                prompt=prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return getattr(response, "content", "") or "—"
        except Exception as e:
            self.logger.error(f"ML explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."

    # === REKOMENDACJE ===
    def generate_recommendations(
        self,
        eda_results: Optional[Dict[str, Any]] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        data_quality: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate actionable recommendations as a list of strings.
        """
        try:
            context = {
                "eda": eda_results or {},
                "ml": ml_results or {},
                "quality": data_quality or {},
            }
            ctx_str = self._trim_context_if_needed(self._safe_stringify(context))

            prompt = RECOMMENDATION_TEMPLATE.format(context=ctx_str)

            # prefer JSON zwrot
            resp_json = self._call_llm_json_with_retry(
                prompt=prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                max_tokens=self.config.json_max_tokens
            )

            recs = []
            if isinstance(resp_json, dict):
                # wspieraj zarówno {"recommendations": [...]}, jak i prostą listę
                if "recommendations" in resp_json and isinstance(resp_json["recommendations"], list):
                    recs = [str(x) for x in resp_json["recommendations"]]
                elif isinstance(resp_json.get("data"), list):
                    recs = [str(x) for x in resp_json["data"]]
            if not recs:
                # fallback: spróbuj zwykły generate i heurystycznie pociąć po liniach
                self.logger.warning("LLM JSON empty/invalid. Falling back to text generation for recommendations.")
                txt = self._call_llm_with_retry(
                    prompt=prompt,
                    system_prompt=MENTOR_SYSTEM_PROMPT,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                content = getattr(txt, "content", "") or ""
                recs = [line.strip("-• ").strip() for line in content.splitlines() if line.strip()]

            # sanity
            recs = [r for r in recs if r]
            return recs or ["Nie udało się wygenerować rekomendacji."]

        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return ["Nie udało się wygenerować rekomendacji."]

    # === KONTEXT ===
    def _build_context(self, context: Dict[str, Any]) -> str:
        """Składa kontekst z priorytetem sekcji i czytelnymi nagłówkami."""
        parts: List[str] = []

        if "eda_results" in context:
            parts.append("**Wyniki EDA:**\n" + self._safe_stringify(context["eda_results"]))
        if "ml_results" in context:
            parts.append("**Wyniki ML:**\n" + self._safe_stringify(context["ml_results"]))
        if "data_info" in context:
            parts.append("**Informacje o danych:**\n" + self._safe_stringify(context["data_info"]))

        return "\n\n".join(parts).strip()

    def _create_prompt(self, query: str, context: str) -> str:
        """Składa kompletny prompt dla LLM (z kontekstem jeśli jest)."""
        if context:
            return f"""Kontekst analizy:
{context}

Pytanie użytkownika:
{query}

Odpowiedz po polsku, rzeczowo i praktycznie. Preferuj krótkie sekcje, listy punktowane i jasne rekomendacje."""
        return query

    # === LLM CALLS Z RETRY/BACKOFF ===
    def _call_llm_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ):
        """Wywołuje LLM.generate z retry/backoff (błędy chwilowe / rate limit)."""
        attempts = self.config.retries + 1
        delay = self.config.initial_backoff_s
        last_err: Optional[Exception] = None

        for i in range(attempts):
            try:
                return self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.request_timeout_s,
                )
            except Exception as e:
                last_err = e
                self.logger.warning(f"LLM generate failed (attempt {i+1}/{attempts}): {e}")
                if i < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"LLM generate failed after {attempts} attempts: {last_err}")

    def _call_llm_json_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int
    ) -> Dict[str, Any]:
        """Wywołuje LLM.generate_json z retry/backoff; zwraca dict lub {}."""
        attempts = self.config.retries + 1
        delay = self.config.initial_backoff_s
        last_err: Optional[Exception] = None

        for i in range(attempts):
            try:
                resp = self.llm_client.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    timeout=self.config.request_timeout_s,
                )
                if isinstance(resp, dict):
                    return resp
                # niektóre klienty zwracają obiekt z .json lub .content
                if hasattr(resp, "json") and isinstance(resp.json, dict):
                    return resp.json
                if hasattr(resp, "content") and isinstance(resp.content, str):
                    # awaryjny parse: prosta heurystyka (jeśli to json string)
                    import json
                    try:
                        return json.loads(resp.content)
                    except Exception:
                        pass
                return {}
            except Exception as e:
                last_err = e
                self.logger.warning(f"LLM generate_json failed (attempt {i+1}/{attempts}): {e}")
                if i < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
        self.logger.error(f"LLM generate_json failed after {attempts} attempts: {last_err}")
        return {}

    # === POMOCNICZE ===
    def _safe_stringify(self, obj: Any) -> str:
        """Bezpieczne i krótkie zamienianie dict/list na string (dla promptu)."""
        try:
            import json
            return json.dumps(obj, ensure_ascii=False, default=str)[: self.config.context_max_chars]
        except Exception:
            return str(obj)[: self.config.context_max_chars]

    def _trim_context_if_needed(self, context: str) -> str:
        """Jeśli kontekst przekracza limit znaków — skraca selektywnie (head + tail)."""
        max_len = self.config.context_max_chars
        if len(context) <= max_len:
            return context
        head = context[: self.config.trim_section_head]
        tail = context[-self.config.trim_section_tail :]
        note = f"\n\n[Uwaga: kontekst skrócony do {max_len} znaków]"
        return head + "\n...\n" + tail + note
