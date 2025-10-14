"""
DataGenius PRO - LLM Client
Unified interface for LLM providers (Claude, OpenAI, Local)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Literal
from abc import ABC, abstractmethod
import json
import time

from loguru import logger
from config.settings import settings
from config.constants import AI_MENTOR_SYSTEM_PROMPT
from core.exceptions import LLMError


# =========================
# Response DTO
# =========================

class LLMResponse:
    """Standardized LLM response"""

    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int = 0,
        finish_reason: str = "stop",
        metadata: Optional[Dict] = None,
        stream_chunks: Optional[List[str]] = None,
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.finish_reason = finish_reason
        self.metadata = metadata or {}
        self.stream_chunks = stream_chunks or []

    def __str__(self) -> str:
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
            "stream_chunks": self.stream_chunks,
        }


# =========================
# Base Provider
# =========================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    DEFAULT_SYSTEM = AI_MENTOR_SYSTEM_PROMPT

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Single-shot completion"""
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Multi-turn conversation"""
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Structured JSON response"""
        raise NotImplementedError


# =========================
# Retry helper (bez dodatkowych zależności)
# =========================

def _retry_call(fn, *, attempts=3, base_delay=0.6, on_error=None):
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if on_error:
                on_error(e, i)
            if i < attempts:
                time.sleep(base_delay * (2 ** (i - 1)))
    raise last_exc  # po próbach wyrzucamy ostatni wyjątek


# =========================
# Claude Provider
# =========================

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider"""

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("Anthropic API key not found")

        try:
            from anthropic import Anthropic  # type: ignore
            self.client = Anthropic(api_key=self.api_key)
            logger.info("Claude provider initialized")
        except ImportError as e:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from e

        # domyślny model (z settings lub przekazany)
        self.default_model = default_model or settings.LLM_MODEL

    def _messages_create(self, **kwargs):
        return self.client.messages.create(**kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from Claude"""

        def _call():
            resp = self._messages_create(
                model=kwargs.pop("model", self.default_model),
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                system=system_prompt or self.DEFAULT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                stream=stream,
                **kwargs
            )
            return resp

        def _on_err(e, i):
            logger.warning(f"[Claude] attempt {i} failed: {e}")

        try:
            logger.info(
                f"[Claude] generate | prompt_len={len(prompt)} | stream={stream}"
            )
            resp = _retry_call(_call, attempts=3, base_delay=0.6, on_error=_on_err)

            # streaming
            if stream:
                chunks: List[str] = []
                full_text = ""
                input_tokens = 0
                output_tokens = 0
                for event in resp:
                    # patrzymy tylko na zdarzenia z tekstem
                    try:
                        if getattr(event, "type", None) == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta and getattr(delta, "type", None) == "text_delta":
                                t = getattr(delta, "text", "") or ""
                                chunks.append(t)
                                full_text += t
                        if getattr(event, "type", None) == "message_start":
                            usage = getattr(event, "message", None)
                            if usage and getattr(usage, "usage", None):
                                input_tokens = usage.usage.input_tokens or 0
                        if getattr(event, "type", None) == "message_delta":
                            u = getattr(event, "usage", None)
                            if u and getattr(u, "output_tokens", None) is not None:
                                output_tokens = u.output_tokens or 0
                    except Exception:  # defensywnie, nie psujemy strumienia
                        continue

                return LLMResponse(
                    content=full_text,
                    model=kwargs.get("model", self.default_model),
                    tokens_used=(input_tokens + output_tokens),
                    finish_reason="stop",
                    metadata={"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}},
                    stream_chunks=chunks,
                )

            # non-stream
            content = resp.content[0].text if resp.content else ""
            tokens_used = (getattr(resp, "usage", None).input_tokens or 0) + \
                          (getattr(resp, "usage", None).output_tokens or 0)

            return LLMResponse(
                content=content,
                model=resp.model,
                tokens_used=tokens_used,
                finish_reason=getattr(resp, "stop_reason", "stop"),
                metadata={
                    "usage": {
                        "input_tokens": getattr(resp.usage, "input_tokens", 0),
                        "output_tokens": getattr(resp.usage, "output_tokens", 0),
                    }
                },
            )

        except Exception as e:  # noqa: BLE001
            logger.error(f"Claude API error: {e}", exc_info=True)
            raise LLMError("Błąd komunikacji z Claude.", details={"original_error": str(e)})

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Natywna rozmowa przez Anthropic Messages API"""

        def _call():
            return self._messages_create(
                model=kwargs.pop("model", self.default_model),
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                system=system_prompt or self.DEFAULT_SYSTEM,
                messages=messages,
                stream=stream,
                **kwargs
            )

        try:
            logger.info(f"[Claude] chat | n_messages={len(messages)} | stream={stream}")
            resp = _retry_call(_call, attempts=3, base_delay=0.6, on_error=lambda e, i: logger.warning(f"[Claude] attempt {i} failed: {e}"))

            if stream:
                chunks: List[str] = []
                full_text = ""
                input_tokens = 0
                output_tokens = 0
                for event in resp:
                    try:
                        if getattr(event, "type", None) == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta and getattr(delta, "type", None) == "text_delta":
                                t = getattr(delta, "text", "") or ""
                                chunks.append(t)
                                full_text += t
                        if getattr(event, "type", None) == "message_start":
                            usage = getattr(event, "message", None)
                            if usage and getattr(usage, "usage", None):
                                input_tokens = usage.usage.input_tokens or 0
                        if getattr(event, "type", None) == "message_delta":
                            u = getattr(event, "usage", None)
                            if u and getattr(u, "output_tokens", None) is not None:
                                output_tokens = u.output_tokens or 0
                    except Exception:
                        continue

                return LLMResponse(
                    content=full_text,
                    model=kwargs.get("model", self.default_model),
                    tokens_used=(input_tokens + output_tokens),
                    finish_reason="stop",
                    metadata={"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}},
                    stream_chunks=chunks,
                )

            content = resp.content[0].text if resp.content else ""
            tokens_used = (getattr(resp, "usage", None).input_tokens or 0) + \
                          (getattr(resp, "usage", None).output_tokens or 0)

            return LLMResponse(
                content=content,
                model=resp.model,
                tokens_used=tokens_used,
                finish_reason=getattr(resp, "stop_reason", "stop"),
                metadata={
                    "usage": {
                        "input_tokens": getattr(resp.usage, "input_tokens", 0),
                        "output_tokens": getattr(resp.usage, "output_tokens", 0),
                    }
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Claude chat error: {e}", exc_info=True)
            raise LLMError("Błąd komunikacji z Claude (chat).", details={"original_error": str(e)})

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate JSON response from Claude"""
        # wymuszamy format w prompt
        json_prompt = f"{prompt}\n\nOdpowiedz TYLKO w formacie JSON, bez dodatkowego tekstu."
        resp = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            **kwargs
        )
        return _parse_json_strict(resp.content)


# =========================
# OpenAI Provider
# =========================

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        try:
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI provider initialized")
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e

        # użyj modelu z settings jeśli wygląda na model OpenAI, inaczej fallback
        if default_model:
            self.default_model = default_model
        elif "gpt" in (settings.LLM_MODEL or "").lower():
            self.default_model = settings.LLM_MODEL
        else:
            self.default_model = "gpt-4-turbo-preview"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from OpenAI"""

        def _call():
            response = self.client.chat.completions.create(
                model=kwargs.pop("model", self.default_model),
                messages=(
                    [{"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM}] +
                    [{"role": "user", "content": prompt}]
                ),
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                stream=stream,
                **kwargs
            )
            return response

        def _on_err(e, i):
            logger.warning(f"[OpenAI] attempt {i} failed: {e}")

        try:
            logger.info(f"[OpenAI] generate | prompt_len={len(prompt)} | stream={stream}")
            resp = _retry_call(_call, attempts=3, base_delay=0.6, on_error=_on_err)

            # streaming
            if stream:
                chunks: List[str] = []
                full_text = ""
                model_name = self.default_model
                total_tokens = 0
                for ev in resp:
                    try:
                        if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                            t = ev.choices[0].delta.content or ""
                            chunks.append(t)
                            full_text += t
                        if getattr(ev, "model", None):
                            model_name = ev.model
                        if getattr(ev, "usage", None) and getattr(ev.usage, "total_tokens", None):
                            total_tokens = ev.usage.total_tokens or total_tokens
                    except Exception:
                        continue
                return LLMResponse(
                    content=full_text,
                    model=model_name,
                    tokens_used=total_tokens,
                    finish_reason="stop",
                    metadata={"usage": {"total_tokens": total_tokens}},
                    stream_chunks=chunks,
                )

            # non-stream
            content = resp.choices[0].message.content if resp.choices else ""
            model_name = getattr(resp, "model", self.default_model)
            total_tokens = getattr(resp.usage, "total_tokens", 0)

            return LLMResponse(
                content=content or "",
                model=model_name,
                tokens_used=total_tokens,
                finish_reason=getattr(resp.choices[0], "finish_reason", "stop") if resp.choices else "stop",
                metadata={
                    "usage": {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                        "total_tokens": total_tokens,
                    }
                },
            )

        except Exception as e:  # noqa: BLE001
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise LLMError("Błąd komunikacji z OpenAI.", details={"original_error": str(e)})

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Natywna rozmowa przez Chat Completions"""

        def _call():
            msgs = messages[:]
            if system_prompt or self.DEFAULT_SYSTEM:
                msgs = [{"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM}] + msgs
            return self.client.chat.completions.create(
                model=kwargs.pop("model", self.default_model),
                messages=msgs,
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                stream=stream,
                **kwargs
            )

        try:
            logger.info(f"[OpenAI] chat | n_messages={len(messages)} | stream={stream}")
            resp = _retry_call(_call, attempts=3, base_delay=0.6, on_error=lambda e, i: logger.warning(f"[OpenAI] attempt {i} failed: {e}"))

            if stream:
                chunks: List[str] = []
                full_text = ""
                model_name = self.default_model
                total_tokens = 0
                for ev in resp:
                    try:
                        if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                            t = ev.choices[0].delta.content or ""
                            chunks.append(t)
                            full_text += t
                        if getattr(ev, "model", None):
                            model_name = ev.model
                        if getattr(ev, "usage", None) and getattr(ev.usage, "total_tokens", None):
                            total_tokens = ev.usage.total_tokens or total_tokens
                    except Exception:
                        continue
                return LLMResponse(
                    content=full_text,
                    model=model_name,
                    tokens_used=total_tokens,
                    finish_reason="stop",
                    metadata={"usage": {"total_tokens": total_tokens}},
                    stream_chunks=chunks,
                )

            content = resp.choices[0].message.content if resp.choices else ""
            model_name = getattr(resp, "model", self.default_model)
            total_tokens = getattr(resp.usage, "total_tokens", 0)

            return LLMResponse(
                content=content or "",
                model=model_name,
                tokens_used=total_tokens,
                finish_reason=getattr(resp.choices[0], "finish_reason", "stop") if resp.choices else "stop",
                metadata={
                    "usage": {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                        "total_tokens": total_tokens,
                    }
                },
            )

        except Exception as e:  # noqa: BLE001
            logger.error(f"OpenAI chat error: {e}", exc_info=True)
            raise LLMError("Błąd komunikacji z OpenAI (chat).", details={"original_error": str(e)})

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate JSON response from OpenAI.
        Używa response_format={"type":"json_object"} jeśli dostępne.
        """
        try:
            response = self.client.chat.completions.create(
                model=kwargs.pop("model", self.default_model),
                messages=(
                    [{"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM}] +
                    [{"role": "user", "content": prompt}]
                ),
                temperature=0.3,
                max_tokens=settings.LLM_MAX_TOKENS,
                response_format={"type": "json_object"},  # JSON mode
                **kwargs
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:  # fallback parsowania
            logger.warning(f"[OpenAI] JSON mode fallback: {e}")
            # klasyczny tryb z instrukcją
            json_prompt = f"{prompt}\n\nOdpowiedz TYLKO w formacie JSON."
            resp = self.generate(
                prompt=json_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                **kwargs
            )
            return _parse_json_strict(resp.content)


# =========================
# Mock Provider
# =========================

class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing"""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        mock_content = "Mock LLM response for testing purposes."
        return LLMResponse(
            content=mock_content,
            model="mock-model",
            tokens_used=42,
            finish_reason="stop",
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        mock_content = "Mock chat response."
        return LLMResponse(
            content=mock_content,
            model="mock-model",
            tokens_used=21,
            finish_reason="stop",
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        return {"mock": True, "message": "Mock JSON response"}


# =========================
# LLM Client (Facade)
# =========================

class LLMClient:
    """
    Unified LLM client with automatic provider selection
    """

    def __init__(
        self,
        provider: Optional[Literal["anthropic", "openai", "mock"]] = None
    ):
        """
        Initialize LLM client

        Args:
            provider: LLM provider to use (default from settings)
        """

        self.provider_name = provider or settings.DEFAULT_LLM_PROVIDER

        # Use mock in test mode
        if settings.USE_MOCK_LLM:
            self.provider_name = "mock"

        # Initialize provider
        self.provider = self._get_provider()
        logger.info(f"LLM Client initialized with provider: {self.provider_name}")

    def _get_provider(self) -> BaseLLMProvider:
        """Get provider instance"""
        if self.provider_name == "anthropic":
            return ClaudeProvider()
        if self.provider_name == "openai":
            return OpenAIProvider()
        if self.provider_name == "mock":
            return MockLLMProvider()
        raise ValueError(f"Unknown provider: {self.provider_name}")

    # High-level API

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Single-shot completion"""
        return self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            stream=stream,
            **kwargs
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Multi-turn chat (natywny dla providerów)"""
        return self.provider.chat(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            stream=stream,
            **kwargs
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Structured JSON response"""
        return self.provider.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )


# =========================
# Utils
# =========================

def _parse_json_strict(content: str) -> Dict[str, Any]:
    """
    Parsowanie JSON z tolerancją na code-blocki.
    Rzuca LLMError przy niepowodzeniu.
    """
    s = content.strip()
    # usuń ewentualne code fences
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:  # noqa: F841
        logger.error(f"Nie można sparsować JSON: {s[:200]}...", exc_info=True)
        raise LLMError("Model nie zwrócił poprawnego JSON.")  # świadome uproszczenie


# =========================
# Global singleton
# =========================

_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get global LLM client instance (singleton)"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
