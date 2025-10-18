# core/llm_client.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — LLM Client v7.0                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE UNIFIED LLM INTERFACE                                        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Multi-Provider Support (Claude, OpenAI, Mock)                         ║
║  ✓ Automatic Retry Logic                                                 ║
║  ✓ Streaming Support                                                     ║
║  ✓ JSON Response Mode                                                    ║
║  ✓ Chat & Completion APIs                                                ║
║  ✓ Token Tracking                                                        ║
║  ✓ Error Handling                                                        ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    LLM Client Structure:
```
    LLMClient (Facade)
    ├── Provider Selection
    │   ├── ClaudeProvider (Anthropic)
    │   ├── OpenAIProvider (OpenAI)
    │   └── MockLLMProvider (Testing)
    ├── Response DTO
    │   ├── Content
    │   ├── Token Usage
    │   ├── Metadata
    │   └── Stream Chunks
    └── Features
        ├── Retry Logic (3 attempts)
        ├── Streaming
        ├── JSON Mode
        └── Chat History
```

Features:
    Multi-Provider:
        • Anthropic Claude
        • OpenAI GPT
        • Mock provider (testing)
        • Automatic selection
    
    Response Modes:
        • Single-shot completion
        • Multi-turn chat
        • Structured JSON
        • Streaming
    
    Resilience:
        • Automatic retry (3 attempts)
        • Exponential backoff
        • Error handling
        • Fallback parsing
    
    Token Management:
        • Usage tracking
        • Cost estimation
        • Metadata capture

Usage:
```python
    from core.llm_client import get_llm_client
    
    # Get client
    llm = get_llm_client()
    
    # Single completion
    response = llm.generate("What is ML?")
    print(response.content)
    
    # Chat conversation
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"}
    ]
    response = llm.chat(messages)
    
    # JSON response
    data = llm.generate_json("List 3 ML algorithms as JSON")
    
    # Streaming
    response = llm.generate("Explain AI", stream=True)
    for chunk in response.stream_chunks:
        print(chunk, end="")
```

Dependencies:
    • anthropic (optional)
    • openai (optional)
    • loguru
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = ["LLMClient", "LLMResponse", "get_llm_client"]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT = """You are an expert AI assistant specializing in data science, 
machine learning, and analytics. Provide clear, accurate, and helpful responses."""


# ═══════════════════════════════════════════════════════════════════════════
# Response DTO
# ═══════════════════════════════════════════════════════════════════════════

class LLMResponse:
    """
    📄 **LLM Response**
    
    Standardized response from LLM providers.
    
    Attributes:
        content: Response text
        model: Model name
        tokens_used: Total tokens consumed
        finish_reason: Completion reason
        metadata: Additional metadata
        stream_chunks: Streaming chunks (if streaming)
    """
    
    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int = 0,
        finish_reason: str = "stop",
        metadata: Optional[Dict] = None,
        stream_chunks: Optional[List[str]] = None
    ):
        """
        Initialize LLM response.
        
        Args:
            content: Response text
            model: Model name
            tokens_used: Total tokens
            finish_reason: Completion reason
            metadata: Additional metadata
            stream_chunks: Streaming chunks
        """
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.finish_reason = finish_reason
        self.metadata = metadata or {}
        self.stream_chunks = stream_chunks or []
    
    def __str__(self) -> str:
        """String representation."""
        return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "content": self.content,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
            "stream_chunks": self.stream_chunks
        }


# ═══════════════════════════════════════════════════════════════════════════
# Base Provider
# ═══════════════════════════════════════════════════════════════════════════

class BaseLLMProvider(ABC):
    """
    🎯 **Base LLM Provider**
    
    Abstract base class for all LLM providers.
    """
    
    DEFAULT_SYSTEM = DEFAULT_SYSTEM_PROMPT
    
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
        """Single-shot completion."""
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
        """Multi-turn conversation."""
        raise NotImplementedError
    
    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Structured JSON response."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Retry Helper
# ═══════════════════════════════════════════════════════════════════════════

def _retry_call(fn, *, attempts=3, base_delay=0.6, on_error=None):
    """
    Retry function with exponential backoff.
    
    Args:
        fn: Function to call
        attempts: Maximum attempts
        base_delay: Base delay in seconds
        on_error: Callback on error
    
    Returns:
        Function result
    
    Raises:
        Last exception if all attempts fail
    """
    last_exc = None
    
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if on_error:
                on_error(e, i)
            if i < attempts:
                delay = base_delay * (2 ** (i - 1))
                time.sleep(delay)
    
    raise last_exc


# ═══════════════════════════════════════════════════════════════════════════
# Claude Provider
# ═══════════════════════════════════════════════════════════════════════════

class ClaudeProvider(BaseLLMProvider):
    """
    🤖 **Anthropic Claude Provider**
    
    Provider for Anthropic's Claude models.
    
    Features:
      • Messages API
      • Streaming support
      • Token tracking
      • Retry logic
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        """
        Initialize Claude provider.
        
        Args:
            api_key: Anthropic API key
            default_model: Default model name
        """
        # Get API key
        try:
            from config.settings import settings
            self.api_key = api_key or settings.ANTHROPIC_API_KEY
            self.default_model = default_model or settings.LLM_MODEL
            self.max_tokens = settings.LLM_MAX_TOKENS
            self.temperature = settings.LLM_TEMPERATURE
        except Exception:
            self.api_key = api_key
            self.default_model = default_model or "claude-3-5-sonnet-20240620"
            self.max_tokens = 4096
            self.temperature = 0.7
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found")
        
        # Initialize client
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
            logger.info("Claude provider initialized")
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            ) from e
    
    def _messages_create(self, **kwargs):
        """Create message with client."""
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
        """Generate completion from Claude."""
        
        def _call():
            return self._messages_create(
                model=kwargs.pop("model", self.default_model),
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                system=system_prompt or self.DEFAULT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                stream=stream,
                **kwargs
            )
        
        def _on_err(e, i):
            logger.warning(f"[Claude] Attempt {i} failed: {e}")
        
        try:
            logger.info(f"[Claude] generate | prompt_len={len(prompt)} | stream={stream}")
            resp = _retry_call(_call, attempts=3, base_delay=0.6, on_error=_on_err)
            
            # Streaming
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
                                text = getattr(delta, "text", "") or ""
                                chunks.append(text)
                                full_text += text
                        
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
                    metadata={
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens
                        }
                    },
                    stream_chunks=chunks
                )
            
            # Non-streaming
            content = resp.content[0].text if resp.content else ""
            tokens_used = (
                (getattr(resp.usage, "input_tokens", 0) or 0) +
                (getattr(resp.usage, "output_tokens", 0) or 0)
            )
            
            return LLMResponse(
                content=content,
                model=resp.model,
                tokens_used=tokens_used,
                finish_reason=getattr(resp, "stop_reason", "stop"),
                metadata={
                    "usage": {
                        "input_tokens": getattr(resp.usage, "input_tokens", 0),
                        "output_tokens": getattr(resp.usage, "output_tokens", 0)
                    }
                }
            )
        
        except Exception as e:
            logger.error(f"Claude API error: {e}", exc_info=True)
            from utils.exceptions import LLMError
            raise LLMError(
                "Claude API communication error",
                details={"original_error": str(e)}
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
        """Native chat through Anthropic Messages API."""
        
        def _call():
            return self._messages_create(
                model=kwargs.pop("model", self.default_model),
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                system=system_prompt or self.DEFAULT_SYSTEM,
                messages=messages,
                stream=stream,
                **kwargs
            )
        
        def _on_err(e, i):
            logger.warning(f"[Claude] Chat attempt {i} failed: {e}")
        
        try:
            logger.info(f"[Claude] chat | n_messages={len(messages)} | stream={stream}")
            resp = _retry_call(_call, attempts=3, base_delay=0.6, on_error=_on_err)
            
            # Streaming
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
                                text = getattr(delta, "text", "") or ""
                                chunks.append(text)
                                full_text += text
                        
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
                    metadata={
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens
                        }
                    },
                    stream_chunks=chunks
                )
            
            # Non-streaming
            content = resp.content[0].text if resp.content else ""
            tokens_used = (
                (getattr(resp.usage, "input_tokens", 0) or 0) +
                (getattr(resp.usage, "output_tokens", 0) or 0)
            )
            
            return LLMResponse(
                content=content,
                model=resp.model,
                tokens_used=tokens_used,
                finish_reason=getattr(resp, "stop_reason", "stop"),
                metadata={
                    "usage": {
                        "input_tokens": getattr(resp.usage, "input_tokens", 0),
                        "output_tokens": getattr(resp.usage, "output_tokens", 0)
                    }
                }
            )
        
        except Exception as e:
            logger.error(f"Claude chat error: {e}", exc_info=True)
            from utils.exceptions import LLMError
            raise LLMError(
                "Claude chat communication error",
                details={"original_error": str(e)}
            )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate JSON response from Claude."""
        json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON, no additional text."
        
        resp = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            **kwargs
        )
        
        return _parse_json_strict(resp.content)


# ═══════════════════════════════════════════════════════════════════════════
# OpenAI Provider
# ═══════════════════════════════════════════════════════════════════════════

class OpenAIProvider(BaseLLMProvider):
    """
    🤖 **OpenAI GPT Provider**
    
    Provider for OpenAI's GPT models.
    
    Features:
      • Chat Completions API
      • Streaming support
      • JSON mode
      • Token tracking
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            default_model: Default model name
        """
        # Get API key
        try:
            from config.settings import settings
            self.api_key = api_key or settings.OPENAI_API_KEY
            
            # Use model from settings if it looks like OpenAI model
            if default_model:
                self.default_model = default_model
            elif "gpt" in (settings.LLM_MODEL or "").lower():
                self.default_model = settings.LLM_MODEL
            else:
                self.default_model = "gpt-4-turbo-preview"
            
            self.max_tokens = settings.LLM_MAX_TOKENS
            self.temperature = settings.LLM_TEMPERATURE
        except Exception:
            self.api_key = api_key
            self.default_model = default_model or "gpt-4-turbo-preview"
            self.max_tokens = 4096
            self.temperature = 0.7
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        # Initialize client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI provider initialized")
        except ImportError as e:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            ) from e
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from OpenAI."""
        
        def _call():
            return self.client.chat.completions.create(
                model=kwargs.pop("model", self.default_model),
                messages=[
                    {"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream,
                **kwargs
            )
        
        def _on_err(e, i):
            logger.warning(f"[OpenAI] Attempt {i} failed: {e}")
        
        try:
            logger.info(f"[OpenAI] generate | prompt_len={len(prompt)} | stream={stream}")
            resp = _retry_call(_call, attempts=3, base_delay=0.6, on_error=_on_err)
            
            # Streaming
            if stream:
                chunks: List[str] = []
                full_text = ""
                model_name = self.default_model
                total_tokens = 0
                
                for ev in resp:
                    try:
                        if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                            text = ev.choices[0].delta.content or ""
                            chunks.append(text)
                            full_text += text
                        
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
                    stream_chunks=chunks
                )
            
            # Non-streaming
            content = resp.choices[0].message.content if resp.choices else ""
            model_name = getattr(resp, "model", self.default_model)
            total_tokens = getattr(resp.usage, "total_tokens", 0)
            
            return LLMResponse(
                content=content or "",
                model=model_name,
                tokens_used=total_tokens,
                finish_reason=(
                    getattr(resp.choices[0], "finish_reason", "stop")
                    if resp.choices else "stop"
                ),
                metadata={
                    "usage": {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                        "total_tokens": total_tokens
                    }
                }
            )
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            from utils.exceptions import LLMError
            raise LLMError(
                "OpenAI API communication error",
                details={"original_error": str(e)}
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
        """Native chat through Chat Completions API."""
        
        def _call():
            msgs = messages[:]
            if system_prompt or self.DEFAULT_SYSTEM:
                msgs = [
                    {"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM}
                ] + msgs
            
            return self.client.chat.completions.create(
                model=kwargs.pop("model", self.default_model),
                messages=msgs,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream,
                **kwargs
            )
        
        def _on_err(e, i):
            logger.warning(f"[OpenAI] Chat attempt {i} failed: {e}")
        
        try:
            logger.info(f"[OpenAI] chat | n_messages={len(messages)} | stream={stream}")
            resp = _retry_call(_call, attempts=3, base_delay=0.6, on_error=_on_err)
            
            # Streaming
            if stream:
                chunks: List[str] = []
                full_text = ""
                model_name = self.default_model
                total_tokens = 0
                
                for ev in resp:
                    try:
                        if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                            text = ev.choices[0].delta.content or ""
                            chunks.append(text)
                            full_text += text
                        
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
                    stream_chunks=chunks
                )
            
            # Non-streaming
            content = resp.choices[0].message.content if resp.choices else ""
            model_name = getattr(resp, "model", self.default_model)
            total_tokens = getattr(resp.usage, "total_tokens", 0)
            
            return LLMResponse(
                content=content or "",
                model=model_name,
                tokens_used=total_tokens,
                finish_reason=(
                    getattr(resp.choices[0], "finish_reason", "stop")
                    if resp.choices else "stop"
                ),
                metadata={
                    "usage": {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                        "total_tokens": total_tokens
                    }
                }
            )
        
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}", exc_info=True)
            from utils.exceptions import LLMError
            raise LLMError(
                "OpenAI chat communication error",
                details={"original_error": str(e)}
            )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate JSON response from OpenAI with JSON mode."""
        try:
            response = self.client.chat.completions.create(
                model=kwargs.pop("model", self.default_model),
                messages=[
                    {"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        
        except Exception as e:
            logger.warning(f"[OpenAI] JSON mode fallback: {e}")
            
            # Fallback to text mode with instruction
            json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON."
            resp = self.generate(
                prompt=json_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                **kwargs
            )
            
            return _parse_json_strict(resp.content)


# ═══════════════════════════════════════════════════════════════════════════
# Mock Provider
# ═══════════════════════════════════════════════════════════════════════════

class MockLLMProvider(BaseLLMProvider):
    """
    🧪 **Mock LLM Provider**
    
    Mock provider for testing without API calls.
    """
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Generate mock response."""
        mock_content = "Mock LLM response for testing purposes."
        return LLMResponse(
            content=mock_content,
            model="mock-model",
            tokens_used=42,
            finish_reason="stop"
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
        """Generate mock chat response."""
        mock_content = "Mock chat response."
        return LLMResponse(
            content=mock_content,
            model="mock-model",
            tokens_used=21,
            finish_reason="stop"
        )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate mock JSON response."""
        return {
            "mock": True,
            "message": "Mock JSON response"
        }


# ═══════════════════════════════════════════════════════════════════════════
# LLM Client (Facade)
# ═══════════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    🎯 **Unified LLM Client**
    
    Facade for multiple LLM providers with automatic selection.
    
    Features:
      • Multi-provider support
      • Automatic provider selection
      • Unified API
      • Error handling
      • Token tracking
    
    Usage:
```python
        llm = LLMClient()
        
        # Generate
        response = llm.generate("What is ML?")
        print(response.content)
        
        # Chat
        messages = [{"role": "user", "content": "Hello"}]
        response = llm.chat(messages)
        
        # JSON
        data = llm.generate_json("List 3 colors as JSON")
"""

def __init__(
    self,
    provider: Optional[Literal["anthropic", "openai", "mock"]] = None
):
    """
    Initialize LLM client.
    
    Args:
        provider: LLM provider to use (default from settings)
    """
    # Get provider name
    try:
        from config.settings import settings
        self.provider_name = provider or settings.DEFAULT_LLM_PROVIDER
        
        # Use mock in test mode
        if settings.USE_MOCK_LLM:
            self.provider_name = "mock"
    except Exception:
        self.provider_name = provider or "anthropic"
    
    # Initialize provider
    self.provider = self._get_provider()
    logger.info(f"LLM Client initialized with provider: {self.provider_name}")

def _get_provider(self) -> BaseLLMProvider:
    """Get provider instance."""
    if self.provider_name == "anthropic":
        return ClaudeProvider()
    elif self.provider_name == "openai":
        return OpenAIProvider()
    elif self.provider_name == "mock":
        return MockLLMProvider()
    else:
        raise ValueError(f"Unknown provider: {self.provider_name}")

def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    **kwargs
) -> LLMResponse:
    """
    🎨 **Generate Completion**
    
    Single-shot text completion.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        stream: Enable streaming
        **kwargs: Provider-specific arguments
    
    Returns:
        LLMResponse
    
    Example:
python            response = llm.generate("Explain machine learning")
            print(response.content)
            print(f"Tokens used: {response.tokens_used}")
    """
    return self.provider.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
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
    """
    💬 **Multi-Turn Chat**
    
    Native multi-turn conversation.
    
    Args:
        messages: Chat message history
        system_prompt: System prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        stream: Enable streaming
        **kwargs: Provider-specific arguments
    
    Returns:
        LLMResponse
    
    Example:
python            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"}
            ]
            response = llm.chat(messages)
    """
    return self.provider.chat(
        messages=messages,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs
    )

def generate_json(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    📋 **Generate JSON Response**
    
    Structured JSON output.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        **kwargs: Provider-specific arguments
    
    Returns:
        Dictionary (parsed JSON)
    
    Example:
python            data = llm.generate_json(
                "List 3 machine learning algorithms with descriptions"
            )
            print(data)
    """
    return self.provider.generate_json(
        prompt=prompt,
        system_prompt=system_prompt,
        **kwargs
    )
# ═══════════════════════════════════════════════════════════════════════════ #
# Utilities                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def _parse_json_strict(content: str) -> Dict[str, Any]:
    """
    Parse JSON with tolerance for code blocks.

    Args:
        content: JSON string (possibly with markdown)

    Returns:
        Parsed dictionary

    Raises:
        LLMError: If parsing fails
    """
    s = content.strip()

    # Remove code fences
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]

    s = s.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {s[:200]}...", exc_info=True)
        from utils.exceptions import LLMError
        raise LLMError(
            "Model did not return valid JSON",
            details={"content_preview": s[:200]}
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Global Singleton                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

_llm_client: Optional[LLMClient] = None

def get_llm_client(
    provider: Optional[Literal["anthropic", "openai", "mock"]] = None
) -> LLMClient:
    """
    🏭 Get LLM Client (Singleton)

    Returns global LLM client instance.

    Args:
        provider: Override provider (optional)

    Returns:
        LLMClient instance

    Example:
        >>> llm = get_llm_client()
        >>> response = llm.generate("Hello!")
        >>> print(response.content)
    """
    global _llm_client

    if _llm_client is None or provider is not None:
        _llm_client = LLMClient(provider=provider)

    return _llm_client


# ═══════════════════════════════════════════════════════════════════════════ #
# Module Self-Test                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("=" * 80)
    print(f"LLM Client v{__version__} - Self Test")
    print("=" * 80)

    # Test with mock provider
    print("\n1. Testing Mock Provider...")
    llm = LLMClient(provider="mock")

    # Test generate
    print("\n2. Testing Generate...")
    response = llm.generate("Test prompt")
    assert response.content == "Mock LLM response for testing purposes."
    print(f"   ✓ Generate works: {response.content[:50]}...")

    # Test chat
    print("\n3. Testing Chat...")
    messages = [{"role": "user", "content": "Hello"}]
    response = llm.chat(messages)
    assert response.content == "Mock chat response."
    print(f"   ✓ Chat works: {response.content}")

    # Test JSON
    print("\n4. Testing JSON Generation...")
    data = llm.generate_json("Test JSON prompt")
    assert data["mock"] is True
    print(f"   ✓ JSON works: {data}")

    # Test response methods
    print("\n5. Testing Response Methods...")
    response = llm.generate("Test")
    response_dict = response.to_dict()
    assert "content" in response_dict
    assert str(response) == response.content
    print("   ✓ Response methods work")

    # Test singleton
    print("\n6. Testing Singleton...")
    llm1 = get_llm_client()
    llm2 = get_llm_client()
    assert llm1 is llm2
    print("   ✓ Singleton works")

    print("\n" + "=" * 80)
    print("USAGE EXAMPLE:")
    print("=" * 80)
    print("""
from core.llm_client import get_llm_client

# === Get Client ===
llm = get_llm_client()

# Or specify provider
llm = get_llm_client(provider="anthropic")
llm = get_llm_client(provider="openai")

# === Single Completion ===
response = llm.generate("What is machine learning?")
print(response.content)
print(f"Tokens used: {response.tokens_used}")
print(f"Model: {response.model}")

# With custom parameters
response = llm.generate(
    "Explain neural networks",
    temperature=0.7,
    max_tokens=1000,
    system_prompt="You are a data science expert"
)

# === Multi-Turn Chat ===
messages = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is artificial intelligence..."},
    {"role": "user", "content": "What about ML?"}
]
response = llm.chat(messages)
print(response.content)

# Continue conversation
messages.append({"role": "assistant", "content": response.content})
messages.append({"role": "user", "content": "Tell me more"})
response = llm.chat(messages)

# === JSON Generation ===
data = llm.generate_json(
    "List 3 machine learning algorithms with their use cases as JSON"
)
print(data)

# Example output:
# {
#   "algorithms": [
#     {"name": "Linear Regression", "use_case": "..."},
#     {"name": "Random Forest", "use_case": "..."},
#     {"name": "Neural Networks", "use_case": "..."}
#   ]
# }

# === Streaming ===
response = llm.generate(
    "Write a story about AI",
    stream=True
)

# Print chunks as they arrive
for chunk in response.stream_chunks:
    print(chunk, end="", flush=True)

print(f"\\n\\nTotal tokens: {response.tokens_used}")

# === Error Handling ===
from utils.exceptions import LLMError

try:
    response = llm.generate("Your prompt")
except LLMError as e:
    print(f"LLM Error: {e.message}")
    print(f"Details: {e.details}")

# === Integration with Agents ===
from core.base_agent import BaseAgent, AgentResult

class LLMAgent(BaseAgent):
    def __init__(self):
        super().__init__("llm_agent")
        self.llm = get_llm_client()

    def execute(self, **kwargs) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        prompt = kwargs.get("prompt")
        response = self.llm.generate(prompt)
        result.add_data(
            response=response.content,
            tokens=response.tokens_used
        )
        return result

# Usage
agent = LLMAgent()
result = agent.run(prompt="Explain machine learning")

# === Custom System Prompt ===
system_prompt = '''
You are a data science expert. Provide clear, accurate,
and practical explanations with code examples when relevant.
'''
response = llm.generate(
    "How do I preprocess data?",
    system_prompt=system_prompt
)

# === Token Tracking ===
response = llm.generate("Long prompt...")
print(f"Input tokens: {response.metadata['usage'].get('input_tokens', 0)}")
print(f"Output tokens: {response.metadata['usage'].get('output_tokens', 0)}")
print(f"Total: {response.tokens_used}")

# === Configuration ===
# In config/settings.py:
# DEFAULT_LLM_PROVIDER = "anthropic"  # or "openai"
# LLM_MODEL = "claude-3-5-sonnet-20240620"
# LLM_MAX_TOKENS = 4096
# LLM_TEMPERATURE = 0.7
# ANTHROPIC_API_KEY = "sk-ant-..."
# OPENAI_API_KEY = "sk-..."

# === Mock for Testing ===
# In config/settings.py:
# USE_MOCK_LLM = True
llm = get_llm_client()  # Automatically uses mock
response = llm.generate("Test")
print(response.content)  # "Mock LLM response for testing purposes."
    """)

    print("\n" + "=" * 80)
    print("✓ Self-test complete")
    print("=" * 80)
