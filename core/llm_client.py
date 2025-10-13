"""
DataGenius PRO - LLM Client
Unified interface for LLM providers (Claude, OpenAI, Local)
"""

from typing import Optional, List, Dict, Any, Literal
from abc import ABC, abstractmethod
import json
from loguru import logger
from config.settings import settings
from config.constants import AI_MENTOR_SYSTEM_PROMPT


class LLMResponse:
    """Standardized LLM response"""
    
    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int = 0,
        finish_reason: str = "stop",
        metadata: Optional[Dict] = None
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.finish_reason = finish_reason
        self.metadata = metadata or {}
    
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
        }


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from prompt"""
        pass
    
    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate JSON response"""
        pass


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found")
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
            logger.info("Claude provider initialized")
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from Claude"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.messages.create(
                model=settings.LLM_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or AI_MENTOR_SYSTEM_PROMPT,
                messages=messages,
                **kwargs
            )
            
            # Extract content
            content = response.content[0].text
            
            # Calculate tokens (approximate)
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return LLMResponse(
                content=content,
                model=response.model,
                tokens_used=tokens_used,
                finish_reason=response.stop_reason,
                metadata={
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                }
            )
        
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate JSON response from Claude"""
        
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nOdpowiedz TYLKO w formacie JSON, bez żadnego dodatkowego tekstu."
        
        response = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for structured output
            **kwargs
        )
        
        try:
            # Try to parse JSON
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Extract JSON from markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]   # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            return json.loads(content)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI provider initialized")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from OpenAI"""
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract content
            content = response.choices[0].message.content
            
            return LLMResponse(
                content=content,
                model=response.model,
                tokens_used=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                }
            )
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate JSON response from OpenAI"""
        
        json_prompt = f"{prompt}\n\nOdpowiedz TYLKO w formacie JSON."
        
        response = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            **kwargs
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing"""
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """Generate mock response"""
        
        mock_content = "Mock LLM response for testing purposes."
        
        return LLMResponse(
            content=mock_content,
            model="mock-model",
            tokens_used=100,
            finish_reason="stop",
        )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate mock JSON response"""
        
        return {
            "mock": True,
            "message": "Mock JSON response",
        }


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
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text completion
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments
        
        Returns:
            LLMResponse object
        """
        
        return self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Additional arguments
        
        Returns:
            Parsed JSON dictionary
        """
        
        return self.provider.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Multi-turn chat conversation
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional arguments
        
        Returns:
            LLMResponse object
        """
        
        # Convert messages to prompt format
        prompt = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        ])
        
        return self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get global LLM client instance (singleton)"""
    global _llm_client
    
    if _llm_client is None:
        _llm_client = LLMClient()
    
    return _llm_client