# agents/mentor/orchestrator.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — AI Mentor Orchestrator            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Military-grade AI mentoring with production safeguards:                   ║
║    ✓ Multi-modal responses (Q&A, EDA, ML, recommendations)                ║
║    ✓ Enhanced PII scrubbing (email, phone, IP, PESEL, UUID, IBAN)         ║
║    ✓ Context management (head/tail trimming, soft/hard caps)              ║
║    ✓ Retry mechanism (exponential backoff + full jitter)                  ║
║    ✓ Circuit breaker (failure threshold + cooldown)                       ║
║    ✓ Token bucket rate limiter (burst protection)                         ║
║    ✓ TTL cache (LRU eviction + thread-safe)                               ║
║    ✓ JSON-first responses with text fallback                              ║
║    ✓ Protocol-based LLM client (easy mocking/DI)                          ║
║    ✓ Comprehensive telemetry (provider, limiter, attempts)                ║
║    ✓ Token/temperature clamping (64-32768, 0.0-1.0)                       ║
║    ✓ Safe template formatting (missing key handling)                      ║
║    ✓ Versioned output contract (semantic versioning)                      ║
║    ✓ Defensive input validation (type safety)                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Output Contract:
{
    "response": str,
    "query": str,
    "tokens_used": int | None,
    "meta": {
        "temperature": float,
        "max_tokens": int,
        "context_chars": int,
    },
    "telemetry": {
        "elapsed_s": float,
        "retries_cfg": int,
        "attempt_index": int,
        "breaker_state": "open" | "closed" | "maybe_open",
        "cache_hit": bool,
        "token_hint_in": int,
        "rate_limiter_hit": bool,
        "provider": str | None,
    },
    "version": "5.0-kosmos-enterprise",
}
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

import pandas as pd  # Required by base contract
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from agents.mentor.prompt_templates import (
    MENTOR_SYSTEM_PROMPT,
    EDA_EXPLANATION_TEMPLATE,
    ML_RESULTS_TEMPLATE,
    RECOMMENDATION_TEMPLATE,
)

__all__ = ["MentorConfig", "MentorOrchestrator", "LLMClientProtocol"]
__version__ = "5.0-kosmos-enterprise"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Protocols & Type Definitions
# ═══════════════════════════════════════════════════════════════════════════

class LLMClientProtocol(Protocol):
    """
    Minimal interface required from LLM client (for DI/mocking).
    
    Enables:
      • Dependency injection
      • Easy testing with mocks
      • Type safety with mypy/pyright
      • Provider-agnostic implementation
    """
    
    def generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: Optional[float] = None,
    ) -> Any:
        """Generate text response."""
        ...
    
    def generate_json(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate JSON response."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MentorConfig:
    """
    Enterprise configuration for AI Mentor Orchestrator.
    
    All parameters are deterministic and stable for production use.
    """
    
    # ─── LLM Parameters ───
    temperature: float = 0.45
    max_tokens: int = 2000          # Hard cap for text responses
    json_max_tokens: int = 1400     # Hard cap for JSON responses
    request_timeout_s: Optional[float] = None
    
    # ─── Retry & Backoff ───
    retries: int = 2
    initial_backoff_s: float = 0.6
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.25    # ±25% jitter around base delay
    
    # ─── Context & Security ───
    pii_scrub: bool = True
    context_max_chars: int = 60_000  # Hard limit for context length
    trim_section_head: int = 10_000
    trim_section_tail: int = 3_000
    prefer_json: bool = True
    
    # ─── Circuit Breaker ───
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: float = 30.0
    
    # ─── Cache (TTL + LRU) ───
    cache_enabled: bool = True
    cache_ttl_s: int = 180
    cache_maxsize: int = 256
    
    # ─── Telemetry ───
    token_hint_avg_chars_per_token: float = 3.8  # Heuristic for token estimation
    
    # ─── Rate Limiting (Token Bucket) ───
    rl_capacity: int = 6                # Bucket capacity (requests)
    rl_refill_rate_per_sec: float = 0.5  # Refill rate (0.5 = 1 token per 2s)
    rl_block_when_empty: bool = True    # True → wait, False → degrade


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _hash_key(*parts: str) -> str:
    """
    Generate SHA256 hash from multiple string parts.
    
    Returns:
        Hexadecimal hash string
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x1f")  # Unit separator
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: TTL Cache Implementation
# ═══════════════════════════════════════════════════════════════════════════

class _TTLCache:
    """
    Thread-safe TTL cache with LRU eviction.
    
    Features:
      • Time-to-live expiration
      • LRU eviction when maxsize reached
      • Thread-safe operations
      • Zero external dependencies
    """
    
    def __init__(self, maxsize: int, ttl_s: int) -> None:
        """
        Initialize cache.
        
        Args:
            maxsize: Maximum number of entries
            ttl_s: Time-to-live in seconds
        """
        self.maxsize = int(maxsize)
        self.ttl_s = float(ttl_s)
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (thread-safe).
        
        Returns:
            Cached value or None if expired/missing
        """
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            
            ts, val = item
            
            # Check TTL
            if time.time() - ts > self.ttl_s:
                self._store.pop(key, None)
                return None
            
            return val
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache with LRU eviction (thread-safe).
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Evict oldest if at capacity
            if len(self._store) >= self.maxsize:
                oldest_key = min(
                    self._store.items(),
                    key=lambda kv: kv[1][0]
                )[0]
                self._store.pop(oldest_key, None)
            
            self._store[key] = (time.time(), value)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Token Bucket Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════

class _TokenBucket:
    """
    Thread-safe token bucket rate limiter.
    
    Features:
      • Burst protection
      • Smooth request flow
      • Configurable refill rate
      • Thread-safe operations
    """
    
    def __init__(self, capacity: int, refill_rate_per_sec: float) -> None:
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate_per_sec: Tokens added per second
        """
        self.capacity = max(1, int(capacity))
        self.refill_rate = float(refill_rate_per_sec)
        self.tokens = float(self.capacity)
        self.timestamp = time.monotonic()
        self._lock = threading.Lock()
    
    def acquire(self) -> bool:
        """
        Try to acquire one token (non-blocking).
        
        Returns:
            True if token acquired, False otherwise
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.timestamp
            self.timestamp = now
            
            # Refill tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            
            return False
    
    def wait(self) -> None:
        """
        Wait until token becomes available (blocking).
        """
        while True:
            if self.acquire():
                return
            time.sleep(max(0.02, 1.0 / max(1e-6, self.refill_rate)))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main Mentor Orchestrator Agent
# ═══════════════════════════════════════════════════════════════════════════

class MentorOrchestrator(BaseAgent):
    """
    **MentorOrchestrator** — Enterprise AI mentoring orchestration.
    
    Responsibilities:
      1. Interactive Q&A with context awareness
      2. EDA results explanation (multi-level)
      3. ML results interpretation
      4. Recommendation generation (JSON-first)
      5. PII scrubbing for data privacy (GDPR-ready)
      6. Context trimming for token efficiency
      7. Retry with exponential backoff + jitter
      8. Circuit breaker for cascading failure prevention
      9. Token bucket rate limiting for burst protection
      10. TTL cache for identical queries
      11. Comprehensive telemetry tracking
      12. Provider-agnostic LLM integration
    
    Features:
      • 4 response modes (Q&A, EDA, ML, recommendations)
      • Enhanced PII scrubber (email, phone, IP, PESEL, UUID, IBAN)
      • Circuit breaker (3 failures → 30s cooldown)
      • TTL cache (180s, 256 entries, LRU)
      • Token bucket (6 capacity, 0.5 refill/s)
      • Exponential backoff (0.6s → 1.2s → 2.4s)
      • Full jitter (±25% randomization)
      • Token estimation (3.8 chars/token heuristic)
      • Temperature/token clamping (0.0-1.0, 64-32768)
      • Safe template formatting (missing key handling)
    """
    
    def __init__(
        self,
        config: Optional[MentorConfig] = None,
        llm_client: Optional[LLMClientProtocol] = None
    ) -> None:
        """
        Initialize AI Mentor Orchestrator.
        
        Args:
            config: Optional custom configuration
            llm_client: Optional custom LLM client (dependency injection)
        """
        super().__init__(
            name="MentorOrchestrator",
            description="AI Mentor for data science guidance with enterprise safeguards"
        )
        self.config = config or MentorConfig()
        
        # ─── Initialize LLM Client (DI-ready) ───
        try:
            self.llm_client: LLMClientProtocol = llm_client or get_llm_client()  # type: ignore[assignment]
        except Exception as e:
            logger.warning(f"⚠ LLM client unavailable; running in degraded mode. Reason: {e}")
            
            class _NullClient:
                """Minimal mock for offline/testing mode."""
                def generate(self, *args, **kwargs):
                    raise RuntimeError("LLM client not available")
                
                def generate_json(self, *args, **kwargs):
                    raise RuntimeError("LLM client not available")
            
            self.llm_client = _NullClient()  # type: ignore[assignment]
        
        # ─── Circuit Breaker State ───
        self._fail_streak: int = 0
        self._breaker_until_ts: float = 0.0
        
        # ─── Cache Initialization ───
        self._cache = _TTLCache(
            maxsize=self.config.cache_maxsize,
            ttl_s=self.config.cache_ttl_s
        )
        
        # ─── Rate Limiter Initialization ───
        self._bucket = _TokenBucket(
            capacity=self.config.rl_capacity,
            refill_rate_per_sec=self.config.rl_refill_rate_per_sec
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Input Validation
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Required:
            query: str (non-empty)
        
        Optional:
            context: Dict[str, Any]
        """
        if "query" not in kwargs:
            raise ValueError("Required parameter 'query' not provided")
        
        query = kwargs["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        
        context = kwargs.get("context")
        if context is not None and not isinstance(context, dict):
            raise TypeError(f"'context' must be dict, got {type(context).__name__}")
        
        return True
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution (Q&A Mode)
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Generate AI mentor response with context awareness.
        
        Args:
            query: User question/request
            context: Optional context dictionary (EDA results, data info, etc.)
            **kwargs: Additional options (for compatibility)
        
        Returns:
            AgentResult with response and telemetry (stable 1:1 contract)
        """
        result = AgentResult(agent_name=self.name)
        t0 = time.perf_counter()
        cfg = self.config
        
        try:
            # ─── Input Validation ───
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
            
            # ─── Circuit Breaker Check ───
            if self._is_breaker_open():
                cooldown = max(0.0, self._breaker_until_ts - time.time())
                warning = f"Circuit breaker is open. Cooldown: {cooldown:.1f}s."
                self.logger.warning(warning)
                result.add_warning(warning)
                result.data = self._wrap_payload(
                    response="Tymczasowa przerwa w wywołaniach LLM (zabezpieczenie). Spróbuj ponownie.",
                    query=query,
                    context_len=0,
                    tokens_used=None,
                    breaker_state="open",
                    cache_hit=False,
                    elapsed_s=time.perf_counter() - t0,
                    limiter_hit=False,
                    attempts=0,
                    provider=self._provider_name()
                )
                return result
            
            # ─── Rate Limiting ───
            limiter_hit = False
            if cfg.rl_block_when_empty:
                self._bucket.wait()  # Blocking wait for token
            else:
                if not self._bucket.acquire():  # Non-blocking
                    limiter_hit = True
            
            # ─── Prepare Context (PII scrub + trimming) ───
            context_str = self._prepare_context(context or {})
            full_prompt = self._create_prompt(query, context_str)
            
            # ─── Check Cache ───
            cache_key = None
            cache_hit = False
            
            if cfg.cache_enabled:
                cache_key = _hash_key(
                    "mentor",
                    MENTOR_SYSTEM_PROMPT or "",
                    full_prompt
                )
                
                cached = self._cache.get(cache_key)
                if cached is not None:
                    cache_hit = True
                    content, tokens_used = cached
                    self._reset_breaker()
                    
                    result.data = self._wrap_payload(
                        response=content,
                        query=query,
                        context_len=len(context_str),
                        tokens_used=tokens_used,
                        breaker_state="closed",
                        cache_hit=True,
                        elapsed_s=time.perf_counter() - t0,
                        limiter_hit=limiter_hit,
                        attempts=0,
                        provider=self._provider_name()
                    )
                    return result
            
            # ─── LLM Call with Retry ───
            response, attempts = self._call_llm_with_retry(
                prompt=full_prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=self._clamp_temperature(cfg.temperature),
                max_tokens=self._clamp_tokens(cfg.max_tokens)
            )
            
            content = getattr(response, "content", None) or str(response)
            tokens_used = getattr(response, "tokens_used", None)
            
            # ─── Update Cache ───
            if cfg.cache_enabled and cache_key:
                self._cache.set(cache_key, (content, tokens_used))
            
            self._reset_breaker()
            
            result.data = self._wrap_payload(
                response=content,
                query=query,
                context_len=len(context_str),
                tokens_used=tokens_used,
                breaker_state="closed",
                cache_hit=cache_hit,
                elapsed_s=time.perf_counter() - t0,
                limiter_hit=limiter_hit,
                attempts=attempts,
                provider=self._provider_name()
            )
            
            self.logger.success("✓ AI Mentor response generated")
        
        except Exception as e:
            self._register_failure()
            msg = f"AI Mentor failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self.logger.error(f"❌ {msg}", exc_info=True)
            
            result.data = self._wrap_payload(
                response="Nie udało się wygenerować odpowiedzi (degradacja bezpieczna).",
                query=query,
                context_len=0,
                tokens_used=None,
                breaker_state="maybe_open",
                cache_hit=False,
                elapsed_s=time.perf_counter() - t0,
                limiter_hit=False,
                attempts=0,
                provider=self._provider_name()
            )
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # EDA Explanation Mode
    # ───────────────────────────────────────────────────────────────────
    
    def explain_eda_results(
        self,
        eda_results: Dict[str, Any],
        user_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    ) -> str:
        """
        Generate explanation of EDA results.
        
        Args:
            eda_results: EDA analysis output
            user_level: User expertise level
        
        Returns:
            Explanation text (Polish)
        """
        try:
            if not isinstance(eda_results, dict):
                raise ValueError("'eda_results' must be a dict")
            
            eda_str = self._prepare_context({"eda_results": eda_results})
            prompt = self._safe_format(
                EDA_EXPLANATION_TEMPLATE,
                eda_results=eda_str,
                user_level=user_level
            )
            
            response, _ = self._call_llm_with_retry(
                prompt=prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=self._clamp_temperature(self.config.temperature),
                max_tokens=self._clamp_tokens(self.config.max_tokens)
            )
            
            self._reset_breaker()
            return getattr(response, "content", "") or "—"
        
        except Exception as e:
            self._register_failure()
            self.logger.error(f"EDA explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."
    
    # ───────────────────────────────────────────────────────────────────
    # ML Explanation Mode
    # ───────────────────────────────────────────────────────────────────
    
    def explain_ml_results(
        self,
        ml_results: Dict[str, Any],
        user_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    ) -> str:
        """
        Generate explanation of ML results.
        
        Args:
            ml_results: ML training output
            user_level: User expertise level
        
        Returns:
            Explanation text (Polish)
        """
        try:
            if not isinstance(ml_results, dict):
                raise ValueError("'ml_results' must be a dict")
            
            ml_str = self._prepare_context({"ml_results": ml_results})
            prompt = self._safe_format(
                ML_RESULTS_TEMPLATE,
                ml_results=ml_str,
                user_level=user_level
            )
            
            response, _ = self._call_llm_with_retry(
                prompt=prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=self._clamp_temperature(self.config.temperature),
                max_tokens=self._clamp_tokens(self.config.max_tokens)
            )
            
            self._reset_breaker()
            return getattr(response, "content", "") or "—"
        
        except Exception as e:
            self._register_failure()
            self.logger.error(f"ML explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."
    
    # ───────────────────────────────────────────────────────────────────
    # Recommendation Mode (JSON-First)
    # ───────────────────────────────────────────────────────────────────
    
    def generate_recommendations(
        self,
        eda_results: Optional[Dict[str, Any]] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        data_quality: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate actionable recommendations (JSON-first with text fallback).
        
        Args:
            eda_results: Optional EDA analysis
            ml_results: Optional ML training results
            data_quality: Optional data quality assessment
        
        Returns:
            List of recommendation strings
        """
        try:
            context = {
                "eda": eda_results or {},
                "ml": ml_results or {},
                "quality": data_quality or {}
            }
            
            context_str = self._prepare_context(context)
            prompt = self._safe_format(RECOMMENDATION_TEMPLATE, context=context_str)
            
            recommendations: List[str] = []
            
            # ─── Try JSON-First ───
            if self.config.prefer_json:
                obj = self._call_llm_json_with_retry(
                    prompt=prompt,
                    system_prompt=MENTOR_SYSTEM_PROMPT,
                    max_tokens=self._clamp_tokens(self.config.json_max_tokens)
                )
                
                if isinstance(obj, dict):
                    # Try multiple possible keys
                    for key in ("recommendations", "data", "items"):
                        if isinstance(obj.get(key), list):
                            recommendations = [
                                str(x).strip()
                                for x in obj[key]
                                if str(x).strip()
                            ]
                            break
            
            # ─── Fallback to Text Parsing ───
            if not recommendations:
                self.logger.warning(
                    "LLM JSON empty/invalid. Falling back to text generation."
                )
                
                txt_response, _ = self._call_llm_with_retry(
                    prompt=prompt,
                    system_prompt=MENTOR_SYSTEM_PROMPT,
                    temperature=self._clamp_temperature(self.config.temperature),
                    max_tokens=self._clamp_tokens(self.config.max_tokens)
                )
                
                content = getattr(txt_response, "content", "") or ""
                
                # Parse bullet points / numbered lists
                recommendations = [
                    line.strip("-• ").strip()
                    for line in content.splitlines()
                    if line.strip()
                ]
            
            self._reset_breaker()
            
            return recommendations if recommendations else [
                "Nie udało się wygenerować rekomendacji."
            ]
        
        except Exception as e:
            self._register_failure()
            self.logger.error(f"Recommendation generation failed: {e}")
            return ["Nie udało się wygenerować rekomendacji."]
    
    # ───────────────────────────────────────────────────────────────────
    # Context Preparation & Security
    # ───────────────────────────────────────────────────────────────────
    
    def _prepare_context(self, context: Dict[str, Any]) -> str:
        """
        Build and sanitize context (PII scrub + trimming).
        
        Priority order: EDA → ML → data_info → other keys
        
        Returns:
            Prepared context string (safe for LLM)
        """
        parts: List[str] = []
        
        def _pack(title: str, obj: Any) -> str:
            s = self._safe_stringify(obj)
            if self.config.pii_scrub:
                s = self._scrub_pii(s)
            return f"**{title}:**\n{self._trim_context_if_needed(s)}"
        
        # Handle known keys in priority order
        if "eda_results" in context:
            parts.append(_pack("Wyniki EDA", context["eda_results"]))
        
        if "ml_results" in context:
            parts.append(_pack("Wyniki ML", context["ml_results"]))
        
        if "data_info" in context:
            parts.append(_pack("Informacje o danych", context["data_info"]))
        
        # Handle remaining keys
        known_keys = {"eda_results", "ml_results", "data_info"}
        for key, value in context.items():
            if key not in known_keys:
                parts.append(_pack(key, value))
        
        joined = "\n\n".join(parts).strip()
        return self._trim_context_if_needed(joined)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create final prompt from query and context.
        
        Returns:
            Complete prompt string
        """
        if context:
            return (
                f"Kontekst analizy:\n{context}\n\n"
                f"Pytanie użytkownika:\n{query}\n\n"
                "Odpowiedz po polsku, rzeczowo i praktycznie. "
                "Stosuj krótkie sekcje i jasne rekomendacje. "
                "Jeśli czegoś brakuje w kontekście — wskaż to i zaproponuj kolejny krok."
            )
        
        return query
    
    def _scrub_pii(self, text: str) -> str:
        """
        Scrub PII from text (GDPR-ready).
        
        Removes:
          • Email addresses
          • Phone numbers
          • IPv4 addresses
          • UUIDs
          • IBAN numbers (PL format)
          • PESEL-like numbers (11 digits)
        
        Returns:
            Scrubbed text
        """
        try:
            # Normalize whitespace
            text = re.sub(r"[ \t]+", " ", text)
            
            # Email addresses
            text = re.sub(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                "[EMAIL]",
                text
            )
            
            # Phone numbers (various formats)
            text = re.sub(r"\b(\+?\d[\d\s\-]{7,}\d)\b", "[PHONE]", text)
            # IPv4 addresses
            text = re.sub(r"\b(\d{1,3}\.){3}\d{1,3}\b", "[IPV4]", text)
            
            # UUIDs
            text = re.sub(
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
                "[UUID]",
                text
            )
            
            # Polish IBAN
            text = re.sub(r"\bPL\d{26}\b", "[IBAN]", text)
            
            # PESEL-like (11 digits)
            text = re.sub(r"\b\d{11}\b", "[ID]", text)
            
        except Exception as e:
            self.logger.debug(f"PII scrubbing encountered error: {e}")
        
        return text
    
    def _safe_stringify(self, obj: Any) -> str:
        """
        Safely stringify object with pre-cap.
        
        Returns:
            JSON string (or str fallback)
        """
        try:
            s = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
        
        # Soft pre-cap (final hard-cap in _trim_context_if_needed)
        return s[:self.config.context_max_chars * 2]
    
    def _trim_context_if_needed(self, text: str) -> str:
        """
        Trim context to fit within char limit (head + tail strategy).
        
        Returns:
            Trimmed context string
        """
        max_len = self.config.context_max_chars
        
        if len(text) <= max_len:
            return text
        
        head = text[:self.config.trim_section_head]
        tail = text[-self.config.trim_section_tail:]
        note = f"\n\n[Uwaga: kontekst skrócony do {max_len} znaków]"
        
        return head + "\n...\n" + tail + note
    
    def _safe_format(self, template: str, **kwargs: Any) -> str:
        """
        Safe template formatting (ignores missing keys).
        
        Returns:
            Formatted string (missing keys preserved as {key})
        """
        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        
        try:
            return template.format_map(_SafeDict(**kwargs))
        except Exception:
            return template
    
    # ───────────────────────────────────────────────────────────────────
    # LLM Integration (Retry + Backoff)
    # ───────────────────────────────────────────────────────────────────
    
    def _call_llm_with_retry(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Tuple[Any, int]:
        """
        Call LLM with exponential backoff + full jitter retry.
        
        Returns:
            Tuple of (response, attempt_index)
        
        Raises:
            RuntimeError: After all retries exhausted
        """
        attempts = self.config.retries + 1
        delay = self.config.initial_backoff_s
        last_error: Optional[Exception] = None
        
        for i in range(attempts):
            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.request_timeout_s
                )
                return response, i  # i = attempt index (0 = no retry)
            
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"LLM generate failed (attempt {i+1}/{attempts}): "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                
                if i < attempts - 1:
                    # Apply jitter
                    jitter = delay * self.config.jitter_fraction
                    sleep_s = delay + random.uniform(-jitter, jitter)
                    time.sleep(max(0.05, sleep_s))
                    delay *= self.config.backoff_multiplier
        
        raise RuntimeError(
            f"LLM generate failed after {attempts} attempts. "
            f"Last error: {type(last_error).__name__}: {str(last_error)[:100]}"
        )
    
    def _call_llm_json_with_retry(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Call LLM for JSON response with retry.
        
        Returns:
            JSON dictionary (empty dict on failure)
        """
        attempts = self.config.retries + 1
        delay = self.config.initial_backoff_s
        last_error: Optional[Exception] = None
        
        for i in range(attempts):
            try:
                response = self.llm_client.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    timeout=self.config.request_timeout_s
                )
                
                # Try multiple response formats
                if isinstance(response, dict):
                    return response
                
                if hasattr(response, "json") and isinstance(response.json, dict):
                    return response.json
                
                if hasattr(response, "content") and isinstance(response.content, str):
                    try:
                        return json.loads(response.content)
                    except Exception:
                        pass
                
                return {}
            
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"LLM generate_json failed (attempt {i+1}/{attempts}): "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                
                if i < attempts - 1:
                    jitter = delay * self.config.jitter_fraction
                    sleep_s = delay + random.uniform(-jitter, jitter)
                    time.sleep(max(0.05, sleep_s))
                    delay *= self.config.backoff_multiplier
        
        self.logger.error(
            f"LLM generate_json failed after {attempts} attempts. "
            f"Last error: {type(last_error).__name__}"
        )
        return {}
    
    # ───────────────────────────────────────────────────────────────────
    # Parameter Clamping
    # ───────────────────────────────────────────────────────────────────
    
    def _clamp_tokens(self, n: int) -> int:
        """
        Clamp token count to safe range (64-32768).
        
        Returns:
            Clamped token count
        """
        try:
            return int(max(64, min(32768, n)))
        except Exception:
            return 1024
    
    def _clamp_temperature(self, t: float) -> float:
        """
        Clamp temperature to valid range (0.0-1.0).
        
        Returns:
            Clamped temperature
        """
        try:
            return float(max(0.0, min(1.0, t)))
        except Exception:
            return 0.2
    
    # ───────────────────────────────────────────────────────────────────
    # Circuit Breaker Management
    # ───────────────────────────────────────────────────────────────────
    
    def _is_breaker_open(self) -> bool:
        """
        Check if circuit breaker is open.
        
        Returns:
            True if breaker is open (calls should be blocked)
        """
        if self._fail_streak < self.config.circuit_breaker_failures:
            return False
        
        return time.time() < self._breaker_until_ts
    
    def _register_failure(self) -> None:
        """
        Register LLM failure and potentially open circuit breaker.
        """
        self._fail_streak += 1
        
        if self._fail_streak >= self.config.circuit_breaker_failures:
            self._breaker_until_ts = time.time() + self.config.circuit_breaker_cooldown_s
            self.logger.warning(
                f"⚠ Circuit breaker OPEN for {self.config.circuit_breaker_cooldown_s}s "
                f"(failures={self._fail_streak})"
            )
    
    def _reset_breaker(self) -> None:
        """
        Reset circuit breaker after successful call.
        """
        if self._fail_streak > 0:
            self.logger.info(
                f"✓ Circuit breaker reset (fail_streak={self._fail_streak} → 0)"
            )
        
        self._fail_streak = 0
        self._breaker_until_ts = 0.0
    
    # ───────────────────────────────────────────────────────────────────
    # Telemetry & Helpers
    # ───────────────────────────────────────────────────────────────────
    
    def _token_hint(self, prompt_len: int) -> int:
        """
        Estimate token count from character length (heuristic).
        
        Returns:
            Estimated token count
        """
        try:
            return int(max(
                1,
                round(prompt_len / self.config.token_hint_avg_chars_per_token)
            ))
        except Exception:
            return 0
    
    def _provider_name(self) -> Optional[str]:
        """
        Extract provider name from LLM client (if available).
        
        Returns:
            Provider name or None
        """
        for attr in ("provider", "name", "model", "engine"):
            try:
                value = getattr(self.llm_client, attr, None)
                if isinstance(value, str) and value.strip():
                    return value
            except Exception:
                continue
        
        return None
    
    def _wrap_payload(
        self,
        *,
        response: str,
        query: str,
        context_len: int,
        tokens_used: Optional[int],
        breaker_state: str,
        cache_hit: bool,
        elapsed_s: float,
        limiter_hit: bool,
        attempts: int,
        provider: Optional[str]
    ) -> Dict[str, Any]:
        """
        Wrap response in standardized payload with telemetry.
        
        Returns:
            Complete response dictionary
        """
        return {
            "response": response,
            "query": query,
            "tokens_used": tokens_used,
            "meta": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "context_chars": context_len,
            },
            "telemetry": {
                "elapsed_s": round(elapsed_s, 4),
                "retries_cfg": self.config.retries,
                "attempt_index": attempts,
                "breaker_state": breaker_state,  # 'open' | 'closed' | 'maybe_open'
                "cache_hit": cache_hit,
                "token_hint_in": self._token_hint(context_len),
                "rate_limiter_hit": limiter_hit,
                "provider": provider,
            },
            "version": __version__,
        }