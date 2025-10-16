# agents/mentor/orchestrator.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — AI Mentor Orchestrator            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade AI-powered mentoring with production safeguards:         ║
║    ✓ Multi-modal responses (Q&A, EDA explanations, ML insights)           ║
║    ✓ PII scrubbing (email, phone, IP, PESEL-like patterns)                ║
║    ✓ Context management (head/tail trimming with size limits)             ║
║    ✓ Retry mechanism (exponential backoff + full jitter)                  ║
║    ✓ Circuit breaker (failure threshold + cooldown)                       ║
║    ✓ TTL cache (LRU eviction + thread-safe)                               ║
║    ✓ JSON-first responses with text fallback                              ║
║    ✓ Comprehensive telemetry (latency, retries, tokens, cache)            ║
║    ✓ Defensive validation (input sanitization)                            ║
║    ✓ Dependency injection (custom LLM client support)                     ║
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
        "token_hints": {
            "prompt_tokens_est": int,
            "response_max_tokens": int,
        },
    },
    "telemetry": {
        "elapsed_s": float,
        "retries_used": int,
        "tokens_used": int | None,
        "breaker_state": "open" | "closed" | "maybe_open",
        "cache_hit": bool,
    },
    "version": "5.0-kosmos-enterprise",
}
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
from functools import wraps

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies
try:
    from core.base_agent import BaseAgent, AgentResult
except ImportError:
    # Fallback for testing
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data = None
            self.errors = []
            self.warnings = []
        
        def add_error(self, msg: str):
            self.errors.append(msg)
        
        def add_warning(self, msg: str):
            self.warnings.append(msg)

try:
    from core.llm_client import get_llm_client
except ImportError:
    get_llm_client = None  # type: ignore
    logger.warning("⚠ core.llm_client unavailable — running in degraded mode")

try:
    from agents.mentor.prompt_templates import (
        MENTOR_SYSTEM_PROMPT,
        EDA_EXPLANATION_TEMPLATE,
        ML_RESULTS_TEMPLATE,
        RECOMMENDATION_TEMPLATE,
    )
except ImportError:
    MENTOR_SYSTEM_PROMPT = None  # type: ignore
    EDA_EXPLANATION_TEMPLATE = None  # type: ignore
    ML_RESULTS_TEMPLATE = None  # type: ignore
    RECOMMENDATION_TEMPLATE = None  # type: ignore
    logger.warning("⚠ prompt templates unavailable")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _now_s() -> float:
    """Get current timestamp in seconds."""
    return time.time()


def _hash_key(*parts: str) -> str:
    """Generate SHA256 hash from multiple string parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x1f")  # Unit separator
    return h.hexdigest()


def _timeit(operation_name: str):
    """Decorator for operation timing with intelligent logging."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(f"⏱ {operation_name}: {elapsed_ms:.2f}ms")
        return wrapper
    return decorator


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
        self.ttl_s = int(ttl_s)
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (thread-safe).
        
        Returns:
            Cached value or None if expired/missing
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            
            timestamp, value = entry
            
            # Check TTL
            if _now_s() - timestamp > self.ttl_s:
                self._store.pop(key, None)
                return None
            
            return value
    
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
            
            self._store[key] = (_now_s(), value)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MentorConfig:
    """Enterprise configuration for AI Mentor Orchestrator."""
    
    # LLM parameters
    temperature: float = 0.45
    max_tokens: int = 2000
    json_max_tokens: int = 1400
    request_timeout_s: Optional[float] = None
    
    # Retry & backoff
    retries: int = 2
    initial_backoff_s: float = 0.6
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.25  # 25% jitter around base delay
    
    # Context & security
    pii_scrub: bool = True
    context_max_chars: int = 60_000  # Hard cap after scrubbing
    trim_section_head: int = 10_000
    trim_section_tail: int = 3_000
    prefer_json: bool = True
    
    # Circuit breaker
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: float = 30.0
    
    # Cache
    cache_enabled: bool = True
    cache_ttl_s: int = 120
    cache_maxsize: int = 256
    
    # Logging
    audit_log_safe_context: bool = True


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
      5. PII scrubbing for data privacy
      6. Context trimming for token efficiency
      7. Retry with exponential backoff + jitter
      8. Circuit breaker for cascading failure prevention
      9. TTL cache for identical queries
      10. Comprehensive telemetry tracking
    
    Features:
      • 3 response modes (Q&A, EDA, ML, recommendations)
      • PII scrubber (email, phone, IP, PESEL)
      • Circuit breaker (3 failures → 30s cooldown)
      • TTL cache (120s, 256 entries)
      • Exponential backoff (0.6s → 1.2s → 2.4s)
      • Full jitter (±25% randomization)
      • Token estimation (4 chars/token heuristic)
    """
    
    def __init__(
        self,
        config: Optional[MentorConfig] = None,
        llm_client: Optional[Any] = None
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
        self._log = logger.bind(agent="MentorOrchestrator")
        warnings.filterwarnings("ignore")
        
        # Initialize LLM client
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = self._safe_get_llm_client()
        
        # Circuit breaker state
        self._fail_streak = 0
        self._breaker_until_ts: float = 0.0
        self._breaker_lock = threading.Lock()
        
        # Cache initialization
        self._cache = _TTLCache(
            maxsize=self.config.cache_maxsize,
            ttl_s=self.config.cache_ttl_s
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
    
    @_timeit("MentorOrchestrator.execute")
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
            # Input validation
            if not isinstance(query, str) or not query.strip():
                msg = "Invalid query: expected non-empty string"
                result.add_error(msg)
                self._log.error(msg)
                return result
            
            if context is not None and not isinstance(context, dict):
                msg = f"Invalid context: expected dict, got {type(context).__name__}"
                result.add_error(msg)
                self._log.error(msg)
                return result
            
            # Check circuit breaker
            if self._is_breaker_open():
                cooldown = max(0.0, self._breaker_until_ts - _now_s())
                warning = (
                    f"Circuit breaker is open. Cooldown: {cooldown:.1f}s. "
                    "Too many LLM failures detected."
                )
                self._log.warning(warning)
                result.add_warning(warning)
                result.data = {
                    "response": (
                        "Tymczasowa przerwa w wywołaniach LLM (zabezpieczenie przed przeciążeniem). "
                        "Spróbuj ponownie za chwilę."
                    ),
                    "query": query,
                    "tokens_used": None,
                    "meta": self._meta_info(0, cfg.max_tokens),
                    "telemetry": self._telemetry(
                        t0,
                        retries_used=0,
                        tokens_used=None,
                        breaker_state="open",
                        cache_hit=False
                    ),
                    "version": "5.0-kosmos-enterprise",
                }
                return result
            
            # Prepare context (scrub + trim)
            context_str = self._prepare_context(context or {})
            full_prompt = self._create_prompt(query, context_str)
            
            # Check cache
            cache_key = None
            cache_hit = False
            
            if cfg.cache_enabled:
                cache_key = _hash_key(
                    "response",
                    full_prompt,
                    MENTOR_SYSTEM_PROMPT or ""
                )
                
                cached = self._cache.get(cache_key)
                if cached is not None:
                    cache_hit = True
                    content, tokens_used = cached
                    self._reset_breaker()
                    
                    self._log.info(f"✓ Cache hit for query: {query[:50]}...")
                    
                    result.data = {
                        "response": content,
                        "query": query,
                        "tokens_used": tokens_used,
                        "meta": self._meta_info(len(context_str), cfg.max_tokens),
                        "telemetry": self._telemetry(
                            t0,
                            retries_used=0,
                            tokens_used=tokens_used,
                            breaker_state="closed",
                            cache_hit=True
                        ),
                        "version": "5.0-kosmos-enterprise",
                    }
                    return result
            
            # LLM call with retry
            response = self._call_llm_with_retry(
                prompt=full_prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens
            )
            
            content = self._extract_response_content(response)
            tokens_used = self._extract_tokens_used(response)
            
            # Update cache
            if cfg.cache_enabled and cache_key:
                try:
                    self._cache.set(cache_key, (content, tokens_used))
                except Exception as e:
                    self._log.debug(f"Cache set failed: {e}")
            
            self._reset_breaker()
            
            result.data = {
                "response": content,
                "query": query,
                "tokens_used": tokens_used,
                "meta": self._meta_info(len(context_str), cfg.max_tokens),
                "telemetry": self._telemetry(
                    t0,
                    retries_used=0,
                    tokens_used=tokens_used,
                    breaker_state="closed",
                    cache_hit=cache_hit
                ),
                "version": "5.0-kosmos-enterprise",
            }
            
            self._log.success(
                f"✓ AI Mentor response generated | "
                f"tokens={tokens_used} | "
                f"cached={cache_hit}"
            )
        
        except Exception as e:
            self._register_failure()
            msg = f"AI Mentor failed: {type(e).__name__}: {str(e)}"
            result.add_error(msg)
            self._log.exception(f"❌ {msg}")
            
            result.data = {
                "response": "Nie udało się wygenerować odpowiedzi. Spróbuj ponownie.",
                "query": query if isinstance(query, str) else "",
                "tokens_used": None,
                "meta": self._meta_info(0, cfg.max_tokens),
                "telemetry": self._telemetry(
                    t0,
                    retries_used=self._fail_streak,
                    tokens_used=None,
                    breaker_state="maybe_open",
                    cache_hit=False
                ),
                "version": "5.0-kosmos-enterprise",
            }
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # EDA Explanation Mode
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("explain_eda")
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
            
            if EDA_EXPLANATION_TEMPLATE is None:
                raise RuntimeError("EDA template unavailable")
            
            eda_str = self._prepare_context({"eda_results": eda_results})
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
            
            self._reset_breaker()
            content = self._extract_response_content(response)
            
            return content if content else "—"
        
        except Exception as e:
            self._register_failure()
            self._log.error(f"EDA explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia EDA."
    
    # ───────────────────────────────────────────────────────────────────
    # ML Explanation Mode
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("explain_ml")
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
            
            if ML_RESULTS_TEMPLATE is None:
                raise RuntimeError("ML template unavailable")
            
            ml_str = self._prepare_context({"ml_results": ml_results})
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
            
            self._reset_breaker()
            content = self._extract_response_content(response)
            
            return content if content else "—"
        
        except Exception as e:
            self._register_failure()
            self._log.error(f"ML explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia ML."
    
    # ───────────────────────────────────────────────────────────────────
    # Recommendation Mode (JSON-First)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("generate_recommendations")
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
            if RECOMMENDATION_TEMPLATE is None:
                raise RuntimeError("Recommendation template unavailable")
            
            context = {
                "eda": eda_results or {},
                "ml": ml_results or {},
                "quality": data_quality or {}
            }
            
            context_str = self._prepare_context(context)
            prompt = RECOMMENDATION_TEMPLATE.format(context=context_str)
            
            recommendations: List[str] = []
            
            # Try JSON-first
            if self.config.prefer_json:
                resp_json = self._call_llm_json_with_retry(
                    prompt=prompt,
                    system_prompt=MENTOR_SYSTEM_PROMPT,
                    max_tokens=self.config.json_max_tokens
                )
                
                if isinstance(resp_json, dict):
                    # Try multiple possible keys
                    for key in ("recommendations", "data", "items"):
                        if isinstance(resp_json.get(key), list):
                            recommendations = [
                                str(x).strip()
                                for x in resp_json[key]
                                if str(x).strip()
                            ]
                            break
            
            # Fallback to text parsing
            if not recommendations:
                self._log.warning("JSON response empty/invalid, falling back to text")
                
                txt_response = self._call_llm_with_retry(
                    prompt=prompt,
                    system_prompt=MENTOR_SYSTEM_PROMPT,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                content = self._extract_response_content(txt_response)
                
                # Parse bullet points / numbered lists
                raw_lines = [ln.strip() for ln in content.splitlines()]
                recommendations = [
                    ln.lstrip("-•123456789. ").strip()
                    for ln in raw_lines
                    if ln and not ln.lower().startswith("recommendation")
                ]
            
            self._reset_breaker()
            
            return recommendations if recommendations else [
                "Nie udało się wygenerować rekomendacji."
            ]
        
        except Exception as e:
            self._register_failure()
            self._log.error(f"Recommendation generation failed: {e}")
            return ["Nie udało się wygenerować rekomendacji."]
    
    # ───────────────────────────────────────────────────────────────────
    # Context Preparation & Security
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("prepare_context")
    def _prepare_context(self, context: Dict[str, Any]) -> str:
        """
        Prepare context string with PII scrubbing and trimming.
        
        Returns:
            Prepared context string (safe for LLM)
        """
        parts: List[str] = []
        
        def _pack(title: str, obj: Any) -> str:
            s = self._safe_stringify(obj)
            if self.config.pii_scrub:
                s = self._scrub_pii(s)
            return f"**{title}:**\n{s}"
        
        # Handle known keys
        if "eda_results" in context:
            parts.append(_pack("Wyniki EDA", context["eda_results"]))
        
        if "ml_results" in context:
            parts.append(_pack("Wyniki ML", context["ml_results"]))
        
        if "data_info" in context:
            parts.append(_pack("Informacje o danych", context["data_info"]))
        
        # Handle other keys
        known_keys = {"eda_results", "ml_results", "data_info"}
        for key, value in context.items():
            if key not in known_keys:
                parts.append(_pack(key, value))
        
        joined = "\n\n".join(parts).strip()
        prepared = self._trim_context_if_needed(joined)
        
        if self.config.audit_log_safe_context:
            self._log.debug(f"Context prepared: {len(prepared)} chars")
        
        return prepared
    
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
                "Odpowiedz po polsku, konkretnie i profesjonalnie. "
                "Używaj krótkich akapitów i list punktowanych tam, gdzie to sensowne. "
                "Jeśli informacji brakuje — zaznacz to i zaproponuj kolejny krok."
            )
        
        return query
    
    def _scrub_pii(self, text: str) -> str:
        """
        Scrub PII from text (email, phone, IP, PESEL-like).
        
        Returns:
            Scrubbed text
        """
        try:
            import re
            
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
            
            # PESEL-like (11 digits)
            text = re.sub(r"\b\d{11}\b", "[ID]", text)
            
            return text
        
        except Exception as e:
            self._log.warning(f"PII scrubbing failed: {e}")
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
        max_pre = self.config.context_max_chars * 2
        return s[:max_pre]
    
    def _trim_context_if_needed(self, context: str) -> str:
        """
        Trim context to fit within char limit (head + tail strategy).
        
        Returns:
            Trimmed context string
        """
        max_len = self.config.context_max_chars
        
        if len(context) <= max_len:
            return context
        
        head = context[:self.config.trim_section_head]
        tail = context[-self.config.trim_section_tail:]
        note = f"\n\n[Uwaga: kontekst skrócony do {max_len} znaków]"
        
        return head + "\n...\n" + tail + note
    
    # ───────────────────────────────────────────────────────────────────
    # LLM Integration (Retry + Backoff)
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("llm_call_with_retry")
    def _call_llm_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Any:
        """
        Call LLM with exponential backoff + full jitter retry.
        
        Returns:
            LLM response object
        
        Raises:
            RuntimeError: After all retries exhausted
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client not initialized")
        
        cfg = self.config
        attempts = cfg.retries + 1
        delay = float(cfg.initial_backoff_s)
        last_error: Optional[Exception] = None
        
        for attempt in range(attempts):
            try:
                return self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    timeout=cfg.request_timeout_s
                )
            
            except Exception as e:
                last_error = e
                self._log.warning(
                    f"LLM generate failed (attempt {attempt+1}/{attempts}): {type(e).name}: {str(e)[:100]}")
            if attempt < attempts - 1:
                sleep_s = self._jittered_delay(delay, cfg.jitter_fraction)
                time.sleep(sleep_s)
                delay *= cfg.backoff_multiplier
    
    raise RuntimeError(
        f"LLM generate failed after {attempts} attempts. "
        f"Last error: {type(last_error).__name__}: {str(last_error)[:100]}"
    )

@_timeit("llm_json_with_retry")
def _call_llm_json_with_retry(
    self,
    prompt: str,
    system_prompt: Optional[str],
    max_tokens: int
) -> Dict[str, Any]:
    """
    Call LLM for JSON response with retry.
    
    Returns:
        JSON dictionary (empty dict on failure)
    """
    if self.llm_client is None:
        return {}
    
    cfg = self.config
    attempts = cfg.retries + 1
    delay = float(cfg.initial_backoff_s)
    last_error: Optional[Exception] = None
    
    for attempt in range(attempts):
        try:
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=int(max_tokens),
                timeout=cfg.request_timeout_s
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
            self._log.warning(
                f"LLM generate_json failed (attempt {attempt+1}/{attempts}): "
                f"{type(e).__name__}: {str(e)[:100]}"
            )
            
            if attempt < attempts - 1:
                sleep_s = self._jittered_delay(delay, cfg.jitter_fraction)
                time.sleep(sleep_s)
                delay *= cfg.backoff_multiplier
    
    self._log.error(
        f"LLM generate_json failed after {attempts} attempts. "
        f"Last error: {type(last_error).__name__}"
    )
    return {}

# ───────────────────────────────────────────────────────────────────
# Circuit Breaker Management
# ───────────────────────────────────────────────────────────────────

def _is_breaker_open(self) -> bool:
    """
    Check if circuit breaker is open (thread-safe).
    
    Returns:
        True if breaker is open (calls should be blocked)
    """
    with self._breaker_lock:
        if self._fail_streak < self.config.circuit_breaker_failures:
            return False
        
        return _now_s() < self._breaker_until_ts

def _register_failure(self) -> None:
    """
    Register LLM failure and potentially open circuit breaker (thread-safe).
    """
    with self._breaker_lock:
        self._fail_streak += 1
        
        if self._fail_streak >= self.config.circuit_breaker_failures:
            self._breaker_until_ts = _now_s() + self.config.circuit_breaker_cooldown_s
            self._log.warning(
                f"⚠ Circuit breaker OPEN for {self.config.circuit_breaker_cooldown_s}s "
                f"(failures={self._fail_streak})"
            )

def _reset_breaker(self) -> None:
    """
    Reset circuit breaker after successful call (thread-safe).
    """
    with self._breaker_lock:
        if self._fail_streak > 0:
            self._log.info(
                f"✓ Circuit breaker reset (fail_streak={self._fail_streak} → 0)"
            )
        
        self._fail_streak = 0
        self._breaker_until_ts = 0.0

# ───────────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────────

def _safe_get_llm_client(self) -> Any:
    """
    Get LLM client with graceful degradation.
    
    Returns:
        LLM client or null client
    """
    if get_llm_client is None:
        self._log.warning("⚠ LLM client module unavailable — degraded mode")
        return self._create_null_client()
    
    try:
        return get_llm_client()
    
    except Exception as e:
        self._log.warning(
            f"⚠ LLM client unavailable — degraded mode. Reason: {e}"
        )
        return self._create_null_client()

@staticmethod
def _create_null_client() -> Any:
    """
    Create null LLM client for degraded mode.
    
    Returns:
        Object that raises on any method call
    """
    class _NullClient:
        def generate(self, *args, **kwargs):
            raise RuntimeError("LLM client not available")
        
        def generate_json(self, *args, **kwargs):
            raise RuntimeError("LLM client not available")
    
    return _NullClient()

@staticmethod
def _jittered_delay(delay_s: float, jitter_fraction: float) -> float:
    """
    Apply full jitter to delay (±fraction).
    
    Returns:
        Jittered delay in seconds
    """
    jitter = max(0.0, float(jitter_fraction)) * delay_s
    jittered = delay_s + random.uniform(-jitter, jitter)
    return max(0.05, jittered)  # Minimum 50ms

@staticmethod
def _extract_response_content(response: Any) -> str:
    """
    Extract text content from LLM response.
    
    Returns:
        Response text
    """
    if response is None:
        return ""
    
    if isinstance(response, str):
        return response
    
    # Try common attributes
    for attr in ("content", "text", "output", "response"):
        if hasattr(response, attr):
            val = getattr(response, attr)
            if isinstance(val, str):
                return val
    
    return str(response)

@staticmethod
def _extract_tokens_used(response: Any) -> Optional[int]:
    """
    Extract token count from LLM response.
    
    Returns:
        Token count or None
    """
    if response is None:
        return None
    
    # Try common attributes
    for attr in ("tokens_used", "usage", "token_count"):
        if hasattr(response, attr):
            val = getattr(response, attr)
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, dict) and "total_tokens" in val:
                return int(val["total_tokens"])
    
    return None

def _meta_info(self, context_chars: int, max_tokens: int) -> Dict[str, Any]:
    """
    Generate metadata dictionary.
    
    Returns:
        Metadata dict
    """
    return {
        "temperature": float(self.config.temperature),
        "max_tokens": int(max_tokens),
        "context_chars": int(context_chars),
        "token_hints": self._token_hints(context_chars, max_tokens),
    }

@staticmethod
def _token_hints(context_chars: int, response_max_tokens: int) -> Dict[str, int]:
    """
    Estimate token counts (4 chars/token heuristic).
    
    Returns:
        Token estimation dict
    """
    try:
        # Rough heuristic: 4 chars per token (GPT-style)
        approx_prompt_tokens = max(1, int(context_chars / 4))
    except Exception:
        approx_prompt_tokens = 1
    
    return {
        "prompt_tokens_est": approx_prompt_tokens,
        "response_max_tokens": int(response_max_tokens),
    }

def _telemetry(
    self,
    t0: float,
    *,
    retries_used: int,
    tokens_used: Optional[int],
    breaker_state: str,
    cache_hit: bool
) -> Dict[str, Any]:
    """
    Generate telemetry dictionary.
    
    Returns:
        Telemetry dict
    """
    return {
        "elapsed_s": round(time.perf_counter() - t0, 4),
        "retries_used": int(retries_used),
        "tokens_used": tokens_used,
        "breaker_state": str(breaker_state),
        "cache_hit": bool(cache_hit),
    }