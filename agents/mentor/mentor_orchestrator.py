# === OPIS MODUŁU ===
"""
DataGenius PRO+++++++++++++ — AI Mentor Orchestrator (Enterprise / KOSMOS)
Wyjaśnienia EDA/ML, rekomendacje i Q&A z LLM w trybie enterprise:
- defensywna walidacja wejść
- PII scrub (email/telefon/IP/PESEL-like) + normalizacja whitespace
- limiter kontekstu (head/tail, soft i hard cap)
- retry z exponential backoff + pełny jitter
- circuit breaker (licznik awarii + cooldown)
- JSON-first z bezpiecznym fallbackiem do tekstu
- telemetry: czasy, próby, token hints, breaker/caching info
- cache (TTL+maxsize) dla identycznych zapytań/kontekstów
- dependency injection klienta LLM

Kontrakt (AgentResult.data):
{
  "response": str,
  "query": str,
  "tokens_used": int | None,
  "meta": {
    "temperature": float,
    "max_tokens": int,
    "context_chars": int,
    "token_hints": {"prompt_tokens_est": int, "response_max_tokens": int}
  },
  "telemetry": {
    "elapsed_s": float,
    "retries_used": int,
    "tokens_used": int | None,
    "breaker_state": "open" | "closed" | "maybe_open",
    "cache_hit": bool
  },
  "version": "4.2-enterprise"
}
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

import pandas as pd  # noqa: F401  (spójność z projektem)
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from agents.mentor.prompt_templates import (
    MENTOR_SYSTEM_PROMPT,
    EDA_EXPLANATION_TEMPLATE,
    ML_RESULTS_TEMPLATE,
    RECOMMENDATION_TEMPLATE,
)


# === UTILS CZASU/HASH ===
def _now_s() -> float:
    return time.time()


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


# === CACHE Z TTL ===
class _TTLCache:
    """Prosty, thread-safe cache z TTL + maxsize (drop najstarszego)."""
    def __init__(self, maxsize: int, ttl_s: int):
        self.maxsize = int(maxsize)
        self.ttl_s = int(ttl_s)
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            row = self._store.get(key)
            if not row:
                return None
            ts, val = row
            if _now_s() - ts > self.ttl_s:
                self._store.pop(key, None)
                return None
            return val

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.maxsize:
                # usuń najstarszy wpis
                oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                self._store.pop(oldest_key, None)
            self._store[key] = (_now_s(), value)


# === KONFIG ===
@dataclass(frozen=True)
class MentorConfig:
    # LLM
    temperature: float = 0.45
    max_tokens: int = 2000
    json_max_tokens: int = 1400
    request_timeout_s: Optional[float] = None

    # retry/backoff
    retries: int = 2
    initial_backoff_s: float = 0.6
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.25  # 25% jitteru wokół bazowego opóźnienia

    # kontekst & scrub
    pii_scrub: bool = True
    context_max_chars: int = 60_000  # hard cap po scrubie
    trim_section_head: int = 10_000
    trim_section_tail: int = 3_000
    prefer_json: bool = True

    # breaker
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: float = 30.0

    # cache
    cache_enabled: bool = True
    cache_ttl_s: int = 120
    cache_maxsize: int = 256

    # logowanie
    audit_log_safe_context: bool = True


# === GŁÓWNY ORKIESTRATOR ===
class MentorOrchestrator(BaseAgent):
    """
    AI Mentor — wyjaśnienia i rekomendacje oparte o wyniki EDA/ML.
    Enterprise-grade: defensywnie, z telemetry, breakerem i cachem.
    """

    def __init__(self, config: Optional[MentorConfig] = None, llm_client: Any = None) -> None:
        super().__init__(name="MentorOrchestrator", description="AI Mentor for data science guidance")
        self.config = config or MentorConfig()
        self._log = logger.bind(agent=self.name)
        self.llm_client = llm_client or self._safe_get_llm_client()
        # circuit breaker
        self._fail_streak = 0
        self._breaker_until_ts: float = 0.0
        # cache
        self._cache = _TTLCache(self.config.cache_maxsize, self.config.cache_ttl_s)

    # === API: GŁÓWNA ODPOWIEDŹ ===
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        t0 = time.perf_counter()
        c = self.config

        try:
            # Walidacja
            if not isinstance(query, str) or not query.strip():
                msg = "MentorOrchestrator: 'query' must be a non-empty string."
                result.add_error(msg)
                self._log.error(msg)
                return result
            if context is not None and not isinstance(context, dict):
                msg = "MentorOrchestrator: 'context' must be a dict if provided."
                result.add_error(msg)
                self._log.error(msg)
                return result

            # Breaker
            if self._is_breaker_open():
                cooldown = max(0.0, self._breaker_until_ts - _now_s())
                warn = f"Circuit breaker is open. Cooldown {cooldown:.1f}s."
                self._log.warning(warn)
                result.add_warning(warn)
                result.data = {
                    "response": "Tymczasowa przerwa w wywołaniach LLM (zabezpieczenie). Spróbuj ponownie.",
                    "telemetry": self._telemetry(t0, retries_used=0, tokens_used=None, breaker_state="open", cache_hit=False),
                    "version": "4.2-enterprise",
                }
                return result

            # Kontekst -> string (scrub + trim)
            context_str = self._prepare_context(context or {})
            full_prompt = self._create_prompt(query, context_str)

            # Cache
            cache_key = None
            cache_hit = False
            if c.cache_enabled:
                cache_key = _hash_key("resp", full_prompt, MENTOR_SYSTEM_PROMPT or "")
                cached = self._cache.get(cache_key)
                if cached is not None:
                    cache_hit = True
                    content, tokens_used = cached
                    self._reset_breaker()
                    result.data = {
                        "response": content,
                        "query": query,
                        "tokens_used": tokens_used,
                        "meta": {
                            "temperature": c.temperature,
                            "max_tokens": c.max_tokens,
                            "context_chars": len(context_str),
                            "token_hints": self._token_hints(full_prompt, c.max_tokens),
                        },
                        "telemetry": self._telemetry(
                            t0, retries_used=0, tokens_used=tokens_used, breaker_state="closed", cache_hit=True
                        ),
                        "version": "4.2-enterprise",
                    }
                    return result

            # LLM call (retry)
            response = self._call_llm_with_retry(
                prompt=full_prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=c.temperature,
                max_tokens=c.max_tokens,
            )
            content = getattr(response, "content", None) or str(response)
            tokens_used = getattr(response, "tokens_used", None)

            # Cache set
            if c.cache_enabled and cache_key:
                try:
                    self._cache.set(cache_key, (content, tokens_used))
                except Exception:
                    pass

            self._reset_breaker()
            result.data = {
                "response": content,
                "query": query,
                "tokens_used": tokens_used,
                "meta": {
                    "temperature": c.temperature,
                    "max_tokens": c.max_tokens,
                    "context_chars": len(context_str),
                    "token_hints": self._token_hints(full_prompt, c.max_tokens),
                },
                "telemetry": self._telemetry(
                    t0, retries_used=0, tokens_used=tokens_used, breaker_state="closed", cache_hit=cache_hit
                ),
                "version": "4.2-enterprise",
            }
            self._log.success("AI Mentor response generated")
            return result

        except Exception as e:
            self._register_failure()
            result.add_error(f"AI Mentor failed: {e}")
            result.data = {
                "response": "Nie udało się wygenerować odpowiedzi (degradacja bezpieczna).",
                "telemetry": self._telemetry(
                    t0, retries_used=self._fail_streak, tokens_used=None, breaker_state="maybe_open", cache_hit=False
                ),
                "version": "4.2-enterprise",
            }
            self._log.exception(f"AI Mentor error: {e}")
            return result

    # === API: WYJAŚNIENIA EDA ===
    def explain_eda_results(
        self,
        eda_results: Dict[str, Any],
        user_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    ) -> str:
        try:
            if not isinstance(eda_results, dict):
                raise ValueError("'eda_results' must be a dict")
            eda_str = self._prepare_context({"eda_results": eda_results})
            prompt = EDA_EXPLANATION_TEMPLATE.format(eda_results=eda_str, user_level=user_level)
            resp = self._call_llm_with_retry(
                prompt=prompt, system_prompt=MENTOR_SYSTEM_PROMPT, temperature=self.config.temperature, max_tokens=self.config.max_tokens
            )
            self._reset_breaker()
            return getattr(resp, "content", "") or "—"
        except Exception as e:
            self._register_failure()
            self._log.error(f"EDA explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."

    # === API: WYJAŚNIENIA ML ===
    def explain_ml_results(
        self,
        ml_results: Dict[str, Any],
        user_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    ) -> str:
        try:
            if not isinstance(ml_results, dict):
                raise ValueError("'ml_results' must be a dict")
            ml_str = self._prepare_context({"ml_results": ml_results})
            prompt = ML_RESULTS_TEMPLATE.format(ml_results=ml_str, user_level=user_level)
            resp = self._call_llm_with_retry(
                prompt=prompt, system_prompt=MENTOR_SYSTEM_PROMPT, temperature=self.config.temperature, max_tokens=self.config.max_tokens
            )
            self._reset_breaker()
            return getattr(resp, "content", "") or "—"
        except Exception as e:
            self._register_failure()
            self._log.error(f"ML explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."

    # === API: REKOMENDACJE (JSON-first) ===
    def generate_recommendations(
        self,
        eda_results: Optional[Dict[str, Any]] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        data_quality: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        try:
            ctx = {"eda": eda_results or {}, "ml": ml_results or {}, "quality": data_quality or {}}
            ctx_str = self._prepare_context(ctx)
            prompt = RECOMMENDATION_TEMPLATE.format(context=ctx_str)

            recs: List[str] = []
            if self.config.prefer_json:
                resp_json = self._call_llm_json_with_retry(
                    prompt=prompt, system_prompt=MENTOR_SYSTEM_PROMPT, max_tokens=self.config.json_max_tokens
                )
                if isinstance(resp_json, dict):
                    if isinstance(resp_json.get("recommendations"), list):
                        recs = [str(x).strip() for x in resp_json["recommendations"] if str(x).strip()]
                    elif isinstance(resp_json.get("data"), list):
                        recs = [str(x).strip() for x in resp_json["data"] if str(x).strip()]

            if not recs:
                self._log.warning("LLM JSON empty/invalid. Fallback → text.")
                txt = self._call_llm_with_retry(
                    prompt=prompt, system_prompt=MENTOR_SYSTEM_PROMPT, temperature=self.config.temperature, max_tokens=self.config.max_tokens
                )
                content = getattr(txt, "content", "") or ""
                # rozbij po liniach / punktorach
                raw_lines = [ln.strip() for ln in content.splitlines()]
                recs = [ln.strip("-• ").strip() for ln in raw_lines if ln and not ln.lower().startswith("recommendation")]

            self._reset_breaker()
            return recs or ["Nie udało się wygenerować rekomendacji."]

        except Exception as e:
            self._register_failure()
            self._log.error(f"Recommendation generation failed: {e}")
            return ["Nie udało się wygenerować rekomendacji."]

    # === KONTEKST / PROMPT ===
    def _prepare_context(self, context: Dict[str, Any]) -> str:
        parts: List[str] = []

        def _pack(title: str, obj: Any) -> str:
            s = self._safe_stringify(obj)
            if self.config.pii_scrub:
                s = self._scrub_pii(s)
            return f"**{title}:**\n{s}"

        if "eda_results" in context:
            parts.append(_pack("Wyniki EDA", context["eda_results"]))
        if "ml_results" in context:
            parts.append(_pack("Wyniki ML", context["ml_results"]))
        if "data_info" in context:
            parts.append(_pack("Informacje o danych", context["data_info"]))

        # inne klucze (bez duplikacji)
        for k, v in context.items():
            if k not in {"eda_results", "ml_results", "data_info"}:
                parts.append(_pack(k, v))

        joined = "\n\n".join(parts).strip()
        prepared = self._trim_context_if_needed(joined)

        if self.config.audit_log_safe_context:
            self._log.debug(f"context_len={len(prepared)} chars")

        return prepared

    def _create_prompt(self, query: str, context: str) -> str:
        if context:
            return (
                f"Kontekst analizy:\n{context}\n\n"
                f"Pytanie użytkownika:\n{query}\n\n"
                "Odpowiedz po polsku, konkretnie, używając krótkich akapitów i list punktowanych tam, gdzie to sensowne. "
                "Jeśli informacji brakuje — zaznacz to i zaproponuj kolejny krok."
            )
        return query

    # === LLM CALLS (retry/backoff+jitter / JSON-first) ===
    def _call_llm_with_retry(self, prompt: str, system_prompt: Optional[str], temperature: float, max_tokens: int):
        if self.llm_client is None:
            raise RuntimeError("LLM client not initialized")

        c = self.config
        attempts = c.retries + 1
        delay = float(c.initial_backoff_s)
        last_err: Optional[Exception] = None

        for i in range(attempts):
            try:
                return self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    timeout=c.request_timeout_s,
                )
            except Exception as e:
                last_err = e
                self._log.warning(f"LLM generate failed (attempt {i+1}/{attempts}): {e}")
                if i < attempts - 1:
                    sleep_s = self._jittered(delay, c.jitter_fraction)
                    time.sleep(sleep_s)
                    delay *= c.backoff_multiplier

        raise RuntimeError(f"LLM generate failed after {attempts} attempts: {last_err}")

    def _call_llm_json_with_retry(self, prompt: str, system_prompt: Optional[str], max_tokens: int) -> Dict[str, Any]:
        if self.llm_client is None:
            return {}

        c = self.config
        attempts = c.retries + 1
        delay = float(c.initial_backoff_s)
        last_err: Optional[Exception] = None

        for i in range(attempts):
            try:
                resp = self.llm_client.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=int(max_tokens),
                    timeout=c.request_timeout_s,
                )
                if isinstance(resp, dict):
                    return resp
                if hasattr(resp, "json") and isinstance(resp.json, dict):
                    return resp.json
                if hasattr(resp, "content") and isinstance(resp.content, str):
                    try:
                        return json.loads(resp.content)
                    except Exception:
                        pass
                return {}
            except Exception as e:
                last_err = e
                self._log.warning(f"LLM generate_json failed (attempt {i+1}/{attempts}): {e}")
                if i < attempts - 1:
                    sleep_s = self._jittered(delay, c.jitter_fraction)
                    time.sleep(sleep_s)
                    delay *= c.backoff_multiplier

        self._log.error(f"LLM generate_json failed after {attempts} attempts: {last_err}")
        return {}

    # === BEZPIECZEŃSTWO / UTILS ===
    def _scrub_pii(self, text: str) -> str:
        """Lekki PII scrubber: e-mail, telefon, IPv4, PESEL-like, normalizacja whitespace."""
        try:
            import re
            # normalize whitespace
            text = re.sub(r"[ \t]+", " ", text)
            # emails
            text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
            # phone (+48 123-456-789, itp.) / dłuższe sekwencje cyfr/łączników/spacji
            text = re.sub(r"\b(\+?\d[\d\s\-]{7,}\d)\b", "[PHONE]", text)
            # IPv4
            text = re.sub(r"\b(\d{1,3}\.){3}\d{1,3}\b", "[IPV4]", text)
            # PESEL-like (11 cyfr)
            text = re.sub(r"\b\d{11}\b", "[ID]", text)
            return text
        except Exception:
            return text

    def _safe_stringify(self, obj: Any) -> str:
        """Bezpieczna serializacja służąca do promptu (z pre-capem)."""
        try:
            s = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
        # soft pre-cap (finalny hard-cap w _trim_context_if_needed)
        max_pre = self.config.context_max_chars * 2
        return s[: max_pre]

    def _trim_context_if_needed(self, context: str) -> str:
        max_len = self.config.context_max_chars
        if len(context) <= max_len:
            return context
        head = context[: self.config.trim_section_head]
        tail = context[-self.config.trim_section_tail :]
        note = f"\n\n[Uwaga: kontekst skrócony do {max_len} znaków]"
        return head + "\n...\n" + tail + note

    def _safe_get_llm_client(self):
        try:
            return get_llm_client()
        except Exception as e:
            logger.warning(f"LLM client unavailable; Mentor runs in degraded mode. Reason: {e}")

            class _NullClient:
                def generate(self, *a, **k):
                    raise RuntimeError("LLM client not available")

                def generate_json(self, *a, **k):
                    raise RuntimeError("LLM client not available")

            return _NullClient()

    # === BREAKER / TELEMETRY / HINTY ===
    def _is_breaker_open(self) -> bool:
        if self._fail_streak < self.config.circuit_breaker_failures:
            return False
        return _now_s() < self._breaker_until_ts

    def _register_failure(self) -> None:
        self._fail_streak += 1
        if self._fail_streak >= self.config.circuit_breaker_failures:
            self._breaker_until_ts = _now_s() + self.config.circuit_breaker_cooldown_s
            self._log.warning(
                f"Circuit breaker OPEN for {self.config.circuit_breaker_cooldown_s}s (failures={self._fail_streak})."
            )

    def _reset_breaker(self) -> None:
        if self._fail_streak > 0:
            self._log.info(f"Circuit breaker reset (fail_streak={self._fail_streak} -> 0).")
        self._fail_streak = 0
        self._breaker_until_ts = 0.0

    def _telemetry(
        self,
        t0: float,
        *,
        retries_used: int,
        tokens_used: Optional[int],
        breaker_state: str,
        cache_hit: bool,
    ) -> Dict[str, Any]:
        return {
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "retries_used": retries_used,
            "tokens_used": tokens_used,
            "breaker_state": breaker_state,
            "cache_hit": bool(cache_hit),
        }

    @staticmethod
    def _jittered(delay_s: float, frac: float) -> float:
        """Zwraca opóźnienie z pełnym jitterem ±frac."""
        jitter = max(0.0, float(frac)) * delay_s
        return max(0.05, delay_s + random.uniform(-jitter, jitter))

    @staticmethod
    def _token_hints(prompt: str, response_max_tokens: int) -> Dict[str, int]:
        """
        Szacowanie liczby tokenów na bazie znaków (~4 znaki/token dla modeli GPT-owych).
        Zwraca estymatę prompt_tokens oraz limit odpowiedzi (max_tokens).
        """
        try:
            approx_prompt_tokens = max(1, int(len(prompt) / 4))  # heurystyka
        except Exception:
            approx_prompt_tokens = 1
        return {
            "prompt_tokens_est": approx_prompt_tokens,
            "response_max_tokens": int(response_max_tokens),
        }
