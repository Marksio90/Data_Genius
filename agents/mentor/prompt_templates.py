# === OPIS MODUŁU ===
"""
DataGenius PRO++++ - AI Mentor Orchestrator (KOSMOS)
Wyjaśnienia EDA/ML, rekomendacje i Q&A z LLM.

Funkcje PRO++++:
- twarde kontrakty wyników + wersjonowanie modułu
- defensywa I/O (walidacja, degradacje kontrolowane)
- PII scrub (email/telefon/IP/ID-like) + normalizacja whitespace
- limiter kontekstu (head/tail, soft i hard cap)
- retry + exponential backoff + pełny jitter
- circuit-breaker (licznik awarii + cooldown)
- JSON-first z bezpiecznym fallbackiem do tekstu
- cache (TTL + maxsize) dla identycznych zapytań/kontekstu
- telemetry: czas, próby, breaker, cache hit, token-hints
- DI klienta LLM (łatwy mocking/testy)
"""

from __future__ import annotations

import json
import random
import re
import time
import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

import pandas as pd  # wymagane przez kontrakt bazowy
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from agents.mentor.prompt_templates import (
    MENTOR_SYSTEM_PROMPT,
    EDA_EXPLANATION_TEMPLATE,
    ML_RESULTS_TEMPLATE,
    RECOMMENDATION_TEMPLATE,
)

__all__ = ["MentorConfig", "MentorOrchestrator"]
__version__ = "4.2-kosmos"


# === KONFIG / PARAMETRY ===
@dataclass(frozen=True)
class MentorConfig:
    """Ustawienia działania Mentora (deterministycznie i stabilnie)."""
    # LLM
    temperature: float = 0.45
    max_tokens: int = 2000
    json_max_tokens: int = 1400
    request_timeout_s: Optional[float] = None

    # retry/backoff
    retries: int = 2
    initial_backoff_s: float = 0.6
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.25  # +/- jitter dla backoffu

    # context & scrub
    pii_scrub: bool = True
    context_max_chars: int = 60_000  # twardy limit znaków kontekstu
    trim_section_head: int = 10_000
    trim_section_tail: int = 3_000
    prefer_json: bool = True

    # circuit breaker
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: float = 30.0

    # cache (TTL + maxsize)
    cache_enabled: bool = True
    cache_ttl_s: int = 180
    cache_maxsize: int = 256

    # telemetry (szacowanie tokenów — przybliżone)
    token_hint_avg_chars_per_token: float = 3.8


# === PROSTY CACHE Z TTL ===
class _TTLCache:
    def __init__(self, maxsize: int, ttl_s: int) -> None:
        self.maxsize = int(maxsize)
        self.ttl_s = float(ttl_s)
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            ts, val = item
            if time.time() - ts > self.ttl_s:
                self._store.pop(key, None)
                return None
            return val

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.maxsize:
                # usuń najstarszy wpis
                oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                self._store.pop(oldest_key, None)
            self._store[key] = (time.time(), value)


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


# === KLASA GŁÓWNA ===
class MentorOrchestrator(BaseAgent):
    """
    AI Mentor - wyjaśnienia i rekomendacje na bazie EDA/ML.
    Stabilny, defensywny i gotowy do pracy w produkcji (PRO++++).
    """

    def __init__(self, config: Optional[MentorConfig] = None, llm_client: Any = None) -> None:
        super().__init__(name="MentorOrchestrator", description="AI Mentor for data science guidance")
        self.config = config or MentorConfig()
        # DI klienta — łatwy mocking/offline
        try:
            self.llm_client = llm_client or get_llm_client()
        except Exception as e:
            logger.warning(f"LLM client unavailable; running in degraded mode. Reason: {e}")
            class _NullClient:
                def generate(self, *a, **k):
                    raise RuntimeError("LLM client not available")
                def generate_json(self, *a, **k):
                    raise RuntimeError("LLM client not available")
            self.llm_client = _NullClient()

        # circuit-breaker
        self._fail_streak: int = 0
        self._breaker_until_ts: float = 0.0

        # cache
        self._cache = _TTLCache(self.config.cache_maxsize, self.config.cache_ttl_s)

    # === API GŁÓWNE ===
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> AgentResult:
        """
        Odpowiada na pytanie użytkownika, uwzględniając (opcjonalnie) kontekst EDA/ML/data_info.
        Zwraca AgentResult z treścią i telemetrią (wersjonowane).
        """
        result = AgentResult(agent_name=self.name)
        t0 = time.perf_counter()
        cfg = self.config

        try:
            # Walidacja
            if not isinstance(query, str) or not query.strip():
                msg = "MentorOrchestrator: 'query' must be a non-empty string."
                result.add_error(msg); self.logger.error(msg); return result
            if context is not None and not isinstance(context, dict):
                msg = "MentorOrchestrator: 'context' must be a dict if provided."
                result.add_error(msg); self.logger.error(msg); return result

            # Circuit breaker
            if self._is_breaker_open():
                cooldown = max(0.0, self._breaker_until_ts - time.time())
                warn = f"Circuit breaker is open. Cooldown {cooldown:.1f}s."
                self.logger.warning(warn)
                result.add_warning(warn)
                result.data = self._wrap_payload(
                    response="Tymczasowa przerwa w wywołaniach LLM (zabezpieczenie). Spróbuj ponownie.",
                    query=query, context_len=0, tokens_used=None,
                    breaker_state="open", cache_hit=False, elapsed_s=time.perf_counter()-t0
                )
                return result

            # Kontekst → prompt (PII scrub + trimming)
            context_str = self._prepare_context(context or {})
            full_prompt = self._create_prompt(query, context_str)

            # Cache (idempotentne)
            cache_key = None
            cache_hit = False
            if cfg.cache_enabled:
                cache_key = _hash_key("mentor", MENTOR_SYSTEM_PROMPT or "", full_prompt)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    cache_hit = True
                    content, tokens_used = cached
                    self._reset_breaker()
                    result.data = self._wrap_payload(
                        response=content, query=query,
                        context_len=len(context_str), tokens_used=tokens_used,
                        breaker_state="closed", cache_hit=True,
                        elapsed_s=time.perf_counter()-t0
                    )
                    return result

            # LLM z retry
            response = self._call_llm_with_retry(
                prompt=full_prompt,
                system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens
            )
            content = getattr(response, "content", None) or str(response)
            tokens_used = getattr(response, "tokens_used", None)

            # set cache
            if cfg.cache_enabled and cache_key:
                self._cache.set(cache_key, (content, tokens_used))

            self._reset_breaker()
            result.data = self._wrap_payload(
                response=content, query=query,
                context_len=len(context_str), tokens_used=tokens_used,
                breaker_state="closed", cache_hit=cache_hit,
                elapsed_s=time.perf_counter()-t0
            )
            self.logger.success("AI Mentor response generated")
            return result

        except Exception as e:
            self._register_failure()
            result.add_error(f"AI Mentor failed: {e}")
            self.logger.error(f"AI Mentor error: {e}", exc_info=True)
            result.data = self._wrap_payload(
                response="Nie udało się wygenerować odpowiedzi (degradacja bezpieczna).",
                query=query, context_len=0, tokens_used=None,
                breaker_state="maybe_open", cache_hit=False,
                elapsed_s=time.perf_counter()-t0
            )
            return result

    # === WYJAŚNIENIA EDA ===
    def explain_eda_results(self, eda_results: Dict[str, Any], user_level: Literal["beginner","intermediate","advanced"]="beginner") -> str:
        try:
            if not isinstance(eda_results, dict):
                raise ValueError("'eda_results' must be a dict")
            eda_str = self._prepare_context({"eda_results": eda_results})
            prompt = EDA_EXPLANATION_TEMPLATE.format(eda_results=eda_str, user_level=user_level)
            resp = self._call_llm_with_retry(
                prompt=prompt, system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=self.config.temperature, max_tokens=self.config.max_tokens
            )
            self._reset_breaker()
            return getattr(resp, "content", "") or "—"
        except Exception as e:
            self._register_failure()
            self.logger.error(f"EDA explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."

    # === WYJAŚNIENIA ML ===
    def explain_ml_results(self, ml_results: Dict[str, Any], user_level: Literal["beginner","intermediate","advanced"]="beginner") -> str:
        try:
            if not isinstance(ml_results, dict):
                raise ValueError("'ml_results' must be a dict")
            ml_str = self._prepare_context({"ml_results": ml_results})
            prompt = ML_RESULTS_TEMPLATE.format(ml_results=ml_str, user_level=user_level)
            resp = self._call_llm_with_retry(
                prompt=prompt, system_prompt=MENTOR_SYSTEM_PROMPT,
                temperature=self.config.temperature, max_tokens=self.config.max_tokens
            )
            self._reset_breaker()
            return getattr(resp, "content", "") or "—"
        except Exception as e:
            self._register_failure()
            self.logger.error(f"ML explanation failed: {e}")
            return "Przepraszam, nie udało się wygenerować wyjaśnienia."

    # === REKOMENDACJE (JSON-first) ===
    def generate_recommendations(
        self,
        eda_results: Optional[Dict[str, Any]] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        data_quality: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generuje listę rekomendacji; preferuje wynik w JSON, z bezpiecznym fallbackiem tekstowym."""
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
                self.logger.warning("LLM JSON empty/invalid. Falling back to text generation for recommendations.")
                txt = self._call_llm_with_retry(
                    prompt=prompt, system_prompt=MENTOR_SYSTEM_PROMPT,
                    temperature=self.config.temperature, max_tokens=self.config.max_tokens
                )
                content = getattr(txt, "content", "") or ""
                recs = [line.strip("-• ").strip() for line in content.splitlines() if line.strip()]

            self._reset_breaker()
            return recs or ["Nie udało się wygenerować rekomendacji."]

        except Exception as e:
            self._register_failure()
            self.logger.error(f"Recommendation generation failed: {e}")
            return ["Nie udało się wygenerować rekomendacji."]

    # === KONTEKST / PROMPTY ===
    def _prepare_context(self, context: Dict[str, Any]) -> str:
        """
        Buduje i sanituzuje kontekst (PII scrub + trimming).
        Priorytety: EDA -> ML -> data_info -> inne.
        """
        parts: List[str] = []

        def _pack(title: str, obj: Any) -> str:
            s = self._safe_stringify(obj)
            if self.config.pii_scrub:
                s = self._scrub_pii(s)
            return f"**{title}:**\n{self._trim_context_if_needed(s)}"

        if "eda_results" in context:
            parts.append(_pack("Wyniki EDA", context["eda_results"]))
        if "ml_results" in context:
            parts.append(_pack("Wyniki ML", context["ml_results"]))
        if "data_info" in context:
            parts.append(_pack("Informacje o danych", context["data_info"]))
        # pozostałe klucze
        for k, v in context.items():
            if k not in {"eda_results", "ml_results", "data_info"}:
                parts.append(_pack(k, v))

        joined = "\n\n".join(parts).strip()
        return self._trim_context_if_needed(joined)

    def _create_prompt(self, query: str, context: str) -> str:
        """Tworzy kompletny prompt (z kontekstem, jeśli jest dostępny)."""
        if context:
            return (
                f"Kontekst analizy:\n{context}\n\n"
                f"Pytanie użytkownika:\n{query}\n\n"
                "Odpowiedz po polsku, rzeczowo i praktycznie. Stosuj krótkie sekcje i jasne rekomendacje. "
                "Jeśli czegoś brakuje w kontekście — wskaż to i zaproponuj kolejny krok."
            )
        return query

    # === LLM CALLS: RETRY + JITTER / JSON-FIRST ===
    def _call_llm_with_retry(self, prompt: str, system_prompt: Optional[str], temperature: float, max_tokens: int):
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
                    jitter = delay * self.config.jitter_fraction
                    sleep_s = delay + random.uniform(-jitter, jitter)
                    time.sleep(max(0.05, sleep_s))
                    delay *= self.config.backoff_multiplier
        raise RuntimeError(f"LLM generate failed after {attempts} attempts: {last_err}")

    def _call_llm_json_with_retry(self, prompt: str, system_prompt: Optional[str], max_tokens: int) -> Dict[str, Any]:
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
                self.logger.warning(f"LLM generate_json failed (attempt {i+1}/{attempts}): {e}")
                if i < attempts - 1:
                    jitter = delay * self.config.jitter_fraction
                    sleep_s = delay + random.uniform(-jitter, jitter)
                    time.sleep(max(0.05, sleep_s))
                    delay *= self.config.backoff_multiplier
        self.logger.error(f"LLM generate_json failed after {attempts} attempts: {last_err}")
        return {}

    # === POMOCNICZE / BEZPIECZEŃSTWO ===
    def _safe_stringify(self, obj: Any) -> str:
        """Bezpieczne stringify; cap 2x hard limit (finalny cap w _trim_context_if_needed)."""
        try:
            s = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
        return s[: self.config.context_max_chars * 2]

    def _trim_context_if_needed(self, text: str) -> str:
        """Skraca tekst kontekstu, zachowując head i tail (np. podsumowania)."""
        max_len = self.config.context_max_chars
        if len(text) <= max_len:
            return text
        head = text[: self.config.trim_section_head]
        tail = text[-self.config.trim_section_tail :]
        note = f"\n\n[Uwaga: kontekst skrócony do {max_len} znaków]"
        return head + "\n...\n" + tail + note

    def _scrub_pii(self, text: str) -> str:
        """Lekki PII scrub (email/telefon/IP/ID-like) + normalizacja whitespace."""
        try:
            text = re.sub(r"[ \t]+", " ", text)  # whitespace normalize
            text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
            text = re.sub(r"\b(\+?\d[\d\s\-]{7,}\d)\b", "[PHONE]", text)
            text = re.sub(r"\b(\d{1,3}\.){3}\d{1,3}\b", "[IPV4]", text)  # IPv4
            text = re.sub(r"\b\d{11}\b", "[ID]", text)  # PESEL-like
        except Exception:
            pass
        return text

    # === CIRCUIT-BREAKER ===
    def _is_breaker_open(self) -> bool:
        if self._fail_streak < self.config.circuit_breaker_failures:
            return False
        return time.time() < self._breaker_until_ts

    def _register_failure(self) -> None:
        self._fail_streak += 1
        if self._fail_streak >= self.config.circuit_breaker_failures:
            self._breaker_until_ts = time.time() + self.config.circuit_breaker_cooldown_s
            self.logger.warning(
                f"Circuit breaker OPEN for {self.config.circuit_breaker_cooldown_s}s (failures={self._fail_streak})."
            )

    def _reset_breaker(self) -> None:
        if self._fail_streak > 0:
            self.logger.info(f"Circuit breaker reset (fail_streak={self._fail_streak} -> 0).")
        self._fail_streak = 0
        self._breaker_until_ts = 0.0

    # === TELEMETRY + WRAPPER KONTRAKTU ===
    def _token_hint(self, prompt_len: int) -> int:
        # przybliżenie — tylko hint do telemetry
        try:
            return int(max(1, round(prompt_len / self.config.token_hint_avg_chars_per_token)))
        except Exception:
            return 0

    def _wrap_payload(
        self,
        *,
        response: str,
        query: str,
        context_len: int,
        tokens_used: Optional[int],
        breaker_state: str,
        cache_hit: bool,
        elapsed_s: float
    ) -> Dict[str, Any]:
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
                "breaker_state": breaker_state,       # 'open' | 'closed' | 'maybe_open'
                "cache_hit": cache_hit,
                "token_hint_in": self._token_hint(context_len),
            },
            "version": __version__,
        }
