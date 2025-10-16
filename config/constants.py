# === core/base_agent.py ===
"""
DataGenius PRO - Base Agent Class (PRO+++++ Enterprise)
Abstrakcyjna klasa bazowa agentów + orkiestratory (pipeline / parallel) w wersji enterprise:
- Retry z eksponent. backoffem (z jitterem), timeout per-run
- Circuit Breaker (half-open, okno czasowe, progi błędów)
- Opcjonalny rate limiter (token bucket, w pamięci)
- Sync i async run() z tym samym kontraktem
- Rozszerzone telemetry hooks (metrics_callback, audit_callback)
- Silna, bezpieczna serializacja (datetime/np/pd) i redakcja wiadomości
- Śledzenie: trace_id, parent_trace_id, correlation_id
- 100% kompatybilne z istniejącymi modułami (AgentResult / BaseAgent API)

Zależności: loguru, pydantic (jak wcześniej). Brak dodatkowych zależności.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional, List, Literal, Tuple, Union

from loguru import logger
from pydantic import BaseModel, Field

# Typy statusów
AgentStatus = Literal["success", "failed", "partial"]

# --------------------------------------------------------------------------------------
# Bezpieczna serializacja do logów/JSON (bez importów ciężkich pakietów)
# --------------------------------------------------------------------------------------
def _safe_json_default(o: Any) -> Any:
    try:
        import numpy as np  # pragma: no cover
        import pandas as pd  # pragma: no cover
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, pd.Series):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return json.loads(o.to_json(orient="records"))
    except Exception:
        pass
    if isinstance(o, datetime):
        return o.isoformat()
    try:
        return str(o)
    except Exception:
        return "<unserializable>"

def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=_safe_json_default, ensure_ascii=False)


# --------------------------------------------------------------------------------------
# AgentResult – kompatybilny + rozbudowane pola śledzenia
# --------------------------------------------------------------------------------------
class AgentResult(BaseModel):
    """Standard result format for all agents (enterprise)."""

    agent_name: str
    status: AgentStatus = Field(default="success", description="success, failed, partial")

    # Timings
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    # Tracing
    trace_id: str = Field(default_factory=lambda: _uuid4_hex())
    parent_trace_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Payload
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # API helpers
    def is_success(self) -> bool:
        return self.status == "success"

    def is_failed(self) -> bool:
        return self.status == "failed"

    def is_partial(self) -> bool:
        return self.status == "partial"

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        if self.status == "success":
            self.status = "failed"

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
        if self.status == "success":
            self.status = "partial"

    def add_data(self, **items: Any) -> None:
        self.data.update(items)

    def add_metadata(self, **items: Any) -> None:
        self.metadata.update(items)

    def to_json(self) -> str:
        return _json_dumps(self.model_dump())


# --------------------------------------------------------------------------------------
# Pomocnicze narzędzia
# --------------------------------------------------------------------------------------
def _uuid4_hex() -> str:
    from uuid import uuid4
    return uuid4().hex


def _exp_backoff(base: float, attempt: int, jitter: float = 0.25, cap: Optional[float] = None) -> float:
    delay = base * (2 ** max(0, attempt - 1))
    if jitter:
        import random
        delay *= (1.0 - jitter) + 2 * jitter * random.random()
    if cap:
        delay = min(delay, cap)
    return delay


@contextmanager
def _deadline(timeout: Optional[float]):
    """
    Prosty "deadline" dla funkcji sync (soft): po czasie ustaw flagę, a call site powinien to respektować.
    Nie używamy sygnałów (zgodność z Windows).
    """
    flag = {"expired": False}
    timer = None
    if timeout and timeout > 0:
        def _mark():
            flag["expired"] = True
        timer = threading.Timer(timeout, _mark)
        timer.daemon = True
        timer.start()
    try:
        yield flag
    finally:
        if timer:
            timer.cancel()


# --------------------------------------------------------------------------------------
# Circuit Breaker (w pamięci – per klasa agenta)
# --------------------------------------------------------------------------------------
@dataclass
class _BreakerState:
    failures: int = 0
    last_failure_ts: float = 0.0
    open_until_ts: float = 0.0
    half_open: bool = False


class _CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        open_seconds: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.open_seconds = open_seconds
        self.half_open_max_calls = half_open_max_calls
        self._state = _BreakerState()
        self._lock = threading.Lock()
        self._half_open_calls = 0

    def can_pass(self) -> bool:
        with self._lock:
            now = time.time()
            if self._state.open_until_ts > now:
                return False
            if self._state.open_until_ts <= now and self._state.failures >= self.failure_threshold:
                # przejście do half-open
                self._state.half_open = True
                self._half_open_calls = 0
            return True

    def on_success(self) -> None:
        with self._lock:
            self._state.failures = 0
            self._state.half_open = False
            self._state.open_until_ts = 0.0
            self._half_open_calls = 0

    def on_failure(self) -> None:
        with self._lock:
            now = time.time()
            self._state.failures += 1
            self._state.last_failure_ts = now
            if self._state.failures >= self.failure_threshold:
                self._state.open_until_ts = now + self.open_seconds
                self._state.half_open = False

    def try_half_open_ticket(self) -> bool:
        with self._lock:
            if not self._state.half_open:
                return True
            if self._half_open_calls >= self.half_open_max_calls:
                return False
            self._half_open_calls += 1
            return True


# --------------------------------------------------------------------------------------
# Rate Limiter (opcjonalny, token bucket – w pamięci)
# --------------------------------------------------------------------------------------
class _TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate = float(rate_per_sec)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self.last = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


# --------------------------------------------------------------------------------------
# BaseAgent (enterprise)
# --------------------------------------------------------------------------------------
class BaseAgent(ABC):
    """
    Abstrakcyjna klasa bazowa agentów (enterprise).

    Nowe możliwości:
    - retry: max_retries, backoff_base, backoff_cap, retry_on (krotka wyjątków)
    - timeout_sec: miękki limit czasu
    - circuit breaker: włącz/wyłącz + progi
    - rate limiting: rate_per_sec / capacity
    - async_execute: jeśli klasa nadpisze execute_async, run_async() użyje tej wersji
    - metrics/audit hooks: metrics_callback(event_dict), audit_callback(event_dict)
    """

    # domyślne polityki (można nadpisać w __init__)
    max_retries: int = 0
    backoff_base: float = 0.4
    backoff_cap: Optional[float] = 10.0
    retry_on: Tuple[type, ...] = (RuntimeError,)

    timeout_sec: Optional[float] = None

    enable_circuit_breaker: bool = False
    breaker_failure_threshold: int = 5
    breaker_open_seconds: float = 30.0
    breaker_half_open_max_calls: int = 1

    rate_limit_rps: Optional[float] = None
    rate_limit_capacity: int = 5

    metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def __init__(self, name: str, description: str = "", version: str = "1.0"):
        self.name = name
        self.description = description
        self.version = version
        self.logger = logger.bind(agent=name, component="agent", version=version)

        # stan breaker/ratelimitera (per instancja; można łatwo współdzielić globalnie)
        self._breaker = _CircuitBreaker(
            failure_threshold=self.breaker_failure_threshold,
            open_seconds=self.breaker_open_seconds,
            half_open_max_calls=self.breaker_half_open_max_calls,
        ) if self.enable_circuit_breaker else None

        self._limiter = (
            _TokenBucket(rate_per_sec=self.rate_limit_rps, capacity=self.rate_limit_capacity)
            if self.rate_limit_rps and self.rate_limit_rps > 0
            else None
        )

        self._result: Optional[AgentResult] = None

    # --- Wymagane przez dzieci ---
    @abstractmethod
    def execute(self, **kwargs) -> AgentResult:
        """Właściwa logika agenta (synchronous)."""
        raise NotImplementedError

    # Opcjonalny wariant asynchroniczny
    async def execute_async(self, **kwargs) -> AgentResult:
        # fallback: uruchom w wątku – kompatybilne z kodem sync
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute, **kwargs)

    # --- Walidacja / Hooki ---
    def validate_input(self, **kwargs) -> bool:
        return True

    def before_execute(self, **kwargs) -> None:
        self.logger.info(f"[{self.name}] Starting execution")

    def after_execute(self, result: AgentResult) -> None:
        self.logger.info(
            f"[{self.name}] Execution completed (status: {result.status}, time: {result.execution_time:.3f}s)"
        )

    # ----------------------------------------------------------------------------------
    # PUBLICZNE: run() i run_async() – 100% backwards compatible
    # ----------------------------------------------------------------------------------
    def run(
        self,
        *,
        parent_trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs,
    ) -> AgentResult:
        """
        Główne wejście (sync). Zarządza retry/timeout/circuit/ratelimit + telemetry.
        """
        return asyncio.run(
            self.run_async(parent_trace_id=parent_trace_id, correlation_id=correlation_id, **kwargs)
        )

    async def run_async(
        self,
        *,
        parent_trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs,
    ) -> AgentResult:
        """
        Wersja asynchroniczna (zalecana w środowiskach async). Zachowuje tę samą semantykę.
        """
        start_perf = time.perf_counter()
        started_at = datetime.now()
        trace_id = _uuid4_hex()

        # ratelimiter
        if self._limiter and not self._limiter.consume():
            msg = f"[{self.name}] Rate limit exceeded"
            self.logger.warning(msg)
            overloaded = AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=0.0,
                started_at=started_at,
                finished_at=datetime.now(),
                parent_trace_id=parent_trace_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
            )
            overloaded.add_error(msg)
            return overloaded

        # circuit breaker
        if self._breaker and not self._breaker.can_pass():
            msg = f"[{self.name}] Circuit open – rejecting execution"
            self.logger.error(msg)
            rejected = AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=0.0,
                started_at=started_at,
                finished_at=datetime.now(),
                parent_trace_id=parent_trace_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
            )
            rejected.add_error(msg)
            return rejected

        # lifecycle + retry
        attempts = max(1, int(self.max_retries) + 1)
        last_exc: Optional[BaseException] = None
        result: Optional[AgentResult] = None

        # Walidacja + before hook (bez retry)
        try:
            self.validate_input(**kwargs)
            self.before_execute(**kwargs)
        except Exception as e:
            self.logger.error(f"[{self.name}] Pre-execution failed: {e}", exc_info=True)
            failed = AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=time.perf_counter() - start_perf,
                started_at=started_at,
                finished_at=datetime.now(),
                parent_trace_id=parent_trace_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
            )
            failed.add_error(str(e))
            self._emit_metrics("agent_pre_fail", failed)
            return failed

        for attempt in range(1, attempts + 1):
            try:
                # soft timeout guard (dla sync/async – kontrolujemy pętlą)
                with _deadline(self.timeout_sec) as dl:
                    if asyncio.iscoroutinefunction(self.execute_async):
                        res = await self.execute_async(**kwargs)
                    else:
                        # teoretycznie nie dojdzie – execute_async istnieje
                        res = await asyncio.get_running_loop().run_in_executor(None, self.execute, **kwargs)

                    # timeout soft: agent sam powinien sprawdzić flagę – tu logujemy ostrzeżenie
                    if dl["expired"]:
                        res.add_warning("Execution exceeded soft timeout.")
                        self.logger.warning(f"[{self.name}] Soft timeout reached.")

                # sanity: zawsze AgentResult
                if not isinstance(res, AgentResult):
                    raise TypeError(f"Invalid result type returned by {self.name} (expected AgentResult)")

                # wzbogacenie tracerami i timingiem
                res.started_at = res.started_at or started_at
                res.finished_at = datetime.now()
                res.execution_time = time.perf_counter() - start_perf
                res.parent_trace_id = parent_trace_id or res.parent_trace_id
                res.correlation_id = correlation_id or res.correlation_id
                res.trace_id = res.trace_id or trace_id
                result = res

                # breaker: sukces
                if self._breaker:
                    self._breaker.on_success()

                # after hook + metrics/audit
                self._result = result
                try:
                    self.after_execute(result)
                finally:
                    self._emit_metrics("agent_success", result)
                    self._emit_audit("agent_success", result)
                return result

            except Exception as e:
                last_exc = e
                # breaker: failure
                if self._breaker:
                    self._breaker.on_failure()

                # retry?
                will_retry = attempt < attempts and isinstance(e, self.retry_on)
                self.logger.error(
                    f"[{self.name}] Attempt {attempt}/{attempts} failed: {e} "
                    f"{'(retrying...)' if will_retry else '(no retry)'}",
                    exc_info=True,
                )
                if will_retry:
                    delay = _exp_backoff(self.backoff_base, attempt, jitter=0.25, cap=self.backoff_cap)
                    await asyncio.sleep(delay)
                else:
                    break

        # final failure
        failed = AgentResult(
            agent_name=self.name,
            status="failed",
            execution_time=time.perf_counter() - start_perf,
            started_at=started_at,
            finished_at=datetime.now(),
            parent_trace_id=parent_trace_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
        )
        if last_exc:
            failed.add_error(str(last_exc))
        self._result = failed
        try:
            self.after_execute(failed)
        finally:
            self._emit_metrics("agent_fail", failed)
            self._emit_audit("agent_fail", failed)
        return failed

    # --- Telemetria ---
    def _emit_metrics(self, event: str, result: AgentResult) -> None:
        if not self.metrics_callback:
            return
        try:
            self.metrics_callback({
                "event": event,
                "agent": self.name,
                "status": result.status,
                "exec_sec": result.execution_time,
                "ts": (result.finished_at or datetime.now()).isoformat(),
                "trace_id": result.trace_id,
                "corr_id": result.correlation_id,
                "errors": len(result.errors),
                "warnings": len(result.warnings),
            })
        except Exception as e:
            self.logger.warning(f"[{self.name}] metrics_callback failed: {e}")

    def _emit_audit(self, event: str, result: AgentResult) -> None:
        if not self.audit_callback:
            return
        try:
            self.audit_callback({
                "event": event,
                "agent": self.name,
                "status": result.status,
                "trace_id": result.trace_id,
                "parent_trace_id": result.parent_trace_id,
                "correlation_id": result.correlation_id,
                "started_at": (result.started_at or datetime.now()).isoformat(),
                "finished_at": (result.finished_at or datetime.now()).isoformat(),
            })
        except Exception as e:
            self.logger.warning(f"[{self.name}] audit_callback failed: {e}")

    # --- Utils ---
    def get_last_result(self) -> Optional[AgentResult]:
        return self._result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"


# --------------------------------------------------------------------------------------
# PipelineAgent – sekwencyjny orkiestrator (enterprise)
# --------------------------------------------------------------------------------------
class PipelineAgent(BaseAgent):
    """
    Sekwencyjna orkiestracja wielu agentów.

    Dodatkowo:
    - obsługa async (z użyciem run_async każdego agenta)
    - stop_on_warning (jak wcześniej)
    - propagacja trace_id/correlation_id
    """

    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        description: str = "",
        stop_on_warning: bool = False,
    ):
        super().__init__(name, description)
        self.agents = agents
        self.stop_on_warning = stop_on_warning

    def execute(self, **kwargs) -> AgentResult:
        # zachowujemy zgodność – wywołanie sync spakujemy w pętlę async
        return asyncio.run(self.execute_async(**kwargs))

    async def execute_async(self, **kwargs) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        pipeline_results: List[AgentResult] = []
        results_index: Dict[str, AgentResult] = {}

        carry = dict(kwargs)
        parent_trace = _uuid4_hex()
        started = time.perf_counter()

        for agent in self.agents:
            self.logger.info(f"[Pipeline: {self.name}] Executing {agent.name}")
            ar = await agent.run_async(parent_trace_id=parent_trace, correlation_id=carry.get("correlation_id"), **carry)
            pipeline_results.append(ar)
            results_index[agent.name] = ar

            if ar.data:
                carry.update(ar.data)

            if ar.is_failed():
                result.add_error(
                    f"Agent {agent.name} failed: {', '.join(ar.errors) or 'unknown error'}"
                )
                break
            if self.stop_on_warning and ar.is_partial():
                result.add_warning(
                    f"Agent {agent.name} returned warnings – pipeline stopped"
                )
                break

        # status zbiorczy
        if any(r.is_failed() for r in pipeline_results):
            result.status = "failed"
        elif any(r.is_partial() for r in pipeline_results):
            result.status = "partial"
        else:
            result.status = "success"

        result.execution_time = time.perf_counter() - started
        result.add_data(
            pipeline_results=pipeline_results,
            agents_executed=len(pipeline_results),
            final_data=carry,
            results_index=results_index,
        )
        return result


# --------------------------------------------------------------------------------------
# ParallelAgent – równoległa orkiestracja (enterprise)
# --------------------------------------------------------------------------------------
class ParallelAgent(BaseAgent):
    """
    Uruchamia wiele agentów równolegle (wątki) z bezpieczną agregacją wyników.
    """

    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        description: str = "",
        max_workers: Optional[int] = None,
    ):
        super().__init__(name, description)
        self.agents = agents
        self.max_workers = max_workers

    def execute(self, **kwargs) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        parallel_results: List[AgentResult] = []
        results_index: Dict[str, AgentResult] = {}

        workers = min(len(self.agents), self.max_workers or len(self.agents))
        if workers <= 0:
            result.add_error("No agents to execute")
            return result

        # Propagacja correlation_id między wątkami
        corr_id = kwargs.get("correlation_id")

        def _runner(agent: BaseAgent) -> AgentResult:
            # sync bridge do run_async
            return asyncio.run(agent.run_async(correlation_id=corr_id, **kwargs))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_runner, agent): agent for agent in self.agents}
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    ar = future.result()
                    parallel_results.append(ar)
                    results_index[agent.name] = ar
                    if not ar.is_success():
                        result.add_warning(
                            f"Agent {agent.name} status: {ar.status} "
                            f"({'; '.join(ar.errors) if ar.errors else 'no errors listed'})"
                        )
                except Exception as e:
                    result.add_error(f"Agent {agent.name} raised exception: {e}")

        # status zbiorczy
        if all(r.is_success() for r in parallel_results) and not result.errors:
            result.status = "success"
        elif any(r.is_success() for r in parallel_results):
            result.status = "partial"
        else:
            result.status = "failed"

        result.add_data(
            parallel_results=parallel_results,
            agents_executed=len(parallel_results),
            results_index=results_index,
        )
        return result
