# === core/base_agent.py ===
"""
DataGenius PRO - Base Agent Class (PRO++++++)
Abstrakcyjna klasa bazowa + orkiestratory (pipeline/parallel)
z defensywnym lifecycle, retry/backoff, opcjonalnym timeoutem,
spójnym AgentResult i telemetrią.

Zewnętrzne zależności: loguru, pydantic
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Literal, Callable, Tuple
from datetime import datetime
from uuid import uuid4
import json
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field


AgentStatus = Literal["success", "failed", "partial"]


# === EXCEPTIONS =================================================================
class AgentError(RuntimeError):
    """Ogólny błąd agenta – pozwala odróżnić błędy wewnętrzne od kontrolowanych."""


# === AGENT RESULT ===============================================================
class AgentResult(BaseModel):
    """Standard result format for all agents"""

    agent_name: str
    status: AgentStatus = Field(default="success", description="success, failed, partial")
    # Czas wykonania i znaczniki czasu
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    # Diagnostyka i dane
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # --- Helpers / API ergonomics ---
    def is_success(self) -> bool:
        return self.status == "success"

    def is_failed(self) -> bool:
        return self.status == "failed"

    def is_partial(self) -> bool:
        return self.status == "partial"

    def add_error(self, error: str) -> None:
        """Add error and set status to failed if it was success."""
        self.errors.append(error)
        if self.status == "success":
            self.status = "failed"

    def add_warning(self, warning: str) -> None:
        """Add warning and mark status partial if it was success."""
        self.warnings.append(warning)
        if self.status == "success":
            self.status = "partial"

    def add_data(self, **items: Any) -> None:
        self.data.update(items)

    def add_metadata(self, **items: Any) -> None:
        self.metadata.update(items)

    # Bezpieczna serializacja (numpy/pandas/datetimes)
    def to_json(self) -> str:
        def _default(o: Any):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            if isinstance(o, pd.Timestamp):
                return o.isoformat()
            if isinstance(o, pd.Series):
                return o.tolist()
            if isinstance(o, pd.DataFrame):
                return json.loads(o.to_json(orient="records"))
            if isinstance(o, datetime):
                return o.isoformat()
            return str(o)
        return json.dumps(self.model_dump(), default=_default, ensure_ascii=False)

    # Łączenie wyników (np. agregacja z wielu kroków)
    def merge(self, other: "AgentResult", *, prefix: Optional[str] = None) -> "AgentResult":
        p = f"{prefix}." if prefix else ""
        self.add_data(**{f"{p}data": other.data})
        self.add_metadata(**{f"{p}meta": other.metadata})
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        if other.is_failed():
            self.status = "failed"
        elif other.is_partial() and self.status == "success":
            self.status = "partial"
        return self


# === BASE AGENT ================================================================
class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.

    Implementuj:
      - execute(**kwargs)
      - (opcjonalnie) validate_input(**kwargs), before_execute(**kwargs), after_execute(result)

    Nowości PRO++++++:
      - wbudowany retry z wykładniczym backoffem,
      - opcjonalny timeout na run,
      - callback on_progress,
      - spójne logowanie start/stop + metryki.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0",
        *,
        retries: int = 0,
        retry_backoff: float = 1.6,     # multiplier (exponential backoff)
        retry_jitter: float = 0.2,      # +-20% losowego rozrzutu
        timeout_sec: Optional[float] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.name = name
        self.description = description
        self.version = version
        self.logger = logger.bind(agent=name, component="agent", version=version)
        self._result: Optional[AgentResult] = None

        # PRO++++++ polityki uruchomienia
        self.retries = max(0, int(retries))
        self.retry_backoff = max(1.0, float(retry_backoff))
        self.retry_jitter = max(0.0, float(retry_jitter))
        self.timeout_sec = timeout_sec
        self.on_progress = on_progress

    # --- Required by children ---
    @abstractmethod
    def execute(self, **kwargs) -> AgentResult:
        """Implementacja właściwej logiki agenta."""
        raise NotImplementedError

    # --- Optional hooks / validation ---
    def validate_input(self, **kwargs) -> bool:
        """Walidacja wejścia (rzucaj ValueError przy błędzie)."""
        return True

    def before_execute(self, **kwargs) -> None:
        """Hook przed execute()."""
        self._emit_progress("start", extra={"kwargs_keys": list(kwargs.keys())})
        self.logger.info(f"[{self.name}] Starting execution")

    def after_execute(self, result: AgentResult) -> None:
        """Hook po execute()."""
        self._emit_progress(
            "end",
            extra={"status": result.status, "execution_time": round(result.execution_time, 3)},
        )
        self.logger.info(
            f"[{self.name}] Execution completed "
            f"(status: {result.status}, time: {result.execution_time:.3f}s)"
        )

    def _emit_progress(self, event: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        if self.on_progress:
            try:
                self.on_progress({"agent": self.name, "event": event, "ts": datetime.utcnow().isoformat(), **(extra or {})})
            except Exception as e:
                self.logger.debug(f"on_progress callback failed: {e}")

    # --- Public entrypoint with lifecycle, retry & timeout ---
    def run(self, **kwargs) -> AgentResult:
        """
        Main entry point – zarządza pełnym cyklem życia agenta:
        walidacja → hook(before) → (retry/timeout execute) → pomiar czasu → hook(after).
        Zwraca AgentResult zawsze (również przy wyjątkach/timeout).
        """
        start_perf = time.perf_counter()
        started_at = datetime.now()
        attempt = 0
        last_exc: Optional[BaseException] = None

        try:
            self.validate_input(**kwargs)
            self.before_execute(**kwargs)

            # Retry loop
            while True:
                attempt += 1
                try:
                    result = self._execute_with_optional_timeout(kwargs)
                    if not isinstance(result, AgentResult):
                        raise AgentError(f"Invalid result type returned by {self.name}")
                    break  # success
                except Exception as e:
                    last_exc = e
                    if attempt > self.retries + 1:  # first attempt + N retries
                        raise
                    sleep_for = self._backoff_seconds(attempt - 1)
                    self.logger.warning(
                        f"[{self.name}] Attempt {attempt}/{self.retries + 1} failed: {e} "
                        f"— retry in {sleep_for:.2f}s"
                    )
                    time.sleep(sleep_for)

            # Pomiar czasu / timestamps
            result.execution_time = time.perf_counter() - start_perf
            result.started_at = started_at
            result.finished_at = datetime.now()
            if not result.timestamp:
                result.timestamp = result.finished_at

            self._result = result
            self.after_execute(result)
            return result

        except Exception as e:
            self.logger.error(f"[{self.name}] Execution failed: {e}", exc_info=True)
            failed = AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=time.perf_counter() - start_perf,
                started_at=started_at,
                finished_at=datetime.now(),
            )
            msg = f"{type(e).__name__}: {e}"
            if isinstance(last_exc, FuturesTimeout) or isinstance(e, FuturesTimeout):
                msg = "TimeoutError: Agent execution exceeded time limit"
            failed.add_error(msg)
            self._result = failed
            try:
                self.after_execute(failed)
            except Exception:
                pass
            return failed

    # --- Internals ---
    def _execute_with_optional_timeout(self, kwargs: Dict[str, Any]) -> AgentResult:
        """
        Jeśli ustawiono timeout_sec, uruchamia execute w wątku i czeka z limitem czasu.
        W przeciwnym razie uruchamia bezpośrednio.
        """
        if not self.timeout_sec or self.timeout_sec <= 0:
            return self.execute(**kwargs)

        # uruchom w wątku
        res_holder: Dict[str, Any] = {}
        exc_holder: Dict[str, BaseException] = {}
        done_evt = threading.Event()

        def _runner():
            try:
                res_holder["res"] = self.execute(**kwargs)
            except BaseException as ex:  # łapiemy BaseException, by przenieść do głównego wątku
                exc_holder["exc"] = ex
            finally:
                done_evt.set()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        finished = done_evt.wait(timeout=self.timeout_sec)
        if not finished:
            # Uwaga: nie przerywamy twardo – oznaczamy timeout i podnosimy wyjątek
            raise FuturesTimeout()

        if "exc" in exc_holder:
            raise exc_holder["exc"]
        return res_holder["res"]

    def _backoff_seconds(self, retry_index: int) -> float:
        """
        retry_index: 0 for first retry sleep, 1 for second, ...
        """
        base = self.retry_backoff ** max(0, retry_index)
        # jitter w zakresie +/- retry_jitter * base
        jitter = (np.random.rand() * 2 - 1) * self.retry_jitter * base if self.retry_jitter > 0 else 0.0
        return max(0.05, base + jitter)

    # --- Utils ---
    def get_last_result(self) -> Optional[AgentResult]:
        return self._result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"


# === PIPELINE AGENT ============================================================
class PipelineAgent(BaseAgent):
    """
    Agent orkiestrujący sekwencyjnie wiele pod–agentów.

    Parametry:
        stop_on_warning: jeśli True, przerwij pipeline na statusie 'partial'.
        propagate_metadata: włącz przenoszenie metadata między krokami (prefiksowane).
    """

    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        description: str = "",
        stop_on_warning: bool = False,
        propagate_metadata: bool = True,
        **kwargs,
    ):
        super().__init__(name, description, **kwargs)
        self.agents = agents
        self.stop_on_warning = stop_on_warning
        self.propagate_metadata = propagate_metadata

    def execute(self, **kwargs) -> AgentResult:
        """Execute all agents in sequence."""
        result = AgentResult(agent_name=self.name)
        pipeline_results: List[AgentResult] = []
        results_index: Dict[str, AgentResult] = {}

        carry = dict(kwargs)  # wejście bazowe

        for idx, agent in enumerate(self.agents):
            step_label = f"{self.name}:{agent.name}"
            self.logger.info(f"[Pipeline] Executing {step_label}")

            agent_result = agent.run(**carry)
            pipeline_results.append(agent_result)
            results_index[agent.name] = agent_result

            # Przeniesienie danych do kolejnego kroku
            if agent_result.data:
                carry.update(agent_result.data)
            if self.propagate_metadata and agent_result.metadata:
                carry.setdefault("_pipeline_meta", {})[agent.name] = agent_result.metadata

            # Kontrola statusu
            if agent_result.is_failed():
                result.add_error(
                    f"Agent {agent.name} failed: {', '.join(agent_result.errors) or 'unknown error'}"
                )
                break
            if self.stop_on_warning and agent_result.is_partial():
                result.add_warning(f"Agent {agent.name} returned warnings – pipeline stopped")
                break

        # Status końcowy
        if any(r.is_failed() for r in pipeline_results):
            result.status = "failed"
        elif any(r.is_partial() for r in pipeline_results):
            result.status = "partial"
        else:
            result.status = "success"

        # Agregacja
        result.add_data(
            pipeline_results=pipeline_results,
            agents_executed=len(pipeline_results),
            final_data=carry,
            results_index=results_index,
        )
        return result

    def add_agent(self, agent: BaseAgent) -> None:
        self.agents.append(agent)

    def remove_agent(self, agent_name: str) -> bool:
        for i, agent in enumerate(self.agents):
            if agent.name == agent_name:
                self.agents.pop(i)
                return True
        return False


# === PARALLEL AGENT ============================================================
class ParallelAgent(BaseAgent):
    """
    Agent uruchamiający wiele pod–agentów równolegle (wątkowo).
    Każdy sub–agent zachowuje własne retry/timeout/telemetrię.
    """

    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        description: str = "",
        max_workers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name, description, **kwargs)
        self.agents = agents
        self.max_workers = max_workers

    def execute(self, **kwargs) -> AgentResult:
        """Execute all agents in parallel."""
        result = AgentResult(agent_name=self.name)
        parallel_results: List[AgentResult] = []
        results_index: Dict[str, AgentResult] = {}

        workers = min(len(self.agents), self.max_workers or len(self.agents))
        if workers <= 0:
            result.add_error("No agents to execute")
            return result

        # Wysyłamy progres (start)
        self._emit_progress("parallel_start", extra={"workers": workers, "n_agents": len(self.agents)})

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(agent.run, **kwargs): agent for agent in self.agents}

            for future in as_completed(futures):
                agent = futures[future]
                try:
                    agent_result = future.result(timeout=getattr(agent, "timeout_sec", None) or None)
                    parallel_results.append(agent_result)
                    results_index[agent.name] = agent_result

                    if not agent_result.is_success():
                        result.add_warning(
                            f"Agent {agent.name} status: {agent_result.status} "
                            f"({'; '.join(agent_result.errors) if agent_result.errors else 'no errors listed'})"
                        )
                except FuturesTimeout:
                    msg = f"TimeoutError: Agent {agent.name} execution exceeded time limit"
                    self.logger.error(msg)
                    failed = AgentResult(agent_name=agent.name, status="failed")
                    failed.add_error(msg)
                    parallel_results.append(failed)
                    results_index[agent.name] = failed
                except Exception as e:
                    msg = f"Agent {agent.name} raised exception: {e}"
                    self.logger.error(msg)
                    failed = AgentResult(agent_name=agent.name, status="failed")
                    failed.add_error(msg)
                    parallel_results.append(failed)
                    results_index[agent.name] = failed

        # Status zbiorczy
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
        self._emit_progress("parallel_end", extra={"status": result.status})
        return result
