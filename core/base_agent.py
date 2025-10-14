# === core/base_agent.py ===
"""
DataGenius PRO - Base Agent Class (PRO+++)
Abstrakcyjna klasa bazowa i orkiestratory (pipeline / parallel)
z defensywnym lifecycle, logowaniem i spójnym AgentResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Literal
from datetime import datetime
from uuid import uuid4
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from pydantic import BaseModel, Field


AgentStatus = Literal["success", "failed", "partial"]


# === AGENT RESULT =============================================================
class AgentResult(BaseModel):
    """Standard result format for all agents"""

    agent_name: str
    status: AgentStatus = Field(
        default="success", description="success, failed, partial"
    )
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

    def to_json(self) -> str:
        """Safe JSON for logging/export (datetime → isoformat)."""
        def _default(o: Any):
            if isinstance(o, datetime):
                return o.isoformat()
            return str(o)
        return json.dumps(self.model_dump(), default=_default)


# === BASE AGENT ===============================================================
class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.

    Każdy Agent powinien nadpisać:
      - execute(**kwargs)
      - opcjonalnie validate_input(**kwargs), before_execute(**kwargs), after_execute(result)
    """

    def __init__(self, name: str, description: str = "", version: str = "1.0"):
        self.name = name
        self.description = description
        self.version = version
        self.logger = logger.bind(agent=name, component="agent", version=version)
        self._result: Optional[AgentResult] = None

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
        self.logger.info(f"[{self.name}] Starting execution")

    def after_execute(self, result: AgentResult) -> None:
        """Hook po execute()."""
        self.logger.info(
            f"[{self.name}] Execution completed "
            f"(status: {result.status}, time: {result.execution_time:.3f}s)"
        )

    # --- Public entrypoint with lifecycle & error handling ---
    def run(self, **kwargs) -> AgentResult:
        """
        Main entry point – zarządza pełnym cyklem życia agenta i obsługą wyjątków.

        Zwraca:
            AgentResult – zawsze (także w wypadku wyjątku).
        """
        start_perf = time.perf_counter()
        started_at = datetime.now()
        try:
            # 1) Walidacja
            self.validate_input(**kwargs)

            # 2) Hook "before"
            self.before_execute(**kwargs)

            # 3) Właściwa praca
            result = self.execute(**kwargs)
            if not isinstance(result, AgentResult):
                # Defensywnie: jeśli ktoś zwrócił inny typ
                result = AgentResult(
                    agent_name=self.name,
                    status="failed",
                    errors=[f"Invalid result type returned by {self.name}"],
                )

            # 4) Pomiar czasu + znaczniki
            result.execution_time = time.perf_counter() - start_perf
            result.started_at = started_at
            result.finished_at = datetime.now()
            if not result.timestamp:
                result.timestamp = result.finished_at  # spójność

            # 5) Zachowaj & Hook "after"
            self._result = result
            self.after_execute(result)
            return result

        except Exception as e:
            # Defensywna obsługa błędu – zawsze zwróć AgentResult
            self.logger.error(f"[{self.name}] Execution failed: {e}", exc_info=True)
            failed = AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=time.perf_counter() - start_perf,
                started_at=started_at,
                finished_at=datetime.now(),
            )
            failed.add_error(str(e))
            self._result = failed
            # Warto mimo błędu wywołać after_execute dla ujednolicenia logów
            try:
                self.after_execute(failed)
            except Exception:
                # nie blokuj zwrotu wyniku, jeśli after_execute ma błąd
                pass
            return failed

    # --- Utils ---
    def get_last_result(self) -> Optional[AgentResult]:
        return self._result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"


# === PIPELINE AGENT ===========================================================
class PipelineAgent(BaseAgent):
    """
    Agent orkiestrujący wiele pod–agentów sekwencyjnie.

    Parametry:
        stop_on_warning: jeśli True, przerwij pipeline na statusie 'partial'.
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
        """Execute all agents in sequence."""
        result = AgentResult(agent_name=self.name)
        pipeline_results: List[AgentResult] = []
        results_index: Dict[str, AgentResult] = {}

        # Zachowujemy oryginalne wejście
        carry = dict(kwargs)

        for agent in self.agents:
            self.logger.info(f"[Pipeline: {self.name}] Executing {agent.name}")

            agent_result = agent.run(**carry)
            pipeline_results.append(agent_result)
            results_index[agent.name] = agent_result

            # Aktualizuj 'carry' danymi z poprzedniego kroku
            if agent_result.data:
                carry.update(agent_result.data)

            # Kontrola statusu
            if agent_result.is_failed():
                result.add_error(
                    f"Agent {agent.name} failed: {', '.join(agent_result.errors) or 'unknown error'}"
                )
                break
            if self.stop_on_warning and agent_result.is_partial():
                result.add_warning(
                    f"Agent {agent.name} returned warnings – pipeline stopped"
                )
                break

        # Ustal status końcowy
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


# === PARALLEL AGENT ===========================================================
class ParallelAgent(BaseAgent):
    """
    Agent uruchamiający wiele pod–agentów równolegle (wątkowo).
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
        """Execute all agents in parallel."""
        result = AgentResult(agent_name=self.name)
        parallel_results: List[AgentResult] = []
        results_index: Dict[str, AgentResult] = {}

        workers = min(len(self.agents), self.max_workers or len(self.agents))
        if workers <= 0:
            result.add_error("No agents to execute")
            return result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(agent.run, **kwargs): agent for agent in self.agents}

            for future in as_completed(futures):
                agent = futures[future]
                try:
                    agent_result = future.result()
                    parallel_results.append(agent_result)
                    results_index[agent.name] = agent_result

                    if not agent_result.is_success():
                        result.add_warning(
                            f"Agent {agent.name} status: {agent_result.status} "
                            f"({'; '.join(agent_result.errors) if agent_result.errors else 'no errors listed'})"
                        )
                except Exception as e:
                    result.add_error(f"Agent {agent.name} raised exception: {e}")

        # Ustal status zbiorczy
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
