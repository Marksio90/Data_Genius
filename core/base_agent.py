# core/base_agent.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Base Agent v7.0 (Unified)        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE AGENT FRAMEWORK WITH ENTERPRISE FEATURES                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Abstract Base Agent Class                                             ║
║  ✓ Retry with Exponential Backoff + Jitter                               ║
║  ✓ Timeout Support (Thread-based)                                        ║
║  ✓ Progress Callbacks                                                    ║
║  ✓ Lifecycle Hooks (before/after)                                        ║
║  ✓ Safe JSON Serialization                                               ║
║  ✓ Pipeline Orchestration                                                ║
║  ✓ Parallel Execution                                                    ║
║  ✓ Result Merging                                                        ║
╚════════════════════════════════════════════════════════════════════════════╝

Features:
    Core Framework:
        • Abstract base class for all agents
        • Lifecycle management (validate → before → execute → after)
        • Retry with exponential backoff + jitter
        • Optional timeout support
        • Progress callbacks
        • Structured logging
    
    Resilience:
        • Configurable retry logic
        • Exponential backoff with jitter
        • Thread-based timeout
        • Graceful error handling
        • Status tracking
    
    Result Management:
        • Standard AgentResult format
        • Safe JSON serialization (numpy/pandas/datetime)
        • Result merging
        • Status helpers
        • Metadata tracking
    
    Orchestration:
        • Sequential pipelines (PipelineAgent)
        • Parallel execution (ParallelAgent)
        • Result aggregation
        • Error propagation

Usage:
```python
    from core.base_agent import BaseAgent, AgentResult
    
    class MyAgent(BaseAgent):
        def __init__(self):
            super().__init__(
                name="my_agent",
                description="Custom agent",
                version="1.0",
                retries=2,
                retry_backoff=1.6,
                timeout_sec=30
            )
        
        def execute(self, **kwargs) -> AgentResult:
            result = AgentResult(agent_name=self.name)
            
            # Your logic
            data = kwargs.get("data")
            processed = self.process(data)
            
            result.add_data(output=processed)
            return result
    
    # Execute with automatic retry/timeout
    agent = MyAgent()
    result = agent.run(data=input_data)
```

Dependencies:
    • loguru
    • pydantic
    • numpy (optional, for serialization)
    • pandas (optional, for serialization)
"""

from __future__ import annotations

import json
import math
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate-unified"
__author__ = "DataGenius Enterprise Team"

__all__ = [
    "BaseAgent",
    "AgentResult",
    "AgentStatus",
    "AgentError",
    "PipelineAgent",
    "ParallelAgent"
]


# ═══════════════════════════════════════════════════════════════════════════
# Type Definitions
# ═══════════════════════════════════════════════════════════════════════════

AgentStatus = Literal["success", "failed", "partial"]


# ═══════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class AgentError(RuntimeError):
    """Base exception for agent-related errors."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Agent Result
# ═══════════════════════════════════════════════════════════════════════════

class AgentResult(BaseModel):
    """
    📊 **Agent Execution Result**
    
    Standard result format with safe serialization and helper methods.
    
    Attributes:
        agent_name: Name of the agent
        status: Execution status (success/failed/partial)
        execution_time: Duration in seconds
        timestamp: Result timestamp
        started_at: Start timestamp
        finished_at: Finish timestamp
        trace_id: Unique trace identifier
        data: Result data dictionary
        metadata: Additional metadata
        errors: List of error messages
        warnings: List of warning messages
    
    Features:
        • Status helpers (is_success, is_failed, is_partial)
        • Mutation helpers (add_error, add_warning, add_data)
        • Safe JSON serialization (numpy/pandas/datetime)
        • Result merging
    """
    
    agent_name: str
    status: AgentStatus = Field(default="success")
    
    # Timing
    execution_time: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    
    # Tracing
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    
    # Payload
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # ───────────────────────────────────────────────────────────────────
    # Status Checks
    # ───────────────────────────────────────────────────────────────────
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == "success"
    
    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status == "failed"
    
    def is_partial(self) -> bool:
        """Check if execution partially succeeded."""
        return self.status == "partial"
    
    # ───────────────────────────────────────────────────────────────────
    # Mutations
    # ───────────────────────────────────────────────────────────────────
    
    def add_error(self, error: str) -> None:
        """Add error and mark as failed."""
        self.errors.append(error)
        if self.status == "success":
            self.status = "failed"
    
    def add_warning(self, warning: str) -> None:
        """Add warning and mark as partial if success."""
        self.warnings.append(warning)
        if self.status == "success":
            self.status = "partial"
    
    def add_data(self, **items: Any) -> None:
        """Add data items."""
        self.data.update(items)
    
    def add_metadata(self, **items: Any) -> None:
        """Add metadata items."""
        self.metadata.update(items)
    
    # ───────────────────────────────────────────────────────────────────
    # Serialization
    # ───────────────────────────────────────────────────────────────────
    
    def to_json(self) -> str:
        """
        Convert to JSON string with safe serialization.
        
        Handles: numpy, pandas, datetime objects
        """
        def _safe_default(obj: Any) -> Any:
            # Try numpy
            try:
                import numpy as np
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
            except ImportError:
                pass
            
            # Try pandas
            try:
                import pandas as pd
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                if isinstance(obj, pd.Series):
                    return obj.tolist()
                if isinstance(obj, pd.DataFrame):
                    return json.loads(obj.to_json(orient="records"))
            except ImportError:
                pass
            
            # Datetime
            if isinstance(obj, datetime):
                return obj.isoformat()
            
            # Fallback
            return str(obj)
        
        return json.dumps(
            self.model_dump(),
            default=_safe_default,
            ensure_ascii=False
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Result Merging
    # ───────────────────────────────────────────────────────────────────
    
    def merge(
        self,
        other: "AgentResult",
        *,
        prefix: Optional[str] = None
    ) -> "AgentResult":
        """
        Merge another result into this one.
        
        Args:
            other: Result to merge
            prefix: Optional prefix for merged data/metadata
        
        Returns:
            Self (for chaining)
        """
        p = f"{prefix}." if prefix else ""
        
        # Merge data and metadata
        self.add_data(**{f"{p}data": other.data})
        self.add_metadata(**{f"{p}meta": other.metadata})
        
        # Merge warnings and errors
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        
        # Update status
        if other.is_failed():
            self.status = "failed"
        elif other.is_partial() and self.status == "success":
            self.status = "partial"
        
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Base Agent
# ═══════════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """
    🤖 **Base Agent Class**
    
    Abstract base class for all agents with lifecycle management.
    
    Features:
      • Retry with exponential backoff + jitter
      • Optional timeout support
      • Progress callbacks
      • Lifecycle hooks (validate, before, after)
      • Structured logging
      • Safe error handling
    
    Configuration:
```python
        class MyAgent(BaseAgent):
            def __init__(self):
                super().__init__(
                    name="my_agent",
                    description="Custom agent",
                    version="1.0",
                    retries=2,              # Number of retries
                    retry_backoff=1.6,      # Exponential multiplier
                    retry_jitter=0.2,       # ±20% random jitter
                    timeout_sec=30,         # Optional timeout
                    on_progress=callback    # Progress callback
                )
```
    
    Lifecycle:
```
        run() → validate_input()
              → before_execute()
              → [retry loop]
                  → execute()
              → measure time
              → after_execute()
              → return AgentResult
```
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0",
        *,
        retries: int = 0,
        retry_backoff: float = 1.6,
        retry_jitter: float = 0.2,
        timeout_sec: Optional[float] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            description: Agent description
            version: Agent version
            retries: Number of retry attempts
            retry_backoff: Exponential backoff multiplier
            retry_jitter: Random jitter fraction (±)
            timeout_sec: Optional execution timeout
            on_progress: Optional progress callback
        """
        self.name = name
        self.description = description
        self.version = version
        
        self.logger = logger.bind(
            agent=name,
            component="agent",
            version=version
        )
        
        self._result: Optional[AgentResult] = None
        
        # Retry configuration
        self.retries = max(0, int(retries))
        self.retry_backoff = max(1.0, float(retry_backoff))
        self.retry_jitter = max(0.0, float(retry_jitter))
        self.timeout_sec = timeout_sec
        self.on_progress = on_progress
    
    # ───────────────────────────────────────────────────────────────────
    # Abstract Methods
    # ───────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def execute(self, **kwargs) -> AgentResult:
        """
        Execute agent logic.
        
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Agent-specific arguments
        
        Returns:
            AgentResult
        """
        raise NotImplementedError
    
    # ───────────────────────────────────────────────────────────────────
    # Lifecycle Hooks
    # ───────────────────────────────────────────────────────────────────
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input before execution.
        
        Override to add custom validation.
        
        Args:
            **kwargs: Input arguments
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If validation fails
        """
        return True
    
    def before_execute(self, **kwargs) -> None:
        """
        Hook called before execution.
        
        Override to add pre-execution logic.
        """
        self._emit_progress("start", extra={"kwargs_keys": list(kwargs.keys())})
        self.logger.info(f"[{self.name}] Starting execution")
    
    def after_execute(self, result: AgentResult) -> None:
        """
        Hook called after execution.
        
        Override to add post-execution logic.
        
        Args:
            result: Execution result
        """
        self._emit_progress(
            "end",
            extra={
                "status": result.status,
                "execution_time": round(result.execution_time, 3)
            }
        )
        self.logger.info(
            f"[{self.name}] Execution completed: "
            f"status={result.status}, time={result.execution_time:.3f}s"
        )
    
    def _emit_progress(
        self,
        event: str,
        *,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit progress event to callback."""
        if self.on_progress:
            try:
                self.on_progress({
                    "agent": self.name,
                    "event": event,
                    "ts": datetime.utcnow().isoformat(),
                    **(extra or {})
                })
            except Exception as e:
                self.logger.debug(f"on_progress callback failed: {e}")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def run(self, **kwargs) -> AgentResult:
        """
        🚀 **Execute Agent**
        
        Main entry point with full lifecycle management.
        
        Flow:
          1. Validate input
          2. Before hook
          3. Execute with retry/timeout
          4. Measure time
          5. After hook
          6. Return result
        
        Args:
            **kwargs: Agent-specific arguments
        
        Returns:
            AgentResult (always, even on failure)
        
        Example:
```python
            agent = MyAgent()
            result = agent.run(data=input_data)
            
            if result.is_success():
                print(result.data)
            else:
                print(result.errors)
```
        """
        start_perf = time.perf_counter()
        started_at = datetime.now()
        attempt = 0
        last_exc: Optional[BaseException] = None
        
        try:
            # Validate input
            self.validate_input(**kwargs)
            self.before_execute(**kwargs)
            
            # Retry loop
            max_attempts = self.retries + 1
            
            while attempt < max_attempts:
                attempt += 1
                
                try:
                    # Execute with optional timeout
                    result = self._execute_with_timeout(kwargs)
                    
                    # Validate result type
                    if not isinstance(result, AgentResult):
                        raise AgentError(
                            f"Invalid result type returned by {self.name}: "
                            f"expected AgentResult, got {type(result).__name__}"
                        )
                    
                    # Success - break retry loop
                    break
                
                except Exception as e:
                    last_exc = e
                    
                    # Check if should retry
                    if attempt >= max_attempts:
                        raise
                    
                    # Calculate backoff
                    sleep_for = self._calculate_backoff(attempt - 1)
                    
                    self.logger.warning(
                        f"[{self.name}] Attempt {attempt}/{max_attempts} failed: {e} "
                        f"— retry in {sleep_for:.2f}s"
                    )
                    
                    time.sleep(sleep_for)
            
            # Enrich result with timing
            result.execution_time = time.perf_counter() - start_perf
            result.started_at = started_at
            result.finished_at = datetime.now()
            
            if not result.timestamp:
                result.timestamp = result.finished_at
            
            # Store and call after hook
            self._result = result
            self.after_execute(result)
            
            return result
        
        except Exception as e:
            # Handle all failures
            self.logger.error(f"[{self.name}] Execution failed: {e}", exc_info=True)
            
            # Create failure result
            failed = AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=time.perf_counter() - start_perf,
                started_at=started_at,
                finished_at=datetime.now()
            )
            
            # Determine error message
            if isinstance(last_exc, FuturesTimeout) or isinstance(e, FuturesTimeout):
                msg = "TimeoutError: Agent execution exceeded time limit"
            else:
                msg = f"{type(e).__name__}: {str(e)}"
            
            failed.add_error(msg)
            
            # Store and try after hook
            self._result = failed
            
            try:
                self.after_execute(failed)
            except Exception:
                pass
            
            return failed
    
    # ───────────────────────────────────────────────────────────────────
    # Internals
    # ───────────────────────────────────────────────────────────────────
    
    def _execute_with_timeout(self, kwargs: Dict[str, Any]) -> AgentResult:
        """
        Execute with optional timeout.
        
        If timeout is set, runs in thread with timeout.
        Otherwise, runs directly.
        """
        if not self.timeout_sec or self.timeout_sec <= 0:
            return self.execute(**kwargs)
        
        # Thread-based timeout
        result_holder: Dict[str, Any] = {}
        exception_holder: Dict[str, BaseException] = {}
        done_event = threading.Event()
        
        def _runner():
            try:
                result_holder["result"] = self.execute(**kwargs)
            except BaseException as ex:
                exception_holder["exception"] = ex
            finally:
                done_event.set()
        
        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        
        finished = done_event.wait(timeout=self.timeout_sec)
        
        if not finished:
            raise FuturesTimeout()
        
        if "exception" in exception_holder:
            raise exception_holder["exception"]
        
        return result_holder["result"]
    
    def _calculate_backoff(self, retry_index: int) -> float:
        """
        Calculate backoff delay with exponential backoff + jitter.
        
        Args:
            retry_index: 0 for first retry, 1 for second, etc.
        
        Returns:
            Sleep duration in seconds
        """
        # Exponential backoff
        base = self.retry_backoff ** max(0, retry_index)
        
        # Add jitter (±retry_jitter * base)
        if self.retry_jitter > 0:
            import random
            jitter = (random.random() * 2 - 1) * self.retry_jitter * base
        else:
            jitter = 0.0
        
        return max(0.05, base + jitter)
    
    # ───────────────────────────────────────────────────────────────────
    # Utilities
    # ───────────────────────────────────────────────────────────────────
    
    def get_last_result(self) -> Optional[AgentResult]:
        """Get last execution result."""
        return self._result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Agent
# ═══════════════════════════════════════════════════════════════════════════

class PipelineAgent(BaseAgent):
    """
    🔄 **Pipeline Agent (Sequential Orchestration)**
    
    Executes multiple agents sequentially with result propagation.
    
    Features:
      • Sequential execution
      • Context propagation
      • Stop on warning option
      • Metadata propagation
      • Result aggregation
    
    Usage:
```python
        pipeline = PipelineAgent(
            name="ml_pipeline",
            agents=[
                LoaderAgent(),
                ProcessorAgent(),
                TrainerAgent()
            ],
            stop_on_warning=False,
            propagate_metadata=True
        )
        
        result = pipeline.run(file_path="data.csv")
```
    """
    
    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        description: str = "",
        stop_on_warning: bool = False,
        propagate_metadata: bool = True,
        **kwargs
    ):
        """
        Initialize pipeline agent.
        
        Args:
            name: Pipeline name
            agents: List of agents to execute sequentially
            description: Pipeline description
            stop_on_warning: Stop pipeline on first warning
            propagate_metadata: Propagate metadata between steps
            **kwargs: Additional BaseAgent arguments
        """
        super().__init__(name, description, **kwargs)
        self.agents = agents
        self.stop_on_warning = stop_on_warning
        self.propagate_metadata = propagate_metadata
    
    def execute(self, **kwargs) -> AgentResult:
        """Execute all agents sequentially."""
        result = AgentResult(agent_name=self.name)
        pipeline_results: List[AgentResult] = []
        results_index: Dict[str, AgentResult] = {}
        
        # Carry context between steps
        carry = dict(kwargs)
        
        for agent in self.agents:
            step_label = f"{self.name}:{agent.name}"
            self.logger.info(f"[Pipeline] Executing {step_label}")
            
            # Execute agent
            agent_result = agent.run(**carry)
            pipeline_results.append(agent_result)
            results_index[agent.name] = agent_result
            
            # Propagate data
            if agent_result.data:
                carry.update(agent_result.data)
            
            # Propagate metadata (prefixed)
            if self.propagate_metadata and agent_result.metadata:
                carry.setdefault("_pipeline_meta", {})[agent.name] = agent_result.metadata
            
            # Check status
            if agent_result.is_failed():
                result.add_error(
                    f"Agent {agent.name} failed: "
                    f"{', '.join(agent_result.errors) or 'unknown error'}"
                )
                break
            
            if self.stop_on_warning and agent_result.is_partial():
                result.add_warning(
                    f"Agent {agent.name} returned warnings - pipeline stopped"
                )
                break
        
        # Determine overall status
        if any(r.is_failed() for r in pipeline_results):
            result.status = "failed"
        elif any(r.is_partial() for r in pipeline_results):
            result.status = "partial"
        else:
            result.status = "success"
        
        # Aggregate results
        result.add_data(
            pipeline_results=pipeline_results,
            agents_executed=len(pipeline_results),
            final_data=carry,
            results_index=results_index
        )
        
        return result
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add agent to pipeline."""
        self.agents.append(agent)
    
    def remove_agent(self, agent_name: str) -> bool:
        """Remove agent from pipeline by name."""
        for i, agent in enumerate(self.agents):
            if agent.name == agent_name:
                self.agents.pop(i)
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Parallel Agent
# ═══════════════════════════════════════════════════════════════════════════

class ParallelAgent(BaseAgent):
    """
    ⚡ **Parallel Agent (Concurrent Orchestration)**
    
    Executes multiple agents concurrently using thread pool.
    
    Features:
      • Parallel execution
      • Thread pool management
      • Timeout handling
      • Result aggregation
      • Error collection
    
    Usage:
```python
        parallel = ParallelAgent(
            name="feature_engineering",
            agents=[
                FeatureAgent1(),
                FeatureAgent2(),
                FeatureAgent3()
            ],
            max_workers=3
        )
        
        result = parallel.run(data=input_data)
```
    """
    
    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        description: str = "",
        max_workers: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize parallel agent.
        
        Args:
            name: Parallel agent name
            agents: List of agents to execute in parallel
            description: Agent description
            max_workers: Maximum thread pool workers
            **kwargs: Additional BaseAgent arguments
        """
        super().__init__(name, description, **kwargs)
        self.agents = agents
        self.max_workers = max_workers
    
    def execute(self, **kwargs) -> AgentResult:
        """Execute all agents in parallel."""
        result = AgentResult(agent_name=self.name)
        parallel_results: List[AgentResult] = []
        results_index: Dict[str, AgentResult] = {}
        
        workers = min(
            len(self.agents),
            self.max_workers or len(self.agents)
        )
        
        if workers <= 0:
            result.add_error("No agents to execute")
            return result
        
        # Emit progress
        self._emit_progress(
            "parallel_start",
            extra={"workers": workers, "n_agents": len(self.agents)}
        )
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(agent.run, **kwargs): agent
                for agent in self.agents
            }
            
            for future in as_completed(futures):
                agent = futures[future]
                
                try:
                    # Get result with optional timeout
                    timeout = getattr(agent, "timeout_sec", None)
                    agent_result = future.result(timeout=timeout)
                    
                    parallel_results.append(agent_result)
                    results_index[agent.name] = agent_result
                    
                    if not agent_result.is_success():
                        result.add_warning(
                            f"Agent {agent.name} status: {agent_result.status} "
                            f"({'; '.join(agent_result.errors) if agent_result.errors else 'no errors'})"
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
        
        # Determine overall status
        if all(r.is_success() for r in parallel_results) and not result.errors:
            result.status = "success"
        elif any(r.is_success() for r in parallel_results):
            result.status = "partial"
        else:
            result.status = "failed"
        
        # Aggregate results
        result.add_data(
            parallel_results=parallel_results,
            agents_executed=len(parallel_results),
            results_index=results_index
        )
        
        self._emit_progress("parallel_end", extra={"status": result.status})
        
        return result