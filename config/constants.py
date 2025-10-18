# core/base_agent.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Base Agent v7.0                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE AGENT FRAMEWORK WITH ENTERPRISE FEATURES                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Abstract Base Agent Class                                             ║
║  ✓ Retry with Exponential Backoff                                        ║
║  ✓ Circuit Breaker Pattern                                               ║
║  ✓ Rate Limiting (Token Bucket)                                          ║
║  ✓ Soft Timeouts                                                         ║
║  ✓ Distributed Tracing                                                   ║
║  ✓ Metrics & Audit Hooks                                                 ║
║  ✓ Sync & Async Execution                                                ║
║  ✓ Pipeline Orchestration                                                ║
║  ✓ Parallel Execution                                                    ║
║  ✓ Safe JSON Serialization                                               ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Agent Hierarchy:
```
    BaseAgent (Abstract)
    ├── Retry Logic
    ├── Circuit Breaker
    ├── Rate Limiter
    ├── Timeout Control
    ├── Tracing
    └── Telemetry
    
    PipelineAgent (Sequential)
    └── Orchestrates multiple agents in sequence
    
    ParallelAgent (Concurrent)
    └── Executes multiple agents in parallel
```

Features:
    Resilience:
        • Retry with exponential backoff + jitter
        • Circuit breaker (open/half-open/closed)
        • Rate limiting (token bucket)
        • Soft timeouts
    
    Observability:
        • Distributed tracing (trace_id, parent_trace_id)
        • Correlation IDs
        • Metrics hooks
        • Audit hooks
        • Structured logging
    
    Execution:
        • Sync and async execution
        • Lifecycle hooks (before/after)
        • Input validation
        • Safe serialization
    
    Orchestration:
        • Sequential pipelines
        • Parallel execution
        • Result aggregation
        • Error propagation

Usage:
```python
    from core.base_agent import BaseAgent, AgentResult
    
    class MyAgent(BaseAgent):
        def __init__(self):
            super().__init__(
                name="my_agent",
                description="Example agent",
                version="1.0"
            )
            # Configure resilience
            self.max_retries = 3
            self.backoff_base = 0.5
            self.timeout_sec = 30
            self.enable_circuit_breaker = True
        
        def execute(self, **kwargs) -> AgentResult:
            result = AgentResult(agent_name=self.name)
            
            # Your logic here
            data = kwargs.get("data")
            processed = self.process(data)
            
            result.add_data(output=processed)
            return result
    
    # Execute
    agent = MyAgent()
    result = agent.run(data=input_data)
    
    if result.is_success():
        print(f"Success: {result.data}")
    else:
        print(f"Failed: {result.errors}")
```

Dependencies:
    • loguru
    • pydantic
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = [
    "BaseAgent",
    "AgentResult",
    "AgentStatus",
    "PipelineAgent",
    "ParallelAgent"
]


# ═══════════════════════════════════════════════════════════════════════════
# Type Definitions
# ═══════════════════════════════════════════════════════════════════════════

AgentStatus = Literal["success", "failed", "partial"]


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _safe_json_default(obj: Any) -> Any:
    """
    Safe JSON serialization for complex types.
    
    Handles: numpy, pandas, datetime, and other non-serializable types.
    """
    # Try numpy/pandas
    try:
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
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
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def _json_dumps(obj: Any) -> str:
    """JSON dumps with safe serialization."""
    return json.dumps(obj, default=_safe_json_default, ensure_ascii=False)


def _uuid4_hex() -> str:
    """Generate UUID4 hex string."""
    return uuid4().hex


def _exp_backoff(
    base: float,
    attempt: int,
    jitter: float = 0.25,
    cap: Optional[float] = None
) -> float:
    """
    Calculate exponential backoff with jitter.
    
    Args:
        base: Base delay in seconds
        attempt: Attempt number (1-indexed)
        jitter: Jitter fraction (0.0 to 1.0)
        cap: Maximum delay cap
    
    Returns:
        Delay in seconds
    """
    delay = base * (2 ** max(0, attempt - 1))
    
    if jitter:
        delay *= (1.0 - jitter) + 2 * jitter * random.random()
    
    if cap:
        delay = min(delay, cap)
    
    return delay


@contextmanager
def _deadline(timeout: Optional[float]):
    """
    Soft deadline context manager.
    
    Sets a flag when timeout expires but doesn't interrupt execution.
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


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _BreakerState:
    """Circuit breaker internal state."""
    failures: int = 0
    last_failure_ts: float = 0.0
    open_until_ts: float = 0.0
    half_open: bool = False


class _CircuitBreaker:
    """
    🔌 **Circuit Breaker**
    
    Implements circuit breaker pattern to prevent cascading failures.
    
    States:
      • Closed: Normal operation
      • Open: Reject all requests
      • Half-Open: Allow limited requests to test recovery
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        open_seconds: float = 30.0,
        half_open_max_calls: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.open_seconds = open_seconds
        self.half_open_max_calls = half_open_max_calls
        
        self._state = _BreakerState()
        self._lock = threading.Lock()
        self._half_open_calls = 0
    
    def can_pass(self) -> bool:
        """Check if request can pass through breaker."""
        with self._lock:
            now = time.time()
            
            # Circuit open?
            if self._state.open_until_ts > now:
                return False
            
            # Transition to half-open
            if (self._state.open_until_ts <= now and 
                self._state.failures >= self.failure_threshold):
                self._state.half_open = True
                self._half_open_calls = 0
            
            return True
    
    def try_half_open_ticket(self) -> bool:
        """Try to get ticket for half-open state."""
        with self._lock:
            if not self._state.half_open:
                return True
            
            if self._half_open_calls >= self.half_open_max_calls:
                return False
            
            self._half_open_calls += 1
            return True
    
    def on_success(self) -> None:
        """Record successful execution."""
        with self._lock:
            self._state.failures = 0
            self._state.half_open = False
            self._state.open_until_ts = 0.0
            self._half_open_calls = 0
    
    def on_failure(self) -> None:
        """Record failed execution."""
        with self._lock:
            now = time.time()
            self._state.failures += 1
            self._state.last_failure_ts = now
            
            if self._state.failures >= self.failure_threshold:
                self._state.open_until_ts = now + self.open_seconds
                self._state.half_open = False


# ═══════════════════════════════════════════════════════════════════════════
# Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════

class _TokenBucket:
    """
    🪣 **Token Bucket Rate Limiter**
    
    Implements token bucket algorithm for rate limiting.
    """
    
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate = float(rate_per_sec)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self.last = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if tokens available, False otherwise
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            
            # Refill tokens
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            
            # Consume if available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Agent Result
# ═══════════════════════════════════════════════════════════════════════════

class AgentResult(BaseModel):
    """
    📊 **Agent Execution Result**
    
    Standard result format for all agents with enterprise features.
    
    Attributes:
        agent_name: Name of the agent
        status: Execution status (success/failed/partial)
        execution_time: Duration in seconds
        timestamp: Result timestamp
        started_at: Start timestamp
        finished_at: Finish timestamp
        trace_id: Unique trace identifier
        parent_trace_id: Parent trace for distributed tracing
        correlation_id: Correlation ID for request tracking
        data: Result data dictionary
        metadata: Additional metadata
        errors: List of error messages
        warnings: List of warning messages
    
    Example:
```python
        result = AgentResult(agent_name="my_agent")
        result.add_data(output="processed data")
        result.add_metadata(rows_processed=1000)
        
        if has_warning:
            result.add_warning("Minor issue detected")
        
        return result
```
    """
    
    agent_name: str
    status: AgentStatus = Field(default="success")
    
    # Timing
    execution_time: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    
    # Tracing
    trace_id: str = Field(default_factory=_uuid4_hex)
    parent_trace_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
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
        """Convert to JSON string with safe serialization."""
        return _json_dumps(self.model_dump())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


# ═══════════════════════════════════════════════════════════════════════════
# Base Agent
# ═══════════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """
    🤖 **Base Agent Class (Enterprise)**
    
    Abstract base class for all agents with enterprise features.
    
    Features:
      • Retry with exponential backoff
      • Circuit breaker pattern
      • Rate limiting
      • Soft timeouts
      • Distributed tracing
      • Metrics & audit hooks
      • Sync & async execution
    
    Configuration:
```python
        class MyAgent(BaseAgent):
            def __init__(self):
                super().__init__("my_agent", "My custom agent")
                
                # Retry config
                self.max_retries = 3
                self.backoff_base = 0.5
                self.backoff_cap = 10.0
                self.retry_on = (RuntimeError, ConnectionError)
                
                # Timeout
                self.timeout_sec = 30.0
                
                # Circuit breaker
                self.enable_circuit_breaker = True
                self.breaker_failure_threshold = 5
                self.breaker_open_seconds = 30.0
                
                # Rate limiting
                self.rate_limit_rps = 10.0
                self.rate_limit_capacity = 20
                
                # Telemetry
                self.metrics_callback = my_metrics_func
                self.audit_callback = my_audit_func
```
    
    Usage:
```python
        agent = MyAgent()
        
        # Synchronous
        result = agent.run(data=input_data)
        
        # Asynchronous
        result = await agent.run_async(data=input_data)
        
        # With tracing
        result = agent.run(
            parent_trace_id="parent-123",
            correlation_id="corr-456",
            data=input_data
        )
```
    """
    
    # Default resilience config
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
    
    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0"
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            description: Agent description
            version: Agent version
        """
        self.name = name
        self.description = description
        self.version = version
        
        self.logger = logger.bind(
            agent=name,
            component="agent",
            version=version
        )
        
        # Initialize circuit breaker
        self._breaker = (
            _CircuitBreaker(
                failure_threshold=self.breaker_failure_threshold,
                open_seconds=self.breaker_open_seconds,
                half_open_max_calls=self.breaker_half_open_max_calls
            )
            if self.enable_circuit_breaker
            else None
        )
        
        # Initialize rate limiter
        self._limiter = (
            _TokenBucket(
                rate_per_sec=self.rate_limit_rps,
                capacity=self.rate_limit_capacity
            )
            if self.rate_limit_rps and self.rate_limit_rps > 0
            else None
        )
        
        self._result: Optional[AgentResult] = None
    
    # ───────────────────────────────────────────────────────────────────
    # Abstract Methods
    # ───────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def execute(self, **kwargs) -> AgentResult:
        """
        Execute agent logic (synchronous).
        
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Agent-specific arguments
        
        Returns:
            AgentResult
        """
        raise NotImplementedError
    
    async def execute_async(self, **kwargs) -> AgentResult:
        """
        Execute agent logic (asynchronous).
        
        Default implementation runs execute() in thread pool.
        Override for true async execution.
        
        Args:
            **kwargs: Agent-specific arguments
        
        Returns:
            AgentResult
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute, **kwargs)
    
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
        self.logger.info(f"[{self.name}] Starting execution")
    
    def after_execute(self, result: AgentResult) -> None:
        """
        Hook called after execution.
        
        Override to add post-execution logic.
        
        Args:
            result: Execution result
        """
        self.logger.info(
            f"[{self.name}] Execution completed: "
            f"status={result.status}, time={result.execution_time:.3f}s"
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def run(
        self,
        *,
        parent_trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        🚀 **Execute Agent (Synchronous)**
        
        Main entry point for synchronous execution.
        Handles retry, timeout, circuit breaker, rate limiting.
        
        Args:
            parent_trace_id: Parent trace ID for distributed tracing
            correlation_id: Correlation ID for request tracking
            **kwargs: Agent-specific arguments
        
        Returns:
            AgentResult
        
        Example:
```python
            result = agent.run(
                parent_trace_id="parent-123",
                data=input_data
            )
```
        """
        return asyncio.run(
            self.run_async(
                parent_trace_id=parent_trace_id,
                correlation_id=correlation_id,
                **kwargs
            )
        )
    
    async def run_async(
        self,
        *,
        parent_trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """
        🚀 **Execute Agent (Asynchronous)**
        
        Main entry point for asynchronous execution.
        Handles retry, timeout, circuit breaker, rate limiting.
        
        Args:
            parent_trace_id: Parent trace ID for distributed tracing
            correlation_id: Correlation ID for request tracking
            **kwargs: Agent-specific arguments
        
        Returns:
            AgentResult
        
        Example:
```python
            result = await agent.run_async(
                parent_trace_id="parent-123",
                data=input_data
            )
```
        """
        start_perf = time.perf_counter()
        started_at = datetime.now()
        trace_id = _uuid4_hex()
        
        # Rate limiting
        if self._limiter and not self._limiter.consume():
            msg = f"[{self.name}] Rate limit exceeded"
            self.logger.warning(msg)
            
            return AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=0.0,
                started_at=started_at,
                finished_at=datetime.now(),
                parent_trace_id=parent_trace_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
                errors=[msg]
            )
        
        # Circuit breaker
        if self._breaker and not self._breaker.can_pass():
            msg = f"[{self.name}] Circuit open - rejecting execution"
            self.logger.error(msg)
            
            return AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=0.0,
                started_at=started_at,
                finished_at=datetime.now(),
                parent_trace_id=parent_trace_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
                errors=[msg]
            )
        
        # Validate input
        try:
            self.validate_input(**kwargs)
            self.before_execute(**kwargs)
        except Exception as e:
            self.logger.error(
                f"[{self.name}] Pre-execution failed: {e}",
                exc_info=True
            )
            
            failed = AgentResult(
                agent_name=self.name,
                status="failed",
                execution_time=time.perf_counter() - start_perf,
                started_at=started_at,
                finished_at=datetime.now(),
                parent_trace_id=parent_trace_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
                errors=[str(e)]
            )
            
            self._emit_metrics("agent_pre_fail", failed)
            return failed
        
        # Retry loop
        attempts = max(1, self.max_retries + 1)
        last_exc: Optional[BaseException] = None
        result: Optional[AgentResult] = None
        
        for attempt in range(1, attempts + 1):
            try:
                # Execute with soft timeout
                with _deadline(self.timeout_sec) as dl:
                    res = await self.execute_async(**kwargs)
                    
                    # Check timeout
                    if dl["expired"]:
                        res.add_warning("Execution exceeded soft timeout")
                        self.logger.warning(f"[{self.name}] Soft timeout reached")
                
                # Validate result type
                if not isinstance(res, AgentResult):
                    raise TypeError(
                        f"Invalid result type: expected AgentResult, "
                        f"got {type(res).__name__}"
                    )
                
                # Enrich result
                res.started_at = res.started_at or started_at
                res.finished_at = datetime.now()
                res.execution_time = time.perf_counter() - start_perf
                res.parent_trace_id = parent_trace_id or res.parent_trace_id
                res.correlation_id = correlation_id or res.correlation_id
                res.trace_id = res.trace_id or trace_id
                
                result = res
                
                # Circuit breaker: success
                if self._breaker:
                    self._breaker.on_success()
                
                # After hook
                self._result = result
                try:
                    self.after_execute(result)
                finally:
                    self._emit_metrics("agent_success", result)
                    self._emit_audit("agent_success", result)
                
                return result
            
            except Exception as e:
                last_exc = e
                
                # Circuit breaker: failure
                if self._breaker:
                    self._breaker.on_failure()
                
                # Should retry?
                will_retry = (
                    attempt < attempts and
                    isinstance(e, self.retry_on)
                )
                
                self.logger.error(
                    f"[{self.name}] Attempt {attempt}/{attempts} failed: {e} "
                    f"{'(retrying...)' if will_retry else '(no retry)'}",
                    exc_info=True
                )
                
                if will_retry:
                    delay = _exp_backoff(
                        self.backoff_base,
                        attempt,
                        jitter=0.25,
                        cap=self.backoff_cap
                    )
                    await asyncio.sleep(delay)
                else:
                    break
        
        # Final failure
        failed = AgentResult(
            agent_name=self.name,
            status="failed",
            execution_time=time.perf_counter() - start_perf,
            started_at=started_at,
            finished_at=datetime.now(),
            parent_trace_id=parent_trace_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            errors=[str(last_exc)] if last_exc else []
        )
        
        self._result = failed
        try:
            self.after_execute(failed)
        finally:
            self._emit_metrics("agent_fail", failed)
            self._emit_audit("agent_fail", failed)
        
        return failed
    
    # ───────────────────────────────────────────────────────────────────
    # Telemetry
    # ───────────────────────────────────────────────────────────────────
    
    def _emit_metrics(self, event: str, result: AgentResult) -> None:
        """Emit metrics event."""
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
                "warnings": len(result.warnings)
            })
        except Exception as e:
            self.logger.warning(f"[{self.name}] metrics_callback failed: {e}")
    
    def _emit_audit(self, event: str, result: AgentResult) -> None:
        """Emit audit event."""
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
                "finished_at": (result.finished_at or datetime.now()).isoformat()
            })
        except Exception as e:
            self.logger.warning(f"[{self.name}] audit_callback failed: {e}")
    
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
      • Context propagation between agents
      • Stop on warning option
      • Distributed tracing
      • Result aggregation
    
    Usage:
```python
        pipeline = PipelineAgent(
            name="ml_pipeline",
            agents=[
                DataLoaderAgent(),
                PreprocessingAgent(),
                TrainingAgent(),
                EvaluationAgent()
            ],
            stop_on_warning=False
        )
        
        result = pipeline.run(input_data=data)
        
        # Access individual results
        for step_result in result.data["pipeline_results"]:
            print(f"{step_result.agent_name}: {step_result.status}")
        
        # Access aggregated data
        final_data = result.data["final_data"]
```
    """
    
    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        description: str = "",
        stop_on_warning: bool = False
    ):
        """
        Initialize pipeline agent.
        
        Args:
            name: Pipeline name
            agents: List of agents to execute sequentially
            description: Pipeline description
            stop_on_warning: Stop pipeline on first warning
        """
        super().__init__(name, description)
        self.agents = agents
        self.stop_on_warning = stop_on_warning
    
    def execute(self, **kwargs) -> AgentResult:
        """Execute pipeline (sync wrapper)."""
        return asyncio.run(self.execute_async(**kwargs))
    
    async def execute_async(self, **kwargs) -> AgentResult:
        """
        Execute pipeline asynchronously.
        
        Args:
            **kwargs: Initial context
        
        Returns:
            AgentResult with aggregated results
        """
        result = AgentResult(agent_name=self.name)
        pipeline_results: List[AgentResult] = []
        results_index: Dict[str, AgentResult] = {}
        
        # Carry context between agents
        carry = dict(kwargs)
        parent_trace = _uuid4_hex()
        started = time.perf_counter()
        
        # Execute agents sequentially
        for agent in self.agents:
            self.logger.info(f"[Pipeline: {self.name}] Executing {agent.name}")
            
            # Execute agent
            agent_result = await agent.run_async(
                parent_trace_id=parent_trace,
                correlation_id=carry.get("correlation_id"),
                **carry
            )
            
            pipeline_results.append(agent_result)
            results_index[agent.name] = agent_result
            
            # Propagate data to next agent
            if agent_result.data:
                carry.update(agent_result.data)
            
            # Check for failure
            if agent_result.is_failed():
                result.add_error(
                    f"Agent {agent.name} failed: "
                    f"{', '.join(agent_result.errors) or 'unknown error'}"
                )
                break
            
            # Check for warnings
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
        
        result.execution_time = time.perf_counter() - started
        
        # Add results
        result.add_data(
            pipeline_results=pipeline_results,
            agents_executed=len(pipeline_results),
            final_data=carry,
            results_index=results_index
        )
        
        result.add_metadata(
            total_agents=len(self.agents),
            successful_agents=sum(1 for r in pipeline_results if r.is_success()),
            failed_agents=sum(1 for r in pipeline_results if r.is_failed()),
            partial_agents=sum(1 for r in pipeline_results if r.is_partial())
        )
        
        return result


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
      • Result aggregation
      • Error collection
      • Distributed tracing
    
    Usage:
```python
        parallel = ParallelAgent(
            name="parallel_processing",
            agents=[
                FeatureEngineering1(),
                FeatureEngineering2(),
                FeatureEngineering3()
            ],
            max_workers=3
        )
        
        result = parallel.run(data=input_data)
        
        # Access individual results
        for agent_result in result.data["parallel_results"]:
            print(f"{agent_result.agent_name}: {agent_result.status}")
        
        # Check for failures
        if result.is_failed():
            for error in result.errors:
                print(f"Error: {error}")
```
    """
    
    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        description: str = "",
        max_workers: Optional[int] = None
    ):
        """
        Initialize parallel agent.
        
        Args:
            name: Parallel agent name
            agents: List of agents to execute in parallel
            description: Agent description
            max_workers: Maximum thread pool workers (None = len(agents))
        """
        super().__init__(name, description)
        self.agents = agents
        self.max_workers = max_workers
    
    def execute(self, **kwargs) -> AgentResult:
        """
        Execute agents in parallel.
        
        Args:
            **kwargs: Context for all agents
        
        Returns:
            AgentResult with aggregated results
        """
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
        
        # Propagate correlation ID
        correlation_id = kwargs.get("correlation_id")
        
        def _runner(agent: BaseAgent) -> AgentResult:
            """Sync bridge to run_async."""
            return asyncio.run(
                agent.run_async(
                    correlation_id=correlation_id,
                    **kwargs
                )
            )
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_runner, agent): agent
                for agent in self.agents
            }
            
            for future in as_completed(futures):
                agent = futures[future]
                
                try:
                    agent_result = future.result()
                    parallel_results.append(agent_result)
                    results_index[agent.name] = agent_result
                    
                    if not agent_result.is_success():
                        result.add_warning(
                            f"Agent {agent.name} status: {agent_result.status} "
                            f"({'; '.join(agent_result.errors) if agent_result.errors else 'no errors'})"
                        )
                
                except Exception as e:
                    result.add_error(f"Agent {agent.name} raised exception: {e}")
        
        # Determine overall status
        if all(r.is_success() for r in parallel_results) and not result.errors:
            result.status = "success"
        elif any(r.is_success() for r in parallel_results):
            result.status = "partial"
        else:
            result.status = "failed"
        
        # Add results
        result.add_data(
            parallel_results=parallel_results,
            agents_executed=len(parallel_results),
            results_index=results_index
        )
        
        result.add_metadata(
            total_agents=len(self.agents),
            successful_agents=sum(1 for r in parallel_results if r.is_success()),
            failed_agents=sum(1 for r in parallel_results if r.is_failed()),
            partial_agents=sum(1 for r in parallel_results if r.is_partial()),
            max_workers=workers
        )
        
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Base Agent v{__version__} - Self Test")
    print("="*80)
    
    # Test agent
    class TestAgent(BaseAgent):
        def __init__(self, name: str, should_fail: bool = False):
            super().__init__(name, f"Test agent: {name}")
            self.should_fail = should_fail
            self.max_retries = 2
        
        def execute(self, **kwargs) -> AgentResult:
            result = AgentResult(agent_name=self.name)
            
            if self.should_fail:
                result.add_error("Simulated failure")
            else:
                result.add_data(
                    output=f"Processed by {self.name}",
                    input_data=kwargs.get("data")
                )
            
            return result
    
    # Test basic execution
    print("\n1. Testing Basic Execution...")
    agent = TestAgent("test_agent_1")
    result = agent.run(data="test_data")
    
    print(f"   Status: {result.status}")
    print(f"   Duration: {result.execution_time:.3f}s")
    print(f"   Trace ID: {result.trace_id}")
    
    # Test failure
    print("\n2. Testing Failure Handling...")
    failing_agent = TestAgent("test_agent_2", should_fail=True)
    result = failing_agent.run(data="test_data")
    
    print(f"   Status: {result.status}")
    print(f"   Errors: {result.errors}")
    
    # Test pipeline
    print("\n3. Testing Pipeline...")
    pipeline = PipelineAgent(
        name="test_pipeline",
        agents=[
            TestAgent("step_1"),
            TestAgent("step_2"),
            TestAgent("step_3")
        ]
    )
    
    result = pipeline.run(data="pipeline_data")
    
    print(f"   Status: {result.status}")
    print(f"   Agents executed: {result.data['agents_executed']}")
    print(f"   Duration: {result.execution_time:.3f}s")
    
    # Test parallel
    print("\n4. Testing Parallel Execution...")
    parallel = ParallelAgent(
        name="test_parallel",
        agents=[
            TestAgent("parallel_1"),
            TestAgent("parallel_2"),
            TestAgent("parallel_3")
        ],
        max_workers=3
    )
    
    result = parallel.run(data="parallel_data")
    
    print(f"   Status: {result.status}")
    print(f"   Agents executed: {result.data['agents_executed']}")
    print(f"   Duration: {result.execution_time:.3f}s")
    
    # Test serialization
    print("\n5. Testing Serialization...")
    try:
        json_str = result.to_json()
        print(f"   ✓ JSON serialization successful ({len(json_str)} chars)")
    except Exception as e:
        print(f"   ✗ JSON serialization failed: {e}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from core.base_agent import BaseAgent, AgentResult

# === Custom Agent ===
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="my_agent",
            description="Custom processing agent",
            version="1.0"
        )
        
        # Configure resilience
        self.max_retries = 3
        self.backoff_base = 0.5
        self.timeout_sec = 30
        self.enable_circuit_breaker = True
        self.rate_limit_rps = 10.0
    
    def execute(self, **kwargs) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        
        # Your logic
        data = kwargs.get("data")
        processed = self.process(data)
        
        result.add_data(output=processed)
        result.add_metadata(rows_processed=len(data))
        
        return result

# === Execution ===
agent = MyAgent()

# Synchronous
result = agent.run(data=input_data)

# Asynchronous
result = await agent.run_async(data=input_data)

# With tracing
result = agent.run(
    parent_trace_id="parent-123",
    correlation_id="corr-456",
    data=input_data
)

# === Pipeline ===
from core.base_agent import PipelineAgent

pipeline = PipelineAgent(
    name="ml_pipeline",
    agents=[
        LoaderAgent(),
        ProcessorAgent(),
        TrainerAgent()
    ]
)

result = pipeline.run(file_path="data.csv")

# === Parallel ===
from core.base_agent import ParallelAgent

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
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)
