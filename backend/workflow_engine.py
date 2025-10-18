# backend/workflow_engine.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Workflow Engine v7.0             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE DAG-BASED WORKFLOW ORCHESTRATION ENGINE                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ DAG Workflow Definition & Validation                                  ║
║  ✓ Task Registry & Custom Tasks                                          ║
║  ✓ Retry Logic with Exponential Backoff                                  ║
║  ✓ Soft Timeouts & Error Handling                                        ║
║  ✓ Event Hooks for Real-Time Monitoring                                  ║
║  ✓ State Persistence & Checkpointing                                     ║
║  ✓ Context & Artifact Management                                         ║
║  ✓ Built-in ML Pipeline Tasks                                            ║
║  ✓ Topological Execution Order                                           ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Workflow Definition (DAG):
```
        Task A
         ├─→ Task B
         │    └─→ Task D
         └─→ Task C
              └─→ Task D
```
    
    Execution Flow:
    1. Validate DAG (no cycles, valid dependencies)
    2. Topological sort (Kahn's algorithm)
    3. Execute tasks in order:
       • Check dependencies
       • Retry with exponential backoff
       • Update context & artifacts
       • Log events
       • Persist state
    4. Finalize workflow run

Features:
    Task Management:
        • Task registry system
        • Custom task registration
        • Built-in ML pipeline tasks
        • Parameter passing
    
    Execution:
        • Topological ordering
        • Dependency resolution
        • Retry with backoff
        • Soft timeouts
        • Continue-on-error mode
    
    State Management:
        • Run state persistence
        • Context sharing
        • Artifact tracking
        • Event logging
    
    Built-in Tasks:
        • pipeline_e2e: Complete ML pipeline
        • drift_check: Data drift detection
        • retrain_decision: Retraining logic
        • save_report_to_session: Report storage

Usage:
```python
    from backend.workflow_engine import (
        WorkflowEngine,
        WorkflowDefinition,
        TaskDefinition
    )
    
    # Create engine
    engine = WorkflowEngine()
    
    # Define workflow
    workflow = WorkflowDefinition(
        name="ml_pipeline",
        tasks=[
            TaskDefinition(
                name="train",
                func="pipeline_e2e",
                params={
                    "session_id": "abc123",
                    "dataset_name": "train_data"
                }
            ),
            TaskDefinition(
                name="check_drift",
                func="drift_check"
            ),
            TaskDefinition(
                name="decide_retrain",
                func="retrain_decision"
            )
        ],
        dependencies=[
            ("train", "check_drift"),
            ("check_drift", "decide_retrain")
        ]
    )
    
    # Execute
    run = engine.run(
        workflow,
        initial_context={"session_id": "abc123"}
    )
    
    print(f"Status: {run.status}")
    print(f"Artifacts: {run.artifacts}")
```

Dependencies:
    • pandas
    • loguru
    • backend.pipeline_executor
    • backend.session_manager
    • agents.monitoring.*
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = [
    "WorkflowEngine",
    "TaskRegistry",
    "WorkflowDefinition",
    "TaskDefinition",
    "WorkflowRun",
    "TaskRun",
    "TaskStatus"
]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

try:
    from config.settings import settings
except ImportError:
    logger.warning("⚠ config.settings not found - using defaults")
    
    class _FallbackSettings:
        WORKFLOWS_PATH = Path.cwd() / "workflows"
        WORKFLOW_MAX_RETRIES = 2
        WORKFLOW_BACKOFF_BASE = 1.8
        WORKFLOW_BACKOFF_MAX_SEC = 60.0
        WORKFLOW_TASK_SOFT_TIMEOUT_SEC = 3600
        WORKFLOW_CONTINUE_ON_ERROR = True
        LOG_JSON_INDENT = 2
    
    settings = _FallbackSettings()  # type: ignore


# Import components with fallback
try:
    from backend.pipeline_executor import PipelineExecutor, PipelineConfig, PipelineResult
except ImportError:
    logger.warning("⚠ PipelineExecutor not available")
    PipelineExecutor = None  # type: ignore
    PipelineConfig = None  # type: ignore
    PipelineResult = None  # type: ignore

try:
    from backend.session_manager import SessionManager
except ImportError:
    logger.warning("⚠ SessionManager not available")
    SessionManager = None  # type: ignore

try:
    from agents.monitoring.drift_detector import DriftDetector
except ImportError:
    logger.warning("⚠ DriftDetector not available")
    DriftDetector = None  # type: ignore

try:
    from agents.monitoring.retraining_scheduler import RetrainingScheduler
except ImportError:
    logger.warning("⚠ RetrainingScheduler not available")
    RetrainingScheduler = None  # type: ignore


# Constants
WORKFLOWS_PATH: Path = Path(getattr(settings, "WORKFLOWS_PATH", Path.cwd() / "workflows"))
JSON_INDENT: int = int(getattr(settings, "LOG_JSON_INDENT", 2))


# ═══════════════════════════════════════════════════════════════════════════
# Type Definitions
# ═══════════════════════════════════════════════════════════════════════════

TaskStatus = Literal["pending", "running", "success", "failed", "skipped"]
OnEvent = Optional[Callable[[Dict[str, Any]], None]]
TaskCallable = Callable[[Dict[str, Any]], Dict[str, Any]]


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _emit_event(callback: OnEvent, payload: Dict[str, Any]) -> None:
    """Emit event to callback."""
    if callback:
        try:
            callback(payload)
        except Exception as e:
            logger.warning(f"Event callback failed: {e}")


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _save_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Save JSON atomically."""
    _ensure_dir(path.parent)
    
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=JSON_INDENT),
        encoding="utf-8"
    )
    tmp.replace(path)


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TaskDefinition:
    """
    📋 **Task Definition**
    
    Defines a single task in the workflow.
    
    Attributes:
        name: Unique task name
        func: Registered function name
        params: Task parameters
        retry: Number of retry attempts
        soft_timeout_sec: Soft timeout in seconds
        continue_on_error: Continue workflow on task failure
    """
    
    name: str
    func: str
    params: Dict[str, Any] = field(default_factory=dict)
    retry: int = field(default_factory=lambda: int(getattr(settings, "WORKFLOW_MAX_RETRIES", 2)))
    soft_timeout_sec: int = field(default_factory=lambda: int(getattr(settings, "WORKFLOW_TASK_SOFT_TIMEOUT_SEC", 3600)))
    continue_on_error: bool = field(default_factory=lambda: bool(getattr(settings, "WORKFLOW_CONTINUE_ON_ERROR", True)))
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowDefinition:
    """
    🔄 **Workflow Definition (DAG)**
    
    Defines complete workflow with tasks and dependencies.
    
    Attributes:
        name: Workflow name
        tasks: List of task definitions
        dependencies: List of (from_task, to_task) edges
        continue_on_error: Global continue-on-error setting
    
    Example:
```python
        workflow = WorkflowDefinition(
            name="ml_pipeline",
            tasks=[
                TaskDefinition(name="load", func="load_data"),
                TaskDefinition(name="train", func="train_model"),
                TaskDefinition(name="eval", func="evaluate")
            ],
            dependencies=[
                ("load", "train"),
                ("train", "eval")
            ]
        )
```
    """
    
    name: str
    tasks: List[TaskDefinition]
    dependencies: List[Tuple[str, str]] = field(default_factory=list)
    continue_on_error: bool = field(default_factory=lambda: bool(getattr(settings, "WORKFLOW_CONTINUE_ON_ERROR", True)))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tasks": [t.to_dict() for t in self.tasks],
            "dependencies": self.dependencies,
            "continue_on_error": self.continue_on_error
        }


@dataclass
class TaskRun:
    """
    ▶️ **Task Execution Result**
    
    Tracks single task execution.
    """
    
    name: str
    func: str
    status: TaskStatus = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_sec: float = 0.0
    try_index: int = 0
    retries: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    output: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowRun:
    """
    🎯 **Workflow Execution Run**
    
    Complete workflow execution state.
    
    Attributes:
        run_id: Unique run identifier
        workflow_name: Workflow name
        started_at: Start timestamp
        finished_at: Finish timestamp
        duration_sec: Total duration
        status: Run status
        tasks: Task execution results
        context: Shared context
        artifacts: Generated artifacts
        event_log: Event history
    """
    
    run_id: str
    workflow_name: str
    started_at: str
    finished_at: Optional[str] = None
    duration_sec: float = 0.0
    status: Literal["running", "success", "failed"] = "running"
    tasks: Dict[str, TaskRun] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    event_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "status": self.status,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "context": self.context,
            "artifacts": self.artifacts,
            "event_log": self.event_log
        }
    
    def get_failed_tasks(self) -> List[TaskRun]:
        """Get all failed tasks."""
        return [t for t in self.tasks.values() if t.status == "failed"]
    
    def get_successful_tasks(self) -> List[TaskRun]:
        """Get all successful tasks."""
        return [t for t in self.tasks.values() if t.status == "success"]


# ═══════════════════════════════════════════════════════════════════════════
# Task Registry
# ═══════════════════════════════════════════════════════════════════════════

class TaskRegistry:
    """
    📚 **Task Registry**
    
    Registry for workflow task functions.
    
    Usage:
```python
        registry = TaskRegistry()
        
        @registry.register_function("my_task")
        def my_task(context: Dict[str, Any]) -> Dict[str, Any]:
            return {"output": "result"}
        
        # Or manually
        registry.register("task_name", task_function)
```
    """
    
    def __init__(self) -> None:
        self._registry: Dict[str, TaskCallable] = {}
        self.logger = logger.bind(component="TaskRegistry")
    
    def register(self, name: str, func: TaskCallable) -> None:
        """
        Register task function.
        
        Args:
            name: Task name
            func: Callable taking context dict and returning result dict
        """
        if not callable(func):
            raise ValueError("func must be callable")
        
        self._registry[name] = func
        self.logger.info(f"✓ Registered task: {name}")
    
    def register_function(self, name: str) -> Callable:
        """Decorator for registering functions."""
        def decorator(func: TaskCallable) -> TaskCallable:
            self.register(name, func)
            return func
        return decorator
    
    def get(self, name: str) -> TaskCallable:
        """Get registered task function."""
        if name not in self._registry:
            raise KeyError(f"Task '{name}' not registered")
        return self._registry[name]
    
    def has(self, name: str) -> bool:
        """Check if task is registered."""
        return name in self._registry
    
    def list_tasks(self) -> List[str]:
        """List all registered tasks."""
        return sorted(self._registry.keys())


# ═══════════════════════════════════════════════════════════════════════════
# DAG Validator
# ═══════════════════════════════════════════════════════════════════════════

class DAGValidator:
    """
    ✓ **DAG Validator**
    
    Validates workflow DAG structure.
    """
    
    @staticmethod
    def validate(definition: WorkflowDefinition) -> None:
        """
        Validate workflow definition.
        
        Checks:
          • No duplicate task names
          • All dependencies reference existing tasks
          • No cycles (using Kahn's algorithm)
        
        Args:
            definition: Workflow definition
        
        Raises:
            ValueError: If validation fails
        """
        task_names = {t.name for t in definition.tasks}
        
        # Check duplicates
        if len(task_names) != len(definition.tasks):
            raise ValueError("Duplicate task names in workflow")
        
        # Check all dependencies reference existing tasks
        for from_task, to_task in definition.dependencies:
            if from_task not in task_names:
                raise ValueError(f"Dependency references unknown task: {from_task}")
            if to_task not in task_names:
                raise ValueError(f"Dependency references unknown task: {to_task}")
        
        # Check for cycles using Kahn's algorithm
        indegree: Dict[str, int] = {name: 0 for name in task_names}
        adjacency: Dict[str, List[str]] = {name: [] for name in task_names}
        
        for from_task, to_task in definition.dependencies:
            indegree[to_task] += 1
            adjacency[from_task].append(to_task)
        
        # Topological sort
        queue = [name for name in task_names if indegree[name] == 0]
        visited = 0
        
        while queue:
            node = queue.pop(0)
            visited += 1
            
            for neighbor in adjacency[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        if visited != len(task_names):
            raise ValueError("Workflow contains a cycle")


# ═══════════════════════════════════════════════════════════════════════════
# Main Workflow Engine
# ═══════════════════════════════════════════════════════════════════════════

class WorkflowEngine:
    """
    🎯 **Ultimate Workflow Engine**
    
    DAG-based workflow orchestration with retry, backoff, and state persistence.
    
    Features:
      • DAG validation
      • Topological execution
      • Retry with exponential backoff
      • Soft timeouts
      • Event hooks
      • State persistence
      • Built-in ML tasks
    
    Usage:
```python
        engine = WorkflowEngine()
        
        workflow = WorkflowDefinition(
            name="ml_pipeline",
            tasks=[...],
            dependencies=[...]
        )
        
        run = engine.run(
            workflow,
            initial_context={"session_id": "abc123"}
        )
```
    """
    
    version: str = __version__
    
    def __init__(self, registry: Optional[TaskRegistry] = None):
        """
        Initialize workflow engine.
        
        Args:
            registry: Optional custom task registry
        """
        self.logger = logger.bind(component="WorkflowEngine", version=self.version)
        self.registry = registry or TaskRegistry()
        
        _ensure_dir(WORKFLOWS_PATH)
        
        # Register built-in tasks
        self._register_builtin_tasks()
        
        self.logger.info(f"✓ WorkflowEngine v{self.version} initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def run(
        self,
        definition: WorkflowDefinition,
        *,
        initial_context: Optional[Dict[str, Any]] = None,
        on_event: OnEvent = None,
        run_id: Optional[str] = None
    ) -> WorkflowRun:
        """
        🚀 **Execute Workflow**
        
        Executes workflow with topological ordering.
        
        Args:
            definition: Workflow definition
            initial_context: Initial context dict
            on_event: Event callback function
            run_id: Optional custom run ID
        
        Returns:
            WorkflowRun with complete execution state
        
        Example:
```python
            run = engine.run(
                workflow,
                initial_context={"session_id": "abc123"},
                on_event=lambda e: print(e)
            )
            
            if run.status == "success":
                print("Workflow succeeded!")
                print(f"Artifacts: {run.artifacts}")
```
        """
        # Validate DAG
        DAGValidator.validate(definition)
        
        # Initialize run
        rid = run_id or uuid.uuid4().hex[:12]
        run = WorkflowRun(
            run_id=rid,
            workflow_name=definition.name,
            started_at=_now_iso(),
            context=initial_context.copy() if initial_context else {}
        )
        
        # Initialize task runs
        for task in definition.tasks:
            run.tasks[task.name] = TaskRun(
                name=task.name,
                func=task.func,
                retries=task.retry,
                params=task.params
            )
        
        self._persist_run(run)
        _emit_event(on_event, {
            "type": "workflow_start",
            "run_id": rid,
            "name": definition.name,
            "ts": run.started_at
        })
        
        # Get execution order
        execution_order = self._topological_sort(definition)
        self.logger.info(f"Workflow '{definition.name}' execution order: {execution_order}")
        
        # Build dependency map
        deps_map = self._build_dependency_map(definition)
        
        # Execute tasks
        t_start = time.perf_counter()
        failed_tasks: Set[str] = set()
        
        for task_name in execution_order:
            task_def = self._get_task_definition(definition, task_name)
            task_run = run.tasks[task_name]
            
            # Check if prerequisites met
            predecessors = deps_map["predecessors"].get(task_name, [])
            if any(run.tasks[pred].status != "success" for pred in predecessors):
                # Skip task
                task_run.status = "skipped"
                task_run.started_at = _now_iso()
                task_run.finished_at = task_run.started_at
                task_run.warnings.append("Skipped due to failed dependencies")
                
                run.event_log.append({
                    "ts": task_run.finished_at,
                    "type": "task_skipped",
                    "task": task_name
                })
                
                _emit_event(on_event, {
                    "type": "task_skipped",
                    "run_id": rid,
                    "task": task_name
                })
                
                self._persist_run(run)
                continue
            
            # Execute task with retry
            _emit_event(on_event, {
                "type": "task_start",
                "run_id": rid,
                "task": task_name,
                "ts": _now_iso()
            })
            
            task_run.status = "running"
            task_run.started_at = _now_iso()
            self._persist_run(run)
            
            # Retry loop
            backoff_base = float(getattr(settings, "WORKFLOW_BACKOFF_BASE", 1.8))
            backoff_max = float(getattr(settings, "WORKFLOW_BACKOFF_MAX_SEC", 60.0))
            max_attempts = task_def.retry + 1
            
            for attempt in range(max_attempts):
                task_run.try_index = attempt
                t0 = time.perf_counter()
                
                try:
                    output = self._execute_task(
                        task_def=task_def,
                        task_run=task_run,
                        context=run.context,
                        on_event=on_event
                    )
                    
                    task_run.status = "success"
                    task_run.output = output or {}
                    break
                
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    task_run.errors.append(error_msg)
                    task_run.status = "failed"
                    
                    # Retry with backoff
                    if attempt < max_attempts - 1:
                        delay = min(backoff_max, backoff_base ** (attempt + 1))
                        
                        _emit_event(on_event, {
                            "type": "task_retry",
                            "run_id": rid,
                            "task": task_name,
                            "attempt": attempt + 1,
                            "delay_sec": round(delay, 2),
                            "error": error_msg
                        })
                        
                        time.sleep(delay)
                
                finally:
                    task_run.duration_sec += (time.perf_counter() - t0)
            
            # Finalize task
            task_run.finished_at = _now_iso()
            
            run.event_log.append({
                "ts": task_run.finished_at,
                "type": f"task_{task_run.status}",
                "task": task_name,
                "attempts": task_run.try_index + 1
            })
            
            _emit_event(on_event, {
                "type": f"task_{task_run.status}",
                "run_id": rid,
                "task": task_name,
                "duration_sec": task_run.duration_sec
            })
            
            # Update context and artifacts
            if task_run.status == "success":
                # Merge output
                context_updates = task_run.output.get("context_updates") or {}
                artifacts_updates = task_run.output.get("artifacts") or {}
                
                if context_updates:
                    run.context.update(context_updates)
                
                if artifacts_updates:
                    run.artifacts.update(artifacts_updates)
            else:
                failed_tasks.add(task_name)
                
                # Check if we should stop
                if not task_def.continue_on_error and not definition.continue_on_error:
                    self.logger.warning(f"Task '{task_name}' failed - stopping workflow")
                    break
            
            self._persist_run(run)
        
        # Finalize workflow
        run.finished_at = _now_iso()
        run.duration_sec = time.perf_counter() - t_start
        run.status = "failed" if failed_tasks else "success"
        
        self._persist_run(run)
        
        _emit_event(on_event, {
            "type": "workflow_end",
            "run_id": rid,
            "status": run.status,
            "duration_sec": round(run.duration_sec, 3),
            "ts": run.finished_at
        })
        
        status_icon = "✓" if run.status == "success" else "✗"
        self.logger.info(
            f"{status_icon} Workflow '{definition.name}' {run.status}: "
            f"{len(run.get_successful_tasks())}/{len(run.tasks)} tasks succeeded "
            f"in {run.duration_sec:.1f}s"
        )
        
        return run
    
    # ───────────────────────────────────────────────────────────────────
    # Built-in Tasks
    # ───────────────────────────────────────────────────────────────────
    
    def _register_builtin_tasks(self) -> None:
        """Register built-in workflow tasks."""
        
        # pipeline_e2e
        @self.registry.register_function("pipeline_e2e")
        def pipeline_e2e(ctx: Dict[str, Any]) -> Dict[str, Any]:
            """
            Complete ML pipeline execution.
            
            Required context:
              - session_id: str
              - dataset_name: str
            
            Optional context:
              - target_column: str
              - problem_type: str
              - pipeline_config: dict
            
            Returns:
              - artifacts: dict
              - context_updates: dict
            """
            if SessionManager is None or PipelineExecutor is None:
                raise RuntimeError("Required components not available")
            
            sm = SessionManager()
            session_id = ctx.get("session_id")
            dataset_name = ctx.get("dataset_name")
            
            if not session_id or not dataset_name:
                raise ValueError("pipeline_e2e requires session_id and dataset_name")
            
            # Load data
            df = sm.get_dataframe(session_id, dataset_name)
            
            # Execute pipeline
            executor = PipelineExecutor()
            result = executor.run(
                df,
                target_column=ctx.get("target_column"),
                problem_type=ctx.get("problem_type"),
                config=PipelineConfig(**ctx.get("pipeline_config", {})) if PipelineConfig else None
            )
            
            # Save report to session
            artifacts = {}
            report_path = result.artifacts.get("report_path")
            if report_path:
                try:
                    with open(report_path, "rb") as f:
                        art_ref = sm.put_artifact(
                            session_id,
                            "eda_report",
                            f.read(),
                            filename=Path(report_path).name
                        )
                    artifacts["eda_report"] = art_ref.file.get("path")
                except Exception as e:
                    logger.warning(f"Failed to save report: {e}")
            
            # Extract results
            artifacts.update({
                "best_model": result.summary.get("best_model"),
                "best_score": result.summary.get("best_score")
            })
            
            context_updates = {
                "target_column": result.summary.get("target_column"),
                "problem_type": result.summary.get("problem_type")
            }
            
            return {
                "artifacts": artifacts,
                "context_updates": context_updates
            }
        
        # drift_check
        @self.registry.register_function("drift_check")
        def drift_check(ctx: Dict[str, Any]) -> Dict[str, Any]:
            """Check for data drift."""
            if SessionManager is None or DriftDetector is None:
                raise RuntimeError("Required components not available")
            
            sm = SessionManager()
            session_id = ctx.get("session_id")
            dataset_name = ctx.get("dataset_name")
            
            if not session_id or not dataset_name:
                raise ValueError("drift_check requires session_id and dataset_name")
            
            df = sm.get_dataframe(session_id, dataset_name)
            
            detector = DriftDetector()
            result = detector.check(data=df, target_column=ctx.get("target_column"))
            
            return {
                "artifacts": {"drift_summary": result},
                "context_updates": {"drift_status": result.get("status", "unknown")}
            }
        
        # retrain_decision
        @self.registry.register_function("retrain_decision")
        def retrain_decision(ctx: Dict[str, Any]) -> Dict[str, Any]:
            """Decide if retraining is needed."""
            if RetrainingScheduler is None:
                raise RuntimeError("RetrainingScheduler not available")
            
            scheduler = RetrainingScheduler()
            decision = scheduler.check_should_retrain(
                recent_metrics=ctx.get
                ("recent_metrics") or {},
                drift_status=ctx.get("drift_status")
            )
            
            return {"context_updates": decision}
        
        # save_report_to_session
        @self.registry.register_function("save_report_to_session")
        def save_report_to_session(ctx: Dict[str, Any]) -> Dict[str, Any]:
            """Save report to session artifacts."""
            if SessionManager is None:
                raise RuntimeError("SessionManager not available")
            
            sm = SessionManager()
            session_id = ctx.get("session_id")
            
            if not session_id:
                raise ValueError("save_report_to_session requires session_id")
            
            artifacts = {}
            
            if ctx.get("report_html"):
                data = ctx["report_html"].encode("utf-8")
                ref = sm.put_artifact(
                    session_id,
                    "custom_report",
                    data,
                    filename="custom_report.html"
                )
                artifacts["saved_report"] = ref.file.get("path")
            
            elif ctx.get("report_path"):
                path = Path(ctx["report_path"])
                with open(path, "rb") as f:
                    ref = sm.put_artifact(
                        session_id,
                        path.stem,
                        f.read(),
                        filename=path.name
                    )
                artifacts["saved_report"] = ref.file.get("path")
            
            return {"artifacts": artifacts}
        
        self.logger.info(f"✓ Registered {len(self.registry.list_tasks())} built-in tasks")
    
    # ───────────────────────────────────────────────────────────────────
    # Task Execution
    # ───────────────────────────────────────────────────────────────────
    
    def _execute_task(
        self,
        *,
        task_def: TaskDefinition,
        task_run: TaskRun,
        context: Dict[str, Any],
        on_event: OnEvent
    ) -> Dict[str, Any]:
        """
        Execute single task.
        
        Args:
            task_def: Task definition
            task_run: Task run state
            context: Current workflow context
            on_event: Event callback
        
        Returns:
            Task output dictionary
        """
        func = self.registry.get(task_def.func)
        
        # Merge context and params
        task_context = {**context, **task_def.params}
        
        # Check soft timeout
        deadline = time.perf_counter() + max(1, task_def.soft_timeout_sec)
        
        _emit_event(on_event, {
            "type": "task_call",
            "task": task_def.name,
            "func": task_def.func
        })
        
        # Execute
        result = func(task_context) or {}
        
        # Check timeout
        if time.perf_counter() > deadline:
            warning = f"Soft timeout exceeded: {task_def.soft_timeout_sec}s"
            task_run.warnings.append(warning)
            self.logger.warning(f"Task '{task_def.name}' {warning}")
        
        # Validate output
        if not isinstance(result, dict):
            raise ValueError("Task must return Dict[str, Any]")
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # DAG Operations
    # ───────────────────────────────────────────────────────────────────
    
    def _topological_sort(self, definition: WorkflowDefinition) -> List[str]:
        """
        Topological sort using Kahn's algorithm.
        
        Args:
            definition: Workflow definition
        
        Returns:
            Ordered list of task names
        """
        task_names = [t.name for t in definition.tasks]
        
        indegree: Dict[str, int] = {name: 0 for name in task_names}
        adjacency: Dict[str, List[str]] = {name: [] for name in task_names}
        
        for from_task, to_task in definition.dependencies:
            indegree[to_task] += 1
            adjacency[from_task].append(to_task)
        
        queue = [name for name in task_names if indegree[name] == 0]
        order: List[str] = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            for neighbor in adjacency[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        return order
    
    def _build_dependency_map(
        self,
        definition: WorkflowDefinition
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Build predecessor and successor maps.
        
        Args:
            definition: Workflow definition
        
        Returns:
            Dict with 'predecessors' and 'successors' mappings
        """
        task_names = [t.name for t in definition.tasks]
        
        predecessors: Dict[str, List[str]] = {name: [] for name in task_names}
        successors: Dict[str, List[str]] = {name: [] for name in task_names}
        
        for from_task, to_task in definition.dependencies:
            successors[from_task].append(to_task)
            predecessors[to_task].append(from_task)
        
        return {
            "predecessors": predecessors,
            "successors": successors
        }
    
    def _get_task_definition(
        self,
        definition: WorkflowDefinition,
        name: str
    ) -> TaskDefinition:
        """Get task definition by name."""
        for task in definition.tasks:
            if task.name == name:
                return task
        raise KeyError(f"Task not found: {name}")
    
    # ───────────────────────────────────────────────────────────────────
    # State Persistence
    # ───────────────────────────────────────────────────────────────────
    
    def _persist_run(self, run: WorkflowRun) -> None:
        """Persist workflow run state."""
        run_dir = WORKFLOWS_PATH / run.run_id
        _ensure_dir(run_dir)
        _save_json_atomic(run_dir / "run_state.json", run.to_dict())
    
    def load_run(self, run_id: str) -> Optional[WorkflowRun]:
        """
        Load workflow run from disk.
        
        Args:
            run_id: Run identifier
        
        Returns:
            WorkflowRun or None if not found
        """
        run_path = WORKFLOWS_PATH / run_id / "run_state.json"
        
        if not run_path.exists():
            return None
        
        try:
            data = json.loads(run_path.read_text(encoding="utf-8"))
            
            # Reconstruct WorkflowRun
            tasks = {
                name: TaskRun(**task_data)
                for name, task_data in data["tasks"].items()
            }
            
            return WorkflowRun(
                run_id=data["run_id"],
                workflow_name=data["workflow_name"],
                started_at=data["started_at"],
                finished_at=data.get("finished_at"),
                duration_sec=data.get("duration_sec", 0.0),
                status=data.get("status", "running"),
                tasks=tasks,
                context=data.get("context", {}),
                artifacts=data.get("artifacts", {}),
                event_log=data.get("event_log", [])
            )
        
        except Exception as e:
            self.logger.error(f"Failed to load run {run_id}: {e}")
            return None
    
    def list_runs(self) -> List[str]:
        """
        List all workflow run IDs.
        
        Returns:
            List of run IDs
        """
        _ensure_dir(WORKFLOWS_PATH)
        return sorted([
            p.name
            for p in WORKFLOWS_PATH.iterdir()
            if p.is_dir() and (p / "run_state.json").exists()
        ])


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def create_simple_workflow(
    name: str,
    tasks: List[Tuple[str, str, Dict[str, Any]]],
    dependencies: Optional[List[Tuple[str, str]]] = None
) -> WorkflowDefinition:
    """
    Create simple workflow from task tuples.
    
    Args:
        name: Workflow name
        tasks: List of (task_name, func_name, params) tuples
        dependencies: Optional list of (from, to) tuples
    
    Returns:
        WorkflowDefinition
    
    Example:
```python
        workflow = create_simple_workflow(
            "my_pipeline",
            [
                ("load", "load_data", {"file": "data.csv"}),
                ("train", "train_model", {"epochs": 10}),
                ("eval", "evaluate", {})
            ],
            [("load", "train"), ("train", "eval")]
        )
```
    """
    task_defs = [
        TaskDefinition(name=name, func=func, params=params)
        for name, func, params in tasks
    ]
    
    return WorkflowDefinition(
        name=name,
        tasks=task_defs,
        dependencies=dependencies or []
    )


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print("WorkflowEngine v7.0 - Self Test")
    print("="*80)
    
    # Create engine
    engine = WorkflowEngine()
    print(f"\n✓ Engine initialized: {engine.version}")
    
    # Register custom task
    @engine.registry.register_function("hello")
    def hello(ctx: Dict[str, Any]) -> Dict[str, Any]:
        name = ctx.get("name", "World")
        return {
            "context_updates": {"greeting": f"Hello, {name}!"},
            "artifacts": {"message": f"Hello, {name}!"}
        }
    
    @engine.registry.register_function("goodbye")
    def goodbye(ctx: Dict[str, Any]) -> Dict[str, Any]:
        greeting = ctx.get("greeting", "Hello")
        return {
            "context_updates": {"farewell": "Goodbye!"},
            "artifacts": {"final_message": f"{greeting} -> Goodbye!"}
        }
    
    print(f"✓ Registered custom tasks")
    
    # Create workflow
    workflow = WorkflowDefinition(
        name="test_workflow",
        tasks=[
            TaskDefinition(
                name="greet",
                func="hello",
                params={"name": "DataGenius"}
            ),
            TaskDefinition(
                name="farewell",
                func="goodbye"
            )
        ],
        dependencies=[("greet", "farewell")]
    )
    
    print(f"✓ Created workflow: {workflow.name}")
    
    # Execute
    print("\n" + "="*80)
    print("Executing workflow...")
    print("="*80)
    
    def event_handler(event: Dict[str, Any]) -> None:
        event_type = event.get("type", "unknown")
        if event_type == "workflow_start":
            print(f"\n🚀 Workflow started: {event['name']}")
        elif event_type == "task_start":
            print(f"  ▶️  Task started: {event['task']}")
        elif event_type == "task_success":
            print(f"  ✓ Task succeeded: {event['task']} ({event['duration_sec']:.2f}s)")
        elif event_type == "workflow_end":
            print(f"\n✓ Workflow finished: {event['status']} ({event['duration_sec']:.2f}s)")
    
    run = engine.run(
        workflow,
        initial_context={"test": True},
        on_event=event_handler
    )
    
    print("\n" + "="*80)
    print("Workflow Result:")
    print("="*80)
    
    print(f"\nStatus: {run.status}")
    print(f"Duration: {run.duration_sec:.2f}s")
    print(f"Tasks: {len(run.get_successful_tasks())}/{len(run.tasks)} succeeded")
    
    print(f"\nContext:")
    for key, value in run.context.items():
        print(f"  {key}: {value}")
    
    print(f"\nArtifacts:")
    for key, value in run.artifacts.items():
        print(f"  {key}: {value}")
    
    # List all tasks
    print(f"\n✓ Available tasks: {', '.join(engine.registry.list_tasks())}")
    
    # List runs
    print(f"✓ Saved runs: {len(engine.list_runs())}")
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(WORKFLOWS_PATH)
        print(f"✓ Cleaned up test workflows")
    except:
        pass
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from backend.workflow_engine import (
    WorkflowEngine,
    WorkflowDefinition,
    TaskDefinition
)

# Create engine
engine = WorkflowEngine()

# Define workflow
workflow = WorkflowDefinition(
    name="ml_pipeline",
    tasks=[
        TaskDefinition(
            name="load_and_train",
            func="pipeline_e2e",
            params={
                "session_id": "abc123",
                "dataset_name": "train_data",
                "pipeline_config": {
                    "ml_enabled": True,
                    "generate_report": True
                }
            },
            retry=2
        ),
        TaskDefinition(
            name="check_drift",
            func="drift_check",
            params={
                "session_id": "abc123",
                "dataset_name": "train_data"
            }
        ),
        TaskDefinition(
            name="decide_retrain",
            func="retrain_decision"
        )
    ],
    dependencies=[
        ("load_and_train", "check_drift"),
        ("check_drift", "decide_retrain")
    ]
)

# Execute with event tracking
def on_event(event):
    print(f"[{event['type']}] {event}")

run = engine.run(
    workflow,
    initial_context={"session_id": "abc123"},
    on_event=on_event
)

# Check results
if run.status == "success":
    print(f"✓ Workflow succeeded!")
    print(f"  Best model: {run.artifacts.get('best_model')}")
    print(f"  Best score: {run.artifacts.get('best_score')}")
    print(f"  Should retrain: {run.context.get('should_retrain')}")
else:
    print(f"✗ Workflow failed")
    for task in run.get_failed_tasks():
        print(f"  {task.name}: {task.errors}")

# Register custom task
@engine.registry.register_function("my_custom_task")
def my_custom_task(context):
    # Your custom logic here
    return {
        "context_updates": {"custom_value": 42},
        "artifacts": {"custom_result": "success"}
    }
    """)
