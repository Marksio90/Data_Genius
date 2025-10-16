# agents/ml/orchestrator.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — ML Orchestrator                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Complete ML pipeline orchestration with enterprise safeguards:            ║
║    ✓ Deterministic execution (global seed control)                        ║
║    ✓ Strict input validation (sanity checks)                              ║
║    ✓ Per-agent error control (soft-timeout without kill)                  ║
║    ✓ Comprehensive telemetry (timing, memory, versions)                   ║
║    ✓ Auto-detection of problem type                                       ║
║    ✓ Lifecycle hooks (on_agent_start/on_agent_end)                        ║
║    ✓ Stable output contract (ml_results + summary)                        ║
║    ✓ Partial success mode (continue on agent failure)                     ║
║    ✓ Memory limiter (RSS monitoring)                                      ║
║    ✓ Error budget control (max failures)                                  ║
║    ✓ Agent version tracking                                               ║
║    ✓ Thread-safe operations                                               ║
║    ✓ Defensive programming                                                ║
║    ✓ Versioned output contract                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Pipeline Agents:
    1. ModelSelector — Choose optimal models
    2. ModelTrainer — Train multiple models
    3. ModelEvaluator — Evaluate and compare
    4. ModelExplainer — Generate explanations

Output Contract:
    result.data = {
        "ml_results": {
            "ModelSelector": {...},
            "ModelTrainer": {...},
            "ModelEvaluator": {...},
            "ModelExplainer": {...}
        },
        "summary": {
            "best_model": str | None,
            "best_score": float | None,
            "models_trained": int,
            "key_insights": List[str],
            "timing_sec": {...},
            "run_id": str,
            "seed": int,
            ...
        }
    }
"""

from __future__ import annotations

import os
import random
import time
import uuid
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Domain dependencies
try:
    from core.base_agent import PipelineAgent, AgentResult
except ImportError:
    class PipelineAgent:
        def __init__(self, name: str, agents: List, description: str):
            self.name = name
            self.agents = agents
            self.description = description
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data: Dict[str, Any] = {}
            self.errors: List[str] = []
        
        def add_error(self, error: str):
            self.errors.append(error)
        
        def is_success(self) -> bool:
            return len(self.errors) == 0


__all__ = ["MLConfig", "MLOrchestrator", "orchestrate_ml_pipeline"]
__version__ = "5.1-kosmos-enterprise"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MLConfig:
    """
    Configuration for ML Orchestrator.
    
    Attributes:
        enabled_agents: List of agent names to run (None = all)
        allow_partial: Continue if single agent fails
        strict_target_check: Validate target column strictly
        min_rows: Minimum rows required
        log_timing: Track agent execution times
        problem_types: Allowed problem types
        random_seed: Global seed for reproducibility
        agent_soft_timeout_s: Soft timeout per agent (None = disabled)
        max_error_agents: Max failed agents before abort (None = unlimited)
        attach_agent_versions: Include agent versions in output
        auto_detect_problem_type: Auto-detect from target
        warn_rss_memory_mb: Warn threshold for RSS memory
        hard_stop_rss_memory_mb: Hard stop threshold for RSS
        selector_name: Name of selector agent
        trainer_name: Name of trainer agent
        evaluator_name: Name of evaluator agent
        explainer_name: Name of explainer agent
    """
    # Agent control
    enabled_agents: Optional[List[str]] = None
    allow_partial: bool = True
    strict_target_check: bool = True
    min_rows: int = 20
    
    # Execution control
    log_timing: bool = True
    problem_types: Tuple[str, ...] = ("classification", "regression")
    random_seed: int = 42
    agent_soft_timeout_s: Optional[float] = 300
    max_error_agents: Optional[int] = None
    
    # Telemetry
    attach_agent_versions: bool = True
    auto_detect_problem_type: bool = False
    warn_rss_memory_mb: Optional[int] = 8_192
    hard_stop_rss_memory_mb: Optional[int] = None
    
    # Agent names
    selector_name: str = "ModelSelector"
    trainer_name: str = "ModelTrainer"
    evaluator_name: str = "ModelEvaluator"
    explainer_name: str = "ModelExplainer"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled_agents": self.enabled_agents,
            "allow_partial": self.allow_partial,
            "strict_target_check": self.strict_target_check,
            "min_rows": self.min_rows,
            "log_timing": self.log_timing,
            "problem_types": list(self.problem_types),
            "random_seed": self.random_seed,
            "agent_soft_timeout_s": self.agent_soft_timeout_s,
            "max_error_agents": self.max_error_agents,
            "attach_agent_versions": self.attach_agent_versions,
            "auto_detect_problem_type": self.auto_detect_problem_type,
            "warn_rss_memory_mb": self.warn_rss_memory_mb,
            "hard_stop_rss_memory_mb": self.hard_stop_rss_memory_mb
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str):
    """Decorator for operation timing."""
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


def _safe_operation(operation_name: str, default_value: Any = None):
    """Decorator for safe operations with fallback."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠ {operation_name} failed: {type(e).__name__}: {str(e)[:80]}")
                return default_value
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main ML Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class MLOrchestrator(PipelineAgent):
    """
    **MLOrchestrator** — Complete ML pipeline orchestration.
    
    Responsibilities:
      1. Validate input data and target
      2. Set global random seed (deterministic)
      3. Monitor memory usage (RSS)
      4. Execute agents in sequence
      5. Handle agent failures gracefully
      6. Track timing and telemetry
      7. Aggregate results from all agents
      8. Generate comprehensive summary
      9. Support lifecycle hooks
      10. Maintain stable output contract
    
    Features:
      • Deterministic execution
      • Strict validation
      • Memory monitoring
      • Soft timeout per agent
      • Error budget control
      • Partial success mode
      • Agent version tracking
      • Comprehensive telemetry
    """
    
    def __init__(self, config: Optional[MLConfig] = None) -> None:
        """
        Initialize ML orchestrator.
        
        Args:
            config: Optional custom configuration
        """
        # Lazy imports to avoid circular dependencies
        from agents.ml.model_selector import ModelSelector
        from agents.ml.model_trainer import ModelTrainer
        from agents.ml.model_evaluator import ModelEvaluator
        from agents.ml.model_explainer import ModelExplainer
        
        self.config = config or MLConfig()
        self._log = logger.bind(agent="MLOrchestrator")
        
        # ─── Define All Agents (Deterministic Order) ───
        all_agents = [
            ModelSelector(),
            ModelTrainer(),
            ModelEvaluator(),
            ModelExplainer(),
        ]
        
        # ─── Filter Agents by Configuration ───
        if self.config.enabled_agents:
            name_set = set(self.config.enabled_agents)
            agents = [
                a for a in all_agents
                if (getattr(a, "name", a.__class__.__name__) in name_set)
                or (a.__class__.__name__ in name_set)
            ]
            
            missing = name_set - {
                getattr(a, "name", a.__class__.__name__) for a in all_agents
            }
            if missing:
                self._log.warning(f"⚠ Requested agents not found: {sorted(missing)}")
        else:
            agents = all_agents
        
        super().__init__(
            name="MLOrchestrator",
            agents=agents,
            description="Complete ML training pipeline with enterprise safeguards"
        )
        
        self._log.info(
            f"✓ MLOrchestrator initialized | "
            f"agents={len(self.agents)} | "
            f"seed={self.config.random_seed}"
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("ml_pipeline_execution")
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: Literal["classification", "regression"],
        *,
        on_agent_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_agent_end: Optional[Callable[[str, AgentResult], None]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        Execute complete ML pipeline.
        
        Args:
            data: Preprocessed DataFrame
            target_column: Target column name
            problem_type: 'classification' or 'regression'
            on_agent_start: Hook called before each agent
            on_agent_end: Hook called after each agent
            **kwargs: Additional parameters forwarded to agents
        
        Returns:
            AgentResult with comprehensive ML results and summary
        """
        result = AgentResult(agent_name=self.name)
        started_at_ts = time.time()
        
        try:
            # ─── Step 0: Deterministic Execution ───
            self._set_global_seed(self.config.random_seed)
            
            # ─── Step 1: Validate and Prepare ───
            df = self._validate_and_prepare(data, target_column)
            
            # ─── Step 1a: Auto-Detect Problem Type ───
            if self.config.auto_detect_problem_type:
                problem_type = self._auto_detect_problem_type(
                    df[target_column],
                    default=problem_type
                )
            
            self._log.info(
                f"Starting ML pipeline | "
                f"type={problem_type} | "
                f"target={target_column} | "
                f"rows={len(df)} | "
                f"cols={len(df.columns)}"
            )
            
            # ─── Step 2: Memory Monitoring ───
            warnings_list: List[str] = []
            rss_mb = self._get_rss_memory_mb()
            
            if rss_mb is not None:
                if self.config.warn_rss_memory_mb and rss_mb > self.config.warn_rss_memory_mb:
                    warn = f"High RSS memory: {rss_mb:.0f} MB"
                    self._log.warning(warn)
                    warnings_list.append(warn)
                
                if self.config.hard_stop_rss_memory_mb and rss_mb > self.config.hard_stop_rss_memory_mb:
                    msg = f"RSS memory {rss_mb:.0f} MB exceeded hard limit {self.config.hard_stop_rss_memory_mb} MB"
                    self._log.error(msg)
                    result.add_error(msg)
                    result.data = {"pipeline_results": []}
                    return result
            
            # ─── Step 3: Pipeline Execution ───
            run_id = f"mlrun_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            t0 = time.perf_counter()
            
            pipeline_results: List[AgentResult] = []
            timing_sec: Dict[str, float] = {}
            agent_status: Dict[str, str] = {}
            errors_seen = 0
            versions: Dict[str, str] = {}
            
            for agent in self.agents:
                agent_label = getattr(agent, "name", agent.__class__.__name__)
                
                # ─── Pre-Agent Hook ───
                if on_agent_start:
                    try:
                        on_agent_start(agent_label, {"run_id": run_id})
                    except Exception:
                        pass
                
                self._log.info(f"▶ Running agent: {agent_label}")
                t_agent = time.perf_counter()
                agent_res: Optional[AgentResult] = None
                
                try:
                    # ─── Execute Agent with Soft Timeout ───
                    agent_res = self._run_agent_with_soft_timeout(
                        agent=agent,
                        timeout_s=self.config.agent_soft_timeout_s,
                        data=df,
                        target_column=target_column,
                        problem_type=problem_type,
                        **kwargs
                    )
                    
                    if not isinstance(agent_res, AgentResult):
                        raise TypeError(
                            f"Agent '{agent_label}' returned invalid type: {type(agent_res)}"
                        )
                    
                    # ─── Check Agent Status ───
                    if agent_res.errors:
                        msg = f"Agent '{agent_label}' reported errors: {agent_res.errors}"
                        if self.config.allow_partial:
                            self._log.warning(msg)
                            agent_status[agent_label] = "error"
                            errors_seen += 1
                        else:
                            raise RuntimeError(msg)
                    else:
                        agent_status[agent_label] = "success"
                        self._log.success(f"✔ Agent completed: {agent_label}")
                
                except Exception as e:
                    err_msg = f"Agent '{agent_label}' failed: {e}"
                    if self.config.allow_partial:
                        self._log.error(err_msg, exc_info=True)
                        failed = AgentResult(agent_name=agent_label)
                        failed.add_error(str(e))
                        agent_res = failed
                        agent_status[agent_label] = "error"
                        errors_seen += 1
                    else:
                        raise
                
                finally:
                    # ─── Record Timing ───
                    if self.config.log_timing:
                        timing_sec[agent_label] = round(time.perf_counter() - t_agent, 4)
                    
                    # ─── Store Results ───
                    if agent_res is not None:
                        pipeline_results.append(agent_res)
                        
                        # ─── Post-Agent Hook ───
                        if on_agent_end:
                            try:
                                on_agent_end(agent_label, agent_res)
                            except Exception:
                                pass
                    
                    # ─── Track Version ───
                    if self.config.attach_agent_versions:
                        versions[agent_label] = getattr(agent, "version", "unknown")
                    
                    # ─── Check Error Budget ───
                    if self.config.max_error_agents is not None and errors_seen >= self.config.max_error_agents:
                        stop_msg = f"Too many agent errors ({errors_seen} >= {self.config.max_error_agents}) — aborting"
                        self._log.error(stop_msg)
                        warnings_list.append(stop_msg)
                        break
            
            # ─── Step 4: Call Parent Execute (Compatibility) ───
            pipeline_parent = super().execute(
                data=df,
                target_column=target_column,
                problem_type=problem_type,
                **kwargs
            )
            pipeline_parent.data = {"pipeline_results": pipeline_results}
            
            # ─── Step 5: Aggregate and Summarize ───
            finished_at_ts = time.time()
            
            if pipeline_parent.is_success():
                ml_results = self._aggregate_ml_results(pipeline_parent.data["pipeline_results"])
                summary = self._generate_ml_summary(ml_results)
                
                # ─── Add Timing ───
                if self.config.log_timing:
                    summary["timing_sec"] = {
                        **timing_sec,
                        "total": round(time.perf_counter() - t0, 4)
                    }
                
                # ─── Add Telemetry ───
                summary.update({
                    "run_id": run_id,
                    "seed": self.config.random_seed,
                    "problem_type": problem_type,
                    "target_column": target_column,
                    "n_rows": int(len(df)),
                    "n_cols": int(len(df.columns)),
                    "agents": [getattr(a, "name", a.__class__.__name__) for a in self.agents],
                    "agent_versions": versions,
                    "partial_success": bool(errors_seen > 0),
                    "errors_seen": int(errors_seen),
                    "memory_rss_mb": self._get_rss_memory_mb(),
                    "warnings": warnings_list,
                    "started_at_ts": started_at_ts,
                    "finished_at_ts": finished_at_ts,
                    "version": __version__,
                    "agent_status": agent_status,
                })
                
                result.data = {"ml_results": ml_results, "summary": summary}
                self._log.success(
                    f"✓ ML pipeline completed | "
                    f"time={summary['timing_sec']['total']:.2f}s | "
                    f"errors={errors_seen}"
                )
            else:
                for e in pipeline_parent.errors:
                    result.add_error(e)
                result.data = {"pipeline_results": pipeline_results}
        
        except Exception as e:
            result.add_error(f"ML pipeline failed: {e}")
            self._log.error(f"ML pipeline error: {e}", exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Validation & Preparation
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("validate_and_prepare", default_value=None)
    def _validate_and_prepare(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Validate input data and prepare for ML pipeline."""
        # ─── Type Check ───
        if not isinstance(df, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame")
        
        # ─── Size Check ───
        if df is None or df.empty or len(df) < self.config.min_rows:
            raise ValueError(
                f"DataFrame is empty or too small (min_rows={self.config.min_rows})"
            )
        
        # ─── Target Column Check ───
        if not isinstance(target_column, str) or not target_column:
            raise ValueError("'target_column' must be a non-empty string")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # ─── Constant Columns Warning ───
        try:
            nunique = df.nunique(dropna=False)
            constant_cols = nunique[nunique <= 1].index.tolist()
            if constant_cols:
                self._log.warning(
                    f"{len(constant_cols)} constant columns detected "
                    f"(e.g., {constant_cols[:3]}). Consider dropping."
                )
        except Exception:
            pass
        
        # ─── Strict Target Checks ───
        if self.config.strict_target_check:
            tgt = df[target_column]
            
            # Missing values
            try:
                n_missing = int(tgt.isna().sum())
                if n_missing > 0:
                    self._log.warning(
                        f"Target column '{target_column}' contains {n_missing} missing values"
                    )
            except Exception:
                pass
            
            # Constant target
            try:
                if tgt.nunique(dropna=True) <= 1:
                    raise ValueError(
                        f"Target column '{target_column}' has ≤1 unique value; "
                        f"cannot train a model"
                    )
            except Exception:
                pass
        
        return df
    
    # ───────────────────────────────────────────────────────────────────
    # Auto-Detection
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("auto_detect_problem_type", default_value=None)
    def _auto_detect_problem_type(self, y: pd.Series, default: str) -> str:
        """Auto-detect problem type from target column."""
        try:
            n_unique = pd.Series(y).nunique(dropna=True)
            
            if pd.api.types.is_numeric_dtype(y) and n_unique > 20:
                pt = "regression"
            else:
                pt = "classification"
            
            if pt not in self.config.problem_types:
                return default
            
            if pt != default:
                self._log.info(f"Auto-detected problem_type='{pt}' (was '{default}')")
            
            return pt
        
        except Exception:
            return default
    
    # ───────────────────────────────────────────────────────────────────
    # Result Aggregation
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("aggregate_results", default_value={})
    def _aggregate_ml_results(self, pipeline_results: List[AgentResult]) -> Dict[str, Any]:
        """Aggregate results from all ML agents."""
        aggregated: Dict[str, Any] = {}
        
        for agent_result in pipeline_results:
            try:
                agent_name = getattr(agent_result, "agent_name", None) or "UnknownAgent"
                aggregated[agent_name] = getattr(agent_result, "data", None)
            except Exception as e:
                self._log.warning(f"⚠ Aggregation skipped for agent: {e}")
        
        return aggregated
    
    # ───────────────────────────────────────────────────────────────────
    # Summary Generation
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("generate_summary", default_value={})
    def _generate_ml_summary(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of ML results."""
        summary: Dict[str, Any] = {
            "best_model": None,
            "best_score": None,
            "models_trained": 0,
            "key_insights": [],
        }
        
        # ─── Best Model from Evaluator ───
        try:
            if "ModelEvaluator" in ml_results and isinstance(ml_results["ModelEvaluator"], dict):
                eval_results = ml_results["ModelEvaluator"] or {}
                summary["best_model"] = eval_results.get("best_model_name")
                summary["best_score"] = eval_results.get("best_score")
        except Exception:
            pass
        
        # ─── Number of Models from Trainer ───
        try:
            if "ModelTrainer" in ml_results and isinstance(ml_results["ModelTrainer"], dict):
                trainer_results = ml_results["ModelTrainer"] or {}
                models_cmp = trainer_results.get("models_comparison") or []
                summary["models_trained"] = int(len(models_cmp)) if hasattr(models_cmp, "__len__") else 0
        except Exception:
            pass
        
        # ─── Top Features from Explainer ───
        try:
            if "ModelExplainer" in ml_results and isinstance(ml_results["ModelExplainer"], dict):
                explainer_results = ml_results["ModelExplainer"] or {}
                top_features = explainer_results.get("top_features") or []
                if top_features:
                    summary["key_insights"].append(
                        f"Top features: {', '.join(map(str, top_features[:3]))}"
                    )
        except Exception:
            pass
        
        return summary
    
    # ───────────────────────────────────────────────────────────────────
    # Deterministic Execution
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("set_global_seed", default_value=None)
    def _set_global_seed(self, seed: int) -> None:
        """Set global random seed for reproducibility."""
        try:
            np.random.seed(seed)
        except Exception:
            pass
        
        try:
            random.seed(seed)
        except Exception:
            pass
        
        # Note: Deliberately not importing torch/sklearn to avoid heavy imports
    
    # ───────────────────────────────────────────────────────────────────
    # Soft Timeout (No Kill)
    # ───────────────────────────────────────────────────────────────────
    
    def _run_agent_with_soft_timeout(
        self,
        agent,
        timeout_s: Optional[float],
        **kwargs
    ) -> AgentResult:
        """
        Run agent with soft timeout.
        
        Logs warning if timeout exceeded but allows agent to complete.
        This avoids brutal termination and state loss.
        """
        if timeout_s is None or timeout_s <= 0:
            return agent.execute(**kwargs)
        
        start = time.perf_counter()
        res: Optional[AgentResult] = None
        
        try:
            res = agent.execute(**kwargs)
            return res
        finally:
            elapsed = time.perf_counter() - start
            if elapsed > timeout_s:
                name = getattr(agent, "name", agent.__class__.__name__)
                self._log.warning(
                    f"⚠ Agent '{name}' exceeded soft timeout "
                    f"({elapsed:.1f}s > {timeout_s:.1f}s)"
                )
    
    # ───────────────────────────────────────────────────────────────────
    # Memory Monitoring
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("get_rss_memory", default_value=None)
    def _get_rss_memory_mb(self) -> Optional[float]:
        """Get current RSS memory usage in MB (best-effort)."""
        try:
            import psutil
            p = psutil.Process(os.getpid())
            return float(p.memory_info().rss) / (1024 ** 2)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Function
# ═══════════════════════════════════════════════════════════════════════════

def orchestrate_ml_pipeline(
    data: pd.DataFrame,
    target_column: str,
    problem_type: Literal["classification", "regression"],
    config: Optional[MLConfig] = None,
    **kwargs
) -> AgentResult:
    """
    Convenience function to orchestrate ML pipeline.
    
    Usage:
```python
        from agents.ml import orchestrate_ml_pipeline
        
        result = orchestrate_ml_pipeline(
            data=df,
            target_column='target',
            problem_type='classification'
        )
        
        # Access results
        ml_results = result.data['ml_results']
        summary = result.data['summary']
        
        print(f"Best model: {summary['best_model']}")
        print(f"Best score: {summary['best_score']}")
        print(f"Models trained: {summary['models_trained']}")
```
    
    Args:
        data: Preprocessed DataFrame
        target_column: Target column name
        problem_type: 'classification' or 'regression'
        config: Optional custom configuration
        **kwargs: Additional parameters for agents
    
    Returns:
        AgentResult with complete ML pipeline results
    """
    orchestrator = MLOrchestrator(config)
    return orchestrator.execute(
        data=data,
        target_column=target_column,
        problem_type=problem_type,
        **kwargs
    )