# backend/pipeline_executor.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Pipeline Executor v7.0           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE END-TO-END ML PIPELINE ORCHESTRATION ENGINE                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Complete E2E Pipeline (Schema → ML → Monitoring → Report)             ║
║  ✓ Retry Logic with Exponential Backoff                                  ║
║  ✓ Soft Timeouts & Hard Memory Limits                                    ║
║  ✓ Step Caching (read-only operations)                                   ║
║  ✓ Event Hooks for Real-Time Monitoring                                  ║
║  ✓ Deterministic Execution (global seed)                                 ║
║  ✓ Graceful Error Handling                                               ║
║  ✓ Comprehensive Telemetry                                               ║
║  ✓ Artifact Management                                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Pipeline Stages:
    1. Schema Analysis       → Column types, distributions
    2. Data Profiling        → Statistics, quality metrics
    3. Target Detection      → Auto-detect target column
    4. Problem Classification → Classification vs Regression
    5. Missing Data Handling → Imputation strategies
    6. Feature Engineering   → Derived features
    7. Preprocessing Build   → Complete preprocessing pipeline
    8. EDA                   → Exploratory data analysis
    9. ML Training           → Model selection & training
    10. Performance Tracking → Metrics logging
    11. Drift Baseline       → Set reference for monitoring
    12. Report Generation    → HTML/PDF/Markdown report

Features:
    • Retry with exponential backoff + jitter
    • Soft timeouts (warning) vs hard timeouts (stop)
    • Memory monitoring (RSS with psutil)
    • Step-level caching for immutable operations
    • Event hooks for UI integration
    • Deterministic execution via global seed
    • Continue-on-error mode
    • Artifact tracking

Usage:
```python
    from backend.pipeline_executor import PipelineExecutor, PipelineConfig
    
    # Configure pipeline
    config = PipelineConfig(
        steps=["schema", "profile", "eda", "ml"],
        continue_on_error=True,
        retries_per_step=2,
        random_seed=42,
        ml_enabled=True,
        generate_report=True
    )
    
    # Execute
    executor = PipelineExecutor()
    result = executor.run(
        df=train_df,
        target_column="target",
        problem_type="classification",
        config=config
    )
    
    # Check results
    if result.ok:
        print(f"Pipeline succeeded in {result.duration_sec:.1f}s")
        print(f"Best model: {result.summary['best_model']}")
    else:
        print(f"Pipeline failed: {result.errors_seen} errors")
```

Dependencies:
    • pandas, numpy
    • loguru
    • psutil (optional, for memory monitoring)
    • All agents.* modules
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Type Definitions
# ═══════════════════════════════════════════════════════════════════════════

ProblemType = Literal["classification", "regression"]

StepName = Literal[
    "schema",
    "profile",
    "target_detect",
    "problem_classify",
    "missing_handle",
    "feature_engineer",
    "preprocess_build",
    "eda",
    "ml",
    "performance_track",
    "drift_baseline",
    "report",
]

DEFAULT_STEPS: List[StepName] = [
    "schema",
    "profile",
    "target_detect",
    "problem_classify",
    "missing_handle",
    "feature_engineer",
    "preprocess_build",
    "eda",
    "ml",
    "performance_track",
    "drift_baseline",
    "report",
]


# ═══════════════════════════════════════════════════════════════════════════
# Agent Imports
# ═══════════════════════════════════════════════════════════════════════════

from core.base_agent import AgentResult

# EDA
try:
    from agents.eda.schema_analyzer import SchemaAnalyzer
    from agents.eda.data_profiler import DataProfiler
    from agents.eda.eda_orchestrator import EDAOrchestrator
    from agents.eda.problem_classifier import ProblemClassifier
except ImportError as e:
    logger.warning(f"⚠ EDA agents not available: {e}")

# Target
try:
    from agents.target.target_detector import TargetDetector
except ImportError as e:
    logger.warning(f"⚠ Target detector not available: {e}")

# Preprocessing
try:
    from agents.preprocessing.missing_data_handler import MissingDataHandler
    from agents.preprocessing.feature_engineer import FeatureEngineer
    from agents.preprocessing.pipeline_builder import PipelineBuilder
    from agents.preprocessing.encoder_selector import EncoderSelector, EncoderPolicy
    from agents.preprocessing.scaler_selector import ScalerSelector, ScalerSelectorConfig
except ImportError as e:
    logger.warning(f"⚠ Preprocessing agents not available: {e}")

# ML
try:
    from agents.ml.ml_orchestrator import MLOrchestrator
except ImportError as e:
    logger.warning(f"⚠ ML orchestrator not available: {e}")

# Monitoring
try:
    from agents.monitoring.performance_tracker import PerformanceTracker
    from agents.monitoring.drift_detector import DriftDetector
except ImportError as e:
    logger.warning(f"⚠ Monitoring agents not available: {e}")

# Reporting
try:
    from agents.reporting.report_generator import ReportGenerator
except ImportError as e:
    logger.warning(f"⚠ Report generator not available: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _hash_dataframe(df: pd.DataFrame, max_rows: int = 100_000) -> str:
    """
    Create stable DataFrame hash for caching.
    
    Args:
        df: DataFrame to hash
        max_rows: Max rows to sample for large datasets
    
    Returns:
        Hex hash string
    """
    try:
        sample = df if len(df) <= max_rows else df.sample(
            n=max_rows,
            random_state=42
        )
        
        col_sig = "|".join(map(str, sample.columns))
        data_sig = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
        
        return f"h{hash((col_sig, data_sig)) & 0xFFFFFFFF:X}"
    
    except Exception:
        return f"h{hash((tuple(df.columns), df.shape)) & 0xFFFFFFFF:X}"


def _now_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _rss_memory_mb() -> Optional[float]:
    """
    Get RSS memory usage in MB (best-effort with psutil).
    
    Returns:
        RSS memory in MB, or None if psutil unavailable
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss) / (1024 ** 2)
    except Exception:
        return None


def _set_global_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    # Python random
    try:
        random.seed(seed)
    except Exception:
        pass
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """
    📊 **Pipeline Step Result**
    
    Complete result information for a single pipeline step.
    
    Attributes:
        name: Step name
        ok: Success status
        started_at: Start timestamp (ISO 8601)
        finished_at: Finish timestamp (ISO 8601)
        duration_sec: Duration in seconds
        data: Step output data
        warnings: Warning messages
        errors: Error messages
        attempts: Number of attempts made
        soft_timeout_exceeded: Soft timeout warning
        memory_rss_mb: RSS memory at completion
        agent_version: Agent version string
    """
    
    name: StepName
    ok: bool
    started_at: str
    finished_at: str
    duration_sec: float
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    attempts: int = 1
    soft_timeout_exceeded: bool = False
    memory_rss_mb: Optional[float] = None
    agent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PipelineConfig:
    """
    ⚙️ **Pipeline Configuration**
    
    Complete configuration for pipeline execution.
    
    Execution Control:
        steps: List of steps to execute
        continue_on_error: Continue after step failures
        max_failed_steps: Max failures before stopping (None = no limit)
    
    Retry & Timeout:
        retries_per_step: Additional retry attempts
        initial_backoff_s: Initial backoff delay
        backoff_multiplier: Backoff multiplier
        jitter_fraction: Random jitter fraction
        soft_timeout_per_step_s: Soft timeout (warning only)
    
    Determinism:
        random_seed: Global random seed
    
    Preprocessing:
        missing_strategy: Imputation strategy
        enable_feature_engineering: Enable feature engineering
        enable_preprocessing_builder: Enable preprocessing pipeline
    
    Recommendations:
        enable_encoder_reco: Enable encoder recommendations
        enable_scaler_reco: Enable scaler recommendations
    
    ML & Monitoring:
        ml_enabled: Enable ML training
        track_performance: Track performance metrics
        set_drift_baseline: Set drift baseline
    
    Reporting:
        report_format: Report output format
        generate_report: Generate report
    
    Memory:
        warn_rss_memory_mb: Warning threshold (MB)
        hard_stop_rss_memory_mb: Hard stop threshold (MB)
    
    Callbacks:
        on_event: Event callback function
    """
    
    # Execution
    steps: List[StepName] = field(default_factory=lambda: DEFAULT_STEPS.copy())
    continue_on_error: bool = True
    max_failed_steps: Optional[int] = None
    
    # Retry & Timeout
    retries_per_step: int = 1
    initial_backoff_s: float = 0.5
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.2
    soft_timeout_per_step_s: Optional[float] = 300.0
    
    # Determinism
    random_seed: int = 42
    
    # Preprocessing
    missing_strategy: Literal["auto", "mean", "median", "mode", "knn", "drop"] = "auto"
    enable_feature_engineering: bool = True
    enable_preprocessing_builder: bool = True
    
    # Recommendations (telemetry only)
    enable_encoder_reco: bool = False
    enable_scaler_reco: bool = False
    
    # ML & Monitoring
    ml_enabled: bool = True
    track_performance: bool = True
    set_drift_baseline: bool = True
    
    # Reporting
    report_format: Literal["html", "pdf", "markdown"] = "html"
    generate_report: bool = True
    
    # Memory
    warn_rss_memory_mb: Optional[int] = 8_192
    hard_stop_rss_memory_mb: Optional[int] = None
    
    # Callbacks
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding callback)."""
        d = asdict(self)
        d.pop("on_event", None)
        return d


@dataclass
class PipelineResult:
    """
    📦 **Complete Pipeline Result**
    
    Final pipeline execution result with all steps and artifacts.
    
    Attributes:
        ok: Overall success status
        started_at: Pipeline start timestamp
        finished_at: Pipeline finish timestamp
        duration_sec: Total duration
        df_hash: DataFrame content hash
        steps: List of step results
        artifacts: Pipeline artifacts
        summary: Execution summary
        memory_rss_mb: Final RSS memory
        errors_seen: Total errors encountered
        version: Pipeline executor version
    """
    
    ok: bool
    started_at: str
    finished_at: str
    duration_sec: float
    df_hash: str
    steps: List[StepResult]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    memory_rss_mb: Optional[float] = None
    errors_seen: int = 0
    version: str = "7.0-ultimate"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d
    
    def get_step(self, name: StepName) -> Optional[StepResult]:
        """Get result for specific step."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def get_failed_steps(self) -> List[StepResult]:
        """Get all failed steps."""
        return [s for s in self.steps if not s.ok]
    
    def get_warnings(self) -> List[Tuple[StepName, List[str]]]:
        """Get all warnings by step."""
        return [(s.name, s.warnings) for s in self.steps if s.warnings]


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline Executor
# ═══════════════════════════════════════════════════════════════════════════

class PipelineExecutor:
    """
    🎯 **Ultimate Pipeline Executor**
    
    Enterprise-grade end-to-end ML pipeline orchestration.
    
    Features:
      • 12-stage complete ML workflow
      • Retry logic with exponential backoff
      • Soft & hard timeouts
      • Memory monitoring
      • Step-level caching
      • Event hooks
      • Deterministic execution
      • Graceful error handling
    
    Architecture:
```
        ┌────────────────────────────────────────────────┐
        │         PipelineExecutor                       │
        ├────────────────────────────────────────────────┤
        │  Cache: (df_hash, step) → results             │
        │  Agents: Schema, Profile, EDA, ML, etc.       │
        ├────────────────────────────────────────────────┤
        │  Execute Pipeline:                             │
        │    1. Validate input                           │
        │    2. Set global seed                          │
        │    3. For each step:                           │
        │       • Check cache                            │
        │       • Execute with retry                     │
        │       • Check timeouts                         │
        │       • Monitor memory                         │
        │       • Emit events                            │
        │    4. Build result                             │
        └────────────────────────────────────────────────┘
```
    
    Usage:
```python
        executor = PipelineExecutor()
        
        config = PipelineConfig(
            steps=["schema", "eda", "ml"],
            retries_per_step=2,
            ml_enabled=True
        )
        
        result = executor.run(
            df=train_df,
            target_column="target",
            problem_type="classification",
            config=config
        )
```
    """
    
    version: str = "7.0-ultimate"
    
    def __init__(self) -> None:
        """Initialize pipeline executor with agent instances."""
        self.logger = logger.bind(component="PipelineExecutor", version=self.version)
        
        # Initialize agents
        self._init_agents()
        
        # Step cache: (df_hash, step_name) → data
        self._cache: Dict[Tuple[str, StepName], Dict[str, Any]] = {}
        
        self.logger.info(f"✓ PipelineExecutor v{self.version} initialized")
    
    def _init_agents(self) -> None:
        """Initialize all agent instances."""
        try:
            # Analysis
            self._schema = SchemaAnalyzer()
            self._profiler = DataProfiler()
            self._eda = EDAOrchestrator()
            self._problem = ProblemClassifier()
            self._target = TargetDetector()
            
            # Preprocessing
            self._missing = MissingDataHandler()
            self._fe = FeatureEngineer()
            self._pre = PipelineBuilder()
            self._enc = EncoderSelector(EncoderPolicy())
            self._scl = ScalerSelector(ScalerSelectorConfig())
            
            # ML
            self._ml = MLOrchestrator()
            
            # Monitoring
            self._perf = PerformanceTracker()
            self._drift = DriftDetector()
            
            # Reporting
            self._report = ReportGenerator()
            
            self.logger.debug("All agents initialized")
        
        except Exception as e:
            self.logger.error(f"Agent initialization warning: {e}")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def run(
        self,
        df: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
        problem_type: Optional[ProblemType] = None,
        config: Optional[PipelineConfig] = None
    ) -> PipelineResult:
        """
        🚀 **Execute Complete Pipeline**
        
        Runs end-to-end ML pipeline with all configured steps.
        
        Args:
            df: Input DataFrame
            target_column: Target column (auto-detect if None)
            problem_type: Problem type (auto-detect if None)
            config: Pipeline configuration
        
        Returns:
            PipelineResult with complete execution information
        
        Raises:
            ValueError: If input validation fails
        
        Example:
```python
            executor = PipelineExecutor()
            
            result = executor.run(
                df=train_df,
                target_column="target",
                problem_type="classification",
                config=PipelineConfig(
                    ml_enabled=True,
                    generate_report=True
                )
            )
            
            if result.ok:
                print(f"✓ Pipeline succeeded")
                print(f"  Best model: {result.summary['best_model']}")
                print(f"  Best score: {result.summary['best_score']}")
            else:
                print(f"✗ Pipeline failed")
                for step in result.get_failed_steps():
                    print(f"  {step.name}: {step.errors}")
```
        """
        cfg = config or PipelineConfig()
        
        # Validation
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be non-empty DataFrame")
        
        # Set global seed for reproducibility
        _set_global_seed(cfg.random_seed)
        
        # Initialize tracking
        t_start = time.perf_counter()
        ts_start = _now_iso()
        df_hash = _hash_dataframe(df)
        
        steps_log: List[StepResult] = []
        artifacts: Dict[str, Any] = {}
        context: Dict[str, Any] = {}
        errors_seen = 0
        
        self.logger.info("="*80)
        self.logger.info(
            f"🚀 Pipeline starting: df_hash={df_hash}, "
            f"shape={df.shape}, seed={cfg.random_seed}"
        )
        self.logger.info("="*80)
        
        # Check initial memory
        rss_mb = _rss_memory_mb()
        if rss_mb and cfg.warn_rss_memory_mb and rss_mb > cfg.warn_rss_memory_mb:
            warn_msg = f"High RSS memory at start: {rss_mb:.0f}MB"
            self.logger.warning(warn_msg)
            self._emit_event(cfg, {
                "type": "memory_warn",
                "rss_mb": rss_mb,
                "ts": _now_iso(),
                "where": "start"
            })
        
        # Execute steps
        try:
            # 1. Schema
            if "schema" in cfg.steps:
                result = self._execute_step(
                    "schema",
                    lambda: self._schema.execute(data=df),
                    cfg, df_hash
                )
                steps_log.append(result)
                context["schema"] = result.data
                errors_seen += (0 if result.ok else 1)
                
                if not result.ok and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # 2. Profile
            if "profile" in cfg.steps:
                result = self._execute_step(
                    "profile",
                    lambda: self._profiler.execute(data=df),
                    cfg, df_hash
                )
                steps_log.append(result)
                context["profile"] = result.data
                errors_seen += (0 if result.ok else 1)
                
                if not result.ok and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # 3. Target Detection
            if "target_detect" in cfg.steps and not target_column:
                col_info = context.get("schema", {}).get("columns", [])
                result = self._execute_step(
                    "target_detect",
                    lambda: self._target.execute(
                        data=df,
                        column_info=col_info,
                        user_target=None
                    ),
                    cfg, df_hash
                )
                steps_log.append(result)
                
                if result.ok:
                    target_column = result.data.get("target_column")
                    context["target_detect"] = result.data
                
                errors_seen += (0 if result.ok else 1)
                
                if (not result.ok or not target_column) and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            if not target_column and "ml" in cfg.steps:
                self.logger.warning("Target column not resolved - ML step may be skipped")
            
            # 4. Problem Classification
            if "problem_classify" in cfg.steps and target_column and not problem_type:
                result = self._execute_step(
                    "problem_classify",
                    lambda: self._problem.execute(
                        data=df,
                        target_column=target_column  # type: ignore
                    ),
                    cfg, df_hash
                )
                steps_log.append(result)
                
                if result.ok:
                    problem_type = result.data.get("problem_type")
                    context["problem_classify"] = result.data
                
                errors_seen += (0 if result.ok else 1)
                
                if (not result.ok or not problem_type) and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # 5. Missing Data Handling
            if "missing_handle" in cfg.steps and target_column:
                result = self._execute_step(
                    "missing_handle",
                    lambda: self._missing.execute(
                        data=df,
                        target_column=target_column,  # type: ignore
                        strategy=cfg.missing_strategy
                    ),
                    cfg, df_hash
                )
                steps_log.append(result)
                
                if result.ok and "data" in result.data:
                    df = result.data["data"]
                    context["missing"] = result.data
                
                errors_seen += (0 if result.ok else 1)
                
                if not result.ok and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # 6. Feature Engineering
            if "feature_engineer" in cfg.steps and cfg.enable_feature_engineering and target_column:
                result = self._execute_step(
                    "feature_engineer",
                    lambda: self._fe.execute(
                        data=df,
                        target_column=target_column  # type: ignore
                    ),
                    cfg, df_hash
                )
                steps_log.append(result)
                
                if result.ok and "engineered_data" in result.data:
                    df = result.data["engineered_data"]
                    context["fe"] = result.data
                
                errors_seen += (0 if result.ok else 1)
                
                if not result.ok and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # 7. Preprocessing Pipeline Build
            if ("preprocess_build" in cfg.steps and 
                cfg.enable_preprocessing_builder and 
                target_column and problem_type):
                
                result = self._execute_step(
                    "preprocess_build",
                    lambda: self._pre.execute(
                        data=df,
                        target_column=target_column,  # type: ignore
                        problem_type=problem_type  # type: ignore
                    ),
                    cfg, df_hash
                )
                steps_log.append(result)
                
                if result.ok:
                    context["preprocess"] = result.data
                
                errors_seen += (0 if result.ok else 1)
                
                if not result.ok and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # Optional: Encoder/Scaler recommendations (telemetry only)
            if cfg.enable_encoder_reco and target_column and problem_type:
                try:
                    enc_res = self._enc.execute(
                        data=df,
                        target_column=target_column,
                        problem_type=problem_type  # type: ignore
                    )
                    context["encoder_reco"] = enc_res.data
                except Exception as e:
                    self.logger.warning(f"Encoder recommendation skipped: {e}")
            
            if cfg.enable_scaler_reco and target_column:
                try:
                    scl_res = self._scl.execute(
                        data=df,
                        target_column=target_column
                    )
                    context["scaler_reco"] = scl_res.data
                except Exception as e:
                    self.logger.warning(f"Scaler recommendation skipped: {e}")
            
            # 8. EDA
            if "eda" in cfg.steps:
                result = self._execute_step(
                    "eda",
                    lambda: self._eda.execute(
                        data=df,
                        target_column=target_column
                    ),
                    cfg, df_hash
                )
                steps_log.append(result)
                
                if result.ok:
                    context["eda"] = result.data
                
                errors_seen += (0 if result.ok else 1)
                
                if not result.ok and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # 9. ML Training
            if "ml" in cfg.steps and cfg.ml_enabled and target_column and problem_type:
                result = self._execute_step(
                    "ml",
                    lambda: self._ml.execute(
                        data=df,
                        target_column=target_column,  # type: ignore
                        problem_type=problem_type  # type: ignore
                    ),
                    cfg, df_hash
                )
                steps_log.append(result)
                
                if result.ok:
                    context["ml"] = result.data
                    
                    # Extract artifacts
                    summary = result.data.get("summary", {})
                    artifacts["best_model"] = summary.get("best_model")
                    artifacts["best_score"] = summary.get("best_score")
                
                errors_seen += (0 if result.ok else 1)
                
                if not result.ok and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # 10. Performance Tracking
            if "performance_track" in cfg.steps and cfg.track_performance and "ml" in context:
                ml_block = context["ml"]
                eval_block = ml_block.get("ml_results", {}).get("ModelEvaluator", {})
                
                y_true = eval_block.get("y_true")
                y_pred = eval_block.get("y_pred")
                y_proba = eval_block.get("y_proba")
                
                if y_true is not None and y_pred is not None and problem_type:
                    result = self._execute_step(
                        "performance_track",
                        lambda: self._perf.execute(
                            problem_type=problem_type,  # type: ignore
                            y_true=y_true,
                            y_pred=y_pred,
                            y_proba=y_proba,
                            run_id=df_hash,
                            model_name=str(eval_block.get("best_model_name", "model")),
                            model_version=str(eval_block.get("best_model_version", "")),
                            dataset_name=f"pipeline_{df_hash}",
                            metadata={"df_hash": df_hash, "source": "pipeline"},
                            compare_to="last"
                        ),
                        cfg, df_hash
                    )
                    steps_log.append(result)
                    
                    if result.ok:
                        context["perf"] = result.data
                    
                    errors_seen += (0 if result.ok else 1)
                    
                    if not result.ok and not cfg.continue_on_error:
                        return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
                else:
                    self.logger.warning("Performance tracking skipped (missing y_true/y_pred)")
            
            # 11. Drift Baseline
            if "drift_baseline" in cfg.steps and cfg.set_drift_baseline:
                result = self._execute_step(
                    "drift_baseline",
                    lambda: self._drift.execute(
                        reference_data=df,
                        current_data=df,
                        target_column=target_column
                    ),
                    cfg, df_hash
                )
                steps_log.append(result)
                
                if result.ok:
                    context["drift_baseline"] = result.data
                
                errors_seen += (0 if result.ok else 1)
                
                if not result.ok and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # 12. Report Generation
            if "report" in cfg.steps and cfg.generate_report and "eda" in context:
                data_info = {
                    "n_rows": len(df),
                    "n_columns": len(df.columns),
                    "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2)
                }
                
                result = self._execute_step(
                    "report",
                    lambda: self._report.execute(
                        eda_results=context["eda"],
                        data_info=data_info,
                        format=cfg.report_format
                    ),
                    cfg, df_hash
                )
                steps_log.append(result)
                
                if result.ok:
                    artifacts["report_path"] = result.data.get("report_path")
                
                errors_seen += (0 if result.ok else 1)
                
                if not result.ok and not cfg.continue_on_error:
                    return self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            # Pipeline complete
            result = self._finish(df_hash, t_start, ts_start, steps_log, artifacts, errors_seen, context)
            
            self.logger.success("="*80)
            self.logger.success(
                f"✓ Pipeline completed: {result.summary['ok_steps']}/{result.summary['n_steps']} "
                f"steps succeeded in {result.duration_sec:.1f}s"
            )
            self.logger.success("="*80)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            
            # Emergency finish
            return self._finish(
                df_hash, t_start, ts_start, steps_log,
                artifacts, errors_seen + 1, context
            )
    
    # ───────────────────────────────────────────────────────────────────
    # Step Execution
    # ───────────────────────────────────────────────────────────────────
    
    def _execute_step(
        self,
        name: StepName,
        func: Callable[[], AgentResult],
        config: PipelineConfig,
        df_hash: str
    ) -> StepResult:
        """
        Execute single pipeline step with retry and monitoring.
        
        Args:
            name: Step name
            func: Step execution function
            config: Pipeline configuration
            df_hash: DataFrame hash for caching
        
        Returns:
            StepResult
        """
        attempts_total = 1 + max(0, config.retries_per_step)
        delay = config.initial_backoff_s
        attempt = 0
        soft_timeout_exceeded = False
        last_result: Optional[AgentResult] = None
        warnings_acc: List[str] = []
        errors_acc: List[str] = []
        
        ts_start = _now_iso()
        self._emit_event(config, {"type": "step_start", "step": name, "ts": ts_start})
        
        t_start = time.perf_counter()
        
        # Check cache for read-only steps
        cacheable = name in ("schema", "profile")
        cache_key = (df_hash, name)
        
        if cacheable and cache_key in self._cache:
            elapsed = time.perf_counter() - t_start
            ts_finish = _now_iso()
            
            self.logger.info(f"✓ {name}: served from cache ({elapsed:.3f}s)")
            
            return StepResult(
                name=name,
                ok=True,
                started_at=ts_start,
                finished_at=ts_finish,
                duration_sec=elapsed,
                data=self._cache[cache_key],
                attempts=0,
                memory_rss_mb=_rss_memory_mb(),
                agent_version=getattr(self._get_agent(name), "version", None)
            )
        
        # Execute with retry
        while attempt < attempts_total:
            attempt += 1
            
            try:
                t_attempt = time.perf_counter()
                last_result = func()
                elapsed_attempt = time.perf_counter() - t_attempt
                
                # Check soft timeout
                if (config.soft_timeout_per_step_s and 
                    elapsed_attempt > config.soft_timeout_per_step_s):
                    soft_timeout_exceeded = True
                    warn_msg = (
                        f"Step '{name}' exceeded soft timeout: "
                        f"{elapsed_attempt:.1f}s > {config.soft_timeout_per_step_s:.1f}s"
                    )
                    self.logger.warning(warn_msg)
                    warnings_acc.append(warn_msg)
                
                # Collect errors/warnings
                if last_result.errors:
                    errors_acc.extend(last_result.errors)
                
                if last_result.warnings:
                    warnings_acc.extend(last_result.warnings)
                
                # Success?
                if last_result.is_success():
                    # Store in cache if applicable
                    if cacheable:
                        self._cache[cache_key] = last_result.data or {}
                    break
            
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                errors_acc.append(error_msg)
                self.logger.warning(
                    f"Step {name} attempt {attempt}/{attempts_total} failed: {error_msg}"
                )
                
                # Retry with backoff
                if attempt < attempts_total:
                    jitter = delay * config.jitter_fraction
                    sleep_s = delay + random.uniform(-jitter, jitter)
                    time.sleep(max(0.05, sleep_s))
                    delay *= config.backoff_multiplier
        
        # Finalize
        elapsed = time.perf_counter() - t_start
        ts_finish = _now_iso()
        
        ok = bool(last_result and last_result.is_success())
        data = (last_result.data if last_result else {}) or {}
        warnings = (last_result.warnings if last_result else []) + warnings_acc
        errors = (last_result.errors if last_result else []) + errors_acc
        
        # Check memory limits
        mem = _rss_memory_mb()
        if (mem and config.hard_stop_rss_memory_mb and 
            mem > config.hard_stop_rss_memory_mb):
            error_msg = (
                f"RSS memory {mem:.0f}MB exceeded hard limit "
                f"{config.hard_stop_rss_memory_mb}MB"
            )
            errors.append(error_msg)
            ok = False
        
        # Emit event
        self._emit_event(config, {
            "type": "step_end",
            "step": name,
            "ts": ts_finish,
            "ok": ok,
            "duration_sec": elapsed,
            "attempts": attempt,
            "soft_timeout": soft_timeout_exceeded,
            "rss_mb": mem
        })
        
        status = "✓" if ok else "✗"
        self.logger.info(
            f"{status} {name}: {'success' if ok else 'failed'} "
            f"({elapsed:.2f}s, {attempt} attempts)"
        )
        
        return StepResult(
            name=name,
            ok=ok,
            started_at=ts_start,
            finished_at=ts_finish,
            duration_sec=elapsed,
            data=data,
            warnings=warnings,
            errors=errors,
            attempts=attempt,
            soft_timeout_exceeded=soft_timeout_exceeded,
            memory_rss_mb=mem,
            agent_version=getattr(self._get_agent(name), "version", None)
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Result Finalization
    # ───────────────────────────────────────────────────────────────────
    
    def _finish(
        self,
        df_hash: str,
        t_start: float,
        ts_start: str,
        steps: List[StepResult],
        artifacts: Dict[str, Any],
        errors_seen: int,
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Finalize pipeline result.
        
        Args:
            df_hash: DataFrame hash
            t_start: Start time (perf_counter)
            ts_start: Start timestamp
            steps: List of step results
            artifacts: Artifacts dictionary
            errors_seen: Total errors
            context: Execution context
        
        Returns:
            PipelineResult
        """
        ok = all(s.ok for s in steps) and len(steps) > 0
        duration = time.perf_counter() - t_start
        ts_finish = _now_iso()
        mem = _rss_memory_mb()
        
        # Build summary
        ctx = context or {}
        
        summary: Dict[str, Any] = {
            "n_steps": len(steps),
            "ok_steps": sum(1 for s in steps if s.ok),
            "failed_steps": [s.name for s in steps if not s.ok],
            "target_column": ctx.get("target_detect", {}).get("target_column"),
            "problem_type": ctx.get("problem_classify", {}).get("problem_type"),
            "best_model": ctx.get("ml", {}).get("summary", {}).get("best_model"),
            "best_score": ctx.get("ml", {}).get("summary", {}).get("best_score"),
            "report_path": artifacts.get("report_path"),
            "agent_versions": {
                s.name: s.agent_version 
                for s in steps 
                if s.agent_version
            }
        }
        
        return PipelineResult(
            ok=ok,
            started_at=ts_start,
            finished_at=ts_finish,
            duration_sec=duration,
            df_hash=df_hash,
            steps=steps,
            artifacts=artifacts,
            summary=summary,
            memory_rss_mb=mem,
            errors_seen=errors_seen,
            version=self.version
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Utilities
    # ───────────────────────────────────────────────────────────────────
    
    def _emit_event(
        self,
        config: PipelineConfig,
        event: Dict[str, Any]
    ) -> None:
        """Emit event to callback."""
        try:
            if config.on_event:
                config.on_event(event)
        except Exception as e:
            self.logger.warning(f"Event callback failed: {e}")
    
    def _get_agent(self, step: StepName) -> Any:
        """Get agent instance for step."""
        agent_map = {
            "schema": self._schema,
            "profile": self._profiler,
            "target_detect": self._target,
            "problem_classify": self._problem,
            "missing_handle": self._missing,
            "feature_engineer": self._fe,
            "preprocess_build": self._pre,
            "eda": self._eda,
            "ml": self._ml,
            "performance_track": self._perf,
            "drift_baseline": self._drift,
            "report": self._report
        }
        return agent_map.get(step)
    
    def clear_cache(self) -> None:
        """Clear step cache."""
        self._cache.clear()
        self.logger.info("✓ Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "keys": list(self._cache.keys())
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "PipelineExecutor",
    "PipelineConfig",
    "PipelineResult",
    "StepResult",
    "ProblemType",
    "StepName",
    "DEFAULT_STEPS"
]


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print("PipelineExecutor v7.0 - Self Test")
    print("="*80)
    
    # Create sample data
    import numpy as np
    
    np.random.seed(42)
    
    sample_df = pd.DataFrame({
        "age": np.random.randint(20, 70, 100),
        "salary": np.random.randint(30000, 150000, 100),
        "experience": np.random.randint(0, 40, 100),
        "department": np.random.choice(["Sales", "Engineering", "Marketing"], 100),
        "promoted": np.random.choice([0, 1], 100)
    })
    
    print(f"\n✓ Sample data created: {sample_df.shape}")
    
    # Create executor
    executor = PipelineExecutor()
    print(f"✓ Executor initialized: v{executor.version}")
    
    # Configure minimal pipeline
    config = PipelineConfig(
        steps=["schema", "profile", "problem_classify"],
        continue_on_error=True,
        retries_per_step=1,
        ml_enabled=False,
        generate_report=False
    )
    
    print(f"\n✓ Config: {len(config.steps)} steps")
    
    # Execute
    print("\n" + "="*80)
    print("Executing pipeline...")
    print("="*80)
    
    result = executor.run(
        df=sample_df,
        target_column="promoted",
        problem_type="classification",
        config=config
    )
    
    print("\n" + "="*80)
    print("Pipeline Result:")
    print("="*80)
    
    print(f"\nStatus: {'✓ SUCCESS' if result.ok else '✗ FAILED'}")
    print(f"Duration: {result.duration_sec:.2f}s")
    print(f"Steps: {result.summary['ok_steps']}/{result.summary['n_steps']} succeeded")
    
    if result.summary['failed_steps']:
        print(f"Failed: {result.summary['failed_steps']}")
    
    print(f"\nSteps executed:")
    for step in result.steps:
        status = "✓" if step.ok else "✗"
        print(f"  {status} {step.name}: {step.duration_sec:.3f}s ({step.attempts} attempts)")
    
    print(f"\nCache info: {executor.get_cache_info()}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from backend.pipeline_executor import PipelineExecutor, PipelineConfig

# Create executor
executor = PipelineExecutor()

# Configure pipeline
config = PipelineConfig(
    steps=["schema", "profile", "eda", "ml", "report"],
    continue_on_error=True,
    retries_per_step=2,
    random_seed=42,
    ml_enabled=True,
    generate_report=True,
    soft_timeout_per_step_s=300.0
)

# Execute
result = executor.run(
    df=train_df,
    target_column="target",
    problem_type="classification",
    config=config
)

# Check results
if result.ok:
    print(f"✓ Pipeline succeeded in {result.duration_sec:.1f}s")
    print(f"  Best model: {result.summary['best_model']}")
    print(f"  Best score: {result.summary['best_score']:.4f}")
    print(f"  Report: {result.artifacts.get('report_path')}")
else:
    print(f"✗ Pipeline failed with {result.errors_seen} errors")
    for step in result.get_failed_steps():
        print(f"  {step.name}: {step.errors}")

# Get warnings
for step_name, warnings in result.get_warnings():
    print(f"Warnings in {step_name}:")
    for warn in warnings:
        print(f"  - {warn}")
    """)
