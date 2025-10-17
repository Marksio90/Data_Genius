# backend/app_controller.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — App Controller v7.0              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 CENTRAL APPLICATION ORCHESTRATION CONTROLLER                          ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ End-to-End ML Pipeline Coordination                                   ║
║  ✓ Agent Lifecycle Management                                            ║
║  ✓ State Management & Caching                                            ║
║  ✓ Monitoring & Drift Detection                                          ║
║  ✓ Performance Tracking                                                  ║
║  ✓ Report Generation                                                     ║
║  ✓ Comprehensive Error Handling                                          ║
║  ✓ Telemetry & Logging                                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    AppController                             │
    ├──────────────────────────────────────────────────────────────┤
    │  • Data Loading & Validation                                 │
    │  • Schema Analysis & Profiling                               │
    │  • Target Detection & Problem Classification                 │
    │  • EDA Orchestration                                         │
    │  • Preprocessing Pipeline                                    │
    │  • ML Training & Evaluation                                  │
    │  • Model Explanation                                         │
    │  • Report Generation                                         │
    │  • Drift Detection                                           │
    │  • Performance Tracking                                      │
    │  • Retraining Scheduling                                     │
    └──────────────────────────────────────────────────────────────┘

Usage:
```python
    from backend.app_controller import AppController
    
    # Initialize controller
    ctrl = AppController()
    
    # Load data
    df = ctrl.load_dataframe(csv_text="age,salary\\n25,50000\\n30,60000")
    
    # Run end-to-end pipeline
    results = ctrl.run_end_to_end(
        df=df,
        target_column="salary",
        problem_type="regression"
    )
    
    # Or step-by-step
    schema = ctrl.analyze_schema(df)
    target = ctrl.detect_target(df)
    eda = ctrl.run_eda(df, target_column=target.data['target_column'])
    ml = ctrl.run_ml_pipeline(df, target_column, problem_type)
```

Dependencies:
    • pandas, numpy
    • loguru
    • agents.* (all agent modules)
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

try:
    from config.settings import settings
except ImportError:
    logger.warning("⚠ config.settings not found - using defaults")
    settings = type('Settings', (), {})()

# ═══════════════════════════════════════════════════════════════════════════
# Core Imports
# ═══════════════════════════════════════════════════════════════════════════

from core.base_agent import AgentResult

# EDA Agents
try:
    from agents.eda.schema_analyzer import SchemaAnalyzer
    from agents.eda.data_profiler import DataProfiler
    from agents.eda.eda_orchestrator import EDAOrchestrator
    from agents.eda.problem_classifier import ProblemClassifier
    from agents.eda.missing_data_analyzer import MissingDataAnalyzer
    from agents.eda.correlation_analyzer import CorrelationAnalyzer
except ImportError as e:
    logger.warning(f"⚠ EDA agents not fully available: {e}")

# Target Detection
try:
    from agents.target.target_detector import TargetDetector
except ImportError as e:
    logger.warning(f"⚠ Target detector not available: {e}")

# Mentor (LLM)
try:
    from agents.mentor.mentor_orchestrator import MentorOrchestrator
except ImportError as e:
    logger.warning(f"⚠ Mentor orchestrator not available: {e}")

# Preprocessing
try:
    from agents.preprocessing.missing_data_handler import MissingDataHandler
    from agents.preprocessing.feature_engineer import FeatureEngineer
    from agents.preprocessing.pipeline_builder import PipelineBuilder
    from agents.preprocessing.encoder_selector import EncoderSelector, EncoderSelectorConfig
    from agents.preprocessing.scaler_selector import ScalerSelector, ScalerSelectorConfig
except ImportError as e:
    logger.warning(f"⚠ Preprocessing agents not fully available: {e}")

# ML
try:
    from agents.ml.ml_orchestrator import MLOrchestrator
    from agents.ml.model_evaluator import ModelEvaluator
    from agents.ml.model_explainer import ModelExplainer
except ImportError as e:
    logger.warning(f"⚠ ML agents not fully available: {e}")

# Monitoring
try:
    from agents.monitoring.drift_detector import DriftDetector, DriftReference
    from agents.monitoring.performance_tracker import PerformanceTracker, RunMetadata
    from agents.monitoring.retraining_scheduler import RetrainingScheduler, RetrainingPolicy
except ImportError as e:
    logger.warning(f"⚠ Monitoring agents not fully available: {e}")

# Reporting
try:
    from agents.reporting.report_generator import ReportGenerator
except ImportError as e:
    logger.warning(f"⚠ Report generator not available: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Type Definitions
# ═══════════════════════════════════════════════════════════════════════════

ProblemType = Literal["classification", "regression"]
ImputationStrategy = Literal["auto", "mean", "median", "mode", "knn", "drop"]
ReportFormat = Literal["html", "pdf", "markdown"]
RetrainingSchedule = Literal["now", "nightly", "weekly"]


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

LOG_TRUNCATE = 500  # Max characters for logged structures
DEFAULT_DRIFT_THRESHOLD = 0.1
MAX_HASH_ROWS = 100_000


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _safe_len(x: Any) -> int:
    """Safe length calculation."""
    try:
        return len(x)  # type: ignore
    except Exception:
        return 0


def _truncate_for_log(obj: Any, limit: int = LOG_TRUNCATE) -> str:
    """
    Truncate object representation for logging.
    
    Args:
        obj: Object to truncate
        limit: Character limit
    
    Returns:
        Truncated string representation
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    
    if len(s) > limit:
        return s[:limit] + f"... (+{len(s)-limit} chars)"
    return s


def _hash_dataframe(df: pd.DataFrame, max_rows: int = MAX_HASH_ROWS) -> str:
    """
    Create stable DataFrame hash for caching/telemetry.
    
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
        # Fallback hash
        return f"h{hash((tuple(df.columns), df.shape)) & 0xFFFFFFFF:X}"


def _timed_exec(
    name: str,
    func: Callable[..., AgentResult],
    /,
    **kwargs: Any
) -> AgentResult:
    """
    ⏱️ **Execute Agent with Timing & Error Handling**
    
    Wraps agent execution with:
      • Performance timing
      • Structured logging
      • Graceful error recovery
      • Always returns AgentResult
    
    Args:
        name: Agent name for logging
        func: Agent execute function
        **kwargs: Arguments to pass to function
    
    Returns:
        AgentResult (always, even on exception)
    """
    t0 = time.perf_counter()
    
    try:
        res = func(**kwargs)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"{name}: exception after {elapsed_ms:.1f}ms: {e}")
        return AgentResult(
            agent_name=name,
            errors=[str(e)],
            data={}
        )
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    # Validate result
    try:
        is_success = res.is_success()
    except Exception:
        logger.warning(f"{name}: returned non-standard result in {elapsed_ms:.1f}ms")
        return res
    
    if is_success:
        logger.info(f"✓ {name}: success in {elapsed_ms:.1f}ms")
    else:
        logger.warning(
            f"✗ {name}: failed in {elapsed_ms:.1f}ms → "
            f"{_truncate_for_log(res.errors)}"
        )
    
    return res


# ═══════════════════════════════════════════════════════════════════════════
# Application State
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    """
    💾 **Application Session State**
    
    Lightweight state tracking current analysis session.
    Does not store heavy binary artifacts (models, large DataFrames).
    
    Attributes:
        df_hash: DataFrame content hash
        n_rows: Number of rows
        n_cols: Number of columns
        columns: Column names
        target_column: Detected/specified target
        problem_type: ML problem type
        last_eda_summary: EDA results summary
        last_ml_summary: ML results summary
        baseline_reference: Drift detection baseline
        last_metrics: Latest model metrics
    """
    
    # Data info
    df_hash: Optional[str] = None
    n_rows: int = 0
    n_cols: int = 0
    columns: List[str] = field(default_factory=list)
    
    # ML info
    target_column: Optional[str] = None
    problem_type: Optional[ProblemType] = None
    
    # Results snapshots
    last_eda_summary: Dict[str, Any] = field(default_factory=dict)
    last_ml_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring
    baseline_reference: Optional[DriftReference] = None
    last_metrics: Dict[str, float] = field(default_factory=dict)
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)
    
    def reset(self) -> None:
        """Reset state to initial values."""
        self.df_hash = None
        self.n_rows = 0
        self.n_cols = 0
        self.columns = []
        self.target_column = None
        self.problem_type = None
        self.last_eda_summary = {}
        self.last_ml_summary = {}
        self.baseline_reference = None
        self.last_metrics = {}


# ═══════════════════════════════════════════════════════════════════════════
# Main Controller
# ═══════════════════════════════════════════════════════════════════════════

class AppController:
    """
    🎮 **Central Application Controller**
    
    Orchestrates complete ML workflow:
      1. Data loading & validation
      2. Schema analysis & profiling
      3. Target detection & problem classification
      4. EDA & visualization
      5. Preprocessing pipeline
      6. ML training & evaluation
      7. Model explanation
      8. Report generation
      9. Monitoring & drift detection
      10. Performance tracking
      11. Retraining scheduling
    
    Features:
      • Stateful session management
      • Agent lifecycle coordination
      • Comprehensive error handling
      • Performance monitoring
      • Structured logging
    
    Usage:
```python
        ctrl = AppController()
        
        # Load data
        df = ctrl.load_dataframe(csv_text="...")
        
        # Run complete pipeline
        results = ctrl.run_end_to_end(df)
        
        # Or step-by-step
        schema = ctrl.analyze_schema(df)
        profile = ctrl.profile_data(df)
        target = ctrl.detect_target(df)
        eda = ctrl.run_eda(df, target.data['target_column'])
        ml = ctrl.run_ml_pipeline(df, target_column, problem_type)
```
    """
    
    def __init__(self) -> None:
        """Initialize controller with agent instances."""
        self.logger = logger.bind(component="AppController")
        self.state = AppState()
        
        # Initialize agent instances (singletons for reuse)
        self._init_agents()
        
        self.logger.info("✓ AppController initialized")
    
    def _init_agents(self) -> None:
        """Initialize all agent instances."""
        try:
            # EDA
            self._schema_analyzer = SchemaAnalyzer()
            self._data_profiler = DataProfiler()
            self._eda = EDAOrchestrator()
            self._missing_an = MissingDataAnalyzer()
            self._corr_an = CorrelationAnalyzer()
            self._problem_cls = ProblemClassifier()
            
            # Target & Mentor
            self._target_det = TargetDetector()
            self._mentor = MentorOrchestrator()
            
            # Preprocessing
            self._imputer = MissingDataHandler()
            self._fe = FeatureEngineer()
            self._pipe = PipelineBuilder()
            self._enc_sel = EncoderSelector(EncoderSelectorConfig())
            self._scaler_sel = ScalerSelector(ScalerSelectorConfig())
            
            # ML
            self._ml = MLOrchestrator()
            self._evaluator = ModelEvaluator()
            self._explainer = ModelExplainer()
            
            # Monitoring
            self._drift = DriftDetector()
            self._perf = PerformanceTracker()
            self._retrain = RetrainingScheduler()
            
            # Reporting
            self._reporter = ReportGenerator()
            
            self.logger.debug("All agents initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Agent initialization warning: {e}")
    
    # ───────────────────────────────────────────────────────────────────
    # Data Loading
    # ───────────────────────────────────────────────────────────────────
    
    def load_dataframe(
        self,
        *,
        records: Optional[List[Dict[str, Any]]] = None,
        csv_text: Optional[str] = None,
        sep: str = ",",
        encoding: str = "utf-8",
        enforce_non_empty: bool = True
    ) -> pd.DataFrame:
        """
        📊 **Load DataFrame from Records or CSV**
        
        Args:
            records: List of dicts (row = dict column→value)
            csv_text: CSV content as string
            sep: CSV separator
            encoding: Text encoding
            enforce_non_empty: Raise if empty
        
        Returns:
            Loaded DataFrame
        
        Raises:
            ValueError: If no data source or DataFrame empty
            TypeError: If invalid data type
        
        Example:
```python
            # From records
            df = ctrl.load_dataframe(
                records=[{"age": 25}, {"age": 30}]
            )
            
            # From CSV
            df = ctrl.load_dataframe(
                csv_text="age,salary\\n25,50000\\n30,60000"
            )
```
        """
        if not records and not csv_text:
            raise ValueError("Provide either 'records' or 'csv_text'")
        
        try:
            if records is not None:
                if not isinstance(records, list):
                    raise TypeError("'records' must be a list[dict]")
                if len(records) > 0 and not isinstance(records[0], dict):
                    raise TypeError("'records' must be list[dict]")
                
                df = pd.DataFrame(records)
            
            else:
                if not isinstance(csv_text, str):
                    raise TypeError("'csv_text' must be a string")
                
                df = pd.read_csv(
                    io.StringIO(csv_text),
                    sep=sep,
                    encoding=encoding
                )
        
        except Exception as e:
            self.logger.error(f"DataFrame loading failed: {e}")
            raise
        
        if enforce_non_empty and (df is None or df.empty):
            raise ValueError("Parsed DataFrame is empty")
        
        # Update state
        self.state.df_hash = _hash_dataframe(df)
        self.state.n_rows, self.state.n_cols = df.shape
        self.state.columns = df.columns.tolist()
        
        self.logger.success(
            f"✓ Loaded DataFrame: shape={df.shape}, hash={self.state.df_hash}, "
            f"cols={len(self.state.columns)}"
        )
        
        return df
    
    # ───────────────────────────────────────────────────────────────────
    # Schema & Profiling
    # ───────────────────────────────────────────────────────────────────
    
    def analyze_schema(self, df: pd.DataFrame) -> AgentResult:
        """
        🔍 **Analyze Data Schema**
        
        Analyzes column types, distributions, and characteristics.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            AgentResult with schema information
        """
        res = _timed_exec(
            "SchemaAnalyzer.execute",
            self._schema_analyzer.execute,
            data=df
        )
        
        if res.is_success():
            n_cols = _safe_len(res.data.get("columns", []))
            self.logger.info(f"Schema analyzed: {n_cols} columns")
        
        return res
    
    def profile_data(self, df: pd.DataFrame) -> AgentResult:
        """
        📈 **Generate Data Profile**
        
        Comprehensive data profiling with statistics and distributions.
        
        Args:
            df: DataFrame to profile
        
        Returns:
            AgentResult with profiling information
        """
        return _timed_exec(
            "DataProfiler.execute",
            self._data_profiler.execute,
            data=df
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Target Detection & Problem Classification
    # ───────────────────────────────────────────────────────────────────
    
    def detect_target(
        self,
        df: pd.DataFrame,
        schema_columns_info: Optional[List[Dict[str, Any]]] = None,
        user_target: Optional[str] = None
    ) -> AgentResult:
        """
        🎯 **Auto-Detect Target Column**
        
        Uses LLM and heuristics to detect target column.
        
        Args:
            df: DataFrame
            schema_columns_info: Pre-computed schema info
            user_target: User-specified target (overrides detection)
        
        Returns:
            AgentResult with detected target
        """
        if schema_columns_info is None:
            schema_res = self._schema_analyzer.execute(data=df)
            if not schema_res.is_success():
                return schema_res
            schema_columns_info = schema_res.data.get("columns", [])
        
        res = _timed_exec(
            "TargetDetector.execute",
            self._target_det.execute,
            data=df,
            column_info=schema_columns_info,
            user_target=user_target
        )
        
        if res.is_success():
            self.state.target_column = res.data.get("target_column")
            self.logger.info(f"Target detected: {self.state.target_column}")
        
        return res
    
    def classify_problem(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> AgentResult:
        """
        🏷️ **Classify ML Problem Type**
        
        Determines if problem is classification or regression.
        
        Args:
            df: DataFrame
            target_column: Target column name
        
        Returns:
            AgentResult with problem type
        """
        res = _timed_exec(
            "ProblemClassifier.execute",
            self._problem_cls.execute,
            data=df,
            target_column=target_column
        )
        
        if res.is_success():
            self.state.problem_type = res.data.get("problem_type")
            self.logger.info(f"Problem type: {self.state.problem_type}")
        
        return res
    
    # ───────────────────────────────────────────────────────────────────
    # EDA
    # ───────────────────────────────────────────────────────────────────
    
    def run_eda(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> AgentResult:
        """
        🔬 **Run Exploratory Data Analysis**
        
        Complete EDA with visualizations and insights.
        
        Args:
            df: DataFrame to analyze
            target_column: Target column (optional)
        
        Returns:
            AgentResult with EDA results
        """
        res = _timed_exec(
            "EDAOrchestrator.execute",
            self._eda.execute,
            data=df,
            target_column=target_column
        )
        
        if res.is_success():
            self.state.last_eda_summary = res.data.get("summary", {})
        
        return res
    
    # ───────────────────────────────────────────────────────────────────
    # Preprocessing
    # ───────────────────────────────────────────────────────────────────
    
    def handle_missing(
        self,
        df: pd.DataFrame,
        target_column: str,
        strategy: ImputationStrategy = "auto"
    ) -> AgentResult:
        """
        🔧 **Handle Missing Values**
        
        Imputes missing values using specified strategy.
        
        Args:
            df: DataFrame
            target_column: Target column
            strategy: Imputation strategy
        
        Returns:
            AgentResult with imputed data
        """
        return _timed_exec(
            "MissingDataHandler.execute",
            self._imputer.execute,
            data=df,
            target_column=target_column,
            strategy=strategy
        )
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> AgentResult:
        """
        ⚙️ **Engineer Features**
        
        Creates derived features (dates, interactions, polynomial, binning).
        
        Args:
            df: DataFrame
            target_column: Target column
        
        Returns:
            AgentResult with engineered features
        """
        return _timed_exec(
            "FeatureEngineer.execute",
            self._fe.execute,
            data=df,
            target_column=target_column
        )
    
    def select_encoders(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType
    ) -> AgentResult:
        """
        🔤 **Select Categorical Encoders**
        
        Recommends encoding strategies for categorical features.
        
        Args:
            df: DataFrame
            target_column: Target column
            problem_type: ML problem type
        
        Returns:
            AgentResult with encoder recommendations
        """
        return _timed_exec(
            "EncoderSelector.execute",
            self._enc_sel.execute,
            data=df,
            target_column=target_column,
            problem_type=problem_type
        )
    
    def select_scalers(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        estimator_hint: Optional[Literal[
            "tree", "linear", "svm", "nn", "boosting", "knn"
        ]] = None,
        prefer_global: Optional[bool] = None
    ) -> AgentResult:
        """
        📏 **Select Scaling Strategy**
        
        Recommends scaling strategies for numeric features.
        
        Args:
            df: DataFrame
            target_column: Target column
            estimator_hint: Algorithm hint
            prefer_global: Use global strategy
        
        Returns:
            AgentResult with scaler recommendations
        """
        return _timed_exec(
            "ScalerSelector.execute",
            self._scaler_sel.execute,
            data=df,
            target_column=target_column,
            estimator_hint=estimator_hint,
            prefer_global=prefer_global
        )
    
    def build_preprocessing(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType
    ) -> AgentResult:
        """
        🔧 **Build Preprocessing Pipeline**
        
        Complete preprocessing pipeline (impute + encode + scale).
        
        Args:
            df: DataFrame
            target_column: Target column
            problem_type: ML problem type
        
        Returns:
            AgentResult with preprocessing pipeline
        """
        return _timed_exec(
            "PipelineBuilder.execute",
            self._pipe.execute,
            data=df,
            target_column=target_column,
            problem_type=problem_type
        )
    
    # ───────────────────────────────────────────────────────────────────
    # ML Pipeline
    # ───────────────────────────────────────────────────────────────────
    
    def run_ml_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType
    ) -> AgentResult:
        """
        🤖 **Run ML Pipeline**
        
        Complete ML pipeline: select + train + evaluate + explain.
        
        Args:
            df: DataFrame
            target_column: Target column
            problem_type: ML problem type
        
        Returns:
            AgentResult with ML results
        """
        res = _timed_exec(
            "MLOrchestrator.execute",
            self._ml.execute,
            data=df,
            target_column=target_column,
            problem_type=problem_type
        )
        
        if res.is_success():
            self.state.last_ml_summary = res.data.get("summary", {})
        
        return res
    
    # ───────────────────────────────────────────────────────────────────
    # LLM Mentoring
    # ───────────────────────────────────────────────────────────────────
    
    def mentor_explain_eda(
        self,
        eda_results: Dict[str, Any],
        user_level: str = "beginner"
    ) -> str:
        """
        💬 **Get LLM Explanation of EDA Results**
        
        Args:
            eda_results: EDA results dict
            user_level: User expertise level
        
        Returns:
            Human-readable explanation
        """
        try:
            return self._mentor.explain_eda_results(
                eda_results,
                user_level=user_level
            )
        except Exception as e:
            self.logger.error(f"Mentor EDA explanation failed: {e}")
            return "Failed to generate EDA explanation"
    
    def mentor_explain_ml(
        self,
        ml_results: Dict[str, Any],
        user_level: str = "beginner"
    ) -> str:
        """
        💬 **Get LLM Explanation of ML Results**
        
        Args:
            ml_results: ML results dict
            user_level: User expertise level
        
        Returns:
            Human-readable explanation
        """
        try:
            return self._mentor.explain_ml_results(
                ml_results,
                user_level=user_level
            )
        except Exception as e:
            self.logger.error(f"Mentor ML explanation failed: {e}")
            return "Failed to generate ML explanation"
    
    # ───────────────────────────────────────────────────────────────────
    # Reporting
    # ───────────────────────────────────────────────────────────────────
    
    def generate_report(
        self,
        eda_results: Dict[str, Any],
        data_info: Dict[str, Any],
        fmt: ReportFormat = "html",
        output_path: Optional[Path] = None
    ) -> AgentResult:
        """
        📄 **Generate EDA Report**
        
        Args:
            eda_results: EDA results
            data_info: Dataset metadata
            fmt: Output format (html/pdf/markdown)
            output_path: Optional output path
        
        Returns:
            AgentResult with report path
        """
        return _timed_exec(
            "ReportGenerator.execute",
            self._reporter.execute,
            eda_results=eda_results,
            data_info=data_info,
            format=fmt,
            output_path=output_path
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Monitoring & Drift
    # ───────────────────────────────────────────────────────────────────
    
    def track_performance(
        self,
        model_name: str,
        problem_type: ProblemType,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> AgentResult:
        """
        📊 **Track Model Performance**
        
        Logs metrics and run metadata.
        
        Args:
            model_name: Model identifier
            problem_type:
            Problem type
            metrics: Performance metrics
            params: Model parameters
            tags: Run tags
        
        Returns:
            AgentResult with tracking info
        """
        meta = RunMetadata(
            model_name=model_name,
            problem_type=problem_type,
            params=params or {},
            tags=tags or []
        )
        
        res = _timed_exec(
            "PerformanceTracker.log_run",
            self._perf.log_run,
            metrics=metrics,
            metadata=meta
        )
        
        if res.is_success():
            self.state.last_metrics = metrics
        
        return res
    
    def set_drift_baseline(
        self,
        df_reference: pd.DataFrame,
        target_column: Optional[str] = None,
        name: str = "baseline"
    ) -> AgentResult:
        """
        📍 **Set Drift Detection Baseline**
        
        Sets reference distribution for drift detection.
        
        Args:
            df_reference: Reference DataFrame (training data)
            target_column: Target column
            name: Baseline name
        
        Returns:
            AgentResult with baseline info
        """
        res = _timed_exec(
            "DriftDetector.set_reference",
            self._drift.set_reference,
            data=df_reference,
            target_column=target_column,
            name=name
        )
        
        if res.is_success():
            self.state.baseline_reference = res.data.get("reference")
            self.logger.info(f"Drift baseline set: {name}")
        
        return res
    
    def check_drift(
        self,
        df_current: pd.DataFrame,
        threshold: float = DEFAULT_DRIFT_THRESHOLD
    ) -> AgentResult:
        """
        🔍 **Check for Data Drift**
        
        Detects drift vs. baseline using statistical tests.
        
        Args:
            df_current: Current data
            threshold: Drift threshold
        
        Returns:
            AgentResult with drift detection results
        
        Raises:
            ValueError: If no baseline set
        """
        if not self.state.baseline_reference:
            return AgentResult(
                agent_name="AppController",
                errors=["No drift baseline set. Use set_drift_baseline() first."],
                data={}
            )
        
        return _timed_exec(
            "DriftDetector.execute",
            self._drift.execute,
            current_data=df_current,
            reference=self.state.baseline_reference,
            threshold=threshold
        )
    
    def maybe_schedule_retraining(
        self,
        project: str,
        model_name: str,
        when: RetrainingSchedule = "nightly",
        drift_alert: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        🔄 **Schedule Model Retraining**
        
        Schedules retraining based on drift or policy.
        
        Args:
            project: Project identifier
            model_name: Model identifier
            when: Schedule (now/nightly/weekly)
            drift_alert: Drift detection results
        
        Returns:
            AgentResult with schedule info
        """
        # Map schedule to policy
        if when == "now":
            policy = RetrainingPolicy.asap()
        elif when == "nightly":
            policy = RetrainingPolicy.daily(hour=2, minute=30)
        else:  # weekly
            policy = RetrainingPolicy.weekly(day_of_week="SUN", hour=3, minute=0)
        
        return _timed_exec(
            "RetrainingScheduler.schedule",
            self._retrain.schedule,
            project=project,
            model_name=model_name,
            policy=policy,
            reason=drift_alert or {}
        )
    
    # ───────────────────────────────────────────────────────────────────
    # End-to-End Pipeline
    # ───────────────────────────────────────────────────────────────────
    
    def run_end_to_end(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: Optional[ProblemType] = None,
        generate_html_report: bool = True,
        track_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        🚀 **Run Complete End-to-End Pipeline**
        
        Executes complete ML workflow:
          1. Schema analysis & profiling
          2. Target detection & problem classification
          3. EDA with visualizations
          4. Preprocessing pipeline
          5. ML training & evaluation
          6. Report generation (optional)
          7. Performance tracking (optional)
        
        Args:
            df: Input DataFrame
            target_column: Target column (auto-detect if None)
            problem_type: Problem type (auto-detect if None)
            generate_html_report: Generate HTML report
            track_metrics: Track performance metrics
        
        Returns:
            Dictionary with all pipeline outputs
        
        Example:
```python
            ctrl = AppController()
            df = ctrl.load_dataframe(csv_text="...")
            
            results = ctrl.run_end_to_end(
                df=df,
                generate_html_report=True
            )
            
            # Access results
            print(f"Target: {results['target_column']}")
            print(f"Problem: {results['problem_type']}")
            print(f"Best model: {results['ml']['best_model']}")
```
        """
        outputs: Dict[str, Any] = {
            "ts": datetime.now().isoformat(),
            "status": "running"
        }
        
        t_start = time.perf_counter()
        
        try:
            self.logger.info("="*80)
            self.logger.info("🚀 Starting end-to-end ML pipeline")
            self.logger.info("="*80)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: Schema & Profiling
            # ═══════════════════════════════════════════════════════════
            
            self.logger.info("📊 Stage 1: Schema analysis & profiling")
            
            schema = self.analyze_schema(df)
            if not schema.is_success():
                raise RuntimeError(f"Schema analysis failed: {schema.errors}")
            
            profile = self.profile_data(df)
            if not profile.is_success():
                self.logger.warning("Data profiling failed - continuing anyway")
            
            outputs["schema"] = schema.data
            outputs["profile"] = profile.data if profile.is_success() else {}
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: Target Detection & Problem Classification
            # ═══════════════════════════════════════════════════════════
            
            self.logger.info("🎯 Stage 2: Target detection & problem classification")
            
            # Target detection
            if not target_column:
                tgt = self.detect_target(
                    df,
                    schema_columns_info=schema.data.get("columns", [])
                )
                if tgt.is_success():
                    target_column = tgt.data.get("target_column")
                    outputs["target_detection"] = tgt.data
            
            if not target_column:
                raise ValueError(
                    "Target column not provided and could not be detected. "
                    "Please specify target_column explicitly."
                )
            
            outputs["target_column"] = target_column
            self.logger.info(f"✓ Target column: {target_column}")
            
            # Problem type classification
            if not problem_type:
                pcls = self.classify_problem(df, target_column=target_column)
                if pcls.is_success():
                    problem_type = pcls.data.get("problem_type")
                    outputs["problem_classification"] = pcls.data
            
            if not problem_type:
                raise ValueError(
                    "Problem type not provided and could not be inferred. "
                    "Please specify problem_type explicitly."
                )
            
            outputs["problem_type"] = problem_type
            self.logger.info(f"✓ Problem type: {problem_type}")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: EDA
            # ═══════════════════════════════════════════════════════════
            
            self.logger.info("🔬 Stage 3: Exploratory data analysis")
            
            eda = self.run_eda(df, target_column=target_column)
            if not eda.is_success():
                self.logger.warning("EDA failed - continuing with limited analysis")
                outputs["eda"] = {"error": eda.errors}
            else:
                outputs["eda"] = eda.data
                self.logger.info("✓ EDA completed successfully")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 4: Preprocessing Pipeline
            # ═══════════════════════════════════════════════════════════
            
            self.logger.info("🔧 Stage 4: Preprocessing pipeline")
            
            preproc = self.build_preprocessing(
                df,
                target_column=target_column,
                problem_type=problem_type
            )
            
            if not preproc.is_success():
                self.logger.warning("Preprocessing failed - ML may use raw data")
                outputs["preprocessing"] = {"error": preproc.errors}
            else:
                outputs["preprocessing"] = preproc.data
                self.logger.info("✓ Preprocessing pipeline built")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 5: ML Training & Evaluation
            # ═══════════════════════════════════════════════════════════
            
            self.logger.info("🤖 Stage 5: ML training & evaluation")
            
            ml = self.run_ml_pipeline(
                df,
                target_column=target_column,
                problem_type=problem_type
            )
            
            if not ml.is_success():
                self.logger.error("ML pipeline failed")
                outputs["ml"] = {"error": ml.errors}
            else:
                outputs["ml"] = ml.data
                self.logger.info("✓ ML pipeline completed")
                
                # Track metrics if requested
                if track_metrics and ml.data.get("metrics"):
                    try:
                        self.track_performance(
                            model_name=ml.data.get("best_model", "unknown"),
                            problem_type=problem_type,
                            metrics=ml.data["metrics"],
                            params=ml.data.get("params", {}),
                            tags=["end_to_end"]
                        )
                    except Exception as e:
                        self.logger.warning(f"Performance tracking failed: {e}")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 6: Report Generation (Optional)
            # ═══════════════════════════════════════════════════════════
            
            if generate_html_report and eda.is_success():
                self.logger.info("📄 Stage 6: Report generation")
                
                data_info = {
                    "n_rows": int(len(df)),
                    "n_columns": int(len(df.columns)),
                    "memory_mb": float(
                        df.memory_usage(deep=True).sum() / 1024**2
                    ),
                    "target_column": target_column,
                    "problem_type": problem_type
                }
                
                rep = self.generate_report(
                    eda_results=eda.data,
                    data_info=data_info,
                    fmt="html"
                )
                
                if rep.is_success():
                    outputs["report_path"] = rep.data.get("report_path")
                    self.logger.info(f"✓ Report generated: {outputs['report_path']}")
            
            # ═══════════════════════════════════════════════════════════
            # Pipeline Complete
            # ═══════════════════════════════════════════════════════════
            
            elapsed_s = time.perf_counter() - t_start
            
            outputs["status"] = "success"
            outputs["elapsed_s"] = round(elapsed_s, 2)
            
            self.logger.success("="*80)
            self.logger.success(
                f"✓ End-to-end pipeline completed in {elapsed_s:.1f}s"
            )
            self.logger.success("="*80)
            
            return outputs
        
        except Exception as e:
            elapsed_s = time.perf_counter() - t_start
            
            self.logger.error("="*80)
            self.logger.error(f"✗ Pipeline failed after {elapsed_s:.1f}s: {e}")
            self.logger.error("="*80)
            
            outputs["status"] = "error"
            outputs["error"] = str(e)
            outputs["elapsed_s"] = round(elapsed_s, 2)
            
            return outputs
    
    # ───────────────────────────────────────────────────────────────────
    # Utility Methods
    # ───────────────────────────────────────────────────────────────────
    
    def get_state(self) -> Dict[str, Any]:
        """Get current application state."""
        return self.state.as_dict()
    
    def reset_state(self) -> None:
        """Reset application state."""
        self.state.reset()
        self.logger.info("✓ Application state reset")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get pipeline summary.
        
        Returns:
            Summary of current state and last results
        """
        return {
            "state": self.state.as_dict(),
            "last_eda": self.state.last_eda_summary,
            "last_ml": self.state.last_ml_summary,
            "last_metrics": self.state.last_metrics
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "AppController",
    "AppState",
    "ProblemType",
    "ImputationStrategy",
    "ReportFormat",
    "RetrainingSchedule"
]


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print("AppController v7.0 - Self Test")
    print("="*80)
    
    # Create controller
    ctrl = AppController()
    
    # Create sample data
    sample_data = """
age,salary,department,years_experience,performance_rating,promoted
25,50000,Engineering,2,3.5,0
30,60000,Sales,5,4.2,1
35,75000,Engineering,8,4.8,1
28,55000,Marketing,3,3.8,0
32,70000,Sales,6,4.5,1
    """.strip()
    
    print("\n✓ Testing data loading...")
    try:
        df = ctrl.load_dataframe(csv_text=sample_data)
        print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Hash: {ctrl.state.df_hash}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n✓ Testing schema analysis...")
    try:
        schema = ctrl.analyze_schema(df)
        if schema.is_success():
            print(f"  Analyzed: {len(schema.data.get('columns', []))} columns")
        else:
            print(f"  ✗ Failed: {schema.errors}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    print("\n✓ Testing state management...")
    state = ctrl.get_state()
    print(f"  State keys: {list(state.keys())}")
    print(f"  Rows: {state['n_rows']}, Cols: {state['n_cols']}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from backend.app_controller import AppController

# Initialize
ctrl = AppController()

# Load data
df = ctrl.load_dataframe(csv_text="age,salary\\n25,50000\\n30,60000")

# Run complete pipeline
results = ctrl.run_end_to_end(
    df=df,
    target_column="salary",
    problem_type="regression",
    generate_html_report=True
)

# Check results
print(f"Status: {results['status']}")
print(f"Elapsed: {results['elapsed_s']}s")
print(f"Report: {results.get('report_path')}")

# Or step-by-step
schema = ctrl.analyze_schema(df)
target = ctrl.detect_target(df)
eda = ctrl.run_eda(df, target.data['target_column'])
ml = ctrl.run_ml_pipeline(df, target_column, problem_type)

# Access state
state = ctrl.get_state()
summary = ctrl.get_summary()
    """)
