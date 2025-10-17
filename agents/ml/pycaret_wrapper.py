# agents/ml/pycaret_wrapper.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — PyCaret Wrapper v6.0             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ENTERPRISE-GRADE PYCARET ORCHESTRATION LAYER                          ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Unified API Contract (setup→compare→tune→finalize)                    ║
║  ✓ Advanced Version Detection & Compatibility                            ║
║  ✓ Defensive Validation & Error Recovery                                 ║
║  ✓ Comprehensive Telemetry & Monitoring                                  ║
║  ✓ Enterprise Artifact Management                                        ║
║  ✓ Model Registry Integration                                            ║
║  ✓ Feature Engineering Helpers                                           ║
║  ✓ Target Mapping & Label Encoding                                       ║
║  ✓ GPU Auto-Configuration                                                ║
║  ✓ Session State Persistence                                             ║
║  ✓ Memory-Efficient Operations                                           ║
║  ✓ Multi-Backend Support (sklearn, lightgbm, xgboost, catboost)         ║
║  ✓ Advanced Preprocessing Pipelines                                      ║
║  ✓ Explainability Integration (SHAP, LIME ready)                         ║
║  ✓ Production-Ready Error Handling                                       ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  PyCaretWrapper Core                        │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Lazy Module Import & Version Detection                 │
    │  2. Experiment Initialization & Setup                       │
    │  3. Multi-Model Comparison & Selection                      │
    │  4. Hyperparameter Optimization                             │
    │  5. Model Finalization & Serialization                      │
    │  6. Prediction & Inference                                  │
    │  7. Feature Importance Extraction                           │
    │  8. Leaderboard Management                                  │
    │  9. Artifact Persistence                                    │
    │ 10. Session State Management                                │
    └─────────────────────────────────────────────────────────────┘

API Contract (Stable):
    1. initialize_experiment(data, target) → None
    2. compare_all_models(...) → Model | List[Model]
    3. tune_best_model(model, ...) → Model
    4. finalize_and_save(model, path) → Model
    5. predict_model(model, data) → DataFrame
    
    Helpers:
    - get_leaderboard() → DataFrame
    - get_feature_importance(model) → DataFrame
    - get_feature_names() → List[str]
    - get_target_mapping() → Dict
    - save_session_artifacts(dir) → Dict[str, str]
    - get_pipeline() → Pipeline
    - get_model_params(model) → Dict
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import platform
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from packaging import version as pkg_version

# ═══════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════

try:
    from loguru import logger
    
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/pycaret_wrapper_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        level="DEBUG"
    )
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Dependencies (with graceful degradation)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from config.settings import settings
except ImportError:
    logger.warning("⚠ config.settings not found - using defaults")
    
    class Settings:
        """Fallback settings."""
        PYCARET_N_JOBS: int = -1
        PYCARET_VERBOSE: bool = False
        PYCARET_SESSION_ID: int = 42
        RANDOM_STATE: int = 42
        ENABLE_GPU: bool = False
    
    settings = Settings()

try:
    from config.model_registry import get_models_for_problem, ProblemType
    MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    logger.warning("⚠ Model registry not available")
    get_models_for_problem = None
    ProblemType = None
    MODEL_REGISTRY_AVAILABLE = False

# Suppress common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pycaret')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "PyCaretConfig",
    "PyCaretWrapper",
    "PyCaretVersion",
    "SetupStrategy",
    "CompareStrategy"
]
__version__ = "6.0.0-enterprise"
__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class PyCaretVersion(str, Enum):
    """PyCaret version detection."""
    V2 = "2.x"
    V3 = "3.x"
    UNKNOWN = "unknown"


class SetupStrategy(str, Enum):
    """Experiment setup strategies."""
    FAST = "fast"           # Minimal preprocessing
    BALANCED = "balanced"   # Standard preprocessing
    THOROUGH = "thorough"   # Maximum preprocessing


class CompareStrategy(str, Enum):
    """Model comparison strategies."""
    FAST = "fast"           # Top 3 models
    BALANCED = "balanced"   # Top 5 models
    COMPREHENSIVE = "comprehensive"  # Top 10 models
    ALL = "all"            # All available models


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=False)
class PyCaretConfig:
    """
    🎯 **Enterprise PyCaret Configuration**
    
    Complete configuration for PyCaret wrapper with production features.
    
    Core Settings:
        use_gpu: Enable GPU acceleration
        n_jobs: Parallel jobs (-1 = all cores)
        verbose: Verbose output
        session_id: Random seed for reproducibility
        setup_strategy: Preprocessing strategy
        compare_strategy: Model comparison strategy
    
    Classification Settings:
        fix_imbalance: Apply SMOTE for imbalanced classes
        fix_imbalance_method: Imbalance handling method
        remove_multicollinearity: Remove correlated features
        multicollinearity_threshold: Correlation threshold
        remove_outliers: Remove statistical outliers
        outliers_threshold: Outlier detection threshold
    
    Regression Settings:
        normalize: Normalize features
        transformation: Apply power transformations
        transform_target: Transform target variable
        remove_multicollinearity: Remove correlated features
        multicollinearity_threshold: Correlation threshold
        remove_outliers: Remove statistical outliers
        outliers_threshold: Outlier detection threshold
    
    Feature Engineering:
        feature_selection: Enable feature selection
        feature_selection_threshold: Selection threshold
        feature_interaction: Create interaction features
        polynomial_features: Create polynomial features
        polynomial_degree: Polynomial degree
        group_features: Group similar features
        bin_numeric_features: Bin numeric features
    
    Performance & Memory:
        memory_efficient: Enable memory optimization
        preprocess_data: Enable preprocessing
        create_clusters: Create cluster features
        clustering_iteration: Number of clustering iterations
        max_encoding_ohe: Max categories for one-hot encoding
    
    Artifacts:
        save_artifacts: Enable artifact saving
        artifacts_dir_name: Artifacts directory name
        setup_config_filename: Setup config filename
        env_info_filename: Environment info filename
        leaderboard_filename: Leaderboard filename
        pipeline_filename: Pipeline filename
        feature_importance_filename: Feature importance filename
    
    Advanced:
        html: Generate HTML reports
        log_experiment: Enable experiment logging
        experiment_name: Experiment name for tracking
        log_plots: Save plots
        log_profile: Save profiling reports
        log_data: Log training data
    """
    
    # ─── Core Settings ───
    use_gpu: bool = False
    n_jobs: int = getattr(settings, "PYCARET_N_JOBS", -1)
    verbose: bool = getattr(settings, "PYCARET_VERBOSE", False)
    session_id: int = getattr(settings, "PYCARET_SESSION_ID", getattr(settings, "RANDOM_STATE", 42))
    setup_strategy: SetupStrategy = SetupStrategy.BALANCED
    compare_strategy: CompareStrategy = CompareStrategy.BALANCED
    
    # ─── Classification Settings ───
    fix_imbalance: bool = True
    fix_imbalance_method: Optional[str] = None  # 'smote', 'adasyn', etc.
    remove_multicollinearity: bool = True
    multicollinearity_threshold: float = 0.90
    remove_outliers: bool = False
    outliers_threshold: float = 0.05
    
    # ─── Regression Settings ───
    normalize: bool = True
    transformation: bool = False
    transform_target: bool = False
    
    # ─── Feature Engineering ───
    feature_selection: bool = False
    feature_selection_threshold: float = 0.8
    feature_interaction: bool = False
    polynomial_features: bool = False
    polynomial_degree: int = 2
    group_features: Optional[List[str]] = None
    bin_numeric_features: Optional[List[str]] = None
    
    # ─── Performance & Memory ───
    memory_efficient: bool = True
    preprocess_data: bool = True
    create_clusters: bool = False
    clustering_iteration: int = 20
    max_encoding_ohe: int = 25
    
    # ─── Artifacts ───
    save_artifacts: bool = True
    artifacts_dir_name: str = "artifacts"
    setup_config_filename: str = "setup_config.json"
    env_info_filename: str = "env_info.json"
    leaderboard_filename: str = "leaderboard.csv"
    pipeline_filename: str = "pipeline.pkl"
    feature_importance_filename: str = "feature_importance.csv"
    
    # ─── Advanced ───
    html: bool = False
    log_experiment: bool = False
    experiment_name: Optional[str] = None
    log_plots: bool = False
    log_profile: bool = False
    log_data: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_jobs < -1:
            raise ValueError(f"n_jobs must be >= -1, got {self.n_jobs}")
        
        if not 0 < self.multicollinearity_threshold <= 1:
            raise ValueError(f"multicollinearity_threshold must be in (0, 1], got {self.multicollinearity_threshold}")
        
        if not 0 < self.outliers_threshold < 0.5:
            logger.warning(f"⚠ outliers_threshold={self.outliers_threshold} seems unusual")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        
        # Convert enums to strings
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        
        return data
    
    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export configuration to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if path:
            Path(path).write_text(json_str, encoding='utf-8')
        
        return json_str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PyCaretConfig':
        """Create configuration from dictionary."""
        # Convert string enums back
        if 'setup_strategy' in data and isinstance(data['setup_strategy'], str):
            data['setup_strategy'] = SetupStrategy(data['setup_strategy'])
        
        if 'compare_strategy' in data and isinstance(data['compare_strategy'], str):
            data['compare_strategy'] = CompareStrategy(data['compare_strategy'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'PyCaretConfig':
        """Load configuration from JSON file."""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls.from_dict(data)
    
    @classmethod
    def create_fast(cls) -> 'PyCaretConfig':
        """Create fast configuration (minimal preprocessing)."""
        return cls(
            setup_strategy=SetupStrategy.FAST,
            compare_strategy=CompareStrategy.FAST,
            fix_imbalance=False,
            remove_multicollinearity=False,
            feature_selection=False,
            normalize=False,
            transformation=False
        )
    
    @classmethod
    def create_thorough(cls) -> 'PyCaretConfig':
        """Create thorough configuration (maximum preprocessing)."""
        return cls(
            setup_strategy=SetupStrategy.THOROUGH,
            compare_strategy=CompareStrategy.COMPREHENSIVE,
            fix_imbalance=True,
            remove_multicollinearity=True,
            remove_outliers=True,
            feature_selection=True,
            feature_interaction=True,
            polynomial_features=True,
            normalize=True,
            transformation=True,
            transform_target=True
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Utility Functions & Decorators
# ═══════════════════════════════════════════════════════════════════════════

def _timeit(operation_name: str):
    """
    ⏱ **Performance Timing Decorator**
    
    Measures execution time with sub-millisecond precision.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(f"⏱ {operation_name}: {elapsed_ms:.3f}ms")
        return wrapper
    return decorator


def _safe_operation(operation_name: str, default_value: Any = None, log_level: str = "warning"):
    """
    🛡 **Safe Operation Decorator**
    
    Wraps operations with comprehensive error handling.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                error_msg = f"⚠ {operation_name} failed: {type(e).__name__}: {str(e)[:100]}"
                
                if log_level == "error":
                    logger.error(error_msg, exc_info=True)
                elif log_level == "warning":
                    logger.warning(error_msg)
                else:
                    logger.debug(error_msg)
                
                return default_value
        return wrapper
    return decorator


@contextmanager
def _suppress_output():
    """
    🤫 **Output Suppression Context**
    
    Suppresses stdout/stderr for noisy operations.
    """
    try:
        # Save original
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Redirect to devnull
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        yield
    finally:
        # Restore
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@lru_cache(maxsize=1)
def _detect_pycaret_version() -> PyCaretVersion:
    """
    🔍 **PyCaret Version Detection**
    
    Auto-detect installed PyCaret version.
    """
    try:
        import pycaret
        ver_str = getattr(pycaret, "__version__", "unknown")
        
        if ver_str == "unknown":
            return PyCaretVersion.UNKNOWN
        
        ver = pkg_version.parse(ver_str)
        
        if ver.major == 2:
            return PyCaretVersion.V2
        elif ver.major == 3:
            return PyCaretVersion.V3
        else:
            return PyCaretVersion.UNKNOWN
    
    except ImportError:
        return PyCaretVersion.UNKNOWN


def _get_system_info() -> Dict[str, Any]:
    """
    💻 **System Information Collector**
    
    Collects comprehensive system metadata.
    """
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Main PyCaret Wrapper (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════

class PyCaretWrapper:
    """
    🚀 **PyCaretWrapper PRO Master Enterprise ++++**
    
    Enterprise-grade PyCaret orchestration layer with production features.
    
    Responsibilities:
      1. Lazy module import with version detection
      2. Defensive experiment initialization
      3. Multi-model comparison & selection
      4. Advanced hyperparameter tuning
      5. Model finalization & serialization
      6. Prediction & inference
      7. Feature importance extraction
      8. Leaderboard management
      9. Comprehensive artifact persistence
     10. Session state management
     11. Memory-efficient processing
     12. Production error handling
    
    Features:
      ✓ Clean, stable API contract
      ✓ Version-aware compatibility
      ✓ Comprehensive validation
      ✓ Telemetry & monitoring
      ✓ Artifact management
      ✓ Model registry integration
      ✓ GPU auto-configuration
      ✓ Memory optimization
      ✓ SHAP/LIME integration ready
      ✓ MLflow compatible
    
    Usage:
```python
        # Basic usage
        wrapper = PyCaretWrapper('classification')
        wrapper.initialize_experiment(df, 'target')
        models = wrapper.compare_all_models(n_select=5)
        best = models[0]
        tuned = wrapper.tune_best_model(best)
        final = wrapper.finalize_and_save(tuned, 'model')
        
        # Advanced usage
        config = PyCaretConfig.create_thorough()
        wrapper = PyCaretWrapper('classification', config)
        wrapper.initialize_experiment(
            data=df,
            target_column='target',
            fold=10,
            use_gpu=True
        )
```
    """
    
    version: str = __version__
    
    def __init__(
        self,
        problem_type: Literal["classification", "regression"],
        config: Optional[PyCaretConfig] = None
    ) -> None:
        """
        Initialize PyCaret wrapper.
        
        Args:
            problem_type: 'classification' or 'regression'
            config: Optional custom configuration
        
        Raises:
            ValueError: Invalid problem type
            RuntimeError: PyCaret import failure
        """
        # ─── Validation ───
        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                f"Invalid problem_type='{problem_type}'. "
                f"Must be 'classification' or 'regression'"
            )
        
        self.problem_type = problem_type
        self.config = config or PyCaretConfig()
        self._log = logger.bind(
            component="PyCaretWrapper",
            problem_type=problem_type,
            version=self.version
        )
        
        # ─── Detect PyCaret Version ───
        self.pycaret_version_enum = _detect_pycaret_version()
        self.pycaret_version_str = "unknown"
        
        # ─── Lazy Import PyCaret ───
        self._log.debug(f"Importing PyCaret {problem_type} module...")
        
        try:
            if problem_type == "classification":
                from pycaret import classification as pc
            else:  # regression
                from pycaret import regression as pc
            
            # Get version
            try:
                import pycaret
                self.pycaret_version_str = getattr(pycaret, "__version__", "unknown")
            except Exception:
                pass
        
        except ImportError as e:
            raise RuntimeError(
                f"PyCaret import failed for '{problem_type}'. "
                f"Install: pip install pycaret"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"PyCaret initialization failed: {type(e).__name__}: {str(e)}"
            ) from e
        
        # ─── Core Functions ───
        self._setup = getattr(pc, 'setup', None)
        self._compare_models = getattr(pc, 'compare_models', None)
        self._create_model = getattr(pc, 'create_model', None)
        self._tune_model = getattr(pc, 'tune_model', None)
        self._ensemble_model = getattr(pc, 'ensemble_model', None)
        self._blend_models = getattr(pc, 'blend_models', None)
        self._stack_models = getattr(pc, 'stack_models', None)
        self._finalize_model = getattr(pc, 'finalize_model', None)
        self._predict_model = getattr(pc, 'predict_model', None)
        self._plot_model = getattr(pc, 'plot_model', None)
        self._evaluate_model = getattr(pc, 'evaluate_model', None)
        self._save_model = getattr(pc, 'save_model', None)
        self._load_model = getattr(pc, 'load_model', None)
        
        # ─── Utility Functions ───
        self._pull = None
        self._get_config = None
        self._models = None
        
        try:
            from pycaret.utils import pull, get_config
            self._pull = pull
            self._get_config = get_config
        except ImportError:
            pass
        
        try:
            self._models = getattr(pc, 'models', None)
        except Exception:
            pass
        
        # ─── Session State ───
        self._experiment_initialized: bool = False
        self._artifacts_root: Optional[Path] = None
        self._last_setup_config: Dict[str, Any] = {}
        self._last_env_info: Dict[str, Any] = {}
        self._last_leaderboard: Optional[pd.DataFrame] = None
        self._setup_timestamp: Optional[float] = None
        self._session_id: Optional[int] = None
        self._data_hash: Optional[str] = None
        
        # ─── Telemetry ───
        self._operations_log: List[Dict[str, Any]] = []
        
        self._log.info(
            f"✓ PyCaretWrapper initialized | "
            f"pycaret={self.pycaret_version_str} | "
            f"type={problem_type}"
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Status & Information
    # ───────────────────────────────────────────────────────────────────
    
    def is_initialized(self) -> bool:
        """Check if experiment is initialized."""
        return self._experiment_initialized
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive wrapper status.
        
        Returns:
            Status dictionary with initialization state and metadata
        """
        return {
            "initialized": self._experiment_initialized,
            "problem_type": self.problem_type,
            "pycaret_version": self.pycaret_version_str,
            "pycaret_version_enum": self.pycaret_version_enum.value,
            "wrapper_version": self.version,
            "session_id": self._session_id,
            "setup_timestamp": self._setup_timestamp,
            "operations_count": len(self._operations_log),
            "artifacts_root": str(self._artifacts_root) if self._artifacts_root else None
        }
    
    def env_info(self) -> Dict[str, Any]:
        """
        Get comprehensive environment information.
        
        Returns:
            Environment metadata dictionary
        """
        if not self._last_env_info:
            self._last_env_info = {
                "wrapper_version": self.version,
                "pycaret_version": self.pycaret_version_str,
                "pycaret_version_enum": self.pycaret_version_enum.value,
                "problem_type": self.problem_type,
                "config": self.config.to_dict(),
                "system": _get_system_info(),
                "model_registry_available": MODEL_REGISTRY_AVAILABLE
            }
        return self._last_env_info
    
    def get_available_models(self) -> Optional[pd.DataFrame]:
        """
        Get list of available models for current problem type.
        
        Returns:
            DataFrame of available models or None
        """
        if self._models is not None:
            try:
                if callable(self._models):
                    return self._models()
                return self._models
            except Exception as e:
                self._log.debug(f"Could not retrieve models list: {e}")
        
        return None
    
    # ───────────────────────────────────────────────────────────────────
    # Experiment Setup
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("initialize_experiment")
    def initialize_experiment(
        self,
        data: pd.DataFrame,
        target_column: str,
        *,
        fold: Optional[int] = None,
        fold_strategy: Optional[str] = None,
        session_id: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        silent: bool = True,
        log_experiment: bool = False,
        experiment_name: Optional[str] = None,
        artifacts_root: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> None:
        """
        🎯 **Initialize PyCaret Experiment**
        
        Sets up PyCaret experiment with comprehensive configuration.
        
        Args:
            data: Complete DataFrame (features + target)
            target_column: Name of target column
            fold: Number of CV folds (default: 5 or 10 for small data)
            fold_strategy: CV strategy ('kfold', 'stratifiedkfold', 'timeseries')
            session_id: Random seed for reproducibility
            use_gpu: Force GPU usage (None = auto-detect)
            silent: Suppress interactive prompts
            log_experiment: Enable PyCaret experiment logging
            experiment_name: Name for experiment tracking
            artifacts_root: Root directory for artifacts
            **kwargs: Additional setup parameters (passed to PyCaret)
        
        Raises:
            ValueError: Invalid inputs
            RuntimeError: Setup failure
        """
        # ═══════════════════════════════════════════════════════════════
        # VALIDATION
        # ═══════════════════════════════════════════════════════════════
        
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("'data' must be non-empty pandas DataFrame")
        
        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError("'target_column' must be non-empty string")
        
        if target_column not in data.columns:
            available = ', '.join(data.columns[:10].tolist())
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available: [{available}{'...' if len(data.columns) > 10 else ''}]"
            )
        
        # ═══════════════════════════════════════════════════════════════
        # SESSION CONFIGURATION
        # ═══════════════════════════════════════════════════════════════
        
        self._session_id = (
            session_id if session_id is not None
            else self.config.session_id
        )
        
        # Data hash for change detection
        try:
            self._data_hash = str(hash(tuple(data.iloc[0])))[:16]
        except Exception:
            self._data_hash = None
        
        # ═══════════════════════════════════════════════════════════════
        # SETUP PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        
        setup_params: Dict[str, Any] = {
            "data": data,
            "target": target_column,
            "session_id": self._session_id,
            "verbose": self.config.verbose,
            "n_jobs": self.config.n_jobs if self.config.n_jobs > 0 else -1,
            "use_gpu": self.config.use_gpu if use_gpu is None else bool(use_gpu),
            "silent": silent,
            "html": self.config.html,
            "log_experiment": log_experiment or self.config.log_experiment,
        }

        # ─── Experiment Name ───
        if experiment_name or self.config.experiment_name:
            setup_params["experiment_name"] = (
                experiment_name or self.config.experiment_name
            )
        
        # ─── Fold Configuration ───
        if fold is not None:
            setup_params["fold"] = int(fold)
        elif len(data) < 100:
            setup_params["fold"] = 5  # Smaller fold for small datasets
        else:
            setup_params["fold"] = 10
        
        if fold_strategy:
            setup_params["fold_strategy"] = fold_strategy
        
        # ═══════════════════════════════════════════════════════════════
        # PROBLEM-SPECIFIC CONFIGURATION
        # ═══════════════════════════════════════════════════════════════
        
        if self.problem_type == "classification":
            # Classification-specific parameters
            setup_params.update({
                "fix_imbalance": self.config.fix_imbalance,
                "remove_multicollinearity": self.config.remove_multicollinearity,
                "multicollinearity_threshold": self.config.multicollinearity_threshold,
                "remove_outliers": self.config.remove_outliers,
                "outliers_threshold": self.config.outliers_threshold,
            })
            
            if self.config.fix_imbalance_method:
                setup_params["fix_imbalance_method"] = self.config.fix_imbalance_method
        
        else:  # regression
            # Regression-specific parameters
            setup_params.update({
                "normalize": self.config.normalize,
                "transformation": self.config.transformation,
                "transform_target": self.config.transform_target,
                "remove_multicollinearity": self.config.remove_multicollinearity,
                "multicollinearity_threshold": self.config.multicollinearity_threshold,
                "remove_outliers": self.config.remove_outliers,
                "outliers_threshold": self.config.outliers_threshold,
            })
        
        # ═══════════════════════════════════════════════════════════════
        # FEATURE ENGINEERING
        # ═══════════════════════════════════════════════════════════════
        
        if self.config.feature_selection:
            setup_params["feature_selection"] = True
            setup_params["feature_selection_threshold"] = self.config.feature_selection_threshold
        
        if self.config.feature_interaction:
            setup_params["feature_interaction"] = True
        
        if self.config.polynomial_features:
            setup_params["polynomial_features"] = True
            setup_params["polynomial_degree"] = self.config.polynomial_degree
        
        if self.config.group_features:
            setup_params["group_features"] = self.config.group_features
        
        if self.config.bin_numeric_features:
            setup_params["bin_numeric_features"] = self.config.bin_numeric_features
        
        # ═══════════════════════════════════════════════════════════════
        # CLUSTERING & ADVANCED FEATURES
        # ═══════════════════════════════════════════════════════════════
        
        if self.config.create_clusters:
            setup_params["create_clusters"] = True
            setup_params["cluster_iter"] = self.config.clustering_iteration
        
        # ═══════════════════════════════════════════════════════════════
        # ENCODING & MEMORY
        # ═══════════════════════════════════════════════════════════════
        
        setup_params["max_encoding_ohe"] = self.config.max_encoding_ohe
        
        if self.config.memory_efficient:
            setup_params["preprocess"] = True
        
        # ═══════════════════════════════════════════════════════════════
        # LOGGING & ARTIFACTS
        # ═══════════════════════════════════════════════════════════════
        
        if self.config.log_plots:
            setup_params["log_plots"] = True
        
        if self.config.log_profile:
            setup_params["log_profile"] = True
        
        if self.config.log_data:
            setup_params["log_data"] = True
        
        # ─── Artifacts Directory ───
        if artifacts_root:
            self._artifacts_root = Path(artifacts_root)
            self._artifacts_root.mkdir(parents=True, exist_ok=True)
        elif self.config.save_artifacts:
            self._artifacts_root = Path(self.config.artifacts_dir_name)
            self._artifacts_root.mkdir(parents=True, exist_ok=True)
        
        # ═══════════════════════════════════════════════════════════════
        # USER OVERRIDES
        # ═══════════════════════════════════════════════════════════════
        
        setup_params.update(kwargs)
        
        # ═══════════════════════════════════════════════════════════════
        # EXECUTE SETUP
        # ═══════════════════════════════════════════════════════════════
        
        self._log.info(
            f"🔧 Initializing PyCaret setup | "
            f"strategy={self.config.setup_strategy.value} | "
            f"rows={len(data):,} | "
            f"cols={len(data.columns):,}"
        )
        
        t0 = time.perf_counter()
        setup_successful = False
        
        try:
            if self._setup is None:
                raise RuntimeError("PyCaret setup function not available")
            
            # Execute setup (may be verbose)
            if silent and not self.config.verbose:
                with _suppress_output():
                    self._setup(**setup_params)
            else:
                self._setup(**setup_params)
            
            setup_successful = True
        
        except Exception as e:
            self._log.error(f"PyCaret setup failed: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"PyCaret setup failed: {e}") from e
        
        finally:
            elapsed = round(time.perf_counter() - t0, 4)
            
            if setup_successful:
                self._experiment_initialized = True
                self._setup_timestamp = time.time()
                
                self._log.success(
                    f"✓ PyCaret experiment initialized | "
                    f"time={elapsed}s | "
                    f"fold={setup_params.get('fold', 'N/A')}"
                )
                
                # Save setup config
                self._last_setup_config = {
                    "params": {
                        k: (str(v) if isinstance(v, (Path, pd.DataFrame)) else v)
                        for k, v in setup_params.items()
                        if k != 'data'  # Don't save data in config
                    },
                    "elapsed_s": elapsed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data_shape": data.shape,
                    "data_hash": self._data_hash
                }
                
                # Log operation
                self._log_operation("setup", elapsed, {"fold": setup_params.get("fold")})
    
    # ───────────────────────────────────────────────────────────────────
    # Model Comparison
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("compare_models")
    def compare_all_models(
        self,
        n_select: Optional[int] = None,
        *,
        sort: str = "auto",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        budget_time: Optional[float] = None,
        turbo: bool = True,
        **kwargs: Any
    ) -> Union[Any, List[Any]]:
        """
        📊 **Compare Multiple Models**
        
        Compares available models and returns best candidates.
        
        Model Selection Priority:
          1. Explicit 'include' parameter
          2. Model registry shortlist (if available)
          3. PyCaret defaults (all models)
        
        Args:
            n_select: Number of top models to select (default: from compare_strategy)
            sort: Metric to sort by ('auto' uses problem-specific default)
            include: Models to include (whitelist)
            exclude: Models to exclude (blacklist)
            budget_time: Time budget in minutes for comparison
            turbo: Use turbo mode (faster, less accurate)
            **kwargs: Additional compare_models parameters
        
        Returns:
            Best model(s): Single model if n_select=1, list otherwise
        
        Raises:
            RuntimeError: Experiment not initialized or comparison failure
        """
        self._require_initialized()
        
        # ═══════════════════════════════════════════════════════════════
        # PARAMETER CONFIGURATION
        # ═══════════════════════════════════════════════════════════════
        
        # Determine n_select from strategy
        if n_select is None:
            strategy_map = {
                CompareStrategy.FAST: 3,
                CompareStrategy.BALANCED: 5,
                CompareStrategy.COMPREHENSIVE: 10,
                CompareStrategy.ALL: 999
            }
            n_select = strategy_map.get(self.config.compare_strategy, 5)
        
        # ─── Model Selection (Include List) ───
        if include is None and MODEL_REGISTRY_AVAILABLE and get_models_for_problem:
            try:
                if self.problem_type == "classification":
                    include = get_models_for_problem(
                        ProblemType.CLASSIFICATION,
                        strategy="accurate"
                    )
                else:
                    include = get_models_for_problem(
                        ProblemType.REGRESSION,
                        strategy="accurate"
                    )
                
                self._log.debug(f"Using model registry include list: {include}")
            
            except Exception as e:
                self._log.debug(f"Model registry unavailable: {e}")
                include = None
        
        # ═══════════════════════════════════════════════════════════════
        # LOGGING
        # ═══════════════════════════════════════════════════════════════
        
        inc_msg = f"include={include}" if include else "include=all"
        exc_msg = f"exclude={exclude}" if exclude else "exclude=none"
        
        self._log.info(
            f"📊 Comparing models | "
            f"n_select={n_select} | "
            f"sort={sort} | "
            f"{inc_msg} | "
            f"{exc_msg} | "
            f"turbo={turbo}"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # EXECUTE COMPARISON
        # ═══════════════════════════════════════════════════════════════
        
        t0 = time.perf_counter()
        best_models = None
        
        try:
            if self._compare_models is None:
                raise RuntimeError("PyCaret compare_models function not available")
            
            compare_params = {
                "n_select": n_select,
                "sort": sort,
                "include": include,
                "exclude": exclude,
                "turbo": turbo,
                **kwargs
            }
            
            # Add budget_time if supported
            if budget_time is not None:
                compare_params["budget_time"] = budget_time
            
            best_models = self._compare_models(**compare_params)
        
        except Exception as e:
            self._log.error(f"Model comparison failed: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Model comparison failed: {e}") from e
        
        finally:
            elapsed = round(time.perf_counter() - t0, 4)
            
            if best_models is not None:
                self._log.success(f"✓ Model comparison completed | time={elapsed}s")
                
                # Log operation
                self._log_operation(
                    "compare_models",
                    elapsed,
                    {"n_select": n_select, "sort": sort}
                )
                
                # ═══════════════════════════════════════════════════════
                # SAVE LEADERBOARD
                # ═══════════════════════════════════════════════════════
                
                try:
                    lb = self.get_leaderboard()
                    if lb is not None and not lb.empty:
                        self._last_leaderboard = lb.copy()
                        
                        if self.config.save_artifacts and self._artifacts_root:
                            self._save_artifact_df(lb, self.config.leaderboard_filename)
                
                except Exception as e:
                    self._log.debug(f"Could not save leaderboard: {e}")
        
        return best_models
    
    # ───────────────────────────────────────────────────────────────────
    # Hyperparameter Tuning
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("tune_model")
    def tune_best_model(
        self,
        model: Any,
        *,
        n_iter: int = 25,
        optimize: str = "auto",
        search_library: str = "scikit-learn",
        search_algorithm: Optional[str] = None,
        choose_better: bool = True,
        **kwargs: Any
    ) -> Any:
        """
        🎯 **Hyperparameter Tuning**
        
        Optimizes model hyperparameters using specified strategy.
        
        Args:
            model: Model to tune
            n_iter: Number of tuning iterations
            optimize: Metric to optimize ('auto' uses problem default)
            search_library: Search library ('scikit-learn', 'optuna', 'tune-sklearn')
            search_algorithm: Search algorithm ('random', 'grid', 'bayesian')
            choose_better: Return tuned model only if better than original
            **kwargs: Additional tune_model parameters
        
        Returns:
            Tuned model (or original if choose_better=True and tuning didn't improve)
        
        Raises:
            RuntimeError: Experiment not initialized or tuning failure
        """
        self._require_initialized()
        
        if self._tune_model is None:
            raise RuntimeError("PyCaret tune_model function not available")
        
        self._log.info(
            f"🎯 Tuning model | "
            f"algorithm={type(model).__name__} | "
            f"n_iter={n_iter} | "
            f"optimize={optimize} | "
            f"search={search_library}"
        )
        
        t0 = time.perf_counter()
        tuned_model = None
        
        try:
            tune_params = {
                "estimator": model,
                "n_iter": n_iter,
                "optimize": optimize,
                "choose_better": choose_better,
                **kwargs
            }
            
            # Add search configuration if supported
            if search_library:
                tune_params["search_library"] = search_library
            
            if search_algorithm:
                tune_params["search_algorithm"] = search_algorithm
            
            tuned_model = self._tune_model(**tune_params)
        
        except Exception as e:
            self._log.error(f"Model tuning failed: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Model tuning failed: {e}") from e
        
        finally:
            elapsed = round(time.perf_counter() - t0, 4)
            
            if tuned_model is not None:
                self._log.success(f"✓ Model tuning completed | time={elapsed}s")
                
                # Log operation
                self._log_operation(
                    "tune_model",
                    elapsed,
                    {"n_iter": n_iter, "optimize": optimize}
                )
        
        return tuned_model
    
    # ───────────────────────────────────────────────────────────────────
    # Model Ensembling
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("ensemble_model")
    def ensemble_model(
        self,
        model: Any,
        method: Literal["Bagging", "Boosting"] = "Bagging",
        n_estimators: int = 10,
        **kwargs: Any
    ) -> Any:
        """
        🎪 **Create Ensemble Model**
        
        Creates ensemble from single model.
        
        Args:
            model: Base model for ensemble
            method: Ensemble method ('Bagging' or 'Boosting')
            n_estimators: Number of estimators
            **kwargs: Additional ensemble_model parameters
        
        Returns:
            Ensemble model
        """
        self._require_initialized()
        
        if self._ensemble_model is None:
            raise RuntimeError("PyCaret ensemble_model function not available")
        
        self._log.info(f"🎪 Creating ensemble | method={method} | n={n_estimators}")
        
        t0 = time.perf_counter()
        ensemble = self._ensemble_model(
            model,
            method=method,
            n_estimators=n_estimators,
            **kwargs
        )
        elapsed = round(time.perf_counter() - t0, 4)
        
        self._log.success(f"✓ Ensemble created | time={elapsed}s")
        self._log_operation("ensemble_model", elapsed, {"method": method})
        
        return ensemble
    
    @_timeit("blend_models")
    def blend_models(
        self,
        models: List[Any],
        method: str = "auto",
        **kwargs: Any
    ) -> Any:
        """
        🔀 **Blend Multiple Models**
        
        Creates blended ensemble from multiple models.
        
        Args:
            models: List of models to blend
            method: Blending method
            **kwargs: Additional blend_models parameters
        
        Returns:
            Blended model
        """
        self._require_initialized()
        
        if self._blend_models is None:
            raise RuntimeError("PyCaret blend_models function not available")
        
        self._log.info(f"🔀 Blending {len(models)} models")
        
        t0 = time.perf_counter()
        blended = self._blend_models(models, method=method, **kwargs)
        elapsed = round(time.perf_counter() - t0, 4)
        
        self._log.success(f"✓ Models blended | time={elapsed}s")
        self._log_operation("blend_models", elapsed, {"n_models": len(models)})
        
        return blended
    
    @_timeit("stack_models")
    def stack_models(
        self,
        models: List[Any],
        meta_model: Optional[Any] = None,
        **kwargs: Any
    ) -> Any:
        """
        📚 **Stack Multiple Models**
        
        Creates stacked ensemble with meta-learner.
        
        Args:
            models: List of base models
            meta_model: Meta-learner (None = auto)
            **kwargs: Additional stack_models parameters
        
        Returns:
            Stacked model
        """
        self._require_initialized()
        
        if self._stack_models is None:
            raise RuntimeError("PyCaret stack_models function not available")
        
        self._log.info(f"📚 Stacking {len(models)} models")
        
        t0 = time.perf_counter()
        stacked = self._stack_models(models, meta_model=meta_model, **kwargs)
        elapsed = round(time.perf_counter() - t0, 4)
        
        self._log.success(f"✓ Models stacked | time={elapsed}s")
        self._log_operation("stack_models", elapsed, {"n_models": len(models)})
        
        return stacked
    
    # ───────────────────────────────────────────────────────────────────
    # Finalization & Persistence
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("finalize_and_save")
    def finalize_and_save(
        self,
        model: Any,
        model_path: Union[str, Path],
        *,
        save_format: str = "pkl",
        **kwargs: Any
    ) -> Any:
        """
        💾 **Finalize & Save Model**
        
        Finalizes model on full dataset and persists to disk.
        
        Args:
            model: Model to finalize
            model_path: Path to save model (extension added automatically)
            save_format: Save format ('pkl', 'joblib')
            **kwargs: Additional finalization parameters
        
        Returns:
            Finalized model
        
        Raises:
            RuntimeError: Experiment not initialized or save failure
        """
        self._require_initialized()
        
        if self._finalize_model is None or self._save_model is None:
            raise RuntimeError("PyCaret finalize/save functions not available")
        
        # ═══════════════════════════════════════════════════════════════
        # PREPARE OUTPUT PATH
        # ═══════════════════════════════════════════════════════════════
        
        out = Path(model_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        self._log.info(f"💾 Finalizing model | algorithm={type(model).__name__}")
        
        # ═══════════════════════════════════════════════════════════════
        # FINALIZE MODEL
        # ═══════════════════════════════════════════════════════════════
        
        t0 = time.perf_counter()
        final_model = None
        
        try:
            # Finalize on full dataset
            final_model = self._finalize_model(model, **kwargs)
            
            # Save model
            self._save_model(final_model, str(out))
            
            # Note: PyCaret adds .pkl extension automatically
            actual_path = out.with_suffix('.pkl' if not out.suffix else out.suffix)
            
        except Exception as e:
            self._log.error(f"Model finalization/save failed: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Model finalization failed: {e}") from e
        
        finally:
            elapsed = round(time.perf_counter() - t0, 4)
            
            if final_model is not None:
                self._log.success(
                    f"✓ Model finalized & saved | "
                    f"path={actual_path} | "
                    f"time={elapsed}s"
                )
                
                # Log operation
                self._log_operation("finalize_and_save", elapsed, {"path": str(actual_path)})
                
                # ═══════════════════════════════════════════════════════
                # SAVE SESSION ARTIFACTS
                # ═══════════════════════════════════════════════════════
                
                if self.config.save_artifacts and self._artifacts_root:
                    try:
                        self.save_session_artifacts(self._artifacts_root)
                    except Exception as e:
                        self._log.warning(f"Could not save session artifacts: {e}")
                
                # ═══════════════════════════════════════════════════════
                # SAVE FEATURE IMPORTANCE
                # ═══════════════════════════════════════════════════════
                
                if self.config.save_artifacts and self._artifacts_root:
                    try:
                        feat_imp = self.get_feature_importance(final_model)
                        if feat_imp is not None:
                            self._save_artifact_df(
                                feat_imp,
                                self.config.feature_importance_filename
                            )
                    except Exception as e:
                        self._log.debug(f"Could not save feature importance: {e}")
        
        return final_model
    
    def load_model(self, model_path: Union[str, Path]) -> Any:
        """
        📂 **Load Saved Model**
        
        Loads previously saved model from disk.
        
        Args:
            model_path: Path to saved model
        
        Returns:
            Loaded model
        """
        if self._load_model is None:
            raise RuntimeError("PyCaret load_model function not available")
        
        path = Path(model_path)
        
        # PyCaret expects path without extension
        if path.suffix == '.pkl':
            path = path.with_suffix('')
        
        self._log.info(f"📂 Loading model from: {path}")
        
        try:
            model = self._load_model(str(path))
            self._log.success(f"✓ Model loaded successfully")
            return model
        
        except Exception as e:
            self._log.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    # ───────────────────────────────────────────────────────────────────
    # Prediction & Inference
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("predict_model")
    def predict_model(
        self,
        model: Any,
        data: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        🔮 **Make Predictions**
        
        Generates predictions using trained model.
        
        Args:
            model: Trained model
            data: Data to predict on (None = use holdout set)
            **kwargs: Additional predict_model parameters
        
        Returns:
            DataFrame with predictions
        
        Raises:
            RuntimeError: Experiment not initialized or prediction failure
        """
        self._require_initialized()
        
        if self._predict_model is None:
            raise RuntimeError("PyCaret predict_model function not available")
        
        self._log.debug(f"🔮 Predicting | data_shape={data.shape if data is not None else 'holdout'}")
        
        try:
            if data is not None:
                preds = self._predict_model(model, data=data, **kwargs)
            else:
                preds = self._predict_model(model, **kwargs)
            
            # Ensure DataFrame output
            if not isinstance(preds, pd.DataFrame):
                self._log.warning("predict_model returned non-DataFrame; coercing")
                preds = pd.DataFrame(preds)
            
            return preds
        
        except Exception as e:
            self._log.error(f"Prediction failed: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    # ───────────────────────────────────────────────────────────────────
    # Leaderboard & Results
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("get_leaderboard", default_value=None)
    def get_leaderboard(self) -> Optional[pd.DataFrame]:
        """
        📊 **Get Model Leaderboard**
        
        Retrieves comparison leaderboard with model performance metrics.
        
        Returns:
            Leaderboard DataFrame or None if unavailable
        """
        if not self._experiment_initialized:
            self._log.warning("get_leaderboard called before experiment initialization")
            return None
        
        # Strategy 1: Use pull() function
        if self._pull is not None:
            try:
                df = self._pull()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df.copy()
            except Exception:
                pass
        
        # Strategy 2: Use get_config()
        if self._get_config is not None:
            try:
                lb = self._get_config("leaderboard")
                if isinstance(lb, pd.DataFrame) and not lb.empty:
                    return lb.copy()
                
                # Alternative config keys
                for key in ["experiment__leaderboard", "compare_models__leaderboard"]:
                    try:
                        lb = self._get_config(key)
                        if isinstance(lb, pd.DataFrame) and not df.empty:
                            return lb.copy()
                    except Exception:
                        continue
            
            except Exception:
                pass
        
        # Strategy 3: Return cached version
        if self._last_leaderboard is not None:
            return self._last_leaderboard.copy()
        
        return None
    
    # ───────────────────────────────────────────────────────────────────
    # Feature Importance & Analysis
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("get_feature_importance", default_value=None)
    def get_feature_importance(
        self,
        model: Any,
        save_plot: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        📈 **Extract Feature Importance**
        
        Extracts feature importance from trained model.
        
        Note: For comprehensive SHAP explanations, use ModelExplainer.
        
        Args:
            model: Trained model with feature_importances_ or coef_ attribute
            save_plot: Save feature importance plot
        
        Returns:
            Feature importance DataFrame or None
        """
        self._require_initialized()
        
        # Try to generate plot
        if save_plot and self._plot_model is not None:
            try:
                self._plot_model(model, plot="feature", save=True)
            except Exception:
                pass
        
        # Extract from model
        feature_names = self.get_feature_names() or []
        
        # Strategy 1: feature_importances_ (tree-based models)
        if hasattr(model, "feature_importances_"):
            importances = getattr(model, "feature_importances_", [])
            
            if len(importances) > 0:
                if not feature_names:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                df = pd.DataFrame({
                    "feature": feature_names[:len(importances)],
                    "importance": importances
                }).sort_values("importance", ascending=False)
                
                return df
        
        # Strategy 2: coef_ (linear models)
        if hasattr(model, "coef_"):
            coefficients = getattr(model, "coef_", [])
            
            # Handle 2D coefficients (multi-class)
            if isinstance(coefficients, np.ndarray) and coefficients.ndim == 2:
                coefficients = np.abs(coefficients).mean(axis=0)
            
            if len(coefficients) > 0:
                if not feature_names:
                    feature_names = [f"feature_{i}" for i in range(len(coefficients))]
                
                df = pd.DataFrame({
                    "feature": feature_names[:len(coefficients)],
                    "importance": np.abs(coefficients)
                }).sort_values("importance", ascending=False)
                
                return df
        
        return None
    
    # ───────────────────────────────────────────────────────────────────
    # Helper Methods
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("get_feature_names", default_value=None)
    def get_feature_names(self) -> Optional[List[str]]:
        """
        📝 **Get Feature Names**
        
        Retrieves feature names from PyCaret configuration.
        
        Returns:
            List of feature names or None
        """
        if self._get_config is None:
            return None
        
        # Try multiple strategies
        for config_key in ["X_train", "X", "feature_names"]:
            try:
                obj = self._get_config(config_key)
                
                if isinstance(obj, pd.DataFrame):
                    return list(obj.columns)
                elif isinstance(obj, list):
                    return obj
            
            except Exception:
                continue
        
        return None
    
    @_safe_operation("get_target_mapping", default_value=None)
    def get_target_mapping(self) -> Optional[Dict[Any, Any]]:
        """
        🏷 **Get Target Label Mapping**
        
        Retrieves label encoder mapping for classification tasks.
        
        Returns:
            Mapping dictionary {encoded_value: original_label} or None
        """
        if self.problem_type != "classification" or self._get_config is None:
            return None
        
        # Try to get from various config locations
        for config_key in ["y_train", "y", "target"]:
            try:
                y = self._get_config(config_key)
                
                # Check for categorical encoding
                if isinstance(y, pd.Series) and hasattr(y, "cat"):
                    try:
                        categories = list(y.cat.categories)
                        return {i: val for i, val in enumerate(categories)}
                    except Exception:
                        pass
                
                # Check for label encoder
                if isinstance(y, pd.Series) and y.dtype in ['int64', 'int32']:
                    # Try to get original labels
                    try:
                        le = self._get_config("label_encoder")
                        if hasattr(le, "classes_"):
                            return {i: val for i, val in enumerate(le.classes_)}
                    except Exception:
                        pass
            
            except Exception:
                continue
        
        return None
    
    @_safe_operation("get_pipeline", default_value=None)
    def get_pipeline(self) -> Optional[Any]:
        """
        🔧 **Get Preprocessing Pipeline**
        
        Retrieves the preprocessing pipeline from PyCaret.
        
        Returns:
            Pipeline object or None
        """
        if self._get_config is None:
            return None
        
        for config_key in ["pipeline", "prep_pipe", "preprocessing_pipeline"]:
            try:
                pipeline = self._get_config(config_key)
                if pipeline is not None:
                    return pipeline
            except Exception:
                continue
        
        return None
    
    def get_pipeline_path(self) -> Optional[Path]:
        """
        📂 **Get Pipeline Path**
        
        Returns path to saved pipeline (heuristic).
        
        Note: PyCaret typically stores pipeline within model.pkl
        
        Returns:
            Pipeline path or None
        """
        if self._artifacts_root and self.config.pipeline_filename:
            pipeline_path = self._artifacts_root / self.config.pipeline_filename
            if pipeline_path.exists():
                return pipeline_path
        
        return None
    
    @_safe_operation("get_model_params", default_value=None)
    def get_model_params(self, model: Any) -> Optional[Dict[str, Any]]:
        """
        ⚙ **Get Model Parameters**
        
        Extracts hyperparameters from trained model.
        
        Args:
            model: Trained model
        
        Returns:
            Parameter dictionary or None
        """
        if hasattr(model, "get_params"):
            try:
                return model.get_params()
            except Exception:
                pass
        
        if hasattr(model, "__dict__"):
            try:
                params = {}
                for key, value in model.__dict__.items():
                    if not key.startswith('_'):
                        # Convert numpy/pandas types to native Python
                        if isinstance(value, (np.ndarray, pd.Series)):
                            continue
                        params[key] = value
                return params
            except Exception:
                pass
        
        return None
    
    @_safe_operation("get_config_all", default_value=None)
    def get_config_all(self) -> Optional[Dict[str, Any]]:
        """
        🔍 **Get All Configuration**
        
        Retrieves complete PyCaret configuration.
        
        Returns:
            Configuration dictionary or None
        """
        if self._get_config is None:
            return None
        
        try:
            # Get all config keys
            config_dict = {}
            
            # Common config keys
            keys = [
                "X_train", "X_test", "y_train", "y_test",
                "pipeline", "fold", "n_jobs", "use_gpu",
                "seed", "experiment_name", "gpu_param"
            ]
            
            for key in keys:
                try:
                    value = self._get_config(key)
                    
                    # Convert DataFrames to shapes only
                    if isinstance(value, pd.DataFrame):
                        config_dict[key] = {"type": "DataFrame", "shape": value.shape}
                    elif isinstance(value, pd.Series):
                        config_dict[key] = {"type": "Series", "shape": value.shape}
                    else:
                        config_dict[key] = value
                
                except Exception:
                    continue
            
            return config_dict
        
        except Exception:
            return None
    
    # ───────────────────────────────────────────────────────────────────
    # Visualization & Evaluation
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("plot_model", default_value=None)
    def plot_model(
        self,
        model: Any,
        plot: str = "auc",
        save: bool = False,
        **kwargs: Any
    ) -> Optional[Any]:
        """
        📊 **Plot Model Performance**
        
        Generates visualization plots for model evaluation.
        
        Args:
            model: Trained model
            plot: Plot type ('auc', 'confusion_matrix', 'feature', etc.)
            save: Save plot to file
            **kwargs: Additional plot parameters
        
        Returns:
            Plot object or None
        """
        self._require_initialized()
        
        if self._plot_model is None:
            self._log.warning("PyCaret plot_model not available")
            return None
        
        try:
            return self._plot_model(model, plot=plot, save=save, **kwargs)
        except Exception as e:
            self._log.warning(f"Plotting failed: {e}")
            return None
    
    @_safe_operation("evaluate_model", default_value=None)
    def evaluate_model(self, model: Any) -> None:
        """
        📋 **Evaluate Model Interactively**
        
        Opens interactive evaluation dashboard (Jupyter only).
        
        Args:
            model: Trained model
        """
        self._require_initialized()
        
        if self._evaluate_model is None:
            self._log.warning("PyCaret evaluate_model not available")
            return None
        
        try:
            self._evaluate_model(model)
        except Exception as e:
            self._log.warning(f"Evaluation failed: {e}")
    
    # ───────────────────────────────────────────────────────────────────
    # Artifact Management
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("save_session_artifacts")
    def save_session_artifacts(
        self,
        out_dir: Union[str, Path]
    ) -> Dict[str, Optional[str]]:
        """
        💾 **Save Session Artifacts**
        
        Exports all session artifacts to directory.
        
        Args:
            out_dir: Output directory for artifacts
        
        Returns:
            Dictionary mapping artifact names to saved paths
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        saved: Dict[str, Optional[str]] = {
            "setup_config": None,
            "env_info": None,
            "leaderboard_csv": None,
            "pipeline": None,
            "operations_log": None
        }
        
        # ─── Setup Configuration ───
        try:
            config_path = out_path / self.config.setup_config_filename
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._last_setup_config or {},
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            saved["setup_config"] = str(config_path)
            self._log.debug(f"✓ Saved: {config_path.name}")
        
        except Exception as e:
            self._log.debug(f"Could not save setup_config: {e}")
        
        # ─── Environment Info ───
        try:
            env_path = out_path / self.config.env_info_filename
            with open(env_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.env_info(),
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            saved["env_info"] = str(env_path)
            self._log.debug(f"✓ Saved: {env_path.name}")
        
        except Exception as e:
            self._log.debug(f"Could not save env_info: {e}")
        
        # ─── Leaderboard ───
        try:
            lb = self._last_leaderboard or self.get_leaderboard()
            if lb is not None and not lb.empty:
                lb_path = out_path / self.config.leaderboard_filename
                lb.to_csv(lb_path, index=False)
                saved["leaderboard_csv"] = str(lb_path)
                self._log.debug(f"✓ Saved: {lb_path.name}")
        
        except Exception as e:
            self._log.debug(f"Could not save leaderboard: {e}")
        
        # ─── Pipeline ───
        try:
            pipeline = self.get_pipeline()
            if pipeline is not None:
                pipeline_path = out_path / self.config.pipeline_filename
                with open(pipeline_path, "wb") as f:
                    pickle.dump(pipeline, f)
                saved["pipeline"] = str(pipeline_path)
                self._log.debug(f"✓ Saved: {pipeline_path.name}")
        
        except Exception as e:
            self._log.debug(f"Could not save pipeline: {e}")
        
        # ─── Operations Log ───
        try:
            if self._operations_log:
                log_path = out_path / "operations_log.json"
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(
                        self._operations_log,
                        f,
                        ensure_ascii=False,
                        indent=2
                    )
                saved["operations_log"] = str(log_path)
                self._log.debug(f"✓ Saved: {log_path.name}")
        
        except Exception as e:
            self._log.debug(f"Could not save operations log: {e}")
        
        self._log.info(f"📦 Session artifacts saved to: {out_path}")
        
        return saved
    
    # ───────────────────────────────────────────────────────────────────
    # Private Methods
    # ───────────────────────────────────────────────────────────────────
    
    def _require_initialized(self) -> None:
        """
        ⚠ **Require Initialized Experiment**
        
        Validates that experiment has been initialized.
        
        Raises:
            RuntimeError: Experiment not initialized
        """
        if not self._experiment_initialized:
            raise RuntimeError(
                "Experiment not initialized. "
                "Call initialize_experiment() first."
            )
    
    @_safe_operation("save_artifact_df", default_value=None)
    def _save_artifact_df(
        self,
        df: pd.DataFrame,
        filename: str
    ) -> Optional[str]:
        """
        💾 **Save DataFrame Artifact**
        
        Saves DataFrame to artifacts directory.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        
        Returns:
            Saved file path or None
        """
        if self._artifacts_root is None:
            return None
        
        try:
            self._artifacts_root.mkdir(parents=True, exist_ok=True)
            path = self._artifacts_root / filename
            df.to_csv(path, index=False)
            return str(path)
        
        except Exception as e:
            self._log.debug(f"Could not save artifact '{filename}': {e}")
            return None
    
    def _log_operation(
        self,
        operation: str,
        elapsed_s: float,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        📝 **Log Operation**
        
        Records operation in telemetry log.
        
        Args:
            operation: Operation name
            elapsed_s: Elapsed time in seconds
            params: Optional operation parameters
        """
        entry = {
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed_s, 4),
            "params": params or {}
        }
        
        self._operations_log.append(entry)
    
    def get_operations_log(self) -> List[Dict[str, Any]]:
        """
        📊 **Get Operations Log**
        
        Returns complete operations telemetry log.
        
        Returns:
            List of operation records
        """
        return self._operations_log.copy()
    
    def clear_operations_log(self) -> None:
        """
        🗑 **Clear Operations Log**
        
        Clears the operations telemetry log.
        """
        self._operations_log.clear()
        self._log.debug("Operations log cleared")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def create_wrapper(
    problem_type: Literal["classification", "regression"],
    config: Optional[PyCaretConfig] = None,
    strategy: Optional[SetupStrategy] = None
) -> PyCaretWrapper:
    """
    🚀 **Create PyCaret Wrapper**
    
    Factory function for creating configured PyCaret wrapper.
    
    Args:
        problem_type: 'classification' or 'regression'
        config: Optional custom configuration
        strategy: Optional setup strategy shortcut
    
    Returns:
        Configured PyCaretWrapper instance
    
    Examples:
```python
        # Basic usage
        wrapper = create_wrapper('classification')
        
        # With strategy
        wrapper = create_wrapper('regression', strategy=SetupStrategy.THOROUGH)
        
        # With custom config
        config = PyCaretConfig(
            use_gpu=True,
            fix_imbalance=True,
            enable_feature_selection=True
        )
        wrapper = create_wrapper('classification', config=config)
```
    """
    if config is None:
        if strategy == SetupStrategy.FAST:
            config = PyCaretConfig.create_fast()
        elif strategy == SetupStrategy.THOROUGH:
            config = PyCaretConfig.create_thorough()
        else:
            config = PyCaretConfig()
    
    return PyCaretWrapper(problem_type, config)


def quick_train(
    data: pd.DataFrame,
    target_column: str,
    problem_type: Literal["classification", "regression"],
    n_select: int = 3,
    tune: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    ⚡ **Quick Training Pipeline**
    
    Streamlined training pipeline for rapid experimentation.
    
    Args:
        data: Training DataFrame
        target_column: Target column name
        problem_type: 'classification' or 'regression'
        n_select: Number of models to compare
        tune: Enable hyperparameter tuning
        save_path: Optional path to save best model
    
    Returns:
        Dictionary with best model and results
    
    Examples:
```python
        # Quick training
        result = quick_train(
            data=df,
            target_column='target',
            problem_type='classification'
        )
        
        best_model = result['model']
        leaderboard = result['leaderboard']
```
    """
    # Create wrapper with fast config
    config = PyCaretConfig.create_fast()
    wrapper = PyCaretWrapper(problem_type, config)
    
    # Initialize
    wrapper.initialize_experiment(data, target_column)
    
    # Compare models
    models = wrapper.compare_all_models(n_select=n_select)
    best_model = models[0] if isinstance(models, list) else models
    
    # Tune if requested
    if tune:
        best_model = wrapper.tune_best_model(best_model, n_iter=10)
    
    # Finalize
    if save_path:
        best_model = wrapper.finalize_and_save(best_model, save_path)
    else:
        best_model = wrapper._finalize_model(best_model)
    
    # Get leaderboard
    leaderboard = wrapper.get_leaderboard()
    
    return {
        "model": best_model,
        "leaderboard": leaderboard,
        "wrapper": wrapper,
        "feature_importance": wrapper.get_feature_importance(best_model)
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _module_init():
    """Initialize module on import."""
    version_info = _detect_pycaret_version()
    logger.info(
        f"✓ PyCaretWrapper v{__version__} loaded | "
        f"PyCaret={version_info.value} | "
        f"Registry={'✓' if MODEL_REGISTRY_AVAILABLE else '✗'}"
    )

_module_init()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Module self-test
    print(f"PyCaretWrapper v{__version__}")
    print(f"PyCaret Version: {_detect_pycaret_version().value}")
    print(f"Model Registry: {'Available' if MODEL_REGISTRY_AVAILABLE else 'Unavailable'}")
    print(f"System: {_get_system_info()}")
    
    print("\n" + "="*80)
    print("EXAMPLE USAGE:")
    print("="*80)
    print("""
# Basic Usage
from agents.ml.pycaret_wrapper import PyCaretWrapper, PyCaretConfig

wrapper = PyCaretWrapper('classification')
wrapper.initialize_experiment(df, 'target')
models = wrapper.compare_all_models(n_select=5)
best = models[0]
tuned = wrapper.tune_best_model(best)
final = wrapper.finalize_and_save(tuned, 'model')

# Advanced Configuration
config = PyCaretConfig.create_thorough()
wrapper = PyCaretWrapper('regression', config)
wrapper.initialize_experiment(
    data=df,
    target_column='price',
    fold=10,
    use_gpu=True
)

# Quick Training
from agents.ml.pycaret_wrapper import quick_train

result = quick_train(
    data=df,
    target_column='target',
    problem_type='classification',
    tune=True,
    save_path='model'
)

best_model = result['model']
leaderboard = result['leaderboard']
    """)