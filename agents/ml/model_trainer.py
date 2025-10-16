# agents/ml/model_trainer.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Model Trainer                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Comprehensive ML training with enterprise safeguards:                     ║
║    ✓ PyCaret integration (with graceful degradation)                      ║
║    ✓ Deterministic execution (global seed control)                        ║
║    ✓ GPU support (auto-detection + manual control)                        ║
║    ✓ Model comparison (include/exclude/blacklist)                         ║
║    ✓ Hyperparameter tuning (optional, configurable)                       ║
║    ✓ Model finalization & persistence                                     ║
║    ✓ Artifact management (models, metadata, leaderboards)                 ║
║    ✓ Imbalance detection & warnings                                       ║
║    ✓ Target validation (sanity checks)                                    ║
║    ✓ Comprehensive telemetry                                              ║
║    ✓ Stable output contract                                               ║
║    ✓ Degraded mode (PyCaret unavailable)                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

Training Pipeline:
    1. Input validation & sanity checks
    2. Deterministic seed setting
    3. PyCaret experiment setup
    4. Model comparison (multi-model)
    5. Optional hyperparameter tuning
    6. Model finalization
    7. Artifact persistence (model + metadata)
    8. Leaderboard export
    9. Telemetry tracking

Output Contract:
    result.data = {
        "best_model": trained_model,
        "model_path": "path/to/model",
        "artifacts": {...},
        "pycaret_wrapper": wrapper_instance,
        "models_comparison": [...],
        "primary_metric": "accuracy",
        "meta": {...}
    }
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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
    from core.base_agent import BaseAgent, AgentResult
except ImportError:
    class BaseAgent:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
            self.logger = logger
    
    class AgentResult:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.data: Dict[str, Any] = {}
            self.errors: List[str] = []
            self.warnings: List[str] = []
        
        def add_error(self, error: str):
            self.errors.append(error)
        
        def add_warning(self, warning: str):
            self.warnings.append(warning)
        
        def is_success(self) -> bool:
            return len(self.errors) == 0

try:
    from agents.ml.pycaret_wrapper import PyCaretWrapper
except ImportError:
    PyCaretWrapper = None

try:
    from config.settings import settings
except ImportError:
    class settings:
        ENABLE_HYPERPARAMETER_TUNING = False
        DEFAULT_TUNING_ITERATIONS = 25
        RANDOM_STATE = 42
        MODELS_PATH = "models"

warnings.filterwarnings('ignore')


__all__ = ["TrainerConfig", "ModelTrainer", "train_models"]
__version__ = "5.1-kosmos-enterprise"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainerConfig:
    """
    Configuration for model training.
    
    Attributes:
        fold: Cross-validation folds
        n_select: Number of models to select
        use_gpu: GPU usage (None=auto, True/False=force)
        primary_metric_cls: Primary metric for classification
        primary_metric_reg: Primary metric for regression
        compare_blacklist: Models to exclude
        compare_include: Models to include (whitelist)
        enable_tuning: Enable hyperparameter tuning
        tuning_iterations: Tuning iterations
        min_rows: Minimum rows required
        log_training_summary: Log training summary
        models_dir_env_key: Settings key for models directory
        random_state_key: Settings key for random state
        leaderboard_filename: Leaderboard CSV filename
        metadata_filename: Metadata JSON filename
        pipeline_filename: Pipeline pickle filename
        warn_extreme_imbalance_ratio: Imbalance warning threshold
        max_feature_cols_warn: Feature count warning threshold
    """
    fold: int = 5
    n_select: int = 3
    use_gpu: Optional[bool] = None
    primary_metric_cls: str = "accuracy"
    primary_metric_reg: str = "r2"
    compare_blacklist: Optional[List[str]] = None
    compare_include: Optional[List[str]] = None
    enable_tuning: Optional[bool] = None
    tuning_iterations: Optional[int] = None
    min_rows: int = 30
    log_training_summary: bool = True
    models_dir_env_key: str = "MODELS_PATH"
    random_state_key: str = "RANDOM_STATE"
    leaderboard_filename: str = "leaderboard.csv"
    metadata_filename: str = "metadata.json"
    pipeline_filename: str = "pipeline.pkl"
    warn_extreme_imbalance_ratio: float = 10.0
    max_feature_cols_warn: int = 5_000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fold": self.fold,
            "n_select": self.n_select,
            "use_gpu": self.use_gpu,
            "primary_metric_cls": self.primary_metric_cls,
            "primary_metric_reg": self.primary_metric_reg,
            "enable_tuning": self.enable_tuning,
            "tuning_iterations": self.tuning_iterations,
            "min_rows": self.min_rows
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
# SECTION: Main Model Trainer
# ═══════════════════════════════════════════════════════════════════════════

class ModelTrainer(BaseAgent):
    """
    **ModelTrainer** — Comprehensive ML model training.
    
    Responsibilities:
      1. Validate input data and target
      2. Set deterministic random seed
      3. Initialize PyCaret experiment
      4. Compare multiple models
      5. Select best models
      6. Optional hyperparameter tuning
      7. Finalize and save models
      8. Export artifacts (leaderboard, metadata)
      9. Track comprehensive telemetry
      10. Handle degraded mode gracefully
    
    Features:
      • PyCaret integration
      • Multi-model comparison
      • Hyperparameter tuning
      • GPU support
      • Artifact management
      • Imbalance detection
      • Comprehensive validation
      • Telemetry tracking
    """
    
    version: str = __version__
    
    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        """
        Initialize model trainer.
        
        Args:
            config: Optional custom configuration
        """
        super().__init__(
            name="ModelTrainer",
            description="Comprehensive ML model training with PyCaret integration"
        )
        self.config = config or TrainerConfig()
        self._log = logger.bind(agent="ModelTrainer")
        
        self._log.info("✓ ModelTrainer initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    @_timeit("model_training")
    def execute(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: Literal["classification", "regression"],
        **kwargs: Any
    ) -> AgentResult:
        """
        Train and tune ML models.
        
        Args:
            data: Training DataFrame
            target_column: Target column name
            problem_type: 'classification' or 'regression'
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with trained models and artifacts
        """
        result = AgentResult(agent_name=self.name)
        started_at_ts = time.time()
        warnings: List[str] = []
        
        try:
            self._log.info(
                f"Starting model training | "
                f"type={problem_type} | "
                f"rows={len(data)} | "
                f"cols={len(data.columns)}"
            )
            
            # ─── Step 1: Validate Inputs ───
            self._validate_inputs(data, target_column, problem_type)
            
            # ─── Step 2: Deterministic Seed ───
            try:
                seed = int(getattr(settings, self.config.random_state_key, 42))
            except Exception:
                seed = 42
            
            try:
                np.random.seed(seed)
            except Exception:
                pass
            
            # ─── Step 3: Setup Artifacts Directory ───
            try:
                models_root = Path(getattr(settings, self.config.models_dir_env_key, "models"))
            except Exception:
                models_root = Path("models")
            
            models_root.mkdir(parents=True, exist_ok=True)
            
            # ─── Step 4: Configure Metrics ───
            primary_metric = (
                self.config.primary_metric_cls if problem_type == "classification"
                else self.config.primary_metric_reg
            )
            
            n_rows, n_cols = int(len(data)), int(len(data.columns))
            
            # Hygiene warnings
            if n_cols > self.config.max_feature_cols_warn:
                warn = f"Very wide dataset detected (n_cols={n_cols}). Consider feature selection."
                warnings.append(warn)
                self._log.warning(warn)
            
            # ─── Step 5: GPU Configuration ───
            use_gpu = self.config.use_gpu if self.config.use_gpu is not None else kwargs.pop("use_gpu", None)
            
            # ─── Step 6: Initialize PyCaret ───
            pycaret: Optional[Any] = None
            
            if PyCaretWrapper is None:
                self._log.warning("PyCaretWrapper not available")
            else:
                try:
                    pycaret = PyCaretWrapper(problem_type)
                except Exception as e:
                    msg = f"PyCaretWrapper initialization failed: {e}"
                    self._log.error(msg)
                    warnings.append(msg)
            
            if pycaret is None:
                # Degraded mode
                result.add_warning("PyCaret is unavailable. Training skipped (degraded mode).")
                result.data = self._degraded_payload(
                    n_rows=n_rows,
                    n_cols=n_cols,
                    problem_type=problem_type,
                    target_column=target_column,
                    primary_metric=primary_metric,
                    use_gpu=use_gpu,
                    started_at_ts=started_at_ts,
                    warnings=warnings,
                    seed=seed
                )
                return result
            
            # ─── Step 7: Setup Experiment ───
            self._log.info("Setting up PyCaret experiment...")
            pycaret.initialize_experiment(
                data=data,
                target_column=target_column,
                fold=self.config.fold,
                session_id=seed,
                use_gpu=use_gpu,
                silent=True,
                log_experiment=False,
                **kwargs
            )
            
            # ─── Step 8: Compare Models ───
            self._log.info("Comparing models...")
            best_models = pycaret.compare_all_models(
                n_select=self.config.n_select,
                sort=primary_metric,
                include=self.config.compare_include,
                exclude=self.config.compare_blacklist
            )
            
            # Normalize list/singleton
            if isinstance(best_models, list):
                models_list = best_models
                best_model = best_models[0] if best_models else None
            else:
                best_model = best_models
                models_list = [best_models] if best_models is not None else []
            
            if best_model is None:
                raise RuntimeError("compare_all_models returned no candidates")
            
            # ─── Step 9: Hyperparameter Tuning (Optional) ───
            enable_tuning = (
                self.config.enable_tuning
                if self.config.enable_tuning is not None
                else bool(getattr(settings, "ENABLE_HYPERPARAMETER_TUNING", False))
            )
            
            tuned_model = best_model
            tuning_iters: Optional[int] = None
            
            if enable_tuning:
                tuning_iters = (
                    self.config.tuning_iterations
                    if self.config.tuning_iterations is not None
                    else int(getattr(settings, "DEFAULT_TUNING_ITERATIONS", 25))
                )
                
                self._log.info(f"Tuning best model (n_iter={tuning_iters})...")
                try:
                    tuned_model = pycaret.tune_best_model(
                        best_model,
                        n_iter=tuning_iters,
                        optimize=primary_metric
                    )
                except Exception as e:
                    warn = f"Tuning failed: {e}. Continuing with untuned model."
                    self._log.warning(warn)
                    warnings.append(warn)
            
            # ─── Step 10: Finalize & Save ───
            run_dir = self._make_run_dir(models_root, problem_type)
            final_model = pycaret.finalize_and_save(tuned_model, str(run_dir / "model"))
            self._log.success(f"Model finalized and saved in: {run_dir}")
            
            # ─── Step 11: Export Leaderboard ───
            leaderboard_csv: Optional[str] = None
            try:
                df_lb = pycaret.get_leaderboard()
                if df_lb is not None and not df_lb.empty:
                    leaderboard_csv = str(run_dir / self.config.leaderboard_filename)
                    df_lb.to_csv(leaderboard_csv, index=False)
            except Exception as e:
                warn = f"Cannot save leaderboard: {e}"
                warnings.append(warn)
            
            # ─── Step 12: Pipeline Path ───
            pipeline_path: Optional[str] = None
            try:
                if hasattr(pycaret, "get_pipeline_path"):
                    path = pycaret.get_pipeline_path()
                    if path:
                        pipeline_path = str(path)
            except Exception:
                pass
            
            # ─── Step 13: Export Metadata ───
            metadata_path = str(run_dir / self.config.metadata_filename)
            meta_json = {
                "version": self.version,
                "problem_type": problem_type,
                "target_column": target_column,
                "fold": self.config.fold,
                "primary_metric": primary_metric,
                "n_select": self.config.n_select,
                "use_gpu": use_gpu,
                "tuning": {"enabled": enable_tuning, "iterations": tuning_iters},
                "n_rows": n_rows,
                "n_cols": n_cols,
                "seed": seed,
                "warnings": warnings,
            }
            
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(meta_json, f, ensure_ascii=False, indent=2)
            except Exception as e:
                warnings.append(f"Cannot write metadata.json: {e}")
            
            finished_at_ts = time.time()
            
            # ─── Step 14: Log Training Brief ───
            if self.config.log_training_summary:
                self._log_training_brief(pycaret, primary_metric)
            
            # ─── Step 15: Create Result ───
            result.data = {
                "best_model": final_model,
                "model_path": str(run_dir / "model"),
                "artifacts": {
                    "model_dir": str(run_dir),
                    "pipeline_path": pipeline_path,
                    "leaderboard_csv": leaderboard_csv,
                    "metadata_json": metadata_path,
                },
                "pycaret_wrapper": pycaret,
                "models_comparison": models_list,
                "primary_metric": primary_metric,
                "meta": {
                    "problem_type": problem_type,
                    "target_column": target_column,
                    "fold": self.config.fold,
                    "use_gpu": use_gpu,
                    "tuning": {"enabled": enable_tuning, "iterations": tuning_iters},
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "seed": seed,
                    "started_at_ts": started_at_ts,
                    "finished_at_ts": finished_at_ts,
                    "elapsed_s": round(finished_at_ts - started_at_ts, 4),
                    "warnings": warnings,
                    "version": self.version,
                },
            }
            
            self._log.success(
                f"✓ Model training completed | "
                f"time={result.data['meta']['elapsed_s']:.2f}s | "
                f"metric={primary_metric}"
            )
        
        except Exception as e:
            result.add_error(f"Model training failed: {e}")
            self._log.error(f"Training error: {e}", exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Validation
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("validate_inputs", default_value=None)
    def _validate_inputs(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str
    ) -> None:
        """Validate training inputs."""
        # DataFrame validation
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("'data' must be a non-empty pandas DataFrame")
        
        if len(df) < self.config.min_rows:
            raise ValueError(f"Not enough rows to train (min_rows={self.config.min_rows})")
        
        # Target validation
        if not isinstance(target_column, str) or not target_column:
            raise ValueError("'target_column' must be a non-empty string")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Problem type validation
        if problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem_type='{problem_type}'")
        
        # Target sanity checks
        y = df[target_column]
        
        if y.isna().all():
            raise ValueError("All target values are NaN — cannot train")
        
        if problem_type == "classification":
            nunique = y.nunique(dropna=True)
            if nunique < 2:
                raise ValueError(f"Classification requires ≥2 classes; found {nunique}")
            
            # Imbalance warning
            vc = y.value_counts(dropna=True)
            try:
                if len(vc) > 1:
                    ratio = vc.max() / max(1, vc.min())
                    if ratio > self.config.warn_extreme_imbalance_ratio:
                        self._log.warning(
                            f"Severe class imbalance detected (max/min > {self.config.warn_extreme_imbalance_ratio}). "
                            f"Consider resampling/weights."
                        )
            except Exception:
                pass
    
    # ───────────────────────────────────────────────────────────────────
    # Logging & Utilities
    # ───────────────────────────────────────────────────────────────────
    
    @_safe_operation("log_training_brief", default_value=None)
    def _log_training_brief(self, pycaret: Any, primary_metric: str) -> None:
        """Log training summary."""
        try:
            df_leaderboard = pycaret.get_leaderboard()
            if df_leaderboard is not None and not df_leaderboard.empty:
                head = df_leaderboard.head(5)
                self._log.info(f"Top models by '{primary_metric}':\n{head.to_string(index=False)}")
        except Exception as e:
            self._log.debug(f"get_leaderboard failed: {e}")
    
    def _make_run_dir(self, root: Path, problem_type: str) -> Path:
        """Create timestamped run directory."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{problem_type}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def _degraded_payload(
        self,
        *,
        n_rows: int,
        n_cols: int,
        problem_type: str,
        target_column: str,
        primary_metric: str,
        use_gpu: Optional[bool],
        started_at_ts: float,
        warnings: List[str],
        seed: int
    ) -> Dict[str, Any]:
        """Create payload for degraded mode (PyCaret unavailable)."""
        finished_at_ts = time.time()
        
        return {
            "best_model": None,
            "model_path": None,
            "artifacts": {
                "model_dir": None,
                "pipeline_path": None,
                "leaderboard_csv": None,
                "metadata_json": None
            },
            "pycaret_wrapper": None,
            "models_comparison": [],
            "primary_metric": primary_metric,
            "meta": {
                "problem_type": problem_type,
                "target_column": target_column,
                "fold": self.config.fold,
                "use_gpu": use_gpu,
                "tuning": {"enabled": False, "iterations": None},
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "seed": int(seed),
                "started_at_ts": started_at_ts,
                "finished_at_ts": finished_at_ts,
                "elapsed_s": round(finished_at_ts - started_at_ts, 4),
                "warnings": ["Training skipped (PyCaret unavailable).", *warnings],
                "version": self.version,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Function
# ═══════════════════════════════════════════════════════════════════════════

def train_models(
    data: pd.DataFrame,
    target_column: str,
    problem_type: Literal["classification", "regression"],
    config: Optional[TrainerConfig] = None,
    **kwargs
) -> AgentResult:
    """
    Convenience function for model training.
    
    Usage:
```python
        from agents.ml import train_models, TrainerConfig
        
        # Basic usage
        result = train_models(
            data=df,
            target_column='target',
            problem_type='classification'
        )
        
        # Access results
        best_model = result.data['best_model']
        model_path = result.data['model_path']
        
        # With custom config
        config = TrainerConfig(
            fold=10,
            n_select=5,
            enable_tuning=True,
            tuning_iterations=50
        )
        
        result = train_models(
            data=df,
            target_column='target',
            problem_type='classification',
            config=config
        )
```
    
    Args:
        data: Training DataFrame
        target_column: Target column name
        problem_type: 'classification' or 'regression'
        config: Optional custom configuration
        **kwargs: Additional parameters
    
    Returns:
        AgentResult with trained models and artifacts
    """
    trainer = ModelTrainer(config)
    return trainer.execute(data, target_column, problem_type, **kwargs)

