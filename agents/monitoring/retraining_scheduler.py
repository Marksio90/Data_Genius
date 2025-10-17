# agents/monitoring/retraining_scheduler.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Retraining Scheduler v6.0        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ENTERPRISE-GRADE INTELLIGENT RETRAINING ORCHESTRATION                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Multi-Signal Decision Engine (drift, performance, age, volume)        ║
║  ✓ Advanced Scoring System with Configurable Weights                     ║
║  ✓ Hard Triggers for Critical Conditions                                 ║
║  ✓ Operational Policies (cooldown, rate limits, volume gates)            ║
║  ✓ Intelligent Schedule Generation (cron + iCal)                         ║
║  ✓ Optional Immediate Retraining Execution                               ║
║  ✓ Comprehensive Audit Logging                                           ║
║  ✓ Decision Reasoning & Explainability                                   ║
║  ✓ Production-Ready with Safety Guards                                   ║
║  ✓ Time Zone Support                                                     ║
║  ✓ Integration with DriftDetector & PerformanceTracker                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │              RetrainingScheduler Core                           │
    ├─────────────────────────────────────────────────────────────────┤
    │  1. Signal Collection (drift, performance, age, volume)         │
    │  2. Scoring Engine (weighted combination)                       │
    │  3. Trigger Detection (hard & soft triggers)                    │
    │  4. Policy Enforcement (cooldown, limits, gates)                │
    │  5. Schedule Generation (cron, iCal, next window)               │
    │  6. Decision Making (should_retrain logic)                      │
    │  7. Optional Execution (immediate retraining)                   │
    │  8. Audit Logging (persistent history)                          │
    └─────────────────────────────────────────────────────────────────┘

Decision Factors:
    • Data Drift       → % of drifted features
    • Target Drift     → Distribution shift in target
    • Performance      → Metric degradation vs baseline
    • Model Age        → Days since last training
    • Data Volume      → New samples available
    
Policies:
    • Cooldown Period  → Minimum days between retrains
    • Weekly Limit     → Maximum retrains per week
    • Volume Gate      → Minimum samples required
    
Triggers:
    • HARD: Critical drift, target drift, critical age
    • SOFT: Warnings, performance drops
    
Schedule:
    • Preferred time window
    • Days of week restrictions
    • Time zone aware
    • Cron expression
    • iCal VEVENT format

Usage:
```python
    from agents.monitoring import RetrainingScheduler, RetrainPolicy
    
    # Basic usage
    scheduler = RetrainingScheduler()
    
    result = scheduler.execute(
        problem_type='classification',
        drift_report=drift_detector_result.data,
        performance_data=tracker_result.data,
        model_path='models/my_model.pkl',
        new_samples=10000
    )
    
    # Check decision
    if result.data['decision']['should_retrain']:
        priority = result.data['decision']['priority']
        schedule = result.data['schedule']
        print(f"🔄 Retraining recommended: {priority} priority")
        print(f"Next window: {schedule['next_time_local_iso']}")
        print(f"Cron: {schedule['cron']}")
    
    # Custom policy
    policy = RetrainPolicy(
        drift_crit_pct=25.0,
        cooldown_days=5,
        max_retrains_per_week=3
    )
    scheduler = RetrainingScheduler(policy)
    
    # With immediate execution
    result = scheduler.execute(
        problem_type='classification',
        drift_report=drift_data,
        train_data=new_training_data,
        target_column='target',
        orchestrator=ml_orchestrator
    )
```
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

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
        "logs/retraining_scheduler_{time:YYYY-MM-DD}.log",
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
# Dependencies
# ═══════════════════════════════════════════════════════════════════════════

try:
    from core.base_agent import BaseAgent, AgentResult
except ImportError:
    logger.warning("⚠ core.base_agent not found - using fallback")
    
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
    from config.settings import settings
except ImportError:
    logger.warning("⚠ config.settings not found - using defaults")
    
    class Settings:
        METRICS_PATH: str = "metrics"
        TIMEZONE: str = "UTC"
    
    settings = Settings()

# Time zone support
try:
    from zoneinfo import ZoneInfo
    HAS_ZONEINFO = True
except ImportError:
    ZoneInfo = None
    HAS_ZONEINFO = False
    logger.warning("⚠ zoneinfo not available - using UTC only")

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["RetrainPolicy", "RetrainingScheduler", "schedule_retraining"]
__version__ = "6.0.0-enterprise"
__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=False)
class RetrainPolicy:
    """
    🎯 **Retraining Policy Configuration**
    
    Complete policy for intelligent retraining decisions.
    
    Drift Thresholds:
        drift_warn_pct: Warning threshold for % drifted features
        drift_crit_pct: Critical threshold for % drifted features
        target_drift_triggers: Target drift always triggers retraining
        
    Performance Thresholds:
        max_accuracy_drop_pct: Max acceptable accuracy drop (percentage points)
        max_f1_drop_pct: Max acceptable F1 drop (percentage points)
        max_r2_drop_abs: Max acceptable R² drop (absolute)
        max_rmse_increase_pct: Max RMSE increase vs baseline (%)
        max_mae_increase_pct: Max MAE increase vs baseline (%)
        
    Age & Volume:
        age_warn_days: Warning threshold for model age (days)
        age_crit_days: Critical threshold for model age (days)
        min_new_samples: Minimum new samples required
        
    Operational Policies:
        cooldown_days: Minimum days between retrains
        max_retrains_per_week: Maximum retrains per 7 days
        require_approval_for_production: Require manual approval
        
    Scheduling:
        preferred_hour: Preferred hour for retraining (0-23)
        preferred_minute: Preferred minute (0-59)
        days_of_week: Allowed days (0=Mon, 6=Sun), None=all days
        timezone: Time zone for scheduling
        
    Scoring:
        weight_drift: Weight for drift component
        weight_performance: Weight for performance component
        weight_age: Weight for age component
        priority_high_threshold: Score threshold for high priority
        priority_medium_threshold: Score threshold for medium priority
        
    Advanced:
        enable_automatic_execution: Allow automatic retraining
        notify_on_decision: Send notifications
        audit_retention_days: Days to retain audit logs
    """
    
    # Drift thresholds
    drift_warn_pct: float = 10.0
    drift_crit_pct: float = 30.0
    target_drift_triggers: bool = True
    
    # Performance thresholds
    max_accuracy_drop_pct: float = 2.0
    max_f1_drop_pct: float = 2.0
    max_r2_drop_abs: float = 0.03
    max_rmse_increase_pct: float = 15.0
    max_mae_increase_pct: float = 15.0
    
    # Age & volume
    age_warn_days: int = 14
    age_crit_days: int = 30
    min_new_samples: int = 5_000
    
    # Operational policies
    cooldown_days: int = 3
    max_retrains_per_week: int = 2
    require_approval_for_production: bool = False
    
    # Scheduling
    preferred_hour: int = 2
    preferred_minute: int = 30
    days_of_week: Optional[List[int]] = None
    timezone: str = getattr(settings, "TIMEZONE", "UTC")
    
    # Scoring
    weight_drift: float = 0.5
    weight_performance: float = 0.3
    weight_age: float = 0.2
    priority_high_threshold: float = 0.7
    priority_medium_threshold: float = 0.4
    
    # Advanced
    enable_automatic_execution: bool = False
    notify_on_decision: bool = True
    audit_retention_days: int = 90
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.preferred_hour <= 23:
            raise ValueError(f"preferred_hour must be 0-23, got {self.preferred_hour}")
        
        if not 0 <= self.preferred_minute <= 59:
            raise ValueError(f"preferred_minute must be 0-59, got {self.preferred_minute}")
        
        total_weight = self.weight_drift + self.weight_performance + self.weight_age
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(
                f"Weights must sum to ~1.0, got {total_weight:.3f}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def create_aggressive(cls) -> 'RetrainPolicy':
        """Create aggressive policy (lower thresholds, more frequent)."""
        return cls(
            drift_crit_pct=20.0,
            age_crit_days=14,
            cooldown_days=1,
            max_retrains_per_week=5,
            priority_medium_threshold=0.3
        )
    
    @classmethod
    def create_conservative(cls) -> 'RetrainPolicy':
        """Create conservative policy (higher thresholds, less frequent)."""
        return cls(
            drift_crit_pct=40.0,
            age_crit_days=60,
            cooldown_days=7,
            max_retrains_per_week=1,
            priority_high_threshold=0.8
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Retraining Scheduler (Main Class)
# ═══════════════════════════════════════════════════════════════════════════

class RetrainingScheduler(BaseAgent):
    """
    🚀 **RetrainingScheduler PRO Master Enterprise ++++**
    
    Enterprise-grade intelligent retraining orchestration system.
    
    Capabilities:
      1. Multi-signal analysis (drift, performance, age, volume)
      2. Weighted scoring system
      3. Hard & soft trigger detection
      4. Operational policy enforcement
      5. Intelligent schedule generation
      6. Optional immediate execution
      7. Comprehensive audit logging
      8. Decision reasoning & explainability
      9. Production safety guards
     10. Time zone aware scheduling
    
    Decision Flow:
```
        ┌──────────────────┐
        │ Signal Collection│
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Score Computation│
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Trigger Detection│
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Policy Enforcement│
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Decision + Schedule│
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Optional Execution│
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Audit Logging    │
        └──────────────────┘
```
    
    Usage:
```python
        # Basic decision
        scheduler = RetrainingScheduler()
        
        result = scheduler.execute(
            problem_type='classification',
            drift_report=drift_data,
            performance_data=perf_data,
            model_path='model.pkl',
            new_samples=15000
        )
        
        # With immediate execution
        result = scheduler.execute(
            problem_type='classification',
            drift_report=drift_data,
            train_data=train_df,
            target_column='target',
            orchestrator=ml_pipeline
        )
```
    """
    
    version: str = __version__
    
    def __init__(self, policy: Optional[RetrainPolicy] = None):
        """
        Initialize retraining scheduler.
        
        Args:
            policy: Optional custom policy
        """
        super().__init__(
            name="RetrainingScheduler",
            description="Enterprise intelligent retraining orchestration"
        )
        
        self.policy = policy or RetrainPolicy()
        self._log = logger.bind(agent="RetrainingScheduler", version=self.version)
        
        # Setup paths
        metrics_path = getattr(settings, "METRICS_PATH", "metrics")
        self.metrics_path = Path(metrics_path)
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        
        self.log_path = self.metrics_path / "retraining_log.csv"
        
        self._log.info(f"✓ RetrainingScheduler v{self.version} initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Main Execution
    # ───────────────────────────────────────────────────────────────────
    
    def execute(
        self,
        problem_type: Literal["classification", "regression"],
        *,
        drift_report: Optional[Dict[str, Any]] = None,
        performance_data: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        last_train_ts: Optional[str] = None,
        new_samples: Optional[int] = None,
        force: bool = False,
        # Immediate retraining
        train_data: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        orchestrator: Optional[Any] = None,
        orchestrator_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """
        🎯 **Execute Retraining Decision & Scheduling**
        
        Analyzes signals, makes retraining decision, generates schedule,
        and optionally executes immediate retraining.
        
        Args:
            problem_type: 'classification' or 'regression'
            drift_report: Output from DriftDetector
            performance_data: Output from PerformanceTracker
            model_path: Path to current model file
            last_train_ts: Last training timestamp (ISO format)
            new_samples: Number of new samples available
            force: Force retraining regardless of signals
            train_data: Training data (for immediate execution)
            target_column: Target column name
            orchestrator: ML orchestrator instance
            orchestrator_kwargs: Additional orchestrator arguments
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with decision, schedule, and audit info
        """
        result = AgentResult(agent_name=self.name)
        t_start = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        
        try:
            self._log.info(
                f"🔄 Starting retraining analysis | "
                f"type={problem_type} | "
                f"samples={new_samples or 'unknown'}"
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: Signal Collection
            # ═══════════════════════════════════════════════════════════
            
            drift_signals = self._extract_drift_signals(drift_report)
            perf_signals = self._extract_performance_signals(
                performance_data, problem_type
            )
            age_days = self._calculate_model_age(model_path, last_train_ts)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: Score Computation
            # ═══════════════════════════════════════════════════════════
            
            score, score_parts = self._compute_score(
                drift_signals, perf_signals, age_days
            )
            priority = self._determine_priority(score)
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: Trigger Detection
            # ═══════════════════════════════════════════════════════════
            
            triggers, has_hard_trigger = self._detect_triggers(
                drift_signals, perf_signals, age_days
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 4: Policy Enforcement
            # ═══════════════════════════════════════════════════════════
            
            cooldown_ok, cooldown_info = self._check_cooldown()
            weekly_ok, weekly_info = self._check_weekly_limit()
            volume_ok = self._check_volume_gate(new_samples)
            
            policy_gates_passed = cooldown_ok and weekly_ok and volume_ok
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 5: Decision Making
            # ═══════════════════════════════════════════════════════════
            
            should_retrain = self._make_decision(
                score=score,
                has_hard_trigger=has_hard_trigger,
                policy_gates_passed=policy_gates_passed,
                force=force
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 6: Schedule Generation
            # ═══════════════════════════════════════════════════════════
            
            schedule = self._generate_schedule()
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 7: Optional Immediate Execution
            # ═══════════════════════════════════════════════════════════
            
            retrain_result = None
            execution_status = "DECIDED_NO_ACTION"
            
            if should_retrain:
                if orchestrator and train_data is not None and target_column:
                    self._log.info("🚀 Executing immediate retraining...")
                    
                    retrain_ok, retrain_result = self._execute_retraining(
                        orchestrator=orchestrator,
                        train_data=train_data,
                        target_column=target_column,
                        problem_type=problem_type,
                        orchestrator_kwargs=orchestrator_kwargs or {}
                    )
                    
                    execution_status = "RETRAIN_OK" if retrain_ok else "RETRAIN_FAILED"
                else:
                    execution_status = "SCHEDULED"
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 8: Decision Reasoning
            # ═══════════════════════════════════════════════════════════
            
            reasoning = self._generate_reasoning(
                should_retrain=should_retrain,
                score=score,
                priority=priority,
                score_parts=score_parts,
                triggers=triggers,
                cooldown_ok=cooldown_ok,
                weekly_ok=weekly_ok,
                volume_ok=volume_ok,
                force=force
            )
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 9: Audit Logging
            # ═══════════════════════════════════════════════════════════
            
            self._append_audit_log({
                "timestamp": timestamp.isoformat(),
                "problem_type": problem_type,
                "decision": should_retrain,
                "priority": priority,
                "score": round(score, 6),
                "drift_pct": drift_signals["pct"],
                "target_drift": drift_signals["target_drift"],
                "perf_delta": perf_signals["primary_delta"],
                "age_days": age_days,
                "new_samples": new_samples or -1,
                "cooldown_ok": cooldown_ok,
                "weekly_ok": weekly_ok,
                "volume_ok": volume_ok,
                "status": execution_status,
                "forced": force
            })
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 10: Telemetry
            # ═══════════════════════════════════════════════════════════
            
            elapsed_s = time.perf_counter() - t_start
            
            telemetry = {
                "elapsed_s": round(elapsed_s, 4),
                "version": self.version,
                "timestamp": timestamp.isoformat(),
                "policy_summary": {
                    "cooldown_days": self.policy.cooldown_days,
                    "max_retrains_per_week": self.policy.max_retrains_per_week,
                    "min_new_samples": self.policy.min_new_samples
                }
            }
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 11: Assemble Result
            # ═══════════════════════════════════════════════════════════
            
            result.data = {
                "decision": {
                    "should_retrain": should_retrain,
                    "priority": priority,
                    "score": float(score),
                    "score_parts": score_parts,
                    "triggers": triggers,
                    "has_hard_trigger": has_hard_trigger,
                    "force": force,
                    "reasoning": reasoning
                },
                "gates": {
                    "cooldown": cooldown_info,
                    "weekly_limit": weekly_info,
                    "volume_ok": volume_ok,
                    "all_passed": policy_gates_passed
                },
                "signals": {
                    "drift": drift_signals,
                    "performance": perf_signals,
                    "age_days": age_days,
                    "new_samples": new_samples or -1
                },
                "schedule": schedule,
                "execution": {
                    "status": execution_status,
                    "result": retrain_result
                },
                "audit_log_path": str(self.log_path),
                "telemetry": telemetry
            }
            
            # Log summary
            action = "✓ Retraining recommended" if should_retrain else "○ No retraining needed"
            self._log.success(
                f"{action} | "
                f"priority={priority} | "
                f"score={score:.3f} | "
                f"time={elapsed_s:.2f}s"
            )
        
        except Exception as e:
            error_msg = f"Retraining scheduling failed: {type(e).__name__}: {str(e)}"
            result.add_error(error_msg)
            self._log.error(error_msg, exc_info=True)
        
        return result
    
    # ───────────────────────────────────────────────────────────────────
    # Signal Extraction
    # ───────────────────────────────────────────────────────────────────
    
    def _extract_drift_signals(
        self,
        drift_report: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract drift signals from DriftDetector output.
        
        Args:
            drift_report: DriftDetector result data
        
        Returns:
            Dictionary with drift signals
        """
        signals = {
            "pct": 0.0,
            "target_drift": False,
            "top_features": [],
            "n_drifted": 0
        }
        
        if not drift_report:
            return signals
        
        try:
            # Handle nested structure
            data_drift = drift_report.get("data_drift", drift_report)
            
            if isinstance(data_drift, dict):
                # Multiple key names for flexibility
                signals["pct"] = float(
                    data_drift.get("drift_score") or
                    data_drift.get("pct_drifted_features") or
                    0.0
                )
                
                drifted = data_drift.get("drifted_features", []) or []
                signals["top_features"] = list(drifted[:5])
                signals["n_drifted"] = len(drifted)
            
            # Target drift
            target_drift = drift_report.get("target_drift", {})
            if isinstance(target_drift, dict):
                signals["target_drift"] = bool(target_drift.get("is_drift", False))
        
        except Exception as e:
            self._log.debug(f"Drift signal extraction failed: {e}")
        
        return signals
    
    def _extract_performance_signals(
        self,
        performance_data: Optional[Dict[str, Any]],
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Extract performance signals from PerformanceTracker output.
        
        Args:
            performance_data: PerformanceTracker result data
            problem_type: Problem type
        
        Returns:
            Dictionary with performance signals
        """
        signals = {
            "primary_delta": 0.0,
            "primary_metric": "accuracy" if problem_type == "classification" else "r2",
            "all_deltas": {}
        }
        
        if not performance_data:
            return signals
        
        try:
            comparison = performance_data.get("comparison", {})
            
            if not isinstance(comparison, dict):
                return signals
            
            # Get primary metric
            primary_key = signals["primary_metric"]
            metrics = comparison.get("metrics", comparison)
            
            if isinstance(metrics, dict) and primary_key in metrics:
                metric_data = metrics[primary_key]
                
                if isinstance(metric_data, dict):
                    signals["primary_delta"] = float(metric_data.get("delta", 0.0))
                    signals["all_deltas"] = {
                        k: v.get("delta", 0) if isinstance(v, dict) else 0
                        for k, v in metrics.items()
                    }
        
        except Exception as e:
            self._log.debug(f"Performance signal extraction failed: {e}")
        
        return signals
    
    def _calculate_model_age(
        self,
        model_path: Optional[str],
        last_train_ts: Optional[str]
    ) -> int:
        """
        Calculate model age in days.
        
        Args:
            model_path: Path to model file
            last_train_ts: Last training timestamp (ISO format)
        
        Returns:
            Age in days, or -1 if unknown
        """
        # Try timestamp first
        if last_train_ts:
            try:
                dt = datetime.fromisoformat(last_train_ts.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                age = (now - dt.astimezone(timezone.utc)).days
                return max(0, age)
            except Exception as e:
                self._log.debug(f"Timestamp parsing failed: {e}")
        
        # Try file modification time
        if model_path:
            try:
                path = Path(model_path)
                if path.exists():
                    mtime = datetime.fromtimestamp(
                        path.stat().st_mtime,
                        tz=timezone.utc
                    )
                    now = datetime.now(timezone.utc)
                    age = (now - mtime).days
                    return max(0, age)
            except Exception as e:
                self._log.debug(f"File age calculation failed: {e}")
        
        return -1  # Unknown age
    
    # ───────────────────────────────────────────────────────────────────
    # Scoring & Triggers
    # ───────────────────────────────────────────────────────────────────
    
    def _compute_score(
        self,
        drift_signals:Dict[str, Any],
        perf_signals: Dict[str, Any],
        age_days: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted retraining score.
        
        Args:
            drift_signals: Drift signals
            perf_signals: Performance signals
            age_days: Model age in days
        
        Returns:
            Tuple of (total_score, component_scores)
        """
        policy = self.policy
        
        # ═══════════════════════════════════════════════════════════════
        # Drift Component (0-1)
        # ═══════════════════════════════════════════════════════════════
        
        drift_pct = drift_signals["pct"]
        
        if drift_pct >= policy.drift_crit_pct:
            drift_score = 1.0
        elif drift_pct >= policy.drift_warn_pct:
            # Linear interpolation between warn and crit
            range_size = policy.drift_crit_pct - policy.drift_warn_pct
            drift_score = 0.5 + 0.5 * (
                (drift_pct - policy.drift_warn_pct) / max(range_size, 1.0)
            )
        else:
            # Below warning
            drift_score = drift_pct / max(policy.drift_warn_pct, 1.0) * 0.5
        
        # ═══════════════════════════════════════════════════════════════
        # Performance Component (0-1)
        # ═══════════════════════════════════════════════════════════════
        
        primary_delta = perf_signals["primary_delta"]
        
        if primary_delta >= 0:
            # No degradation or improvement
            perf_score = 0.0
        else:
            # Degradation detected
            if perf_signals["primary_metric"] in ["accuracy", "f1"]:
                # Classification metrics (percentage points)
                threshold = policy.max_accuracy_drop_pct / 100.0
                perf_score = min(1.0, abs(primary_delta) / max(threshold, 0.001))
            else:
                # Regression metrics (absolute)
                threshold = policy.max_r2_drop_abs
                perf_score = min(1.0, abs(primary_delta) / max(threshold, 0.001))
        
        # ═══════════════════════════════════════════════════════════════
        # Age Component (0-1)
        # ═══════════════════════════════════════════════════════════════
        
        if age_days < 0:
            age_score = 0.0
        elif age_days >= policy.age_crit_days:
            age_score = 1.0
        elif age_days >= policy.age_warn_days:
            # Linear interpolation between warn and crit
            range_size = policy.age_crit_days - policy.age_warn_days
            age_score = 0.5 + 0.5 * (
                (age_days - policy.age_warn_days) / max(range_size, 1)
            )
        else:
            # Below warning
            age_score = age_days / max(policy.age_warn_days, 1) * 0.5
        
        # ═══════════════════════════════════════════════════════════════
        # Weighted Combination
        # ═══════════════════════════════════════════════════════════════
        
        total_score = (
            policy.weight_drift * drift_score +
            policy.weight_performance * perf_score +
            policy.weight_age * age_score
        )
        
        score_parts = {
            "drift": round(float(drift_score), 6),
            "performance": round(float(perf_score), 6),
            "age": round(float(age_score), 6)
        }
        
        return float(total_score), score_parts
    
    def _determine_priority(self, score: float) -> Literal["low", "medium", "high"]:
        """
        Determine priority level from score.
        
        Args:
            score: Computed score (0-1)
        
        Returns:
            Priority level
        """
        if score >= self.policy.priority_high_threshold:
            return "high"
        elif score >= self.policy.priority_medium_threshold:
            return "medium"
        else:
            return "low"
    
    def _detect_triggers(
        self,
        drift_signals: Dict[str, Any],
        perf_signals: Dict[str, Any],
        age_days: int
    ) -> Tuple[List[str], bool]:
        """
        Detect retraining triggers.
        
        Args:
            drift_signals: Drift signals
            perf_signals: Performance signals
            age_days: Model age
        
        Returns:
            Tuple of (trigger_list, has_hard_trigger)
        """
        triggers: List[str] = []
        has_hard = False
        policy = self.policy
        
        # ═══════════════════════════════════════════════════════════════
        # Drift Triggers
        # ═══════════════════════════════════════════════════════════════
        
        drift_pct = drift_signals["pct"]
        
        if drift_pct >= policy.drift_crit_pct:
            triggers.append(f"🚨 CRITICAL_DRIFT ({drift_pct:.1f}%)")
            has_hard = True
        elif drift_pct >= policy.drift_warn_pct:
            triggers.append(f"⚠️ WARNING_DRIFT ({drift_pct:.1f}%)")
        
        # Target drift
        if policy.target_drift_triggers and drift_signals["target_drift"]:
            triggers.append("🚨 TARGET_DRIFT_DETECTED")
            has_hard = True
        
        # ═══════════════════════════════════════════════════════════════
        # Performance Triggers
        # ═══════════════════════════════════════════════════════════════
        
        primary_delta = perf_signals["primary_delta"]
        
        if primary_delta < 0:
            metric_name = perf_signals["primary_metric"]
            triggers.append(
                f"📉 PERFORMANCE_DROP ({metric_name}: {primary_delta:.4f})"
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Age Triggers
        # ═══════════════════════════════════════════════════════════════
        
        if age_days >= policy.age_crit_days:
            triggers.append(f"🚨 CRITICAL_AGE ({age_days}d)")
            has_hard = True
        elif age_days >= policy.age_warn_days:
            triggers.append(f"⚠️ WARNING_AGE ({age_days}d)")
        
        return triggers, has_hard
    
    # ───────────────────────────────────────────────────────────────────
    # Policy Enforcement
    # ───────────────────────────────────────────────────────────────────
    
    def _check_cooldown(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check cooldown period constraint.
        
        Returns:
            Tuple of (passed, info)
        """
        cooldown_days = self.policy.cooldown_days
        
        if cooldown_days <= 0:
            return True, {
                "required_days": 0,
                "last_retrain": None,
                "passed": True
            }
        
        history = self._read_audit_log()
        
        if history.empty:
            return True, {
                "required_days": cooldown_days,
                "last_retrain": None,
                "passed": True
            }
        
        # Find last successful retrain
        successful = history[history["status"] == "RETRAIN_OK"]
        
        if successful.empty:
            return True, {
                "required_days": cooldown_days,
                "last_retrain": None,
                "passed": True
            }
        
        last_row = successful.sort_values("timestamp").iloc[-1]
        
        try:
            last_ts = pd.to_datetime(last_row["timestamp"], utc=True)
            now = pd.Timestamp.now(tz='UTC')
            days_since = (now - last_ts).days
            
            passed = days_since >= cooldown_days
            
            return passed, {
                "required_days": cooldown_days,
                "last_retrain": last_ts.isoformat(),
                "days_since": days_since,
                "passed": passed
            }
        
        except Exception as e:
            self._log.debug(f"Cooldown check failed: {e}")
            return True, {
                "required_days": cooldown_days,
                "last_retrain": None,
                "passed": True
            }
    
    def _check_weekly_limit(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check weekly retraining limit.
        
        Returns:
            Tuple of (passed, info)
        """
        limit = self.policy.max_retrains_per_week
        
        if limit <= 0:
            return True, {
                "limit": 0,
                "count_last_7d": 0,
                "passed": True
            }
        
        history = self._read_audit_log()
        
        if history.empty:
            return True, {
                "limit": limit,
                "count_last_7d": 0,
                "passed": True
            }
        
        try:
            now = pd.Timestamp.now(tz='UTC')
            week_ago = now - pd.Timedelta(days=7)
            
            # Filter to last 7 days
            history['timestamp'] = pd.to_datetime(history['timestamp'], utc=True)
            recent = history[history['timestamp'] >= week_ago]
            
            # Count successful retrains
            count = int((recent['status'] == 'RETRAIN_OK').sum())
            
            passed = count < limit
            
            return passed, {
                "limit": limit,
                "count_last_7d": count,
                "remaining": max(0, limit - count),
                "passed": passed
            }
        
        except Exception as e:
            self._log.debug(f"Weekly limit check failed: {e}")
            return True, {
                "limit": limit,
                "count_last_7d": 0,
                "passed": True
            }
    
    def _check_volume_gate(self, new_samples: Optional[int]) -> bool:
        """
        Check if sufficient new samples are available.
        
        Args:
            new_samples: Number of new samples
        
        Returns:
            True if gate passed
        """
        if new_samples is None:
            return True  # Unknown - allow
        
        return new_samples >= self.policy.min_new_samples
    
    # ───────────────────────────────────────────────────────────────────
    # Decision Making
    # ───────────────────────────────────────────────────────────────────
    
    def _make_decision(
        self,
        score: float,
        has_hard_trigger: bool,
        policy_gates_passed: bool,
        force: bool
    ) -> bool:
        """
        Make final retraining decision.
        
        Args:
            score: Computed score
            has_hard_trigger: Hard trigger detected
            policy_gates_passed: All policy gates passed
            force: Force flag
        
        Returns:
            True if should retrain
        """
        # Force overrides everything
        if force:
            return True
        
        # Hard trigger + gates = retrain
        if has_hard_trigger and policy_gates_passed:
            return True
        
        # Score threshold + gates = retrain
        if score >= self.policy.priority_medium_threshold and policy_gates_passed:
            return True
        
        return False
    
    # ───────────────────────────────────────────────────────────────────
    # Schedule Generation
    # ───────────────────────────────────────────────────────────────────
    
    def _generate_schedule(self) -> Dict[str, Any]:
        """
        Generate recommended retraining schedule.
        
        Returns:
            Schedule dictionary with multiple formats
        """
        now_local = self._get_local_time()
        
        hour = self.policy.preferred_hour
        minute = self.policy.preferred_minute
        days_of_week = self.policy.days_of_week
        
        # ═══════════════════════════════════════════════════════════════
        # Calculate Next Window
        # ═══════════════════════════════════════════════════════════════
        
        next_time = now_local.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0
        )
        
        # If time has passed today, move to tomorrow
        if next_time <= now_local:
            next_time += timedelta(days=1)
        
        # Apply day-of-week restrictions
        if days_of_week:
            allowed_days = set(days_of_week)
            
            while next_time.weekday() not in allowed_days:
                next_time += timedelta(days=1)
        
        # ═══════════════════════════════════════════════════════════════
        # Generate Cron Expression
        # ═══════════════════════════════════════════════════════════════
        
        cron_dow = self._generate_cron_dow(days_of_week)
        cron = f"{minute} {hour} * * {cron_dow}"
        
        # ═══════════════════════════════════════════════════════════════
        # Generate iCal VEVENT
        # ═══════════════════════════════════════════════════════════════
        
        vevent = self._generate_ical_vevent(next_time)
        
        # ═══════════════════════════════════════════════════════════════
        # Assemble Schedule
        # ═══════════════════════════════════════════════════════════════
        
        return {
            "next_time_local_iso": next_time.isoformat(),
            "next_time_utc": next_time.astimezone(timezone.utc).isoformat(),
            "cron": cron,
            "window": {
                "hour": hour,
                "minute": minute,
                "days_of_week": days_of_week or "daily",
                "timezone": self.policy.timezone
            },
            "ical_vevent": vevent
        }
    
    def _get_local_time(self) -> datetime:
        """Get current time in configured timezone."""
        tz_name = self.policy.timezone
        
        if HAS_ZONEINFO:
            try:
                tz = ZoneInfo(tz_name)
                return datetime.now(tz)
            except Exception as e:
                self._log.warning(f"Timezone '{tz_name}' not available: {e}")
        
        # Fallback to UTC
        return datetime.now(timezone.utc)
    
    def _generate_cron_dow(self, days_of_week: Optional[List[int]]) -> str:
        """
        Generate cron day-of-week expression.
        
        Args:
            days_of_week: List of days (0=Mon, 6=Sun) or None
        
        Returns:
            Cron DOW string
        """
        if not days_of_week:
            return "*"
        
        # Convert Python weekday (0=Mon) to cron (0=Sun, 1=Mon)
        cron_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
        cron_days = sorted([cron_map[d] for d in days_of_week if d in cron_map])
        
        return ",".join(str(d) for d in cron_days)
    
    def _generate_ical_vevent(self, dt: datetime) -> str:
        """
        Generate iCal VEVENT format.
        
        Args:
            dt: Datetime for event
        
        Returns:
            iCal VEVENT string
        """
        dtstart = dt.strftime("%Y%m%dT%H%M%S")
        tzid = self.policy.timezone if HAS_ZONEINFO else "UTC"
        
        return (
            "BEGIN:VEVENT\n"
            f"DTSTART;TZID={tzid}:{dtstart}\n"
            f"SUMMARY:Model Retraining\n"
            f"DESCRIPTION:Recommended retraining window by {self.name}\n"
            f"LOCATION:Production Environment\n"
            "END:VEVENT"
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Immediate Retraining Execution
    # ───────────────────────────────────────────────────────────────────
    
    def _execute_retraining(
        self,
        orchestrator: Any,
        train_data: pd.DataFrame,
        target_column: str,
        problem_type: str,
        orchestrator_kwargs: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Execute immediate retraining via orchestrator.
        
        Args:
            orchestrator: ML orchestrator instance
            train_data: Training data
            target_column: Target column
            problem_type: Problem type
            orchestrator_kwargs: Additional kwargs
        
        Returns:
            Tuple of (success, result_data)
        """
        if not hasattr(orchestrator, "execute"):
            raise ValueError("Orchestrator must have 'execute' method")
        
        try:
            result = orchestrator.execute(
                data=train_data,
                target_column=target_column,
                problem_type=problem_type,
                **orchestrator_kwargs
            )
            
            # Check success
            success = False
            if hasattr(result, "is_success"):
                success = result.is_success()
            elif isinstance(result, dict):
                success = result.get("success", False)
            
            # Extract data
            data = None
            if hasattr(result, "data"):
                data = result.data
            elif isinstance(result, dict):
                data = result
            
            return success, data
        
        except Exception as e:
            self._log.error(f"Retraining execution failed: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    # ───────────────────────────────────────────────────────────────────
    # Decision Reasoning
    # ───────────────────────────────────────────────────────────────────
    
    def _generate_reasoning(
        self,
        should_retrain: bool,
        score: float,
        priority: str,
        score_parts: Dict[str, float],
        triggers: List[str],
        cooldown_ok: bool,
        weekly_ok: bool,
        volume_ok: bool,
        force: bool
    ) -> str:
        """
        Generate human-readable decision reasoning.
        
        Args:
            should_retrain: Decision result
            score: Computed score
            priority: Priority level
            score_parts: Score components
            triggers: Detected triggers
            cooldown_ok: Cooldown gate
            weekly_ok: Weekly limit gate
            volume_ok: Volume gate
            force: Force flag
        
        Returns:
            Reasoning string
        """
        parts = []
        
        # Score breakdown
        parts.append(
            f"Score: {score:.3f} (drift={score_parts['drift']:.2f}, "
            f"perf={score_parts['performance']:.2f}, "
            f"age={score_parts['age']:.2f})"
        )
        
        # Priority
        parts.append(f"Priority: {priority.upper()}")
        
        # Triggers
        if triggers:
            parts.append(f"Triggers: {', '.join(triggers)}")
        else:
            parts.append("Triggers: None")
        
        # Gates
        gates = []
        if not cooldown_ok:
            gates.append("cooldown_blocked")
        if not weekly_ok:
            gates.append("weekly_limit_reached")
        if not volume_ok:
            gates.append("insufficient_volume")
        
        if gates:
            parts.append(f"Gates: BLOCKED ({', '.join(gates)})")
        else:
            parts.append("Gates: PASSED")
        
        # Force
        if force:
            parts.append("FORCED retraining")
        
        # Decision
        decision_text = "RETRAIN RECOMMENDED" if should_retrain else "NO RETRAINING NEEDED"
        parts.append(f"Decision: {decision_text}")
        
        return " | ".join(parts)
    
    # ───────────────────────────────────────────────────────────────────
    # Audit Logging
    # ───────────────────────────────────────────────────────────────────
    
    def _append_audit_log(self, record: Dict[str, Any]) -> None:
        """
        Append record to audit log.
        
        Args:
            record: Record to append
        """
        try:
            df = pd.DataFrame([record])
            
            header_needed = not self.log_path.exists()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(
                self.log_path,
                mode='a',
                header=header_needed,
                index=False,
                encoding='utf-8'
            )
        
        except Exception as e:
            self._log.warning(f"Audit log append failed: {e}")
    
    def _read_audit_log(self) -> pd.DataFrame:
        """
        Read audit log history.
        
        Returns:
            DataFrame with audit log
        """
        if not self.log_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.log_path, encoding='utf-8')
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            return df
        
        except Exception as e:
            self._log.warning(f"Audit log read failed: {e}")
            return pd.DataFrame()
    
    # ───────────────────────────────────────────────────────────────────
    # Public API Methods
    # ───────────────────────────────────────────────────────────────────
    
    def get_history(
        self,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get retraining history.
        
        Args:
            limit: Maximum number of records
            since: Only records after this time
        
        Returns:
            DataFrame with history
        """
        df = self._read_audit_log()
        
        if df.empty:
            return df
        
        if since:
            df = df[df['timestamp'] >= since]
        
        df = df.sort_values('timestamp', ascending=False)
        
        if limit:
            df = df.head(limit)
        
        return df.reset_index(drop=True)
    
    def clear_history(self, confirm: bool = False) -> bool:
        """
        Clear audit log history.
        
        Args:
            confirm: Confirmation flag
        
        Returns:
            True if cleared
        """
        if not confirm:
            self._log.warning("clear_history() requires confirm=True")
            return False
        
        try:
            if self.log_path.exists():
                self.log_path.unlink()
            
            self._log.warning("⚠️ Audit log cleared")
            return True
        
        except Exception as e:
            self._log.error(f"Failed to clear history: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def schedule_retraining(
    problem_type: Literal["classification", "regression"],
    drift_report: Optional[Dict[str, Any]] = None,
    performance_data: Optional[Dict[str, Any]] = None,
    policy: Optional[RetrainPolicy] = None,
    **kwargs
) -> AgentResult:
    """
    🚀 **Convenience Function: Schedule Retraining**
    
    High-level API for retraining decisions.
    
    Args:
        problem_type: 'classification' or 'regression'
        drift_report: DriftDetector output
        performance_data: PerformanceTracker output
        policy: Optional custom policy
        **kwargs: Additional parameters
    
    Returns:
        AgentResult with decision and schedule
    
    Examples:
```python
        from agents.monitoring import schedule_retraining
        
        # Basic decision
        result = schedule_retraining(
            problem_type='classification',
            drift_report=drift_data,
            performance_data=perf_data,
            model_path='model.pkl',
            new_samples=15000
        )
        
        # Check decision
        if result.data['decision']['should_retrain']:
            print("Retraining recommended!")
            print(result.data['decision']['reasoning'])
            print(f"Next window: {result.data['schedule']['next_time_local_iso']}")
        
        # Custom policy
        policy = RetrainPolicy.create_aggressive()
        result = schedule_retraining(
            problem_type='classification',
            drift_report=drift_data,
            policy=policy
        )
```
    """
    scheduler = RetrainingScheduler(policy)
    return scheduler.execute(
        problem_type=problem_type,
        drift_report=drift_report,
        performance_data=performance_data,
        **kwargs
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _module_init():
    """Initialize module on import."""
    logger.info(f"✓ RetrainingScheduler v{__version__} loaded")

_module_init()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{'='*80}")
    print(f"RetrainingScheduler v{__version__}")
    print(f"{'='*80}")
    
    # Generate synthetic signals
    print("\n✓ Testing retraining scheduler...")
    
    # Mock drift report (high drift)
    drift_report = {
        "data_drift": {
            "drift_score": 35.0,
            "drifted_features": ["feature_1", "feature_2", "feature_3"],
            "n_drifted": 3
        },
        "target_drift": {
            "is_drift": True
        }
    }
    
    # Mock performance data (degradation)
    performance_data = {
        "comparison": {
            "metrics": {
                "accuracy": {
                    "current": 0.82,
                    "baseline": 0.88,
                    "delta": -0.06
                }
            }
        }
    }
    
    scheduler = RetrainingScheduler()
    
    result = scheduler.execute(
        problem_type='classification',
        drift_report=drift_report,
        performance_data=performance_data,
        model_path='model.pkl',
        last_train_ts='2024-09-15T00:00:00Z',
        new_samples=10000
    )
    
    if result.is_success():
        print(f"\n✓ Scheduling completed")
        
        decision = result.data['decision']
        schedule = result.data['schedule']
        
        print(f"\nDecision:")
        print(f"  Should Retrain: {decision['should_retrain']}")
        print(f"  Priority: {decision['priority'].upper()}")
        print(f"  Score: {decision['score']:.3f}")
        
        print(f"\nTriggers:")
        for trigger in decision['triggers']:
            print(f"  {trigger}")
        
        print(f"\nReasoning:")
        print(f"  {decision['reasoning']}")
        
        print(f"\nSchedule:")
        print(f"  Next Window: {schedule['next_time_local_iso']}")
        print(f"  Cron: {schedule['cron']}")
        
        print(f"\nGates:")
        gates = result.data['gates']
        print(f"  Cooldown: {'✓' if gates['cooldown']['passed'] else '✗'}")
        print(f"  Weekly Limit: {'✓' if gates['weekly_limit']['passed'] else '✗'}")
        print(f"  Volume: {'✓' if gates['volume_ok'] else '✗'}")
    
    else:
        print(f"\n✗ Scheduling failed:")
        for error in result.errors:
            print(f"  - {error}")
    
    print(f"\n{'='*80}")
    print("USAGE EXAMPLES:")
    print(f"{'='*80}")
    print("""
from agents.monitoring import RetrainingScheduler, RetrainPolicy

# Basic usage
scheduler = RetrainingScheduler()

result = scheduler.execute(
    problem_type='classification',
    drift_report=drift_detector_result.data,
    performance_data=tracker_result.data,
    model_path='models/my_model.pkl',
    new_samples=15000
)

# Check decision
if result.data['decision']['should_retrain']:
    priority = result.data['decision']['priority']
    schedule = result.data['schedule']
    
    print(f"Retraining recommended: {priority} priority")
    print(f"Next window: {schedule['next_time_local_iso']}")

# Custom policy
policy = RetrainPolicy(
    drift_crit_pct=25.0,
    cooldown_days=5,
    max_retrains_per_week=3
)

scheduler = RetrainingScheduler(policy)

# Convenience function
from agents.monitoring import schedule_retraining

result = schedule_retraining(
    problem_type='classification',
    drift_report=drift_data,
    performance_data=perf_data
)
    """)