# agents/monitoring/__init__.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Monitoring Package               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ENTERPRISE-GRADE ML MONITORING & RETRAINING ORCHESTRATION             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Lazy exports of core monitoring components:                              ║
║  • DriftDetector / DriftConfig                                            ║
║  • PerformanceTracker / PerformanceConfig                                 ║
║  • RetrainingScheduler / RetrainPolicy                                    ║
║  • Convenience functions (detect_drift, track_performance, etc.)          ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌────────────────────────────────────────────────────────────┐
    │                  Monitoring Package                        │
    ├────────────────────────────────────────────────────────────┤
    │  1. Drift Detection (statistical + ML)                     │
    │  2. Performance Tracking (metrics + trends)                │
    │  3. Retraining Orchestration (intelligent decisions)       │
    └────────────────────────────────────────────────────────────┘

Usage:
```python
    # Basic imports
    from agents.monitoring import DriftDetector, PerformanceTracker
    
    detector = DriftDetector()
    tracker = PerformanceTracker()
    
    # Convenience functions
    from agents.monitoring import detect_drift, track_performance
    
    drift_result = detect_drift(train_df, prod_df)
    perf_result = track_performance('classification', y_true, y_pred)
    
    # Full workflow
    from agents.monitoring import (
        DriftDetector, DriftConfig,
        PerformanceTracker, PerformanceConfig,
        RetrainingScheduler, RetrainPolicy
    )
```

Examples:
    # Simple drift detection
    >>> from agents.monitoring import detect_drift
    >>> result = detect_drift(train_df, prod_df, target_column='target')
    >>> print(f"Drift: {result.data['data_drift']['drift_score']:.1f}%")
    
    # Performance tracking
    >>> from agents.monitoring import track_performance
    >>> result = track_performance('classification', y_true, y_pred, y_proba)
    >>> print(f"Accuracy: {result.data['metrics']['accuracy']:.4f}")
    
    # Retraining decision
    >>> from agents.monitoring import schedule_retraining
    >>> result = schedule_retraining(
    ...     'classification',
    ...     drift_report=drift_data,
    ...     performance_data=perf_data
    ... )
    >>> print(f"Retrain: {result.data['decision']['should_retrain']}")
    
    # Full pipeline
    >>> from agents.monitoring import (
    ...     DriftDetector, PerformanceTracker, RetrainingScheduler
    ... )
    >>> detector = DriftDetector()
    >>> tracker = PerformanceTracker()
    >>> scheduler = RetrainingScheduler()
    >>> 
    >>> drift = detector.execute(train_df, prod_df)
    >>> perf = tracker.execute('classification', y_true, y_pred)
    >>> decision = scheduler.execute(
    ...     'classification',
    ...     drift_report=drift.data,
    ...     performance_data=perf.data
    ... )
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "6.0.0-enterprise"
__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# Lazy Export Definitions
# ═══════════════════════════════════════════════════════════════════════════

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # ─────────────────────────────────────────────────────────────────────
    # Drift Detection
    # ─────────────────────────────────────────────────────────────────────
    "DriftDetector": (
        "agents.monitoring.drift_detector",
        "DriftDetector"
    ),
    "DriftConfig": (
        "agents.monitoring.drift_detector",
        "DriftConfig"
    ),
    "detect_drift": (
        "agents.monitoring.drift_detector",
        "detect_drift"
    ),
    "quick_drift_check": (
        "agents.monitoring.drift_detector",
        "quick_drift_check"
    ),
    
    # ─────────────────────────────────────────────────────────────────────
    # Performance Tracking
    # ─────────────────────────────────────────────────────────────────────
    "PerformanceTracker": (
        "agents.monitoring.performance_tracker",
        "PerformanceTracker"
    ),
    "PerformanceConfig": (
        "agents.monitoring.performance_tracker",
        "PerformanceConfig"
    ),
    "track_performance": (
        "agents.monitoring.performance_tracker",
        "track_performance"
    ),
    "quick_performance_check": (
        "agents.monitoring.performance_tracker",
        "quick_performance_check"
    ),
    
    # ─────────────────────────────────────────────────────────────────────
    # Retraining Scheduling
    # ─────────────────────────────────────────────────────────────────────
    "RetrainingScheduler": (
        "agents.monitoring.retraining_scheduler",
        "RetrainingScheduler"
    ),
    "RetrainPolicy": (
        "agents.monitoring.retraining_scheduler",
        "RetrainPolicy"
    ),
    "schedule_retraining": (
        "agents.monitoring.retraining_scheduler",
        "schedule_retraining"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = tuple(_LAZY_EXPORTS.keys()) + (
    "__version__",
    "get_version_info",
    "list_components"
)


# ═══════════════════════════════════════════════════════════════════════════
# Lazy Loading Implementation
# ═══════════════════════════════════════════════════════════════════════════

def __getattr__(name: str):
    """
    Lazy attribute resolution with caching.
    
    Dynamically imports and caches monitoring components on first access.
    This reduces initial import time and memory footprint.
    
    Args:
        name: Attribute name to resolve
    
    Returns:
        Resolved attribute (class, function, or constant)
    
    Raises:
        AttributeError: If attribute not found in exports
    
    Example:
        >>> from agents.monitoring import DriftDetector  # Triggers lazy load
        >>> detector = DriftDetector()
    """
    if name in _LAZY_EXPORTS:
        module_name, symbol_name = _LAZY_EXPORTS[name]
        
        try:
            # Import module
            module: ModuleType = import_module(module_name)
            
            # Get symbol from module
            obj = getattr(module, symbol_name)
            
            # Cache in globals for future access
            globals()[name] = obj
            
            return obj
        
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import '{symbol_name}' from '{module_name}': {e}"
            ) from e
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'"
    )


def __dir__():
    """
    Return list of available attributes for tab-completion.
    
    Returns:
        Sorted list of all available attributes
    """
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def get_version_info() -> Dict[str, str]:
    """
    Get detailed version information.
    
    Returns:
        Dictionary with version details
    
    Example:
        >>> from agents.monitoring import get_version_info
        >>> info = get_version_info()
        >>> print(info['version'])
        6.0.0-enterprise
    """
    return {
        "version": __version__,
        "package": "agents.monitoring",
        "author": __author__,
        "license": __license__,
        "components": {
            "drift_detector": "6.0.0-enterprise",
            "performance_tracker": "6.0.0-enterprise",
            "retraining_scheduler": "6.0.0-enterprise"
        }
    }


def list_components() -> Dict[str, Dict[str, str]]:
    """
    List all available monitoring components.
    
    Returns:
        Dictionary mapping component names to their details
    
    Example:
        >>> from agents.monitoring import list_components
        >>> components = list_components()
        >>> for name, info in components.items():
        ...     print(f"{name}: {info['description']}")
    """
    return {
        "DriftDetector": {
            "type": "class",
            "module": "drift_detector",
            "description": "Statistical drift detection for ML features",
            "main_methods": ["execute", "validate_input"],
            "config": "DriftConfig"
        },
        "PerformanceTracker": {
            "type": "class",
            "module": "performance_tracker",
            "description": "Model performance monitoring and SLO tracking",
            "main_methods": ["execute", "get_history", "get_latest"],
            "config": "PerformanceConfig"
        },
        "RetrainingScheduler": {
            "type": "class",
            "module": "retraining_scheduler",
            "description": "Intelligent retraining decision and scheduling",
            "main_methods": ["execute", "get_history"],
            "config": "RetrainPolicy"
        },
        "detect_drift": {
            "type": "function",
            "module": "drift_detector",
            "description": "Convenience function for drift detection",
            "signature": "(reference_data, current_data, ...)"
        },
        "track_performance": {
            "type": "function",
            "module": "performance_tracker",
            "description": "Convenience function for performance tracking",
            "signature": "(problem_type, y_true, y_pred, ...)"
        },
        "schedule_retraining": {
            "type": "function",
            "module": "retraining_scheduler",
            "description": "Convenience function for retraining decisions",
            "signature": "(problem_type, drift_report, ...)"
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _init_module():
    """Initialize module on import."""
    try:
        from loguru import logger
        logger.info(f"✓ Monitoring package v{__version__} loaded (lazy)")
    except ImportError:
        import logging
        logging.getLogger(__name__).info(
            f"Monitoring package v{__version__} loaded (lazy)"
        )

_init_module()


# ═══════════════════════════════════════════════════════════════════════════
# Quick Reference Documentation
# ═══════════════════════════════════════════════════════════════════════════

QUICK_START = """
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius Monitoring — Quick Start Guide                               ║
╚════════════════════════════════════════════════════════════════════════════╝

1. DRIFT DETECTION
   ────────────────────────────────────────────────────────────────────────
   from agents.monitoring import detect_drift
   
   result = detect_drift(
       reference_data=train_df,
       current_data=prod_df,
       target_column='target'
   )
   
   drift_score = result.data['data_drift']['drift_score']
   print(f"Drift: {drift_score:.1f}%")

2. PERFORMANCE TRACKING
   ────────────────────────────────────────────────────────────────────────
   from agents.monitoring import track_performance
   
   result = track_performance(
       problem_type='classification',
       y_true=y_test,
       y_pred=predictions,
       y_proba=probabilities,
       model_name='my_model',
       compare_to='best'
   )
   
   metrics = result.data['metrics']
   print(f"Accuracy: {metrics['accuracy']:.4f}")

3. RETRAINING DECISION
   ────────────────────────────────────────────────────────────────────────
   from agents.monitoring import schedule_retraining
   
   result = schedule_retraining(
       problem_type='classification',
       drift_report=drift_result.data,
       performance_data=perf_result.data,
       model_path='model.pkl',
       new_samples=10000
   )
   
   decision = result.data['decision']
   if decision['should_retrain']:
       print(f"Retraining recommended: {decision['priority']} priority")

4. FULL PIPELINE
   ────────────────────────────────────────────────────────────────────────
   from agents.monitoring import (
       DriftDetector, PerformanceTracker, RetrainingScheduler
   )
   
   # Initialize
   detector = DriftDetector()
   tracker = PerformanceTracker()
   scheduler = RetrainingScheduler()
   
   # Execute pipeline
   drift = detector.execute(train_df, prod_df)
   perf = tracker.execute('classification', y_true, y_pred)
   decision = scheduler.execute(
       'classification',
       drift_report=drift.data,
       performance_data=perf.data
   )
   
   # Act on decision
   if decision.data['decision']['should_retrain']:
       print("🔄 Retraining required!")

5. CUSTOM CONFIGURATION
   ────────────────────────────────────────────────────────────────────────
   from agents.monitoring import (
       DriftConfig, PerformanceConfig, RetrainPolicy
   )
   
   # Strict monitoring
   drift_cfg = DriftConfig.create_strict()
   perf_cfg = PerformanceConfig.create_strict()
   policy = RetrainPolicy.create_aggressive()
   
   detector = DriftDetector(drift_cfg)
   tracker = PerformanceTracker(perf_cfg)
   scheduler = RetrainingScheduler(policy)

For more information:
  • Documentation: https://docs.datagenius.ai/monitoring
  • Examples: https://github.com/datagenius/examples/monitoring
  • Support: support@datagenius.ai
"""


def print_quick_start():
    """Print quick start guide."""
    print(QUICK_START)


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{'='*80}")
    print(f"DataGenius Monitoring Package v{__version__}")
    print(f"{'='*80}")
    
    # Show version info
    print("\n✓ Version Information:")
    info = get_version_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Show available components
    print("\n✓ Available Components:")
    components = list_components()
    for name, details in components.items():
        print(f"  • {name}")
        print(f"    Type: {details['type']}")
        print(f"    Description: {details['description']}")
    
    # Show exports
    print(f"\n✓ Lazy Exports ({len(_LAZY_EXPORTS)} items):")
    for name in sorted(_LAZY_EXPORTS.keys()):
        module_name, symbol_name = _LAZY_EXPORTS[name]
        print(f"  • {name:30s} → {module_name}")
    
    # Quick start
    print(f"\n{'='*80}")
    print_quick_start()