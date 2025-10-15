# agents/monitoring/__init__.py
"""
DataGenius PRO — monitoring package (lazy exports)

Eksporty:
- DriftDetector / DriftConfig
- PerformanceTracker / PerformanceConfig
- RetrainingScheduler / RetrainPolicy

Użycie:
    from agents.monitoring import DriftDetector, PerformanceTracker, RetrainingScheduler
    dd = DriftDetector(); pt = PerformanceTracker(); rs = RetrainingScheduler()
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple

# mapowanie: symbol -> (moduł, nazwa_obiektu)
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # drift
    "DriftDetector": ("agents.monitoring.drift_detector", "DriftDetector"),
    "DriftConfig": ("agents.monitoring.drift_detector", "DriftConfig"),

    # performance tracking
    "PerformanceTracker": ("agents.monitoring.performance_tracker", "PerformanceTracker"),
    "PerformanceConfig": ("agents.monitoring.performance_tracker", "PerformanceConfig"),

    # retraining scheduler
    "RetrainingScheduler": ("agents.monitoring.retraining_scheduler", "RetrainingScheduler"),
    "RetrainPolicy": ("agents.monitoring.retraining_scheduler", "RetrainPolicy"),
}

__all__ = tuple(_LAZY_EXPORTS.keys())

def __getattr__(name: str):
    """Leniwe rozwiązywanie symboli i cache w module globals()."""
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        module: ModuleType = import_module(mod_name)
        obj = getattr(module, symbol)
        globals()[name] = obj  # cache na przyszłość
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
