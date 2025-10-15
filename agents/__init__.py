# agents/__init__.py
"""
DataGenius PRO — agents package
Leniwe eksporty głównych agentów, tak by import pakietu był lekki,
a cięższe zależności (np. PyCaret) ładowały się dopiero przy użyciu.

Przykład:
    from agents import MentorOrchestrator, MLOrchestrator
    mentor = MentorOrchestrator()
    ml = MLOrchestrator()
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from types import ModuleType
from typing import Dict

# --- wersja pakietu (bez twardej zależności od setuptools) ---
try:
    __version__ = _pkg_version("datagenius-pro")  # jeśli masz nazwę pakietu w pyproject
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

# --- mapa leniwych eksportów: nazwa → (moduł, symbol) ---
_LAZY_EXPORTS: Dict[str, tuple[str, str]] = {
    # Orchestratory
    "MentorOrchestrator": ("agents.mentor.orchestrator", "MentorOrchestrator"),
    "MentorConfig": ("agents.mentor.orchestrator", "MentorConfig"),
    "MLOrchestrator": ("agents.ml.orchestrator", "MLOrchestrator"),
    "MLConfig": ("agents.ml.orchestrator", "MLConfig"),

    # ML core (użyteczne w testach / ręcznie)
    "ModelTrainer": ("agents.ml.model_trainer", "ModelTrainer"),
    "TrainerConfig": ("agents.ml.model_trainer", "TrainerConfig"),
    "ModelSelector": ("agents.ml.model_trainer", "ModelSelector"),

    # Monitoring & ops
    "DriftDetector": ("agents.monitoring.drift_detector", "DriftDetector"),
    "DriftConfig": ("agents.monitoring.drift_detector", "DriftConfig"),
    "PerformanceTracker": ("agents.monitoring.performance_tracker", "PerformanceTracker"),
    "PerformanceConfig": ("agents.monitoring.performance_tracker", "PerformanceConfig"),
    "RetrainingScheduler": ("agents.monitoring.retraining_scheduler", "RetrainingScheduler"),
    "RetrainPolicy": ("agents.monitoring.retraining_scheduler", "RetrainPolicy"),

    # Prep / feature pipeline
    "MissingDataHandler": ("agents.features.missing_data_handler", "MissingDataHandler"),
    "MissingHandlerConfig": ("agents.features.missing_data_handler", "MissingHandlerConfig"),
    "FeatureEngineer": ("agents.features.feature_engineer", "FeatureEngineer"),
    "FeatureConfig": ("agents.features.feature_engineer", "FeatureConfig"),
    "EncoderSelector": ("agents.features.encoder_selector", "EncoderSelector"),
    "EncoderPolicy": ("agents.features.encoder_selector", "EncoderPolicy"),
    "ScalerSelector": ("agents.features.scaler_selector", "ScalerSelector"),
    "ScalerSelectorConfig": ("agents.features.scaler_selector", "ScalerSelectorConfig"),
}

__all__ = tuple(_LAZY_EXPORTS.keys()) + ("__version__",)

# --- mechanizm leniwych atrybutów ---
def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        module: ModuleType = import_module(mod_name)
        obj = getattr(module, symbol)
        globals()[name] = obj  # cache w module, by kolejne odczyty były szybkie
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
