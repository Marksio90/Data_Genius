# agents/__init__.py
"""
DataGenius PRO — agents package (Enterprise / KOSMOS)
Leniwe eksporty głównych agentów, tak by import pakietu był lekki,
a cięższe zależności (np. PyCaret) ładowały się dopiero przy użyciu.

Przykład:
    from agents import MentorOrchestrator, MLOrchestrator
    mentor = MentorOrchestrator()
    ml = MLOrchestrator()
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from types import ModuleType
from typing import Any, Dict, Final, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Wersja pakietu (bez twardej zależności od setuptools; bez faila w dev)
# ──────────────────────────────────────────────────────────────────────────────
try:
    __version__ = _pkg_version("datagenius-pro")  # zmień, jeśli inna nazwa w pyproject
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

# ──────────────────────────────────────────────────────────────────────────────
# Spec leniwego eksportu
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _LazySpec:
    module: str
    symbol: str

# Mapa leniwych eksportów: publiczna_nazwa -> (_LazySpec)
_LAZY_EXPORTS: Dict[str, _LazySpec] = {
    # Orchestratory
    "MentorOrchestrator": _LazySpec("agents.mentor.orchestrator", "MentorOrchestrator"),
    "MentorConfig":       _LazySpec("agents.mentor.orchestrator", "MentorConfig"),
    "MLOrchestrator":     _LazySpec("agents.ml.orchestrator", "MLOrchestrator"),
    "MLConfig":           _LazySpec("agents.ml.orchestrator", "MLConfig"),

    # ML core (użyteczne w testach / ręcznie)
    "ModelTrainer":  _LazySpec("agents.ml.model_trainer", "ModelTrainer"),
    "TrainerConfig": _LazySpec("agents.ml.model_trainer", "TrainerConfig"),
    "ModelSelector": _LazySpec("agents.ml.model_trainer", "ModelSelector"),

    # Monitoring & ops
    "DriftDetector":       _LazySpec("agents.monitoring.drift_detector", "DriftDetector"),
    "DriftConfig":         _LazySpec("agents.monitoring.drift_detector", "DriftConfig"),
    "PerformanceTracker":  _LazySpec("agents.monitoring.performance_tracker", "PerformanceTracker"),
    "PerformanceConfig":   _LazySpec("agents.monitoring.performance_tracker", "PerformanceConfig"),
    "RetrainingScheduler": _LazySpec("agents.monitoring.retraining_scheduler", "RetrainingScheduler"),
    "RetrainPolicy":       _LazySpec("agents.monitoring.retraining_scheduler", "RetrainPolicy"),

    # Prep / feature pipeline
    "MissingDataHandler":   _LazySpec("agents.features.missing_data_handler", "MissingDataHandler"),
    "MissingHandlerConfig": _LazySpec("agents.features.missing_data_handler", "MissingHandlerConfig"),
    "FeatureEngineer":      _LazySpec("agents.features.feature_engineer", "FeatureEngineer"),
    "FeatureConfig":        _LazySpec("agents.features.feature_engineer", "FeatureConfig"),
    "EncoderSelector":      _LazySpec("agents.features.encoder_selector", "EncoderSelector"),
    "EncoderPolicy":        _LazySpec("agents.features.encoder_selector", "EncoderPolicy"),
    "ScalerSelector":       _LazySpec("agents.features.scaler_selector", "ScalerSelector"),
    "ScalerSelectorConfig": _LazySpec("agents.features.scaler_selector", "ScalerSelectorConfig"),
}

# Publiczny interfejs modułu
__all__: Final[Tuple[str, ...]] = tuple(_LAZY_EXPORTS.keys()) + ("__version__",)

# ──────────────────────────────────────────────────────────────────────────────
# Leniwe rozwiązywanie + mikro-cache w module
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=len(_LAZY_EXPORTS) or None)
def _resolve(name: str) -> Any:
    """
    Importuje moduł i pobiera symbol wskazany w _LAZY_EXPORTS.
    Rezultat jest cache'owany (LRU) — zapobiega wielokrotnemu importowi
    i jest bezpieczny na wyścigi przy równoległych importach.
    """
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    try:
        module: ModuleType = importlib.import_module(spec.module)
    except Exception as e:
        # Czytelny komunikat, gdy moduł nie istnieje / ma błąd inicjalizacji
        raise ImportError(
            f"Failed to import module '{spec.module}' required for attribute '{name}'."
        ) from e

    try:
        obj = getattr(module, spec.symbol)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{spec.module}' does not define expected attribute '{spec.symbol}' "
            f"(needed for '{name}')."
        ) from e

    return obj


def __getattr__(name: str) -> Any:  # PEP 562
    """
    Leniwe rozwiązywanie symboli + wstrzyknięcie do globals() po pierwszym użyciu.
    """
    obj = _resolve(name)
    globals()[name] = obj  # mikro-cache w module (przyspiesza kolejne odczyty)
    return obj


def __dir__() -> list[str]:
    """
    Zwraca standardowe atrybuty + nazwy dostępne przez leniwe eksporty (dla autocompletion).
    """
    return sorted(list(globals().keys()) + list(__all__))
