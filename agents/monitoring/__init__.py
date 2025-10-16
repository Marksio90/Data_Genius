# agents/monitoring/__init__.py
"""
DataGenius PRO — Monitoring package (lazy exports, Enterprise/KOSMOS)

Eksporty (leniwe, stabilne):
- DriftDetector / DriftConfig
- PerformanceTracker / PerformanceConfig
- RetrainingScheduler / RetrainPolicy

Użycie:
    from agents.monitoring import DriftDetector, PerformanceTracker, RetrainingScheduler
    dd = DriftDetector(); pt = PerformanceTracker(); rs = RetrainingScheduler()
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any, Dict, Final, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA LENIWEGO EKSPORTU
# Klucz = publiczna nazwa symbolu eksportowanego przez `agents.monitoring`.
# Wartość = (pełna_ścieżka_modułu, nazwa_symbolu_w_modułowym_namespace)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _LazySpec:
    module: str
    symbol: str

_LAZY_EXPORTS: Dict[str, _LazySpec] = {
    # drift
    "DriftDetector":   _LazySpec("agents.monitoring.drift_detector", "DriftDetector"),
    "DriftConfig":     _LazySpec("agents.monitoring.drift_detector", "DriftConfig"),

    # performance tracking
    "PerformanceTracker": _LazySpec("agents.monitoring.performance_tracker", "PerformanceTracker"),
    "PerformanceConfig":  _LazySpec("agents.monitoring.performance_tracker", "PerformanceConfig"),

    # retraining scheduler
    "RetrainingScheduler": _LazySpec("agents.monitoring.retraining_scheduler", "RetrainingScheduler"),
    "RetrainPolicy":       _LazySpec("agents.monitoring.retraining_scheduler", "RetrainPolicy"),
}

# Publiczny interfejs modułu
__all__: Final[Tuple[str, ...]] = tuple(_LAZY_EXPORTS.keys())

# ──────────────────────────────────────────────────────────────────────────────
# IMPLEMENTACJA LENIWEGO ROZWIĄZYWANIA + CACHE
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=len(_LazySpec.__annotations__) or None)  # defensywny maxsize; i tak <= liczby eksportów
def _resolve(name: str) -> Any:
    """
    Importuje moduł i pobiera symbol wskazany w _LAZY_EXPORTS.
    Rezultat jest cache'owany (LRU) — zapobiega wielokrotnemu importowi
    i jest bezpieczny na wyścigi przy równoległych importach.
    """
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        # Konsekwentna semantyka AttributeError dla nieistniejących symboli.
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    try:
        module: ModuleType = importlib.import_module(spec.module)
    except Exception as e:
        # Czytelniejszy komunikat, gdy moduł nie istnieje / ma błąd inicjalizacji
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
    # Mikro-cache w module przyspiesza dostęp w kolejnych odczytach
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    """
    Zwraca standardowe atrybuty + nazwy dostępne przez leniwe eksporty (dla autocompletion).
    """
    return sorted(list(globals().keys()) + list(__all__))
