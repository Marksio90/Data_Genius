# agents/ml/__init__.py
"""
DataGenius PRO — ML package (lazy exports)

Eksporty (leniwe):
- ModelSelector
- ModelTrainer, TrainerConfig
- ModelEvaluator
- ModelExplainer
- PyCaretWrapper, PyCaretConfig

Użycie:
    from agents.ml import ModelTrainer, ModelSelector
    trainer = ModelTrainer()
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple

# Mapowanie: publiczny symbol -> (ścieżka_modułu, nazwa_obiektu)
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # selection
    "ModelSelector": ("agents.ml.model_selector", "ModelSelector"),

    # training
    "ModelTrainer": ("agents.ml.model_trainer", "ModelTrainer"),
    "TrainerConfig": ("agents.ml.model_trainer", "TrainerConfig"),

    # evaluation (jeśli masz osobny moduł)
    "ModelEvaluator": ("agents.ml.model_evaluator", "ModelEvaluator"),

    # explainability (jeśli masz osobny moduł)
    "ModelExplainer": ("agents.ml.model_explainer", "ModelExplainer"),

    # pycaret wrapper
    "PyCaretWrapper": ("agents.ml.pycaret_wrapper", "PyCaretWrapper"),
    "PyCaretConfig": ("agents.ml.pycaret_wrapper", "PyCaretConfig"),
}

__all__ = tuple(_LAZY_EXPORTS.keys())

def __getattr__(name: str):
    """Leniwe rozwiązywanie symboli + cache w module globals()."""
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        module: ModuleType = import_module(mod_name)
        obj = getattr(module, symbol)
        globals()[name] = obj  # cache na przyszłość
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
