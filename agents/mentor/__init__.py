# agents/mentor/__init__.py
"""
DataGenius PRO — Mentor package (lazy exports)

Eksporty (leniwe):
- MentorOrchestrator, MentorConfig
- MENTOR_SYSTEM_PROMPT, EDA_EXPLANATION_TEMPLATE, ML_RESULTS_TEMPLATE, RECOMMENDATION_TEMPLATE

Użycie:
    from agents.mentor import MentorOrchestrator, MENTOR_SYSTEM_PROMPT
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple

# Dostosuj ścieżki modułów jeśli Twoje pliki mają inne nazwy.
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # główny agent
    "MentorOrchestrator": ("agents.mentor.orchestrator", "MentorOrchestrator"),
    "MentorConfig": ("agents.mentor.orchestrator", "MentorConfig"),

    # szablony promptów
    "MENTOR_SYSTEM_PROMPT": ("agents.mentor.prompt_templates", "MENTOR_SYSTEM_PROMPT"),
    "EDA_EXPLANATION_TEMPLATE": ("agents.mentor.prompt_templates", "EDA_EXPLANATION_TEMPLATE"),
    "ML_RESULTS_TEMPLATE": ("agents.mentor.prompt_templates", "ML_RESULTS_TEMPLATE"),
    "RECOMMENDATION_TEMPLATE": ("agents.mentor.prompt_templates", "RECOMMENDATION_TEMPLATE"),
}

__all__ = tuple(_LAZY_EXPORTS.keys())

def __getattr__(name: str):
    """Leniwe rozwiązywanie symboli + cache w module globals()."""
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        module: ModuleType = import_module(mod_name)
        obj = getattr(module, symbol)
        globals()[name] = obj  # cache
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
