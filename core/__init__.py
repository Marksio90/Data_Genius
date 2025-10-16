# core/__init__.py
"""
DataGenius PRO — core package (PRO++++)
Lekkie, leniwe eksporty najważniejszych prymitywów rdzenia.

Eksporty:
- BaseAgent, PipelineAgent, AgentResult        (z core.base_agent)
- get_llm_client                               (z core.llm_client)

Użycie:
    from core import BaseAgent, AgentResult, get_llm_client
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from types import ModuleType
from typing import Dict, Tuple

# --- wersja pakietu (bez twardej zależności od setuptools) ---
try:
    __version__ = _pkg_version("datagenius-pro")  # dopasuj do nazwy w pyproject.toml
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

# --- mapa leniwych eksportów: publiczny_symbol -> (moduł, symbol_w_modułowym_namespace) ---
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Agenci i kontrakty wyników
    "BaseAgent": ("core.base_agent", "BaseAgent"),
    "PipelineAgent": ("core.base_agent", "PipelineAgent"),
    "AgentResult": ("core.base_agent", "AgentResult"),

    # LLM client (DI/factory)
    "get_llm_client": ("core.llm_client", "get_llm_client"),
}

__all__ = tuple(_LAZY_EXPORTS.keys()) + ("__version__",)

def __getattr__(name: str):
    """Leniwe rozwiązywanie symboli + cache w module globals()."""
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        module: ModuleType = import_module(mod_name)
        obj = getattr(module, symbol)
        globals()[name] = obj  # cache kolejnych odczytów
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
