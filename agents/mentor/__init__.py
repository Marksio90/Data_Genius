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

import importlib
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Dict, Tuple, Final, Any

# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA LENIWEGO EKSPORTU
# Jeżeli nazwy modułów/plików różnią się w Twoim repozytorium, dostosuj mapowanie.
# Klucz = publiczna nazwa symbolu eksportowanego przez `agents.mentor`.
# Wartość = (pełna_ścieżka_modułu, nazwa_symbolu_w_modułowym_namespace)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _LazySpec:
    module: str
    symbol: str

# Mapa eksportów (zgodna z wcześniejszą wersją; bezpieczna dla mypy/pyright).
_LAZY_EXPORTS: Dict[str, _LazySpec] = {
    # główny agent
    "MentorOrchestrator": _LazySpec("agents.mentor.orchestrator", "MentorOrchestrator"),
    "MentorConfig": _LazySpec("agents.mentor.orchestrator", "MentorConfig"),

    # szablony promptów
    "MENTOR_SYSTEM_PROMPT": _LazySpec("agents.mentor.prompt_templates", "MENTOR_SYSTEM_PROMPT"),
    "EDA_EXPLANATION_TEMPLATE": _LazySpec("agents.mentor.prompt_templates", "EDA_EXPLANATION_TEMPLATE"),
    "ML_RESULTS_TEMPLATE": _LazySpec("agents.mentor.prompt_templates", "ML_RESULTS_TEMPLATE"),
    "RECOMMENDATION_TEMPLATE": _LazySpec("agents.mentor.prompt_templates", "RECOMMENDATION_TEMPLATE"),
}

# Publiczny interfejs modułu
__all__: Final[Tuple[str, ...]] = tuple(_LAZY_EXPORTS.keys())

# ──────────────────────────────────────────────────────────────────────────────
# IMPLEMENTACJA LENIWEGO ROZWIĄZYWANIA
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=len(_LAZY_EXPORTS) or None)
def _resolve(name: str) -> Any:
    """
    Importuje moduł i pobiera symbol wskazany w _LAZY_EXPORTS.
    Rezultat jest cache'owany (LRU) — identyczne jak ręczny cache w globals(),
    ale dodatkowo chroni przed powtarzającym się importem przy wyścigach.
    """
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
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
    globals()[name] = obj  # mikro-cache w module (przyspiesza dostęp po pierwszym imporcie)
    return obj


def __dir__() -> list[str]:
    """
    Zwraca standardowe atrybuty + nazwy dostępne przez leniwe eksporty.
    """
    return sorted(list(globals().keys()) + list(__all__))
