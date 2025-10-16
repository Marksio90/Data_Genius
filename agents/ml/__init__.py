# agents/ml/__init__.py
# === OPIS MODUŁU ===
"""
DataGenius PRO — ML package (lazy exports, Enterprise)

Leniwe, defensywne eksporty publicznego API pakietu ML.
Szybszy start procesu, niższe zużycie RAM, czytelne błędy i stabilny interfejs.

Eksporty:
- ModelSelector
- ModelTrainer, TrainerConfig
- ModelEvaluator
- ModelExplainer
- PyCaretWrapper, PyCaretConfig

Użycie:
    from agents.ml import ModelTrainer, ModelSelector
    trainer = ModelTrainer()

Wersja modułu: 5.0-kosmos-enterprise
"""

from __future__ import annotations

# === NAZWA_SEKCJI === IMPORTY SYSTEMOWE ===
import importlib
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any, Dict, Final, Tuple

__version__: Final[str] = "5.0-kosmos-enterprise"

# === NAZWA_SEKCJI === SPECYFIKACJA LENIWEGO EKSPORTU ===
@dataclass(frozen=True)
class _LazySpec:
    """Opis pojedynczego eksportu: skąd i co pobrać."""
    module: str
    symbol: str

# Mapowanie: publiczny symbol -> _LazySpec(module_path, symbol_name)
_LAZY_EXPORTS: Dict[str, _LazySpec] = {
    # selection
    "ModelSelector": _LazySpec("agents.ml.model_selector", "ModelSelector"),

    # training
    "ModelTrainer":  _LazySpec("agents.ml.model_trainer", "ModelTrainer"),
    "TrainerConfig": _LazySpec("agents.ml.model_trainer", "TrainerConfig"),

    # evaluation
    "ModelEvaluator": _LazySpec("agents.ml.model_evaluator", "ModelEvaluator"),

    # explainability
    "ModelExplainer": _LazySpec("agents.ml.model_explainer", "ModelExplainer"),

    # pycaret wrapper
    "PyCaretWrapper": _LazySpec("agents.ml.pycaret_wrapper", "PyCaretWrapper"),
    "PyCaretConfig":  _LazySpec("agents.ml.pycaret_wrapper", "PyCaretConfig"),
}

# Publiczny interfejs modułu
__all__: Final[Tuple[str, ...]] = tuple(_LAZY_EXPORTS.keys())

# === NAZWA_SEKCJI === ROZWIĄZYWANIE SYMBOLI (LENIWE + CACHE) ===
@lru_cache(maxsize=len(_LAZY_EXPORTS) or None)
def _resolve(name: str) -> Any:
    """
    Importuje moduł i pobiera symbol wskazany w _LAZY_EXPORTS.
    Wynik jest LRU-cache'owany, co redukuje koszty powtarzanych dostępów.

    Raises:
        AttributeError: jeżeli symbol nie jest zdefiniowany w mapie eksportów.
        ImportError:   jeżeli moduł nie może zostać zaimportowany.
        AttributeError: jeżeli w module brakuje oczekiwanego symbolu.
    """
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    try:
        module: ModuleType = importlib.import_module(spec.module)
    except Exception as e:
        # Komunikat z kontekstem — który moduł zawiódł
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

# === NAZWA_SEKCJI === PEP 562: __getattr__ / __dir__ ===
def __getattr__(name: str) -> Any:
    """
    Leniwe rozwiązywanie symboli.
    Po pierwszym odczycie wstrzykujemy obiekt do globals() (mikro-cache).
    """
    obj = _resolve(name)
    globals()[name] = obj
    return obj

def __dir__() -> list[str]:
    """Zwraca standardowe atrybuty + nazwy dostępne przez leniwe eksporty."""
    return sorted(list(globals().keys()) + list(__all__))
