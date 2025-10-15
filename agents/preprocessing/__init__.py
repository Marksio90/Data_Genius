# agents/preprocessing/__init__.py
"""
DataGenius PRO — preprocessing package

Leniwe eksporty najważniejszych komponentów:
- MissingDataHandler / MissingHandlerConfig
- FeatureEngineer / FeatureConfig
- EncoderSelector / EncoderPolicy
- ScalerSelector / ScalerSelectorConfig

Przykład:
    from agents.preprocessing import MissingDataHandler, FeatureEngineer
    mdh = MissingDataHandler()
    fe = FeatureEngineer()
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # imputacja
    "MissingDataHandler": ("agents.preprocessing.missing_data_handler", "MissingDataHandler"),
    "MissingHandlerConfig": ("agents.preprocessing.missing_data_handler", "MissingHandlerConfig"),

    # feature engineering
    "FeatureEngineer": ("agents.preprocessing.feature_engineer", "FeatureEngineer"),
    "FeatureConfig": ("agents.preprocessing.feature_engineer", "FeatureConfig"),

    # encodery
    "EncoderSelector": ("agents.preprocessing.encoder_selector", "EncoderSelector"),
    "EncoderPolicy": ("agents.preprocessing.encoder_selector", "EncoderPolicy"),

    # skalery
    "ScalerSelector": ("agents.preprocessing.scaler_selector", "ScalerSelector"),
    "ScalerSelectorConfig": ("agents.preprocessing.scaler_selector", "ScalerSelectorConfig"),
}

__all__ = tuple(_LAZY_EXPORTS.keys())

def __getattr__(name: str):
    """Leniwe rozwiązywanie symboli, cache’owane w globals()."""
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        module: ModuleType = import_module(mod_name)
        obj = getattr(module, symbol)
        globals()[name] = obj  # cache
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
