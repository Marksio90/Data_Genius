# config/__init__.py
"""
DataGenius PRO++++ — config package (KOSMOS)
Lekki punkt wejścia do konfiguracji projektu z:
- leniwymi eksportami (settings, registry itp.),
- wersjonowaniem pakietu i bezpiecznym fallbackiem,
- walidacją podstawowych ustawień środowiska,
- helperami do pracy z konfiguracją w testach/CLI.

Użycie:
    from config import settings, ProblemType, get_models_for_problem
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from types import ModuleType
from typing import Any, Dict, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Wersja pakietu (bez twardej zależności od build tooling)
# ──────────────────────────────────────────────────────────────────────────────
try:
    __version__ = _pkg_version("datagenius-pro")
except PackageNotFoundError:
    # fallback na tryb deweloperski / brak zainstalowanego pakietu
    __version__ = "0.0.0-dev"

# ──────────────────────────────────────────────────────────────────────────────
# Leniwe eksporty — nic ciężkiego nie importujemy dopóki nie jest potrzebne
# ──────────────────────────────────────────────────────────────────────────────
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Settings
    "settings": ("config.settings", "settings"),
    "Settings": ("config.settings", "Settings"),

    # Model registry
    "ProblemType": ("config.model_registry", "ProblemType"),
    "get_models_for_problem": ("config.model_registry", "get_models_for_problem"),
    "CLASSIFICATION_MODELS": ("config.model_registry", "CLASSIFICATION_MODELS"),
    "REGRESSION_MODELS": ("config.model_registry", "REGRESSION_MODELS"),

    # (opcjonalnie) klucze/metadane globalne
    "DEFAULT_RANDOM_STATE": ("config.settings", "RANDOM_STATE"),
}

__all__ = tuple(_LAZY_EXPORTS.keys()) + ("__version__",)

# ──────────────────────────────────────────────────────────────────────────────
# Leniwe rozwiązywanie i mikro-cache w module
# ──────────────────────────────────────────────────────────────────────────────
def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, symbol = _LAZY_EXPORTS[name]
        module: ModuleType = import_module(mod_name)
        obj = getattr(module, symbol)
        globals()[name] = obj  # cache w przestrzeni modułu
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))

# ──────────────────────────────────────────────────────────────────────────────
# Walidacja podstawowych ustawień (best-effort) — do wywołania na starcie appki
# ──────────────────────────────────────────────────────────────────────────────
def validate_settings(*, strict: bool = False) -> Dict[str, str]:
    """
    Lekka walidacja kluczowych pól w settings. Zwraca mapę ostrzeżeń.
    Gdy strict=True, rzuca ValueError przy krytycznych brakach.
    """
    warnings: Dict[str, str] = {}
    try:
        from pathlib import Path
        from config import settings as _s  # leniwy import

        # Ścieżki robocze
        base = getattr(_s, "BASE_PATH", Path.cwd())
        data = getattr(_s, "DATA_PATH", base / "data")
        sessions = getattr(_s, "SESSIONS_PATH", base / "sessions")

        for label, p in (("DATA_PATH", data), ("SESSIONS_PATH", sessions)):
            try:
                Path(p).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                msg = f"{label} not writable/creatable: {e}"
                if strict:
                    raise ValueError(msg)
                warnings[label] = msg

        # Losowe seed / API klucz (opcjonalnie)
        rs = getattr(_s, "RANDOM_STATE", None)
        if not isinstance(rs, int):
            warnings["RANDOM_STATE"] = "RANDOM_STATE is not set or not an int; using 42 by default."

        # Limit rozmiaru CSV
        max_csv = int(getattr(_s, "API_MAX_CSV_BYTES", 25_000_000))
        if max_csv < 1_000_000:
            warnings["API_MAX_CSV_BYTES"] = "API_MAX_CSV_BYTES is very low — uploads may fail unexpectedly."

    except Exception as e:
        if strict:
            raise
        warnings["config"] = f"Validation encountered an issue: {e}"

    return warnings

# ──────────────────────────────────────────────────────────────────────────────
# Helpery do testów / narzędzi
# ──────────────────────────────────────────────────────────────────────────────
def use_test_settings(**overrides: Any) -> None:
    """
    Szybkie nadpisanie wybranych pól w `config.settings` (np. w testach).
    Przykład:
        from config import use_test_settings
        use_test_settings(API_MAX_CSV_BYTES=1_000_000, SESSION_TTL_HOURS=1)
    """
    mod: ModuleType = import_module("config.settings")
    s = getattr(mod, "settings", None)
    if s is None:
        raise RuntimeError("config.settings.settings is not available")
    for k, v in overrides.items():
        setattr(s, k, v)
