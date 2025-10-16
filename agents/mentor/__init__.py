# agents/mentor/__init__.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Mentor Package                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Enterprise-grade lazy import system with intelligent caching:             ║
║    ✓ PEP 562 lazy attribute resolution (__getattr__)                      ║
║    ✓ LRU cache with globals() injection                                   ║
║    ✓ TYPE_CHECKING branch for static analysis (mypy/IDE)                  ║
║    ✓ Eager import mode (env: DATAGENIUS_DISABLE_LAZY_EXPORTS=1)           ║
║    ✓ Runtime registration for dynamic plugins                             ║
║    ✓ Hot-reload support for development                                   ║
║    ✓ Comprehensive introspection utilities                                ║
║    ✓ Defensive error handling with context                                ║
║    ✓ Zero performance overhead after first access                         ║
║    ✓ Thread-safe caching with LRU                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

Public API (Stable Contract):
    
    Exports:
        - MentorOrchestrator: Main orchestrator agent
        - MentorConfig: Configuration dataclass
        - MENTOR_SYSTEM_PROMPT: System prompt template
        - EDA_EXPLANATION_TEMPLATE: EDA results template
        - ML_RESULTS_TEMPLATE: ML results template
        - RECOMMENDATION_TEMPLATE: Recommendations template
    
    Utilities:
        - preload(names): Eager-load specified exports
        - available_exports(): Introspect available exports
        - register_lazy_export(): Register custom exports
        - reload_export(): Hot-reload for development
    
Usage:
    from agents.mentor import MentorOrchestrator, MENTOR_SYSTEM_PROMPT
    
    orchestrator = MentorOrchestrator()
    result = orchestrator.execute(data=df, context={})

Environment Variables:
    DATAGENIUS_DISABLE_LAZY_EXPORTS=1  # Enable eager import mode
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Configuration & Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _LazySpec:
    """
    Specification for lazy-loaded export.
    
    Attributes:
        module: Fully qualified module path
        symbol: Name of the symbol to import from module
    """
    module: str
    symbol: str


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Export Registry
# ═══════════════════════════════════════════════════════════════════════════

# Registry of all lazy exports with their module paths
_LAZY_EXPORTS: Dict[str, _LazySpec] = {
    # Main orchestrator and configuration
    "MentorOrchestrator": _LazySpec(
        module="agents.mentor.orchestrator",
        symbol="MentorOrchestrator"
    ),
    "MentorConfig": _LazySpec(
        module="agents.mentor.orchestrator",
        symbol="MentorConfig"
    ),
    
    # Prompt templates
    "MENTOR_SYSTEM_PROMPT": _LazySpec(
        module="agents.mentor.prompt_templates",
        symbol="MENTOR_SYSTEM_PROMPT"
    ),
    "EDA_EXPLANATION_TEMPLATE": _LazySpec(
        module="agents.mentor.prompt_templates",
        symbol="EDA_EXPLANATION_TEMPLATE"
    ),
    "ML_RESULTS_TEMPLATE": _LazySpec(
        module="agents.mentor.prompt_templates",
        symbol="ML_RESULTS_TEMPLATE"
    ),
    "RECOMMENDATION_TEMPLATE": _LazySpec(
        module="agents.mentor.prompt_templates",
        symbol="RECOMMENDATION_TEMPLATE"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Public Interface Definition
# ═══════════════════════════════════════════════════════════════════════════

__all__: Final[Tuple[str, ...]] = (
    # Exported classes and constants
    "MentorOrchestrator",
    "MentorConfig",
    "MENTOR_SYSTEM_PROMPT",
    "EDA_EXPLANATION_TEMPLATE",
    "ML_RESULTS_TEMPLATE",
    "RECOMMENDATION_TEMPLATE",
    
    # Utility functions (public API for introspection and control)
    "preload",
    "available_exports",
    "register_lazy_export",
    "reload_export",
)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Static Type Checking Support
# ═══════════════════════════════════════════════════════════════════════════

if TYPE_CHECKING:
    # Import symbols statically for type checkers (mypy, pyright, IDE)
    # This branch is never executed at runtime
    try:
        from agents.mentor.orchestrator import (  # type: ignore
            MentorOrchestrator,
            MentorConfig,
        )
        from agents.mentor.prompt_templates import (  # type: ignore
            MENTOR_SYSTEM_PROMPT,
            EDA_EXPLANATION_TEMPLATE,
            ML_RESULTS_TEMPLATE,
            RECOMMENDATION_TEMPLATE,
        )
    except ImportError:
        # Graceful degradation if modules aren't available yet
        # Type checkers will use 'Any' type
        pass


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Logging & Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

_log = logger.bind(package="agents.mentor", layer="lazy_exports")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Lazy Resolution Implementation
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def _resolve(name: str) -> Any:
    """
    Resolve and import a lazy export by name.
    
    This function is cached with LRU to ensure each export is only
    imported once. After resolution, the object is also injected
    into globals() for direct access on subsequent lookups.
    
    Args:
        name: Public name of the export to resolve
    
    Returns:
        The imported object (class, function, or constant)
    
    Raises:
        AttributeError: If export name is not registered
        ImportError: If module import fails
        AttributeError: If symbol not found in module
    
    Thread Safety:
        LRU cache is thread-safe in CPython due to GIL
    """
    spec = _LAZY_EXPORTS.get(name)
    
    if spec is None:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Available exports: {', '.join(sorted(_LAZY_EXPORTS.keys()))}"
        )
    
    # Import module
    try:
        module: ModuleType = importlib.import_module(spec.module)
    except ImportError as e:
        _log.error(
            f"Failed to import module '{spec.module}' "
            f"required for '{name}': {type(e).__name__}: {e}"
        )
        raise ImportError(
            f"[Mentor] Cannot import module '{spec.module}' "
            f"(required for '{name}'). "
            f"Ensure the module exists and has no import errors."
        ) from e
    
    # Extract symbol from module
    try:
        obj = getattr(module, spec.symbol)
    except AttributeError as e:
        _log.error(
            f"Module '{spec.module}' missing expected symbol '{spec.symbol}' "
            f"(needed for '{name}')"
        )
        raise AttributeError(
            f"[Mentor] Module '{spec.module}' does not define "
            f"expected attribute '{spec.symbol}' (needed for '{name}'). "
            f"Check module implementation."
        ) from e
    
    _log.debug(f"✓ Lazy-loaded: {name} from {spec.module}:{spec.symbol}")
    return obj


def __getattr__(name: str) -> Any:
    """
    PEP 562 module-level __getattr__ for lazy attribute resolution.
    
    This function is called when an attribute is not found in the
    module's __dict__. It performs lazy import and caches the result
    in globals() for subsequent direct access.
    
    Args:
        name: Attribute name being accessed
    
    Returns:
        The resolved object
    
    Raises:
        AttributeError: If attribute is not a registered export
    
    Performance:
        - First access: O(1) dict lookup + import + cache
        - Subsequent access: O(1) globals() lookup (bypasses this function)
    """
    if name in _LAZY_EXPORTS:
        obj = _resolve(name)
        
        # Inject into globals for direct access on next lookup
        # This bypasses __getattr__ for subsequent accesses
        globals()[name] = obj
        
        return obj
    
    # Provide helpful error message with suggestions
    available = sorted(_LAZY_EXPORTS.keys())
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Available exports: {', '.join(available)}"
    )


def __dir__() -> List[str]:
    """
    Return list of available attributes for introspection.
    
    Combines:
      • Built-in module attributes (globals)
      • Lazy export names
      • Utility functions
    
    Used by:
      • dir(module)
      • IDE autocomplete
      • help() documentation
    
    Returns:
        Sorted list of attribute names
    """
    return sorted(
        set(
            list(globals().keys()) +
            list(_LAZY_EXPORTS.keys()) +
            list(__all__)
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Introspection & Management Utilities
# ═══════════════════════════════════════════════════════════════════════════

def available_exports() -> Dict[str, Tuple[str, str]]:
    """
    Get registry of all available lazy exports.
    
    Returns:
        Dictionary mapping public names to (module_path, symbol_name) tuples
    
    Example:
        >>> exports = available_exports()
        >>> exports['MentorOrchestrator']
        ('agents.mentor.orchestrator', 'MentorOrchestrator')
    
    Use Cases:
        • Testing and validation
        • Documentation generation
        • Debugging import issues
        • Plugin discovery
    """
    return {
        name: (spec.module, spec.symbol)
        for name, spec in _LAZY_EXPORTS.items()
    }


def register_lazy_export(
    public_name: str,
    module_path: str,
    symbol_name: str
) -> None:
    """
    Register a new lazy export at runtime.
    
    This enables dynamic plugin systems where exports can be registered
    after package initialization.
    
    Args:
        public_name: Name to expose in module namespace
        module_path: Fully qualified module path
        symbol_name: Name of symbol to import from module
    
    Raises:
        ValueError: If any argument is invalid
    
    Side Effects:
        • Adds entry to _LAZY_EXPORTS registry
        • Clears resolution cache for this name
        • Removes any existing cached object from globals()
    
    Example:
        >>> register_lazy_export(
        ...     public_name='CustomMentor',
        ...     module_path='plugins.custom_mentor',
        ...     symbol_name='CustomMentorClass'
        ... )
    
    Thread Safety:
        Not thread-safe. Call during initialization or with external locking.
    """
    # Validation
    if not public_name or not isinstance(public_name, str):
        raise ValueError("public_name must be a non-empty string")
    
    if not module_path or not isinstance(module_path, str):
        raise ValueError("module_path must be a non-empty string")
    
    if not symbol_name or not isinstance(symbol_name, str):
        raise ValueError("symbol_name must be a non-empty string")
    
    # Register new export
    _LAZY_EXPORTS[public_name] = _LazySpec(
        module=module_path,
        symbol=symbol_name
    )
    
    # Clear caches
    try:
        _resolve.cache_clear()
    except Exception as e:
        _log.warning(f"Failed to clear resolution cache: {e}")
    
    # Remove stale object from globals
    globals().pop(public_name, None)
    
    _log.info(
        f"✓ Registered lazy export: {public_name} → "
        f"{module_path}:{symbol_name}"
    )


def reload_export(public_name: str) -> Any:
    """
    Force reload of a specific export (hot-reload for development).
    
    Useful during development to pick up code changes without
    restarting the Python interpreter.
    
    Args:
        public_name: Name of export to reload
    
    Returns:
        Freshly imported object
    
    Raises:
        AttributeError: If export name not registered
    
    Side Effects:
        • Clears LRU cache
        • Re-imports module
        • Updates globals() with new object
    
    Example:
        >>> # After modifying orchestrator.py
        >>> MentorOrchestrator = reload_export('MentorOrchestrator')
    
    Warning:
        This does NOT reload the actual module file. It only
        re-executes the import. For true hot-reload, use
        importlib.reload() on the module first.
    """
    if public_name not in _LAZY_EXPORTS:
        available = sorted(_LAZY_EXPORTS.keys())
        raise AttributeError(
            f"Unknown export '{public_name}'. "
            f"Available: {', '.join(available)}"
        )
    
    # Clear all caches
    try:
        _resolve.cache_clear()
    except Exception as e:
        _log.warning(f"Failed to clear resolution cache: {e}")
    
    # Remove cached object
    globals().pop(public_name, None)
    
    # Re-resolve
    obj = _resolve(public_name)
    globals()[public_name] = obj
    
    _log.info(f"✓ Reloaded export: {public_name}")
    return obj


def preload(names: Optional[Iterable[str]] = None) -> None:
    """
    Eagerly import specified exports (or all if names=None).
    
    Use this to:
      • Warm up caches during application startup
      • Avoid lazy import overhead on first request
      • Catch import errors early in serverless functions
      • Pre-validate module integrity in CI/CD
    
    Args:
        names: Specific export names to preload, or None for all
    
    Raises:
        AttributeError: If unknown export name specified
        ImportError: If module import fails
    
    Side Effects:
        • Imports modules immediately
        • Populates globals() with imported objects
        • Populates LRU cache
    
    Example:
        >>> # Preload all exports at startup
        >>> preload()
        
        >>> # Preload only specific exports
        >>> preload(['MentorOrchestrator', 'MentorConfig'])
    
    Performance:
        Each import happens only once due to Python's module cache.
    """
    targets = list(names) if names is not None else list(_LAZY_EXPORTS.keys())
    
    _log.info(f"Preloading {len(targets)} export(s)...")
    
    failed: List[Tuple[str, Exception]] = []
    
    for name in targets:
        if name not in _LAZY_EXPORTS:
            available = sorted(_LAZY_EXPORTS.keys())
            raise AttributeError(
                f"Unknown export '{name}'. "
                f"Available: {', '.join(available)}"
            )
        
        try:
            obj = _resolve(name)
            globals()[name] = obj
            _log.debug(f"  ✓ {name}")
        except Exception as e:
            failed.append((name, e))
            _log.error(f"  ✗ {name}: {type(e).__name__}: {e}")
    
    if failed:
        failed_names = [name for name, _ in failed]
        raise ImportError(
            f"Preload failed for {len(failed)} export(s): "
            f"{', '.join(failed_names)}. "
            f"See logs for details."
        )
    
    _log.success(f"✓ Preloaded {len(targets)} export(s) successfully")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Eager Import Mode (Environment-Driven)
# ═══════════════════════════════════════════════════════════════════════════

# Check for eager import mode via environment variable
_eager_mode_enabled = os.environ.get(
    "DATAGENIUS_DISABLE_LAZY_EXPORTS",
    ""
).strip().lower() in {"1", "true", "yes"}

if _eager_mode_enabled:
    _log.info("Eager import mode enabled (DATAGENIUS_DISABLE_LAZY_EXPORTS=1)")
    
    try:
        preload()
        _log.success("✓ All exports preloaded successfully")
    except Exception as e:
        # Don't crash application - fall back to lazy mode
        _log.warning(
            f"⚠ Eager import failed, falling back to lazy mode: "
            f"{type(e).__name__}: {e}"
        )