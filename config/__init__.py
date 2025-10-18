# config/__init__.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Configuration Package v7.0       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE LAZY-LOADING CONFIGURATION SYSTEM                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Lazy Module Loading                                                   ║
║  ✓ Package Versioning                                                    ║
║  ✓ Settings Validation                                                   ║
║  ✓ Test Utilities                                                        ║
║  ✓ Safe Fallbacks                                                        ║
║  ✓ Auto-Caching                                                          ║
║  ✓ Type Safety                                                           ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Lazy Loading Pattern:
```
    config/
    ├── __init__.py          # Lazy exports (this file)
    ├── settings.py          # Application settings
    └── model_registry.py    # ML model registry
    
    Import Flow:
    1. from config import settings
       └─→ Lazy load config.settings
           └─→ Cache in globals()
           └─→ Return cached on next access
```

Features:
    Lazy Loading:
        • Import modules only when accessed
        • Auto-caching in module globals
        • No circular dependencies
    
    Settings Management:
        • Centralized configuration
        • Environment-aware
        • Validation utilities
    
    Model Registry:
        • ML model configurations
        • Problem-type specific models
        • Easy extensibility
    
    Test Support:
        • Runtime setting overrides
        • Test fixtures
        • Isolated configuration

Usage:
```python
    # Lazy imports (loaded on first access)
    from config import settings
    from config import ProblemType, get_models_for_problem
    
    # Access settings
    print(settings.BASE_PATH)
    print(settings.API_MAX_ROWS)
    
    # Get models for problem
    models = get_models_for_problem("classification")
    
    # Validate configuration
    from config import validate_settings
    warnings = validate_settings()
    
    # Test utilities
    from config import use_test_settings
    use_test_settings(API_MAX_ROWS=1000)
```

Export Map:
    Settings:
      • settings: Application settings instance
      • Settings: Settings class
    
    Model Registry:
      • ProblemType: Type hint for problem types
      • get_models_for_problem: Get models by type
      • CLASSIFICATION_MODELS: Classification model list
      • REGRESSION_MODELS: Regression model list
    
    Utilities:
      • validate_settings: Validate configuration
      • use_test_settings: Override for tests
      • __version__: Package version

Dependencies:
    • None (pure Python)
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Package Metadata
# ═══════════════════════════════════════════════════════════════════════════

try:
    __version__ = _pkg_version("datagenius-pro")
except PackageNotFoundError:
    # Development mode / uninstalled package
    __version__ = "7.0.0-dev"

__author__ = "DataGenius Enterprise Team"


# ═══════════════════════════════════════════════════════════════════════════
# Lazy Export Definitions
# ═══════════════════════════════════════════════════════════════════════════

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Settings
    "settings": ("config.settings", "settings"),
    "Settings": ("config.settings", "Settings"),
    
    # Model Registry
    "ProblemType": ("config.model_registry", "ProblemType"),
    "get_models_for_problem": ("config.model_registry", "get_models_for_problem"),
    "CLASSIFICATION_MODELS": ("config.model_registry", "CLASSIFICATION_MODELS"),
    "REGRESSION_MODELS": ("config.model_registry", "REGRESSION_MODELS"),
    
    # Constants
    "DEFAULT_RANDOM_STATE": ("config.settings", "RANDOM_STATE"),
}

__all__ = (
    # Metadata
    "__version__",
    
    # Settings
    "settings",
    "Settings",
    
    # Model Registry
    "ProblemType",
    "get_models_for_problem",
    "CLASSIFICATION_MODELS",
    "REGRESSION_MODELS",
    
    # Constants
    "DEFAULT_RANDOM_STATE",
    
    # Utilities
    "validate_settings",
    "use_test_settings",
    "get_config_info"
)


# ═══════════════════════════════════════════════════════════════════════════
# Lazy Loading Implementation
# ═══════════════════════════════════════════════════════════════════════════

def __getattr__(name: str) -> Any:
    """
    Lazy attribute resolution.
    
    Loads modules only when their exports are first accessed.
    Caches loaded objects in module globals for fast subsequent access.
    
    Args:
        name: Attribute name to resolve
    
    Returns:
        Resolved attribute value
    
    Raises:
        AttributeError: If attribute not found
    """
    if name in _LAZY_EXPORTS:
        module_name, symbol_name = _LAZY_EXPORTS[name]
        
        try:
            # Import module
            module: ModuleType = import_module(module_name)
            
            # Get symbol from module
            obj = getattr(module, symbol_name)
            
            # Cache in globals for fast subsequent access
            globals()[name] = obj
            
            return obj
        
        except (ImportError, AttributeError) as e:
            raise AttributeError(
                f"Failed to load '{name}' from '{module_name}': {e}"
            )
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> List[str]:
    """
    Return module directory including lazy exports.
    
    Returns:
        Sorted list of all available attributes
    """
    return sorted(set(list(globals().keys()) + list(_LAZY_EXPORTS.keys())))


# ═══════════════════════════════════════════════════════════════════════════
# Configuration Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_settings(*, strict: bool = False) -> Dict[str, str]:
    """
    🔍 **Validate Configuration Settings**
    
    Performs basic validation of critical configuration settings.
    
    Args:
        strict: If True, raises ValueError on critical issues
    
    Returns:
        Dictionary of warnings (empty if all OK)
    
    Raises:
        ValueError: If strict=True and critical issues found
    
    Example:
```python
        from config import validate_settings
        
        # Check for warnings
        warnings = validate_settings()
        if warnings:
            for key, msg in warnings.items():
                print(f"⚠ {key}: {msg}")
        
        # Strict mode (raises on errors)
        try:
            validate_settings(strict=True)
        except ValueError as e:
            print(f"✗ Configuration error: {e}")
```
    """
    warnings: Dict[str, str] = {}
    
    try:
        from pathlib import Path
        from config import settings as _s
        
        # Validate paths
        base = getattr(_s, "BASE_PATH", Path.cwd())
        data = getattr(_s, "DATA_PATH", base / "data")
        sessions = getattr(_s, "SESSIONS_PATH", base / "sessions")
        workflows = getattr(_s, "WORKFLOWS_PATH", base / "workflows")
        
        for label, path in [
            ("BASE_PATH", base),
            ("DATA_PATH", data),
            ("SESSIONS_PATH", sessions),
            ("WORKFLOWS_PATH", workflows)
        ]:
            try:
                p = Path(path)
                p.mkdir(parents=True, exist_ok=True)
                
                # Check writability
                test_file = p / ".write_test"
                try:
                    test_file.touch()
                    test_file.unlink()
                except Exception:
                    msg = f"{label} ({path}) is not writable"
                    if strict:
                        raise ValueError(msg)
                    warnings[label] = msg
            
            except Exception as e:
                msg = f"{label} ({path}) is not accessible: {e}"
                if strict:
                    raise ValueError(msg)
                warnings[label] = msg
        
        # Validate random state
        random_state = getattr(_s, "RANDOM_STATE", None)
        if not isinstance(random_state, int):
            warnings["RANDOM_STATE"] = (
                "RANDOM_STATE is not set or not an integer; "
                "using 42 by default"
            )
        elif random_state < 0:
            warnings["RANDOM_STATE"] = "RANDOM_STATE should be non-negative"
        
        # Validate API limits
        max_csv = int(getattr(_s, "API_MAX_CSV_BYTES", 25_000_000))
        if max_csv < 1_000_000:
            warnings["API_MAX_CSV_BYTES"] = (
                f"API_MAX_CSV_BYTES is very low ({max_csv:,} bytes); "
                "uploads may fail unexpectedly"
            )
        
        max_rows = int(getattr(_s, "API_MAX_ROWS", 2_000_000))
        if max_rows < 1000:
            warnings["API_MAX_ROWS"] = (
                f"API_MAX_ROWS is very low ({max_rows:,}); "
                "may limit dataset processing"
            )
        
        max_cols = int(getattr(_s, "API_MAX_COLUMNS", 2_000))
        if max_cols < 10:
            warnings["API_MAX_COLUMNS"] = (
                f"API_MAX_COLUMNS is very low ({max_cols}); "
                "may limit dataset processing"
            )
        
        # Validate session TTL
        session_ttl = int(getattr(_s, "SESSION_TTL_HOURS", 12))
        if session_ttl < 1:
            warnings["SESSION_TTL_HOURS"] = (
                "SESSION_TTL_HOURS is less than 1; "
                "sessions may expire too quickly"
            )
        
        # Validate workflow settings
        workflow_retries = int(getattr(_s, "WORKFLOW_MAX_RETRIES", 2))
        if workflow_retries < 0:
            warnings["WORKFLOW_MAX_RETRIES"] = (
                "WORKFLOW_MAX_RETRIES should be non-negative"
            )
    
    except Exception as e:
        if strict:
            raise
        warnings["config"] = f"Validation encountered an issue: {e}"
    
    return warnings


# ═══════════════════════════════════════════════════════════════════════════
# Test Utilities
# ═══════════════════════════════════════════════════════════════════════════

def use_test_settings(**overrides: Any) -> None:
    """
    🧪 **Override Settings for Tests**
    
    Temporarily override configuration settings for testing.
    
    Args:
        **overrides: Settings to override
    
    Example:
```python
        from config import use_test_settings
        
        # Override for tests
        use_test_settings(
            API_MAX_CSV_BYTES=1_000_000,
            SESSION_TTL_HOURS=1,
            RANDOM_STATE=123
        )
```
    
    Warning:
        These changes are global and persist until process restart.
        Use with caution outside of test environments.
    """
    try:
        mod: ModuleType = import_module("config.settings")
        settings_instance = getattr(mod, "settings", None)
        
        if settings_instance is None:
            raise RuntimeError("config.settings.settings is not available")
        
        for key, value in overrides.items():
            if not hasattr(settings_instance, key):
                raise AttributeError(
                    f"Setting '{key}' does not exist in configuration"
                )
            
            setattr(settings_instance, key, value)
    
    except ImportError as e:
        raise RuntimeError(f"Failed to load config.settings: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration Info
# ═══════════════════════════════════════════════════════════════════════════

def get_config_info() -> Dict[str, Any]:
    """
    📊 **Get Configuration Information**
    
    Returns summary of current configuration.
    
    Returns:
        Dictionary with configuration info
    
    Example:
```python
        from config import get_config_info
        
        info = get_config_info()
        print(f"Version: {info['version']}")
        print(f"Loaded modules: {info['loaded_modules']}")
        print(f"Available exports: {info['exports']}")
```
    """
    from pathlib import Path
    
    info: Dict[str, Any] = {
        "version": __version__,
        "loaded_modules": [],
        "exports": list(_LAZY_EXPORTS.keys()),
        "cached": [],
        "settings_loaded": False,
        "registry_loaded": False
    }
    
    # Check which modules are cached
    for name in _LAZY_EXPORTS.keys():
        if name in globals():
            info["cached"].append(name)
    
    # Check if settings module loaded
    try:
        from config import settings
        info["settings_loaded"] = True
        info["settings"] = {
            "BASE_PATH": str(getattr(settings, "BASE_PATH", "N/A")),
            "DATA_PATH": str(getattr(settings, "DATA_PATH", "N/A")),
            "RANDOM_STATE": getattr(settings, "RANDOM_STATE", "N/A"),
            "API_MAX_ROWS": getattr(settings, "API_MAX_ROWS", "N/A")
        }
    except Exception:
        pass
    
    # Check if model registry loaded
    try:
        from config import CLASSIFICATION_MODELS, REGRESSION_MODELS
        info["registry_loaded"] = True
        info["models"] = {
            "classification_count": len(CLASSIFICATION_MODELS) if CLASSIFICATION_MODELS else 0,
            "regression_count": len(REGRESSION_MODELS) if REGRESSION_MODELS else 0
        }
    except Exception:
        pass
    
    return info


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Configuration Package v{__version__} - Self Test")
    print("="*80)
    
    # Test lazy loading
    print("\n1. Testing Lazy Loading...")
    print(f"   Available exports: {len(__all__)}")
    print(f"   Lazy exports defined: {len(_LAZY_EXPORTS)}")
    
    # Test settings import
    print("\n2. Testing Settings Import...")
    try:
        from config import settings
        print(f"   ✓ Settings loaded")
        print(f"   BASE_PATH: {settings.BASE_PATH}")
        print(f"   RANDOM_STATE: {settings.RANDOM_STATE}")
    except Exception as e:
        print(f"   ✗ Settings failed: {e}")
    
    # Test model registry
    print("\n3. Testing Model Registry...")
    try:
        from config import get_models_for_problem
        clf_models = get_models_for_problem("classification")
        reg_models = get_models_for_problem("regression")
        print(f"   ✓ Model registry loaded")
        print(f"   Classification models: {len(clf_models)}")
        print(f"   Regression models: {len(reg_models)}")
    except Exception as e:
        print(f"   ✗ Model registry failed: {e}")
    
    # Test validation
    print("\n4. Testing Configuration Validation...")
    warnings = validate_settings()
    if warnings:
        print(f"   ⚠ Found {len(warnings)} warning(s):")
        for key, msg in warnings.items():
            print(f"     • {key}: {msg}")
    else:
        print(f"   ✓ All settings valid")
    
    # Test config info
    print("\n5. Testing Config Info...")
    info = get_config_info()
    print(f"   Version: {info['version']}")
    print(f"   Cached exports: {len(info['cached'])}")
    print(f"   Settings loaded: {info['settings_loaded']}")
    print(f"   Registry loaded: {info['registry_loaded']}")
    
    # Test override
    print("\n6. Testing Setting Override...")
    try:
        original = settings.RANDOM_STATE
        use_test_settings(RANDOM_STATE=999)
        print(f"   ✓ Override successful: {original} → {settings.RANDOM_STATE}")
        use_test_settings(RANDOM_STATE=original)  # Restore
        print(f"   ✓ Restored: {settings.RANDOM_STATE}")
    except Exception as e:
        print(f"   ✗ Override failed: {e}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
# === Basic Usage ===
from config import settings, get_models_for_problem

# Access settings
print(settings.BASE_PATH)
print(settings.API_MAX_ROWS)

# Get models
clf_models = get_models_for_problem("classification")
reg_models = get_models_for_problem("regression")

# === Validation ===
from config import validate_settings

warnings = validate_settings()
if warnings:
    for key, msg in warnings.items():
        print(f"⚠ {key}: {msg}")

# === Testing ===
from config import use_test_settings

use_test_settings(
    API_MAX_ROWS=1000,
    SESSION_TTL_HOURS=1
)

# === Configuration Info ===
from config import get_config_info

info = get_config_info()
print(f"Version: {info['version']}")
print(f"Settings loaded: {info['settings_loaded']}")
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)
