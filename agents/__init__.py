# agents/__init__.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Agents Package v7.0              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE LAZY-LOADING AGENT ORCHESTRATION SYSTEM                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Lazy Import System (PEP 562)                                          ║
║  ✓ LRU Caching for Performance                                           ║
║  ✓ Comprehensive Agent Ecosystem                                         ║
║  ✓ Clean Public API                                                      ║
║  ✓ Version Management                                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Agent Categories:
    • Orchestrators: High-level workflow coordination
    • ML Agents: Model training, selection, tuning
    • Preprocessing: Data transformation pipeline
    • Monitoring: Drift detection, performance tracking
    • Analysis: EDA, insights, recommendations

Lazy Loading Benefits:
    • Fast initial import (< 100ms)
    • Load dependencies only when needed
    • Memory efficient
    • Thread-safe with LRU cache

Usage:
```python
    # Import is fast - no heavy dependencies loaded yet
    from agents import MentorOrchestrator, PipelineBuilder
    
    # Dependencies load on first use
    mentor = MentorOrchestrator()  # Loads mentor + dependencies
    pipeline = PipelineBuilder()    # Loads preprocessing + sklearn
```
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from types import ModuleType
from typing import Any, Dict, Final, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Package Metadata
# ═══════════════════════════════════════════════════════════════════════════

try:
    __version__ = _pkg_version("datagenius-pro")
except PackageNotFoundError:
    __version__ = "7.0.0-ultimate-dev"

__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# Lazy Export Specification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _LazySpec:
    """Specification for lazy-loaded symbol."""
    module: str
    symbol: str
    category: str = "other"


# Map: public_name → _LazySpec
_LAZY_EXPORTS: Dict[str, _LazySpec] = {
    
    # ═══════════════════════════════════════════════════════════════════════
    # ORCHESTRATORS
    # ═══════════════════════════════════════════════════════════════════════
    
    "MentorOrchestrator": _LazySpec(
        "agents.mentor.orchestrator",
        "MentorOrchestrator",
        "orchestrator"
    ),
    "MentorConfig": _LazySpec(
        "agents.mentor.orchestrator",
        "MentorConfig",
        "orchestrator"
    ),
    
    "MLOrchestrator": _LazySpec(
        "agents.ml.orchestrator",
        "MLOrchestrator",
        "orchestrator"
    ),
    "MLConfig": _LazySpec(
        "agents.ml.orchestrator",
        "MLConfig",
        "orchestrator"
    ),
    
    # ═══════════════════════════════════════════════════════════════════════
    # ML AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    
    "ModelTrainer": _LazySpec(
        "agents.ml.model_trainer",
        "ModelTrainer",
        "ml"
    ),
    "TrainerConfig": _LazySpec(
        "agents.ml.model_trainer",
        "TrainerConfig",
        "ml"
    ),
    "ModelSelector": _LazySpec(
        "agents.ml.model_selector",
        "ModelSelector",
        "ml"
    ),
    "SelectorConfig": _LazySpec(
        "agents.ml.model_selector",
        "SelectorConfig",
        "ml"
    ),
    
    # ═══════════════════════════════════════════════════════════════════════
    # PREPROCESSING AGENTS (UPDATED!)
    # ═══════════════════════════════════════════════════════════════════════
    
    "PipelineBuilder": _LazySpec(
        "agents.preprocessing.pipeline_builder",
        "PipelineBuilder",
        "preprocessing"
    ),
    "PipelineConfig": _LazySpec(
        "agents.preprocessing.pipeline_builder",
        "PipelineConfig",
        "preprocessing"
    ),
    
    "MissingDataHandler": _LazySpec(
        "agents.preprocessing.missing_data_handler",
        "MissingDataHandler",
        "preprocessing"
    ),
    "MissingHandlerConfig": _LazySpec(
        "agents.preprocessing.missing_data_handler",
        "MissingHandlerConfig",
        "preprocessing"
    ),
    
    "FeatureEngineer": _LazySpec(
        "agents.preprocessing.feature_engineer",
        "FeatureEngineer",
        "preprocessing"
    ),
    "FeatureConfig": _LazySpec(
        "agents.preprocessing.feature_engineer",
        "FeatureConfig",
        "preprocessing"
    ),
    
    "EncoderSelector": _LazySpec(
        "agents.preprocessing.encoder_selector",
        "EncoderSelector",
        "preprocessing"
    ),
    "EncoderPolicy": _LazySpec(
        "agents.preprocessing.encoder_selector",
        "EncoderPolicy",
        "preprocessing"
    ),
    
    "ScalerSelector": _LazySpec(
        "agents.preprocessing.scaler_selector",
        "ScalerSelector",
        "preprocessing"
    ),
    "ScalerSelectorConfig": _LazySpec(
        "agents.preprocessing.scaler_selector",
        "ScalerSelectorConfig",
        "preprocessing"
    ),
    
    # ═══════════════════════════════════════════════════════════════════════
    # MONITORING AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    
    "DriftDetector": _LazySpec(
        "agents.monitoring.drift_detector",
        "DriftDetector",
        "monitoring"
    ),
    "DriftConfig": _LazySpec(
        "agents.monitoring.drift_detector",
        "DriftConfig",
        "monitoring"
    ),
    
    "PerformanceTracker": _LazySpec(
        "agents.monitoring.performance_tracker",
        "PerformanceTracker",
        "monitoring"
    ),
    "PerformanceConfig": _LazySpec(
        "agents.monitoring.performance_tracker",
        "PerformanceConfig",
        "monitoring"
    ),
    
    "RetrainingScheduler": _LazySpec(
        "agents.monitoring.retraining_scheduler",
        "RetrainingScheduler",
        "monitoring"
    ),
    "RetrainPolicy": _LazySpec(
        "agents.monitoring.retraining_scheduler",
        "RetrainPolicy",
        "monitoring"
    ),
    
    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    
    "EDAAgent": _LazySpec(
        "agents.analysis.eda_agent",
        "EDAAgent",
        "analysis"
    ),
    "EDAConfig": _LazySpec(
        "agents.analysis.eda_agent",
        "EDAConfig",
        "analysis"
    ),
    
    "InsightsGenerator": _LazySpec(
        "agents.analysis.insights_generator",
        "InsightsGenerator",
        "analysis"
    ),
    "InsightsConfig": _LazySpec(
        "agents.analysis.insights_generator",
        "InsightsConfig",
        "analysis"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

__all__: Final[Tuple[str, ...]] = tuple(_LAZY_EXPORTS.keys()) + (
    "__version__",
    "__author__",
    "__license__",
    "get_version_info",
    "list_agents",
    "list_agents_by_category"
)


# ═══════════════════════════════════════════════════════════════════════════
# Lazy Resolution with LRU Cache
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=len(_LAZY_EXPORTS) or 128)
def _resolve(name: str) -> Any:
    """
    🔄 **Lazy Symbol Resolution with Caching**
    
    Imports module and retrieves symbol on first access.
    Results are cached (LRU) to prevent redundant imports.
    Thread-safe for concurrent access.
    
    Args:
        name: Symbol name to resolve
    
    Returns:
        Resolved symbol
    
    Raises:
        AttributeError: Symbol not in lazy exports
        ImportError: Module import failed
    """
    spec = _LAZY_EXPORTS.get(name)
    
    if spec is None:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Available: {', '.join(sorted(_LAZY_EXPORTS.keys()))}"
        )
    
    try:
        module: ModuleType = importlib.import_module(spec.module)
    except Exception as e:
        raise ImportError(
            f"Failed to import module '{spec.module}' required for '{name}': {e}"
        ) from e
    
    try:
        obj = getattr(module, spec.symbol)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{spec.module}' does not define '{spec.symbol}' "
            f"(needed for '{name}')"
        ) from e
    
    return obj


def __getattr__(name: str) -> Any:
    """
    🎯 **PEP 562 Lazy Attribute Access**
    
    Resolves symbols on first access and caches in module globals.
    """
    obj = _resolve(name)
    globals()[name] = obj  # Cache in module for faster subsequent access
    return obj


def __dir__() -> List[str]:
    """
    📋 **Enhanced Directory Listing**
    
    Returns all available attributes for IDE autocompletion.
    """
    return sorted(list(globals().keys()) + list(__all__))


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def get_version_info() -> Dict[str, Any]:
    """
    📊 **Get Package Version Information**
    
    Returns:
        Dictionary with version details
    
    Example:
```python
        from agents import get_version_info
        
        info = get_version_info()
        print(f"Version: {info['version']}")
        print(f"Agents: {info['total_agents']}")
```
    """
    categories = {}
    for name, spec in _LAZY_EXPORTS.items():
        categories.setdefault(spec.category, []).append(name)
    
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "total_agents": len(_LAZY_EXPORTS),
        "categories": {cat: len(agents) for cat, agents in categories.items()},
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }


def list_agents() -> List[str]:
    """
    📋 **List All Available Agents**
    
    Returns:
        Sorted list of agent names
    
    Example:
```python
        from agents import list_agents
        
        agents = list_agents()
        print(f"Available agents: {len(agents)}")
        for agent in agents:
            print(f"  • {agent}")
```
    """
    return sorted(_LAZY_EXPORTS.keys())


def list_agents_by_category() -> Dict[str, List[str]]:
    """
    🏷️ **List Agents Grouped by Category**
    
    Returns:
        Dictionary mapping category → agent list
    
    Example:
```python
        from agents import list_agents_by_category
        
        by_category = list_agents_by_category()
        
        for category, agents in by_category.items():
            print(f"\n{category.upper()}:")
            for agent in agents:
                print(f"  • {agent}")
```
    """
    categories: Dict[str, List[str]] = {}
    
    for name, spec in _LAZY_EXPORTS.items():
        categories.setdefault(spec.category, []).append(name)
    
    # Sort both categories and agents within each category
    return {
        cat: sorted(agents)
        for cat, agents in sorted(categories.items())
    }


# ═══════════════════════════════════════════════════════════════════════════
# Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _module_init():
    """Initialize module on import."""
    # Optional: Log initialization (disable in production)
    if sys.flags.dev_mode or sys.flags.verbose:
        print(f"✓ DataGenius agents v{__version__} initialized (lazy loading enabled)")

_module_init()


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"DataGenius Agents Package v{__version__}")
    print("="*80)
    
    # Version info
    print("\n📊 Version Information:")
    info = get_version_info()
    for key, value in info.items():
        if key == "categories":
            print(f"  {key}:")
            for cat, count in value.items():
                print(f"    {cat}: {count} agents")
        else:
            print(f"  {key}: {value}")
    
    # List by category
    print("\n🏷️ Agents by Category:")
    by_category = list_agents_by_category()
    
    for category, agents in by_category.items():
        print(f"\n  {category.upper()} ({len(agents)} agents):")
        for agent in agents:
            print(f"    • {agent}")
    
    # Test lazy loading
    print("\n🔄 Testing Lazy Loading:")
    print("  Importing PipelineBuilder...")
    
    try:
        from agents import PipelineBuilder
        print(f"  ✓ PipelineBuilder loaded: {PipelineBuilder}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES:")
    print("="*80)
    print("""
# Fast import - no heavy dependencies loaded yet
from agents import get_version_info, list_agents_by_category

# Check what's available
print(get_version_info())
print(list_agents_by_category())

# Import specific agents (loads on first use)
from agents import (
    PipelineBuilder,
    MissingDataHandler,
    FeatureEngineer,
    EncoderSelector,
    ScalerSelector
)

# Build preprocessing pipeline
pipeline = PipelineBuilder()
pipeline.fit(train_df, 'target')
test_processed = pipeline.transform(test_df)

# Or use individual agents
missing_handler = MissingDataHandler()
feature_engineer = FeatureEngineer()
encoder = EncoderSelector()
scaler = ScalerSelector()

# Import orchestrators for high-level workflows
from agents import MentorOrchestrator, MLOrchestrator

mentor = MentorOrchestrator()
result = mentor.execute(data_path='data.csv', target='target')
    """)