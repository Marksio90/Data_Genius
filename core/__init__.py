# core/__init__.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Core Package v7.0                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE CORE FRAMEWORK PACKAGE                                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Lazy Module Loading                                                   ║
║  ✓ Clean Public API                                                      ║
║  ✓ Auto-Caching                                                          ║
║  ✓ Type Safety                                                           ║
║  ✓ Version Management                                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Core Package Structure:
```
    core/
    ├── __init__.py          # Lazy exports (this file)
    ├── base_agent.py        # Agent framework
    └── llm_client.py        # LLM client (optional)
```

Lazy Loading Pattern:
```
    Import Flow:
    1. from core import BaseAgent
       └─→ Lazy load core.base_agent
           └─→ Cache in globals()
           └─→ Return cached on next access
```

Features:
    Lazy Loading:
        • Import modules only when accessed
        • Auto-caching in module globals
        • No circular dependencies
        • Fast subsequent access
    
    Public API:
        • BaseAgent - Abstract base class
        • PipelineAgent - Sequential orchestration
        • ParallelAgent - Concurrent execution
        • AgentResult - Standard result format
        • get_llm_client - LLM client factory (if available)
    
    Version Management:
        • Automatic version detection
        • Fallback for development
        • Package metadata support

Usage:
```python
    # Import core components (lazy-loaded)
    from core import BaseAgent, AgentResult
    
    # Create custom agent
    class MyAgent(BaseAgent):
        def execute(self, **kwargs) -> AgentResult:
            result = AgentResult(agent_name=self.name)
            # Your logic here
            return result
    
    # Use agent
    agent = MyAgent("my_agent")
    result = agent.run(data=input_data)
    
    # Pipeline orchestration
    from core import PipelineAgent
    
    pipeline = PipelineAgent(
        name="my_pipeline",
        agents=[agent1, agent2, agent3]
    )
    result = pipeline.run(data=input_data)
    
    # LLM client (if available)
    from core import get_llm_client
    
    llm = get_llm_client()
    response = llm.complete("Your prompt here")
```

Export Map:
    Agents:
      • BaseAgent: Abstract base class for all agents
      • PipelineAgent: Sequential agent orchestration
      • ParallelAgent: Parallel agent execution
      • AgentResult: Standard result format
      • AgentStatus: Status type literal
    
    LLM:
      • get_llm_client: LLM client factory (optional)

Dependencies:
    • None (pure Python lazy loading)
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from types import ModuleType
from typing import Any, Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
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
    # Agent Framework
    "BaseAgent": ("core.base_agent", "BaseAgent"),
    "PipelineAgent": ("core.base_agent", "PipelineAgent"),
    "ParallelAgent": ("core.base_agent", "ParallelAgent"),
    "AgentResult": ("core.base_agent", "AgentResult"),
    "AgentStatus": ("core.base_agent", "AgentStatus"),
    
    # LLM Client (optional - may not exist)
    "get_llm_client": ("core.llm_client", "get_llm_client"),
}

__all__ = (
    # Metadata
    "__version__",
    
    # Agent Framework
    "BaseAgent",
    "PipelineAgent",
    "ParallelAgent",
    "AgentResult",
    "AgentStatus",
    
    # LLM Client
    "get_llm_client"
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
            # Handle optional modules gracefully
            if module_name == "core.llm_client":
                raise AttributeError(
                    f"'{name}' not available - core.llm_client module not found. "
                    "This is optional functionality."
                )
            
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
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Core Package v{__version__} - Self Test")
    print("="*80)
    
    # Test version
    print(f"\n1. Testing Version...")
    print(f"   Version: {__version__}")
    
    # Test lazy imports
    print(f"\n2. Testing Lazy Imports...")
    print(f"   Available exports: {len(__all__)}")
    
    # Test BaseAgent
    print(f"\n3. Testing BaseAgent Import...")
    try:
        from core import BaseAgent
        print(f"   ✓ BaseAgent imported: {BaseAgent.__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test AgentResult
    print(f"\n4. Testing AgentResult Import...")
    try:
        from core import AgentResult
        print(f"   ✓ AgentResult imported: {AgentResult.__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test PipelineAgent
    print(f"\n5. Testing PipelineAgent Import...")
    try:
        from core import PipelineAgent
        print(f"   ✓ PipelineAgent imported: {PipelineAgent.__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test ParallelAgent
    print(f"\n6. Testing ParallelAgent Import...")
    try:
        from core import ParallelAgent
        print(f"   ✓ ParallelAgent imported: {ParallelAgent.__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test LLM Client (optional)
    print(f"\n7. Testing LLM Client Import (Optional)...")
    try:
        from core import get_llm_client
        print(f"   ✓ get_llm_client imported: {get_llm_client.__name__}")
    except AttributeError as e:
        print(f"   ⚠ LLM client not available (optional): {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
    
    # Test __dir__
    print(f"\n8. Testing __dir__...")
    available = dir()
    print(f"   Available attributes: {len(available)}")
    print(f"   Sample: {', '.join(sorted(available)[:5])}...")
    
    # Test caching
    print(f"\n9. Testing Caching...")
    try:
        from core import BaseAgent as BA1
        from core import BaseAgent as BA2
        if BA1 is BA2:
            print(f"   ✓ Caching works (same object)")
        else:
            print(f"   ✗ Caching failed (different objects)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
# === Basic Import ===
from core import BaseAgent, AgentResult

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="my_agent",
            description="Custom agent",
            version="1.0"
        )
    
    def execute(self, **kwargs) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        
        # Your logic here
        data = kwargs.get("data")
        processed = self.process(data)
        
        result.add_data(output=processed)
        return result

# === Agent Execution ===
agent = MyAgent()
result = agent.run(data=input_data)

if result.is_success():
    print(f"Success: {result.data}")
else:
    print(f"Failed: {result.errors}")

# === Pipeline Orchestration ===
from core import PipelineAgent

pipeline = PipelineAgent(
    name="ml_pipeline",
    agents=[
        LoaderAgent(),
        ProcessorAgent(),
        TrainerAgent()
    ]
)

result = pipeline.run(file_path="data.csv")

# === Parallel Execution ===
from core import ParallelAgent

parallel = ParallelAgent(
    name="feature_engineering",
    agents=[
        FeatureAgent1(),
        FeatureAgent2(),
        FeatureAgent3()
    ],
    max_workers=3
)

result = parallel.run(data=input_data)

# === LLM Client (Optional) ===
try:
    from core import get_llm_client
    
    llm = get_llm_client()
    response = llm.complete("Analyze this data")
    print(response)
except AttributeError:
    print("LLM client not available")

# === Type Hints ===
from core import BaseAgent, AgentResult
from typing import List

def process_agents(agents: List[BaseAgent]) -> List[AgentResult]:
    return [agent.run() for agent in agents]

# === Custom Agent with Resilience ===
from core import BaseAgent, AgentResult

class ResilientAgent(BaseAgent):
    def __init__(self):
        super().__init__("resilient_agent")
        
        # Configure resilience
        self.max_retries = 3
        self.backoff_base = 0.5
        self.timeout_sec = 30
        self.enable_circuit_breaker = True
        self.rate_limit_rps = 10.0
    
    def execute(self, **kwargs) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        
        try:
            # Your logic with automatic retry/circuit breaker
            output = self.risky_operation(kwargs)
            result.add_data(output=output)
        except Exception as e:
            result.add_error(str(e))
        
        return result

agent = ResilientAgent()
result = agent.run(data=input_data)
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)
