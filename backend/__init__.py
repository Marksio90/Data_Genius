# backend/__init__.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Backend Module v7.0              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 FastAPI Backend Application Layer                                     ║
╚════════════════════════════════════════════════════════════════════════════╝

Backend Components:
    • API Layer: RESTful endpoints (FastAPI)
    • Pipeline Executor: ML pipeline orchestration
    • Workflow Engine: Multi-step workflow management
    • Session Manager: State and session handling
    • File Handler: Upload/download management
    • App Controller: Application lifecycle

Integration:
    Backend connects to agents/ for ML functionality:
    - agents.preprocessing → Data preprocessing
    - agents.ml → Model training
    - agents.monitoring → Performance tracking
"""

from __future__ import annotations

__version__ = "7.0.0-ultimate"
__author__ = "DataGenius Enterprise Team"

# Backend components are imported by FastAPI app
# No need for lazy loading here as this is application layer

__all__ = [
    "__version__",
    "__author__"
]