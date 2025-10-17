# backend/api/__init__.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — FastAPI Application v7.0         ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE FASTAPI APP FACTORY WITH ENTERPRISE FEATURES                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ CORS & GZip Middleware                                                ║
║  ✓ Loguru Integration (uvicorn bridge)                                   ║
║  ✓ Structured Error Handling                                             ║
║  ✓ ORJSON Performance Boost                                              ║
║  ✓ Async Lifespan Management                                             ║
║  ✓ OpenAPI Documentation                                                 ║
║  ✓ Production-Ready Configuration                                        ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    FastAPI Application                       │
    ├──────────────────────────────────────────────────────────────┤
    │  Middleware Stack:                                           │
    │    1. CORS (Cross-Origin Resource Sharing)                   │
    │    2. GZip (Response Compression)                            │
    │    3. Custom Error Handlers                                  │
    ├──────────────────────────────────────────────────────────────┤
    │  Logging:                                                    │
    │    • Loguru (structured, colored)                            │
    │    • Uvicorn bridge (intercept stdlib logs)                  │
    ├──────────────────────────────────────────────────────────────┤
    │  Routing:                                                    │
    │    • /          → Root info                                  │
    │    • /docs      → Swagger UI                                 │
    │    • /redoc     → ReDoc                                      │
    │    • /api/*     → API routes (from routes.py)                │
    └──────────────────────────────────────────────────────────────┘

Features:
    • ORJSON: Fast JSON serialization (if available)
    • Lifespan: Async startup/shutdown hooks
    • Error Handlers: Consistent JSON error responses
    • CORS: Configurable via settings
    • Compression: GZip for responses > 1KB

Usage:
    # Development
    uvicorn backend.api:get_app --reload --port 8000 --log-level debug
    
    # Production
    gunicorn backend.api:get_app \
        -w 4 \
        -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 \
        --access-logfile - \
        --error-logfile -
    
    # Docker
    CMD ["uvicorn", "backend.api:get_app", "--host", "0.0.0.0", "--port", "8000"]
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from typing import Any, Callable, Dict, Iterable, Optional

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# ═══════════════════════════════════════════════════════════════════════════
# Optional Performance Boost: ORJSON
# ═══════════════════════════════════════════════════════════════════════════

try:
    import orjson
    
    def _orjson_dumps(v: Any, *, default: Any) -> bytes:
        return orjson.dumps(v, default=default)
    
    _HAS_ORJSON = True

except ImportError:
    orjson = None
    _orjson_dumps = None
    _HAS_ORJSON = False


# ═══════════════════════════════════════════════════════════════════════════
# Local Imports
# ═══════════════════════════════════════════════════════════════════════════

from config.settings import settings

# Import router - create if doesn't exist
try:
    from backend.api.routes import router as api_router
except ImportError:
    from fastapi import APIRouter
    api_router = APIRouter(prefix="/api")
    logger.warning("⚠ backend.api.routes not found - using empty router")


# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = ["create_app", "get_app", "__version__"]


# ═══════════════════════════════════════════════════════════════════════════
# Loguru ↔ Stdlib Bridge (Beautiful Uvicorn Logs)
# ═══════════════════════════════════════════════════════════════════════════

class InterceptHandler(logging.Handler):
    """Intercept stdlib logging and route to Loguru."""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where log originated
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _configure_logging() -> None:
    """Configure Loguru to intercept all stdlib logging."""
    # Remove default Loguru handler
    logger.remove()
    
    # Add custom handler with formatting
    log_level = getattr(settings, "LOG_LEVEL", "INFO")
    
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=False
    )
    
    # Optional: Add file handler for production
    if getattr(settings, "LOG_TO_FILE", False):
        logger.add(
            "logs/app_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            compression="zip",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )
    
    # Intercept stdlib logging
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(log_level)
    
    # Intercept specific loggers
    for logger_name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "sqlalchemy",
        "alembic"
    ):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


# ═══════════════════════════════════════════════════════════════════════════
# Performance Monitoring Middleware (Optional)
# ═══════════════════════════════════════════════════════════════════════════

class TimingMiddleware(BaseHTTPMiddleware):
    """Log request timing for monitoring."""
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.2f}s"
            )
        
        return response


# ═══════════════════════════════════════════════════════════════════════════
# Lifespan Management (Startup/Shutdown Hooks)
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for application lifespan.
    
    Startup:
        • Initialize connections (DB, Redis, etc.)
        • Load ML models
        • Setup monitoring
    
    Shutdown:
        • Close connections
        • Cleanup resources
        • Flush logs
    """
    # ─────────────────────────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("="*80)
    logger.info("🚀 Starting DataGenius PRO Master Enterprise ++++")
    logger.info("="*80)
    
    logger.info(f"Version: {__version__}")
    logger.info(f"Environment: {getattr(settings, 'ENV', 'development')}")
    logger.info(f"ORJSON: {'✓ Enabled' if _HAS_ORJSON else '✗ Disabled'}")
    
    # Initialize resources here
    # Example:
    # - await init_database()
    # - await load_ml_models()
    # - await setup_cache()
    
    logger.info("✓ Application startup complete")
    
    # ─────────────────────────────────────────────────────────────────
    # APPLICATION RUNNING
    # ─────────────────────────────────────────────────────────────────
    
    try:
        yield
    
    # ─────────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────────
    
    finally:
        logger.info("="*80)
        logger.info("🛑 Shutting down DataGenius PRO")
        logger.info("="*80)
        
        # Cleanup resources here
        # Example:
        # - await close_database()
        # - await cleanup_cache()
        
        logger.info("✓ Application shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════
# App Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    """
    🏭 **FastAPI Application Factory**
    
    Creates and configures the FastAPI application with:
      • Middleware (CORS, GZip, Timing)
      • Error handlers
      • Routing
      • Documentation
    
    Returns:
        Configured FastAPI application instance
    """
    # Configure logging first
    _configure_logging()
    
    # ─────────────────────────────────────────────────────────────────
    # Create FastAPI App
    # ─────────────────────────────────────────────────────────────────
    
    app_config = {
        "title": getattr(settings, "APP_NAME", "DataGenius PRO API"),
        "version": getattr(settings, "APP_VERSION", __version__),
        "description": "Ultimate Enterprise-Grade ML Platform API",
        "docs_url": getattr(settings, "DOCS_URL", "/docs"),
        "redoc_url": getattr(settings, "REDOC_URL", "/redoc"),
        "openapi_url": getattr(settings, "OPENAPI_URL", "/openapi.json"),
        "lifespan": lifespan
    }
    
    # Use ORJSON if available
    if _HAS_ORJSON:
        class ORJSONResponse(JSONResponse):
            media_type = "application/json"
            
            def render(self, content: Any) -> bytes:
                return _orjson_dumps(content, default=str)
        
        app_config["default_response_class"] = ORJSONResponse
    
    app = FastAPI(**app_config)
    
    # ─────────────────────────────────────────────────────────────────
    # Middleware Stack
    # ─────────────────────────────────────────────────────────────────
    
    # 1. CORS
    allow_origins = getattr(settings, "CORS_ALLOW_ORIGINS", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time"]
    )
    
    # 2. GZip Compression
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1024,
        compresslevel=6
    )
    
    # 3. Timing (optional, for monitoring)
    if getattr(settings, "ENABLE_TIMING_MIDDLEWARE", False):
        app.add_middleware(TimingMiddleware)
    
    # ─────────────────────────────────────────────────────────────────
    # Error Handlers
    # ─────────────────────────────────────────────────────────────────
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions with structured JSON."""
        payload = {
            "status": "error",
            "code": exc.status_code,
            "errors": [exc.detail] if isinstance(exc.detail, str) else exc.detail,
            "path": str(request.url.path)
        }
        
        logger.error(
            f"HTTP {exc.status_code}: {request.method} {request.url.path} - {exc.detail}"
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=payload
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle validation errors with detailed messages."""
        payload = {
            "status": "error",
            "code": 422,
            "errors": ["Validation error"],
            "details": json.loads(exc.json()),
            "path": str(request.url.path)
        }
        
        logger.warning(
            f"Validation error: {request.method} {request.url.path}"
        )
        
        return JSONResponse(
            status_code=422,
            content=payload
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        payload = {
            "status": "error",
            "code": 500,
            "errors": ["Internal server error"],
            "message": str(exc) if getattr(settings, "DEBUG", False) else "An unexpected error occurred",
            "path": str(request.url.path)
        }
        
        logger.exception(
            f"Unhandled exception: {request.method} {request.url.path}"
        )
        
        return JSONResponse(
            status_code=500,
            content=payload
        )
    
    # ─────────────────────────────────────────────────────────────────
    # Routes
    # ─────────────────────────────────────────────────────────────────
    
    # Mount API router
    app.include_router(api_router)
    
    # Root endpoint
    @app.get("/", tags=["root"], summary="API Information")
    async def root():
        """Get API information and available endpoints."""
        return {
            "service": getattr(settings, "APP_NAME", "DataGenius PRO API"),
            "version": getattr(settings, "APP_VERSION", __version__),
            "status": "operational",
            "documentation": {
                "swagger": app.docs_url,
                "redoc": app.redoc_url,
                "openapi": app.openapi_url
            },
            "endpoints": {
                "api": "/api",
                "health": "/health"
            }
        }
    
    # Health check endpoint
    @app.get("/health", tags=["health"], summary="Health Check")
    async def health_check():
        """Simple health check endpoint."""
        return {
            "status": "healthy",
            "version": __version__,
            "timestamp": time.time()
        }
    
    logger.info("✓ FastAPI application configured")
    
    return app


# ═══════════════════════════════════════════════════════════════════════════
# Uvicorn Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def get_app() -> FastAPI:
    """
    🚀 **Get FastAPI Application Instance**
    
    Entry point for ASGI servers (Uvicorn, Gunicorn).
    
    Usage:
        uvicorn backend.api:get_app --reload
    
    Returns:
        FastAPI application
    """
    return create_app()


# ═══════════════════════════════════════════════════════════════════════════
# Development Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.api:get_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )