# backend/api/__init__.py
"""
DataGenius PRO - FastAPI App (PRO++++++)
A solid app factory with CORS, gzip, structured errors, Loguru logging,
and router mounting (see backend/api/routes.py).
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any, Dict, Iterable, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

try:
    # orjson przyspiesza JSON (opcjonalnie)
    import orjson  # type: ignore

    def _orjson_dumps(v: Any, *, default: Any) -> bytes:  # pragma: no cover
        return orjson.dumps(v, default=default)
except Exception:  # pragma: no cover
    orjson = None
    _orjson_dumps = None  # type: ignore

# lokalne importy
from config.settings import settings
from backend.api.routes import router as api_router  # <- twój router z /api/*


# ---- Loguru ↔ stdlib bridge (ładniejsze logi uvicorn) ------------------------
class InterceptHandler(logging.Handler):  # pragma: no cover
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        logger.bind(name=record.name).opt(exception=record.exc_info).log(level, record.getMessage())


def _configure_logging() -> None:  # pragma: no cover
    logging.getLogger().handlers = [InterceptHandler()]
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [InterceptHandler()]
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=getattr(settings, "LOG_LEVEL", "INFO"),
        backtrace=False,
        diagnose=False,
        colorize=True,
        enqueue=False,
    )


# ---- Lifespan: init/teardown zasobów (np. połączenia, cache) -----------------
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting DataGenius PRO …")
    # tu możesz zainicjalizować np. globalny feature store, cache, itp.
    try:
        yield
    finally:
        logger.info("🛑 Shutting down DataGenius PRO …")


# ---- App factory --------------------------------------------------------------
def create_app() -> FastAPI:
    _configure_logging()

    app = FastAPI(
        title=getattr(settings, "APP_NAME", "DataGenius PRO API"),
        version=getattr(settings, "APP_VERSION", "0.0.0"),
        docs_url=getattr(settings, "DOCS_URL", "/docs"),
        redoc_url=getattr(settings, "REDOC_URL", "/redoc"),
        openapi_url=getattr(settings, "OPENAPI_URL", "/openapi.json"),
        default_response_class=JSONResponse if orjson is None else type(
            "ORJSONResponse",
            (JSONResponse,),
            {"render": staticmethod(lambda c, *_, **__: _orjson_dumps(c, default=str))},  # type: ignore
        ),
        lifespan=lifespan,
    )

    # CORS
    allow_origins: Iterable[str] = getattr(settings, "CORS_ALLOW_ORIGINS", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # GZip (kompresja odpowiedzi)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # Mount router
    app.include_router(api_router)

    # Custom handlers → spójny JSON dla błędów
    @app.exception_handler(StarletteHTTPException)
    async def http_exc_handler(_: Request, exc: StarletteHTTPException):
        payload: Dict[str, Any] = {
            "status": "error",
            "errors": [exc.detail] if isinstance(exc.detail, str) else exc.detail,
        }
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def validation_exc_handler(_: Request, exc: RequestValidationError):
        payload = {
            "status": "error",
            "errors": ["Validation error"],
            "details": json.loads(exc.json()),
        }
        return JSONResponse(status_code=422, content=payload)

    @app.get("/", tags=["root"])
    async def root():
        return {
            "service": getattr(settings, "APP_NAME", "DataGenius PRO API"),
            "version": getattr(settings, "APP_VERSION", "0.0.0"),
            "docs": app.docs_url,
            "api_base": "/api",
        }

    return app


# Uvicorn entrypoint: `uvicorn backend.api:get_app`
def get_app() -> FastAPI:
    return create_app()


__all__ = ["create_app", "get_app"]
