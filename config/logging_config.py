# === logging_config.py ===
"""
DataGenius PRO - Logging Configuration (PRO+++)
Centralized logging setup with structured logging, redaction, and stdlib intercept.

Funkcje kluczowe:
- setup_logging(): konsola + pliki (app.log, errors.log, agents.log) + opcj. JSONL
- InterceptHandler: przekierowanie stdlib logging -> loguru
- Redakcja sekretów (API keys, Bearer tokens) przez patcher
- Dekoratory: log_execution_time, log_agent_activity, log_exceptions
- Kontekst: LogContext (contextualize) i get_logger(bind)
- add_run_file_sinks(): per-run logi (np. dla workflow_engine)
"""

from __future__ import annotations

import sys
import re
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Iterable, Tuple, List

from loguru import logger, Logger

# === Bezpieczny fallback dla settings ===
try:
    from config.settings import settings  # type: ignore
except Exception:  # pragma: no cover
    class _FallbackSettings:
        APP_NAME = "DataGenius PRO"
        LOG_LEVEL = "INFO"
        LOGS_PATH = Path.cwd() / "logs"
        TEST_MODE = False
        LOG_ROTATION = "10 MB"
        LOG_RETENTION = "30 days"
        LOG_JSON_ENABLED = True
    settings = _FallbackSettings()  # type: ignore


# === Intercept stdlib -> loguru ===
class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno

        # Ustal głębokość stosu, żeby wskazać prawdziwe miejsce wywołania
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# === Redakcja sekretów przez patcher ===
_SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key\s*=\s*)([A-Za-z0-9_\-\.]{12,})", re.IGNORECASE),
    re.compile(r"(token\s*=\s*)([A-Za-z0-9_\-\.]{12,})", re.IGNORECASE),
    re.compile(r"(authorization:\s*Bearer\s+)([A-Za-z0-9\.\-_]+)", re.IGNORECASE),
    re.compile(r"(password\s*=\s*)([^&\s]+)", re.IGNORECASE),
]

def _redact(message: str) -> str:
    red = message
    for pat in _SECRET_PATTERNS:
        red = pat.sub(r"\1***", red)
    return red

def _patch_record(record: Dict[str, Any]) -> None:
    # Redaguj message i extra (jeśli są stringami)
    msg = record.get("message")
    if isinstance(msg, str):
        record["message"] = _redact(msg)
    # Czasem w extra bywa 'payload' lub 'body'
    extra = record.get("extra") or {}
    for key in ("payload", "body", "headers"):
        val = extra.get(key)
        if isinstance(val, str):
            extra[key] = _redact(val)
    record["extra"] = extra


# === Filtry ===
def _agents_filter(rec: Dict[str, Any]) -> bool:
    """
    Filtruje wpisy powiązane z agentami (nazwy lub extra.component/agent).
    """
    name = (rec.get("name") or "").lower()
    extra = rec.get("extra") or {}
    comp = str(extra.get("component", "")).lower()
    agent = str(extra.get("agent", "")).lower()
    return ("agent" in name) or ("agent" in comp) or (agent != "")


# === Formatery ===
LOG_FORMAT_HUMAN = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>{extra}</magenta> | "
    "<level>{message}</level>"
)

LOG_FORMAT_COMPACT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)


# === Główna konfiguracja ===
def setup_logging(
    app_name: Optional[str] = None,
    log_level: Optional[str] = None,
    *,
    enable_json: Optional[bool] = None,
    console_compact: bool = False,
) -> None:
    """
    Inicjalizuje centralny system logowania.

    Args:
        app_name: Nazwa aplikacji (domyślnie settings.APP_NAME)
        log_level: Poziom logowania (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        enable_json: Włącza JSONL sink (domyślnie settings.LOG_JSON_ENABLED=True)
        console_compact: Uproszczony format konsoli
    """
    app_name = app_name or getattr(settings, "APP_NAME", "DataGenius PRO")
    log_level = log_level or getattr(settings, "LOG_LEVEL", "INFO")
    logs_path: Path = Path(getattr(settings, "LOGS_PATH", Path.cwd() / "logs"))
    rotation = getattr(settings, "LOG_ROTATION", "10 MB")
    retention = getattr(settings, "LOG_RETENTION", "30 days")
    enable_json = enable_json if enable_json is not None else bool(getattr(settings, "LOG_JSON_ENABLED", True))
    test_mode = bool(getattr(settings, "TEST_MODE", False))

    logs_path.mkdir(parents=True, exist_ok=True)

    # Usuń domyślne handlery
    logger.remove()

    # Konsola (kolorowa)
    logger.add(
        sys.stdout,
        format=LOG_FORMAT_COMPACT if console_compact else LOG_FORMAT_HUMAN,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=False,  # True tylko przy dev-debug
        enqueue=True,    # bezpieczne dla wątków/procesów
        patcher=_patch_record,
    )

    if not test_mode:
        # Plik główny
        logger.add(
            logs_path / "app.log",
            format=LOG_FORMAT_HUMAN,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            encoding="utf-8",
            enqueue=True,
            patcher=_patch_record,
        )

        # Plik błędów
        logger.add(
            logs_path / "errors.log",
            format=LOG_FORMAT_HUMAN,
            level="ERROR",
            rotation=rotation,
            retention="90 days",
            compression="zip",
            encoding="utf-8",
            enqueue=True,
            patcher=_patch_record,
        )

        # Plik agentów
        logger.add(
            logs_path / "agents.log",
            format=LOG_FORMAT_HUMAN,
            level="INFO",
            rotation=rotation,
            retention=retention,
            compression="zip",
            encoding="utf-8",
            enqueue=True,
            filter=_agents_filter,
            patcher=_patch_record,
        )

        # JSONL (strukturalny)
        if enable_json:
            logger.add(
                logs_path / "app.jsonl",
                serialize=True,   # JSON na każdą linię
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                encoding="utf-8",
                enqueue=True,
                patcher=_patch_record,
            )

    # Przechwyć stdlib logging i warnings
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for lib in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "sqlalchemy", "streamlit"):
        logging.getLogger(lib).handlers = [InterceptHandler()]
        logging.getLogger(lib).propagate = False

    warnings.simplefilter("default")
    logging.captureWarnings(True)

    # Flaga w extra, ułatwia filtrowanie
    logger.bind(app=app_name)
    logger.info(f"Logging initialized for {app_name} (level: {log_level}, json: {enable_json})")


# === Per-run sinki (np. workflow run) ===
def add_run_file_sinks(run_id: str, run_dir: Path) -> List[int]:
    """
    Dodaje dodatkowe sinki plikowe na czas konkretnego runu (np. workflow).
    Zwraca listę ID sinków (do ewentualnego usunięcia przez logger.remove(id)).

    Logi trafią do: <run_dir>/run.log oraz <run_dir>/run.jsonl (serialize).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    sink_ids: List[int] = []

    sink_ids.append(
        logger.add(
            run_dir / "run.log",
            format=LOG_FORMAT_HUMAN,
            level=getattr(settings, "LOG_LEVEL", "INFO"),
            rotation=getattr(settings, "LOG_ROTATION", "10 MB"),
            retention=getattr(settings, "LOG_RETENTION", "30 days"),
            encoding="utf-8",
            enqueue=True,
            patcher=_patch_record,
        )
    )
    sink_ids.append(
        logger.add(
            run_dir / "run.jsonl",
            serialize=True,
            level=getattr(settings, "LOG_LEVEL", "INFO"),
            rotation=getattr(settings, "LOG_ROTATION", "10 MB"),
            retention=getattr(settings, "LOG_RETENTION", "30 days"),
            encoding="utf-8",
            enqueue=True,
            patcher=_patch_record,
        )
    )
    return sink_ids


# === API pomocnicze ===
def get_logger(name: Optional[str] = None, **binds: Any) -> Logger:
    """
    Zwraca bindowany logger. Użycie:
        log = get_logger(__name__, component="agent", agent="EDAOrchestrator")
        log.info("Start")
    """
    lgr = logger
    if name:
        lgr = lgr.bind(name=name)
    if binds:
        lgr = lgr.bind(**binds)
    return lgr


class LogContext:
    """Context manager oparty o logger.contextualize(**kwargs)."""
    def __init__(self, **kwargs: Any) -> None:
        self._ctx = kwargs
        self._token = None

    def __enter__(self):
        self._token = logger.contextualize(**self._ctx)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # contextlib-like – zwolnienie contextu następuje automatycznie po wyjściu
        if exc_type:
            logger.error("Exception in context: {exc}", exc=repr(exc_val))
        return False


# === Dekoratory ===
def log_execution_time(func):
    """Loguje czas wykonania funkcji (INFO) z obsługą wyjątków (ERROR)."""
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        logger.info(f"Starting {func.__name__}")
        try:
            res = func(*args, **kwargs)
            dur = time.perf_counter() - start
            logger.info(f"Completed {func.__name__} in {dur:.3f}s")
            return res
        except Exception as e:
            dur = time.perf_counter() - start
            logger.error(f"Failed {func.__name__} after {dur:.3f}s: {e}")
            raise
    return wrapper


def log_agent_activity(agent_name: str):
    """Loguje aktywność agenta: start/success/failure z bindowaniem 'agent'."""
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            with LogContext(agent=agent_name, component="agent"):
                logger.info(f"[{agent_name}] Starting {func.__name__}")
                try:
                    res = func(*args, **kwargs)
                    logger.success(f"[{agent_name}] Completed {func.__name__}")
                    return res
                except Exception as e:
                    logger.error(f"[{agent_name}] Failed {func.__name__}: {e}")
                    raise
        return wrapper
    return decorator


def log_exceptions(level: str = "ERROR"):
    """Dekorator: loguje wyjątki (bez czasu wykonania)."""
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(level, f"Exception in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


# === Inicjalizacja przy imporcie (bez side-effectów w testach) ===
if not bool(getattr(settings, "TEST_MODE", False)):
    setup_logging()
