# === logging_config.py ===
"""
DataGenius PRO - Logging Configuration (PRO++++++)
Centralny setup logowania: konsola, pliki, JSONL, redakcja sekretów, kontekst żądań
oraz przechwycenie stdlib logging do loguru.

Kluczowe:
- setup_logging(): konsola + app.log + errors.log + agents.log + (opcjonalnie) app.jsonl
- InterceptHandler: stdlib -> loguru (uvicorn/fastapi/sqlalchemy itp.)
- Redakcja sekretów w message i extra (rekurencyjnie)
- ContextVars: request_id/user_id/run_id i helpery do ustawiania
- Dekoratory: log_execution_time, log_agent_activity, log_exceptions
- add_run_file_sinks(): dodatkowe sinki dla pojedynczego “run”

Bezpieczny fallback na settings (bez side-effectów w testach).
"""

from __future__ import annotations

import sys
import re
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Iterable, Tuple, List, Union

from loguru import logger, Logger

# === Bezpieczny fallback dla settings ===
try:  # pragma: no cover
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
        LOG_CONSOLE_COMPACT = False
    settings = _FallbackSettings()  # type: ignore


# === Context variables (request/run) ===
from contextvars import ContextVar
_ctx_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_ctx_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_ctx_run_id: ContextVar[Optional[str]] = ContextVar("run_id", default=None)

def set_request_context(*, request_id: Optional[str] = None, user_id: Optional[str] = None, run_id: Optional[str] = None) -> None:
    if request_id is not None:
        _ctx_request_id.set(request_id)
    if user_id is not None:
        _ctx_user_id.set(user_id)
    if run_id is not None:
        _ctx_run_id.set(run_id)

def clear_request_context() -> None:
    _ctx_request_id.set(None)
    _ctx_user_id.set(None)
    _ctx_run_id.set(None)


# === Intercept stdlib -> loguru ===
class InterceptHandler(logging.Handler):
    """Przechwytuje standardowe logi i przekazuje do loguru."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno

        # Głębokość stosu: wskaż faktyczne źródło wywołania
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        # Dołącz kontekst żądania
        extra = {
            "request_id": _ctx_request_id.get(),
            "user_id": _ctx_user_id.get(),
            "run_id": _ctx_run_id.get(),
            "module": record.module,
        }
        logger.bind(**{k: v for k, v in extra.items() if v is not None}) \
              .opt(depth=depth, exception=record.exc_info) \
              .log(level, record.getMessage())


# === Redakcja sekretów ===
_SECRET_PATTERNS = [
    # key=XXXX lub "apiKey":"XXXX"
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([A-Za-z0-9_\-\.]{12,})"),
    re.compile(r"(?i)(token\s*[:=]\s*)([A-Za-z0-9_\-\.]{12,})"),
    re.compile(r"(?i)(authorization\s*:\s*Bearer\s+)([A-Za-z0-9\.\-_]+)"),
    re.compile(r"(?i)(password\s*[:=]\s*)([^&\s]+)"),
]

def _redact_str(message: str) -> str:
    red = message
    for pat in _SECRET_PATTERNS:
        red = pat.sub(r"\1***", red)
    return red

def _redact_obj(obj: Any) -> Any:
    # Rekurencyjna redakcja w strukturach
    if isinstance(obj, str):
        return _redact_str(obj)
    if isinstance(obj, dict):
        return {k: _redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_obj(v) for v in obj]
    return obj

def _patch_record(record: Dict[str, Any]) -> None:
    # Merged extra: dodaj context vars
    extra = record.get("extra") or {}
    if _ctx_request_id.get():
        extra.setdefault("request_id", _ctx_request_id.get())
    if _ctx_user_id.get():
        extra.setdefault("user_id", _ctx_user_id.get())
    if _ctx_run_id.get():
        extra.setdefault("run_id", _ctx_run_id.get())

    # Redakcja message/extra
    if isinstance(record.get("message"), str):
        record["message"] = _redact_str(record["message"])
    # typowe pola payload/headers/body + wszystko rekurencyjnie
    for key in ("payload", "body", "headers", "params", "query"):
        if key in extra:
            extra[key] = _redact_obj(extra[key])
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
    "pid={process} | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "req=<blue>{extra[request_id]}</blue> run=<blue>{extra[run_id]}</blue> user=<blue>{extra[user_id]}</blue> | "
    "<magenta>{extra}</magenta> | "
    "<level>{message}</level>"
)

LOG_FORMAT_COMPACT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | pid={process} | {message}"
)


# === Główna konfiguracja ===
def setup_logging(
    app_name: Optional[str] = None,
    log_level: Optional[str] = None,
    *,
    enable_json: Optional[bool] = None,
    console_compact: Optional[bool] = None,
    logs_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Inicjalizuje centralny system logowania.

    Args:
        app_name: Nazwa aplikacji (domyślnie settings.APP_NAME)
        log_level: Poziom logowania (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        enable_json: Włącza JSONL sink (domyślnie settings.LOG_JSON_ENABLED=True)
        console_compact: Uproszczony format w konsoli
        logs_path: Ścieżka do katalogu logów (domyślnie settings.LOGS_PATH)
    """
    app_name = app_name or getattr(settings, "APP_NAME", "DataGenius PRO")
    log_level = log_level or getattr(settings, "LOG_LEVEL", "INFO")
    logs_dir: Path = Path(logs_path or getattr(settings, "LOGS_PATH", Path.cwd() / "logs")).resolve()
    rotation = getattr(settings, "LOG_ROTATION", "10 MB")
    retention = getattr(settings, "LOG_RETENTION", "30 days")
    enable_json = enable_json if enable_json is not None else bool(getattr(settings, "LOG_JSON_ENABLED", True))
    test_mode = bool(getattr(settings, "TEST_MODE", False))
    console_compact = bool(getattr(settings, "LOG_CONSOLE_COMPACT", False)) if console_compact is None else console_compact

    logs_dir.mkdir(parents=True, exist_ok=True)

    # Usuń domyślne handlery
    logger.remove()

    # Konsola (kolorowa)
    logger.add(
        sys.stdout,
        format=LOG_FORMAT_COMPACT if console_compact else LOG_FORMAT_HUMAN,
        level=log_level,
        colorize=True,
        backtrace=log_level == "DEBUG",
        diagnose=False,   # True tylko przy głębokim debugowaniu
        enqueue=True,     # bezpieczne dla wątków/procesów
        patcher=_patch_record,
    )

    if not test_mode:
        # Plik główny
        logger.add(
            logs_dir / "app.log",
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
            logs_dir / "errors.log",
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
            logs_dir / "agents.log",
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
                logs_dir / "app.jsonl",
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
        lg = logging.getLogger(lib)
        lg.handlers = [InterceptHandler()]
        lg.propagate = False

    warnings.simplefilter("default")
    logging.captureWarnings(True)

    # Flaga/kontekst
    logger.bind(app=app_name)
    logger.info(f"Logging initialized for {app_name} (level={log_level}, json={enable_json}, logs_dir={str(logs_dir)})")


# === Per-run sinki (np. workflow run) ===
def add_run_file_sinks(run_id: str, run_dir: Path) -> List[int]:
    """
    Dodaje dodatkowe sinki plikowe na czas konkretnego runu (np. workflow).
    Zwraca listę ID sinków (do ewentualnego usunięcia przez logger.remove(id)).

    Tworzy:
      - <run_dir>/run-{run_id}.log
      - <run_dir>/run-{run_id}.jsonl
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    sink_ids: List[int] = []

    sink_ids.append(
        logger.add(
            run_dir / f"run-{run_id}.log",
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
            run_dir / f"run-{run_id}.jsonl",
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
    """Loguje aktywność agenta: start/sukces/porażka z bindowaniem 'agent'."""
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
    """Dekorator: loguje wyjątki (bez mierzenia czasu)."""
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


def set_log_level(level: str) -> None:
    """
    Zmienia poziom logowania w locie (dotyczy nowych wpisów; istniejące sinki zachowują filtry).
    """
    try:
        logger.remove()
    except Exception:
        pass
    # Re-setup konsoli z nowym poziomem i domyślną konfiguracją
    setup_logging(log_level=level)


# === Inicjalizacja przy imporcie (bez side-effectów w testach) ===
if not bool(getattr(settings, "TEST_MODE", False)):  # pragma: no cover
    setup_logging()


__all__ = [
    "setup_logging",
    "add_run_file_sinks",
    "get_logger",
    "LogContext",
    "log_execution_time",
    "log_agent_activity",
    "log_exceptions",
    "set_request_context",
    "clear_request_context",
    "set_log_level",
]
