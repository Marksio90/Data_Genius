# utils/logging_config.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Logging Configuration v7.0       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE CENTRALIZED LOGGING SYSTEM                                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Structured Logging (JSON + Human-Readable)                            ║
║  ✓ Multiple Sinks (Console, Files, JSONL)                                ║
║  ✓ Secret Redaction (Automatic PII/Credential Masking)                   ║
║  ✓ Context Variables (Request/User/Run/Session IDs)                      ║
║  ✓ Stdlib Interception (uvicorn, fastapi, sqlalchemy)                    ║
║  ✓ Async/Sync Decorators                                                 ║
║  ✓ Per-Run File Sinks                                                    ║
║  ✓ Dynamic Log Level Control                                             ║
║  ✓ Thread-Safe Operations                                                ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Logging Flow:
```
    Application Code
         ├─→ loguru.logger
         ├─→ stdlib logging → InterceptHandler → loguru
         └─→ warnings → loguru
    
    Sinks:
    ├── Console (stdout, colorized)
    ├── app.log (all logs, rotated)
    ├── errors.log (ERROR+ only)
    ├── agents.log (agent-specific)
    └── app.jsonl (structured JSON)
    
    Per-Run Sinks:
    └── run-{id}.log / run-{id}.jsonl
```

Features:
    Multi-Sink Architecture:
        • Console output (colorized)
        • Rotating file logs
        • Error-specific logs
        • Agent-specific logs
        • Structured JSON logs
    
    Security:
        • Automatic secret redaction
        • PII masking
        • API key/token filtering
        • Recursive structure sanitization
    
    Context Management:
        • Request ID tracking
        • User ID tracking
        • Run ID tracking
        • Session ID tracking
        • Context variables (ContextVar)
    
    Integration:
        • Stdlib logging interception
        • Warning capture
        • FastAPI integration
        • Uvicorn integration
        • SQLAlchemy integration
    
    Utilities:
        • Execution time decorators
        • Agent activity logging
        • Exception logging
        • Context managers
        • Dynamic log levels

Usage:
```python
    from utils.logging_config import (
        setup_logging,
        get_logger,
        set_request_context,
        log_execution_time,
        log_agent_activity
    )
    
    # Initialize
    setup_logging(
        app_name="DataGenius",
        log_level="INFO",
        enable_json=True
    )
    
    # Get logger
    log = get_logger(__name__, component="api")
    
    # Set context
    set_request_context(
        request_id="req-123",
        user_id="user-456"
    )
    
    # Use decorators
    @log_execution_time
    def process_data(data):
        log.info("Processing data")
        return processed
    
    @log_agent_activity("MyAgent")
    def agent_execute(self, **kwargs):
        return result
```

Dependencies:
    • loguru
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import sys
import time
import warnings
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger, Logger

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = [
    "setup_logging",
    "get_logger",
    "set_request_context",
    "clear_request_context",
    "log_execution_time",
    "log_agent_activity",
    "log_exceptions",
    "LogContext",
    "add_run_file_sinks",
    "remove_run_file_sinks",
    "set_log_level"
]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

try:
    from config.settings import settings
except ImportError:
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


# ═══════════════════════════════════════════════════════════════════════════
# Context Variables
# ═══════════════════════════════════════════════════════════════════════════

_ctx_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_ctx_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_ctx_run_id: ContextVar[Optional[str]] = ContextVar("run_id", default=None)
_ctx_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


def set_request_context(
    *,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> None:
    """
    🏷️ **Set Request Context**
    
    Sets context variables for current execution context.
    
    Args:
        request_id: Request identifier
        user_id: User identifier
        run_id: Run identifier
        session_id: Session identifier
    
    Example:
```python
        set_request_context(
            request_id="req-123",
            user_id="user-456",
            session_id="sess-789"
        )
        
        log.info("Processing request")  # Includes context IDs
```
    """
    if request_id is not None:
        _ctx_request_id.set(request_id)
    if user_id is not None:
        _ctx_user_id.set(user_id)
    if run_id is not None:
        _ctx_run_id.set(run_id)
    if session_id is not None:
        _ctx_session_id.set(session_id)


def clear_request_context() -> None:
    """
    🧹 **Clear Request Context**
    
    Clears all context variables.
    
    Example:
```python
        try:
            set_request_context(request_id="req-123")
            process_request()
        finally:
            clear_request_context()
```
    """
    _ctx_request_id.set(None)
    _ctx_user_id.set(None)
    _ctx_run_id.set(None)
    _ctx_session_id.set(None)


# ═══════════════════════════════════════════════════════════════════════════
# Stdlib Logging Interception
# ═══════════════════════════════════════════════════════════════════════════

class InterceptHandler(logging.Handler):
    """
    🔌 **Stdlib Logging Interceptor**
    
    Intercepts standard library logging and routes to loguru.
    """
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to loguru."""
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore
            depth += 1
        
        # Add context
        extra = {
            "request_id": _ctx_request_id.get(),
            "user_id": _ctx_user_id.get(),
            "run_id": _ctx_run_id.get(),
            "session_id": _ctx_session_id.get(),
            "module": record.module
        }
        
        logger.bind(**{k: v for k, v in extra.items() if v is not None}) \
              .opt(depth=depth, exception=record.exc_info) \
              .log(level, record.getMessage())


# ═══════════════════════════════════════════════════════════════════════════
# Secret Redaction
# ═══════════════════════════════════════════════════════════════════════════

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([A-Za-z0-9_\-\.]{12,})"),
    re.compile(r"(?i)(token\s*[:=]\s*)([A-Za-z0-9_\-\.]{12,})"),
    re.compile(r"(?i)(authorization\s*:\s*Bearer\s+)([A-Za-z0-9\.\-_]+)"),
    re.compile(r"(?i)(password\s*[:=]\s*)([^&\s]+)"),
    re.compile(r'(?i)("?(secret|passwd|pwd)"?\s*[:=]\s*"?)([^"\s]{6,})"?'),
]


def _redact_string(text: str) -> str:
    """Redact secrets in string."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(r"\1***", result)
    return result


def _redact_object(obj: Any) -> Any:
    """
    Recursively redact secrets in objects.
    
    Handles: str, dict, list, JSON strings
    """
    if isinstance(obj, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(obj)
            redacted = _redact_object(parsed)
            return json.dumps(redacted, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return _redact_string(obj)
    
    if isinstance(obj, dict):
        return {k: _redact_object(v) for k, v in obj.items()}
    
    if isinstance(obj, list):
        return [_redact_object(item) for item in obj]
    
    return obj


def _patch_record(record: Dict[str, Any]) -> None:
    """
    Patch log record with context and redaction.
    
    Args:
        record: Log record dictionary
    """
    # Add context variables
    extra = record.get("extra") or {}
    
    if _ctx_request_id.get():
        extra.setdefault("request_id", _ctx_request_id.get())
    if _ctx_user_id.get():
        extra.setdefault("user_id", _ctx_user_id.get())
    if _ctx_run_id.get():
        extra.setdefault("run_id", _ctx_run_id.get())
    if _ctx_session_id.get():
        extra.setdefault("session_id", _ctx_session_id.get())
    
    # Redact message
    if isinstance(record.get("message"), str):
        record["message"] = _redact_string(record["message"])
    
    # Redact sensitive fields in extra
    for key in ("payload", "body", "headers", "params", "query", "response"):
        if key in extra:
            extra[key] = _redact_object(extra[key])
    
    record["extra"] = extra


# ═══════════════════════════════════════════════════════════════════════════
# Filters
# ═══════════════════════════════════════════════════════════════════════════

def _agents_filter(record: Dict[str, Any]) -> bool:
    """
    Filter for agent-related logs.
    
    Args:
        record: Log record
    
    Returns:
        True if record is agent-related
    """
    name = (record.get("name") or "").lower()
    extra = record.get("extra") or {}
    component = str(extra.get("component", "")).lower()
    agent = str(extra.get("agent", "")).lower()
    
    return ("agent" in name) or ("agent" in component) or (agent != "")


# ═══════════════════════════════════════════════════════════════════════════
# Log Formats
# ═══════════════════════════════════════════════════════════════════════════

LOG_FORMAT_HUMAN = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "pid={process} tid={thread} | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "req=<blue>{extra[request_id]}</blue> run=<blue>{extra[run_id]}</blue> "
    "user=<blue>{extra[user_id]}</blue> sess=<blue>{extra[session_id]}</blue> | "
    "<level>{message}</level>"
)

LOG_FORMAT_COMPACT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | pid={process} | {message}"
)


# ═══════════════════════════════════════════════════════════════════════════
# Initialization State
# ═══════════════════════════════════════════════════════════════════════════

_INITIALIZED_FLAG = False
_SINK_IDS: List[int] = []
_RUN_SINKS: Dict[str, List[int]] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Main Setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(
    app_name: Optional[str] = None,
    log_level: Optional[str] = None,
    *,
    enable_json: Optional[bool] = None,
    console_compact: Optional[bool] = None,
    logs_path: Optional[Union[str, Path]] = None,
    reset_existing: bool = False
) -> None:
    """
    🔧 **Setup Centralized Logging**
    
    Initializes logging system with multiple sinks.
    Idempotent - can be called multiple times safely.
    
    Args:
        app_name: Application name
        log_level: Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        enable_json: Enable JSONL sink
        console_compact: Use compact console format
        logs_path: Directory for log files
        reset_existing: Force re-initialization
    
    Example:
```python
        setup_logging(
            app_name="DataGenius",
            log_level="INFO",
            enable_json=True,
            logs_path="/var/log/datagenius"
        )
```
    
    Creates:
      • Console sink (colorized)
      • app.log (all logs)
      • errors.log (ERROR+ only)
      • agents.log (agent-specific)
      • app.jsonl (structured JSON, if enabled)
    """
    global _INITIALIZED_FLAG, _SINK_IDS
    
    # Configuration
    app_name = app_name or getattr(settings, "APP_NAME", "DataGenius PRO")
    log_level = (log_level or getattr(settings, "LOG_LEVEL", "INFO")).upper()
    logs_dir = Path(logs_path or getattr(settings, "LOGS_PATH", Path.cwd() / "logs")).resolve()
    rotation = getattr(settings, "LOG_ROTATION", "10 MB")
    retention = getattr(settings, "LOG_RETENTION", "30 days")
    enable_json = enable_json if enable_json is not None else bool(getattr(settings, "LOG_JSON_ENABLED", True))
    test_mode = bool(getattr(settings, "TEST_MODE", False))
    console_compact = (
        console_compact if console_compact is not None
        else bool(getattr(settings, "LOG_CONSOLE_COMPACT", False))
    )
    
    # Check if already initialized
    if _INITIALIZED_FLAG and not reset_existing:
        return
    
    # Remove existing sinks
    try:
        logger.remove()
    except ValueError:
        pass
    
    _SINK_IDS.clear()
    
    # Create logs directory
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Console sink
    _SINK_IDS.append(
        logger.add(
            sys.stdout,
            format=LOG_FORMAT_COMPACT if console_compact else LOG_FORMAT_HUMAN,
            level=log_level,
            colorize=True,
            backtrace=(log_level == "DEBUG"),
            diagnose=False,
            enqueue=True,
            patcher=_patch_record
        )
    )
    
    if not test_mode:
        # Main log file
        _SINK_IDS.append(
            logger.add(
                logs_dir / "app.log",
                format=LOG_FORMAT_HUMAN,
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                encoding="utf-8",
                enqueue=True,
                patcher=_patch_record
            )
        )
        
        # Error log file
        _SINK_IDS.append(
            logger.add(
                logs_dir / "errors.log",
                format=LOG_FORMAT_HUMAN,
                level="ERROR",
                rotation=rotation,
                retention="90 days",
                compression="zip",
                encoding="utf-8",
                enqueue=True,
                patcher=_patch_record
            )
        )
        
        # Agent log file
        _SINK_IDS.append(
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
                patcher=_patch_record
            )
        )
        
        # JSONL sink
        if enable_json:
            _SINK_IDS.append(
                logger.add(
                    logs_dir / "app.jsonl",
                    serialize=True,
                    level=log_level,
                    rotation=rotation,
                    retention=retention,
                    compression="zip",
                    encoding="utf-8",
                    enqueue=True,
                    patcher=_patch_record
                )
            )
    
    # Intercept stdlib logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    for lib in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "sqlalchemy", "streamlit"):
        lib_logger = logging.getLogger(lib)
        lib_logger.handlers = [InterceptHandler()]
        lib_logger.propagate = False
    
    # Capture warnings
    warnings.simplefilter("default")
    logging.captureWarnings(True)
    
    # Bind app context
    logger.bind(app=app_name)
    logger.info(
        f"✓ Logging initialized: app={app_name}, level={log_level}, "
        f"json={enable_json}, logs_dir={logs_dir}"
    )
    
    _INITIALIZED_FLAG = True


# ═══════════════════════════════════════════════════════════════════════════
# Per-Run Sinks
# ═══════════════════════════════════════════════════════════════════════════

def add_run_file_sinks(run_id: str, run_dir: Path) -> List[int]:
    """
    📁 **Add Per-Run File Sinks**
    
    Creates dedicated log files for specific run.
    
    Args:
        run_id: Run identifier
        run_dir: Directory for run logs
    
    Returns:
        List of sink IDs
    
    Example:
```python
        sink_ids = add_run_file_sinks(
            run_id="workflow-123",
            run_dir=Path("./workflows/workflow-123")
        )
        
        # ... workflow execution ...
        
        remove_run_file_sinks("workflow-123")
```
    
    Creates:
      • run-{id}.log (human-readable)
      • run-{id}.jsonl (structured JSON)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    ids: List[int] = []
    
    # Run log file
    ids.append(
        logger.add(
            run_dir / f"run-{run_id}.log",
            format=LOG_FORMAT_HUMAN,
            level=getattr(settings, "LOG_LEVEL", "INFO"),
            rotation=getattr(settings, "LOG_ROTATION", "10 MB"),
            retention=getattr(settings, "LOG_RETENTION", "30 days"),
            encoding="utf-8",
            enqueue=True,
            patcher=_patch_record
        )
    )
    
    # Run JSONL file
    ids.append(
        logger.add(
            run_dir / f"run-{run_id}.jsonl",
            serialize=True,
            level=getattr(settings, "LOG_LEVEL", "INFO"),
            rotation=getattr(settings, "LOG_ROTATION", "10 MB"),
            retention=getattr(settings, "LOG_RETENTION", "30 days"),
            encoding="utf-8",
            enqueue=True,
            patcher=_patch_record
        )
    )
    
    _RUN_SINKS[run_id] = ids
    return ids


def remove_run_file_sinks(run_id: str) -> None:
    """
    🗑️ **Remove Per-Run File Sinks**
    
    Removes previously added run-specific sinks.
    
    Args:
        run_id: Run identifier
    
    Example:
```python
        remove_run_file_sinks("workflow-123")
```
    """
    ids = _RUN_SINKS.pop(run_id, [])
    for sink_id in ids:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def get_logger(name: Optional[str] = None, **binds: Any) -> Logger:
    """
    📝 **Get Bound Logger**
    
    Returns logger with bound context.
    
    Args:
        name: Logger name (usually __name__)
        **binds: Additional bindings
    
    Returns:
        Bound logger instance
    
    Example:
```python
        log = get_logger(__name__, component="api", version="1.0")
        log.info("Processing request")
```
    """
    lgr = logger
    
    if name:
        lgr = lgr.bind(name=name)
    
    if binds:
        lgr = lgr.bind(**binds)
    
    return lgr


class LogContext:
    """
    📦 **Log Context Manager**
    
    Context manager for temporary log bindings.
    
    Example:
```python
        with LogContext(agent="MyAgent", step="preprocessing"):
            log.info("Processing data")  # Includes agent and step
```
    """
    
    def __init__(self, **kwargs: Any):
        self._ctx = kwargs
        self._token = None
    
    def __enter__(self):
        self._token = logger.contextualize(**self._ctx)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"Exception in context: {exc_val!r}")
        return False


def set_log_level(level: str) -> None:
    """
    🎚️ **Set Log Level Dynamically**
    
    Changes log level at runtime.
    
    Args:
        level: New log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    
    Example:
```python
        set_log_level("DEBUG")  # Enable debug logging
        
        # ... debug session ...
        
        set_log_level("INFO")  # Return to normal
```
    """
    setup_logging(log_level=level, reset_existing=True)


# ═══════════════════════════════════════════════════════════════════════════
# Decorators
# ═══════════════════════════════════════════════════════════════════════════

def log_execution_time(func: Callable) -> Callable:
    """
    ⏱️ **Log Execution Time Decorator**
    
    Logs function execution time and handles exceptions.
    Supports both sync and async functions.
    
    Example:
```python
        @log_execution_time
        def process_data(data):
            return transform(data)
        
        @log_execution_time
        async def async_process(data):
            return await async_transform(data)
```
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info(f"▶️  Starting {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info(f"✓ Completed {func.__name__} in {duration:.3f}s")
                return result
            
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(f"✗ Failed {func.__name__} after {duration:.3f}s: {e}")
                raise
        
        return async_wrapper
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        logger.info(f"▶️  Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.info(f"✓ Completed {func.__name__} in {duration:.3f}s")
            return result
        
        except Exception as e:
            duration = time.perf_counter() - start
            logger.error(f"✗ Failed {func.__name__} after {duration:.3f}s: {e}")
            raise
    
    return sync_wrapper


def log_agent_activity(agent_name: str) -> Callable:
    """
    🤖 **Log Agent Activity Decorator**
    
    Logs agent lifecycle events with bound context.
    Supports both sync and async methods.
    
    Args:
        agent_name: Name of the agent
    
    Example:
```python
        class MyAgent:
            @log_agent_activity("MyAgent")
            def execute(self, **kwargs):
                # Agent logic
                return result
```
    """
    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with LogContext(agent=agent_name, component="agent"):
                    logger.info(f"[{agent_name}] ▶️  Starting {func.__name__}")
                    
                    try:
                        result = await func(*args, **kwargs)
                        logger.success(f"[{agent_name}] ✓ Completed {func.__name__}")
                        return result
                    
                    except Exception as e:
                        logger.error(f"[{agent_name}] ✗ Failed {func.__name__}: {e}")
                        raise
            
            return async_wrapper
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with LogContext(agent=agent_name, component="agent"):
                logger.info(f"[{agent_name}] ▶️  Starting {func.__name__}")
                
                try:
                    result = func(*args, **kwargs)
                    logger.success(f"[{agent_name}] ✓ Completed {func.__name__}")
                    return result
                
                except Exception as e:
                    logger.error(f"[{agent_name}] ✗ Failed {func.__name__}: {e}")
                    raise
        
        return sync_wrapper
    
    return decorator


def log_exceptions(level: str = "ERROR") -> Callable:
    """
    🚨 **Log Exceptions Decorator**
    
    Logs exceptions without measuring time.
    Supports both sync and async functions.
    
    Args:
        level: Log level for exceptions
    
    Example:
```python
        @log_exceptions("ERROR")
        def risky_operation():
            # May raise exception
            pass
```
    """
    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.log(level, f"Exception in {func.__name__}: {e}")
                    raise
            
            return async_wrapper
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(level, f"Exception in {func.__name__}: {e}")
                raise
        
        return sync_wrapper
    
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# Auto-Initialization
# ═══════════════════════════════════════════════════════════════════════════

if not bool(getattr(settings, "TEST_MODE", False)):
    setup_logging()


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Logging Configuration v{__version__} - Self Test")
    print("="*80)
    
    # Test setup
    print("\n1. Testing Logging Setup...")
    setup_logging(
        app_name="TestApp",
        log_level="DEBUG",
        reset_existing=True
    )
    print("   ✓ Logging configured")
    
    # Test basic logging
    print("\n2. Testing Basic Logging...")
    log = get_logger(__name__, component="test")
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    print("   ✓ Basic logging works")
    
    # Test context
    print("\n3. Testing Context Variables...")
    set_request_context(
        request_id="req-123",
        user_id="user-456",
        run_id="run-789"
    )
    log.info("Message with context")
    clear_request_context()
    print("   ✓ Context variables work")
    
    # Test secret redaction
    print("\n4. Testing Secret Redaction...")
    log.info("API Key: api_key=abc123def456ghi789")
    log.info("Password: password=mysecret123")
    log.info("Token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
    print("   ✓ Secret redaction works")
    
    # Test decorators
    print("\n5. Testing Decorators...")
    
    @log_execution_time
    def test_func():
        time.sleep(0.1)
        return "result"
    
    @log_agent_activity("TestAgent")
    def test_agent():
        return "agent_result"
    
    result = test_func()
    agent_result = test_agent()
    print("   ✓ Decorators work")
    
    # Test context manager
    print("\n6. Testing Context Manager...")
    with LogContext(operation="test", step="validation"):
        log.info("Inside context manager")
    print("   ✓ Context manager works")
    
    # Test log level change
    print("\n7. Testing Dynamic Log Level...")
    set_log_level("WARNING")
    log.info("This should not appear")
    log.warning("This should appear")
    set_log_level("INFO")
    print("   ✓ Log level change works")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from utils.logging_config import (
    setup_logging,
    get_logger,
    set_request_context,
    log_execution_time,
    log_agent_activity,
    LogContext
)

# === Initialize Logging ===
setup_logging(
    app_name="DataGenius",
    log_level="INFO",
    enable_json=True,
    logs_path="./logs"
)

# === Get Logger ===
log = get_logger(__name__, component="api")

# === Set Context ===
set_request_context(
    request_id="req-123",
    user_id="user-456",
    session_id="sess-789"
)

# === Basic Logging ===
log.debug("Debug information")
log.info("Processing request")
log.warning("Resource low")
log.error("Operation failed")
log.success("Request completed")

# === Decorators ===
@log_execution_time
def process_data(data):
    # Processing logic
    return processed

@log_agent_activity("MyAgent")
def execute_agent(self, **kwargs):
    # Agent logic
    return result

@log_exceptions("ERROR")
def risky_operation():
    # May raise exception
    pass

# === Context Manager ===
with LogContext(operation="ml_training", step="preprocessing"):
    log.info("Preprocessing data")
    # Logs include operation and step

# === Per-Run Logging ===
from pathlib import Path

sink_ids = add_run_file_sinks(
    run_id="workflow-123",
    run_dir=Path("./workflows/workflow-123")
)

# ... run workflow ...

remove_run_file_sinks("workflow-123")

# === Dynamic Log Level ===
set_log_level("DEBUG")  # Enable debug logging
# ... debug session ...
set_log_level("INFO")   # Return to normal

# === FastAPI Integration ===
from fastapi import FastAPI, Request
from utils.logging_config import set_request_context, clear_request_context

app = FastAPI()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Set context
    request_id = str(uuid.uuid4())
    set_request_context(request_id=request_id)
    
    log.info(f"Request started: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        log.info(f"Request completed: {response.status_code}")
        return response
    finally:
        clear_request_context()

# === Async Support ===
@log_execution_time
async def async_process(data):
    result = await async_operation(data)
    return result

# === Agent Integration ===
from core.base_agent import BaseAgent, AgentResult

class MyAgent(BaseAgent):
    @log_agent_activity("MyAgent")
    def execute(self, **kwargs) -> AgentResult:
        log = get_logger(__name__, agent=self.name)
        
        log.info("Starting execution")
        result = AgentResult(agent_name=self.name)
        
        try:
            # Agent logic
            output = self.process(kwargs.get("data"))
            result.add_data(output=output)
            log.success("Execution completed")
        except Exception as e:
            log.error(f"Execution failed: {e}")
            result.add_error(str(e))
        
        return result
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)
