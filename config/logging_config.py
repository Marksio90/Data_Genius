"""
DataGenius PRO - Logging Configuration
Centralized logging setup with structured logging support
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger
from config.settings import settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru"""
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    app_name: Optional[str] = None,
    log_level: Optional[str] = None
) -> None:
    """
    Setup centralized logging configuration
    
    Args:
        app_name: Application name for log identification
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    
    app_name = app_name or settings.APP_NAME
    log_level = log_level or settings.LOG_LEVEL
    
    # Remove default loguru handler
    logger.remove()
    
    # Log format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Console handler (colorized)
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # File handler - main log
    logger.add(
        settings.LOGS_PATH / "app.log",
        format=log_format,
        level=log_level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )
    
    # File handler - errors only
    logger.add(
        settings.LOGS_PATH / "errors.log",
        format=log_format,
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        compression="zip",
        encoding="utf-8",
    )
    
    # File handler - agents activity
    logger.add(
        settings.LOGS_PATH / "agents.log",
        format=log_format,
        level="INFO",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        filter=lambda record: "agent" in record["name"].lower()
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Configure third-party loggers
    for logger_name in [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "sqlalchemy",
        "streamlit",
    ]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
    
    logger.info(f"Logging initialized for {app_name} (level: {log_level})")


def get_logger(name: str) -> logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


class LoggerContext:
    """Context manager for structured logging"""
    
    def __init__(self, **kwargs):
        self.context = kwargs
    
    def __enter__(self):
        logger.bind(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(
                f"Exception in context: {exc_type.__name__}: {exc_val}",
                **self.context
            )


# Convenience decorators
def log_execution_time(func):
    """Decorator to log function execution time"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                f"Completed {func.__name__} in {execution_time:.2f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Failed {func.__name__} after {execution_time:.2f}s: {e}"
            )
            raise
    
    return wrapper


def log_agent_activity(agent_name: str):
    """Decorator to log agent activity"""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"[{agent_name}] Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                logger.success(f"[{agent_name}] Completed {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"[{agent_name}] Failed {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator


# Initialize logging on module import
if not settings.TEST_MODE:
    setup_logging()