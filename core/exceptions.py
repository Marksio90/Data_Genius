"""
DataGenius PRO - Custom Exceptions (PRO)
Centralized exception definitions + helpers
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Callable, Type, Iterable, ContextManager
from contextlib import contextmanager
from enum import Enum
from http import HTTPStatus

from loguru import logger

# Opcjonalna integracja z Sentry (jeśli włączona w settings)
try:
    from config.settings import settings  # type: ignore
    _SENTRY_ENABLED = bool(getattr(settings, "ENABLE_SENTRY", False) and getattr(settings, "SENTRY_DSN", None))
except Exception:
    settings = None  # type: ignore
    _SENTRY_ENABLED = False

try:
    import sentry_sdk  # type: ignore
except Exception:  # pragma: no cover
    sentry_sdk = None  # type: ignore


# =========================
# Error taxonomy
# =========================

class ErrorSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCode(str, Enum):
    UNKNOWN = "unknown_error"
    DATA_LOAD = "data_load_error"
    DATA_VALIDATION = "data_validation_error"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_TARGET = "invalid_target"
    MODEL_TRAINING = "model_training_error"
    MODEL_PREDICTION = "model_prediction_error"
    LLM = "llm_error"
    CONFIG = "configuration_error"
    AGENT = "agent_execution_error"
    PIPELINE = "pipeline_error"
    FEATURE_ENGINEERING = "feature_engineering_error"
    REPORT = "report_generation_error"
    DATABASE = "database_error"
    CACHE = "cache_error"
    MONITORING = "monitoring_error"


_HTTP_MAP: Dict[ErrorCode, HTTPStatus] = {
    ErrorCode.DATA_LOAD: HTTPStatus.BAD_REQUEST,
    ErrorCode.DATA_VALIDATION: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.INSUFFICIENT_DATA: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.INVALID_TARGET: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.MODEL_TRAINING: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.MODEL_PREDICTION: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.LLM: HTTPStatus.BAD_GATEWAY,
    ErrorCode.CONFIG: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.AGENT: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.PIPELINE: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.FEATURE_ENGINEERING: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.REPORT: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.DATABASE: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.CACHE: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.MONITORING: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.UNKNOWN: HTTPStatus.INTERNAL_SERVER_ERROR,
}


# =========================
# Exception classes
# =========================

class DataGeniusException(Exception):
    """Base exception for DataGenius PRO (rozszerzona)"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        *,
        error_code: ErrorCode = ErrorCode.UNKNOWN,
        status_code: Optional[int] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = details or {}
        self.error_code: ErrorCode = error_code
        self.status_code: int = int(status_code or _HTTP_MAP.get(error_code, HTTPStatus.INTERNAL_SERVER_ERROR))
        self.severity: ErrorSeverity = severity
        self.context: Dict[str, Any] = context or {}
        self.cause = cause

    def __str__(self) -> str:
        base = f"{self.error_code}: {self.message}"
        if self.details:
            base += f" | Details: {self.details}"
        if self.context:
            base += f" | Context: {self.context}"
        if self.cause:
            base += f" | Cause: {type(self.cause).__name__}: {self.cause}"
        return base

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details or {},
                "context": self.context or {},
                "severity": self.severity,
                "status_code": self.status_code,
                "cause": str(self.cause) if self.cause else None,
            }
        }

    @classmethod
    def from_exc(
        cls,
        exc: BaseException,
        *,
        default_code: ErrorCode = ErrorCode.UNKNOWN,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> "DataGeniusException":
        if isinstance(exc, DataGeniusException):
            return exc
        return cls(
            message or str(exc) or "Wystąpił nieoczekiwany błąd.",
            details=details,
            error_code=default_code,
            severity=severity,
            context=context,
            cause=exc,
        )


# Specyficzne wyjątki (zachowują kompatybilność nazw)
class DataLoadError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.DATA_LOAD, **kw)


class DataValidationError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.DATA_VALIDATION, **kw)


class InsufficientDataError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.INSUFFICIENT_DATA, **kw)


class InvalidTargetError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.INVALID_TARGET, **kw)


class ModelTrainingError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.MODEL_TRAINING, **kw)


class ModelPredictionError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.MODEL_PREDICTION, **kw)


class LLMError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.LLM, **kw)


class ConfigurationError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.CONFIG, **kw)


class AgentExecutionError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.AGENT, **kw)


class PipelineError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.PIPELINE, **kw)


class FeatureEngineeringError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.FEATURE_ENGINEERING, **kw)


class ReportGenerationError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.REPORT, **kw)


class DatabaseError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.DATABASE, **kw)


class CacheError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.CACHE, **kw)


class MonitoringError(DataGeniusException):
    def __init__(self, message: str, details: dict = None, **kw):
        super().__init__(message, details, error_code=ErrorCode.MONITORING, **kw)


# =========================
# Public helpers
# =========================

def handle_exception(e: Exception, context: str = "") -> str:
    """
    Format exception for user display (kompatybilne z poprzednią wersją)
    """
    if isinstance(e, DataGeniusException):
        msg = f"❌ **Błąd**: {e.message}"
        if context:
            msg += f"\n\n**Kontekst**: {context}"
        if e.details:
            msg += f"\n\n**Szczegóły**: {e.details}"
        return msg
    else:
        msg = f"❌ **Nieoczekiwany błąd**: {str(e)}"
        if context:
            msg += f"\n\n**Kontekst**: {context}"
        return msg


def exception_to_response(exc: DataGeniusException) -> Dict[str, Any]:
    """
    Zamiana wyjątku na słownik pod odpowiedź HTTP (np. w FastAPI).
    """
    payload = exc.to_dict()
    payload["status_code"] = exc.status_code
    return payload


def _maybe_capture_sentry(exc: DataGeniusException) -> None:
    if _SENTRY_ENABLED and sentry_sdk is not None:  # pragma: no cover
        try:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("error_code", exc.error_code)
                scope.set_tag("severity", exc.severity)
                for k, v in (exc.context or {}).items():
                    scope.set_extra(f"context_{k}", v)
                for k, v in (exc.details or {}).items():
                    scope.set_extra(f"details_{k}", v)
                sentry_sdk.capture_exception(exc.cause or exc)
        except Exception:
            # nie blokujemy przepływu w razie błędu w Sentry
            pass


def safe_execute(
    func: Callable[..., Any],
    *args,
    error_message: str = "Operacja nie powiodła się",
    exc_type: Type[DataGeniusException] = DataGeniusException,
    error_code: ErrorCode = ErrorCode.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    log: bool = True,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Execute function with exception handling.
    W razie błędu rzuca DataGeniusException (lub pochodną).
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        dg_exc = exc_type(
            error_message,
            details={"original_error": str(e)},
            error_code=error_code,
            severity=severity,
            context=context,
            cause=e,
        )
        if log:
            logger.error(str(dg_exc), exc_info=True)
        _maybe_capture_sentry(dg_exc)
        raise dg_exc


def wrap_exceptions(
    *,
    to: Type[DataGeniusException],
    message: str,
    error_code: ErrorCode,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    context_builder: Optional[Callable[[tuple, dict], Dict[str, Any]]] = None,
    log: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Dekorator: owija dowolny wyjątek w dedykowany DataGeniusException.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except DataGeniusException:
                # nie podwajamy
                raise
            except Exception as e:
                ctx = context_builder(*args, **kwargs) if context_builder else {}
                dg_exc = to(
                    message,
                    details={"original_error": str(e)},
                    error_code=error_code,
                    severity=severity,
                    context=ctx,
                    cause=e,
                )
                if log:
                    logger.error(str(dg_exc), exc_info=True)
                _maybe_capture_sentry(dg_exc)
                raise dg_exc
        return wrapper
    return decorator


@contextmanager
def exception_context(
    *,
    to: Type[DataGeniusException] = DataGeniusException,
    message: str = "Operacja nie powiodła się",
    error_code: ErrorCode = ErrorCode.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: Optional[Dict[str, Any]] = None,
    log: bool = True,
) -> ContextManager[None]:
    """
    Kontekst menedżer: każde odstępstwo zamienia w DataGeniusException.
    Idealny do krótkich bloków I/O itp.
    """
    try:
        yield
    except DataGeniusException:
        raise
    except Exception as e:
        dg_exc = to(
            message,
            details={"original_error": str(e)},
            error_code=error_code,
            severity=severity,
            context=context,
            cause=e,
        )
        if log:
            logger.error(str(dg_exc), exc_info=True)
        _maybe_capture_sentry(dg_exc)
        raise dg_exc


# =========================
# __all__ (porządek eksportów)
# =========================

__all__ = [
    # enums
    "ErrorCode", "ErrorSeverity",
    # base + specyficzne
    "DataGeniusException", "DataLoadError", "DataValidationError",
    "InsufficientDataError", "InvalidTargetError", "ModelTrainingError",
    "ModelPredictionError", "LLMError", "ConfigurationError",
    "AgentExecutionError", "PipelineError", "FeatureEngineeringError",
    "ReportGenerationError", "DatabaseError", "CacheError", "MonitoringError",
    # helpers
    "handle_exception", "safe_execute", "wrap_exceptions",
    "exception_context", "exception_to_response",
]
