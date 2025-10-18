# utils/exceptions.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Exceptions v7.0                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ULTIMATE EXCEPTION HANDLING SYSTEM                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Centralized Exception Hierarchy                                       ║
║  ✓ Error Code & Severity System                                          ║
║  ✓ HTTP Status Mapping                                                   ║
║  ✓ Context & Details Tracking                                            ║
║  ✓ Sentry Integration                                                    ║
║  ✓ Decorator & Context Manager                                           ║
║  ✓ Safe Execution Wrappers                                               ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Exception System Structure:
```
    DataGeniusException (Base)
    ├── ErrorCode (taxonomy)
    ├── ErrorSeverity (info/warning/error/critical)
    ├── HTTP Status Mapping
    ├── Context & Details
    └── Sentry Integration
    
    Specific Exceptions:
    ├── DataLoadError
    ├── DataValidationError
    ├── ModelTrainingError
    ├── LLMError
    ├── AgentExecutionError
    └── ... (14 total)
    
    Helpers:
    ├── safe_execute()
    ├── wrap_exceptions() decorator
    ├── exception_context() manager
    └── exception_to_response()
```

Features:
    Exception Hierarchy:
        • Base exception class
        • 14 specific exception types
        • Error code taxonomy
        • Severity levels
    
    Context Tracking:
        • Error details
        • Execution context
        • Original cause
        • Stack trace
    
    HTTP Integration:
        • Status code mapping
        • Response formatting
        • FastAPI compatible
    
    Sentry Integration:
        • Automatic capture
        • Context tagging
        • Severity tracking
        • Error grouping
    
    Safe Execution:
        • Decorator pattern
        • Context manager
        • Safe wrapper function
        • Automatic logging

Usage:
```python
    from utils.exceptions import (
        DataValidationError,
        safe_execute,
        wrap_exceptions,
        exception_context
    )
    
    # Raise specific exception
    raise DataValidationError(
        "Invalid data format",
        details={"column": "age", "issue": "negative values"}
    )
    
    # Safe execution
    result = safe_execute(
        risky_function,
        arg1, arg2,
        error_message="Operation failed",
        exc_type=DataLoadError
    )
    
    # Decorator
    @wrap_exceptions(
        to=ModelTrainingError,
        message="Training failed",
        error_code=ErrorCode.MODEL_TRAINING
    )
    def train_model(data):
        # Training logic
        pass
    
    # Context manager
    with exception_context(
        to=DataLoadError,
        message="Failed to load data"
    ):
        df = pd.read_csv("data.csv")
```

Dependencies:
    • loguru
    • sentry-sdk (optional)
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from http import HTTPStatus
from typing import Any, Callable, ContextManager, Dict, Optional, Type

from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "7.0-ultimate"
__author__ = "DataGenius Enterprise Team"

__all__ = [
    # Enums
    "ErrorCode",
    "ErrorSeverity",
    # Base Exception
    "DataGeniusException",
    # Specific Exceptions
    "DataLoadError",
    "DataValidationError",
    "InsufficientDataError",
    "InvalidTargetError",
    "ModelTrainingError",
    "ModelPredictionError",
    "LLMError",
    "ConfigurationError",
    "AgentExecutionError",
    "PipelineError",
    "FeatureEngineeringError",
    "ReportGenerationError",
    "DatabaseError",
    "CacheError",
    "MonitoringError",
    # Helpers
    "handle_exception",
    "safe_execute",
    "wrap_exceptions",
    "exception_context",
    "exception_to_response"
]


# ═══════════════════════════════════════════════════════════════════════════
# Sentry Configuration
# ═══════════════════════════════════════════════════════════════════════════

try:
    from config.settings import settings
    _SENTRY_ENABLED = bool(
        getattr(settings, "ENABLE_SENTRY", False) and
        getattr(settings, "SENTRY_DSN", None)
    )
except Exception:
    settings = None
    _SENTRY_ENABLED = False

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None


# ═══════════════════════════════════════════════════════════════════════════
# Error Taxonomy
# ═══════════════════════════════════════════════════════════════════════════

class ErrorSeverity(str, Enum):
    """
    🚨 **Error Severity Levels**
    
    Severity classification for errors.
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCode(str, Enum):
    """
    🏷️ **Error Code Taxonomy**
    
    Standardized error codes for categorization.
    """
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


# HTTP Status Code Mapping
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
    ErrorCode.UNKNOWN: HTTPStatus.INTERNAL_SERVER_ERROR
}


# ═══════════════════════════════════════════════════════════════════════════
# Base Exception
# ═══════════════════════════════════════════════════════════════════════════

class DataGeniusException(Exception):
    """
    🎯 **Base DataGenius Exception**
    
    Base exception class with rich context and metadata.
    
    Features:
      • Error code classification
      • Severity levels
      • HTTP status mapping
      • Context tracking
      • Details dictionary
      • Original cause tracking
      • Sentry integration
    
    Usage:
```python
        raise DataGeniusException(
            "Operation failed",
            details={"reason": "invalid input"},
            error_code=ErrorCode.DATA_VALIDATION,
            severity=ErrorSeverity.ERROR,
            context={"user_id": "123"}
        )
```
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        *,
        error_code: ErrorCode = ErrorCode.UNKNOWN,
        status_code: Optional[int] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            details: Additional details dictionary
            error_code: Error code classification
            status_code: HTTP status code (auto-mapped if None)
            severity: Error severity level
            context: Execution context dictionary
            cause: Original exception (if wrapping)
        """
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = details or {}
        self.error_code: ErrorCode = error_code
        self.status_code: int = int(
            status_code or
            _HTTP_MAP.get(error_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        )
        self.severity: ErrorSeverity = severity
        self.context: Dict[str, Any] = context or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """String representation with full context."""
        parts = [f"{self.error_code}: {self.message}"]
        
        if self.details:
            parts.append(f"Details: {self.details}")
        
        if self.context:
            parts.append(f"Context: {self.context}")
        
        if self.cause:
            parts.append(f"Cause: {type(self.cause).__name__}: {self.cause}")
        
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details or {},
                "context": self.context or {},
                "severity": self.severity,
                "status_code": self.status_code,
                "cause": str(self.cause) if self.cause else None
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
        severity: ErrorSeverity = ErrorSeverity.ERROR
    ) -> "DataGeniusException":
        """
        Create from existing exception.
        
        Args:
            exc: Original exception
            default_code: Default error code
            message: Override message
            details: Additional details
            context: Execution context
            severity: Error severity
        
        Returns:
            DataGeniusException instance
        """
        if isinstance(exc, DataGeniusException):
            return exc
        
        return cls(
            message or str(exc) or "An unexpected error occurred",
            details=details,
            error_code=default_code,
            severity=severity,
            context=context,
            cause=exc
        )


# ═══════════════════════════════════════════════════════════════════════════
# Specific Exception Classes
# ═══════════════════════════════════════════════════════════════════════════

class DataLoadError(DataGeniusException):
    """❌ Data loading error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.DATA_LOAD, **kwargs)


class DataValidationError(DataGeniusException):
    """⚠️ Data validation error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.DATA_VALIDATION, **kwargs)


class InsufficientDataError(DataGeniusException):
    """📉 Insufficient data error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.INSUFFICIENT_DATA, **kwargs)


class InvalidTargetError(DataGeniusException):
    """🎯 Invalid target column error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.INVALID_TARGET, **kwargs)


class ModelTrainingError(DataGeniusException):
    """🏋️ Model training error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.MODEL_TRAINING, **kwargs)


class ModelPredictionError(DataGeniusException):
    """🔮 Model prediction error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.MODEL_PREDICTION, **kwargs)


class LLMError(DataGeniusException):
    """🤖 LLM API error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.LLM, **kwargs)


class ConfigurationError(DataGeniusException):
    """⚙️ Configuration error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.CONFIG, **kwargs)


class AgentExecutionError(DataGeniusException):
    """🤖 Agent execution error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.AGENT, **kwargs)


class PipelineError(DataGeniusException):
    """🔄 Pipeline execution error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.PIPELINE, **kwargs)


class FeatureEngineeringError(DataGeniusException):
    """🔧 Feature engineering error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.FEATURE_ENGINEERING, **kwargs)


class ReportGenerationError(DataGeniusException):
    """📊 Report generation error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.REPORT, **kwargs)


class DatabaseError(DataGeniusException):
    """🗄️ Database error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.DATABASE, **kwargs)


class CacheError(DataGeniusException):
    """💾 Cache error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.CACHE, **kwargs)


class MonitoringError(DataGeniusException):
    """📈 Monitoring error."""
    def __init__(self, message: str, details: Optional[Dict] = None, **kwargs):
        super().__init__(message, details, error_code=ErrorCode.MONITORING, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def handle_exception(e: Exception, context: str = "") -> str:
    """
    📝 **Format Exception for Display**
    
    Formats exception for user-friendly display.
    
    Args:
        e: Exception to format
        context: Additional context
    
    Returns:
        Formatted message string
    
    Example:
```python
        try:
            risky_operation()
        except Exception as e:
            message = handle_exception(e, "Data processing")
            print(message)
```
    """
    if isinstance(e, DataGeniusException):
        msg = f"❌ **Error**: {e.message}"
        
        if context:
            msg += f"\n\n**Context**: {context}"
        
        if e.details:
            msg += f"\n\n**Details**: {e.details}"
        
        return msg
    else:
        msg = f"❌ **Unexpected Error**: {str(e)}"
        
        if context:
            msg += f"\n\n**Context**: {context}"
        
        return msg


def exception_to_response(exc: DataGeniusException) -> Dict[str, Any]:
    """
    🔄 **Convert Exception to HTTP Response**
    
    Converts exception to dictionary for HTTP response.
    
    Args:
        exc: DataGenius exception
    
    Returns:
        Response dictionary with status_code
    
    Example:
```python
        try:
            process_data()
        except DataGeniusException as e:
            response = exception_to_response(e)
            return JSONResponse(
                content=response,
                status_code=response["status_code"]
            )
```
    """
    payload = exc.to_dict()
    payload["status_code"] = exc.status_code
    return payload


def _maybe_capture_sentry(exc: DataGeniusException) -> None:
    """Capture exception in Sentry if enabled."""
    if _SENTRY_ENABLED and sentry_sdk is not None:
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
            pass


def safe_execute(
    func: Callable[..., Any],
    *args,
    error_message: str = "Operation failed",
    exc_type: Type[DataGeniusException] = DataGeniusException,
    error_code: ErrorCode = ErrorCode.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    log: bool = True,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    🛡️ **Safe Function Execution**
    
    Executes function with automatic exception handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        error_message: Error message on failure
        exc_type: Exception type to raise
        error_code: Error code
        severity: Error severity
        log: Log error
        context: Execution context
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    
    Raises:
        DataGeniusException: On execution failure
    
    Example:
```python
        result = safe_execute(
            load_data,
            "data.csv",
            error_message="Failed to load data",
            exc_type=DataLoadError,
            error_code=ErrorCode.DATA_LOAD
        )
```
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
            cause=e
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
    log: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    🎁 **Exception Wrapping Decorator**
    
    Decorator to wrap exceptions in DataGeniusException.
    
    Args:
        to: Exception type to wrap to
        message: Error message
        error_code: Error code
        severity: Error severity
        context_builder: Function to build context from args/kwargs
        log: Log errors
    
    Returns:
        Decorator function
    
    Example:
```python
        @wrap_exceptions(
            to=ModelTrainingError,
            message="Model training failed",
            error_code=ErrorCode.MODEL_TRAINING
        )
        def train_model(data, model_name):
            # Training logic
            pass
```
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except DataGeniusException:
                raise
            except Exception as e:
                ctx = context_builder(args, kwargs) if context_builder else {}
                
                dg_exc = to(
                    message,
                    details={"original_error": str(e)},
                    error_code=error_code,
                    severity=severity,
                    context=ctx,
                    cause=e
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
    message: str = "Operation failed",
    error_code: ErrorCode = ErrorCode.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: Optional[Dict[str, Any]] = None,
    log: bool = True
) -> ContextManager[None]:
    """
    🔒 **Exception Context Manager**
    
    Context manager to wrap exceptions.
    
    Args:
        to: Exception type to wrap to
        message: Error message
        error_code: Error code
        severity: Error severity
        context: Execution context
        log: Log errors
    
    Yields:
        None
    
    Example:
```python
        with exception_context(
            to=DataLoadError,
            message="Failed to load data",
            error_code=ErrorCode.DATA_LOAD
        ):
            df = pd.read_csv("data.csv")
```
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
            cause=e
        )
        
        if log:
            logger.error(str(dg_exc), exc_info=True)
        
        _maybe_capture_sentry(dg_exc)
        raise dg_exc


# ═══════════════════════════════════════════════════════════════════════════
# Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print(f"Exceptions v{__version__} - Self Test")
    print("="*80)
    
    # Test base exception
    print("\n1. Testing Base Exception...")
    try:
        raise DataGeniusException(
            "Test error",
            details={"key": "value"},
            error_code=ErrorCode.UNKNOWN,
            context={"user": "test"}
        )
    except DataGeniusException as e:
        assert e.message == "Test error"
        assert e.error_code == ErrorCode.UNKNOWN
        print("   ✓ Base exception works")
    
    # Test specific exceptions
    print("\n2. Testing Specific Exceptions...")
    exceptions = [
        (DataLoadError, "Load failed"),
        (DataValidationError, "Validation failed"),
        (ModelTrainingError, "Training failed")
    ]
    
    for exc_class, msg in exceptions:
        try:
            raise exc_class(msg)
        except DataGeniusException as e:
            assert e.message == msg
    
    print("   ✓ Specific exceptions work")
    
    # Test safe_execute
    print("\n3. Testing safe_execute...")
    
    def failing_function():
        raise ValueError("Test error")
    
    try:
        safe_execute(
            failing_function,
            error_message="Execution failed",
            exc_type=DataLoadError,
            log=False
        )
    except DataLoadError as e:
        assert "Execution failed" in str(e)
        print("   ✓ safe_execute works")
    
    # Test decorator
    print("\n4. Testing wrap_exceptions...")
    
    @wrap_exceptions(
        to=ModelTrainingError,
        message="Training failed",
        error_code=ErrorCode.MODEL_TRAINING,
        log=False
    )
    def train_model():
        raise ValueError("Model error")
    
    try:
        train_model()
    except ModelTrainingError as e:
        assert "Training failed" in str(e)
        print("   ✓ wrap_exceptions decorator works")
    
    # Test context manager
    print("\n5. Testing exception_context...")
    
    try:
        with exception_context(
            to=DataLoadError,
            message="Load failed",
            log=False
        ):
            raise ValueError("Context error")
    except DataLoadError as e:
        assert "Load failed" in str(e)
        print("   ✓ exception_context works")
    
    # Test to_dict
    print("\n6. Testing to_dict...")
    exc = DataGeniusException(
        "Test",
        details={"a": 1},
        context={"b": 2}
    )
    d = exc.to_dict()
    assert "error" in d
    assert d["error"]["message"] == "Test"
    print("   ✓ to_dict works")
    
    # Test exception_to_response
    print("\n7. Testing exception_to_response...")
    exc = DataLoadError("Load failed")
    response = exception_to_response(exc)
    assert "status_code" in response
    assert response["status_code"] == 400
    print("   ✓ exception_to_response works")
    
    # Test handle_exception
    print("\n8. Testing handle_exception...")
    exc = DataValidationError("Validation failed", details={"col": "age"})
    msg = handle_exception(exc, "Data processing")
    assert "Validation failed" in msg
    assert "Data processing" in msg
    print("   ✓ handle_exception works")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from utils.exceptions import (
    DataLoadError,
    DataValidationError,
    ModelTrainingError,
    safe_execute,
    wrap_exceptions,
    exception_context,
    ErrorCode,
    ErrorSeverity
)

# === Raise Specific Exception ===

raise DataLoadError(
    "Failed to load CSV file",
    details={
        "file": "data.csv",
        "error": "File not found"
    },
    context={
        "user_id": "123",
        "session_id": "abc"
    }
)

# === Safe Execution ===

result = safe_execute(
    load_data,
    "data.csv",
    error_message="Failed to load data",
    exc_type=DataLoadError,
    error_code=ErrorCode.DATA_LOAD,
    context={"source": "file"}
)

# === Decorator ===

@wrap_exceptions(
    to=ModelTrainingError,
    message="Model training failed",
    error_code=ErrorCode.MODEL_TRAINING,
    severity=ErrorSeverity.ERROR
)
def train_model(data, model_name):
    # Training logic that might fail
    model.fit(data)
    return model

# === Context Manager ===

with exception_context(
    to=DataLoadError,
    message="Failed to load data",
    error_code=ErrorCode.DATA_LOAD,
    context={"file": "data.csv"}
):
    df = pd.read_csv("data.csv")

# === FastAPI Integration ===

from fastapi import FastAPI, HTTPException
from utils.exceptions import (
    DataValidationError,
    exception_to_response
)

app = FastAPI()

@app.post("/validate")
def validate_data(data: dict):
    try:
        # Validation logic
        if not data:
            raise DataValidationError(
                "Empty data",
                details={"received": data}
            )
        return {"status": "valid"}
    except DataValidationError as e:
        response = exception_to_response(e)
        raise HTTPException(
            status_code=response["status_code"],
            detail=response["error"]
        )

# === Error Handling with Context ===

try:
    process_data()
except Exception as e:
    dg_exc = DataGeniusException.from_exc(
        e,
        default_code=ErrorCode.PIPELINE,
        message="Pipeline failed",
        context={"stage": "processing"}
    )
    logger.error(str(dg_exc))
    raise dg_exc

# === Custom Exception with Cause ===

try:
    load_model()
except FileNotFoundError as e:
    raise ModelTrainingError(
        "Model file not found",
        details={"path": "models/model.pkl"},
        cause=e
    )
    """)
    
    print("\n" + "="*80)
    print("✓ Self-test complete")
    print("="*80)