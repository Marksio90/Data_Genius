"""
DataGenius PRO - Custom Exceptions
Centralized exception definitions
"""


class DataGeniusException(Exception):
    """Base exception for DataGenius PRO"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataLoadError(DataGeniusException):
    """Exception raised when data loading fails"""
    pass


class DataValidationError(DataGeniusException):
    """Exception raised when data validation fails"""
    pass


class InsufficientDataError(DataGeniusException):
    """Exception raised when there's not enough data for ML"""
    pass


class InvalidTargetError(DataGeniusException):
    """Exception raised when target column is invalid"""
    pass


class ModelTrainingError(DataGeniusException):
    """Exception raised when model training fails"""
    pass


class ModelPredictionError(DataGeniusException):
    """Exception raised when model prediction fails"""
    pass


class LLMError(DataGeniusException):
    """Exception raised when LLM API call fails"""
    pass


class ConfigurationError(DataGeniusException):
    """Exception raised when configuration is invalid"""
    pass


class AgentExecutionError(DataGeniusException):
    """Exception raised when agent execution fails"""
    pass


class PipelineError(DataGeniusException):
    """Exception raised when pipeline execution fails"""
    pass


class FeatureEngineeringError(DataGeniusException):
    """Exception raised when feature engineering fails"""
    pass


class ReportGenerationError(DataGeniusException):
    """Exception raised when report generation fails"""
    pass


class DatabaseError(DataGeniusException):
    """Exception raised when database operation fails"""
    pass


class CacheError(DataGeniusException):
    """Exception raised when cache operation fails"""
    pass


class MonitoringError(DataGeniusException):
    """Exception raised when monitoring fails"""
    pass


# Utility functions for exception handling

def handle_exception(e: Exception, context: str = "") -> str:
    """
    Format exception for user display
    
    Args:
        e: Exception
        context: Additional context
    
    Returns:
        Formatted error message
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


def safe_execute(func, *args, error_message: str = "Operacja nie powiodła się", **kwargs):
    """
    Execute function with exception handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        error_message: Error message to display
        **kwargs: Function keyword arguments
    
    Returns:
        Function result or None if failed
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        from loguru import logger
        logger.error(f"{error_message}: {e}", exc_info=True)
        raise DataGeniusException(error_message, details={"original_error": str(e)})