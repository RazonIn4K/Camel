"""
Custom exceptions for the Gray Swan Arena framework.
"""

from typing import Any, Dict, Optional

from camel.types import ModelType, ModelPlatformType


class GraySwanError(Exception):
    """Base exception class for Gray Swan Arena errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.details = details or {}


class ModelError(Exception):
    """Custom exception for model-related errors."""
    
    def __init__(self, message: str, model: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModelError.
        
        Args:
            message (str): The error message.
            model (Optional[str]): The name of the model (optional).
            details (Optional[Dict[str, Any]]): Additional error details (optional).
        """
        self.message = message
        self.model = model
        self.details = details or {}
        super().__init__(message)

    def __str__(self):
        """String representation of the error."""
        if self.model:
            return f"{self.message} (Model: {self.model})"
        return self.message


class ModelConfigError(ModelError):
    """Exception raised for model configuration errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the ModelConfigError.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, model="config", details=details)


class ModelBackupError(ModelError):
    """Exception raised when both primary and backup models fail."""

    def __init__(
        self,
        message: str,
        primary_model: Optional[str] = None,
        backup_model: Optional[str] = None,
        operation: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        primary_model_type: Optional[ModelType] = None,
        primary_model_platform: Optional[ModelPlatformType] = None,
        backup_model_type: Optional[ModelType] = None,
        backup_model_platform: Optional[ModelPlatformType] = None,
    ):
        """Initialize the ModelBackupError.

        Args:
            message: Error message
            primary_model: String representation of the primary model (for backward compatibility)
            backup_model: String representation of the backup model (for backward compatibility)
            operation: Operation that failed
            details: Additional error details
            primary_model_type: Type of the primary model
            primary_model_platform: Platform of the primary model
            backup_model_type: Type of the backup model
            backup_model_platform: Platform of the backup model
        """
        # Fix: properly pass parameters to parent constructor
        super().__init__(message, model=operation, details=details)
        
        # Support both the old and new parameter styles
        self.primary_model = primary_model
        self.backup_model = backup_model
        
        # Store the new parameters if provided
        self.primary_model_type = primary_model_type
        self.primary_model_platform = primary_model_platform
        self.backup_model_type = backup_model_type
        self.backup_model_platform = backup_model_platform


class RateLimitError(ModelError):
    """Exception raised when rate limits are hit."""

    def __init__(self, message: str, retry_after: int = 0, details: Optional[Dict[str, Any]] = None):
        """Initialize the RateLimitError.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            details: Additional error details
        """
        super().__init__(message, model="rate_limit", details=details)
        self.retry_after = retry_after


class ModelTimeoutError(ModelError):
    """Exception raised when model operations timeout."""

    def __init__(self, message: str, timeout: float, details: Optional[Dict[str, Any]] = None):
        """Initialize the ModelTimeoutError.

        Args:
            message: Error message
            timeout: Timeout duration in seconds
            details: Additional error details
        """
        super().__init__(message, model="timeout", details=details)
        self.timeout = timeout


class ModelValidationError(ModelError):
    """Exception raised when model inputs/outputs are invalid."""

    def __init__(self, message: str, field: str = "", details: Optional[Dict[str, Any]] = None):
        """Initialize the ModelValidationError.

        Args:
            message: Error message
            field: Field that failed validation
            details: Additional error details
        """
        super().__init__(message, model="validation", details=details)
        self.field = field


class ModelConnectionError(ModelError):
    """Exception raised when model connection fails."""

    def __init__(self, message: str, endpoint: str = "", details: Optional[Dict[str, Any]] = None):
        """Initialize the ModelConnectionError.

        Args:
            message: Error message
            endpoint: Endpoint that failed to connect
            details: Additional error details
        """
        super().__init__(message, model="connection", details=details)
        self.endpoint = endpoint


class ModelAuthenticationError(ModelError):
    """Exception raised when model authentication fails."""

    def __init__(self, message: str, provider: str = "", details: Optional[Dict[str, Any]] = None):
        """Initialize the ModelAuthenticationError.

        Args:
            message: Error message
            provider: Provider that failed authentication
            details: Additional error details
        """
        super().__init__(message, model="authentication", details=details)
        self.provider = provider


class ConfigurationError(GraySwanError):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the configuration error.

        Args:
            message: Error message
            config_key: Configuration key that caused the error
            details: Additional error details
        """
        super().__init__(message, details)
        self.config_key = config_key


class ValidationError(GraySwanError):
    """Exception raised for data validation errors."""

    def __init__(
        self,
        message: str,
        field: str,
        value: Any,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            details: Additional error details
        """
        super().__init__(message, details)
        self.field = field
        self.value = value


class APIError(GraySwanError):
    """Exception raised for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the API error.

        Args:
            message: Error message
            status_code: HTTP status code
            endpoint: API endpoint that failed
            details: Additional error details
        """
        super().__init__(message, details)
        self.status_code = status_code
        self.endpoint = endpoint


class AgentError(GraySwanError):
    """Exception raised for agent-specific errors."""

    def __init__(
        self,
        message: str,
        agent_name: str,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the agent error.

        Args:
            message: Error message
            agent_name: Name of the agent that caused the error
            operation: Operation that failed
            details: Additional error details
        """
        super().__init__(message, details)
        self.agent_name = agent_name
        self.operation = operation


class DataProcessingError(GraySwanError):
    """Exception raised for errors during data processing."""

    def __init__(
        self,
        message: str,
        data_type: str,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the data processing error.

        Args:
            message: Error message
            data_type: Type of data being processed
            operation: Operation that failed
            details: Additional error details
        """
        super().__init__(message, details)
        self.data_type = data_type
        self.operation = operation


class RecoveryError(GraySwanError):
    """Exception raised when error recovery fails."""

    def __init__(
        self,
        message: str,
        original_error: Exception,
        recovery_strategy: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the recovery error.

        Args:
            message: Error message
            original_error: The original error that triggered recovery
            recovery_strategy: The recovery strategy that failed
            details: Additional error details
        """
        super().__init__(message, details)
        self.original_error = original_error
        self.recovery_strategy = recovery_strategy
