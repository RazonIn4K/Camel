"""Exceptions module for Gray Swan Arena agents."""

from typing import Any, Dict, Optional


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


class ModelError(GraySwanError):
    """Exception raised for errors related to model operations."""

    def __init__(
        self,
        message: str,
        model_name: str,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the model error.

        Args:
            message: Error message
            model_name: Name of the model that caused the error
            operation: Operation that failed
            details: Additional error details
        """
        super().__init__(message, details)
        self.model_name = model_name
        self.operation = operation


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


class ModelBackupError(ModelError):
    """Exception raised when both primary and backup models fail."""

    def __init__(
        self,
        message: str,
        primary_model: str,
        backup_model: str,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the model backup error.

        Args:
            message: Error message
            primary_model: Name of the primary model that failed
            backup_model: Name of the backup model that failed
            operation: Operation that failed
            details: Additional error details
        """
        super().__init__(message, primary_model, operation, details)
        self.backup_model = backup_model


class RateLimitError(APIError):
    """Exception raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            details: Additional error details
        """
        super().__init__(message, 429, None, details)
        self.retry_after = retry_after


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
