class CyberSecurityError(Exception):
    """Base exception for all cybersecurity agent errors."""


class ValidationError(CyberSecurityError):
    """Raised when input validation fails."""


class SecurityError(CyberSecurityError):
    """Raised for security-related issues."""


class AgentError(CyberSecurityError):
    """Raised for agent-specific errors."""


class RateLimitError(CyberSecurityError):
    """Raised when rate limit is exceeded."""


class FileError(CyberSecurityError):
    """Raised for file-related issues."""

    def __init__(self, file_path: str, message: str):
        self.file_path = file_path
        self.message = message
        super().__init__(f"File error for {file_path}: {message}")
