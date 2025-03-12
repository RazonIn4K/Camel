"""Base agent implementation for all cybersecurity agents."""

from typing import Any, Dict, Optional


class BaseAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def validate_api_key(self) -> bool:
        """Validate the API key."""
        return bool(self.api_key)

    def execute(self, command: str, **kwargs: Any) -> Dict:
        """Execute a command with the agent."""
        raise NotImplementedError
