"""Network security agent implementation."""

from dataclasses import dataclass
from typing import Dict, Optional

from ..core.base_agent import BaseAgent
from ..utils.credentials import CredentialManager


@dataclass
class NetworkSecurityAgent(BaseAgent):
    provider: str = "anthropic"
    config: Optional[Dict] = None

    def __post_init__(self):
        self.cred_manager = CredentialManager()
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the AI provider."""
        self.api_key = self.cred_manager.get_credential(f"{self.provider}_api_key")
        # Your existing initialization code...
