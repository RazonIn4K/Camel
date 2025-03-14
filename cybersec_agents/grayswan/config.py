"""Configuration management for Gray Swan Arena agents."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator

from .exceptions import ConfigurationError


class ModelConfig(BaseModel):
    """Configuration for AI models."""

    name: str = Field(..., description="Name of the model")
    api_key_env: str = Field(..., description="Environment variable name for API key")
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    timeout: int = Field(30, description="Timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    temperature: float = Field(0.7, description="Model temperature")
    max_tokens: int = Field(2048, description="Maximum tokens per request")

    @validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v


class AgentConfig(BaseModel):
    """Configuration for agents."""

    output_dir: Path = Field(..., description="Output directory for agent results")
    model_name: str = Field(..., description="Primary model name")
    backup_model: Optional[str] = Field(None, description="Backup model name")
    reasoning_model: Optional[str] = Field(
        None, description="Model for reasoning tasks"
    )
    complexity_threshold: float = Field(
        0.7, description="Threshold for switching to backup model"
    )

    @validator("output_dir")
    def validate_output_dir(cls, v: Path) -> Path:
        """Validate and create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @validator("complexity_threshold")
    def validate_complexity_threshold(cls, v: float) -> float:
        """Validate complexity threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Complexity threshold must be between 0 and 1")
        return v


class APIConfig(BaseModel):
    """Configuration for API integrations."""

    discord_token: Optional[str] = Field(None, description="Discord API token")
    agentops_key: Optional[str] = Field(None, description="AgentOps API key")
    rate_limit_delay: float = Field(
        1.0, description="Delay between API calls in seconds"
    )
    max_concurrent: int = Field(5, description="Maximum concurrent API calls")


@dataclass
class Environment:
    """Environment configuration."""

    name: str
    config_path: Path
    is_production: bool = False


class Config:
    """Main configuration class for Gray Swan Arena."""

    def __init__(
        self,
        env: str = "development",
        config_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize configuration.

        Args:
            env: Environment name (development, staging, production)
            config_dir: Directory containing configuration files
        """
        self.env = Environment(
            name=env,
            config_path=Path(config_dir or "config") / f"{env}.yml",
            is_production=env == "production",
        )

        self._load_config()
        self._validate_config()

    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        try:
            if not self.env.config_path.exists():
                raise ConfigurationError(
                    f"Configuration file not found: {self.env.config_path}"
                )

            with open(self.env.config_path) as f:
                config_data = yaml.safe_load(f)

            self.models = {
                name: ModelConfig(**cfg)
                for name, cfg in config_data.get("models", {}).items()
            }

            self.agents = {
                name: AgentConfig(**cfg)
                for name, cfg in config_data.get("agents", {}).items()
            }

            self.api = APIConfig(**config_data.get("api", {}))

            # Load environment variables
            self._load_env_vars()

        except Exception as e:
            raise ConfigurationError(
                f"Error loading configuration: {str(e)}",
                details={"env": self.env.name, "path": str(self.env.config_path)},
            )

    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        # Load API keys for models
        for model in self.models.values():
            api_key = os.getenv(model.api_key_env)
            if not api_key and self.env.is_production:
                raise ConfigurationError(
                    f"Missing API key for model {model.name}",
                    config_key=model.api_key_env,
                )

        # Load API tokens
        if not self.api.discord_token:
            self.api.discord_token = os.getenv("DISCORD_TOKEN")

        if not self.api.agentops_key:
            self.api.agentops_key = os.getenv("AGENTOPS_API_KEY")

    def _validate_config(self) -> None:
        """Validate configuration."""
        # Validate model references in agent configs
        for agent_name, agent_cfg in self.agents.items():
            if agent_cfg.model_name not in self.models:
                raise ConfigurationError(
                    f"Invalid model reference in agent {agent_name}: {agent_cfg.model_name}"
                )

            if agent_cfg.backup_model and agent_cfg.backup_model not in self.models:
                raise ConfigurationError(
                    f"Invalid backup model reference in agent {agent_name}: {agent_cfg.backup_model}"
                )

            if (
                agent_cfg.reasoning_model
                and agent_cfg.reasoning_model not in self.models
            ):
                raise ConfigurationError(
                    f"Invalid reasoning model reference in agent {agent_name}: {agent_cfg.reasoning_model}"
                )

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration

        Raises:
            ConfigurationError: If model configuration is not found
        """
        try:
            return self.models[model_name]
        except KeyError:
            raise ConfigurationError(f"Model configuration not found: {model_name}")

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent configuration

        Raises:
            ConfigurationError: If agent configuration is not found
        """
        try:
            return self.agents[agent_name]
        except KeyError:
            raise ConfigurationError(f"Agent configuration not found: {agent_name}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "env": self.env.name,
            "models": {name: model.dict() for name, model in self.models.items()},
            "agents": {name: agent.dict() for name, agent in self.agents.items()},
            "api": self.api.dict(),
        }

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.

        Args:
            path: Path to save configuration file (defaults to current environment config path)
        """
        save_path = Path(path) if path else self.env.config_path
        try:
            with open(save_path, "w") as f:
                yaml.safe_dump(self.to_dict(), f)
        except Exception as e:
            raise ConfigurationError(
                f"Error saving configuration: {str(e)}",
                details={"path": str(save_path)},
            )
