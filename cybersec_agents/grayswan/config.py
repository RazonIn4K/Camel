"""Configuration management for Gray Swan Arena agents."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator
from camel.types import ModelType, ModelPlatformType

from .exceptions import ConfigurationError


class ModelConfig(BaseModel):
    """Configuration for AI models."""

    model_type: ModelType = Field(..., description="Type of the model")
    model_platform: ModelPlatformType = Field(..., description="Platform of the model")
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

    @validator("base_url", always=True)
    def validate_base_url(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate base_url is set for Perplexity platform."""
        if values.get("model_platform") == ModelPlatformType.PERPLEXITY and not v:
            raise ValueError("base_url must be provided for Perplexity model platform")
        return v


class AgentConfig(BaseModel):
    """Configuration for agents."""

    output_dir: Path = Field(..., description="Output directory for agent results")
    model_type: ModelType = Field(..., description="Primary model type")
    model_platform: ModelPlatformType = Field(..., description="Primary model platform")
    backup_model_type: Optional[ModelType] = Field(None, description="Backup model type")
    backup_model_platform: Optional[ModelPlatformType] = Field(None, description="Backup model platform")
    reasoning_model_type: Optional[ModelType] = Field(None, description="Model type for reasoning tasks")
    reasoning_model_platform: Optional[ModelPlatformType] = Field(None, description="Model platform for reasoning tasks")
    complexity_threshold: float = Field(0.7, description="Threshold for switching to backup model")

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

    @validator("backup_model_platform")
    def validate_backup_model_platform(cls, v: Optional[ModelPlatformType], values: Dict[str, Any]) -> Optional[ModelPlatformType]:
        """Validate that backup_model_platform is provided if backup_model_type is set."""
        if values.get("backup_model_type") is not None and v is None:
            raise ValueError("backup_model_platform must be provided when backup_model_type is set")
        return v

    @validator("reasoning_model_platform")
    def validate_reasoning_model_platform(cls, v: Optional[ModelPlatformType], values: Dict[str, Any]) -> Optional[ModelPlatformType]:
        """Validate that reasoning_model_platform is provided if reasoning_model_type is set."""
        if values.get("reasoning_model_type") is not None and v is None:
            raise ValueError("reasoning_model_platform must be provided when reasoning_model_type is set")
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

            # Load model configurations
            self.models = {}
            for model_cfg in config_data.get("models", []):
                model_type = ModelType[model_cfg.pop("type")]
                model_platform = ModelPlatformType[model_cfg.pop("platform")]
                self.models[(model_type, model_platform)] = ModelConfig(
                    model_type=model_type,
                    model_platform=model_platform,
                    **model_cfg
                )

            # Load agent configurations
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
                    f"Missing API key for model {model.model_type} on {model.model_platform}",
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
            # Check primary model
            if (agent_cfg.model_type, agent_cfg.model_platform) not in self.models:
                raise ConfigurationError(
                    f"Invalid model reference in agent {agent_name}: {agent_cfg.model_type} on {agent_cfg.model_platform}"
                )

            # Check backup model if specified
            if agent_cfg.backup_model_type:
                if (agent_cfg.backup_model_type, agent_cfg.backup_model_platform) not in self.models:
                    raise ConfigurationError(
                        f"Invalid backup model reference in agent {agent_name}: {agent_cfg.backup_model_type} on {agent_cfg.backup_model_platform}"
                    )

            # Check reasoning model if specified
            if agent_cfg.reasoning_model_type:
                if (agent_cfg.reasoning_model_type, agent_cfg.reasoning_model_platform) not in self.models:
                    raise ConfigurationError(
                        f"Invalid reasoning model reference in agent {agent_name}: {agent_cfg.reasoning_model_type} on {agent_cfg.reasoning_model_platform}"
                    )

    def get_model_config(self, model_type: ModelType, model_platform: ModelPlatformType) -> ModelConfig:
        """Get configuration for a specific model.

        Args:
            model_type: Type of the model
            model_platform: Platform of the model

        Returns:
            Model configuration

        Raises:
            ConfigurationError: If model configuration is not found
        """
        try:
            return self.models[(model_type, model_platform)]
        except KeyError:
            raise ConfigurationError(f"Model configuration not found: {model_type} on {model_platform}")

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
            "models": [
                {
                    "type": model_type.name,
                    "platform": model_platform.name,
                    **{k: v for k, v in model.dict().items() if k not in ["model_type", "model_platform"]}
                }
                for (model_type, model_platform), model in self.models.items()
            ],
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
