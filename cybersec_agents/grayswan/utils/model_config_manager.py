"""
Model configuration manager for handling model parameters and settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from camel.types import ModelType, ModelPlatformType
from ..exceptions import ModelConfigError

logger = logging.getLogger(__name__)


class ModelConfigManager:
    """Manages model configurations and parameters."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the ModelConfigManager.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    def _create_default_config(self) -> Dict:
        """Create a default model configuration.

        Returns:
            Dictionary containing default model configurations
        """
        default_config = {
            (ModelType.GPT_4.name, ModelPlatformType.OPENAI.name): {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            (ModelType.GPT_3_5_TURBO.name, ModelPlatformType.OPENAI.name): {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            (ModelType.CLAUDE_3_SONNET.name, ModelPlatformType.ANTHROPIC.name): {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 1.0,
            },
            (ModelType.CLAUDE_3_OPUS.name, ModelPlatformType.ANTHROPIC.name): {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 1.0,
            },
        }
        return default_config

    def get_model_params(self, model_type: ModelType, model_platform: ModelPlatformType) -> Optional[Dict]:
        """Get model parameters for a specific model.

        Args:
            model_type: Type of the model
            model_platform: Platform of the model

        Returns:
            Dictionary containing model parameters or None if not found
        """
        key = (model_type.name, model_platform.name)
        return self.config.get(key, self._create_default_config().get(key))

    def update_model_params(
        self, model_type: ModelType, model_platform: ModelPlatformType, params: Dict
    ) -> None:
        """Update model parameters for a specific model.

        Args:
            model_type: Type of the model
            model_platform: Platform of the model
            params: New parameters to set

        Raises:
            ModelConfigError: If there is an error updating the configuration
        """
        try:
            key = (model_type.name, model_platform.name)
            self.config[key] = params
            logger.info("Updated parameters for model %s on %s", model_type.name, model_platform.name)
        except Exception as e:
            logger.error("Error updating model parameters: %s", str(e))
            raise ModelConfigError(f"Failed to update model parameters: {str(e)}") from e

    def remove_model(self, model_type: ModelType, model_platform: ModelPlatformType) -> None:
        """Remove a model from the configuration.

        Args:
            model_type: Type of the model
            model_platform: Platform of the model

        Raises:
            ModelConfigError: If there is an error removing the model
        """
        try:
            key = (model_type.name, model_platform.name)
            if key in self.config:
                del self.config[key]
                logger.info("Removed model %s on %s from configuration", model_type.name, model_platform.name)
            else:
                logger.warning("Model %s on %s not found in configuration", model_type.name, model_platform.name)
        except Exception as e:
            logger.error("Error removing model: %s", str(e))
            raise ModelConfigError(f"Failed to remove model: {str(e)}") from e

    def get_model_config_by_type_and_platform(
        self, model_type: ModelType, model_platform: ModelPlatformType
    ) -> Dict:
        """Get model configuration by type and platform.

        Args:
            model_type: Type of the model
            model_platform: Platform of the model

        Returns:
            Dictionary containing model configuration

        Raises:
            ModelConfigError: If model configuration is not found
        """
        params = self.get_model_params(model_type, model_platform)
        if params is None:
            raise ModelConfigError(f"Model configuration not found for {model_type.name} on {model_platform.name}")
        return params 