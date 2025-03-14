"""Utility functions for model management."""

import logging
from typing import Any, Dict, Optional

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

logger = logging.getLogger(__name__)


def create_model(
    model_type: ModelType,
    model_platform: ModelPlatformType = ModelPlatformType.OPENAI,
    model_config_dict: Optional[Dict[str, Any]] = None,
):
    """Create a model using Camel's ModelFactory with standardized settings.

    Args:
        model_type: The type of model to create (from camel.types.ModelType)
        model_platform: The platform of the model (from camel.types.ModelPlatformType)
        model_config_dict: Optional configuration dictionary for the model

    Returns:
        The created model or None if creation fails
    """
    try:
        logger.info(f"Creating model: {model_type} on {model_platform}")

        model = ModelFactory.create(
            model_type=model_type,
            model_platform=model_platform,
            **(model_config_dict or {}),
        )

        logger.info(f"Model created successfully: {model_type}")
        return model
    except Exception as e:
        logger.error(
            f"Failed to create model {model_type} on {model_platform}: {str(e)}"
        )
        return None


def get_default_model_type_for_agent(agent_type: str) -> ModelType:
    """Get the default model type for a specific agent type.

    Args:
        agent_type: The type of agent (e.g., "recon", "prompt_engineer", "evaluation")

    Returns:
        The default ModelType for the agent
    """
    # Define defaults - bias toward more powerful models for critical tasks
    defaults = {
        "recon": ModelType.GPT_4,  # Needs strong reasoning for research
        "prompt_engineer": ModelType.GPT_4,  # Creative and nuanced task
        "exploit_delivery": ModelType.GPT_3_5_TURBO,  # Mainly execution
        "evaluation": ModelType.GPT_4,  # Complex analysis
        # Default for unknown agent types
        "default": ModelType.GPT_3_5_TURBO,
    }

    return defaults.get(agent_type.lower(), defaults["default"])


def get_model_info(
    model_type: ModelType, model_platform: ModelPlatformType
) -> Dict[str, Any]:
    """Get information about a model.

    Args:
        model_type: The type of model
        model_platform: The platform of the model

    Returns:
        Dictionary with model information
    """
    # Convert enum to string representation for easier logging/tracking
    model_type_str = str(model_type).split(".")[-1]
    model_platform_str = str(model_platform).split(".")[-1]

    # Map to user-friendly names where possible
    user_friendly_names = {
        "GPT_3_5_TURBO": "GPT-3.5 Turbo",
        "GPT_4": "GPT-4",
        "GPT_4_TURBO": "GPT-4 Turbo",
        "OPENAI": "OpenAI",
    }

    return {
        "model_type": model_type_str,
        "model_platform": model_platform_str,
        "display_name": user_friendly_names.get(model_type_str, model_type_str),
        "platform_display_name": user_friendly_names.get(
            model_platform_str, model_platform_str
        ),
    }
