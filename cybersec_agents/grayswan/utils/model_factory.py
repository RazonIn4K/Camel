"""
Factory functions for creating model instances.
"""

from typing import Any, Dict, Optional, Type, Union
import logging

from camel.agents import ChatAgent
from camel.models import ModelFactory, BaseModelBackend
from camel.types import ModelType, ModelPlatformType
from camel.types.enums import RoleType

from cybersec_agents.grayswan.exceptions import ModelError
from .logging_utils import setup_logging

# Set up logging
logger = logging.getLogger(__name__)
setup_logging(logger_name="model_factory")


def create_model(
    model_type: ModelType,
    model_platform: ModelPlatformType,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseModelBackend:
    """Create a model instance.

    Args:
        model_type: Type of the model.
        model_platform: Platform of the model.
        api_key: API key for the model.
        **kwargs: Additional model configuration.

    Returns:
        A model instance.

    Raises:
        ModelError: If model creation fails.
    """
    try:
        logger.info(f"Creating model of type {model_type} on {model_platform}")
        model = ModelFactory.create(
            model_type=model_type,
            model_platform=model_platform,
            api_key=api_key,
            **kwargs,
        )
        logger.info(f"Model created successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        raise ModelError(f"Failed to create model: {str(e)}", "create") from e


def get_chat_agent(
    model_type: ModelType,
    model_platform: ModelPlatformType,
    api_key: Optional[str] = None,
    role_type: RoleType = RoleType.USER,
    **kwargs: Any,
) -> ChatAgent:
    """Get a chat agent with the specified model.

    Args:
        model_type: Type of the model.
        model_platform: Platform of the model.
        api_key: API key for the model.
        role_type: Role type for the agent.
        **kwargs: Additional model configuration.

    Returns:
        A chat agent instance.

    Raises:
        ModelError: If agent creation fails.
    """
    try:
        logger.info(f"Creating chat agent with role {role_type}")
        model = create_model(
            model_type=model_type,
            model_platform=model_platform,
            api_key=api_key,
            **kwargs,
        )
        agent = ChatAgent(model=model, role_type=role_type)
        logger.info(f"Chat agent created successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to create chat agent: {str(e)}")
        raise ModelError(f"Failed to create chat agent: {str(e)}", "create") from e 