"""Utility functions for model management."""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import openai
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from ..exceptions import ModelBackupError, ModelError, RateLimitError
from .logging_utils import setup_logging

# Set up logging
logger = setup_logging("model_utils")

# Type variable for the return value of model operations
T = TypeVar("T")


def create_model(
    model_type: ModelType,
    model_platform: ModelPlatformType = ModelPlatformType.OPENAI,
    model_config_dict: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
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
    defaults: dict[str, Any] = {
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
    user_friendly_names: dict[str, Any] = {
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


def get_api_key(model_name: str) -> Optional[str]:
    """Get the API key for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        API key if found, None otherwise
    """
    # Handle OpenAI models
    if model_name.startswith("gpt-") or model_name.endswith("-o"):
        return os.getenv("OPENAI_API_KEY")

    # Handle Anthropic models
    if model_name.startswith("claude-"):
        return os.getenv("ANTHROPIC_API_KEY")

    # Handle o3-mini
    if model_name == "o3-mini":
        return os.getenv("O3_MINI_API_KEY")

    # Default to OpenAI key
    return os.getenv("OPENAI_API_KEY")


def with_exponential_backoff(
    func: Callable[..., T],
    max_retries: int = 5,
    base_delay: float = 1.0,
    rate_limit_error_codes: List[int] = [429, 500, 503],
) -> Callable[..., T]:
    """Decorator to add exponential backoff to a function.

    Args:
        func: Function to decorate
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        rate_limit_error_codes: HTTP error codes to retry on

    Returns:
        Decorated function with exponential backoff
    """

    def wrapper(*args: Any, **kwargs: Any) -> T:
        retries: int = 0
        while True:
            try:
                return func(*args, **kwargs)
            except openai.RateLimitError as e:
                if retries >= max_retries:
                    raise RateLimitError(
                        f"Rate limit exceeded after {max_retries} retries",
                        retry_after=int(base_delay * (2**retries)),
                    )

                delay = base_delay * (2**retries)
                logger.warning(
                    f"Rate limit exceeded, retrying in {delay:.2f} seconds (retry {retries + 1}/{max_retries})"
                )
                time.sleep(delay)
                retries += 1
            except Exception as e:
                raise e

    return wrapper


def with_backup_model(
    func: Callable[..., T],
    primary_model_param: str = "model_name",
    backup_model_param: str = "backup_model",
) -> Callable[..., T]:
    """Decorator to add backup model functionality to a function.

    Args:
        func: Function to decorate
        primary_model_param: Name of the parameter containing the primary model
        backup_model_param: Name of the parameter containing the backup model

    Returns:
        Decorated function with backup model functionality
    """

    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Get the model names from the arguments
        primary_model = kwargs.get(primary_model_param)
        backup_model = kwargs.get(backup_model_param)

        # If there's no backup model, just call the function
        if not backup_model:
            return func(*args, **kwargs)

        try:
            # Try with the primary model
            return func(*args, **kwargs)
        except ModelError as e:
            # If we have a backup model, try with that
            logger.warning(
                f"Primary model {primary_model} failed: {str(e)}. "
                f"Falling back to backup model {backup_model}."
            )

            # Swap the models
            kwargs[primary_model_param] = backup_model

            try:
                # Try with the backup model
                return func(*args, **kwargs)
            except ModelError as backup_error:
                # If both models fail, raise a ModelBackupError
                raise ModelBackupError(
                    f"Both primary and backup models failed",
                    primary_model=str(primary_model),
                    backup_model=backup_model,
                    operation=e.operation,
                    details={
                        "primary_error": str(e),
                        "backup_error": str(backup_error),
                    },
                )

    return wrapper


def get_chat_agent(
    model_type: ModelType,
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> ChatAgent:
    """Create a ChatAgent with the specified model.

    Args:
        model_type: Type of the model.
        system_prompt: System prompt for the ChatAgent
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation

    Returns:
        Initialized ChatAgent

    Raises:
        ModelError: If the ChatAgent initialization fails
    """
    try:
        # Get the API key for the model
        model_name = str(model_type)
        api_key = get_api_key(model_name)

        if not api_key:
            raise ModelError(
                f"No API key found for model {model_name}",
                model_name=model_name,
                operation="initialization",
            )

        # Create the ChatAgent
        model_config: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        model = create_model(model_type, model_config_dict=model_config)

        if model is None:
            raise ModelError(
                f"Failed to create model for ChatAgent with model type {model_type}",
                model_name=str(model_type),
                operation="initialization",
            )

        agent = ChatAgent(system_prompt, model)

        return agent
    except Exception as e:
        raise ModelError(
            f"Failed to initialize ChatAgent with model {model_type}: {str(e)}",
            model_name=str(model_type),
            operation="initialization",
            details={"error": str(e)},
        )
