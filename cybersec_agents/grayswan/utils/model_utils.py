"""
Utility functions for model management and configuration.

This module provides utility functions for interacting with different AI models
and handling errors in a consistent way.
"""

import logging
import os
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import agentops
from camel.models import BaseModelBackend
from camel.models.factory import ModelFactory  # Added proper import
from camel.types import ModelPlatformType, ModelType

from cybersec_agents.grayswan.exceptions import ModelBackupError, ModelError, RateLimitError
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("model_utils")

# Check if AgentOps is available
AGENTOPS_AVAILABLE = 'AGENTOPS_API_KEY' in os.environ

# Type variable for generic function return type
T = TypeVar('T')


def get_model_name_from_type(model_type: ModelType, model_platform: ModelPlatformType) -> str:
    """Get the model name based on the model type and platform.
    
    Args:
        model_type: The model type.
        model_platform: The model platform.
        
    Returns:
        The model name.
        
    Raises:
        ModelError: If the model type and platform combination is not supported.
    """
    model_map = {
        (ModelType.GPT_3_5_TURBO, ModelPlatformType.OPENAI): "gpt-3.5-turbo",
        (ModelType.GPT_4, ModelPlatformType.OPENAI): "gpt-4",
        (ModelType.GPT_4_TURBO, ModelPlatformType.OPENAI): "gpt-4-turbo",
        (ModelType.MISTRAL_8X7B, ModelPlatformType.HUGGINGFACE): "mistral-medium",
        (ModelType.GEMINI_PRO, ModelPlatformType.GOOGLE): "gemini-pro",
        (ModelType.GEMINI_2_PRO, ModelPlatformType.GOOGLE): "gemini-2.0-pro-exp-02-05",  # Add Gemini
        (ModelType.CLAUDE_3_SONNET, ModelPlatformType.ANTHROPIC): "claude-3-sonnet",
        (ModelType.CLAUDE_3_7_SONNET, ModelPlatformType.ANTHROPIC): "claude-3-7-sonnet",  # Add Claude
        (ModelType.O3_MINI, ModelPlatformType.OPENAI): "o3-mini",  # Add O3-mini
        (ModelType.GPT_4O, ModelPlatformType.OPENAI): "gpt-4o",  # Add GPT-4o
    }
    try:
        return model_map[(model_type, model_platform)]
    except KeyError:
        raise ModelError(f"Unsupported model type: {model_type} on platform: {model_platform}")


def get_model_type(model_name: str) -> ModelType:
    """Determine the model type from its name.
    
    Args:
        model_name: The name of the model.
        
    Returns:
        The model type.
        
    Raises:
        ModelError: If the model name is unknown.
    """
    model_type_map = {
        "gpt-3.5-turbo": ModelType.GPT_3_5_TURBO,
        "gpt-4": ModelType.GPT_4,
        "gpt-4-turbo": ModelType.GPT_4_TURBO,
        "mistral-medium": ModelType.MISTRAL_8X7B,
        "gemini-pro": ModelType.GEMINI_PRO,
        "gemini-2.0-pro-exp-02-05": ModelType.GEMINI_2_PRO,  # Add Gemini
        "claude-3-sonnet": ModelType.CLAUDE_3_SONNET,
        "claude-3-7-sonnet": ModelType.CLAUDE_3_7_SONNET,  # Add Claude
        "o3-mini": ModelType.O3_MINI,  # Add O3-mini
        "gpt-4o": ModelType.GPT_4O,  # Add GPT-4o
    }
    
    # Handle string representation of ModelType enum
    if model_name and model_name.upper() in [m.name for m in ModelType]:
        try:
            return ModelType[model_name.upper()]
        except (KeyError, AttributeError):
            pass
    
    model_type = model_type_map.get(model_name)
    if not model_type:
        raise ModelError(f"Unknown model name: {model_name}")
    return model_type


def get_model_platform(model_name: str) -> ModelPlatformType:
    """Determine the model platform from its name.
    
    Args:
        model_name: The name of the model.
        
    Returns:
        The model platform.
        
    Raises:
        ModelError: If the model name is unknown.
    """
    model_platform_map = {
        "gpt-3.5-turbo": ModelPlatformType.OPENAI,
        "gpt-4": ModelPlatformType.OPENAI,
        "gpt-4-turbo": ModelPlatformType.OPENAI,
        "mistral-medium": ModelPlatformType.HUGGINGFACE,
        "gemini-pro": ModelPlatformType.GOOGLE,
        "gemini-2.0-pro-exp-02-05": ModelPlatformType.GOOGLE,  # Add Gemini
        "claude-3-sonnet": ModelPlatformType.ANTHROPIC,
        "claude-3-7-sonnet": ModelPlatformType.ANTHROPIC,  # Add Claude
        "o3-mini": ModelPlatformType.OPENAI,  # Add O3-mini
        "gpt-4o": ModelPlatformType.OPENAI,  # Add GPT-4o
    }
    
    # Handle string representation of ModelPlatformType enum
    if model_name and model_name.upper() in [p.name for p in ModelPlatformType]:
        try:
            return ModelPlatformType[model_name.upper()]
        except (KeyError, AttributeError):
            pass
    
    model_platform = model_platform_map.get(model_name)
    if not model_platform:
        raise ModelError(f"Unknown model name: {model_name}")
    return model_platform


def get_api_key(model_type: ModelType, model_platform: ModelPlatformType) -> str:
    """Get the API key for a model platform.
    
    Args:
        model_type: The model type.
        model_platform: The model platform.
        
    Returns:
        The API key, or None if not found.
        
    Raises:
        ModelError: If the API key is not found.
    """
    platform_name = model_platform.name if model_platform else "OPENAI"
    api_key = os.getenv(f"{platform_name}_API_KEY")
    if not api_key:
        raise ModelError(f"API key not found for {model_type} on {model_platform}")
    return api_key


def with_backup_model(
    func: Callable[..., T]
) -> Callable[..., T]:
    """Decorator to retry a function with a backup model if the primary model fails.
    
    Args:
        func: The function to decorate.
        
    Returns:
        The decorated function.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Extract model_type and model_platform from kwargs
        model_type = kwargs.get("model_type")
        model_platform = kwargs.get("model_platform")
        backup_model_type = kwargs.get("backup_model_type")
        backup_model_platform = kwargs.get("backup_model_platform")
        
        # Check if the model_type and model_platform were passed as arguments
        if not model_type or not model_platform:
            raise ModelError("model_type and model_platform are required arguments for with_backup_model")

        # Try with primary model
        try:
            # Log primary model usage
            logger.info(f"Using primary model {model_type.name} on {model_platform.name}")

            # Record with AgentOps
            try:
                if AGENTOPS_AVAILABLE:
                    agentops.record(agentops.ActionEvent(
                        "model_usage",
                        {
                            "model_type": model_type.name,
                            "model_platform": model_platform.name,
                            "primary": True,
                        }
                    ))
            except Exception as e:
                logger.warning(f"Failed to record model usage with AgentOps: {str(e)}")

            # Call the function with the primary model
            result = func(
                *args,
                **kwargs
            )

            return result

        except Exception as e:
            # If no backup is available, re-raise the exception
            if not backup_model_type:
                logger.error(f"Primary model failed and no backup specified: {str(e)}")
                raise

            # Use the same platform as primary if not specified
            actual_backup_platform = backup_model_platform or model_platform

            # Log backup model usage
            logger.warning(
                f"Primary model {model_type.name} failed: {str(e)}. "
                f"Falling back to {backup_model_type.name} on {actual_backup_platform.name}"
            )

            try:
                # Record with AgentOps
                try:
                    if AGENTOPS_AVAILABLE:
                        agentops.record(agentops.ActionEvent(
                            "model_usage",
                            {
                                "model_type": backup_model_type.name,
                                "model_platform": actual_backup_platform.name,
                                "primary": False,
                                "fallback_reason": str(e),
                            }
                        ))
                except Exception as e:
                    logger.warning(f"Failed to record model usage with AgentOps: {str(e)}")

                # Save original model type and platform
                original_model_type = kwargs.get("model_type")
                original_model_platform = kwargs.get("model_platform")
                
                # Replace with backup model
                kwargs["model_type"] = backup_model_type
                kwargs["model_platform"] = actual_backup_platform

                # Call the function with the backup model
                result = func(
                    *args,
                    **kwargs
                )
                
                # Restore original values
                kwargs["model_type"] = original_model_type
                kwargs["model_platform"] = original_model_platform

                return result

            except Exception as backup_e:
                # Both models failed
                logger.error(
                    f"Both models failed: "
                    f"Primary ({model_type.name}): {str(e)}, "
                    f"Backup ({backup_model_type.name}): {str(backup_e)}"
                )

                raise ModelBackupError(
                    message=f"Both models failed: Primary: {str(e)}, Backup: {str(backup_e)}",
                    primary_model=f"{model_type.name}_{model_platform.name}",
                    backup_model=f"{backup_model_type.name}_{actual_backup_platform.name}",
                    operation=func.__name__
                )
    
    return wrapper


def get_chat_agent(
    model_type: ModelType,
    model_platform: ModelPlatformType,
    api_key: Optional[str] = None,
    role_type = None,
    **kwargs: Any
) -> BaseModelBackend:
    """Get a chat agent with the specified model configuration.

    Args:
        model_type: Type of model (e.g. GPT_4, CLAUDE_3_SONNET)
        model_platform: Platform to use (e.g. OPENAI, ANTHROPIC)
        api_key: Optional API key (if not provided, will be retrieved from env)
        role_type: Optional role type for the agent
        **kwargs: Additional arguments to pass to the model factory

    Returns:
        BaseModelBackend: The configured chat agent
        
    Raises:
        ModelError: If failed to create the chat agent
    """
    try:
        # Get model name from type/platform
        model_name = f"{model_type.name.lower()}-{model_platform.name.lower()}"
        
        # Use provided API key or get from environment
        if api_key is None:
            api_key = get_api_key(model_type, model_platform)
        
        # Create and return the model
        return ModelFactory.create(
            model_type=model_type,
            model_platform=model_platform,
            api_key=api_key,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Failed to create chat agent: {str(e)}")
        raise ModelError(f"Failed to create chat agent: {str(e)}", str(model_type))
