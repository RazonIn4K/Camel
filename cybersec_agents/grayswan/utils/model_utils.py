"""
Utility functions for model interactions and error handling.

This module provides utility functions for interacting with different AI models
and handling errors in a consistent way.
"""

import logging
import os
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import agentops
from camel.models import BaseModelBackend
from camel.types import ModelPlatformType, ModelType

from cybersec_agents.grayswan.exceptions import ModelBackupError, ModelError, RateLimitError
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("model_utils")

# Check if AgentOps is available
AGENTOPS_AVAILABLE = 'AGENTOPS_API_KEY' in os.environ

# Type variable for generic function return type
T = TypeVar('T')


def _get_model_type(model_type_str: Optional[str]) -> ModelType:
    """Get a ModelType enum from a string.
    
    Args:
        model_type_str: String representation of the model type.
        
    Returns:
        A ModelType enum value.
        
    Raises:
        ValueError: If the model type is invalid.
    """
    if not model_type_str:
        return ModelType.GPT_4
    
    try:
        return ModelType[model_type_str.upper()]
    except (KeyError, AttributeError):
        logger.warning(f"Invalid model type: {model_type_str}. Using GPT_4 as default.")
        return ModelType.GPT_4


def _get_model_platform(model_platform_str: Optional[str]) -> ModelPlatformType:
    """Get a ModelPlatformType enum from a string.
    
    Args:
        model_platform_str: String representation of the model platform.
        
    Returns:
        A ModelPlatformType enum value.
        
    Raises:
        ValueError: If the model platform is invalid.
    """
    if not model_platform_str:
        return ModelPlatformType.OPENAI
    
    try:
        return ModelPlatformType[model_platform_str.upper()]
    except (KeyError, AttributeError):
        logger.warning(f"Invalid model platform: {model_platform_str}. Using OPENAI as default.")
        return ModelPlatformType.OPENAI


def get_api_key(model_type: ModelType, model_platform: ModelPlatformType) -> Optional[str]:
    """Get the API key for a model platform.
    
    Args:
        model_type: The model type.
        model_platform: The model platform.
        
    Returns:
        The API key, or None if not found.
    """
    # Map of platform types to environment variable names
    platform_to_env = {
        ModelPlatformType.OPENAI: "OPENAI_API_KEY",
        ModelPlatformType.ANTHROPIC: "ANTHROPIC_API_KEY",
        # Uncomment these if the platform types are available
        # ModelPlatformType.GOOGLE: "GOOGLE_API_KEY",
        # ModelPlatformType.PERPLEXITY: "PERPLEXITY_API_KEY",
        # ModelPlatformType.HUGGINGFACE: "HUGGINGFACE_API_KEY",
    }
    
    # Get environment variable name for the platform
    env_var = platform_to_env.get(model_platform)
    
    # If platform is not recognized, return None
    if not env_var:
        return None
    
    # Get API key from environment variable
    api_key = os.getenv(env_var)
    
    # Log if API key is missing (but don't leak the key if it exists)
    if not api_key:
        logger.warning(f"API key for {model_platform.name} not found in environment variable {env_var}")
    
    return api_key


def with_backup_model(
    primary_model_type: ModelType,
    primary_model_platform: ModelPlatformType,
    backup_model_type: Optional[ModelType] = None,
    backup_model_platform: Optional[ModelPlatformType] = None,
):
    """Decorator to retry a function with a backup model if the primary model fails.
    
    Args:
        primary_model_type: Type of the primary model.
        primary_model_platform: Platform of the primary model.
        backup_model_type: Type of the backup model. If None, no backup is used.
        backup_model_platform: Platform of the backup model. If None, uses the primary platform.
        
    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Try with primary model
            try:
                # Log primary model usage
                logger.info(f"Using primary model {primary_model_type.name} on {primary_model_platform.name}")
                
                # Record with AgentOps
                try:
                    if AGENTOPS_AVAILABLE:
                        agentops.record(agentops.ActionEvent(
                            "model_usage",
                            {
                                "model_type": primary_model_type.name,
                                "model_platform": primary_model_platform.name,
                                "primary": True,
                            }
                        ))
                except Exception as e:
                    logger.warning(f"Failed to record model usage with AgentOps: {str(e)}")
                
                # Call the function with the primary model
                result = func(
                    *args,
                    model_type=primary_model_type,
                    model_platform=primary_model_platform,
                    **kwargs
                )
                
                return result
                
            except Exception as e:
                # If no backup is available, re-raise the exception
                if not backup_model_type:
                    logger.error(f"Primary model failed and no backup specified: {str(e)}")
                    raise
                
                # Use the same platform as primary if not specified
                actual_backup_platform = backup_model_platform or primary_model_platform
                
                # Log backup model usage
                logger.warning(
                    f"Primary model {primary_model_type.name} failed: {str(e)}. "
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
                    
                    # Call the function with the backup model
                    result = func(
                        *args,
                        model_type=backup_model_type,
                        model_platform=actual_backup_platform,
                        **kwargs
                    )
                    
                    return result
                    
                except Exception as backup_e:
                    # Both models failed
                    logger.error(
                        f"Both models failed: "
                        f"Primary ({primary_model_type.name}): {str(e)}, "
                        f"Backup ({backup_model_type.name}): {str(backup_e)}"
                    )
                    
                    raise ModelBackupError(
                        message=f"Both models failed: Primary: {str(e)}, Backup: {str(backup_e)}",
                        primary_model=f"{primary_model_type.name}_{primary_model_platform.name}",
                        backup_model=f"{backup_model_type.name}_{actual_backup_platform.name}",
                        operation=func.__name__
                    )
        
        return wrapper
    
    return decorator


def get_chat_agent(
    model_name: str,
    model_type: ModelType,
    model_platform: ModelPlatformType,
    **kwargs: Any
) -> BaseModelBackend:
    """Get a chat agent with the specified model configuration.

    Args:
        model_name: Name of the model to use
        model_type: Type of model (e.g. GPT_4, CLAUDE_3_SONNET)
        model_platform: Platform to use (e.g. OPENAI, ANTHROPIC)
        **kwargs: Additional arguments to pass to the model factory

    Returns:
        BaseModelBackend: The configured chat agent
    """
    return ModelFactory.create(
        model_name=model_name,
        model_type=model_type,
        model_platform=model_platform,
        **kwargs
    )


def with_model_backup(func):
    """Decorator for retrying functions with a backup model.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if (hasattr(self, "backup_model_type") and self.backup_model_type and 
                hasattr(self, "backup_model_platform") and self.backup_model_platform):
                logger.warning(f"Primary model failed, trying backup model: {str(e)}")
                try:
                    # Save original model type and platform
                    original_model_type = self.model_type
                    original_model_platform = self.model_platform
                    
                    # Set backup model type and platform as active
                    self.model_type = self.backup_model_type
                    self.model_platform = self.backup_model_platform
                    
                    # Call function with backup model
                    result = func(self, *args, **kwargs)
                    
                    # Restore original model type and platform
                    self.model_type = original_model_type
                    self.model_platform = original_model_platform
                    
                    return result
                except Exception as backup_e:
                    # If both models fail, raise a ModelBackupError
                    raise ModelBackupError(
                        f"Both primary and backup models failed. Primary: {str(e)}, Backup: {str(backup_e)}",
                        primary_model_type=self.model_type,
                        primary_model_platform=self.model_platform,
                        backup_model_type=self.backup_model_type,
                        backup_model_platform=self.backup_model_platform,
                        operation="execute",
                        details={
                            "primary_error": str(e),
                            "backup_error": str(backup_e),
                        },
                    )
            raise

    return wrapper
