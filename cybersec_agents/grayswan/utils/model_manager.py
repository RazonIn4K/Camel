"""
Model configuration manager for Gray Swan Arena.

This module provides functionality for managing model configurations,
including loading from configuration files and retrieving parameters.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import time
import random
from functools import wraps

import yaml
from camel.models import BaseModelBackend
from camel.types import ModelPlatformType, ModelType

from cybersec_agents.grayswan.utils.config import load_config
from cybersec_agents.grayswan.utils.logging_utils import setup_logging
from cybersec_agents.grayswan.utils.model_factory import (
    create_model,
    get_chat_agent,
)
from cybersec_agents.grayswan.utils.model_type_mapping import get_default_model_type_for_agent
from cybersec_agents.grayswan.utils.model_utils import (
    _get_model_platform,
    _get_model_type,
    get_api_key,
)

# Import model errors for exception handling
from cybersec_agents.grayswan.exceptions import ModelError, RateLimitError, APIError, ModelBackupError

# Set up logging
logger = setup_logging("model_manager")

# Check if AgentOps is available
AGENTOPS_AVAILABLE = 'AGENTOPS_API_KEY' in os.environ

def with_exponential_backoff(func):
    """
    Decorator that implements exponential backoff for API rate limits.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function with exponential backoff retry logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Max retries ({max_retries}) exceeded for rate limit")
                    raise
                
                # Use the retry_after value if provided, otherwise use exponential backoff
                delay = e.retry_after if e.retry_after > 0 else base_delay * (2 ** attempt)
                # Add some jitter to prevent thundering herd
                delay = delay * (0.5 + random.random())
                
                logger.warning(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            except Exception:
                # Don't retry other exceptions
                raise
    
    return wrapper

class ModelConfigManager:
    """
    Manager for model configurations.
    
    This class provides functionality for loading model configurations from
    files and retrieving model parameters based on model type and platform.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ModelConfigManager':
        """
        Get the singleton instance of ModelConfigManager.
        
        Returns:
            The ModelConfigManager instance
        """
        if cls._instance is None:
            cls._instance = ModelConfigManager()
        return cls._instance
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the ModelConfigManager.
        
        Args:
            config_file: Path to the configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        # Load configuration from file if provided
        if config_file:
            self.load_config(config_file)
        else:
            # Load from environment variable or default file
            default_config_file = os.getenv("GRAYSWAN_CONFIG", "config/development.yml")
            if os.path.exists(default_config_file):
                self.load_config(default_config_file)
            else:
                self.logger.warning(f"Config file not found: {default_config_file}")
    
    def load_config(self, config_file: str) -> None:
        """
        Load model configurations from a file.
        
        Args:
            config_file: Path to the configuration file
        """
        try:
            self.config = load_config(config_file)
            self.logger.info(f"Loaded model configurations from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load model configurations: {str(e)}")
            self.config = {}
    
    def get_model_params(
        self, model_type: ModelType, model_platform: ModelPlatformType
    ) -> Dict[str, Any]:
        """
        Get model parameters for a specific model type and platform.
        
        Args:
            model_type: Type of the model
            model_platform: Platform of the model
            
        Returns:
            Dictionary of model parameters
        """
        # Get model parameters from configuration
        model_params = {}
        
        # Check if models section exists in config
        if "models" not in self.config:
            self.logger.warning("No models section found in configuration")
            return model_params
        
        # Get default model parameters if available
        if "default" in self.config["models"]:
            model_params.update(self.config["models"]["default"])
        
        # Get specific model parameters if available
        model_name = model_type.name
        if model_name in self.config["models"]:
            model_params.update(self.config["models"][model_name])
        
        return model_params
    
    def get_agent_model_params(
        self, agent_type: str
    ) -> Dict[str, Any]:
        """
        Get model parameters for a specific agent type.
        
        Args:
            agent_type: Type of the agent
            
        Returns:
            Dictionary of model parameters for the agent
        """
        # Get agent parameters from configuration
        agent_params = {}
        
        # Check if agents section exists in config
        if "agents" not in self.config:
            self.logger.warning("No agents section found in configuration")
            return agent_params
        
        # Get specific agent parameters if available
        if agent_type in self.config["agents"]:
            agent_params.update(self.config["agents"][agent_type])
        
        return agent_params
    
    def create_model_with_config(
        self, model_type: ModelType, model_platform: ModelPlatformType, **kwargs: Any
    ) -> BaseModelBackend:
        """
        Create a model instance with configuration.
        
        Args:
            model_type: Type of the model
            model_platform: Platform of the model
            **kwargs: Additional model configuration parameters
            
        Returns:
            A model instance
            
        Raises:
            ModelError: If model creation fails
        """
        # Get model parameters from configuration
        model_params = self.get_model_params(model_type, model_platform)
        
        # Override with provided parameters
        model_params.update(kwargs)
        
        # Get API key
        api_key = get_api_key(model_type, model_platform)
        
        # Create model
        try:
            # Try to create the model with the retrieved parameters
            model = create_model(
                model_type=model_type,
                model_platform=model_platform,
                api_key=api_key,
                **model_params
            )
            
            # Log model creation (without sensitive information)
            safe_params = {k: v for k, v in model_params.items() if k not in ['api_key']}
            self.logger.info(f"Created model {model_type.name} on {model_platform.name} with params: {safe_params}")
            
            return model
            
        except Exception as e:
            error_msg = f"Failed to create model {model_type.name} on {model_platform.name}: {str(e)}"
            self.logger.error(error_msg)
            
            # Raise as ModelError
            raise ModelError(error_msg) from e
    
    def get_agent_model_types(self, agent_type: str) -> Tuple[ModelType, ModelPlatformType]:
        """
        Get model type and platform for a specific agent type.
        
        Args:
            agent_type: Type of the agent
            
        Returns:
            Tuple of (model_type, model_platform)
        """
        # Get agent parameters from configuration
        agent_params = self.get_agent_model_params(agent_type)
        
        # Get model type from agent parameters or default
        model_type_str = agent_params.get("model_type")
        if model_type_str:
            model_type = _get_model_type(model_type_str)
        else:
            # Default model types based on agent type
            default_model_types = {
                "recon": ModelType.GPT_4,
                "prompt_engineer": ModelType.GPT_4_TURBO,
                "exploit_delivery": ModelType.GPT_4,
                "evaluation": ModelType.CLAUDE_3_SONNET,
            }
            model_type = default_model_types.get(agent_type, ModelType.GPT_4)
        
        # Get model platform from agent parameters or default
        model_platform_str = agent_params.get("model_platform")
        model_platform = _get_model_platform(model_platform_str) if model_platform_str else ModelPlatformType.OPENAI
        
        return model_type, model_platform
