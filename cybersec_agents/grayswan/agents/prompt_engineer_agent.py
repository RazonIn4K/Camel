"""
Prompt Engineer Agent for Gray Swan Arena.

This agent is responsible for generating effective attack prompts
based on reconnaissance data about target AI models.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import logging

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelType, ModelPlatformType
from camel.types.enums import RoleType

# Import agentops directly
import agentops

from ..utils.logging_utils import setup_logging
# Import the ModelConfigManager
from ..utils.model_config_manager import ModelConfigManager
from ..utils.model_utils import get_chat_agent, _get_model_type, _get_model_platform

# Set up logging
logger = logging.getLogger(__name__)


class PromptEngineerAgent:
    """Agent for generating attack prompts for target AI models."""

    def __init__(
        self,
        output_dir: str = "./prompts",
        model_type: Optional[ModelType] = None,
        model_platform: Optional[ModelPlatformType] = None,
        model_name: Optional[str] = None,
        backup_model_type: Optional[ModelType] = None,
        backup_model_platform: Optional[ModelPlatformType] = None,
        reasoning_model_type: Optional[ModelType] = None,
        reasoning_model_platform: Optional[ModelPlatformType] = None,
        reasoning_model: Optional[str] = None,
        config_file: Optional[str] = None,
        config_dir: Optional[str] = None,
        **model_kwargs
    ):
        """Initialize the PromptEngineerAgent.

        Args:
            output_dir: Directory to save generated prompts to
            model_type: Type of model to use (e.g. GPT_4, CLAUDE_3_SONNET)
            model_platform: Platform to use (e.g. OPENAI, ANTHROPIC)
            model_name: Optional name of the model to use (for backwards compatibility)
            backup_model_type: Type of backup model to use if primary fails
            backup_model_platform: Platform of backup model
            reasoning_model_type: Type of model to use for reasoning tasks
            reasoning_model_platform: Platform of reasoning model
            reasoning_model: Optional name of reasoning model (for backwards compatibility)
            config_file: Optional path to a model configuration file
            config_dir: Optional directory for the configuration file
            **model_kwargs: Additional model parameters that override configuration
        """
        self.output_dir = Path(output_dir)
        
        # Determine model parameters
        self.model_name = model_name
        self.model_type = model_type
        self.model_platform = model_platform
        
        # If model_type/platform not provided but model_name is, derive them from model_name
        if (self.model_type is None or self.model_platform is None) and self.model_name:
            from ..utils.model_utils import _get_model_type, _get_model_platform
            self.model_type = self.model_type or _get_model_type(self.model_name)
            self.model_platform = self.model_platform or _get_model_platform(self.model_name)
        
        # Setup backup model configuration
        self.backup_model_type = backup_model_type
        self.backup_model_platform = backup_model_platform
        
        # Determine reasoning model parameters
        self.reasoning_model = reasoning_model
        self.reasoning_model_type = reasoning_model_type or self.model_type
        self.reasoning_model_platform = reasoning_model_platform or self.model_platform
        
        # If reasoning model name is provided but not type/platform, derive them
        if (self.reasoning_model_type is None or self.reasoning_model_platform is None) and self.reasoning_model:
            from ..utils.model_utils import _get_model_type, _get_model_platform
            self.reasoning_model_type = self.reasoning_model_type or _get_model_type(self.reasoning_model)
            self.reasoning_model_platform = self.reasoning_model_platform or _get_model_platform(self.reasoning_model)
        
        # Add AGENTOPS_AVAILABLE flag
        self.AGENTOPS_AVAILABLE = 'AGENTOPS_API_KEY' in os.environ
        
        # Store additional model parameters
        self.model_kwargs = model_kwargs

        # Initialize the ModelConfigManager
        self.model_manager = ModelConfigManager()
        
        # Get the model configurations
        if self.model_type and self.model_platform:
            self.model_config = self.model_manager.get_model_params(self.model_type, self.model_platform) or {}
        else:
            self.model_config = {}
            
        if self.reasoning_model_type and self.reasoning_model_platform:
            self.reasoning_model_config = self.model_manager.get_model_params(
                self.reasoning_model_type,
                self.reasoning_model_platform
            ) or {}
        else:
            self.reasoning_model_config = {}
        
        # Log model configuration
        logger.info("Using model configuration for %s/%s: %s",
                   self.model_type,
                   self.model_platform,
                   self.model_config)
        if self.reasoning_model_type != self.model_type or self.reasoning_model_platform != self.model_platform:
            logger.info("Using reasoning model configuration for %s/%s: %s",
                      self.reasoning_model_type,
                      self.reasoning_model_platform,
                      self.reasoning_model_config)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log initialization
        try:
            if self.AGENTOPS_AVAILABLE:
                # Create appropriate tags for this agent
                agent_tags = ["prompt_engineer_agent"]
                if self.model_type:
                    agent_tags.append(f"model_type:{self.model_type.name}")
                if self.model_platform:
                    agent_tags.append(f"model_platform:{self.model_platform.name}")
                
                agentops.record(agentops.ActionEvent(
                    "agent_initialized",
                    {
                        "agent_type": "prompt_engineer",
                        "output_dir": str(output_dir),
                        "model_type": self.model_type.name if self.model_type else None,
                        "model_platform": self.model_platform.name if self.model_platform else None,
                        "reasoning_model_type": self.reasoning_model_type.name if self.reasoning_model_type else None,
                        "reasoning_model_platform": self.reasoning_model_platform.name if self.reasoning_model_platform else None,
                        "model_config": self.model_config,
                    }
                ))
        except Exception as e:
            logger.warning(f"Failed to record agent initialization with AgentOps: {str(e)}")

        # Improved initialization log
        logger.info("PromptEngineerAgent initialized with model type: %s, platform: %s",
                   self.model_type,
                   self.model_platform)
        
        # If backup model is configured, log it
        if self.backup_model_type:
            logger.info("Backup model configured: %s/%s",
                       self.backup_model_type,
                       self.backup_model_platform)
            
        # If reasoning model differs from main model, log it
        if self.reasoning_model_type != self.model_type or self.reasoning_model_platform != self.model_platform:
            logger.info("Reasoning model configured: %s/%s",
                   self.reasoning_model_type,
                   self.reasoning_model_platform)

        self.templates = self.load_templates()

    def load_templates(self) -> Dict[str, Any]:
        """Load prompt templates from JSON file."""
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils",
            "prompt_templates.json"
        )
        
        try:
            with open(template_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading templates: {e}")
            return {"challenges": {}}
    
    def generate_prompts(self, challenge_name: str, model_name: str, recon_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate prompts for the specified challenge and model.
        
        Args:
            challenge_name: The name of the challenge.
            model_name: The name of the target model.
            recon_data: Optional reconnaissance data to inject into prompts.
            
        Returns:
            A list of generated prompts.
        """
        if not self.templates or "challenges" not in self.templates:
            return ["Default prompt due to missing templates."]
            
        # Get challenge templates
        challenges = self.templates["challenges"]
        if challenge_name not in challenges:
            return [f"No template found for challenge: {challenge_name}"]
        
        # Get model-specific template or default
        models = challenges[challenge_name]["models"]
        template = models.get(model_name, models.get("default", {})).get("template", "")
        
        if not template:
            return [f"No template found for model {model_name} in challenge {challenge_name}"]
        
        # Replace placeholders
        filled_template = self._replace_placeholders(template, challenge_name, model_name, recon_data)
        
        return [filled_template]
    
    def _replace_placeholders(self, template: str, challenge_name: str, model_name: str, 
                             recon_data: Optional[Dict[str, Any]] = None) -> str:
        """Replace placeholders in the template with actual values.
        
        Args:
            template: Template string with placeholders.
            challenge_name: The name of the challenge.
            model_name: The name of the model.
            recon_data: Optional reconnaissance data.
            
        Returns:
            Template with placeholders replaced.
        """
        # Define default values for common placeholders
        default_values = {
            "challenge_name": challenge_name,
            "model_name": model_name,
            "airbnb_username": "user1234",
            "airbnb_password": "vacation2023!",
            "safeguard_type": "prevent harmful outputs",
        }
        
        # If recon data is provided, override defaults
        if recon_data:
            default_values.update(recon_data)
        
        # Find all placeholders in the template
        placeholders = re.findall(r'\{\{(\w+)\}\}', template)
        
        # Replace each placeholder
        result = template
        for placeholder in placeholders:
            if placeholder in default_values:
                result = result.replace(f"{{{{{placeholder}}}}}", str(default_values[placeholder]))
            else:
                # Leave unknown placeholders unchanged for now
                pass
                
        return result
