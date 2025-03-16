"""
Prompt Engineer Agent for Gray Swan Arena.

This agent is responsible for generating effective attack prompts
based on reconnaissance data about target AI models.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import logging

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelType, ModelPlatformType
from camel.types.enums import RoleType

# Import agentops directly
import agentops

from ..exceptions import ModelError
from ..utils.logging_utils import setup_logging
# Import the ModelConfigManager
from ..utils.model_config_manager import ModelConfigManager
# Fix: Update import to use functions without underscore prefix
from ..utils.model_utils import get_chat_agent, get_model_type, get_model_platform, with_backup_model

# Set up logging
logger = logging.getLogger(__name__)

# Agent event types
class AgentEventType:
    INITIALIZATION = "agent_initialized"
    PROMPT_GENERATION = "prompt_generated"
    TEMPLATE_LOADING = "templates_loaded"
    PLACEHOLDER_REPLACEMENT = "placeholders_replaced"

# Agent types
class AgentType:
    PROMPT_ENGINEER = "prompt_engineer"
    RECON = "recon"
    EXPLOIT_DELIVERY = "exploit_delivery" 
    EVALUATION = "evaluation"

def resolve_model_configuration(
    model_type: Optional[ModelType] = None,
    model_platform: Optional[ModelPlatformType] = None,
    model_name: Optional[str] = None
) -> Tuple[ModelType, ModelPlatformType]:
    """Resolve model configuration from various inputs.
    
    This function determines the model type and platform based on provided inputs,
    with fallbacks to ensure valid configuration is returned.
    
    Args:
        model_type: Type of model to use (e.g., GPT_4, CLAUDE_3_SONNET)
        model_platform: Platform to use (e.g., OPENAI, ANTHROPIC)
        model_name: Optional name of model to use (for backwards compatibility)
        
    Returns:
        Tuple containing (ModelType, ModelPlatformType)
    """
    # If both type and platform provided, use them
    if model_type is not None and model_platform is not None:
        return model_type, model_platform
        
    # If model_name is provided, derive type and platform
    # Fix: Update function calls to use versions without underscore prefix
    if model_name:
        derived_type = get_model_type(model_name)
        derived_platform = get_model_platform(model_name)
        
        # Use derived values for any missing parameters
        return model_type or derived_type, model_platform or derived_platform
        
    # Default fallback
    return ModelType.GPT_4_TURBO, ModelPlatformType.OPENAI

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
        template_path: Optional[str] = None,
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
            template_path: Optional path to the prompt templates file
            **model_kwargs: Additional model parameters that override configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Resolve primary model configuration
        self.model_type, self.model_platform = resolve_model_configuration(
            model_type, model_platform, model_name)
        
        # Store original model name for backwards compatibility
        self.model_name = model_name
        
        # Resolve backup model configuration
        if backup_model_type or backup_model_platform:
            self.backup_model_type = backup_model_type
            self.backup_model_platform = backup_model_platform or self.model_platform
        else:
            self.backup_model_type = None
            self.backup_model_platform = None
        
        # Resolve reasoning model configuration
        self.reasoning_model_type, self.reasoning_model_platform = resolve_model_configuration(
            reasoning_model_type, reasoning_model_platform, reasoning_model)
        
        # Store original reasoning model name for backwards compatibility  
        self.reasoning_model = reasoning_model
        
        # Check if AgentOps is available
        self.AGENTOPS_AVAILABLE = 'AGENTOPS_API_KEY' in os.environ
        
        # Store additional model parameters
        self.model_kwargs = model_kwargs

        # Initialize the ModelConfigManager
        self.model_manager = ModelConfigManager()
        
        # Get the model configurations
        self.model_config = self.model_manager.get_model_params(
            self.model_type, self.model_platform) or {}
            
        self.reasoning_model_config = self.model_manager.get_model_params(
            self.reasoning_model_type, self.reasoning_model_platform) or {}
        
        # Set template path
        self.template_path = template_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils",
            "prompt_templates.json"
        )
        
        # Log configuration details
        self._log_configuration()
        
        # Record initialization with AgentOps if available
        self._record_initialization()
        
        # Load templates
        self.templates = self.load_templates()

    def _log_configuration(self) -> None:
        """Log agent configuration details."""
        logger.info("PromptEngineerAgent initialized with model type: %s, platform: %s",
                   self.model_type, self.model_platform)
        
        logger.info("Using model configuration: %s", self.model_config)
        
        if self.backup_model_type:
            logger.info("Backup model configured: %s/%s",
                       self.backup_model_type, self.backup_model_platform)
            
        if self.reasoning_model_type != self.model_type or self.reasoning_model_platform != self.model_platform:
            logger.info("Reasoning model configured: %s/%s",
                      self.reasoning_model_type, self.reasoning_model_platform)
            logger.info("Using reasoning model configuration: %s", self.reasoning_model_config)

    def _record_initialization(self) -> None:
        """Record agent initialization with AgentOps."""
        if not self.AGENTOPS_AVAILABLE:
            return
            
        try:
            # Create appropriate tags for this agent
            agent_tags = ["prompt_engineer_agent"]
            if self.model_type:
                agent_tags.append(f"model_type:{self.model_type.name}")
            if self.model_platform:
                agent_tags.append(f"model_platform:{self.model_platform.name}")
            
            agentops.record(agentops.ActionEvent(
                AgentEventType.INITIALIZATION,
                {
                    "agent_type": AgentType.PROMPT_ENGINEER,
                    "output_dir": str(self.output_dir),
                    "model_type": self.model_type.name if self.model_type else None,
                    "model_platform": self.model_platform.name if self.model_platform else None,
                    "reasoning_model_type": self.reasoning_model_type.name if self.reasoning_model_type else None,
                    "reasoning_model_platform": self.reasoning_model_platform.name if self.reasoning_model_platform else None,
                    "model_config": self.model_config,
                }
            ))
        except Exception as e:
            logger.warning(f"Failed to record agent initialization with AgentOps: {str(e)}")

    def load_templates(self) -> Dict[str, Any]:
        """Load prompt templates from JSON file."""
        try:
            with open(self.template_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError as e:
            logger.error(f"Template file not found at {self.template_path}: {e}")
            raise FileNotFoundError(f"No prompt_templates.json found at {self.template_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in template file {self.template_path}: {e}")
            raise json.JSONDecodeError(f"Invalid JSON in template file", "", 0)
        except Exception as e:
            logger.error(f"Unexpected error loading templates: {e}")
            raise
    
    def generate_prompts(self, challenge_name: str, model_name: str, recon_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate prompts for the specified challenge and model.
        
        Args:
            challenge_name: The name of the challenge.
            model_name: The name of the target model.
            recon_data: Optional reconnaissance data to inject into prompts.
            
        Returns:
            A list of generated prompts.
        """
        if not self.templates:
            return ["Default prompt due to missing templates."]
        
        # Handle flat template structure (array of templates)    
        if "templates" in self.templates:
            # Extract the challenge name part after colon if present
            if ":" in challenge_name:
                challenge_type, challenge_short_name = challenge_name.split(":", 1)
                challenge_short_name = challenge_short_name.strip()
            else:
                challenge_short_name = challenge_name
                
            # Find templates that match the challenge
            matching_templates = []
            for template in self.templates["templates"]:
                if template.get("challenge") == challenge_short_name:
                    matching_templates.append(template)
            
            if not matching_templates:
                return [f"No template found for challenge: {challenge_name}"]
            
            results = []
            
            # Process matching templates
            for template in matching_templates:
                template_text = template.get("template", "")
                if template_text:
                    # Replace placeholders
                    filled_template = self._replace_placeholders(
                        template_text, 
                        challenge_name, 
                        model_name, 
                        template.get("name", ""),
                        template.get("strategy", ""),
                        recon_data
                    )
                    results.append(filled_template)
            
            if results:
                return results
            return [f"No valid templates found for challenge: {challenge_name}"]
        
        # Original nested template structure
        elif "challenges" in self.templates:
            # Original implementation for backward compatibility
            challenges = self.templates["challenges"]
            if challenge_name not in challenges:
                return [f"No template found for challenge: {challenge_name}"]
            
            # Get model-specific template or default
            models = challenges[challenge_name]["models"]
            template = models.get(model_name, models.get("default", {})).get("template", "")
            
            if not template:
                return [f"No template found for model {model_name} in challenge {challenge_name}"]
            
            # Replace placeholders
            filled_template = self._replace_placeholders(
                template, 
                challenge_name, 
                model_name, 
                "", 
                "", 
                recon_data
            )
            
            return [filled_template]
        
        return ["No templates found in the expected format."]
    
    def _replace_placeholders(self, template: str, challenge_name: str, model_name: str, 
                             template_name: str = "", strategy: str = "",
                             recon_data: Optional[Dict[str, Any]] = None) -> str:
        """Replace placeholders in the template with actual values.
        
        Args:
            template: Template string with placeholders.
            challenge_name: The name of the challenge.
            model_name: The name of the target model.
            template_name: The name of the template.
            strategy: The strategy used in the template.
            recon_data: Optional reconnaissance data.
            
        Returns:
            Template with placeholders replaced.
        """
        # Constants for default values
        DEFAULT_VALUES = {
            "challenge_name": challenge_name,
            "model_name": model_name,
            "template_name": template_name,
            "strategy": strategy,
            "airbnb_username": "user1234",
            "airbnb_password": "vacation2023!",
            "safeguard_type": "prevent harmful outputs",
            "harmful_activity": "create a secure password",
            "harmful_request": "explain quantum computing",
            "table_name": "users",
            "fill": "Please carefully read and follow these instructions."
        }
        
        # If recon data is provided, override defaults
        values = DEFAULT_VALUES.copy()
        if recon_data:
            values.update(recon_data)
        
        # Find all placeholders in the template
        placeholders = re.findall(r'\{\{(\w+)\}\}', template)
        
        # Special handling for {fill} in indirect prompts  
        if "{fill}" in template and "fill" in values:
            template = template.replace("{fill}", values["fill"])
        
        # Validate all placeholders have values
        unknown_placeholders = [p for p in placeholders if p not in values]
        if unknown_placeholders:
            logger.warning(f"Unknown placeholders in template: {unknown_placeholders}")
        
        # Replace each placeholder
        result = template
        for placeholder in placeholders:
            if placeholder in values:
                result = result.replace(f"{{{{{placeholder}}}}}", str(values[placeholder]))
            else:
                # Replace unknown placeholders with a warning text
                replacement = f"[MISSING VALUE FOR '{placeholder}']"
                result = result.replace(f"{{{{{placeholder}}}}}", replacement)
                
        return result

    @with_backup_model
    def generate_prompt(
        self,
        challenge_name: str,
        prompt_type: str,
        **model_kwargs
    ) -> str:
        """Generate a prompt for the specified challenge and prompt type.
        
        Args:
            challenge_name: The name of the challenge
            prompt_type: The type of prompt to generate
            **model_kwargs: Model parameters including model_type and model_platform
            
        Returns:
            Generated prompt text
            
        Raises:
            ModelError: If model parameters are missing
            ValueError: If no template is found for the challenge or prompt type
        """
        # Extract model_type and model_platform from model_kwargs
        model_type = model_kwargs.get("model_type")
        model_platform = model_kwargs.get("model_platform")
        
        # Check if the model_type and model_platform were passed as arguments
        if not model_type or not model_platform:
            raise ModelError("model_type and model_platform are required arguments for generate_prompt")
        
        logger.info(f"Generating prompt for challenge: {challenge_name}, prompt type: {prompt_type}")
        
        # Load the templates
        templates = self.load_templates()
        
        # Extract the challenge info
        challenge_info = templates.get(challenge_name)
        if not challenge_info:
            raise ValueError(f"No template found for challenge '{challenge_name}'.")
        
        # Extract the prompt type
        prompt_template = challenge_info.get(prompt_type)
        if not prompt_template:
            raise ValueError(f"No template found for prompt type '{prompt_type}' in challenge '{challenge_name}'.")
        
        # Extract the strategies
        strategies = challenge_info.get("strategies", [])
        
        # Create a dictionary with the placeholders
        placeholders = {
            "challenge": challenge_info.get("description"),
            "task": challenge_info.get("task"),
            "prompt_type": prompt_type,
            "strategies": " ".join(strategies),
            "model_type": str(model_type),
            "model_platform": str(model_platform)
        }
        
        # Replace the placeholders
        prompt = prompt_template
        for placeholder, value in placeholders.items():
            placeholder_str = f"{{{placeholder}}}"
            if placeholder_str in prompt:
                prompt = prompt.replace(placeholder_str, str(value))
        
        return prompt
