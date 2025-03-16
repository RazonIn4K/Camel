"""
Evaluation Agent for Gray Swan Arena.

This agent is responsible for analyzing exploit results, generating visualizations,
and producing comprehensive evaluation reports.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelType, ModelPlatformType
from camel.types.enums import RoleType

from ..utils.advanced_visualization_utils import (
    create_advanced_evaluation_report,
    create_attack_pattern_visualization,
    create_interactive_dashboard,
    create_prompt_similarity_network,
    create_success_prediction_model,
)

# Import specific utilities directly
from ..utils.logging_utils import setup_logging
from ..utils.model_factory import get_chat_agent
from ..utils.visualization_utils import create_evaluation_report
from ..utils.retry_utils import ExponentialBackoffRetryStrategy
from ..utils.retry_manager import RetryManager
from ..utils.model_utils import _get_model_type, _get_model_platform, get_api_key

# Set up logging using our logging utility
logger = setup_logging("evaluation_agent")


def _get_model_name_from_type(model_type: ModelType) -> str:
    """
    Map ModelType enum to model name string.
    
    Args:
        model_type: The model type enum value
        
    Returns:
        The corresponding model name string
    """
    model_name_map = {
        ModelType.GPT_4: "gpt-4",
        ModelType.GPT_4_TURBO: "gpt-4-turbo",
        ModelType.GPT_3_5_TURBO: "gpt-3.5-turbo",
        ModelType.CLAUDE_3_SONNET: "claude-3-sonnet",
        ModelType.CLAUDE_3_OPUS: "claude-3-opus",
        ModelType.O3_MINI: "o3-mini",
    }
    
    return model_name_map.get(model_type, "unknown-model")


class EvaluationAgent:
    """Agent responsible for evaluating exploit results and producing reports."""

    def __init__(
        self,
        output_dir: str = "./evaluations",
        model_type: Optional[ModelType] = None,
        model_platform: Optional[ModelPlatformType] = None,
        model_name: Optional[str] = None,
        backup_model_type: Optional[ModelType] = None,
        backup_model_platform: Optional[ModelPlatformType] = None,
        reasoning_model_type: Optional[ModelType] = None,
        reasoning_model_platform: Optional[ModelPlatformType] = None,
        reasoning_model: Optional[str] = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        **model_kwargs
    ):
        """Initialize the evaluation agent.

        Args:
            output_dir: Directory to store evaluation results.
            model_type: Type of model to use.
            model_platform: Platform to use.
            model_name: Optional name of the model to use (for backwards compatibility)
            backup_model_type: Type of backup model to use if primary fails.
            backup_model_platform: Platform of backup model.
            reasoning_model_type: Type of model to use for reasoning tasks.
            reasoning_model_platform: Platform of reasoning model.
            reasoning_model: Optional name of reasoning model (for backwards compatibility)
            max_retries: Maximum number of retries for model operations.
            initial_retry_delay: Initial delay between retries.
            **model_kwargs: Additional model configuration.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine model parameters
        self.model_name = model_name
        self.model_type = model_type
        self.model_platform = model_platform
        
        # If model_type/platform not provided but model_name is, derive them from model_name
        if (self.model_type is None or self.model_platform is None) and self.model_name:
            self.model_type = self.model_type or _get_model_type(self.model_name)
            self.model_platform = self.model_platform or _get_model_platform(self.model_name)
        
        # Set defaults if still None
        self.model_type = self.model_type or ModelType.GPT_4
        self.model_platform = self.model_platform or ModelPlatformType.OPENAI
        
        # Setup backup model configuration
        self.backup_model_type = backup_model_type
        self.backup_model_platform = backup_model_platform
        
        # Determine reasoning model parameters
        self.reasoning_model = reasoning_model
        self.reasoning_model_type = reasoning_model_type
        self.reasoning_model_platform = reasoning_model_platform
        
        # If reasoning model name is provided but not type/platform, derive them
        if (self.reasoning_model_type is None or self.reasoning_model_platform is None) and self.reasoning_model:
            self.reasoning_model_type = self.reasoning_model_type or _get_model_type(self.reasoning_model)
            self.reasoning_model_platform = self.reasoning_model_platform or _get_model_platform(self.reasoning_model)
        
        # Default reasoning model to main model if not specified
        self.reasoning_model_type = self.reasoning_model_type or self.model_type
        self.reasoning_model_platform = self.reasoning_model_platform or self.model_platform
        
        self.model_kwargs = model_kwargs

        self.retry_strategy = ExponentialBackoffRetryStrategy(
            max_retries=max_retries,
            initial_delay=initial_retry_delay,
        )
        self.retry_manager = RetryManager(self.retry_strategy)
        
        # Set API key
        model_name = self._get_model_name_from_type(self.model_type)
        self.api_key = get_api_key(self.model_type, self.model_platform)

        # Create chat agent
        self.chat_agent = get_chat_agent(
            model_type=self.model_type,
            model_platform=self.model_platform,
            role_type=RoleType.ASSISTANT,
            api_key=self.api_key,
            **self.model_kwargs,
        )
        
        # Add a unique agent_id for tracking retry operations
        self.agent_id = f"evaluation_agent_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create visualizations directory
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Improved initialization log
        logger.info("EvaluationAgent initialized with model type: %s, platform: %s",
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
        
        # Log retry configuration
        logger.info("Retry configuration: max_retries=%d, initial_delay=%f",
                   max_retries,
                   initial_retry_delay)
        
    def _get_model_name_from_type(self, model_type: ModelType) -> str:
        """Convert ModelType to model name string.

        Args:
            model_type: ModelType enum value

        Returns:
            str: Model name as a string
        """
        model_type_to_name = {
            ModelType.GPT_4: "gpt-4",
            ModelType.GPT_4_TURBO: "gpt-4-turbo",
            ModelType.GPT_3_5_TURBO: "gpt-3.5-turbo",
            ModelType.CLAUDE_3_SONNET: "claude-3-sonnet",
            ModelType.CLAUDE_3_OPUS: "claude-3-opus",
        }
        return model_type_to_name.get(model_type, "gpt-4")  # Default to gpt-4 if unknown

    def evaluate_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate results from edge case tests.

        Args:
            test_results: List of test results to evaluate

        Returns:
            Dict containing evaluation results
        """
        logger.info("Starting evaluation of test results")
        
        try:
            # Generate evaluation report
            evaluation_report = create_evaluation_report(
                results=test_results,
                output_dir=self.viz_dir,
                evaluation_agent=self.chat_agent,
                reasoning_agent=self.chat_agent
            )
            
            # Generate advanced visualizations
            advanced_report = create_advanced_evaluation_report(
                results=test_results,
                output_dir=self.viz_dir,
                evaluation_agent=self.chat_agent,
                reasoning_agent=self.chat_agent
            )
            
            # Create interactive dashboard
            dashboard = create_interactive_dashboard(
                results=test_results,
                output_dir=self.viz_dir,
                evaluation_agent=self.chat_agent
            )
            
            # Generate prompt similarity network
            similarity_network = create_prompt_similarity_network(
                results=test_results,
                output_dir=self.viz_dir,
                evaluation_agent=self.chat_agent
            )
            
            # Create success prediction model
            prediction_model = create_success_prediction_model(
                results=test_results,
                output_dir=self.viz_dir,
                evaluation_agent=self.chat_agent
            )
            
            # Combine all results
            evaluation_results = {
                "basic_report": evaluation_report,
                "advanced_report": advanced_report,
                "dashboard": dashboard,
                "similarity_network": similarity_network,
                "prediction_model": prediction_model,
                "timestamp": datetime.now().isoformat(),
                "model_type": str(self.model_type),
                "reasoning_model_type": str(self.reasoning_model_type)
            }
            
            # Save evaluation results
            self.save_evaluation(evaluation_results)
            
            logger.info("Successfully completed evaluation of test results")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            if self.backup_model_type:
                logger.info(f"Attempting evaluation with backup model: {self.backup_model_type}")
                try:
                    # Ensure backup model type is never None before using it
                    if self.backup_model_type is None:
                        self.backup_model_type = ModelType.GPT_4  # Default fallback
                        
                    backup_model_platform = self.backup_model_platform or self.model_platform
                    if backup_model_platform is None:
                        backup_model_platform = ModelPlatformType.OPENAI  # Default fallback
                        
                    # Create backup evaluation agent
                    backup_agent = get_chat_agent(
                        model_type=self.backup_model_type,
                        model_platform=backup_model_platform,
                        role_type=RoleType.ASSISTANT,
                        api_key=get_api_key(self.backup_model_type, backup_model_platform),
                        **self.model_kwargs
                    )
                    
                    # Retry evaluation with backup model
                    evaluation_results = self._retry_evaluation_with_backup(
                        test_results=test_results,
                        backup_agent=backup_agent
                    )
                    
                    logger.info("Successfully completed evaluation with backup model")
                    return evaluation_results
                    
                except Exception as backup_error:
                    logger.error(f"Backup model evaluation failed: {str(backup_error)}")
                    raise
                        else:
                raise

    def generate_summary(self, test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate a summary for each prompt/response pair.
        
        Args:
            test_results: List of test results to summarize
            
        Returns:
            List of dictionaries containing summaries
        """
        summaries = []
        for result in test_results:
            summary = {
                "prompt": result.get("prompt", ""),
                "response": result.get("response", ""),
                "success": result.get("success", False),
                "response_time": result.get("response_time", 0),
                "model_type": result.get("model_type", "Unknown"),
                "prompt_type": result.get("prompt_type", ""),
                "attack_vector": result.get("attack_vector", ""),
                "timestamp": datetime.now().isoformat()
            }
            summaries.append(summary)
        return summaries

    def create_visualizations(self, test_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create visualizations for the test results.
        
        Args:
            test_results: List of test results to visualize
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualization_paths = {}
        
        # Create success rate chart
        success_rate_path = self._create_success_rate_chart(test_results)
        if success_rate_path:
            visualization_paths["success_rate"] = success_rate_path
            
        # Create response time chart
        response_time_path = self._create_response_time_chart(test_results)
        if response_time_path:
            visualization_paths["response_time"] = response_time_path
            
        # Create prompt type effectiveness chart
        prompt_effectiveness_path = self._create_prompt_type_effectiveness_chart(test_results)
        if prompt_effectiveness_path:
            visualization_paths["prompt_effectiveness"] = prompt_effectiveness_path
            
        # Create vulnerability heatmap
        vulnerability_heatmap_path = self._create_vulnerability_heatmap(test_results)
        if vulnerability_heatmap_path:
            visualization_paths["vulnerability_heatmap"] = vulnerability_heatmap_path
            
        return visualization_paths

    def _create_success_rate_chart(self, test_results: List[Dict[str, Any]]) -> Optional[str]:
        """Create a chart showing success rates by model and prompt type.

        Args:
            test_results: List of test results to analyze

        Returns:
            Path to the saved chart file, or None if creation failed
        """
        try:
            # Group results by model and prompt type
            success_data = {}
            for result in test_results:
                model = result.get("model_type", "Unknown")
                prompt_type = result.get("prompt_type", "Unknown")
                success = result.get("success", False)
                
                if model not in success_data:
                    success_data[model] = {}
                if prompt_type not in success_data[model]:
                    success_data[model][prompt_type] = {"success": 0, "total": 0}
                    
                success_data[model][prompt_type]["total"] += 1
                if success:
                    success_data[model][prompt_type]["success"] += 1
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot success rates
            models = list(success_data.keys())
            prompt_types = set()
            for model_data in success_data.values():
                prompt_types.update(model_data.keys())
            prompt_types = sorted(list(prompt_types))
            
            x = np.arange(len(models))
            width = 0.8 / len(prompt_types)
            
            for i, prompt_type in enumerate(prompt_types):
                success_rates = []
                for model in models:
                    if prompt_type in success_data[model]:
                        data = success_data[model][prompt_type]
                        rate = data["success"] / data["total"]
                        success_rates.append(rate)
                    else:
                        success_rates.append(0)
                
                plt.bar(x + i * width, success_rates, width, label=prompt_type)
            
            plt.xlabel("Model")
            plt.ylabel("Success Rate")
            plt.title("Success Rates by Model and Prompt Type")
            plt.xticks(x + width * (len(prompt_types) - 1) / 2, models, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"success_rate_{timestamp}_{self.agent_id}.png"
            filepath = self.viz_dir / filename
            plt.savefig(filepath)
            plt.close()
            
        return str(filepath)

        except Exception as e:
            logger.error(f"Failed to create success rate chart: {str(e)}")
            return None

    def _create_response_time_chart(self, test_results: List[Dict[str, Any]]) -> Optional[str]:
        """Create a chart showing response times by model and prompt type.

        Args:
            test_results: List of test results to analyze

        Returns:
            Path to the saved chart file, or None if creation failed
        """
        try:
            # Group results by model and prompt type
            time_data = {}
            for result in test_results:
                model = result.get("model_type", "Unknown")
                prompt_type = result.get("prompt_type", "Unknown")
                response_time = result.get("response_time", 0)
                
                if model not in time_data:
                    time_data[model] = {}
                if prompt_type not in time_data[model]:
                    time_data[model][prompt_type] = []
                    
                time_data[model][prompt_type].append(response_time)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot response times
            models = list(time_data.keys())
            prompt_types = set()
            for model_data in time_data.values():
                prompt_types.update(model_data.keys())
            prompt_types = sorted(list(prompt_types))
            
            x = np.arange(len(models))
            width = 0.8 / len(prompt_types)
            
            for i, prompt_type in enumerate(prompt_types):
                avg_times = []
                for model in models:
                    if prompt_type in time_data[model]:
                        times = time_data[model][prompt_type]
                        avg_time = np.mean(times)
                        avg_times.append(avg_time)
                    else:
                        avg_times.append(0)
                
                plt.bar(x + i * width, avg_times, width, label=prompt_type)
            
            plt.xlabel("Model")
            plt.ylabel("Average Response Time (s)")
            plt.title("Response Times by Model and Prompt Type")
            plt.xticks(x + width * (len(prompt_types) - 1) / 2, models, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_time_{timestamp}_{self.agent_id}.png"
            filepath = self.viz_dir / filename
            plt.savefig(filepath)
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create response time chart: {str(e)}")
            return None

    def _create_prompt_type_effectiveness_chart(self, test_results: List[Dict[str, Any]]) -> Optional[str]:
        """Create a chart showing prompt type effectiveness.

        Args:
            test_results: List of test results to analyze

        Returns:
            Path to the saved chart file, or None if creation failed
        """
        try:
            # Group results by prompt type
            effectiveness_data = {}
            for result in test_results:
                prompt_type = result.get("prompt_type", "Unknown")
                success = result.get("success", False)
                
                if prompt_type not in effectiveness_data:
                    effectiveness_data[prompt_type] = {"success": 0, "total": 0}
                    
                effectiveness_data[prompt_type]["total"] += 1
                if success:
                    effectiveness_data[prompt_type]["success"] += 1
            
            # Calculate success rates
            prompt_types = list(effectiveness_data.keys())
            success_rates = []
            for prompt_type in prompt_types:
                data = effectiveness_data[prompt_type]
                rate = data["success"] / data["total"]
                success_rates.append(rate)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            plt.bar(prompt_types, success_rates)
            plt.xlabel("Prompt Type")
            plt.ylabel("Success Rate")
            plt.title("Prompt Type Effectiveness")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_effectiveness_{timestamp}_{self.agent_id}.png"
            filepath = self.viz_dir / filename
            plt.savefig(filepath)
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create prompt type effectiveness chart: {str(e)}")
            return None

    def _create_vulnerability_heatmap(self, test_results: List[Dict[str, Any]]) -> Optional[str]:
        """Create a heatmap showing vulnerability patterns.

        Args:
            test_results: List of test results to analyze

        Returns:
            Path to the saved chart file, or None if creation failed
        """
        try:
            # Group results by model and attack vector
            vulnerability_data = {}
            for result in test_results:
                model = result.get("model_type", "Unknown")
                attack_vector = result.get("attack_vector", "Unknown")
                success = result.get("success", False)
                
                if model not in vulnerability_data:
                    vulnerability_data[model] = {}
                if attack_vector not in vulnerability_data[model]:
                    vulnerability_data[model][attack_vector] = {"success": 0, "total": 0}
                    
                vulnerability_data[model][attack_vector]["total"] += 1
                if success:
                    vulnerability_data[model][attack_vector]["success"] += 1
            
            # Create success rate matrix
            models = sorted(list(vulnerability_data.keys()))
            attack_vectors = set()
            for model_data in vulnerability_data.values():
                attack_vectors.update(model_data.keys())
            attack_vectors = sorted(list(attack_vectors))
            
            success_rates = np.zeros((len(models), len(attack_vectors)))
            for i, model in enumerate(models):
                for j, attack_vector in enumerate(attack_vectors):
                    if attack_vector in vulnerability_data[model]:
                        data = vulnerability_data[model][attack_vector]
                        success_rates[i, j] = data["success"] / data["total"]
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                success_rates,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                xticklabels=attack_vectors,
                yticklabels=models
            )
            plt.xlabel("Attack Vector")
            plt.ylabel("Model")
            plt.title("Vulnerability Heatmap")
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vulnerability_heatmap_{timestamp}_{self.agent_id}.png"
            filepath = self.viz_dir / filename
            plt.savefig(filepath)
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to create vulnerability heatmap: {str(e)}")
            return None

    def _retry_evaluation_with_backup(
        self,
        test_results: List[Dict[str, Any]],
        backup_agent: ChatAgent
    ) -> Dict[str, Any]:
        """Retry evaluation using the backup model.
        
        Args:
            test_results: List of test results to evaluate
            backup_agent: Backup agent to use for evaluation
            
        Returns:
            Dict containing evaluation results
        """
        # Ensure reasoning model type is never None
        if self.reasoning_model_type is None:
            self.reasoning_model_type = ModelType.GPT_4  # Default fallback
            
        reasoning_model_platform = self.reasoning_model_platform or self.model_platform
        if reasoning_model_platform is None:
            reasoning_model_platform = ModelPlatformType.OPENAI  # Default fallback
        
        # Get model name for API key using non-None model type
        reasoning_model_name = self._get_model_name_from_type(self.reasoning_model_type)
        
        # Create reasoning agent for analysis with non-None values
        reasoning_agent = get_chat_agent(
            model_type=self.reasoning_model_type,
            model_platform=reasoning_model_platform,
            role_type=RoleType.ASSISTANT,
            api_key=get_api_key(self.reasoning_model_type, reasoning_model_platform),
            **self.model_kwargs
        )
        
        # Generate evaluation report
        evaluation_report = create_evaluation_report(
            results=test_results,
            output_dir=self.viz_dir,
            evaluation_agent=backup_agent,
            reasoning_agent=reasoning_agent
        )
        
        # Generate advanced visualizations
        advanced_report = create_advanced_evaluation_report(
            results=test_results,
            output_dir=self.viz_dir,
            evaluation_agent=backup_agent,
            reasoning_agent=reasoning_agent
        )
        
        # Create interactive dashboard
        dashboard = create_interactive_dashboard(
            results=test_results,
            output_dir=self.viz_dir,
            evaluation_agent=backup_agent
        )
        
        # Generate prompt similarity network
        similarity_network = create_prompt_similarity_network(
            results=test_results,
            output_dir=self.viz_dir,
            evaluation_agent=backup_agent
        )
        
        # Create success prediction model
        prediction_model = create_success_prediction_model(
            results=test_results,
            output_dir=self.viz_dir,
            evaluation_agent=backup_agent
        )
        
        # Combine all results
        evaluation_results = {
            "basic_report": evaluation_report,
            "advanced_report": advanced_report,
            "dashboard": dashboard,
            "similarity_network": similarity_network,
            "prediction_model": prediction_model,
            "timestamp": datetime.now().isoformat(),
            "model_type": str(self.backup_model_type),
            "reasoning_model_type": str(self.reasoning_model_type),
            "backup_used": True
        }
        
        # Save evaluation results
        self.save_evaluation(evaluation_results)
        
        return evaluation_results

    def save_evaluation(self, evaluation_results: Dict[str, Any]) -> None:
        """Save evaluation results to a JSON file.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}_{self.agent_id}_{self.model_type}.json"
        filepath = self.viz_dir / filename
        
        try:
            with open(filepath, "w") as f:
                json.dump(evaluation_results, f, indent=2)
            logger.info(f"Evaluation results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results to file: {str(e)}")
            raise

    def create_advanced_visualizations(
        self,
        test_results: List[Dict[str, Any]],
        evaluation_agent: ChatAgent,
        reasoning_agent: ChatAgent
    ) -> Dict[str, Any]:
        """Create advanced visualizations for the test results.

        Args:
            test_results: List of test results to visualize
            evaluation_agent: Agent to use for evaluation
            reasoning_agent: Agent to use for reasoning tasks

        Returns:
            Dictionary containing visualization data
        """
        try:
            # Create attack pattern visualization
            attack_pattern = create_attack_pattern_visualization(
                results=test_results,
                output_dir=self.viz_dir,
                evaluation_agent=evaluation_agent
            )
            
            # Create prompt similarity network
            similarity_network = create_prompt_similarity_network(
                results=test_results,
                output_dir=self.viz_dir,
                evaluation_agent=evaluation_agent
            )
            
            # Create success prediction model
            prediction_model = create_success_prediction_model(
                results=test_results,
                output_dir=self.viz_dir,
                evaluation_agent=evaluation_agent
            )
            
            # Create interactive dashboard
            dashboard = create_interactive_dashboard(
                results=test_results,
                output_dir=self.viz_dir,
                evaluation_agent=evaluation_agent
            )
            
            # Combine all visualizations
            visualizations = {
                "attack_pattern": attack_pattern,
                "similarity_network": similarity_network,
                "prediction_model": prediction_model,
                "dashboard": dashboard,
                "timestamp": datetime.now().isoformat()
            }
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Failed to create advanced visualizations: {str(e)}")
            return {"error": str(e)}

    def _extract_success_rate(self, content: str) -> float:
        """
        Extract the success rate from the analysis content.

        Args:
            content: The analysis content

        Returns:
            The success rate as a float between 0 and 1
        """
        try:
            # Look for patterns like "success rate: 25%" or "success rate of 25%"
            import re
            success_pattern = r"success rate:?\s*(?:of\s*)?(\d+(?:\.\d+)?)%"
            match = re.search(success_pattern, content.lower())
            
            if match:
                return float(match.group(1)) / 100.0
            
            # Try alternate pattern like "25% of prompts were successful"
            alt_pattern = r"(\d+(?:\.\d+)?)%\s*(?:of\s*)?(?:the\s*)?prompts"
            alt_match = re.search(alt_pattern, content.lower())
            
            if alt_match:
                return float(alt_match.group(1)) / 100.0
                
            # If no clear pattern is found, default to a conservative estimate
            # based on calculating successful vs. total prompts
            return 0.0
            
        except Exception as e:
            logger.error(f"Error extracting success rate: {str(e)}")
            return 0.0

    def _generate_visualizations(self, evaluation: Dict[str, Any]) -> None:
        """
        Generate visualizations for the evaluation.

        Args:
            evaluation: The evaluation data
        """
        try:
            # Basic bar chart of success rate
            successful = len(evaluation.get("successful_prompts", []))
            failed = len(evaluation.get("failed_prompts", []))
            total = successful + failed
            
            if total > 0:
                success_rate = successful / total
                evaluation["success_rate"] = success_rate
                evaluation["successful_attempts"] = successful
                evaluation["failed_attempts"] = failed
                
                logger.info(f"Generated visualization data: {successful} successful, {failed} failed, {success_rate:.2%} success rate")
            else:
                logger.warning("No prompt data available for visualization")
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract sections from report content.

        Args:
            content: Report content

        Returns:
            Dictionary of section titles and content
        """
        sections: dict[str, Any] = {}
        current_section: str = "preamble"
        current_content: list[Any] = []

        for line in content.split("\n"):
            if line.startswith("#"):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                    current_content = []

                # Extract new section title
                current_section = line.strip("# ").lower()
            else:
                current_content.append(line)

        # Save the last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections
