"""Base Attack class for Gray Swan Arena.

This module provides the base Attack class that all specific attack implementations
should inherit from. It defines the common interface and functionality that all
attacks must implement.
"""

import abc
import logging
from typing import Dict, Any, List, Optional

from ..utils.agentops import agentops

# Set up logger
logger = logging.getLogger(__name__)


class Attack(abc.ABC):
    """Abstract base class for all attacks in Gray Swan Arena."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        output_dir: str = "./reports/attacks",
        **kwargs
    ):
        """Initialize the attack.

        Args:
            model_name: Name of the model to use
            output_dir: Directory to save results to
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.AGENTOPS_AVAILABLE = hasattr(agentops, "record")
        
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")

    @abc.abstractmethod
    def gather_intelligence(self) -> Dict[str, Any]:
        """Gather intelligence about the target systems.
        
        This method should be implemented by each attack class to gather
        relevant information about the target systems, such as:
        - Target model capabilities and limitations
        - System architecture and components
        - Known vulnerabilities or weaknesses
        - Previous attack attempts and their outcomes
        
        Returns:
            Dictionary containing gathered intelligence
        """
        pass

    @abc.abstractmethod
    def craft_prompts(self, intelligence: Dict[str, Any]) -> List[str]:
        """Craft prompts for the attack.
        
        This method should be implemented by each attack class to generate
        prompts that will be used to test the target system. The prompts
        should be crafted based on the gathered intelligence and should
        be designed to achieve the attack's objectives.
        
        Args:
            intelligence: Dictionary containing gathered intelligence
            
        Returns:
            List of crafted prompts
        """
        pass

    @abc.abstractmethod
    def execute_attack(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Execute the attack using the crafted prompts.
        
        This method should be implemented by each attack class to execute
        the attack using the crafted prompts. It should handle:
        - Running the prompts against the target system
        - Collecting and recording responses
        - Handling errors and retries
        - Saving results
        
        Args:
            prompts: List of crafted prompts
            
        Returns:
            List of attack results
        """
        pass

    @abc.abstractmethod
    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the results of the attack.
        
        This method should be implemented by each attack class to evaluate
        the results of the attack execution. It should:
        - Analyze the responses and outcomes
        - Generate metrics and statistics
        - Create visualizations
        - Save evaluation results
        
        Args:
            results: List of attack results
            
        Returns:
            Dictionary containing evaluation results
        """
        pass

    def run(self) -> Dict[str, Any]:
        """Run the complete attack workflow.
        
        This method orchestrates the entire attack process by:
        1. Gathering intelligence
        2. Crafting prompts
        3. Executing the attack
        4. Evaluating results
        
        Returns:
            Dictionary containing the complete attack results
        """
        logger.info(f"Starting {self.__class__.__name__} attack workflow")
        
        try:
            # Log attack start
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "attack_started",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                    }
                ))

            # Step 1: Gather intelligence
            intelligence = self.gather_intelligence()
            
            # Step 2: Craft prompts
            prompts = self.craft_prompts(intelligence)
            
            # Step 3: Execute attack
            results = self.execute_attack(prompts)
            
            # Step 4: Evaluate results
            evaluation = self.evaluate_results(results)
            
            # Log attack completion
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "attack_completed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "num_prompts": len(prompts),
                        "num_results": len(results),
                    }
                ))

            return {
                "intelligence": intelligence,
                "prompts": prompts,
                "results": results,
                "evaluation": evaluation
            }

        except Exception as e:
            logger.error(f"Error in attack workflow: {e}", exc_info=True)
            
            # Log attack failure
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "attack_failed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "error": str(e),
                    }
                ))
            
            raise 