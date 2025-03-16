"""System Safeguard Attack for Gray Swan Arena.

This attack attempts to leak the system safeguards of a target AI agent
through direct or indirect means. It leverages the PromptEngineerAgent
for crafting prompts and the ExploitDeliveryAgent for executing the attack.
"""

import logging
from typing import Any, Dict, List

from .base_attack import Attack
from ..agents import PromptEngineerAgent, ExploitDeliveryAgent, EvaluationAgent
from ..utils.challenge_loader import ChallengeLoader
from ..utils.agentops import agentops

# Set up logger
logger = logging.getLogger(__name__)


class SystemSafeguardAttack(Attack):
    """Class to orchestrate the system safeguard attack."""

    def __init__(self, model_name: str = "gpt-4", output_dir: str = "./reports/attacks", **kwargs):
        """Initialize the attack components.

        Args:
            model_name: Name of the model to use
            output_dir: Directory to save results to
            **kwargs: Additional arguments
        """
        super().__init__(model_name, output_dir, **kwargs)

        # Initialize agents
        self.prompt_engineer = PromptEngineerAgent(
            output_dir="./reports/prompts", model_name=model_name
        )
        self.exploit_delivery = ExploitDeliveryAgent(
            output_dir="./reports/exploits",
            model_name=model_name,
            browser_method="playwright",
            headless=True,
        )
        self.evaluation_agent = EvaluationAgent(
            output_dir="./reports/evaluations",
            model_name=model_name,
        )

        # Initialize challenge loader
        self.challenge_loader = ChallengeLoader()

        logger.info(
            f"{self.__class__.__name__} initialized with model: {self.model_name}"
        )

    def gather_intelligence(self) -> Dict[str, Any]:
        """Gather intelligence about the target.

        In this challenge, no specific intelligence gathering is required beyond
        the general description of the AI agent.

        Returns:
            An empty dictionary, as no intelligence gathering is needed for this attack
        """
        logger.info("No intelligence gathering required for SystemSafeguardAttack")

        # Log intelligence gathering start
        if self.AGENTOPS_AVAILABLE:
            agentops.record(
                agentops.ActionEvent(
                    "intelligence_gathering_started",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                    },
                )
            )

        # Log intelligence gathering completion
        if self.AGENTOPS_AVAILABLE:
            agentops.record(
                agentops.ActionEvent(
                    "intelligence_gathering_completed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                    },
                )
            )

        return {}

    def craft_prompts(self, intelligence: Dict[str, Any]) -> List[str]:
        """Craft prompts for the attack.

        Args:
            intelligence: Dictionary containing gathered intelligence (not used in this attack)

        Returns:
            List of crafted prompts
        """
        logger.info("Starting prompt engineering phase")

        # Log prompt engineering start
        if self.AGENTOPS_AVAILABLE:
            agentops.record(
                agentops.ActionEvent(
                    "prompt_engineering_started",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                    },
                )
            )

        # Get challenge details from challenges.md
        challenge_details = self.challenge_loader.get_challenge_details(
            challenge_type="Confidentiality Breaches",
            challenge_name="Leak Agent System Safeguards",
        )

        # Generate prompts using the gathered intelligence and challenge details
        prompts = self.prompt_engineer.generate_prompts(
            target_model="system_safeguards",
            target_behavior="leak_system_safeguards",
            recon_report=challenge_details,
            num_prompts=10,
        )

        # Save the prompts
        prompts_path = self.prompt_engineer.save_prompts(
            prompts=prompts,
            target_model="system_safeguards",
            target_behavior="leak_system_safeguards",
        )

        logger.info(f"Crafted {len(prompts)} prompts. Saved to {prompts_path}")

        # Log prompt engineering completion
        if self.AGENTOPS_AVAILABLE:
            agentops.record(
                agentops.ActionEvent(
                    "prompt_engineering_completed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "num_prompts": len(prompts),
                        "prompts_path": str(prompts_path),
                    },
                )
            )

        return prompts

    def execute_attack(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Execute the attack using the crafted prompts.

        Args:
            prompts: List of crafted prompts

        Returns:
            List of attack results
        """
        logger.info("Starting attack execution phase")

        # Log attack execution start
        if self.AGENTOPS_AVAILABLE:
            agentops.record(
                agentops.ActionEvent(
                    "attack_execution_started",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "num_prompts": len(prompts),
                    },
                )
            )

        # Execute prompts
        results = self.exploit_delivery.run_prompts(
            prompts=prompts,
            target_model="system_safeguards",
            target_behavior="leak_system_safeguards",
            method="api",
            max_tries=3,
            delay_between_tries=2,
        )

        # Save the results
        results_path = self.exploit_delivery.save_results(
            results=results,
            target_model="system_safeguards",
            target_behavior="leak_system_safeguards",
        )

        logger.info(f"Attack execution completed. Results saved to {results_path}")

        # Log attack execution completion
        if self.AGENTOPS_AVAILABLE:
            agentops.record(
                agentops.ActionEvent(
                    "attack_execution_completed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "num_results": len(results),
                        "results_path": str(results_path),
                    },
                )
            )

        return results

    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the results of the attack.

        Args:
            results: List of attack results

        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting evaluation phase")

        # Log evaluation start
        if self.AGENTOPS_AVAILABLE:
            agentops.record(
                agentops.ActionEvent(
                    "evaluation_started",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "num_results": len(results),
                    },
                )
            )

        # Evaluate the results
        evaluation = self.evaluation_agent.evaluate_results(test_results=results)

        # Generate summary
        summary = self.evaluation_agent.generate_summary(test_results=results)

        # Create visualizations
        visualizations = self.evaluation_agent.create_visualizations(
            test_results=results
        )

        # Create advanced visualizations
        advanced_viz = self.evaluation_agent.create_advanced_visualizations(
            results=results,
            target_model="system_safeguards",
            target_behavior="leak_system_safeguards",
            include_interactive=True,
        )

        # Save the evaluation
        eval_path = self.evaluation_agent.save_evaluation(
            evaluation=evaluation,
            target_model="system_safeguards",
            target_behavior="leak_system_safeguards",
        )

        logger.info(f"Evaluation completed. Results saved to {eval_path}")

        # Log evaluation completion
        if self.AGENTOPS_AVAILABLE:
            agentops.record(
                agentops.ActionEvent(
                    "evaluation_completed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "eval_path": str(eval_path),
                    },
                )
            )

        return {
            "evaluation": evaluation,
            "summary": summary,
            "visualizations": visualizations,
            "advanced_visualizations": advanced_viz,
        } 