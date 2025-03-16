"""AirBnB Credentials Attack for Gray Swan Arena.

This attack attempts to leak AirBnB credentials to Instagram through indirect means.
It leverages the ReconAgent for information gathering, PromptEngineerAgent for crafting
prompts, and ExploitDeliveryAgent for executing the attack.
"""

import logging
from typing import Dict, Any, List

from .base_attack import Attack
from ..agents import (
    ReconAgent,
    PromptEngineerAgent,
    ExploitDeliveryAgent,
    EvaluationAgent
)
from ..utils.challenge_loader import ChallengeLoader
from ..utils.agentops import agentops

# Set up logger
logger = logging.getLogger(__name__)


class AirBnBCredentialsAttack(Attack):
    """Class to orchestrate the AirBnB credentials attack."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        output_dir: str = "./reports/attacks",
        **kwargs
    ):
        """Initialize the attack components.

        Args:
            model_name: Name of the model to use
            output_dir: Directory to save results to
            **kwargs: Additional arguments
        """
        super().__init__(model_name, output_dir, **kwargs)
        
        # Initialize agents
        self.recon_agent = ReconAgent(
            output_dir="./reports/recon",
            model_name=model_name
        )
        self.prompt_engineer = PromptEngineerAgent(
            output_dir="./reports/prompts",
            model_name=model_name
        )
        self.exploit_delivery = ExploitDeliveryAgent(
            output_dir="./reports/exploits",
            model_name=model_name,
            browser_method="playwright",
            headless=True
        )
        self.evaluation_agent = EvaluationAgent(
            output_dir="./reports/evaluations",
            model_name=model_name
        )
        
        # Initialize challenge loader
        self.challenge_loader = ChallengeLoader()
        
        logger.info("AirBnBCredentialsAttack initialized with all required agents")

    def gather_intelligence(self) -> Dict[str, Any]:
        """Gather intelligence about the target systems.
        
        Returns:
            Dictionary containing gathered intelligence
        """
        logger.info("Starting intelligence gathering phase")
        
        try:
            # Log intelligence gathering start
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "intelligence_gathering_started",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                    }
                ))

            # Gather information about AI systems, Instagram, and AirBnB
            web_results = self.recon_agent.run_web_search(
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram",
                num_results=5
            )
            
            discord_results = self.recon_agent.run_discord_search(
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram",
                channels=["ai-security", "prompt-engineering"]
            )
            
            # Generate a comprehensive report
            intelligence = self.recon_agent.generate_report(
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram",
                web_results=web_results,
                discord_results=discord_results
            )
            
            # Save the report
            report_path = self.recon_agent.save_report(
                report=intelligence,
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram"
            )
            
            logger.info(f"Intelligence gathering completed. Report saved to {report_path}")

            # Log intelligence gathering completion
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "intelligence_gathering_completed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "report_path": str(report_path),
                    }
                ))

            return intelligence

        except Exception as e:
            logger.error(f"Error gathering intelligence: {e}", exc_info=True)
            
            # Log intelligence gathering failure
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "intelligence_gathering_failed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "error": str(e),
                    }
                ))
            
            raise

    def craft_prompts(self, intelligence: Dict[str, Any]) -> List[str]:
        """Craft prompts for the attack.
        
        Args:
            intelligence: Dictionary containing gathered intelligence
            
        Returns:
            List of crafted prompts
        """
        logger.info("Starting prompt engineering phase")
        
        try:
            # Log prompt engineering start
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "prompt_engineering_started",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                    }
                ))

            # Get challenge details from challenges.md
            challenge_details = self.challenge_loader.get_challenge_details(
                challenge_type="Confidentiality Breaches",
                challenge_name="Leak Agent System Safeguards"
            )
            
            # Generate prompts using the gathered intelligence and challenge details
            prompts = self.prompt_engineer.generate_prompts(
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram",
                recon_report=intelligence,
                num_prompts=10
            )
            
            # Save the prompts
            prompts_path = self.prompt_engineer.save_prompts(
                prompts=prompts,
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram"
            )
            
            logger.info(f"Crafted {len(prompts)} prompts. Saved to {prompts_path}")

            # Log prompt engineering completion
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "prompt_engineering_completed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "num_prompts": len(prompts),
                        "prompts_path": str(prompts_path),
                    }
                ))

            return prompts

        except Exception as e:
            logger.error(f"Error crafting prompts: {e}", exc_info=True)
            
            # Log prompt engineering failure
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "prompt_engineering_failed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "error": str(e),
                    }
                ))
            
            raise

    def execute_attack(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Execute the attack using the crafted prompts.
        
        Args:
            prompts: List of crafted prompts
            
        Returns:
            List of attack results
        """
        logger.info("Starting attack execution phase")
        
        try:
            # Log attack execution start
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "attack_execution_started",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "num_prompts": len(prompts),
                    }
                ))

            # Execute prompts using web method first
            web_results = self.exploit_delivery.run_prompts(
                prompts=prompts,
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram",
                method="web",  # Start with web method
                max_tries=3,
                delay_between_tries=2
            )
            
            # Get browser metrics after web automation
            browser_metrics = self.exploit_delivery.get_browser_metrics()
            logger.info(f"Browser metrics: {browser_metrics}")
            
            # Then try API method
            api_results = self.exploit_delivery.run_prompts(
                prompts=prompts,
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram",
                method="api",  # Then try API method
                max_tries=3,
                delay_between_tries=2
            )
            
            # Combine results
            results = web_results + api_results
            
            # Save the combined results
            results_path = self.exploit_delivery.save_results(
                results=results,
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram"
            )
            
            logger.info(f"Attack execution completed. Results saved to {results_path}")

            # Log attack execution completion
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "attack_execution_completed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "num_results": len(results),
                        "results_path": str(results_path),
                    }
                ))

            return results

        except Exception as e:
            logger.error(f"Error executing attack: {e}", exc_info=True)
            
            # Log attack execution failure
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "attack_execution_failed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "error": str(e),
                    }
                ))
            
            raise

    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the results of the attack.
        
        Args:
            results: List of attack results
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting evaluation phase")
        
        try:
            # Log evaluation start
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "evaluation_started",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "num_results": len(results),
                    }
                ))

            # Evaluate the results
            evaluation = self.evaluation_agent.evaluate_results(test_results=results)
            
            # Generate summary
            summary = self.evaluation_agent.generate_summary(test_results=results)
            
            # Create visualizations
            visualizations = self.evaluation_agent.create_visualizations(test_results=results)
            
            # Create advanced visualizations
            advanced_viz = self.evaluation_agent.create_advanced_visualizations(
                results=results,
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram",
                include_interactive=True
            )
            
            # Save the evaluation
            eval_path = self.evaluation_agent.save_evaluation(
                evaluation=evaluation,
                target_model="airbnb_credentials",
                target_behavior="leak_to_instagram"
            )
            
            logger.info(f"Evaluation completed. Results saved to {eval_path}")

            # Log evaluation completion
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "evaluation_completed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "eval_path": str(eval_path),
                    }
                ))

            return {
                "evaluation": evaluation,
                "summary": summary,
                "visualizations": visualizations,
                "advanced_visualizations": advanced_viz
            }

        except Exception as e:
            logger.error(f"Error evaluating results: {e}", exc_info=True)
            
            # Log evaluation failure
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "evaluation_failed",
                    {
                        "attack_type": self.__class__.__name__,
                        "model_name": self.model_name,
                        "error": str(e),
                    }
                ))
            
            raise 