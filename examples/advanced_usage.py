from typing import Any, Dict, List, Optional, Tuple, Union

#!/usr/bin/env python
"""Example demonstrating advanced usage of the Gray Swan Arena agents with error handling."""

import logging
import sys

from cybersec_agents.grayswan.agents.evaluation_agent import EvaluationAgent
from cybersec_agents.grayswan.agents.exploit_delivery_agent import ExploitDeliveryAgent
from cybersec_agents.grayswan.agents.prompt_engineer_agent import PromptEngineerAgent
from cybersec_agents.grayswan.agents.recon_agent import ReconAgent
from cybersec_agents.grayswan.config import Config
from cybersec_agents.grayswan.exceptions import (
    APIError,
    ConfigurationError,
    ModelBackupError,
    ModelError,
    RecoveryError,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("advanced_usage")


def setup_environment(env_name="development"):
    """Set up the environment with configuration."""
    try:
        # Load configuration from the specified environment
        logger.info(f"Loading configuration for environment: {env_name}")
        config: dict[str, Any] = Config(env=env_name)
        logger.info(f"Configuration loaded successfully: {env_name}")
        return config
    except ConfigurationError as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        logger.error(f"Details: {e.details}")
        sys.exit(1)


def initialize_agents(config):
    """Initialize agents with configuration."""
    try:
        logger.info("Initializing agents...")

        # Get agent configurations
        recon_config = config.get_agent_config("recon")
        prompt_config = config.get_agent_config("prompt_engineer")
        exploit_config = config.get_agent_config("exploit_delivery")
        eval_config = config.get_agent_config("evaluation")

        # Initialize agents with proper configuration
        recon_agent = ReconAgent(
            output_dir=recon_config.output_dir,
            model_name=recon_config.model_name,
            backup_model=recon_config.backup_model,
            reasoning_model=recon_config.reasoning_model,
        )

        prompt_agent = PromptEngineerAgent(
            output_dir=prompt_config.output_dir,
            model_name=prompt_config.model_name,
            reasoning_model=prompt_config.reasoning_model,
        )

        exploit_agent = ExploitDeliveryAgent(
            output_dir=exploit_config.output_dir,
            model_name=exploit_config.model_name,
            backup_model=exploit_config.backup_model,
        )

        eval_agent = EvaluationAgent(
            output_dir=eval_config.output_dir,
            model_name=eval_config.model_name,
            backup_model=eval_config.backup_model,
            reasoning_model=eval_config.reasoning_model,
        )

        logger.info("Agents initialized successfully")
        return recon_agent, prompt_agent, exploit_agent, eval_agent

    except ConfigurationError as e:
        logger.error(f"Configuration error during agent initialization: {str(e)}")
        logger.error(f"Details: {e.details}")
        sys.exit(1)


def run_pipeline(recon_agent, prompt_agent, exploit_agent, eval_agent):
    """Run the Gray Swan Arena pipeline with error handling."""
    target_model: str = "claude-3"
    target_behavior: str = "jailbreak"

    try:
        # Step 1: Reconnaissance
        logger.info(
            f"Starting reconnaissance for {target_model} targeting {target_behavior}"
        )
        try:
            web_results = recon_agent.run_web_search(target_model, target_behavior)
            logger.info(f"Web search completed with {len(web_results)} results")
        except APIError as e:
            logger.warning(f"API error during web search: {str(e)}")
            logger.warning("Continuing with empty web results")
            web_results: dict[str, Any] = {}

        try:
            report = recon_agent.generate_report(
                target_model=target_model,
                target_behavior=target_behavior,
                web_results=web_results,
            )
            report_path = recon_agent.save_report(
                report=report,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            logger.info(f"Reconnaissance report saved to: {report_path}")
        except ModelError as e:
            logger.error(f"Model error during report generation: {str(e)}")
            logger.error(f"Model: {e.model_name}, Operation: {e.operation}")
            return

        # Step 2: Prompt Engineering
        logger.info("Starting prompt engineering phase")
        try:
            prompts = prompt_agent.generate_prompts(
                target_model=target_model,
                target_behavior=target_behavior,
                recon_report=report,
                num_prompts=5,
            )
            prompts_path = prompt_agent.save_prompts(
                prompts=prompts,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            logger.info(f"Generated {len(prompts)} prompts, saved to: {prompts_path}")
        except ModelBackupError as e:
            logger.error(f"Both primary and backup models failed: {str(e)}")
            logger.error(f"Primary: {e.model_name}, Backup: {e.backup_model}")
            return

        # Step 3: Exploit Delivery
        logger.info("Starting exploit delivery phase")
        try:
            results: list[Any] = exploit_agent.run_prompts(
                prompts=prompts,
                target_model=target_model,
                target_behavior=target_behavior,
                method="api",
                max_tries=3,
            )
            results_path = exploit_agent.save_results(
                results=results,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            logger.info(f"Exploit results saved to: {results_path}")
        except APIError as e:
            if e.status_code == 429:
                logger.error(f"Rate limit exceeded: {str(e)}")
                logger.info("Implementing exponential backoff strategy")
                # Implement recovery strategy here
            else:
                logger.error(f"API error during exploit delivery: {str(e)}")
            return

        # Step 4: Evaluation
        logger.info("Starting evaluation phase")
        try:
            evaluation = eval_agent.evaluate_results(
                results=results,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            eval_path = eval_agent.save_evaluation(
                evaluation=evaluation,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            logger.info(f"Evaluation saved to: {eval_path}")

            summary = eval_agent.generate_summary(
                evaluation=evaluation,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            summary_path = eval_agent.save_summary(
                summary=summary,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            logger.info(f"Summary saved to: {summary_path}")

            viz_paths = eval_agent.create_visualizations(
                evaluation=evaluation,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            logger.info(f"Visualizations saved to: {viz_paths}")
        except Exception as e:
            logger.error(f"Error during evaluation phase: {str(e)}")
            return

        logger.info("Gray Swan Arena pipeline completed successfully!")

    except RecoveryError as e:
        logger.critical(f"Recovery failed: {str(e)}")
        logger.critical(f"Original error: {e.original_error}")
        logger.critical(f"Recovery strategy: {e.recovery_strategy}")
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")


def main():
    """Main function to run the example."""
    # Get environment from command line argument or use default
    env_name = sys.argv[1] if len(sys.argv) > 1 else "development"

    # Set up environment
    config: dict[str, Any] = setup_environment(env_name)

    # Initialize agents
    recon_agent, prompt_agent, exploit_agent, eval_agent = initialize_agents(config)

    # Run pipeline
    run_pipeline(recon_agent, prompt_agent, exploit_agent, eval_agent)


if __name__ == "__main__":
    main()
