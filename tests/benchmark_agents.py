from typing import Any, Dict, List, Optional, Tuple, Union
#!/usr/bin/env python
"""
Benchmark script for Gray Swan Arena agents.

This script tests the functionality of our Gray Swan Arena agents
by running a simple reconnaissance task against a target model.
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Add the parent directory to sys.path so we can import the cybersec_agents module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cybersec_agents.grayswan.agents.evaluation_agent import EvaluationAgent
from cybersec_agents.grayswan.agents.exploit_delivery_agent import ExploitDeliveryAgent
from cybersec_agents.grayswan.agents.prompt_engineer_agent import PromptEngineerAgent
from cybersec_agents.grayswan.agents.recon_agent import ReconAgent
from cybersec_agents.grayswan.utils import init_agentops

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark")


def setup_environment():
    """Load environment variables and set up directories."""
    load_dotenv()

    # Create output directories
    os.makedirs("./test_outputs", exist_ok=True)
    os.makedirs("./test_outputs/reports", exist_ok=True)
    os.makedirs("./test_outputs/prompts", exist_ok=True)
    os.makedirs("./test_outputs/exploits", exist_ok=True)
    os.makedirs("./test_outputs/evaluations", exist_ok=True)

    # Initialize AgentOps
    init_agentops(session_name="benchmark_test")


def test_recon_agent():
    """Test the ReconAgent functionality."""
    logger.info("Testing ReconAgent...")

    # Initialize ReconAgent
    agent = ReconAgent(output_dir="./test_outputs/reports")

    # Run web search
    target_model: str = "GPT3.5"
    target_behavior: str = "bypassing content filters"

    web_results = agent.run_web_search(
        target_model=target_model, target_behavior=target_behavior
    )

    # Generate and save report
    report = agent.generate_report(
        target_model=target_model,
        target_behavior=target_behavior,
        web_results=web_results,
    )

    report_path = agent.save_report(
        report=report, target_model=target_model, target_behavior=target_behavior
    )

    logger.info(f"Recon report saved to: {report_path}")
    return report


def test_prompt_engineer_agent(recon_report):
    """Test the PromptEngineerAgent functionality."""
    logger.info("Testing PromptEngineerAgent...")

    # Initialize PromptEngineerAgent
    agent = PromptEngineerAgent(output_dir="./test_outputs/prompts")

    # Generate prompts
    target_model: str = "GPT3.5"
    target_behavior: str = "bypassing content filters"
    num_prompts: int = 3

    prompts = agent.generate_prompts(
        target_model=target_model,
        target_behavior=target_behavior,
        recon_report=recon_report,
        num_prompts=num_prompts,
    )

    # Save prompts
    prompts_path = agent.save_prompts(
        prompts=prompts, target_model=target_model, target_behavior=target_behavior
    )

    logger.info(f"Prompts saved to: {prompts_path}")
    return prompts


def test_exploit_delivery_agent(prompts):
    """Test the ExploitDeliveryAgent functionality."""
    logger.info("Testing ExploitDeliveryAgent...")

    # Initialize ExploitDeliveryAgent
    agent = ExploitDeliveryAgent(output_dir="./test_outputs/exploits")

    # Run prompts
    target_model: str = "GPT3.5"
    target_behavior: str = "bypassing content filters"

    results: list[Any] = agent.run_prompts(
        prompts=prompts,
        target_model="gpt-3.5-turbo",  # Using the model name directly
        target_behavior=target_behavior,
        method="api",
    )

    # Save results
    results_path = agent.save_results(
        results=results, target_model=target_model, target_behavior=target_behavior
    )

    logger.info(f"Exploit results saved to: {results_path}")
    return results


def test_evaluation_agent(results):
    """Test the EvaluationAgent functionality."""
    logger.info("Testing EvaluationAgent...")

    # Initialize EvaluationAgent
    agent = EvaluationAgent(output_dir="./test_outputs/evaluations")

    # Evaluate results
    target_model: str = "GPT3.5"
    target_behavior: str = "bypassing content filters"

    evaluation = agent.evaluate_results(
        results=results, target_model=target_model, target_behavior=target_behavior
    )

    # Save evaluation
    evaluation_path = agent.save_evaluation(
        evaluation=evaluation,
        target_model=target_model,
        target_behavior=target_behavior,
    )

    # Generate summary
    summary = agent.generate_summary(
        evaluation=evaluation,
        target_model=target_model,
        target_behavior=target_behavior,
    )

    # Save summary
    summary_path = agent.save_summary(
        summary=summary, target_model=target_model, target_behavior=target_behavior
    )

    # Create visualizations
    vis_paths = agent.create_visualizations(
        evaluation=evaluation,
        target_model=target_model,
        target_behavior=target_behavior,
    )

    logger.info(f"Evaluation saved to: {evaluation_path}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Visualizations saved to: {', '.join(vis_paths.values())}")


def run_benchmark():
    """Run the full benchmark test."""
    logger.info("Starting Gray Swan Arena benchmark test...")

    # Set up environment
    setup_environment()

    # Test agents in sequence
    try:
        recon_report = test_recon_agent()
        prompts = test_prompt_engineer_agent(recon_report)
        results: list[Any] = test_exploit_delivery_agent(prompts)
        test_evaluation_agent(results)

        logger.info("Benchmark completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
