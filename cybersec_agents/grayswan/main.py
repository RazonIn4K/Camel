"""
Main module for Gray Swan Arena.

This module provides the main functionality for running Gray Swan Arena agents
and executing the full pipeline for AI safety evaluation.
"""

import os
from datetime import datetime
from typing import Any, Dict

# Import agents
from cybersec_agents.grayswan.agents.recon_agent import ReconAgent
from cybersec_agents.grayswan.utils.agentops_utils import (
    initialize_agentops,
    log_agentops_event,
)

# Import utilities
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("grayswan_main")


def run_reconnaissance(
    target_model: str,
    target_behavior: str,
    output_dir: str = "./reports",
    model_name: str = "gpt-4",
) -> Dict[str, Any]:
    """
    Run the reconnaissance phase of the Gray Swan Arena pipeline.

    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save reports
        model_name: Name of the model to use for the agent

    Returns:
        Dictionary containing the reconnaissance report
    """
    logger.info(f"Starting reconnaissance phase for {target_model} - {target_behavior}")

    # Initialize AgentOps session
    api_key = os.getenv("AGENTOPS_API_KEY")
    if api_key:
        initialize_agentops(api_key)

    try:
        # Initialize the ReconAgent
        recon_agent = ReconAgent(output_dir=output_dir, model_name=model_name)

        # Run web search
        web_results = recon_agent.run_web_search(
            target_model=target_model, target_behavior=target_behavior
        )

        # Run Discord search
        discord_results = recon_agent.run_discord_search(
            target_model=target_model, target_behavior=target_behavior
        )

        # Generate and save report
        report = recon_agent.generate_report(
            target_model=target_model,
            target_behavior=target_behavior,
            web_results=web_results,
            discord_results=discord_results,
        )

        report_path = recon_agent.save_report(
            report=report, target_model=target_model, target_behavior=target_behavior
        )

        logger.info(f"Reconnaissance phase completed, report saved to {report_path}")

        # Log completion event
        log_agentops_event(
            "phase_completed",
            {
                "phase": "reconnaissance",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "completed",
                "report_path": report_path,
            },
        )

        return report

    except Exception as e:
        logger.error(f"Reconnaissance phase failed: {str(e)}")

        # Log error event
        log_agentops_event(
            "phase_error",
            {
                "phase": "reconnaissance",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "failed",
                "error": str(e),
            },
        )

        return {
            "error": str(e),
            "target_model": target_model,
            "target_behavior": target_behavior,
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """
    Main function for command-line execution.
    """
    # Set up arguments parser
    import argparse

    parser = argparse.ArgumentParser(
        description="Gray Swan Arena - AI Safety Testing Framework"
    )

    # Add arguments
    parser.add_argument(
        "--target-model", type=str, default="gpt-3.5-turbo", help="Target model to test"
    )
    parser.add_argument(
        "--target-behavior",
        type=str,
        default="bypass content policies",
        help="Behavior to target",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="Directory to save outputs"
    )
    parser.add_argument(
        "--agent-model", type=str, default="gpt-4", help="Model to use for agents"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run reconnaissance
    report = run_reconnaissance(
        target_model=args.target_model,
        target_behavior=args.target_behavior,
        output_dir=os.path.join(args.output_dir, "reports"),
        model_name=args.agent_model,
    )

    print(f"Reconnaissance complete. Report generated for {args.target_model}.")


if __name__ == "__main__":
    main()
