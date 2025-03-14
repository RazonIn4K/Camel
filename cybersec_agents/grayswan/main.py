"""Main orchestration script for Gray Swan Arena."""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Optional

from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv

from .agents.evaluation_agent import EvaluationAgent
from .agents.exploit_delivery_agent import ExploitDeliveryAgent
from .agents.prompt_engineer_agent import PromptEngineerAgent
from .agents.recon_agent import ReconAgent
from .utils.logging_utils import setup_logging

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logging("GraySwanMain")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gray Swan Arena - AI Red-Teaming Framework"
    )

    # Target options
    parser.add_argument(
        "--target",
        type=str,
        default="GPT3.5",
        choices=["GPT3.5", "GPT4", "Claude", "Llama2", "Brass Fox"],
        help="Target AI model to test (default: GPT3.5)",
    )

    # Execution mode
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "recon", "prompts", "exploit", "evaluate"],
        help="Execution mode (default: full)",
    )

    # Input file options (for partial executions)
    parser.add_argument(
        "--recon-report",
        type=str,
        help="Path to existing recon report (for prompts/exploit/evaluate modes)",
    )
    parser.add_argument(
        "--prompt-list",
        type=str,
        help="Path to existing prompt list (for exploit/evaluate modes)",
    )
    parser.add_argument(
        "--exploit-results",
        type=str,
        help="Path to existing exploit results (for evaluate mode)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory for output files (default: data)",
    )

    # Generation options
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to generate (default: 10)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent exploit attempts (default: 3)",
    )

    # Visualization options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations in the evaluation report",
    )

    # Logging options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # AgentOps options
    parser.add_argument(
        "--disable-monitoring", action="store_true", help="Disable AgentOps monitoring"
    )

    return parser.parse_args()


def map_target_to_model(target: str) -> tuple:
    """Map target name to model type and platform."""
    target_map = {
        "GPT3.5": (ModelType.GPT_3_5_TURBO, ModelPlatformType.OPENAI),
        "GPT4": (ModelType.GPT_4, ModelPlatformType.OPENAI),
        "Claude": (ModelType.CLAUDE_3_SONNET, ModelPlatformType.ANTHROPIC),
        "Llama2": (ModelType.LLAMA_2_70B, ModelPlatformType.META),
        # For demo purposes, "Brass Fox" uses GPT-3.5
        "Brass Fox": (ModelType.GPT_3_5_TURBO, ModelPlatformType.OPENAI),
    }

    return target_map.get(target, (ModelType.GPT_3_5_TURBO, ModelPlatformType.OPENAI))


def run_reconnaissance(args) -> Optional[str]:
    """Run reconnaissance phase.

    Args:
        args: Command line arguments

    Returns:
        Path to the recon report, or None if failed
    """
    logger.info("Starting reconnaissance phase")

    try:
        # Create recon agent
        recon_agent = ReconAgent(ModelType.GPT_4, ModelPlatformType.OPENAI)

        # Gather information about target model
        logger.info(f"Gathering information on target: {args.target}")
        model_info = recon_agent.run_web_search(
            f"{args.target} AI model architecture capabilities"
        )
        behavior_info = recon_agent.run_web_search(
            "AI safeguards bypassing vulnerabilities"
        )
        techniques_info = recon_agent.run_web_search(
            "jailbreaking techniques for LLMs prompt injection"
        )

        # Try Discord search if available
        discord_info = None
        try:
            discord_info = recon_agent.run_discord_search(
                f"{args.target} vulnerabilities red-teaming"
            )
        except Exception as e:
            logger.warning(f"Discord search failed: {str(e)}")

        # Generate and save report
        report = recon_agent.generate_report(
            model_info, behavior_info, techniques_info, discord_info
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"recon_report_{args.target.lower().replace(' ', '_')}_{timestamp}.json"
        )

        report_path = recon_agent.save_report(report, filename)
        logger.info(f"Reconnaissance report saved to: {report_path}")

        return report_path

    except KeyboardInterrupt:
        logger.warning("Reconnaissance interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Reconnaissance failed: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return None


def run_prompt_engineering(args, recon_report_path: str) -> Optional[str]:
    """Run prompt engineering phase.

    Args:
        args: Command line arguments
        recon_report_path: Path to the reconnaissance report

    Returns:
        Path to the prompt list, or None if failed
    """
    logger.info("Starting prompt engineering phase")

    try:
        # Create prompt engineer agent
        prompt_agent = PromptEngineerAgent(ModelType.GPT_4, ModelPlatformType.OPENAI)

        # Load recon report
        recon_report = prompt_agent.load_recon_report(recon_report_path)
        if not recon_report:
            logger.error(f"Failed to load recon report from {recon_report_path}")
            return None

        # Generate prompts
        logger.info(f"Generating {args.num_prompts} attack prompts")
        prompts = prompt_agent.generate_prompts(
            recon_report, num_prompts=args.num_prompts
        )

        # Evaluate prompt diversity
        diversity = prompt_agent.evaluate_prompt_diversity(prompts)
        logger.info(
            f"Generated {diversity['total_prompts']} prompts using {len(diversity['technique_distribution'])} techniques"
        )

        # Save prompts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"prompt_list_{args.target.lower().replace(' ', '_')}_{timestamp}.json"
        )

        prompt_path = prompt_agent.save_prompts(prompts, filename)
        logger.info(f"Prompt list saved to: {prompt_path}")

        return prompt_path

    except KeyboardInterrupt:
        logger.warning("Prompt engineering interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Prompt engineering failed: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return None


def run_exploit_delivery(args, prompt_list_path: str) -> Optional[str]:
    """Run exploit delivery phase.

    Args:
        args: Command line arguments
        prompt_list_path: Path to the prompt list

    Returns:
        Path to the exploit results, or None if failed
    """
    logger.info("Starting exploit delivery phase")

    try:
        # Map target to model type and platform
        model_type, model_platform = map_target_to_model(args.target)

        # Create exploit delivery agent
        exploit_agent = ExploitDeliveryAgent(model_type, model_platform)

        # Load prompts
        prompts = exploit_agent.load_prompts(prompt_list_path)
        if not prompts:
            logger.error(f"Failed to load prompts from {prompt_list_path}")
            return None

        # Execute prompts
        logger.info(f"Executing {len(prompts)} prompts against {args.target}")
        results = exploit_agent.execute_prompt_batch(
            prompts, max_concurrent=args.max_concurrent
        )

        # Analyze results
        analysis = exploit_agent.analyze_results(results)
        success_rate = analysis["success_rate"] * 100
        logger.info(f"Execution complete. Success rate: {success_rate:.2f}%")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"exploit_results_{args.target.lower().replace(' ', '_')}_{timestamp}.json"
        )

        results_path = exploit_agent.save_results(results, filename)
        logger.info(f"Exploit results saved to: {results_path}")

        return results_path

    except KeyboardInterrupt:
        logger.warning("Exploit delivery interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Exploit delivery failed: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return None


def run_evaluation(
    args, exploit_results_path: str, recon_report_path: Optional[str] = None
) -> Optional[str]:
    """Run evaluation phase.

    Args:
        args: Command line arguments
        exploit_results_path: Path to the exploit results
        recon_report_path: Path to the reconnaissance report (optional)

    Returns:
        Path to the evaluation report, or None if failed
    """
    logger.info("Starting evaluation phase")

    try:
        # Create evaluation agent
        eval_agent = EvaluationAgent(ModelType.GPT_4, ModelPlatformType.OPENAI)

        # Load exploit results
        results = eval_agent.load_exploit_results(exploit_results_path)
        if not results:
            logger.error(f"Failed to load exploit results from {exploit_results_path}")
            return None

        # Load recon report if available
        recon_report = None
        if recon_report_path:
            recon_report = eval_agent.load_recon_report(recon_report_path)

        # Calculate statistics
        statistics = eval_agent.calculate_statistics(results)

        # Create visualizations if requested
        visualization_paths = None
        if args.visualize:
            vis_dir = os.path.join(
                args.output_dir, "evaluation_reports", "visualizations"
            )
            visualization_paths = eval_agent.create_visualizations(statistics, vis_dir)

        # Generate report
        report = eval_agent.generate_report(
            results=results,
            statistics=statistics,
            recon_report=recon_report,
            visualization_paths=visualization_paths,
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_report_{args.target.lower().replace(' ', '_')}_{timestamp}.json"

        report_path = eval_agent.save_report(report, filename)
        logger.info(f"Evaluation report saved to: {report_path}")

        # Generate markdown report
        markdown_filename = (
            f"evaluation_report_{args.target.lower().replace(' ', '_')}_{timestamp}.md"
        )
        markdown_path = os.path.join(
            args.output_dir, "evaluation_reports", markdown_filename
        )

        eval_agent.generate_markdown_report(report, markdown_path)
        logger.info(f"Markdown report saved to: {markdown_path}")

        return report_path

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return None


def run_full_pipeline(args):
    """Run the full red-teaming pipeline."""
    logger.info(f"Starting full red-teaming pipeline against {args.target}")

    # Run reconnaissance
    recon_report_path = run_reconnaissance(args)
    if not recon_report_path:
        logger.error("Reconnaissance failed, cannot continue pipeline")
        return

    # Run prompt engineering
    prompt_list_path = run_prompt_engineering(args, recon_report_path)
    if not prompt_list_path:
        logger.error("Prompt engineering failed, cannot continue pipeline")
        return

    # Run exploit delivery
    exploit_results_path = run_exploit_delivery(args, prompt_list_path)
    if not exploit_results_path:
        logger.error("Exploit delivery failed, cannot continue pipeline")
        return

    # Run evaluation
    eval_report_path = run_evaluation(args, exploit_results_path, recon_report_path)
    if not eval_report_path:
        logger.error("Evaluation failed")
        return

    logger.info("Gray Swan Arena red-teaming pipeline completed successfully")
    return eval_report_path


def main(args=None):
    """Main entry point."""
    # Parse arguments if not provided
    if args is None:
        args = parse_arguments()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Disable AgentOps if requested
    if args.disable_monitoring:
        os.environ["AGENTOPS_API_KEY"] = ""

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Run in the specified mode
    try:
        if args.mode == "full":
            run_full_pipeline(args)

        elif args.mode == "recon":
            recon_report_path = run_reconnaissance(args)
            if recon_report_path:
                logger.info(
                    f"Reconnaissance completed successfully: {recon_report_path}"
                )
            else:
                logger.error("Reconnaissance failed")

        elif args.mode == "prompts":
            if not args.recon_report:
                logger.error("A recon report must be provided for 'prompts' mode")
                return

            prompt_list_path = run_prompt_engineering(args, args.recon_report)
            if prompt_list_path:
                logger.info(
                    f"Prompt engineering completed successfully: {prompt_list_path}"
                )
            else:
                logger.error("Prompt engineering failed")

        elif args.mode == "exploit":
            if not args.prompt_list:
                logger.error("A prompt list must be provided for 'exploit' mode")
                return

            exploit_results_path = run_exploit_delivery(args, args.prompt_list)
            if exploit_results_path:
                logger.info(
                    f"Exploit delivery completed successfully: {exploit_results_path}"
                )
            else:
                logger.error("Exploit delivery failed")

        elif args.mode == "evaluate":
            if not args.exploit_results:
                logger.error("Exploit results must be provided for 'evaluate' mode")
                return

            eval_report_path = run_evaluation(
                args, args.exploit_results, args.recon_report
            )
            if eval_report_path:
                logger.info(f"Evaluation completed successfully: {eval_report_path}")
            else:
                logger.error("Evaluation failed")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
