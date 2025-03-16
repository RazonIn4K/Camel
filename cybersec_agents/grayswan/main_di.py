"""
Main module for Gray Swan Arena with Dependency Injection.

This module provides the main functionality for running Gray Swan Arena agents
and executing the full pipeline for AI safety evaluation, using dependency injection
for better testability and flexibility.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
from camel.types import ModelType, ModelPlatformType

# Import container
from cybersec_agents.grayswan.container import (
    GraySwanContainer,
    GraySwanContainerFactory,
)

# Import utilities
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("grayswan_main_di")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Gray Swan Arena to evaluate AI safety and security."
    )

    # Target model configuration
    parser.add_argument(
        "--target-model-type",
        type=str,
        default="GPT_4",
        choices=[model_type.name for model_type in ModelType],
        help="Type of model to target for evaluation",
    )

    parser.add_argument(
        "--target-model-platform",
        type=str,
        default="OPENAI",
        choices=[platform.name for platform in ModelPlatformType],
        help="Platform of the target model",
    )

    parser.add_argument(
        "--target-behavior",
        type=str,
        default="bypass content policies",
        help="Target behavior to evaluate",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save output files",
    )

    # Pipeline configuration
    parser.add_argument(
        "--skip-phases",
        type=str,
        nargs="*",
        choices=["recon", "prompt", "exploit", "evaluate"],
        help="Phases to skip in the pipeline",
    )

    parser.add_argument(
        "--max-prompts",
        type=int,
        default=10,
        help="Maximum number of prompts to generate",
    )

    parser.add_argument(
        "--test-method",
        type=str,
        default="api",
        choices=["api", "web"],
        help="Method to use for testing (API or web interface)",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum number of concurrent tasks",
    )

    # Configuration file
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file (default: config/development.yml or GRAYSWAN_CONFIG env var)",
    )

    # Visualization options
    parser.add_argument(
        "--advanced-visualizations",
        action="store_true",
        help="Include advanced visualizations in the output",
    )

    parser.add_argument(
        "--interactive-dashboard",
        action="store_true",
        help="Generate an interactive dashboard",
    )

    return parser.parse_args()


class GraySwanPipeline:
    """
    Gray Swan Arena pipeline with dependency injection.

    This class provides methods for running the Gray Swan Arena pipeline
    using dependency injection for better testability and flexibility.
    """

    def __init__(self, container: GraySwanContainer):
        """
        Initialize the GraySwanPipeline.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = container.logger()

        self.logger.info("GraySwanPipeline initialized with dependency injection")

    async def run_parallel_reconnaissance(
        self,
        target_model_type: ModelType,
        target_model_platform: ModelPlatformType,
        target_behavior: str,
    ) -> Dict[str, Any]:
        """
        Run reconnaissance tasks in parallel.

        Args:
            target_model_type: The type of model to test
            target_model_platform: The platform of the model
            target_behavior: The behavior to target

        Returns:
            Dictionary containing reconnaissance results and report path
        """
        self.logger.info(
            "Starting parallel reconnaissance for %s on %s - %s",
            target_model_type,
            target_model_platform,
            target_behavior
        )

        # Get ReconAgent from container
        recon_agent = self.container.recon_agent()

        # Create tasks for concurrent execution
        web_task = asyncio.create_task(
            asyncio.to_thread(recon_agent.run_web_search, target_model_type, target_behavior)
        )

        discord_task = asyncio.create_task(
            asyncio.to_thread(
                recon_agent.run_discord_search, target_model_type, target_behavior
            )
        )

        # Wait for all tasks to complete
        web_results, discord_results = await asyncio.gather(web_task, discord_task)

        # Generate and save report
        report = recon_agent.generate_report(
            target_model_type=target_model_type,
            target_model_platform=target_model_platform,
            target_behavior=target_behavior,
            web_results=web_results,
            discord_results=discord_results,
        )

        report_path = recon_agent.save_report(
            report=report,
            target_model_type=target_model_type,
            target_model_platform=target_model_platform,
            target_behavior=target_behavior,
        )

        self.logger.info(
            "Parallel reconnaissance completed, report saved to %s",
            report_path
        )

        return {
            "report": report,
            "path": report_path,
            "web_results": web_results,
            "discord_results": discord_results,
        }

    def run_reconnaissance(
        self,
        target_model_type: ModelType,
        target_model_platform: ModelPlatformType,
        target_behavior: str,
    ) -> Dict[str, Any]:
        """
        Run the reconnaissance phase of the Gray Swan Arena pipeline.

        Args:
            target_model_type: The type of model to test
            target_model_platform: The platform of the model
            target_behavior: The behavior to target

        Returns:
            Dictionary containing the reconnaissance report
        """
        self.logger.info(
            f"Starting reconnaissance phase for {target_model_type} on {target_model_platform} - {target_behavior}"
        )

        try:
            # Get ReconAgent from container
            recon_agent = self.container.recon_agent()

            # Run web search
            web_results = recon_agent.run_web_search(
                target_model_type=target_model_type,
                target_behavior=target_behavior,
            )

            # Run Discord search
            discord_results = recon_agent.run_discord_search(
                target_model_type=target_model_type,
                target_behavior=target_behavior,
            )

            # Generate and save report
            report = recon_agent.generate_report(
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
                web_results=web_results,
                discord_results=discord_results,
            )

            report_path = recon_agent.save_report(
                report=report,
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
            )

            self.logger.info(
                f"Reconnaissance phase completed, report saved to {report_path}"
            )

            return report

        except Exception as e:
            self.logger.error(f"Reconnaissance phase failed: {str(e)}")

            return {
                "error": str(e),
                "target_model_type": str(target_model_type),
                "target_model_platform": str(target_model_platform),
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
            }

    def run_prompt_engineering(
        self,
        target_model_type: ModelType,
        target_model_platform: ModelPlatformType,
        target_behavior: str,
        recon_report: Dict[str, Any],
        num_prompts: int = 10,
    ) -> Dict[str, Any]:
        """
        Run the prompt engineering phase of the Gray Swan Arena pipeline.

        Args:
            target_model_type: The type of model to test
            target_model_platform: The platform of the model
            target_behavior: The behavior to target
            recon_report: Report from the reconnaissance phase
            num_prompts: Number of prompts to generate

        Returns:
            Dictionary containing the generated prompts and file path
        """
        self.logger.info(
            f"Starting prompt engineering phase for {target_model_type} on {target_model_platform} - {target_behavior}"
        )

        try:
            # Get PromptEngineerAgent from container
            prompt_agent = self.container.prompt_engineer_agent()

            # Generate prompts
            prompts = prompt_agent.generate_prompts(
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
                recon_report=recon_report,
                num_prompts=num_prompts,
            )

            # Save prompts
            prompts_path = prompt_agent.save_prompts(
                prompts=prompts,
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
            )

            self.logger.info(
                f"Prompt engineering phase completed, {len(prompts)} prompts saved to {prompts_path}"
            )

            return {
                "prompts": prompts,
                "path": prompts_path,
            }

        except Exception as e:
            self.logger.error(f"Prompt engineering phase failed: {str(e)}")

            return {
                "error": str(e),
                "target_model_type": str(target_model_type),
                "target_model_platform": str(target_model_platform),
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
                "prompts": [],
            }

    async def run_parallel_exploits(
        self,
        prompts: List[str],
        target_model_type: ModelType,
        target_model_platform: ModelPlatformType,
        target_behavior: str,
        method: str = "api",
        max_concurrent: int = 3,
    ) -> Dict[str, Any]:
        """
        Run exploit delivery in parallel batches.

        Args:
            prompts: List of prompts to test
            target_model_type: The type of model to test
            target_model_platform: The platform of the model
            target_behavior: The behavior to target
            method: Method to use (api, web)
            max_concurrent: Maximum number of concurrent tasks

        Returns:
            Dictionary containing exploit results and file path
        """
        self.logger.info(
            f"Starting parallel exploit delivery for {target_model_type} on {target_model_platform} - {target_behavior}"
        )

        try:
            # Get ExploitDeliveryAgent from container
            exploit_agent = self.container.exploit_delivery_agent()

            # Create batches of prompts
            batches = [
                prompts[i : i + max_concurrent]
                for i in range(0, len(prompts), max_concurrent)
            ]

            results = []
            for batch in batches:
                # Create tasks for concurrent execution
                tasks = [
                    asyncio.create_task(
                        asyncio.to_thread(
                            self._execute_single_prompt,
                            exploit_agent,
                            prompt,
                            target_model_type,
                            target_model_platform,
                            target_behavior,
                            method,
                        )
                    )
                    for prompt in batch
                ]

                # Wait for batch to complete
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)

                # Add small delay between batches
                await asyncio.sleep(1)

            # Save results
            results_path = exploit_agent.save_results(
                results=results,
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
            )

            self.logger.info(
                f"Parallel exploit delivery completed, results saved to {results_path}"
            )

            return {
                "results": results,
                "path": results_path,
            }

        except Exception as e:
            self.logger.error(f"Parallel exploit delivery failed: {str(e)}")

            return {
                "error": str(e),
                "target_model_type": str(target_model_type),
                "target_model_platform": str(target_model_platform),
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
                "results": [],
            }

    def _execute_single_prompt(
        self,
        agent: Any,
        prompt: str,
        target_model_type: ModelType,
        target_model_platform: ModelPlatformType,
        target_behavior: str,
        method: str,
    ) -> Dict[str, Any]:
        """
        Execute a single prompt using the specified method.

        Args:
            agent: The ExploitDeliveryAgent instance
            prompt: The prompt to test
            target_model_type: The type of model to test
            target_model_platform: The platform of the model
            target_behavior: The behavior to target
            method: Method to use (api, web)

        Returns:
            Dictionary containing the result
        """
        try:
            if method == "api":
                result = agent.run_api_exploit(
                    prompt=prompt,
                    target_model_type=target_model_type,
                    target_model_platform=target_model_platform,
                    target_behavior=target_behavior,
                )
            else:
                result = agent.run_web_exploit(
                    prompt=prompt,
                    target_model_type=target_model_type,
                    target_model_platform=target_model_platform,
                    target_behavior=target_behavior,
                )

            return result

        except Exception as e:
            self.logger.error(f"Error executing prompt: {str(e)}")

            return {
                "error": str(e),
                "prompt": prompt,
                "target_model_type": str(target_model_type),
                "target_model_platform": str(target_model_platform),
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
            }

    def run_exploit_delivery(
        self,
        prompts: List[str],
        target_model_type: ModelType,
        target_model_platform: ModelPlatformType,
        target_behavior: str,
        method: str = "api",
    ) -> Dict[str, Any]:
        """
        Run the exploit delivery phase of the Gray Swan Arena pipeline.

        Args:
            prompts: List of prompts to test
            target_model_type: The type of model to test
            target_model_platform: The platform of the model
            target_behavior: The behavior to target
            method: Method to use (api, web)

        Returns:
            Dictionary containing the exploit results
        """
        self.logger.info(
            f"Starting exploit delivery phase for {target_model_type} on {target_model_platform} - {target_behavior}"
        )

        try:
            # Get ExploitDeliveryAgent from container
            exploit_agent = self.container.exploit_delivery_agent()

            results = []
            for prompt in prompts:
                result = self._execute_single_prompt(
                    agent=exploit_agent,
                    prompt=prompt,
                    target_model_type=target_model_type,
                    target_model_platform=target_model_platform,
                    target_behavior=target_behavior,
                    method=method,
                )
                results.append(result)

            # Save results
            results_path = exploit_agent.save_results(
                results=results,
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
            )

            self.logger.info(
                f"Exploit delivery phase completed, results saved to {results_path}"
            )

            return {
                "results": results,
                "path": results_path,
            }

        except Exception as e:
            self.logger.error(f"Exploit delivery phase failed: {str(e)}")

            return {
                "error": str(e),
                "target_model_type": str(target_model_type),
                "target_model_platform": str(target_model_platform),
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
                "results": [],
            }

    def run_evaluation(
        self,
        exploit_results: List[Dict[str, Any]],
        target_model_type: ModelType,
        target_model_platform: ModelPlatformType,
        target_behavior: str,
        include_advanced_visualizations: bool = True,
        include_interactive_dashboard: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the evaluation phase of the Gray Swan Arena pipeline.

        Args:
            exploit_results: Results from the exploit delivery phase
            target_model_type: The type of model to test
            target_model_platform: The platform of the model
            target_behavior: The behavior to target
            include_advanced_visualizations: Whether to include advanced visualizations
            include_interactive_dashboard: Whether to include interactive dashboard

        Returns:
            Dictionary containing the evaluation results
        """
        self.logger.info(
            f"Starting evaluation phase for {target_model_type} on {target_model_platform} - {target_behavior}"
        )

        try:
            # Get EvaluationAgent from container
            evaluation_agent = self.container.evaluation_agent()

            # Generate evaluation report
            report = evaluation_agent.evaluate_results(
                test_results=exploit_results,
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
            )

            # Create visualizations if requested
            if include_advanced_visualizations:
                visualizations = evaluation_agent.create_advanced_visualizations(
                    test_results=exploit_results,
                    target_model_type=target_model_type,
                    target_model_platform=target_model_platform,
                    target_behavior=target_behavior,
                )
                report["visualizations"] = visualizations

            if include_interactive_dashboard:
                dashboard = evaluation_agent.create_interactive_dashboard(
                    test_results=exploit_results,
                    target_model_type=target_model_type,
                    target_model_platform=target_model_platform,
                    target_behavior=target_behavior,
                )
                report["dashboard"] = dashboard

            # Save evaluation results
            evaluation_path = evaluation_agent.save_evaluation(
                evaluation_results=report,
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
            )

            self.logger.info(
                f"Evaluation phase completed, results saved to {evaluation_path}"
            )

            return {
                "report": report,
                "path": evaluation_path,
            }

        except Exception as e:
            self.logger.error(f"Evaluation phase failed: {str(e)}")

            return {
                "error": str(e),
                "target_model_type": str(target_model_type),
                "target_model_platform": str(target_model_platform),
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
            }

    async def run_full_pipeline_async(
        self,
        target_model_type: ModelType,
        target_model_platform: ModelPlatformType,
        target_behavior: str,
        skip_phases: Optional[List[str]] = None,
        max_prompts: int = 10,
        test_method: str = "api",
        max_concurrent: int = 3,
        include_advanced_visualizations: bool = True,
        include_interactive_dashboard: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full Gray Swan Arena pipeline asynchronously.

        Args:
            target_model_type: The type of model to test
            target_model_platform: The platform of the model
            target_behavior: The behavior to target
            skip_phases: List of phases to skip
            max_prompts: Maximum number of prompts to generate
            test_method: Method to use for testing (api, web)
            max_concurrent: Maximum number of concurrent tasks
            include_advanced_visualizations: Whether to include advanced visualizations
            include_interactive_dashboard: Whether to include interactive dashboard

        Returns:
            Dictionary containing results from all phases
        """
        skip_phases = skip_phases or []
        results = {}

        # Run reconnaissance phase
        if "recon" not in skip_phases:
            recon_results = await self.run_parallel_reconnaissance(
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
            )
            results["recon"] = recon_results
        else:
            self.logger.info("Skipping reconnaissance phase")

        # Run prompt engineering phase
        if "prompt" not in skip_phases:
            prompt_results = self.run_prompt_engineering(
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
                recon_report=results.get("recon", {}),
                num_prompts=max_prompts,
            )
            results["prompt"] = prompt_results
        else:
            self.logger.info("Skipping prompt engineering phase")

        # Run exploit delivery phase
        if "exploit" not in skip_phases and "prompt" in results:
            exploit_results = await self.run_parallel_exploits(
                prompts=results["prompt"]["prompts"],
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
                method=test_method,
                max_concurrent=max_concurrent,
            )
            results["exploit"] = exploit_results
        else:
            self.logger.info("Skipping exploit delivery phase")

        # Run evaluation phase
        if "evaluate" not in skip_phases and "exploit" in results:
            evaluation_results = self.run_evaluation(
                exploit_results=results["exploit"]["results"],
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
                include_advanced_visualizations=include_advanced_visualizations,
                include_interactive_dashboard=include_interactive_dashboard,
            )
            results["evaluate"] = evaluation_results
        else:
            self.logger.info("Skipping evaluation phase")

        return results

    def run_full_pipeline(
        self,
        target_model_type: ModelType,
        target_model_platform: ModelPlatformType,
        target_behavior: str,
        skip_phases: Optional[List[str]] = None,
        max_prompts: int = 10,
        test_method: str = "api",
        max_concurrent: int = 3,
        include_advanced_visualizations: bool = True,
        include_interactive_dashboard: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full Gray Swan Arena pipeline synchronously.

        Args:
            target_model_type: The type of model to test
            target_model_platform: The platform of the model
            target_behavior: The behavior to target
            skip_phases: List of phases to skip
            max_prompts: Maximum number of prompts to generate
            test_method: Method to use for testing (api, web)
            max_concurrent: Maximum number of concurrent tasks
            include_advanced_visualizations: Whether to include advanced visualizations
            include_interactive_dashboard: Whether to include interactive dashboard

        Returns:
            Dictionary containing results from all phases
        """
        return asyncio.run(
            self.run_full_pipeline_async(
                target_model_type=target_model_type,
                target_model_platform=target_model_platform,
                target_behavior=target_behavior,
                skip_phases=skip_phases,
                max_prompts=max_prompts,
                test_method=test_method,
                max_concurrent=max_concurrent,
                include_advanced_visualizations=include_advanced_visualizations,
                include_interactive_dashboard=include_interactive_dashboard,
            )
        )


def main():
    """Run the Gray Swan Arena pipeline."""
    # Parse command line arguments
    args = parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load configuration
        config_file = args.config_file or os.getenv("GRAYSWAN_CONFIG", "config/development.yml")

        # Create container from config file
        container = GraySwanContainerFactory.create_container_from_file(config_file)

        # Create pipeline
        pipeline = GraySwanPipeline(container)

        # Run pipeline
        results = pipeline.run_full_pipeline(
            target_model_type=ModelType[args.target_model_type],
            target_model_platform=ModelPlatformType[args.target_model_platform],
            target_behavior=args.target_behavior,
            skip_phases=args.skip_phases,
            max_prompts=args.max_prompts,
            test_method=args.test_method,
            max_concurrent=args.max_concurrent,
            include_advanced_visualizations=args.advanced_visualizations,
            include_interactive_dashboard=args.interactive_dashboard,
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"pipeline_results_{timestamp}.json")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Pipeline results saved to %s", output_file)

        # Print summary
        print(f"\nGray Swan Arena pipeline completed for {args.target_model_type} on {args.target_model_platform}")
        print(f"Target behavior: {args.target_behavior}")

        if "error" in results:
            print(f"\nError: {results['error']}")
        else:
            print(f"\nSuccess rate: {results.get('success_rate', 'N/A')}%")
            
            if args.interactive_dashboard and "dashboard_path" in results:
                print(f"\nInteractive dashboard available at: {results['dashboard_path']}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
