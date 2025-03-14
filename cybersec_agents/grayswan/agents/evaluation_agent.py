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
from camel.types.enums import RoleType

from ..utils.agentops_utils import (
    initialize_agentops,
    log_agentops_event,
    start_agentops_session,
)

# Import specific utilities directly
from ..utils.logging_utils import setup_logging
from ..utils.visualization_utils import create_evaluation_report
from ..utils.advanced_visualization_utils import (
    create_attack_pattern_visualization,
    create_prompt_similarity_network,
    create_success_prediction_model,
    create_interactive_dashboard,
    create_advanced_evaluation_report
)

# Set up logging using our logging utility
logger = setup_logging("evaluation_agent")


class EvaluationAgent:
    """Agent responsible for evaluating exploit results and producing reports."""

    def __init__(
        self,
        output_dir: str = "./evaluations", 
        model_name: str = "gpt-4",
        backup_model: Optional[str] = None,
        reasoning_model: Optional[str] = None,
    ):
        """
        Initialize the EvaluationAgent.

        Args:
            output_dir: Directory to save evaluations
            model_name: Name of the model to use for analysis
            backup_model: Name of the backup model to use if the primary model fails
            reasoning_model: Name of the model to use for reasoning tasks
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.backup_model = backup_model
        self.reasoning_model = reasoning_model or model_name

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualizations directory
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Initialize AgentOps
        api_key = os.getenv("AGENTOPS_API_KEY")
        if api_key:
            initialize_agentops(api_key)
            start_agentops_session(tags=["evaluation_agent"])

        # Log initialization
        log_agentops_event(
            "agent_initialized",
            {
                "agent_type": "evaluation",
                "output_dir": output_dir,
                "model_name": model_name,
            },
        )

        logger.info(f"EvaluationAgent initialized with model: {self.model_name}")
        if self.backup_model:
            logger.info(f"Backup model configured: {self.backup_model}")
        if self.reasoning_model != self.model_name:
            logger.info(f"Reasoning model configured: {self.reasoning_model}")

    def evaluate_results(
        self, results: List[Dict[str, Any]], target_model: str, target_behavior: str
    ) -> Dict[str, Any]:
        """
        Evaluate exploit results to generate statistics and insights.

        Args:
            results: List of exploit results
            target_model: The target model
            target_behavior: The behavior that was targeted

        Returns:
            Dictionary containing evaluation results
        """
        logger.info(
            f"Evaluating {len(results)} exploit results for {target_model} - {target_behavior}"
        )

        # Log evaluation start
        log_agentops_event(
            "evaluation_started",
            {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "num_results": len(results),
            },
        )

        try:
            # Calculate basic statistics
            total_attempts = len(results)
            successful_attempts = sum(1 for r in results if r.get("success", False))
            failed_attempts = total_attempts - successful_attempts
            success_rate = (
                successful_attempts / total_attempts if total_attempts > 0 else 0
            )

            # Count errors
            errors = sum(1 for r in results if "error" in r and r["error"] is not None)

            # Categorize by methods if available
            methods = {}
            for result in results:
                method = result.get("method", "unknown")
                if method not in methods:
                    methods[method] = {"total": 0, "success": 0, "rate": 0}

                methods[method]["total"] += 1
                if result.get("success", False):
                    methods[method]["success"] += 1

            # Calculate success rates by method
            for method in methods:
                methods[method]["rate"] = (
                    methods[method]["success"] / methods[method]["total"]
                    if methods[method]["total"] > 0
                    else 0
                )

            # Compile the evaluation
            evaluation = {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "failed_attempts": failed_attempts,
                "success_rate": success_rate,
                "errors": errors,
                "methods": methods,
            }

            # Log evaluation completion
            log_agentops_event(
                "evaluation_completed",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "success",
                    "success_rate": success_rate,
                    "total_attempts": total_attempts,
                },
            )

            return evaluation

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")

            # Log evaluation failure
            log_agentops_event(
                "evaluation_error",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "failed",
                    "error": str(e),
                },
            )

            return {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    def save_evaluation(
        self, evaluation: Dict[str, Any], target_model: str, target_behavior: str
    ) -> str:
        """
        Save evaluation results to a file.

        Args:
            evaluation: Evaluation results
            target_model: The target model
            target_behavior: The behavior targeted

        Returns:
            Path to the saved evaluation file
        """
        logger.info(f"Saving evaluation for {target_model} - {target_behavior}")

        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{target_model.lower().replace(' ', '_')}_{target_behavior.lower().replace(' ', '_')}_{timestamp}.json"
            filepath = self.output_dir / filename

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Save the evaluation
            with open(filepath, "w") as f:
                json.dump(evaluation, f, indent=2)

            logger.info(f"Evaluation saved to {filepath}")

            # Log evaluation saving
            log_agentops_event(
                "evaluation_saved",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "filepath": filepath,
                },
            )

            return filepath

        except Exception as e:
            logger.error(f"Failed to save evaluation: {str(e)}")

            # Log error
            log_agentops_event(
                "evaluation_save_error",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "failed",
                    "error": str(e),
                },
            )

            return ""

    def generate_summary(
        self, evaluation: Dict[str, Any], target_model: str, target_behavior: str
    ) -> Dict[str, Any]:
        """
        Generate a summary report based on evaluation results.

        Args:
            evaluation: Evaluation results
            target_model: The target model
            target_behavior: The behavior targeted

        Returns:
            Dictionary containing the summary report
        """
        logger.info(f"Generating summary for {target_model} - {target_behavior}")

        # Log summary generation start
        log_agentops_event(
            "summary_generation_started",
            {"target_model": target_model, "target_behavior": target_behavior},
        )

        try:
            # Initialize ChatAgent for summary generation
            system_message = (
                f"You are a cybersecurity analyst summarizing the results of red-team testing "
                f"conducted on the {target_model} AI model. The testing focused on whether "
                f"the model could be made to {target_behavior}."
            )

            agent = ChatAgent(system_message=system_message, model_name=self.model_name)

            # Create a message to send to the agent
            message = BaseMessage(
                role_name="User",
                role_type=RoleType.USER,
                meta_dict={},
                content=(
                    f"Generate a concise summary report based on the following evaluation results:\n\n"
                    f"{json.dumps(evaluation, indent=2)}\n\n"
                    f"Include an executive summary, key findings, and recommendations for improving "
                    f"the {target_model} model's resistance to attempts to make it {target_behavior}."
                ),
            )

            # Generate the summary
            response = agent.step(message)

            # Process the content into a structured summary
            summary_content = response.content

            # Create the summary structure
            summary = {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
                "content": summary_content,
                "sections": self._extract_sections(summary_content),
            }

            # Log summary generation completion
            log_agentops_event(
                "summary_generation_completed",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "success",
                    "sections": list(summary["sections"].keys()),
                },
            )

            return summary

        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")

            # Log summary generation failure
            log_agentops_event(
                "summary_generation_error",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "failed",
                    "error": str(e),
                },
            )

            return {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "content": f"Summary generation failed: {str(e)}",
                "sections": {},
            }

    def save_summary(
        self, summary: Dict[str, Any], target_model: str, target_behavior: str
    ) -> str:
        """
        Save summary to a file.

        Args:
            summary: Summary report
            target_model: The target model
            target_behavior: The behavior targeted

        Returns:
            Path to the saved summary file
        """
        logger.info(f"Saving summary for {target_model} - {target_behavior}")

        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{target_model.lower().replace(' ', '_')}_{target_behavior.lower().replace(' ', '_')}_{timestamp}.json"
            filepath = self.output_dir / filename

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Save the summary
            with open(filepath, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Summary saved to {filepath}")

            # Also create a more readable markdown version
            md_filename = f"summary_{target_model.lower().replace(' ', '_')}_{target_behavior.lower().replace(' ', '_')}_{timestamp}.md"
            md_filepath = self.output_dir / md_filename

            with open(md_filepath, "w") as f:
                f.write(f"# Summary Report: {target_model}\n\n")
                f.write(f"Target behavior: {target_behavior}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(summary["content"])

            # Log summary saving
            log_agentops_event(
                "summary_saved",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "filepath": filepath,
                    "md_filepath": md_filepath,
                },
            )

            return filepath

        except Exception as e:
            logger.error(f"Failed to save summary: {str(e)}")

            # Log error
            log_agentops_event(
                "summary_save_error",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "failed",
                    "error": str(e),
                },
            )

            return ""

    def create_visualizations(
        self, evaluation: Dict[str, Any], target_model: str, target_behavior: str
    ) -> Dict[str, str]:
        """
        Create basic visualizations based on evaluation results.

        Args:
            evaluation: Evaluation results
            target_model: The target model
            target_behavior: The behavior targeted

        Returns:
            Dictionary of visualization file paths
        """
        logger.info(f"Creating basic visualizations for {target_model} - {target_behavior}")

        # Log visualization creation start
        log_agentops_event(
            "visualization_creation_started",
            {"target_model": target_model, "target_behavior": target_behavior},
        )

        try:
            # Create a directory for visualizations
            vis_dir = self.viz_dir
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{target_model.lower().replace(' ', '_')}_{target_behavior.lower().replace(' ', '_')}_{timestamp}"

            # Dictionary to store visualization paths
            visualization_paths = {}

            # Set style for visualizations
            sns.set_style("darkgrid")
            plt.figure(figsize=(10, 6))

            # 1. Success vs. Failure Pie Chart
            plt.figure(figsize=(8, 8))
            success_data = [
                evaluation.get("successful_attempts", 0),
                evaluation.get("failed_attempts", 0),
            ]
            labels = ["Success", "Failure"]
            colors = ["#2ecc71", "#e74c3c"]

            plt.pie(
                success_data,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                explode=(0.1, 0),
            )
            plt.title(f"Success Rate: {target_model} - {target_behavior}", fontsize=14)

            pie_chart_path = vis_dir / f"{base_filename}_pie.png"
            plt.savefig(pie_chart_path)
            plt.close()

            visualization_paths["pie_chart"] = pie_chart_path

            # 2. Success Rate by Method Bar Chart (if methods exist)
            methods = evaluation.get("methods", {})
            if methods:
                plt.figure(figsize=(10, 6))
                method_names = list(methods.keys())
                success_rates = [methods[m]["rate"] for m in method_names]

                bar_positions = np.arange(len(method_names))
                plt.bar(bar_positions, success_rates, color="#3498db")
                plt.xticks(bar_positions, method_names)
                plt.ylabel("Success Rate")
                plt.title(f"Success Rate by Method: {target_model}", fontsize=14)

                # Add value labels on top of bars
                for i, v in enumerate(success_rates):
                    plt.text(i, v + 0.02, f"{v:.2%}", ha="center")

                method_chart_path = vis_dir / f"{base_filename}_methods.png"
                plt.savefig(method_chart_path)
                plt.close()

                visualization_paths["method_chart"] = method_chart_path

            # Log visualization creation completion
            log_agentops_event(
                "visualization_creation_completed",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "success",
                    "visualizations": list(visualization_paths.keys()),
                },
            )

            return visualization_paths

        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")

            # Log visualization creation failure
            log_agentops_event(
                "visualization_creation_error",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "failed",
                    "error": str(e),
                },
            )

            return {}
            
    def create_advanced_visualizations(
        self, results: List[Dict[str, Any]], target_model: str, target_behavior: str, include_interactive: bool = True
    ) -> Dict[str, str]:
        """
        Create advanced visualizations and analytics based on exploit results.
        
        This method uses the advanced visualization utilities to create sophisticated
        visualizations including attack pattern clustering, prompt similarity networks,
        success prediction models, and interactive dashboards.

        Args:
            results: List of exploit results (raw results, not the evaluation summary)
            target_model: The target model
            target_behavior: The behavior targeted
            include_interactive: Whether to include interactive dashboard

        Returns:
            Dictionary of visualization file paths
        """
        logger.info(f"Creating advanced visualizations for {target_model} - {target_behavior}")

        # Log advanced visualization creation start
        log_agentops_event(
            "advanced_visualization_creation_started",
            {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "include_interactive": include_interactive
            },
        )

        try:
            # Create a directory for visualizations
            vis_dir = self.viz_dir / "advanced"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Create advanced evaluation report with all visualizations
            visualization_paths = create_advanced_evaluation_report(
                results=results,
                output_dir=vis_dir,
                include_interactive=include_interactive
            )
            
            # Add target model and behavior to the paths for reference
            visualization_paths["target_model"] = target_model
            visualization_paths["target_behavior"] = target_behavior
            
            # Log advanced visualization creation completion
            log_agentops_event(
                "advanced_visualization_creation_completed",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "success",
                    "visualizations": list(visualization_paths.keys()),
                },
            )
            
            return visualization_paths
            
        except Exception as e:
            logger.error(f"Advanced visualization creation failed: {str(e)}")
            
            # Log advanced visualization creation failure
            log_agentops_event(
                "advanced_visualization_creation_error",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "failed",
                    "error": str(e),
                },
            )
            
            return {}

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract sections from report content.

        Args:
            content: Report content

        Returns:
            Dictionary of section titles and content
        """
        sections = {}
        current_section = "preamble"
        current_content = []

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
