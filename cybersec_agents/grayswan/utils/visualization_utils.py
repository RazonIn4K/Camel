"""
Visualization utilities for the Gray Swan Arena.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .logging_utils import setup_logging
from camel.agents import ChatAgent

# Set up logger
logger = setup_logging("VisualizationUtils")


def ensure_output_dir(output_dir: str) -> str:
    """Ensure the output directory exists.

    Args:
        output_dir: Directory where visualization files will be saved

    Returns:
        The absolute path to the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directory exists: {output_path.absolute()}")
    return str(output_path.absolute())


def create_success_rate_chart(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Success Rate by Model and Prompt Type",
) -> str:
    """Create a chart showing success rates by model and prompt type.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        title: Title of the chart

    Returns:
        Path to the saved chart file
    """
    # Process results
    data: dict[str, Any] = []
    for result in results:
        model = result.get("model_name", "Unknown")
        prompt_type = result.get("prompt_type", "Unknown")
        success = result.get("success", False)
        data.append(
            {"Model": model, "Prompt Type": prompt_type, "Success": 1 if success else 0}
        )

    if not data:
        logger.warning("No data available for success rate chart")
        return ""

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate success rates
    success_rates = df.groupby(["Model", "Prompt Type"])["Success"].mean().reset_index()

    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # Create plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="Model", y="Success", hue="Prompt Type", data=success_rates)

    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Prompt Type", loc="best")
    plt.tight_layout()

    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Save plot
    output_file = os.path.join(output_dir, "success_rate_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created success rate chart: {output_file}")
    return output_file


def create_response_time_chart(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Response Time by Model",
) -> str:
    """Create a chart showing response times by model.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        title: Title of the chart

    Returns:
        Path to the saved chart file
    """
    # Process results
    data: dict[str, Any] = []
    for result in results:
        model = result.get("model_name", "Unknown")
        response_time = result.get("response_time", 0)
        data.append({"Model": model, "Response Time (s)": response_time})

    if not data:
        logger.warning("No data available for response time chart")
        return ""

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # Create plot
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x="Model", y="Response Time (s)", data=df)

    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Response Time (seconds)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "response_time_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created response time chart: {output_file}")
    return output_file


def create_prompt_type_effectiveness_chart(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Prompt Type Effectiveness",
) -> str:
    """Create a chart showing the effectiveness of different prompt types.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        title: Title of the chart

    Returns:
        Path to the saved chart file
    """
    # Process results
    data: dict[str, Any] = []
    for result in results:
        prompt_type = result.get("prompt_type", "Unknown")
        success = result.get("success", False)
        data.append({"Prompt Type": prompt_type, "Success": 1 if success else 0})

    if not data:
        logger.warning("No data available for prompt type effectiveness chart")
        return ""

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate success rates
    success_rates = df.groupby(["Prompt Type"])["Success"].mean().reset_index()
    counts = df.groupby(["Prompt Type"]).size().reset_index(name="Count")
    success_rates = success_rates.merge(counts, on="Prompt Type")

    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # Create plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="Prompt Type", y="Success", data=success_rates)

    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel("Prompt Type", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        count: int = counts.iloc[i]["Count"]
        ax.annotate(
            f"{p.get_height():.2f} (n={count})",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Save plot
    output_file = os.path.join(output_dir, "prompt_type_effectiveness_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created prompt type effectiveness chart: {output_file}")
    return output_file


def create_vulnerability_heatmap(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Vulnerability Heatmap by Model and Attack Vector",
) -> str:
    """Create a heatmap showing vulnerabilities by model and attack vector.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        title: Title of the chart

    Returns:
        Path to the saved chart file
    """
    # Process results
    data: dict[str, Any] = []
    for result in results:
        model = result.get("model_name", "Unknown")
        attack_vector = result.get("attack_vector", "Unknown")
        success = result.get("success", False)
        data.append(
            {
                "Model": model,
                "Attack Vector": attack_vector,
                "Success": 1 if success else 0,
            }
        )

    if not data:
        logger.warning("No data available for vulnerability heatmap")
        return ""

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate success rates
    success_rates: tuple[Any, ...] = (
        df.groupby(["Model", "Attack Vector"])["Success"].mean().reset_index()
    )

    # Pivot table for heatmap
    heatmap_data = success_rates.pivot(
        index="Model", columns="Attack Vector", values="Success"
    )

    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # Create plot
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        heatmap_data, annot=True, cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.5
    )

    # Customize plot
    plt.title(title, fontsize=16)
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "vulnerability_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created vulnerability heatmap: {output_file}")
    return output_file


def create_evaluation_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent,
    reasoning_agent: ChatAgent
) -> Dict[str, Any]:
    """Create a basic evaluation report.
    
    Args:
        results: List of test results to evaluate
        output_dir: Directory to save the report
        evaluation_agent: Agent to use for evaluation
        reasoning_agent: Agent to use for reasoning tasks
        
    Returns:
        Dict containing the evaluation report
    """
    # Implementation here
    pass


def create_advanced_evaluation_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent,
    reasoning_agent: ChatAgent
) -> Dict[str, Any]:
    """Create an advanced evaluation report with detailed analysis.
    
    Args:
        results: List of test results to evaluate
        output_dir: Directory to save the report
        evaluation_agent: Agent to use for evaluation
        reasoning_agent: Agent to use for reasoning tasks
        
    Returns:
        Dict containing the advanced evaluation report
    """
    # Implementation here
    pass


def create_interactive_dashboard(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent
) -> Dict[str, Any]:
    """Create an interactive dashboard for visualization.
    
    Args:
        results: List of test results to visualize
        output_dir: Directory to save the dashboard
        evaluation_agent: Agent to use for evaluation
        
    Returns:
        Dict containing the dashboard data
    """
    # Implementation here
    pass


def create_prompt_similarity_network(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent
) -> Dict[str, Any]:
    """Create a network visualization of prompt similarities.
    
    Args:
        results: List of test results to analyze
        output_dir: Directory to save the network
        evaluation_agent: Agent to use for evaluation
        
    Returns:
        Dict containing the network data
    """
    # Implementation here
    pass


def create_success_prediction_model(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent
) -> Dict[str, Any]:
    """Create a model for predicting attack success.
    
    Args:
        results: List of test results to train on
        output_dir: Directory to save the model
        evaluation_agent: Agent to use for evaluation
        
    Returns:
        Dict containing the model data
    """
    # Implementation here
    pass
