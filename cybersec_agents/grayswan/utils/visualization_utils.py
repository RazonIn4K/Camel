"""Visualization utilities for Gray Swan Arena."""

import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .logging_utils import setup_logging

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
    data = []
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
    data = []
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
    data = []
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
        count = counts.iloc[i]["Count"]
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
    data = []
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
    success_rates = (
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
    results: List[Dict[str, Any]], output_dir: str
) -> Dict[str, str]:
    """Create a comprehensive evaluation report with multiple visualizations.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved

    Returns:
        Dictionary mapping chart names to file paths
    """
    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # Create visualizations
    report_files = {
        "success_rate": create_success_rate_chart(results, output_dir),
        "response_time": create_response_time_chart(results, output_dir),
        "prompt_effectiveness": create_prompt_type_effectiveness_chart(
            results, output_dir
        ),
        "vulnerability_heatmap": create_vulnerability_heatmap(results, output_dir),
    }

    # Generate HTML report
    html_content = generate_html_report(results, report_files)
    html_path = os.path.join(output_dir, "evaluation_report.html")

    with open(html_path, "w") as f:
        f.write(html_content)

    logger.info(f"Created evaluation report: {html_path}")

    # Add HTML report to files dictionary
    report_files["html_report"] = html_path

    return report_files


def generate_html_report(
    results: List[Dict[str, Any]], chart_files: Dict[str, str]
) -> str:
    """Generate an HTML report from evaluation results and chart files.

    Args:
        results: List of result dictionaries
        chart_files: Dictionary mapping chart names to file paths

    Returns:
        HTML content as a string
    """
    # Calculate overall statistics
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get("success", False))
    success_rate = successful_tests / total_tests if total_tests > 0 else 0

    # Count unique models and prompt types
    models = {r.get("model_name", "Unknown") for r in results}
    prompt_types = {r.get("prompt_type", "Unknown") for r in results}

    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gray Swan Arena Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
                background-color: #f9f9f9;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .stats-container {{
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }}
            .stat-box {{
                flex: 1;
                min-width: 200px;
                padding: 15px;
                margin: 10px;
                background: #f5f5f5;
                border-left: 4px solid #3498db;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .chart-container {{
                margin-bottom: 30px;
            }}
            .chart {{
                max-width: 100%;
                height: auto;
                margin-top: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                text-align: left;
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f5f5f5;
            }}
            .success {{
                color: green;
            }}
            .failure {{
                color: red;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Gray Swan Arena Evaluation Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <div class="stats-container">
                <div class="stat-box">
                    <h3>Total Tests</h3>
                    <p>{total_tests}</p>
                </div>
                <div class="stat-box">
                    <h3>Success Rate</h3>
                    <p>{success_rate:.2%}</p>
                </div>
                <div class="stat-box">
                    <h3>Models Tested</h3>
                    <p>{len(models)}</p>
                </div>
                <div class="stat-box">
                    <h3>Prompt Types</h3>
                    <p>{len(prompt_types)}</p>
                </div>
            </div>
    """

    # Add charts
    for chart_name, chart_path in chart_files.items():
        if chart_name == "html_report" or not chart_path:
            continue

        chart_title = chart_name.replace("_", " ").title()
        relative_path = os.path.basename(chart_path)

        html += f"""
            <div class="chart-container">
                <h2>{chart_title}</h2>
                <img class="chart" src="{relative_path}" alt="{chart_title}">
            </div>
        """

    # Add results table
    html += """
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Prompt Type</th>
                    <th>Attack Vector</th>
                    <th>Success</th>
                    <th>Response Time (s)</th>
                </tr>
    """

    for result in results:
        model = result.get("model_name", "Unknown")
        prompt_type = result.get("prompt_type", "Unknown")
        attack_vector = result.get("attack_vector", "Unknown")
        success = result.get("success", False)
        response_time = result.get("response_time", 0)

        success_class = "success" if success else "failure"
        success_text = "Yes" if success else "No"

        html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{prompt_type}</td>
                    <td>{attack_vector}</td>
                    <td class="{success_class}">{success_text}</td>
                    <td>{response_time:.2f}</td>
                </tr>
        """

    # Close table and HTML
    html += """
            </table>
        </div>
    </body>
    </html>
    """

    return html
