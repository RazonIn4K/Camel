from typing import Any, Dict, List, Optional, Tuple, Union
#!/usr/bin/env python3
"""Example script demonstrating the visualization utilities in Gray Swan Arena.

This script shows how to use the visualization utilities to create charts and reports
from test results.
"""

import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cybersec_agents.grayswan.utils import (
    create_evaluation_report,
    create_prompt_type_effectiveness_chart,
    create_response_time_chart,
    create_success_rate_chart,
    create_vulnerability_heatmap,
    ensure_output_dir,
    setup_logging,
)

# Set up logging
logger = setup_logging("VisualizationExample")


def generate_sample_results(num_results=50):
    """Generate sample test results for demonstration purposes.

    Args:
        num_results: Number of sample results to generate

    Returns:
        List of result dictionaries
    """
    models: dict[str, Any] = ["GPT-3.5", "GPT-4", "Claude-2", "Llama-2"]
    prompt_types: list[Any] = [
        "Direct",
        "Indirect",
        "Chain",
        "Role-playing",
        "Context Manipulation",
    ]
    attack_vectors: list[Any] = [
        "Prompt Injection",
        "Jailbreaking",
        "Data Extraction",
        "Harmful Content",
        "Misinformation",
    ]

    results: list[Any] = []

    for i in range(num_results):
        model = random.choice(models)
        prompt_type = random.choice(prompt_types)
        attack_vector = random.choice(attack_vectors)

        # Simulate different success rates for different combinations
        base_success_rate: dict[str, Any] = {
            "GPT-3.5": 0.4,
            "GPT-4": 0.2,
            "Claude-2": 0.3,
            "Llama-2": 0.5,
        }.get(model, 0.3)

        prompt_type_modifier: dict[str, Any] = {
            "Direct": 0.1,
            "Indirect": 0.2,
            "Chain": 0.3,
            "Role-playing": 0.4,
            "Context Manipulation": 0.5,
        }.get(prompt_type, 0.3)

        success_probability = base_success_rate + prompt_type_modifier
        success = random.random() < success_probability

        # Generate a random response time between 1 and 10 seconds
        response_time = random.uniform(1.0, 10.0)

        # Create a result dictionary
        result: Any = {
            "id": f"test-{i+1}",
            "model_name": model,
            "prompt_type": prompt_type,
            "attack_vector": attack_vector,
            "success": success,
            "response_time": response_time,
            "timestamp": (
                datetime.now() - timedelta(minutes=random.randint(0, 60))
            ).isoformat(),
            "prompt": f"Sample prompt {i+1} using {prompt_type} technique",
            "response": (
                "Sample response from the model"
                if not success
                else "Response containing prohibited content"
            ),
            "score": random.uniform(0.0, 1.0),
        }

        results.append(result)

    return results


def main():
    """Main function demonstrating visualization utilities."""
    # Generate sample results
    logger.info("Generating sample test results")
    results: list[Any] = generate_sample_results(num_results=100)

    # Create output directory
    output_dir: str = "data/visualization_example"
    ensure_output_dir(output_dir)

    # Save sample results to a file
    results_file = os.path.join(output_dir, "sample_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved sample results to {results_file}")

    # Create individual charts
    logger.info("Creating success rate chart")
    success_chart = create_success_rate_chart(
        results, output_dir, title="Success Rate by Model and Prompt Type"
    )

    logger.info("Creating response time chart")
    time_chart = create_response_time_chart(
        results, output_dir, title="Response Time by Model"
    )

    logger.info("Creating prompt type effectiveness chart")
    effectiveness_chart = create_prompt_type_effectiveness_chart(
        results, output_dir, title="Prompt Type Effectiveness"
    )

    logger.info("Creating vulnerability heatmap")
    heatmap = create_vulnerability_heatmap(
        results, output_dir, title="Vulnerability Heatmap by Model and Attack Vector"
    )

    # Create a comprehensive HTML report
    logger.info("Creating comprehensive evaluation report")
    report_files = create_evaluation_report(results, output_dir)

    # Log the generated files
    logger.info("Generated visualization files:")
    for name, path in report_files.items():
        if path:
            logger.info(f"  {name}: {path}")

    logger.info(f"Visualization example completed. Output files are in {output_dir}")


if __name__ == "__main__":
    main()
