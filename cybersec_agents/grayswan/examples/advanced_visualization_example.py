"""
Example script demonstrating the advanced visualization capabilities.

This script shows how to use the advanced visualization utilities to create
sophisticated visualizations and interactive dashboards for Gray Swan Arena results.
"""

import os
import sys
import json
import random
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from cybersec_agents.grayswan.utils.advanced_visualization_utils import (
    create_attack_pattern_visualization,
    create_prompt_similarity_network,
    create_success_prediction_model,
    create_interactive_dashboard,
    create_advanced_evaluation_report
)
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("advanced_visualization_example")


def generate_sample_data(num_samples: int = 50) -> List[Dict[str, Any]]:
    """
    Generate sample data for visualization examples.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of sample result dictionaries
    """
    models: dict[str, Any] = ["GPT-4", "GPT-3.5-Turbo", "Claude-2", "Claude-Instant", "Llama-2-70B"]
    prompt_types: list[Any] = ["Direct", "Indirect", "Role-Playing", "Few-Shot", "Chain-of-Thought"]
    attack_vectors: list[Any] = ["Jailbreak", "Prompt Injection", "Context Manipulation", "Refusal Bypass", "Policy Evasion"]
    
    results: list[Any] = []
    
    for i in range(num_samples):
        # Select model with weighted probabilities
        model = random.choices(
            models, 
            weights=[0.3, 0.3, 0.2, 0.1, 0.1], 
            k=1
        )[0]
        
        # Select prompt type and attack vector
        prompt_type = random.choice(prompt_types)
        attack_vector = random.choice(attack_vectors)
        
        # Generate sample prompt
        prompt_length = random.randint(50, 500)
        prompt = f"Sample prompt {i+1} using {prompt_type} approach with {attack_vector} technique. " + "X" * (prompt_length - 50)
        
        # Determine success based on model and technique
        # Different models have different vulnerabilities to different techniques
        base_success_rate: dict[str, Any] = {
            "GPT-4": 0.2,
            "GPT-3.5-Turbo": 0.4,
            "Claude-2": 0.3,
            "Claude-Instant": 0.5,
            "Llama-2-70B": 0.6
        }.get(model, 0.3)
        
        # Adjust for prompt type
        prompt_type_factor: dict[str, Any] = {
            "Direct": 0.5,
            "Indirect": 0.7,
            "Role-Playing": 0.9,
            "Few-Shot": 0.8,
            "Chain-of-Thought": 0.6
        }.get(prompt_type, 0.7)
        
        # Adjust for attack vector
        attack_vector_factor: dict[str, Any] = {
            "Jailbreak": 0.8,
            "Prompt Injection": 0.7,
            "Context Manipulation": 0.6,
            "Refusal Bypass": 0.9,
            "Policy Evasion": 0.5
        }.get(attack_vector, 0.7)
        
        # Calculate success probability
        success_prob = base_success_rate * prompt_type_factor * attack_vector_factor
        success = random.random() < success_prob
        
        # Generate response time (successful attempts tend to take longer)
        base_response_time = random.uniform(0.5, 3.0)
        if success:
            response_time = base_response_time * random.uniform(1.0, 2.0)
        else:
            response_time = base_response_time * random.uniform(0.5, 1.0)
        
        # Create result dictionary
        result: Any = {
            "model_name": model,
            "prompt_type": prompt_type,
            "attack_vector": attack_vector,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "success": success,
            "response_time": response_time,
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat(),
            "word_count": len(prompt.split()),
            "question_marks": prompt.count("?"),
            "exclamation_marks": prompt.count("!"),
            "has_code_markers": random.random() < 0.3,  # 30% chance of having code markers
            "uppercase_ratio": random.uniform(0.05, 0.2)  # 5-20% uppercase
        }
        
        results.append(result)
    
    return results


def main():
    """Main function for the example script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Visualization Example for Gray Swan Arena"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output/visualizations", 
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=50, 
        help="Number of sample data points to generate"
    )
    parser.add_argument(
        "--save-data", 
        action="store_true", 
        help="Save the generated sample data to a file"
    )
    parser.add_argument(
        "--load-data", 
        type=str, 
        help="Load data from a file instead of generating"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Create interactive dashboard"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get data
    if args.load_data:
        try:
            with open(args.load_data, "r") as f:
                results: list[Any] = json.load(f)
            logger.info(f"Loaded {len(results)} results from {args.load_data}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.info("Generating sample data instead")
            results: list[Any] = generate_sample_data(args.num_samples)
    else:
        results: list[Any] = generate_sample_data(args.num_samples)
        logger.info(f"Generated {len(results)} sample results")
    
    # Save data if requested
    if args.save_data:
        data_file = os.path.join(args.output_dir, "sample_data.json")
        with open(data_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved sample data to {data_file}")
    
    # Create attack pattern visualization
    logger.info("Creating attack pattern visualization...")
    attack_pattern_file = create_attack_pattern_visualization(
        results, 
        args.output_dir,
        title="Sample Attack Pattern Clustering"
    )
    if attack_pattern_file:
        logger.info(f"Created attack pattern visualization: {attack_pattern_file}")
    
    # Create prompt similarity network
    logger.info("Creating prompt similarity network...")
    similarity_network_file = create_prompt_similarity_network(
        results, 
        args.output_dir,
        title="Sample Prompt Similarity Network"
    )
    if similarity_network_file:
        logger.info(f"Created prompt similarity network: {similarity_network_file}")
    
    # Create success prediction model
    logger.info("Creating success prediction model...")
    prediction_file, prediction_metrics = create_success_prediction_model(
        results, 
        args.output_dir,
        title="Sample Success Prediction Model"
    )
    if prediction_file:
        logger.info(f"Created success prediction model: {prediction_file}")
        logger.info(f"Model metrics: {prediction_metrics}")
    
    # Create interactive dashboard if requested
    if args.interactive:
        logger.info("Creating interactive dashboard...")
        dashboard_file = create_interactive_dashboard(
            results, 
            args.output_dir,
            title="Sample Gray Swan Arena Dashboard"
        )
        if dashboard_file:
            logger.info(f"Created interactive dashboard: {dashboard_file}")
    
    # Create comprehensive evaluation report
    logger.info("Creating comprehensive evaluation report...")
    report_files = create_advanced_evaluation_report(
        results, 
        args.output_dir,
        include_interactive=args.interactive
    )
    
    logger.info("Visualization examples completed successfully!")
    logger.info(f"Output files: {list(report_files.values())}")
    
    # Print instructions for viewing the dashboard
    if args.interactive and "interactive_dashboard" in report_files:
        dashboard_path = os.path.abspath(report_files["interactive_dashboard"])
        logger.info("\nTo view the interactive dashboard:")
        logger.info(f"  Open {dashboard_path} in a web browser")
        logger.info("  Or run: python -m http.server --directory {0} 8000".format(
            os.path.dirname(dashboard_path)
        ))
        logger.info("  Then navigate to http://localhost:8000/{0}".format(
            os.path.basename(dashboard_path)
        ))


if __name__ == "__main__":
    main()