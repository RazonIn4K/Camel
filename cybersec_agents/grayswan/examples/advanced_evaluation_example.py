"""
Example script demonstrating the advanced evaluation capabilities.

This script shows how to use the EvaluationAgent with advanced visualization
and analytics capabilities to analyze exploit results.
"""

import os
import sys
import json
import random
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from cybersec_agents.grayswan.agents.evaluation_agent import EvaluationAgent
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("advanced_evaluation_example")


def generate_sample_data(num_samples: int = 50) -> List[Dict[str, Any]]:
    """
    Generate sample data for evaluation examples.
    
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
            "uppercase_ratio": random.uniform(0.05, 0.2),  # 5-20% uppercase
            "method": random.choice(["api", "web"]),  # Method used for testing
            "target_model": model,
            "target_behavior": "generate harmful content"
        }
        
        results.append(result)
    
    return results


def main():
    """Main function for the example script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Evaluation Example for Gray Swan Arena"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output/evaluations", 
        help="Directory to save evaluations"
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
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="gpt-4", 
        help="Model to use for evaluation agent"
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
    
    # Create evaluation agent
    logger.info(f"Creating evaluation agent with model {args.model_name}")
    eval_agent = EvaluationAgent(
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    # Extract target model and behavior from results
    # In a real scenario, these would be provided as parameters
    target_model: str = "Multiple Models"  # We're testing multiple models
    target_behavior: str = "Generate Harmful Content"
    
    # Evaluate results
    logger.info("Evaluating results...")
    evaluation = eval_agent.evaluate_results(
        results=results,
        target_model=target_model,
        target_behavior=target_behavior
    )
    
    # Save evaluation
    logger.info("Saving evaluation...")
    eval_path = eval_agent.save_evaluation(
        evaluation=evaluation,
        target_model=target_model,
        target_behavior=target_behavior
    )
    logger.info(f"Evaluation saved to {eval_path}")
    
    # Create basic visualizations
    logger.info("Creating basic visualizations...")
    basic_vis_paths = eval_agent.create_visualizations(
        evaluation=evaluation,
        target_model=target_model,
        target_behavior=target_behavior
    )
    logger.info(f"Basic visualizations created: {list(basic_vis_paths.keys())}")
    
    # Create advanced visualizations
    logger.info("Creating advanced visualizations...")
    advanced_vis_paths = eval_agent.create_advanced_visualizations(
        results=results,
        target_model=target_model,
        target_behavior=target_behavior,
        include_interactive=args.interactive
    )
    logger.info(f"Advanced visualizations created: {list(advanced_vis_paths.keys())}")
    
    # Generate summary
    logger.info("Generating summary...")
    summary = eval_agent.generate_summary(
        evaluation=evaluation,
        target_model=target_model,
        target_behavior=target_behavior
    )
    
    # Save summary
    logger.info("Saving summary...")
    summary_path = eval_agent.save_summary(
        summary=summary,
        target_model=target_model,
        target_behavior=target_behavior
    )
    logger.info(f"Summary saved to {summary_path}")
    
    # Print instructions for viewing the dashboard
    if args.interactive and "interactive_dashboard" in advanced_vis_paths:
        dashboard_path = os.path.abspath(advanced_vis_paths["interactive_dashboard"])
        logger.info("\nTo view the interactive dashboard:")
        logger.info(f"  Open {dashboard_path} in a web browser")
        logger.info("  Or run: python -m http.server --directory {0} 8000".format(
            os.path.dirname(dashboard_path)
        ))
        logger.info("  Then navigate to http://localhost:8000/{0}".format(
            os.path.basename(dashboard_path)
        ))
    
    logger.info("\nEvaluation example completed successfully!")


if __name__ == "__main__":
    main()