#!/usr/bin/env python3
"""
Automated Refinement Pipeline Script.

This script automates the entire testing pipeline by:
1. Loading prompt/response pairs from a JSON file
2. Running edge case tests
3. Evaluating the results
4. Generating and saving a comprehensive report
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add the parent directory to sys.path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cybersec_agents.grayswan.agents.evaluation_agent import EvaluationAgent
from scripts.run_edge_case_tests import run_prompt_response_tests
from cybersec_agents.grayswan.utils.data_handling_utils import load_json_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_report(evaluation_results: Dict[str, Any], output_path: Path, agent_id: str) -> str:
    """
    Generate a comprehensive report from evaluation results.
    
    Args:
        evaluation_results: Dictionary containing evaluation results
        output_path: Path to save the report
        agent_id: ID of the agent generating the report
        
    Returns:
        Path to the generated report file
    """
    logger.info("Generating comprehensive report...")
    
    try:
        # Validate evaluation results
        if not evaluation_results:
            logger.error("No evaluation results provided")
            raise ValueError("No evaluation results provided")
            
        # Create report directory if it doesn't exist
        report_dir = output_path.parent
        if not report_dir.exists():
            logger.info(f"Creating report directory: {report_dir}")
            report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"refinement_report_{timestamp}_{agent_id}.md"
        report_path = report_dir / report_filename
        
        # Extract data from evaluation results with validation
        num_prompts = evaluation_results.get("num_prompts_tested", 0)
        summaries = evaluation_results.get("summaries", [])
        visualizations = evaluation_results.get("visualizations", {})
        meta = evaluation_results.get("meta", {})
        
        if not summaries:
            logger.warning("No summaries found in evaluation results")
        
        # Calculate statistics
        success_count = sum(1 for s in summaries if s.get("success", False))
        success_rate = (success_count / num_prompts * 100) if num_prompts > 0 else 0
        
        # Generate markdown report
        try:
            with open(report_path, "w") as f:
                # Write header
                f.write("# Automated Refinement Report\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Agent ID: {agent_id}\n")
                f.write(f"Evaluation Model: {meta.get('evaluation_model', 'N/A')}\n\n")
                
                # Write summary statistics
                f.write("## Summary Statistics\n\n")
                f.write(f"- Total Prompts Tested: {num_prompts}\n")
                f.write(f"- Successful Responses: {success_count}\n")
                f.write(f"- Success Rate: {success_rate:.2f}%\n\n")
                
                # Write classification distribution
                f.write("## Classification Distribution\n\n")
                classifications = {}
                for summary in summaries:
                    classification = summary.get("classification", "unknown")
                    classifications[classification] = classifications.get(classification, 0) + 1
                
                for classification, count in classifications.items():
                    percentage = (count / num_prompts * 100) if num_prompts > 0 else 0
                    f.write(f"- {classification}: {count} ({percentage:.2f}%)\n")
                
                f.write("\n")
                
                # Write visualization references
                f.write("## Visualizations\n\n")
                for name, path in visualizations.items():
                    if not name.endswith("_error"):
                        f.write(f"### {name.replace('_', ' ').title()}\n")
                        f.write(f"![{name}]({path})\n\n")
                
                # Write detailed results
                f.write("## Detailed Results\n\n")
                for summary in summaries:
                    f.write(f"### {summary.get('prompt', 'Unknown Prompt')}\n\n")
                    f.write(f"**Response:** {summary.get('response', 'No response')}\n\n")
                    f.write(f"**Success:** {'Yes' if summary.get('success', False) else 'No'}\n")
                    f.write(f"**Classification:** {summary.get('classification', 'Unknown')}\n")
                    
                    if summary.get('error_details'):
                        f.write("\n**Error Details:**\n")
                        for error in summary['error_details']:
                            f.write(f"- {error}\n")
                    
                    f.write("\n")
            
            # Verify file was created and has content
            if not report_path.exists() or report_path.stat().st_size == 0:
                raise RuntimeError(f"Report file was not created or is empty at {report_path}")
                
            logger.info(f"Report generated successfully at {report_path}")
            return str(report_path)
            
        except IOError as e:
            logger.error(f"Failed to write report file: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise

def run_automated_refinement(input_file: str, output_path: str, model_name: str, agent_id: str) -> None:
    """
    Run the automated refinement pipeline.
    
    Args:
        input_file: Path to JSON file containing prompt/response pairs
        output_path: Path to save the report
        model_name: Name of the model to use for evaluation
        agent_id: ID of the agent running the pipeline
    """
    logger.info("Starting automated refinement pipeline...")
    
    try:
        # Load prompt/response pairs
        logger.info(f"Loading prompt/response pairs from {input_file}")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Convert data to prompt/response pairs if needed
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                # Convert list of strings to list of dictionaries
                prompt_response_pairs = [{"prompt": item, "response": "", "input_data": {}} for item in data]
            elif all(isinstance(item, dict) for item in data):
                # Use existing list of dictionaries
                prompt_response_pairs = data
            else:
                raise ValueError("Input file must contain either a list of strings or a list of dictionaries")
        else:
            raise ValueError("Input file must contain a list")
        
        # Run edge case tests
        logger.info("Running edge case tests...")
        test_results = run_prompt_response_tests(prompt_response_pairs)
        
        # Initialize evaluation agent
        logger.info("Initializing evaluation agent...")
        evaluation_agent = EvaluationAgent(
            output_dir=os.path.dirname(output_path),
            model_name=model_name,
            agent_id=agent_id
        )
        
        # Evaluate results
        logger.info("Evaluating test results...")
        evaluation_results = evaluation_agent.evaluate_results(test_results)
        
        # Generate report
        logger.info("Generating report...")
        report_path = generate_report(evaluation_results, Path(output_path), agent_id)
        
        logger.info(f"Automated refinement completed successfully. Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in automated refinement pipeline: {str(e)}")
        raise

def main():
    """Main entry point for the script."""
    # Default values
    input_file = "reports/prompts/prompts_airbnb_credentials_leak_to_instagram_20250315_040405.json"
    output_path = "reports/refinement/refinement_report.json"
    model_name = "gpt-4"
    agent_id = "automated_refinement_agent"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        run_automated_refinement(
            input_file,
            output_path,
            model_name,
            agent_id
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main()) 