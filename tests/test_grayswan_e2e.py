from typing import Any, Dict, List, Optional, Tuple, Union

"""End-to-end test for Gray Swan Arena."""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).parent.parent))

from cybersec_agents.grayswan import (
    EvaluationAgent,
    ExploitDeliveryAgent,
    PromptEngineerAgent,
    ReconAgent,
)

# Test configuration
TEST_MODEL: str = "gpt-3.5-turbo"  # Use GPT-3.5 Turbo for testing
OUTPUT_DIR = Path("tests/output")


def setup_test_environment():
    """Set up the test environment."""
    # Load environment variables
    load_dotenv()

    # Check for required API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY not found in environment variables")
        print("Please set it in a .env file or directly in your environment")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Test environment set up. Using model: {TEST_MODEL}")
    print(f"Output will be saved to {OUTPUT_DIR.absolute()}")


def test_recon_agent():
    """Test the Recon Agent."""
    print("\n=== Testing Recon Agent ===")

    try:
        # Initialize ReconAgent with output directory
        recon_output_dir = OUTPUT_DIR / "recon"
        recon_output_dir.mkdir(exist_ok=True)
        recon_agent = ReconAgent(output_dir=str(recon_output_dir), model_name="gpt-4")

        # Define target model and behavior
        target_model = TEST_MODEL  # "gpt-3.5-turbo"
        target_behavior: str = "jailbreak"

        # Run web search
        print(f"Running web search for {target_model} - {target_behavior}...")
        web_results = recon_agent.run_web_search(
            target_model=target_model,
            target_behavior=target_behavior,
            num_results=2,  # Limit results for testing
        )

        # For E2E testing, we'll mock Discord results
        print("Mocking Discord search results...")
        discord_results: dict[str, Any] = {
            "query": f"{target_model} {target_behavior}",
            "channels": ["ai-ethics", "red-teaming"],
            "results": [
                {
                    "channel": "ai-ethics",
                    "author": "user123",
                    "content": f"Has anyone tried the new prompt injection technique on {target_model}?",
                    "timestamp": "2023-03-15T14:30:00Z",
                }
            ],
            "timestamp": time.time(),
        }

        # Generate report
        print(f"Generating report for {target_model} - {target_behavior}...")
        report = recon_agent.generate_report(
            target_model=target_model,
            target_behavior=target_behavior,
            web_results=web_results,
            discord_results=discord_results,
        )

        # Save the report
        report_path = recon_agent.save_report(
            report=report, target_model=target_model, target_behavior=target_behavior
        )

        print(f"Recon report saved to {report_path}")
        return report_path

    except Exception as e:
        print(f"Error in Recon Agent test: {e}")
        return None


def test_prompt_engineer_agent(recon_report_path):
    """Test the Prompt Engineer Agent."""
    print("\n=== Testing Prompt Engineer Agent ===")

    if not recon_report_path:
        print("Recon report not found. Skipping Prompt Engineer test.")
        return None

    try:
        # Initialize PromptEngineerAgent with output directory
        prompt_output_dir = OUTPUT_DIR / "prompts"
        prompt_output_dir.mkdir(exist_ok=True)
        prompt_engineer = PromptEngineerAgent(
            output_dir=str(prompt_output_dir), model_name="gpt-4"
        )

        # Load the recon report
        print(f"Loading recon report from {recon_report_path}...")
        with open(recon_report_path, "r") as f:
            recon_report = json.load(f)

        # Define target model and behavior (should match what was used in recon)
        target_model = TEST_MODEL  # "gpt-3.5-turbo"
        target_behavior: str = "jailbreak"

        # Generate prompts
        print(f"Generating test prompts for {target_model} - {target_behavior}...")
        prompts = prompt_engineer.generate_prompts(
            target_model=target_model,
            target_behavior=target_behavior,
            recon_report=recon_report,
            num_prompts=2,  # Reduced for testing
        )

        # Save prompts
        prompts_path = prompt_engineer.save_prompts(
            prompts=prompts, target_model=target_model, target_behavior=target_behavior
        )

        print(f"Generated {len(prompts)} prompts and saved to {prompts_path}")
        return prompts_path

    except Exception as e:
        print(f"Error in Prompt Engineer Agent test: {e}")
        return None


def test_exploit_delivery_agent(prompts_path):
    """Test the Exploit Delivery Agent."""
    print("\n=== Testing Exploit Delivery Agent ===")

    if not prompts_path:
        print("Prompts file not found. Skipping Exploit Delivery test.")
        return None

    try:
        # Initialize ExploitDeliveryAgent with output directory
        exploit_output_dir = OUTPUT_DIR / "exploits"
        exploit_output_dir.mkdir(exist_ok=True)
        exploit_agent = ExploitDeliveryAgent(
            output_dir=str(exploit_output_dir), model_name="gpt-4"
        )

        # Load prompts
        print(f"Loading prompts from {prompts_path}...")
        with open(prompts_path, "r") as f:
            prompts = json.load(f)

        # Define target model and behavior (should match what was used in recon and prompt generation)
        target_model = TEST_MODEL  # "gpt-3.5-turbo"
        target_behavior: str = "jailbreak"

        # Execute prompts
        print(f"Executing {len(prompts)} prompts against {target_model}...")
        results: list[Any] = exploit_agent.run_prompts(
            prompts=prompts,
            target_model=target_model,
            target_behavior=target_behavior,
            method="api",  # Use API method for testing
            max_tries=1,  # Limit to 1 try for testing
            delay_between_tries=1,
        )

        # Save results
        results_path = exploit_agent.save_results(
            results=results, target_model=target_model, target_behavior=target_behavior
        )

        print(f"Executed {len(results)} prompts and saved results to {results_path}")
        return results_path

    except Exception as e:
        print(f"Error in Exploit Delivery Agent test: {e}")
        return None


def test_evaluation_agent(exploit_results_path, recon_report_path):
    """Test the Evaluation Agent."""
    print("\n=== Testing Evaluation Agent ===")

    if not exploit_results_path:
        print("Exploit results not found. Skipping Evaluation test.")
        return False

    try:
        # Initialize EvaluationAgent with output directory
        eval_output_dir = OUTPUT_DIR / "evaluations"
        eval_output_dir.mkdir(exist_ok=True)
        eval_agent = EvaluationAgent(
            output_dir=str(eval_output_dir), model_name="gpt-4"
        )

        # Load results
        print(f"Loading exploit results from {exploit_results_path}...")
        with open(exploit_results_path, "r") as f:
            results: list[Any] = json.load(f)

        # Define target model and behavior (should match what was used in previous steps)
        target_model = TEST_MODEL  # "gpt-3.5-turbo"
        target_behavior: str = "jailbreak"

        # Evaluate results
        print(f"Evaluating results for {target_model} - {target_behavior}...")
        evaluation = eval_agent.evaluate_results(
            results=results, target_model=target_model, target_behavior=target_behavior
        )

        # Save evaluation
        eval_path = eval_agent.save_evaluation(
            evaluation=evaluation,
            target_model=target_model,
            target_behavior=target_behavior,
        )
        print(f"Evaluation saved to {eval_path}")

        # Generate summary
        print("Generating summary report...")
        summary = eval_agent.generate_summary(
            evaluation=evaluation,
            target_model=target_model,
            target_behavior=target_behavior,
        )

        # Save summary
        summary_path = eval_agent.save_summary(
            summary=summary, target_model=target_model, target_behavior=target_behavior
        )
        print(f"Summary saved to {summary_path}")

        # Create visualizations
        print("Creating visualizations...")
        viz_paths = eval_agent.create_visualizations(
            evaluation=evaluation,
            target_model=target_model,
            target_behavior=target_behavior,
        )
        print(f"Visualizations saved to: {viz_paths}")

        return True

    except Exception as e:
        print(f"Error in Evaluation Agent test: {e}")
        return False


def run_e2e_test():
    """Run the end-to-end test of the Gray Swan Arena pipeline."""
    print("=" * 60)
    print("GRAY SWAN ARENA: END-TO-END TEST")
    print("=" * 60)

    # Set up test environment
    setup_test_environment()

    start_time = time.time()

    # Step 1: Recon
    recon_report_path = test_recon_agent()

    # Step 2: Prompt Engineering
    prompts_path = test_prompt_engineer_agent(recon_report_path)

    # Step 3: Exploit Delivery
    results_path = test_exploit_delivery_agent(prompts_path)

    # Step 4: Evaluation
    evaluation_success = test_evaluation_agent(results_path, recon_report_path)

    # Calculate total time
    total_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Recon: {'✅ PASSED' if recon_report_path else '❌ FAILED'}")
    print(f"Prompt Engineering: {'✅ PASSED' if prompts_path else '❌ FAILED'}")
    print(f"Exploit Delivery: {'✅ PASSED' if results_path else '❌ FAILED'}")
    print(f"Evaluation: {'✅ PASSED' if evaluation_success else '❌ FAILED'}")
    print(f"Total time: {total_time:.2f} seconds")

    if recon_report_path and prompts_path and results_path and evaluation_success:
        print("\n✅ END-TO-END TEST PASSED!")
        print(f"All outputs saved to {OUTPUT_DIR.absolute()}")
        return True
    else:
        print("\n❌ END-TO-END TEST FAILED!")
        print("See error messages above for details.")
        return False


if __name__ == "__main__":
    run_e2e_test()
