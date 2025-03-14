"""
Example script demonstrating the enhanced browser automation capabilities.

This script shows how to use the EnhancedPlaywrightDriver to interact with
various AI model interfaces with improved reliability and self-healing capabilities.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from cybersec_agents.grayswan.utils.enhanced_browser_utils import EnhancedBrowserAutomationFactory
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("enhanced_browser_example")


def test_model_interface(
    model: str,
    prompt: str,
    headless: bool = False,
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    """
    Test a model interface using the enhanced browser automation.
    
    Args:
        model: The model to test (e.g., "gpt-4", "claude-2", "llama-2")
        prompt: The prompt to send to the model
        headless: Whether to run the browser in headless mode
        retry_attempts: Number of retry attempts for flaky operations
        retry_delay: Base delay between retry attempts
        
    Returns:
        Dictionary containing the result and metrics
    """
    logger.info(f"Testing model interface for {model} with prompt: {prompt[:50]}...")
    
    # Create an enhanced browser driver
    driver: Optional[Any] = None
    result: Any = {
        "model": model,
        "prompt": prompt,
        "response": None,
        "error": None,
        "metrics": None,
        "success": False,
    }
    
    try:
        # Create an enhanced browser driver with retry capabilities
        driver = EnhancedBrowserAutomationFactory.create_driver(
            method="playwright",
            headless=headless,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay
        )
        
        # Initialize the browser
        logger.info("Initializing browser...")
        driver.initialize()
        
        # Navigate to appropriate interface based on model
        logger.info(f"Navigating to interface for {model}...")
        if "gpt" in model.lower():
            driver.navigate("https://chat.openai.com/")
        elif "claude" in model.lower():
            driver.navigate("https://claude.ai/")
        elif "llama" in model.lower():
            driver.navigate("https://www.llama2.ai/")
        elif "gemini" in model.lower() or "bard" in model.lower():
            driver.navigate("https://gemini.google.com/")
        else:
            # Default to OpenAI
            logger.warning(f"No specific web interface defined for {model}, using OpenAI")
            driver.navigate("https://chat.openai.com/")
        
        # Execute the prompt and get the response
        logger.info("Executing prompt...")
        response = driver.execute_prompt(prompt, model, "")
        
        # Get metrics if available
        if hasattr(driver, 'get_metrics') and callable(getattr(driver, 'get_metrics')):
            metrics = driver.get_metrics()
            logger.info(f"Browser automation metrics: {metrics}")
            result["metrics"] = metrics
        
        result["response"] = response
        result["success"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing model interface: {str(e)}")
        result["error"] = str(e)
        return result
        
    finally:
        # Ensure browser is closed
        if driver:
            try:
                logger.info("Closing browser...")
                driver.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {str(e)}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Test enhanced browser automation with AI model interfaces"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-3.5-turbo", 
        help="Model to test (e.g., gpt-4, claude-2, llama-2)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Explain the concept of browser automation in simple terms.", 
        help="Prompt to send to the model"
    )
    parser.add_argument(
        "--visible", 
        action="store_true", 
        help="Run the browser in visible mode (not headless)"
    )
    parser.add_argument(
        "--retry-attempts", 
        type=int, 
        default=3, 
        help="Number of retry attempts for flaky operations"
    )
    parser.add_argument(
        "--retry-delay", 
        type=float, 
        default=1.0, 
        help="Base delay between retry attempts"
    )
    
    args = parser.parse_args()
    
    # Test the model interface
    result: Any = test_model_interface(
        model=args.model,
        prompt=args.prompt,
        headless=not args.visible,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
    )
    
    # Print the result
    print("\n" + "="*80)
    print(f"Model: {result['model']}")
    print(f"Prompt: {result['prompt'][:100]}...")
    print("="*80)
    
    if result["success"]:
        print("\nResponse:")
        print("-"*80)
        print(result["response"])
        print("-"*80)
        
        if result["metrics"]:
            print("\nMetrics:")
            print("-"*80)
            for key, value in result["metrics"].items():
                print(f"{key}: {value}")
            print("-"*80)
    else:
        print(f"\nError: {result['error']}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()