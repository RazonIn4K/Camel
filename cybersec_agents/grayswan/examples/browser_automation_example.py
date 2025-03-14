#!/usr/bin/env python3
"""Example script demonstrating the browser automation utilities in Gray Swan Arena.

This script shows how to use the browser automation utilities to interact with a web-
based AI model interface.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the parent directory to the path so we can import the package
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cybersec_agents.grayswan.utils import (
    BrowserAutomationFactory,
    is_browser_automation_available,
    setup_logging,
)

# Set up logging
logger = setup_logging("BrowserAutomationExample")


def main():
    """Main function demonstrating browser automation utilities."""
    # Load environment variables
    load_dotenv()

    # Check which browser automation methods are available
    available_methods = is_browser_automation_available()
    logger.info(f"Available browser automation methods:")
    logger.info(f"  Playwright: {available_methods['playwright']}")
    logger.info(f"  Selenium: {available_methods['selenium']}")

    # Determine which method to use
    if available_methods["playwright"]:
        method = "playwright"
    elif available_methods["selenium"]:
        method = "selenium"
    else:
        logger.error(
            "No browser automation methods available. Please install Playwright or Selenium."
        )
        return

    logger.info(f"Using {method} for browser automation")

    # Get the target URL from environment variables or use a default
    target_url = os.getenv("GRAY_SWAN_URL", "https://chat.openai.com")

    # Create a browser driver
    try:
        driver = BrowserAutomationFactory.create_driver(
            method=method,
            headless=os.getenv("GRAYSWAN_BROWSER_HEADLESS", "true").lower() == "true",
        )

        # Initialize the driver
        logger.info("Initializing browser driver")
        driver.initialize()

        # Navigate to the target URL
        logger.info(f"Navigating to {target_url}")
        driver.navigate(target_url)

        # Example prompt to test
        prompt = "What is the capital of France?"
        model = "GPT-3.5"
        behavior = "General Knowledge"

        # Execute the prompt
        logger.info(f"Executing prompt: {prompt}")
        response = driver.execute_prompt(prompt, model, behavior)

        # Log the response
        logger.info(f"Response received:")
        logger.info(response)

    except Exception as e:
        logger.error(f"Error during browser automation: {e}")
    finally:
        # Close the driver
        if "driver" in locals():
            logger.info("Closing browser driver")
            driver.close()

    logger.info("Browser automation example completed")


if __name__ == "__main__":
    main()
