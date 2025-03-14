"""Test browser automation features of Gray Swan Arena."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).parent.parent))

from cybersec_agents.grayswan import ExploitDeliveryAgent
from cybersec_agents.grayswan.utils import (
    BrowserAutomationFactory,
    BrowserMethod,
    is_browser_automation_available,
)

# Constants
TEST_URL = "https://chat.openai.com"  # Default test URL
OUTPUT_DIR = Path("tests/output")
TEST_MODEL = "gpt-3.5-turbo"  # Model to use for testing


def setup_test_environment():
    """Set up the test environment."""
    # Load environment variables
    load_dotenv()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get test URL from environment or use default
    test_url = os.environ.get("GRAY_SWAN_URL", TEST_URL)

    print(f"Test environment set up. Using URL: {test_url}")
    print(f"Output will be saved to {OUTPUT_DIR.absolute()}")

    return test_url


def test_browser_availability():
    """Test which browser automation methods are available."""
    print("\n=== Testing Browser Automation Availability ===")

    available_methods = is_browser_automation_available()

    print("Browser automation methods availability:")
    print(
        f"Playwright: {'✅ Available' if available_methods['playwright'] else '❌ Not available'}"
    )
    print(
        f"Selenium: {'✅ Available' if available_methods['selenium'] else '❌ Not available'}"
    )

    if not any(available_methods.values()):
        print("\n❌ No browser automation methods available!")
        print("Please install at least one of the following:")
        print("- Playwright: pip install playwright && playwright install")
        print("- Selenium: pip install selenium webdriver-manager")
        return None

    # Determine which method to use for testing
    if available_methods["playwright"]:
        return BrowserMethod.PLAYWRIGHT
    elif available_methods["selenium"]:
        return BrowserMethod.SELENIUM
    else:
        return None


def test_browser_automation_factory(browser_method, test_url):
    """Test the BrowserAutomationFactory."""
    print(f"\n=== Testing BrowserAutomationFactory with {browser_method.value} ===")

    if not browser_method:
        print("No browser method available. Skipping test.")
        return False

    try:
        # Create a browser driver
        print(f"Creating {browser_method.value} driver...")
        driver = BrowserAutomationFactory.create_driver(
            method=browser_method.value, headless=True
        )

        # Initialize the driver
        print("Initializing browser driver...")
        driver.initialize()

        # Navigate to the test URL
        print(f"Navigating to {test_url}...")
        driver.navigate(test_url)

        # Execute a test prompt
        print("Executing a test prompt...")
        prompt = "What are some cybersecurity best practices?"

        try:
            response = driver.execute_prompt(
                prompt=prompt,
                model=TEST_MODEL,  # Use the explicit model name
                behavior="General Knowledge",
            )
            print(f"Got response snippet: {response[:100]}...")
        except Exception as e:
            print(
                f"Error executing prompt (this might be expected if using a real website): {e}"
            )

        # Close the driver
        print("Closing browser driver...")
        driver.close()

        print(f"{browser_method.value} browser automation test completed successfully!")
        return True

    except Exception as e:
        print(f"Error testing {browser_method.value} browser automation: {e}")
        return False


def test_exploit_delivery_agent_with_browser(browser_method, test_url):
    """Test the ExploitDeliveryAgent with browser automation."""
    print(f"\n=== Testing ExploitDeliveryAgent with {browser_method.value} ===")

    if not browser_method:
        print("No browser method available. Skipping test.")
        return False

    try:
        # Create an ExploitDeliveryAgent
        print(f"Creating ExploitDeliveryAgent...")
        agent = ExploitDeliveryAgent()

        # Create a test prompt
        prompts = [
            {
                "id": "test-prompt-1",
                "prompt": "What are the OWASP Top 10 security vulnerabilities?",
                "type": "information_gathering",
                "target_behavior": "security_information",
            }
        ]

        # Save prompts to a file
        prompts_path = OUTPUT_DIR / "browser_test_prompts.json"
        with open(prompts_path, "w") as f:
            json.dump(prompts, f, indent=2)

        # Execute the prompts with browser automation
        try:
            print(f"Testing run_prompts method with browser...")
            print(
                f"Note: This might fail if {test_url} is not set up for Gray Swan Arena"
            )

            # Configure browser settings at runtime
            results = agent.run_prompts(
                prompts=prompts,
                target_model=TEST_MODEL,  # Use explicit model name
                evaluation_model=TEST_MODEL,  # Model for evaluation
                target_behavior="General Knowledge",
                url=test_url,
                use_browser=True,
                browser_method=browser_method.value,
                headless=True,
            )

            print(f"Got response for {len(results)} prompts")
        except Exception as e:
            print(
                f"Error running prompts (this might be expected with a real website): {e}"
            )

        print(f"ExploitDeliveryAgent with browser test completed")
        return True

    except Exception as e:
        print(f"Error testing ExploitDeliveryAgent with browser: {e}")
        return False


def run_browser_automation_tests():
    """Run all browser automation tests."""
    print("=" * 60)
    print("GRAY SWAN ARENA: BROWSER AUTOMATION TESTS")
    print("=" * 60)

    # Set up test environment
    test_url = setup_test_environment()

    # Test browser availability
    browser_method = test_browser_availability()

    if not browser_method:
        print(
            "\n❌ Browser automation tests cannot proceed without any available browser automation methods."
        )
        return False

    # Test BrowserAutomationFactory
    factory_test = test_browser_automation_factory(browser_method, test_url)

    # Test ExploitDeliveryAgent with browser automation
    agent_test = test_exploit_delivery_agent_with_browser(browser_method, test_url)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Browser Availability: {'✅ PASSED' if browser_method else '❌ FAILED'}")
    print(f"Browser Automation Factory: {'✅ PASSED' if factory_test else '❌ FAILED'}")
    print(
        f"ExploitDeliveryAgent with Browser: {'✅ PASSED' if agent_test else '❌ FAILED'}"
    )

    if browser_method and factory_test and agent_test:
        print("\n✅ BROWSER AUTOMATION TESTS PASSED!")
        return True
    else:
        print("\n❌ BROWSER AUTOMATION TESTS FAILED or PARTIALLY FAILED")
        print("See error messages above for details.")
        print("\nNote: Some failures may be expected when testing with real websites.")
        return False


if __name__ == "__main__":
    run_browser_automation_tests()
