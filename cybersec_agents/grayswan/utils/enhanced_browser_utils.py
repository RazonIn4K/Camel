"""Enhanced browser automation utilities for Gray Swan Arena."""

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .browser_utils import (
    BrowserDriver,
    BrowserMethod,
    PlaywrightDriver,
    SeleniumDriver,
)
from .logging_utils import setup_logging

# Set up logger
logger = setup_logging("EnhancedBrowserUtils")


class EnhancedPlaywrightDriver(PlaywrightDriver):
    """
    Enhanced browser driver using Playwright with adaptive selectors and self-healing capabilities.

    This class extends the base PlaywrightDriver with:
    1. Alternative selectors for different UI patterns
    2. Self-healing capabilities for handling UI changes
    3. Retry mechanisms for flaky interactions
    4. Improved error handling and recovery
    """

    def __init__(
        self, headless: bool = True, retry_attempts: int = 3, retry_delay: float = 1.0
    ):
        """
        Initialize the Enhanced Playwright driver.

        Args:
            headless: Whether to run the browser in headless mode
            retry_attempts: Number of retry attempts for flaky operations
            retry_delay: Base delay between retry attempts (will be adjusted with jitter)
        """
        super().__init__(headless)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Define alternative selectors for different UI patterns
        self.selector_alternatives = {
            "model_select": [
                "#model-select",
                ".model-dropdown",
                "[data-testid='model-selector']",
                "select[name='model']",
                "div[role='combobox'][aria-label*='model']",
            ],
            "behavior_select": [
                "#behavior-select",
                ".behavior-dropdown",
                "[data-testid='behavior-selector']",
                "select[name='behavior']",
                "div[role='combobox'][aria-label*='behavior']",
            ],
            "prompt_input": [
                "#prompt-textarea",
                ".prompt-input",
                "[data-testid='prompt-input']",
                "textarea",
                "div[contenteditable='true']",
                ".input-area textarea",
            ],
            "submit_button": [
                "#submit-button",
                "button[type='submit']",
                ".submit-btn",
                "button:has-text('Submit')",
                "button:has-text('Send')",
                "button.primary",
                "[data-testid='submit-button']",
            ],
            "response_output": [
                "#response-container",
                ".response-content",
                "[data-testid='response']",
                ".output-area",
                ".model-response",
                ".response-text",
            ],
            "success_indicator": [
                ".success-indicator",
                "[data-testid='success-indicator']",
                ".success-badge",
                "div.success",
                "span.success-text",
            ],
        }

        # Keep track of which selectors worked last time
        self.working_selectors = {}

        # Track metrics for self-healing
        self.metrics = {
            "selector_fallbacks": 0,
            "retry_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
        }

    def with_retry(self, operation: Callable, error_message: str) -> Any:
        """
        Execute an operation with retry logic.

        Args:
            operation: The function to execute
            error_message: Message to log on failure

        Returns:
            The result of the operation

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception: Optional[Any] = None

        for attempt in range(self.retry_attempts):
            try:
                return operation()
            except Exception as e:
                last_exception = e
                self.metrics["retry_attempts"] += 1

                if attempt < self.retry_attempts - 1:
                    # Calculate delay with jitter
                    jitter = random.uniform(0.75, 1.25)
                    delay = self.retry_delay * (attempt + 1) * jitter

                    logger.warning(
                        f"Attempt {attempt + 1}/{self.retry_attempts} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)

        # If we get here, all retries failed
        self.metrics["failed_recoveries"] += 1
        logger.error(f"{error_message}: {str(last_exception)}")
        raise last_exception

    def try_selectors(self, selector_type: str, action_fn: Callable[[str], Any]) -> Any:
        """
        Try multiple selectors to find a working one.

        Args:
            selector_type: The type of selector to try (e.g., "model_select")
            action_fn: Function that takes a selector and performs an action

        Returns:
            The result of the action function

        Raises:
            ValueError: If all selectors fail
        """
        # First try the selector that worked last time
        if selector_type in self.working_selectors:
            try:
                result: Any = action_fn(self.working_selectors[selector_type])
                return result
            except Exception:
                # If it fails, continue to try alternatives
                pass

        # Try the default selector
        default_selector = self.selectors.get(selector_type)
        if default_selector:
            try:
                result: Any = action_fn(default_selector)
                self.working_selectors[selector_type] = default_selector
                return result
            except Exception:
                # If it fails, continue to try alternatives
                pass

        # Try alternative selectors
        selectors = self.selector_alternatives.get(selector_type, [])
        for selector in selectors:
            try:
                result: Any = action_fn(selector)
                # Update the working selector for future use
                self.working_selectors[selector_type] = selector
                self.metrics["selector_fallbacks"] += 1
                self.metrics["successful_recoveries"] += 1
                logger.info(f"Found working selector for {selector_type}: {selector}")
                return result
            except Exception:
                continue

        # If all selectors fail, raise an exception
        self.metrics["failed_recoveries"] += 1
        raise ValueError(f"All selectors failed for {selector_type}")

    def initialize(self):
        """Initialize the Playwright browser session with retry logic."""

        def _initialize():
            return super().initialize()

        return self.with_retry(
            _initialize, "Error initializing enhanced Playwright browser"
        )

    def navigate(self, url: str):
        """
        Navigate to a URL with retry logic.

        Args:
            url: The URL to navigate to
        """
        if not self.page:
            raise ValueError("Browser not initialized. Call initialize() first.")

        def _navigate():
            super().navigate(url)

        return self.with_retry(_navigate, f"Error navigating to {url}")

    def execute_prompt(self, prompt: str, model: str, behavior: str) -> str:
        """
        Execute a prompt against a model with enhanced reliability.

        Args:
            prompt: The prompt to execute
            model: The model to execute against
            behavior: The behavior to test for

        Returns:
            The model's response
        """
        if not self.page:
            raise ValueError("Browser not initialized. Call initialize() first.")

        try:
            # Select model if available
            def select_model(selector):
                if self.page.locator(selector).count() > 0:
                    self.page.select_option(selector, model)
                    logger.info(f"Selected model: {model}")
                return True

            try:
                self.try_selectors("model_select", select_model)
            except ValueError as e:
                logger.warning(f"Could not select model: {e}")

            # Select behavior if available
            def select_behavior(selector):
                if self.page.locator(selector).count() > 0:
                    self.page.select_option(selector, behavior)
                    logger.info(f"Selected behavior: {behavior}")
                return True

            try:
                self.try_selectors("behavior_select", select_behavior)
            except ValueError as e:
                logger.warning(f"Could not select behavior: {e}")

            # Enter prompt
            def enter_prompt(selector):
                # Check if it's a contenteditable div
                if "contenteditable" in selector:
                    self.page.evaluate(
                        f'document.querySelector("{selector}").innerText = "{prompt}"'
                    )
                else:
                    self.page.fill(selector, prompt)
                logger.info("Entered prompt")
                return True

            self.try_selectors("prompt_input", enter_prompt)

            # Submit
            def click_submit(selector):
                self.page.click(selector)
                logger.info("Submitted prompt")
                return True

            self.try_selectors("submit_button", click_submit)

            # Wait for response with retry logic
            def wait_for_response(selector):
                self.page.wait_for_selector(selector, state="visible", timeout=60000)
                return self.page.text_content(selector)

            response = self.try_selectors("response_output", wait_for_response)
            logger.info("Received response")

            return response
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            return f"Error: {str(e)}"

    def get_metrics(self) -> Dict[str, int]:
        """
        Get metrics about the browser automation.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()

    def close(self):
        """Close the Playwright browser session with retry logic."""
        if not (self.browser or hasattr(self, "_playwright")):
            return

        def _close():
            super().close()

        try:
            self.with_retry(_close, "Error closing Playwright browser")
        except Exception as e:
            logger.error(f"Failed to close browser gracefully: {e}")
            # Force close as a last resort
            if self.browser:
                try:
                    self.browser.close()
                except:
                    pass
            if hasattr(self, "_playwright"):
                try:
                    self._playwright.stop()
                except:
                    pass


class EnhancedSeleniumDriver(SeleniumDriver):
    """
    Enhanced browser driver using Selenium with adaptive selectors and self-healing capabilities.

    This class extends the base SeleniumDriver with:
    1. Alternative selectors for different UI patterns
    2. Self-healing capabilities for handling UI changes
    3. Retry mechanisms for flaky interactions
    4. Improved error handling and recovery
    """

    def __init__(
        self, headless: bool = True, retry_attempts: int = 3, retry_delay: float = 1.0
    ):
        """
        Initialize the Enhanced Selenium driver.

        Args:
            headless: Whether to run the browser in headless mode
            retry_attempts: Number of retry attempts for flaky operations
            retry_delay: Base delay between retry attempts (will be adjusted with jitter)
        """
        super().__init__(headless)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Define alternative selectors for different UI patterns
        self.selector_alternatives = {
            "model_select": [
                "#model-select",
                ".model-dropdown",
                "[data-testid='model-selector']",
                "select[name='model']",
                "div[role='combobox'][aria-label*='model']",
            ],
            "behavior_select": [
                "#behavior-select",
                ".behavior-dropdown",
                "[data-testid='behavior-selector']",
                "select[name='behavior']",
                "div[role='combobox'][aria-label*='behavior']",
            ],
            "prompt_input": [
                "#prompt-textarea",
                ".prompt-input",
                "[data-testid='prompt-input']",
                "textarea",
                "div[contenteditable='true']",
                ".input-area textarea",
            ],
            "submit_button": [
                "#submit-button",
                "button[type='submit']",
                ".submit-btn",
                "button:contains('Submit')",
                "button:contains('Send')",
                "button.primary",
                "[data-testid='submit-button']",
            ],
            "response_output": [
                "#response-container",
                ".response-content",
                "[data-testid='response']",
                ".output-area",
                ".model-response",
                ".response-text",
            ],
            "success_indicator": [
                ".success-indicator",
                "[data-testid='success-indicator']",
                ".success-badge",
                "div.success",
                "span.success-text",
            ],
        }

        # Keep track of which selectors worked last time
        self.working_selectors = {}

        # Track metrics for self-healing
        self.metrics = {
            "selector_fallbacks": 0,
            "retry_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
        }

    # Implementation of EnhancedSeleniumDriver methods would go here
    # Similar to EnhancedPlaywrightDriver but using Selenium APIs


class EnhancedBrowserAutomationFactory:
    """Factory for creating enhanced browser drivers."""

    @staticmethod
    def create_driver(
        method: str,
        headless: bool = True,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ) -> BrowserDriver:
        """
        Create an enhanced browser driver.

        Args:
            method: The browser automation method to use ("playwright" or "selenium")
            headless: Whether to run the browser in headless mode
            retry_attempts: Number of retry attempts for flaky operations
            retry_delay: Base delay between retry attempts

        Returns:
            A BrowserDriver instance
        """
        method = method.lower()

        if method == "playwright":
            driver = EnhancedPlaywrightDriver(
                headless=headless,
                retry_attempts=retry_attempts,
                retry_delay=retry_delay,
            )
        elif method == "selenium":
            driver = EnhancedSeleniumDriver(
                headless=headless,
                retry_attempts=retry_attempts,
                retry_delay=retry_delay,
            )
        else:
            raise ValueError(f"Unsupported browser automation method: {method}")

        return driver
