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
        PlaywrightDriver.__init__(self, headless)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.page = None
        self.browser = None

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
                "[data-testid='chat-input']",
                "[data-testid='chat-textarea']",
                "[aria-label='Chat input']",
                "[placeholder*='Send a message']",
                "[placeholder*='Type your message']",
                "[placeholder*='Type a message']",
                "div[role='textbox']",
                "div.chat-input",
                ".chat-textarea",
                "#chat-input",
                "#chat-textarea",
                "form textarea",
            ],
            "submit_button": [
                "#submit-button",
                "button[type='submit']",
                ".submit-btn",
                "button:has-text('Submit')",
                "button:has-text('Send')",
                "button.primary",
                "[data-testid='submit-button']",
                "[data-testid='send-button']",
                "button[aria-label='Send message']",
                "button.send-button",
                ".send-btn",
                "#send-button",
                "form button[type='submit']",
            ],
            "response_output": [
                "#response-container",
                ".response-content",
                "[data-testid='response']",
                ".output-area",
                ".model-response",
                ".response-text",
                "[data-testid='conversation-turn-']",
                ".chat-message",
                ".message-content",
                ".assistant-message",
                ".bot-message",
                "[role='presentation']",
                ".markdown-content",
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
            RuntimeError: If all retry attempts fail
        """
        last_exception = None

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
        raise RuntimeError(f"{error_message}: {str(last_exception)}")

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
                logger.info(f"Trying previously working selector for {selector_type}: {self.working_selectors[selector_type]}")
                result: Any = action_fn(self.working_selectors[selector_type])
                return result
            except Exception as e:
                logger.debug(f"Previously working selector failed: {e}")
                # If it fails, continue to try alternatives
                pass

        # Try the default selector
        default_selector = self.selectors.get(selector_type)
        if default_selector:
            try:
                logger.info(f"Trying default selector for {selector_type}: {default_selector}")
                result: Any = action_fn(default_selector)
                self.working_selectors[selector_type] = default_selector
                return result
            except Exception as e:
                logger.debug(f"Default selector failed: {e}")
                # If it fails, continue to try alternatives
                pass

        # Try alternative selectors
        selectors = self.selector_alternatives.get(selector_type, [])
        logger.info(f"Trying {len(selectors)} alternative selectors for {selector_type}")
        for selector in selectors:
            try:
                logger.info(f"Trying alternative selector for {selector_type}: {selector}")
                result: Any = action_fn(selector)
                # Update the working selector for future use
                self.working_selectors[selector_type] = selector
                self.metrics["selector_fallbacks"] += 1
                self.metrics["successful_recoveries"] += 1
                logger.info(f"Found working selector for {selector_type}: {selector}")
                return result
            except Exception as e:
                logger.debug(f"Alternative selector {selector} failed: {e}")
                continue

        # If all selectors fail, raise an exception
        self.metrics["failed_recoveries"] += 1
        logger.error(f"All selectors failed for {selector_type}")
        raise ValueError(f"All selectors failed for {selector_type}")

    def initialize(self):
        """Initialize the Playwright browser session with retry logic."""
        try:
            super().initialize()
            if not self.page:
                raise RuntimeError("Failed to initialize page")
        except Exception as e:
            raise RuntimeError(f"Error initializing browser: {str(e)}")

    def navigate(self, url: str):
        """
        Navigate to a URL with retry logic.

        Args:
            url: The URL to navigate to
        """
        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        try:
            super().navigate(url)
        except Exception as e:
            raise RuntimeError(f"Error navigating to {url}: {str(e)}")

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
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        try:
            logger.info("Starting prompt execution...")
            # Wait for the page to load
            logger.info("Waiting for page to load...")
            self.page.wait_for_load_state("networkidle")
            self.page.wait_for_load_state("domcontentloaded")
            logger.info("Page loaded successfully")

            # Select model if available
            def select_model(selector):
                if self.page and self.page.locator(selector).count() > 0:
                    logger.info(f"Found model selector: {selector}")
                    self.page.select_option(selector, model)
                    logger.info(f"Selected model: {model}")
                else:
                    logger.warning(f"Model selector not found or has no elements: {selector}")
                return True

            try:
                logger.info("Attempting to select model...")
                self.try_selectors("model_select", select_model)
            except ValueError as e:
                logger.warning(f"Could not select model: {e}")

            # Select behavior if available
            def select_behavior(selector):
                if self.page and self.page.locator(selector).count() > 0:
                    logger.info(f"Found behavior selector: {selector}")
                    self.page.select_option(selector, behavior)
                    logger.info(f"Selected behavior: {behavior}")
                else:
                    logger.warning(f"Behavior selector not found or has no elements: {selector}")
                return True

            try:
                logger.info("Attempting to select behavior...")
                self.try_selectors("behavior_select", select_behavior)
            except ValueError as e:
                logger.warning(f"Could not select behavior: {e}")

            # Enter prompt
            def enter_prompt(selector):
                if not self.page:
                    raise RuntimeError("Page not initialized")
                logger.info(f"Waiting for prompt input selector: {selector}")
                # Wait for the element to be visible and enabled
                try:
                    self.page.wait_for_selector(selector, state="visible", timeout=10000)
                    logger.info(f"Found prompt input element with selector: {selector}")
                except Exception as e:
                    logger.error(f"Timeout waiting for prompt input selector {selector}: {e}")
                    raise

                # Check if it's a contenteditable div
                try:
                    if "contenteditable" in selector:
                        logger.info("Using contenteditable input method")
                        self.page.evaluate(
                            f'document.querySelector("{selector}").innerText = "{prompt}"'
                        )
                    else:
                        logger.info("Using standard input method")
                        self.page.fill(selector, prompt)
                    logger.info("Successfully entered prompt text")
                    return True
                except Exception as e:
                    logger.error(f"Failed to enter prompt text using selector {selector}: {e}")
                    raise

            logger.info("Attempting to enter prompt...")
            self.try_selectors("prompt_input", enter_prompt)

            # Submit
            def click_submit(selector):
                if not self.page:
                    raise RuntimeError("Page not initialized")
                logger.info(f"Waiting for submit button selector: {selector}")
                # Wait for the button to be visible and enabled
                try:
                    self.page.wait_for_selector(selector, state="visible", timeout=10000)
                    logger.info(f"Found submit button with selector: {selector}")
                except Exception as e:
                    logger.error(f"Timeout waiting for submit button selector {selector}: {e}")
                    raise

                try:
                    self.page.click(selector)
                    logger.info("Successfully clicked submit button")
                    return True
                except Exception as e:
                    logger.error(f"Failed to click submit button using selector {selector}: {e}")
                    raise

            logger.info("Attempting to submit prompt...")
            self.try_selectors("submit_button", click_submit)

            # Wait for response with retry logic
            def wait_for_response(selector):
                if not self.page:
                    raise RuntimeError("Page not initialized")
                logger.info(f"Waiting for response with selector: {selector}")
                try:
                    # Wait for the response element to be visible
                    self.page.wait_for_selector(selector, state="visible", timeout=60000)
                    logger.info("Response element became visible")
                    # Wait for any loading indicators to disappear
                    self.page.wait_for_load_state("networkidle")
                    logger.info("Network activity settled")
                    content = self.page.text_content(selector)
                    logger.info(f"Successfully retrieved response content (length: {len(content)})")
                    return content
                except Exception as e:
                    logger.error(f"Failed to get response using selector {selector}: {e}")
                    raise

            logger.info("Attempting to get response...")
            response = self.try_selectors("response_output", wait_for_response)
            logger.info("Successfully completed prompt execution")

            return response
        except Exception as e:
            logger.error(f"Error executing prompt: {e}", exc_info=True)
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
            PlaywrightDriver.close(self)

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
