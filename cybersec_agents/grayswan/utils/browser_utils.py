"""Browser automation utilities for Gray Swan Arena."""

from enum import Enum
from typing import Dict

from .logging_utils import setup_logging

# Set up logger
logger = setup_logging("BrowserUtils")


class BrowserMethod(Enum):
    """Supported browser automation methods."""

    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"


class BrowserDriver:
    """Base interface for browser drivers."""

    def __init__(self, headless: bool = True):
        """Initialize the browser driver.

        Args:
            headless: Whether to run the browser in headless mode
        """
        self.headless = headless
        self.selectors = {
            "model_select": "#model-select",
            "behavior_select": "#behavior-select",
            "prompt_input": "#prompt-textarea",
            "submit_button": "#submit-button",
            "response_output": "#response-container",
            "success_indicator": ".success-indicator",
        }

    def initialize(self):
        """Initialize the browser session."""
        raise NotImplementedError("Subclasses must implement initialize()")

    def navigate(self, url: str):
        """Navigate to a URL.

        Args:
            url: The URL to navigate to
        """
        raise NotImplementedError("Subclasses must implement navigate()")

    def execute_prompt(self, prompt: str, model: str, behavior: str) -> str:
        """Execute a prompt against a model.

        Args:
            prompt: The prompt to execute
            model: The model to execute against
            behavior: The behavior to test for

        Returns:
            The model's response
        """
        raise NotImplementedError("Subclasses must implement execute_prompt()")

    def close(self):
        """Close the browser session."""
        raise NotImplementedError("Subclasses must implement close()")


class PlaywrightDriver(BrowserDriver):
    """Browser driver using Playwright."""

    def __init__(self, headless: bool = True):
        """Initialize the Playwright driver.

        Args:
            headless: Whether to run the browser in headless mode
        """
        super().__init__(headless)
        self.playwright_available = self._check_playwright_available()
        self.browser = None
        self.page = None

    def _check_playwright_available(self) -> bool:
        """Check if Playwright is available.

        Returns:
            True if Playwright is available, False otherwise
        """
        try:
            from playwright.sync_api import sync_playwright

            logger.info("Playwright is available")
            return True
        except ImportError:
            logger.warning(
                "Playwright not installed. Install with 'pip install playwright' and run 'playwright install'"
            )
            return False

    def initialize(self):
        """Initialize the Playwright browser session."""
        if not self.playwright_available:
            raise ImportError("Playwright not installed")

        try:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self.browser = self._playwright.chromium.launch(headless=self.headless)
            self.page = self.browser.new_page()
            logger.info("Playwright browser session initialized")
        except Exception as e:
            logger.error(f"Error initializing Playwright browser: {e}")
            raise

    def navigate(self, url: str):
        """Navigate to a URL.

        Args:
            url: The URL to navigate to
        """
        if not self.page:
            raise ValueError("Browser not initialized. Call initialize() first.")

        try:
            self.page.goto(url)
            logger.info(f"Navigated to {url}")
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            raise

    def execute_prompt(self, prompt: str, model: str, behavior: str) -> str:
        """Execute a prompt against a model.

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
            try:
                if self.page.locator(self.selectors["model_select"]).count() > 0:
                    self.page.select_option(self.selectors["model_select"], model)
                    logger.info(f"Selected model: {model}")
            except Exception as e:
                logger.warning(f"Could not select model: {e}")

            # Select behavior if available
            try:
                if self.page.locator(self.selectors["behavior_select"]).count() > 0:
                    self.page.select_option(self.selectors["behavior_select"], behavior)
                    logger.info(f"Selected behavior: {behavior}")
            except Exception as e:
                logger.warning(f"Could not select behavior: {e}")

            # Enter prompt
            self.page.fill(self.selectors["prompt_input"], prompt)
            logger.info(f"Entered prompt")

            # Submit
            self.page.click(self.selectors["submit_button"])
            logger.info(f"Submitted prompt")

            # Wait for response
            self.page.wait_for_selector(
                self.selectors["response_output"], state="visible", timeout=60000
            )

            # Get response
            response = self.page.text_content(self.selectors["response_output"])
            logger.info(f"Received response")

            return response
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            return f"Error: {str(e)}"

    def close(self):
        """Close the Playwright browser session."""
        if self.browser:
            self.browser.close()
        if hasattr(self, "_playwright"):
            self._playwright.stop()
        logger.info("Playwright browser session closed")


class SeleniumDriver(BrowserDriver):
    """Browser driver using Selenium."""

    def __init__(self, headless: bool = True):
        """Initialize the Selenium driver.

        Args:
            headless: Whether to run the browser in headless mode
        """
        super().__init__(headless)
        self.selenium_available = self._check_selenium_available()
        self.driver = None
        self.selenium = {}

    def _check_selenium_available(self) -> bool:
        """Check if Selenium is available.

        Returns:
            True if Selenium is available, False otherwise
        """
        try:
            from selenium import webdriver
            from selenium.common.exceptions import (
                NoSuchElementException,
                TimeoutException,
            )
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.support.ui import Select, WebDriverWait

            self.selenium = {
                "webdriver": webdriver,
                "By": By,
                "WebDriverWait": WebDriverWait,
                "Select": Select,
                "EC": EC,
                "TimeoutException": TimeoutException,
                "NoSuchElementException": NoSuchElementException,
            }

            logger.info("Selenium is available")
            return True
        except ImportError:
            logger.warning(
                "Selenium not installed. Install with 'pip install selenium webdriver-manager'"
            )
            return False

    def initialize(self):
        """Initialize the Selenium browser session."""
        if not self.selenium_available:
            raise ImportError("Selenium not installed")

        try:
            webdriver = self.selenium["webdriver"]

            # Use Chrome by default
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service as ChromeService
            from webdriver_manager.chrome import ChromeDriverManager

            options = Options()
            if self.headless:
                options.add_argument("--headless")
                options.add_argument("--disable-gpu")

            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            service = ChromeService(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)

            logger.info("Selenium browser session initialized")
        except Exception as e:
            logger.error(f"Error initializing Selenium browser: {e}")
            raise

    def navigate(self, url: str):
        """Navigate to a URL.

        Args:
            url: The URL to navigate to
        """
        if not self.driver:
            raise ValueError("Browser not initialized. Call initialize() first.")

        try:
            self.driver.get(url)
            logger.info(f"Navigated to {url}")
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            raise

    def execute_prompt(self, prompt: str, model: str, behavior: str) -> str:
        """Execute a prompt against a model.

        Args:
            prompt: The prompt to execute
            model: The model to execute against
            behavior: The behavior to test for

        Returns:
            The model's response
        """
        if not self.driver:
            raise ValueError("Browser not initialized. Call initialize() first.")

        try:
            By = self.selenium["By"]
            WebDriverWait = self.selenium["WebDriverWait"]
            Select = self.selenium["Select"]
            EC = self.selenium["EC"]

            # Select model if available
            try:
                model_select = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, self.selectors["model_select"])
                    )
                )
                Select(model_select).select_by_visible_text(model)
                logger.info(f"Selected model: {model}")
            except Exception as e:
                logger.warning(f"Could not select model: {e}")

            # Select behavior if available
            try:
                behavior_select = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, self.selectors["behavior_select"])
                    )
                )
                Select(behavior_select).select_by_visible_text(behavior)
                logger.info(f"Selected behavior: {behavior}")
            except Exception as e:
                logger.warning(f"Could not select behavior: {e}")

            # Enter prompt
            prompt_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.selectors["prompt_input"])
                )
            )
            prompt_input.clear()
            prompt_input.send_keys(prompt)
            logger.info(f"Entered prompt")

            # Submit
            submit_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, self.selectors["submit_button"])
                )
            )
            submit_button.click()
            logger.info(f"Submitted prompt")

            # Wait for response
            response_element = WebDriverWait(self.driver, 60).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.selectors["response_output"])
                )
            )

            # Get response
            response = response_element.text
            logger.info(f"Received response")

            return response
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            return f"Error: {str(e)}"

    def close(self):
        """Close the Selenium browser session."""
        if self.driver:
            self.driver.quit()
        logger.info("Selenium browser session closed")


class BrowserAutomationFactory:
    """Factory for creating browser drivers."""

    @staticmethod
    def create_driver(method: str, headless: bool = True) -> BrowserDriver:
        """Create a browser driver.

        Args:
            method: The browser automation method to use ("playwright" or "selenium")
            headless: Whether to run the browser in headless mode

        Returns:
            A BrowserDriver instance
        """
        method = method.lower()

        if method == "playwright":
            driver = PlaywrightDriver(headless=headless)
        elif method == "selenium":
            driver = SeleniumDriver(headless=headless)
        else:
            raise ValueError(f"Unsupported browser automation method: {method}")

        return driver


def is_browser_automation_available() -> Dict[str, bool]:
    """Check which browser automation methods are available.

    Returns:
        A dictionary with keys "playwright" and "selenium" and boolean values
    """
    result = {"playwright": False, "selenium": False}

    # Check Playwright
    try:
        from playwright.sync_api import sync_playwright

        result["playwright"] = True
    except ImportError:
        result["playwright"] = False

    # Check Selenium
    try:
        from selenium import webdriver

        result["selenium"] = True
    except ImportError:
        result["selenium"] = False

    return result
