from typing import Any, Dict, List, Optional, Tuple, Union
"""
Tests for the enhanced browser automation utilities.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cybersec_agents.grayswan.utils.enhanced_browser_utils import (
    EnhancedPlaywrightDriver,
    EnhancedBrowserAutomationFactory
)


class TestEnhancedBrowserAutomation(unittest.TestCase):
    """Test cases for enhanced browser automation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock for the playwright module
        self.playwright_patcher = patch('cybersec_agents.grayswan.utils.browser_utils.sync_playwright')
        self.mock_playwright = self.playwright_patcher.start()
        
        # Set up mock browser, page, etc.
        self.mock_browser = MagicMock()
        self.mock_page = MagicMock()
        self.mock_browser.new_page.return_value = self.mock_page
        
        self.mock_playwright_instance = MagicMock()
        self.mock_playwright_instance.chromium.launch.return_value = self.mock_browser
        
        self.mock_playwright.return_value.start.return_value = self.mock_playwright_instance

    def tearDown(self):
        """Tear down test fixtures."""
        self.playwright_patcher.stop()

    def test_enhanced_playwright_driver_initialization(self):
        """Test initialization of EnhancedPlaywrightDriver."""
        driver = EnhancedPlaywrightDriver(headless=True, retry_attempts=3, retry_delay=1.0)
        
        # Check that the driver was initialized correctly
        self.assertTrue(driver.headless)
        self.assertEqual(driver.retry_attempts, 3)
        self.assertEqual(driver.retry_delay, 1.0)
        
        # Check that the selector alternatives were set up
        self.assertIn("model_select", driver.selector_alternatives)
        self.assertIn("prompt_input", driver.selector_alternatives)
        self.assertIn("submit_button", driver.selector_alternatives)
        self.assertIn("response_output", driver.selector_alternatives)
        
        # Check that metrics were initialized
        self.assertEqual(driver.metrics["selector_fallbacks"], 0)
        self.assertEqual(driver.metrics["retry_attempts"], 0)
        self.assertEqual(driver.metrics["successful_recoveries"], 0)
        self.assertEqual(driver.metrics["failed_recoveries"], 0)

    def test_enhanced_playwright_driver_initialize(self):
        """Test initialize method of EnhancedPlaywrightDriver."""
        driver = EnhancedPlaywrightDriver(headless=True)
        driver.initialize()
        
        # Check that the browser was launched with the correct parameters
        self.mock_playwright.return_value.start.assert_called_once()
        self.mock_playwright_instance.chromium.launch.assert_called_once_with(headless=True)
        self.mock_browser.new_page.assert_called_once()
        
        # Check that the driver's browser and page attributes were set
        self.assertEqual(driver.browser, self.mock_browser)
        self.assertEqual(driver.page, self.mock_page)

    def test_enhanced_playwright_driver_navigate(self):
        """Test navigate method of EnhancedPlaywrightDriver."""
        driver = EnhancedPlaywrightDriver(headless=True)
        driver.initialize()
        
        # Navigate to a URL
        url = "https://example.com"
        driver.navigate(url)
        
        # Check that the page's goto method was called with the correct URL
        self.mock_page.goto.assert_called_once_with(url)

    def test_enhanced_playwright_driver_try_selectors(self):
        """Test try_selectors method of EnhancedPlaywrightDriver."""
        driver = EnhancedPlaywrightDriver(headless=True)
        driver.initialize()
        
        # Set up mock locator
        mock_locator = MagicMock()
        mock_locator.count.return_value = 1
        self.mock_page.locator.return_value = mock_locator
        
        # Define a test action function
        def test_action(selector):
            return f"Action performed on {selector}"
        
        # Test with a selector that works on the first try
        result: Any = driver.try_selectors("model_select", test_action)
        
        # Check that the result is correct
        self.assertEqual(result, "Action performed on #model-select")
        
        # Check that the working selector was remembered
        self.assertEqual(driver.working_selectors["model_select"], "#model-select")
        
        # Test with a selector that fails on the first try but works on the second
        self.mock_page.locator.side_effect = [
            MagicMock(count=lambda: 0),  # First selector fails
            MagicMock(count=lambda: 1)   # Second selector works
        ]
        
        # Define a test action function that raises an exception for the first selector
        def test_action_with_exception(selector):
            if selector == "#model-select":
                raise ValueError("Selector not found")
            return f"Action performed on {selector}"
        
        # Test with a selector that fails on the first try but works on the second
        result: Any = driver.try_selectors("model_select", test_action_with_exception)
        
        # Check that the result is correct and metrics were updated
        self.assertTrue(result.startswith("Action performed on"))
        self.assertEqual(driver.metrics["selector_fallbacks"], 1)
        self.assertEqual(driver.metrics["successful_recoveries"], 1)

    def test_enhanced_playwright_driver_execute_prompt(self):
        """Test execute_prompt method of EnhancedPlaywrightDriver."""
        driver = EnhancedPlaywrightDriver(headless=True)
        driver.initialize()
        
        # Set up mock locators and responses
        mock_model_locator = MagicMock()
        mock_model_locator.count.return_value = 1
        
        mock_prompt_locator = MagicMock()
        mock_prompt_locator.count.return_value = 1
        
        mock_response_locator = MagicMock()
        mock_response_locator.count.return_value = 1
        
        self.mock_page.locator.side_effect = [
            mock_model_locator,   # For model select
            mock_prompt_locator,  # For prompt input
            mock_response_locator # For response output
        ]
        
        # Set up mock text_content
        self.mock_page.text_content.return_value = "Test response"
        
        # Execute a prompt
        response = driver.execute_prompt(
            prompt="Test prompt",
            model="gpt-4",
            behavior="test behavior"
        )
        
        # Check that the response is correct
        self.assertEqual(response, "Test response")
        
        # Check that the page methods were called correctly
        self.mock_page.fill.assert_called_once()
        self.mock_page.click.assert_called_once()
        self.mock_page.wait_for_selector.assert_called_once()
        self.mock_page.text_content.assert_called_once()

    def test_enhanced_browser_automation_factory(self):
        """Test EnhancedBrowserAutomationFactory."""
        # Test creating a Playwright driver
        driver = EnhancedBrowserAutomationFactory.create_driver(
            method="playwright",
            headless=True,
            retry_attempts=3,
            retry_delay=1.0
        )
        
        # Check that the driver is an instance of EnhancedPlaywrightDriver
        self.assertIsInstance(driver, EnhancedPlaywrightDriver)
        
        # Check that the driver was initialized with the correct parameters
        self.assertTrue(driver.headless)
        self.assertEqual(driver.retry_attempts, 3)
        self.assertEqual(driver.retry_delay, 1.0)
        
        # Test with an invalid method
        with self.assertRaises(ValueError):
            EnhancedBrowserAutomationFactory.create_driver(
                method="invalid",
                headless=True
            )


if __name__ == '__main__':
    unittest.main()