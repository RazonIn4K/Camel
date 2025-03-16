"""
Tests for the retry functionality in Gray Swan Arena.

This module provides tests for the retry functionality, including the RetryManager
and its integration with agent methods.
"""

import unittest
import time
import logging
from unittest.mock import MagicMock, patch

from ..utils.retry_utils import (
    RetryStrategy, 
    FixedDelayRetryStrategy, 
    ExponentialBackoffRetryStrategy, 
    CircuitBreakerRetryStrategy
)
from ..utils.retry_manager import RetryManager
from ..agents.evaluation_agent import EvaluationAgent

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestRetryManager(unittest.TestCase):
    """Test case for the RetryManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.retry_strategy = FixedDelayRetryStrategy(max_retries=3, delay=0.01)
        self.retry_manager = RetryManager(
            retry_strategy=self.retry_strategy,
            operation_name="test_operation",
            agent_id="test_agent"
        )

    def test_retry_success_after_failures(self):
        """Test that an operation succeeds after multiple failures."""
        # Mock function that fails twice and then succeeds
        counter = {"attempts": 0}
        
        def test_function():
            counter["attempts"] += 1
            if counter["attempts"] < 3:
                raise ValueError("Test failure")
            return "success"
        
        # Call the function with retry
        result = self.retry_manager.retry(test_function)
        
        # Verify that the function was called multiple times and eventually succeeded
        self.assertEqual(result, "success")
        self.assertEqual(counter["attempts"], 3)
    
    def test_retry_failure_after_max_retries(self):
        """Test that an operation fails after max retries."""
        # Mock function that always fails
        counter = {"attempts": 0}
        
        def test_function():
            counter["attempts"] += 1
            raise ValueError("Test failure")
        
        # Call the function with retry and verify it raises an exception
        with self.assertRaises(ValueError):
            self.retry_manager.retry(test_function)
        
        # Verify that the function was called the expected number of times
        self.assertEqual(counter["attempts"], self.retry_strategy.max_retries + 1)
    
    def test_retry_context_success_after_failures(self):
        """Test that a code block in a retry context succeeds after multiple failures."""
        # Counter to track attempts
        counter = {"attempts": 0, "result": None}
        
        # Execute code in retry context
        try:
            with self.retry_manager.retry_context("test_context"):
                counter["attempts"] += 1
                if counter["attempts"] < 3:
                    raise ValueError("Test failure")
                counter["result"] = "success"
        except Exception:
            pass
        
        # Verify that the block was executed multiple times and eventually succeeded
        self.assertEqual(counter["result"], "success")
        self.assertEqual(counter["attempts"], 3)
    
    def test_retry_context_failure_after_max_retries(self):
        """Test that a code block in a retry context fails after max retries."""
        # Counter to track attempts
        counter = {"attempts": 0}
        
        # Execute code in retry context and verify it raises an exception
        with self.assertRaises(ValueError):
            with self.retry_manager.retry_context("test_context"):
                counter["attempts"] += 1
                raise ValueError("Test failure")
        
        # Verify that the block was executed the expected number of times
        self.assertEqual(counter["attempts"], self.retry_strategy.max_retries + 1)


class TestRetryWithEvaluationAgent(unittest.TestCase):
    """Test case for retry functionality with EvaluationAgent."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a retry strategy for the test
        self.retry_strategy = FixedDelayRetryStrategy(max_retries=3, delay=0.01)
        
        # Create a retry manager for the test
        self.retry_manager = RetryManager(
            retry_strategy=self.retry_strategy,
            operation_name="test_operation",
            agent_id="test_agent"
        )
        
        # Create an evaluation agent with a fixed delay retry strategy
        self.agent = EvaluationAgent(
            output_dir="./test_evaluations",
            model_name="gpt-4",
            max_retries=3,
            initial_retry_delay=0.01
        )

    @patch('cybersec_agents.grayswan.agents.evaluation_agent.get_chat_agent')
    def test_agent_evaluate_results_with_retry(self, mock_get_chat_agent):
        """Test that the evaluate_results method retries on failure."""
        # Set up the mock
        mock_agent = MagicMock()
        mock_step_response = MagicMock()
        
        # Configure the mock to fail twice and then succeed
        mock_step_responses = [
            # First call - fails
            Exception("Test failure 1"),
            # Second call - fails
            Exception("Test failure 2"),
            # Third call - succeeds
            mock_step_response
        ]
        
        # Configure the mock_agent.step method to raise exceptions or return response
        mock_agent.step.side_effect = mock_step_responses
        
        # Configure mock_get_chat_agent to return our mock_agent
        mock_get_chat_agent.return_value = mock_agent
        
        # Configure mock_step_response to have required attributes
        mock_step_response.__dict__ = {'content': "Test response with 25% success rate"}
        
        # Set up test data
        results = [
            {"prompt": "test prompt", "response": "test response", "success": True},
            {"prompt": "test prompt 2", "response": "test response 2", "success": False}
        ]
        
        # Call the method
        with self.retry_manager.retry_context("test_evaluate_results"):
            evaluation = self.agent.evaluate_results(results, "test-model", "some-behavior")
        
        # Verify that get_chat_agent was called
        mock_get_chat_agent.assert_called()
        
        # Verify that the agent.step method was called multiple times
        self.assertEqual(mock_agent.step.call_count, 3)
        
        # Verify that the evaluation contains the expected data
        self.assertEqual(evaluation["target_model"], "test-model")
        self.assertEqual(evaluation["target_behavior"], "some-behavior")
        self.assertEqual(evaluation["success_rate"], 0.25)


if __name__ == '__main__':
    unittest.main() 