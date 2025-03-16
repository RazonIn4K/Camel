"""
Tests for the parallel processing utilities.

This module contains tests for the parallel processing utilities in the Gray Swan Arena framework.
"""

import asyncio
import os
import sys
import time
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cybersec_agents.grayswan.utils.parallel_processing import (
    RateLimiter,
    TaskManager,
    parallel_map,
    run_parallel_sync,
    run_parallel_tasks,
    run_parallel_tasks_with_kwargs,
    run_parallel_with_kwargs_sync,
)


class TestRateLimiter(unittest.TestCase):
    """Tests for the RateLimiter class."""

    async def test_rate_limiter_acquire(self):
        """Test that the rate limiter acquires tokens correctly."""
        # Create a rate limiter with 60 requests per minute (1 per second)
        rate_limiter = RateLimiter(requests_per_minute=60, burst_size=3, jitter=False)

        # Initial tokens should be equal to burst size
        self.assertEqual(rate_limiter.tokens, 3)

        # First three requests should not be delayed
        for _ in range(3):
            delay = await rate_limiter.acquire()
            self.assertEqual(delay, 0.0)

        # Fourth request should be delayed
        delay = await rate_limiter.acquire()
        self.assertGreater(delay, 0.0)

    async def test_rate_limiter_refill(self):
        """Test that the rate limiter refills tokens correctly."""
        # Create a rate limiter with 60 requests per minute (1 per second)
        rate_limiter = RateLimiter(requests_per_minute=60, burst_size=3, jitter=False)

        # Use all tokens
        for _ in range(3):
            await rate_limiter.acquire()

        # Wait for tokens to refill
        await asyncio.sleep(1.0)

        # Should have refilled about 1 token
        self.assertGreaterEqual(rate_limiter.tokens, 0.9)
        self.assertLessEqual(rate_limiter.tokens, 1.1)

    async def test_rate_limiter_jitter(self):
        """Test that the rate limiter adds jitter to delays."""
        # Create a rate limiter with jitter
        rate_limiter_with_jitter = RateLimiter(
            requests_per_minute=60, burst_size=1, jitter=True
        )

        # Use the token
        await rate_limiter_with_jitter.acquire()

        # Get delay with jitter
        delay_with_jitter = await rate_limiter_with_jitter.acquire()

        # Create a rate limiter without jitter
        rate_limiter_without_jitter = RateLimiter(
            requests_per_minute=60, burst_size=1, jitter=False
        )

        # Use the token
        await rate_limiter_without_jitter.acquire()

        # Get delay without jitter
        delay_without_jitter = await rate_limiter_without_jitter.acquire()

        # Delays should be different due to jitter
        self.assertNotEqual(delay_with_jitter, delay_without_jitter)


class TestTaskManager(unittest.TestCase):
    """Tests for the TaskManager class."""

    async def test_execute_task_success(self):
        """Test that the task manager executes tasks successfully."""
        # Create a task manager
        task_manager = TaskManager(max_workers=1, requests_per_minute=60, max_retries=0)

        # Define a simple task function
        async def task_func(x):
            return x * 2

        # Execute the task
        result, metrics = await task_manager.execute_task(task_func, 5)

        # Check the result
        self.assertEqual(result, 10)

        # Check the metrics
        self.assertTrue(metrics["success"])
        self.assertGreaterEqual(metrics["duration"], 0.0)
        self.assertEqual(metrics["retries"], 0)
        self.assertIsNone(metrics["error"])

    async def test_execute_task_failure(self):
        """Test that the task manager handles task failures correctly."""
        # Create a task manager
        task_manager = TaskManager(max_workers=1, requests_per_minute=60, max_retries=0)

        # Define a task function that raises an exception
        async def task_func(x):
            raise ValueError("Test error")

        # Execute the task and expect an exception
        with self.assertRaises(ValueError):
            await task_manager.execute_task(task_func, 5)

        # Check the metrics
        self.assertEqual(task_manager.metrics["tasks_failed"], 1)

    async def test_execute_task_retry(self):
        """Test that the task manager retries failed tasks."""
        # Create a task manager with retries
        task_manager = TaskManager(
            max_workers=1, requests_per_minute=60, max_retries=2, retry_delay=0.1
        )

        # Define a task function that fails on first attempt but succeeds on retry
        attempt: list[Any] = [0]

        async def task_func(x):
            attempt[0] += 1
            if attempt[0] == 1:
                raise ValueError("Test error")
            return x * 2

        # Execute the task
        result, metrics = await task_manager.execute_task(task_func, 5)

        # Check the result
        self.assertEqual(result, 10)

        # Check the metrics
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["retries"], 1)
        self.assertIsNone(metrics["error"])

        # Check the task manager metrics
        self.assertEqual(task_manager.metrics["retries"], 1)

    async def test_map(self):
        """Test that the task manager maps functions over items correctly."""
        # Create a task manager
        task_manager = TaskManager(max_workers=2, requests_per_minute=60, max_retries=0)

        # Define a simple task function
        async def task_func(x):
            await asyncio.sleep(0.1)
            return x * 2

        # Map the function over items
        items: list[Any] = [1, 2, 3, 4, 5]
        results_with_metrics = await task_manager.map(task_func, items, batch_size=2)

        # Extract results
        results: list[Any] = [r for r, _ in results_with_metrics]

        # Check the results
        self.assertEqual(results, [2, 4, 6, 8, 10])

        # Check the task manager metrics
        self.assertEqual(task_manager.metrics["tasks_submitted"], 5)
        self.assertEqual(task_manager.metrics["tasks_completed"], 5)
        self.assertEqual(task_manager.metrics["tasks_failed"], 0)

    async def test_map_with_kwargs(self):
        """Test that the task manager maps functions with kwargs correctly."""
        # Create a task manager
        task_manager = TaskManager(max_workers=2, requests_per_minute=60, max_retries=0)

        # Define a task function with kwargs
        async def task_func(x, multiplier=1, add=0):
            await asyncio.sleep(0.1)
            return x * multiplier + add

        # Create items with kwargs
        items_with_kwargs: list[Any] = [
            {"x": 1, "multiplier": 2, "add": 1},
            {"x": 2, "multiplier": 3, "add": 2},
            {"x": 3, "multiplier": 4, "add": 3},
        ]

        # Map the function over items with kwargs
        results_with_metrics = await task_manager.map_with_kwargs(
            task_func, items_with_kwargs
        )

        # Extract results
        results: list[Any] = [r for r, _ in results_with_metrics]

        # Check the results
        self.assertEqual(results, [3, 8, 15])


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for the convenience functions."""

    async def test_run_parallel_tasks(self):
        """Test that run_parallel_tasks works correctly."""

        # Define a simple task function
        async def task_func(x):
            await asyncio.sleep(0.1)
            return x * 2

        # Run tasks in parallel
        items: list[Any] = [1, 2, 3, 4, 5]
        results, metrics = await run_parallel_tasks(task_func, items, max_workers=2)

        # Check the results
        self.assertEqual(results, [2, 4, 6, 8, 10])

        # Check the metrics
        self.assertEqual(metrics["tasks_submitted"], 5)
        self.assertEqual(metrics["tasks_completed"], 5)
        self.assertEqual(metrics["tasks_failed"], 0)

    async def test_run_parallel_tasks_with_kwargs(self):
        """Test that run_parallel_tasks_with_kwargs works correctly."""

        # Define a task function with kwargs
        async def task_func(x, multiplier=1, add=0):
            await asyncio.sleep(0.1)
            return x * multiplier + add

        # Create items with kwargs
        items_with_kwargs: list[Any] = [
            {"x": 1, "multiplier": 2, "add": 1},
            {"x": 2, "multiplier": 3, "add": 2},
            {"x": 3, "multiplier": 4, "add": 3},
        ]

        # Run tasks in parallel with kwargs
        results, metrics = await run_parallel_tasks_with_kwargs(
            task_func, items_with_kwargs, max_workers=2
        )

        # Check the results
        self.assertEqual(results, [3, 8, 15])

        # Check the metrics
        self.assertEqual(metrics["tasks_submitted"], 3)
        self.assertEqual(metrics["tasks_completed"], 3)
        self.assertEqual(metrics["tasks_failed"], 0)

    def test_run_parallel_sync(self):
        """Test that run_parallel_sync works correctly."""

        # Define a simple task function
        def task_func(x):
            time.sleep(0.1)
            return x * 2

        # Run tasks in parallel
        items: list[Any] = [1, 2, 3, 4, 5]
        results, metrics = run_parallel_sync(task_func, items, max_workers=2)

        # Check the results
        self.assertEqual(results, [2, 4, 6, 8, 10])

        # Check the metrics
        self.assertEqual(metrics["tasks_submitted"], 5)
        self.assertEqual(metrics["tasks_completed"], 5)
        self.assertEqual(metrics["tasks_failed"], 0)

    def test_run_parallel_with_kwargs_sync(self):
        """Test that run_parallel_with_kwargs_sync works correctly."""

        # Define a task function with kwargs
        def task_func(x, multiplier=1, add=0):
            time.sleep(0.1)
            return x * multiplier + add

        # Create items with kwargs
        items_with_kwargs: list[Any] = [
            {"x": 1, "multiplier": 2, "add": 1},
            {"x": 2, "multiplier": 3, "add": 2},
            {"x": 3, "multiplier": 4, "add": 3},
        ]

        # Run tasks in parallel with kwargs
        results, metrics = run_parallel_with_kwargs_sync(
            task_func, items_with_kwargs, max_workers=2
        )

        # Check the results
        self.assertEqual(results, [3, 8, 15])

        # Check the metrics
        self.assertEqual(metrics["tasks_submitted"], 3)
        self.assertEqual(metrics["tasks_completed"], 3)
        self.assertEqual(metrics["tasks_failed"], 0)

    def test_parallel_map_decorator(self):
        """Test that the parallel_map decorator works correctly."""

        # Define a function with the parallel_map decorator
        @parallel_map(max_workers=2)
        def process_data(item):
            time.sleep(0.1)
            return item * 2

        # Process items using the decorated function
        items: list[Any] = [1, 2, 3, 4, 5]
        results, metrics = process_data(items)

        # Check the results
        self.assertEqual(results, [2, 4, 6, 8, 10])

        # Check the metrics
        self.assertEqual(metrics["tasks_submitted"], 5)
        self.assertEqual(metrics["tasks_completed"], 5)
        self.assertEqual(metrics["tasks_failed"], 0)


class TestGraySwanIntegration(unittest.TestCase):
    """Tests for the Gray Swan Arena integration."""

    @patch("cybersec_agents.grayswan.agents.recon_agent.ReconAgent")
    async def test_run_parallel_reconnaissance(self, mock_recon_agent):
        """Test that run_parallel_reconnaissance works correctly."""
        from cybersec_agents.grayswan.utils.parallel_processing import (
            run_parallel_reconnaissance,
        )

        # Mock the ReconAgent
        mock_instance = MagicMock()
        mock_recon_agent.return_value = mock_instance

        # Mock the web search and Discord search methods
        mock_instance.run_web_search.return_value = {"web": "results"}
        mock_instance.run_discord_search.return_value = {"discord": "results"}

        # Mock the generate_report and save_report methods
        mock_instance.generate_report.return_value = {"report": "data"}
        mock_instance.save_report.return_value = "report_path"

        # Run parallel reconnaissance
        result: Any = await run_parallel_reconnaissance(
            target_model="GPT-4",
            target_behavior="Generate harmful content",
            output_dir="./output",
            model_name="gpt-4",
            max_workers=2,
        )

        # Check that the methods were called
        mock_instance.run_web_search.assert_called_once()
        mock_instance.run_discord_search.assert_called_once()
        mock_instance.generate_report.assert_called_once()
        mock_instance.save_report.assert_called_once()

        # Check the result
        self.assertEqual(result["report"], {"report": "data"})
        self.assertEqual(result["path"], "report_path")
        self.assertEqual(result["search_results"]["web"], {"web": "results"})
        self.assertEqual(result["search_results"]["discord"], {"discord": "results"})

    @patch("cybersec_agents.grayswan.agents.prompt_engineer_agent.PromptEngineerAgent")
    async def test_run_parallel_prompt_engineering(self, mock_prompt_agent):
        """Test that run_parallel_prompt_engineering works correctly."""
        from cybersec_agents.grayswan.utils.parallel_processing import (
            run_parallel_prompt_engineering,
        )

        # Mock the PromptEngineerAgent
        mock_instance = MagicMock()
        mock_prompt_agent.return_value = mock_instance

        # Mock the get_prompt_types method
        mock_instance.get_prompt_types.return_value = ["direct", "indirect"]

        # Mock the generate_prompts_of_type method
        mock_instance.generate_prompts_of_type.side_effect = lambda **kwargs: [
            "prompt1",
            "prompt2",
        ]

        # Mock the save_prompts method
        mock_instance.save_prompts.return_value = "prompts_path"

        # Run parallel prompt engineering
        result: Any = await run_parallel_prompt_engineering(
            target_model="GPT-4",
            target_behavior="Generate harmful content",
            recon_report={"report": "data"},
            output_dir="./output",
            model_name="gpt-4",
            num_prompts=4,
            max_workers=2,
        )

        # Check that the methods were called
        mock_instance.get_prompt_types.assert_called_once()
        self.assertEqual(mock_instance.generate_prompts_of_type.call_count, 2)
        mock_instance.save_prompts.assert_called_once()

        # Check the result
        self.assertEqual(len(result["prompts"]), 4)
        self.assertEqual(result["path"], "prompts_path")

    @patch(
        "cybersec_agents.grayswan.agents.exploit_delivery_agent.ExploitDeliveryAgent"
    )
    async def test_run_parallel_exploits(self, mock_exploit_agent):
        """Test that run_parallel_exploits works correctly."""
        from cybersec_agents.grayswan.utils.parallel_processing import (
            run_parallel_exploits,
        )

        # Mock the ExploitDeliveryAgent
        mock_instance = MagicMock()
        mock_exploit_agent.return_value = mock_instance

        # Mock the _execute_via_api method
        mock_instance._execute_via_api.side_effect = (
            lambda prompt, model: f"Response to {prompt}"
        )

        # Mock the _analyze_response method
        mock_instance._analyze_response.side_effect = lambda response, behavior: (
            True,
            "Success",
        )

        # Mock the save_results method
        mock_instance.save_results.return_value = "results_path"

        # Run parallel exploits
        prompts: list[Any] = ["prompt1", "prompt2", "prompt3"]
        result: Any = await run_parallel_exploits(
            prompts=prompts,
            target_model="GPT-4",
            target_behavior="Generate harmful content",
            output_dir="./output",
            model_name="gpt-4",
            method="api",
            max_workers=2,
        )

        # Check that the methods were called
        self.assertEqual(mock_instance._execute_via_api.call_count, 3)
        self.assertEqual(mock_instance._analyze_response.call_count, 3)
        mock_instance.save_results.assert_called_once()

        # Check the result
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(result["path"], "results_path")

        # Check that all results have the expected fields
        for i, res in enumerate(result["results"]):
            self.assertEqual(res["prompt"], prompts[i])
            self.assertEqual(res["target_model"], "GPT-4")
            self.assertEqual(res["target_behavior"], "Generate harmful content")
            self.assertEqual(res["method"], "api")
            self.assertTrue(res["success"])
            self.assertEqual(res["reason"], "Success")
            self.assertEqual(res["response"], f"Response to {prompts[i]}")
            self.assertIsNone(res["error"])

    @patch("cybersec_agents.grayswan.agents.evaluation_agent.EvaluationAgent")
    async def test_run_parallel_evaluation(self, mock_eval_agent):
        """Test that run_parallel_evaluation works correctly."""
        from cybersec_agents.grayswan.utils.parallel_processing import (
            run_parallel_evaluation,
        )

        # Mock the EvaluationAgent
        mock_instance = MagicMock()
        mock_eval_agent.return_value = mock_instance

        # Mock the evaluate_results method
        mock_instance.evaluate_results.return_value = {"evaluation": "data"}

        # Mock the create_visualizations method
        mock_instance.create_visualizations.return_value = {"vis": "paths"}

        # Mock the create_advanced_visualizations method
        mock_instance.create_advanced_visualizations.return_value = {"adv_vis": "paths"}

        # Mock the generate_summary method
        mock_instance.generate_summary.return_value = {"summary": "data"}

        # Mock the save_evaluation and save_summary methods
        mock_instance.save_evaluation.return_value = "eval_path"
        mock_instance.save_summary.return_value = "summary_path"

        # Run parallel evaluation
        exploit_results = [{"result": "data"}]
        result: Any = await run_parallel_evaluation(
            exploit_results=exploit_results,
            target_model="GPT-4",
            target_behavior="Generate harmful content",
            output_dir="./output",
            model_name="gpt-4",
            include_visualizations=True,
            include_advanced_visualizations=True,
            max_workers=2,
        )

        # Check that the methods were called
        mock_instance.evaluate_results.assert_called_once()
        mock_instance.create_visualizations.assert_called_once()
        mock_instance.create_advanced_visualizations.assert_called_once()
        mock_instance.generate_summary.assert_called_once()
        mock_instance.save_evaluation.assert_called_once()
        mock_instance.save_summary.assert_called_once()

        # Check the result
        self.assertEqual(result["evaluation"], {"evaluation": "data"})
        self.assertEqual(result["summary"], {"summary": "data"})
        self.assertEqual(result["paths"]["evaluation"], "eval_path")
        self.assertEqual(result["paths"]["summary"], "summary_path")
        self.assertEqual(result["visualizations"], {"vis": "paths", "adv_vis": "paths"})


if __name__ == "__main__":
    # Run the async tests
    loop = asyncio.get_event_loop()
    result: Any = unittest.main(exit=False)
    loop.close()

    # Exit with the test result
    sys.exit(not result.result.wasSuccessful())
