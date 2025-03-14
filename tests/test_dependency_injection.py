"""
Tests for the dependency injection implementation in Gray Swan Arena.

This module contains tests for the dependency injection container and pipeline.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from cybersec_agents.grayswan.container import GraySwanContainer, GraySwanContainerFactory
from cybersec_agents.grayswan.main_di import GraySwanPipeline


class TestGraySwanContainer(unittest.TestCase):
    """Tests for the GraySwanContainer class."""
    
    def test_default_configuration(self):
        """Test that the container has default configuration values."""
        container = GraySwanContainer()
        
        # Check that default configuration values are set
        self.assertEqual(container.config.output_dir(), './output')
        self.assertEqual(container.config.agents.recon.model_name(), 'gpt-4')
        self.assertEqual(container.config.agents.recon.output_dir(), './output/recon_reports')
        self.assertEqual(container.config.browser.method(), 'playwright')
        self.assertEqual(container.config.browser.headless(), True)
        self.assertEqual(container.config.visualization.dpi(), 300)
    
    def test_custom_configuration(self):
        """Test that the container can be configured with custom values."""
        container = GraySwanContainer()
        
        # Override configuration values
        container.config.output_dir.override('./custom_output')
        container.config.agents.recon.model_name.override('gpt-3.5-turbo')
        container.config.browser.headless.override(False)
        
        # Check that configuration values are overridden
        self.assertEqual(container.config.output_dir(), './custom_output')
        self.assertEqual(container.config.agents.recon.model_name(), 'gpt-3.5-turbo')
        self.assertEqual(container.config.browser.headless(), False)
        
        # Check that other values are still default
        self.assertEqual(container.config.agents.recon.output_dir(), './output/recon_reports')
        self.assertEqual(container.config.browser.method(), 'playwright')
        self.assertEqual(container.config.visualization.dpi(), 300)
    
    def test_configuration_from_dict(self):
        """Test that the container can be configured from a dictionary."""
        config_dict = {
            'output_dir': './dict_output',
            'agents': {
                'recon': {
                    'model_name': 'claude-2',
                    'output_dir': './dict_output/recon_reports',
                },
            },
            'browser': {
                'method': 'selenium',
                'headless': False,
            },
        }
        
        container = GraySwanContainerFactory.create_container(config_dict)
        
        # Check that configuration values are set from dictionary
        self.assertEqual(container.config.output_dir(), './dict_output')
        self.assertEqual(container.config.agents.recon.model_name(), 'claude-2')
        self.assertEqual(container.config.agents.recon.output_dir(), './dict_output/recon_reports')
        self.assertEqual(container.config.browser.method(), 'selenium')
        self.assertEqual(container.config.browser.headless(), False)
        
        # Check that other values are still default
        self.assertEqual(container.config.visualization.dpi(), 300)


class TestGraySwanPipeline(unittest.TestCase):
    """Tests for the GraySwanPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a container with mocked dependencies
        self.container = GraySwanContainer()
        
        # Mock the logger
        self.mock_logger = MagicMock()
        self.container.logger = lambda: self.mock_logger
        
        # Mock the agents
        self.mock_recon_agent = MagicMock()
        self.container.recon_agent = lambda: self.mock_recon_agent
        
        self.mock_prompt_agent = MagicMock()
        self.container.prompt_engineer_agent = lambda: self.mock_prompt_agent
        
        self.mock_exploit_agent = MagicMock()
        self.container.exploit_delivery_agent = lambda: self.mock_exploit_agent
        
        self.mock_eval_agent = MagicMock()
        self.container.evaluation_agent = lambda: self.mock_eval_agent
        
        # Create pipeline with mocked container
        self.pipeline = GraySwanPipeline(self.container)
    
    def test_run_reconnaissance(self):
        """Test the run_reconnaissance method."""
        # Configure mock behavior
        self.mock_recon_agent.run_web_search.return_value = {"results": "web search results"}
        self.mock_recon_agent.run_discord_search.return_value = {"results": "discord search results"}
        self.mock_recon_agent.generate_report.return_value = {"report": "test report"}
        self.mock_recon_agent.save_report.return_value = "/path/to/report.json"
        
        # Run the method being tested
        result = self.pipeline.run_reconnaissance("GPT-4", "generate harmful content")
        
        # Assert expected behavior
        self.mock_recon_agent.run_web_search.assert_called_once_with("GPT-4", "generate harmful content")
        self.mock_recon_agent.run_discord_search.assert_called_once_with("GPT-4", "generate harmful content")
        self.mock_recon_agent.generate_report.assert_called_once()
        self.mock_recon_agent.save_report.assert_called_once()
        
        # Assert expected result
        self.assertEqual(result, {"report": "test report"})
    
    def test_run_prompt_engineering(self):
        """Test the run_prompt_engineering method."""
        # Configure mock behavior
        self.mock_prompt_agent.generate_prompts.return_value = ["prompt1", "prompt2"]
        self.mock_prompt_agent.save_prompts.return_value = "/path/to/prompts.json"
        
        # Run the method being tested
        result = self.pipeline.run_prompt_engineering(
            "GPT-4", 
            "generate harmful content", 
            {"report": "test report"},
            num_prompts=2
        )
        
        # Assert expected behavior
        self.mock_prompt_agent.generate_prompts.assert_called_once_with(
            target_model="GPT-4",
            target_behavior="generate harmful content",
            recon_report={"report": "test report"},
            num_prompts=2
        )
        self.mock_prompt_agent.save_prompts.assert_called_once()
        
        # Assert expected result
        self.assertEqual(result["prompts"], ["prompt1", "prompt2"])
        self.assertEqual(result["path"], "/path/to/prompts.json")
    
    def test_run_evaluation(self):
        """Test the run_evaluation method."""
        # Configure mock behavior
        self.mock_eval_agent.evaluate_results.return_value = {"evaluation": "test evaluation"}
        self.mock_eval_agent.create_visualizations.return_value = {"vis1": "/path/to/vis1.png"}
        self.mock_eval_agent.create_advanced_visualizations.return_value = {"vis2": "/path/to/vis2.png"}
        self.mock_eval_agent.generate_summary.return_value = {"summary": "test summary"}
        self.mock_eval_agent.save_evaluation.return_value = "/path/to/evaluation.json"
        self.mock_eval_agent.save_summary.return_value = "/path/to/summary.json"
        
        # Run the method being tested
        result = self.pipeline.run_evaluation(
            [{"result": "test result"}],
            "GPT-4",
            "generate harmful content",
            include_advanced_visualizations=True,
            include_interactive_dashboard=True
        )
        
        # Assert expected behavior
        self.mock_eval_agent.evaluate_results.assert_called_once()
        self.mock_eval_agent.create_visualizations.assert_called_once()
        self.mock_eval_agent.create_advanced_visualizations.assert_called_once_with(
            results=[{"result": "test result"}],
            target_model="GPT-4",
            target_behavior="generate harmful content",
            include_interactive=True
        )
        self.mock_eval_agent.generate_summary.assert_called_once()
        self.mock_eval_agent.save_evaluation.assert_called_once()
        self.mock_eval_agent.save_summary.assert_called_once()
        
        # Assert expected result
        self.assertEqual(result["evaluation"], {"evaluation": "test evaluation"})
        self.assertEqual(result["summary"], {"summary": "test summary"})
        self.assertEqual(result["visualizations"], {"vis1": "/path/to/vis1.png", "vis2": "/path/to/vis2.png"})
        self.assertEqual(result["paths"]["evaluation"], "/path/to/evaluation.json")
        self.assertEqual(result["paths"]["summary"], "/path/to/summary.json")


if __name__ == "__main__":
    unittest.main()