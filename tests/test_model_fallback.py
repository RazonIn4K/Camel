from typing import Any, Dict, List, Optional, Tuple, Union
"""
Tests for the Model Integration and Fallback System.

This module contains tests for the ModelManager and ModelManagerProvider classes.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from cybersec_agents.grayswan.utils.model_manager_di import ModelManager, ModelManagerProvider
from cybersec_agents.grayswan.container import GraySwanContainerFactory


class TestModelManager(unittest.TestCase):
    """Tests for the ModelManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model
        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = {"text": "Test response"}
        
        # Create a ModelManager with mocked models
        self.model_manager = ModelManager(
            primary_model="gpt-4",
            backup_model="gpt-3.5-turbo",
            complexity_threshold=0.7
        )
        
        # Replace the models dictionary with our mocks
        self.model_manager.models = {
            "gpt-4": self.mock_model,
            "gpt-3.5-turbo": self.mock_model,
        }
    
    def test_generate(self):
        """Test the generate method."""
        # Call the generate method
        response = self.model_manager.generate("Test prompt")
        
        # Assert that the model was called
        self.mock_model.generate.assert_called_once_with("Test prompt")
        
        # Assert that the response is correct
        self.assertEqual(response, {"text": "Test response"})
        
        # Assert that the metrics were updated
        self.assertEqual(self.model_manager.metrics["primary_calls"], 1)
        self.assertEqual(self.model_manager.metrics["backup_calls"], 0)
        self.assertEqual(self.model_manager.metrics["failures"], 0)
    
    def test_generate_with_complexity(self):
        """Test the generate method with complexity-based model selection."""
        # Call the generate method with high complexity
        response = self.model_manager.generate("Test prompt", complexity=0.8)
        
        # Assert that the model was called
        self.mock_model.generate.assert_called_once_with("Test prompt")
        
        # Assert that the response is correct
        self.assertEqual(response, {"text": "Test response"})
        
        # Assert that the metrics were updated
        self.assertEqual(self.model_manager.metrics["primary_calls"], 0)
        self.assertEqual(self.model_manager.metrics["backup_calls"], 1)
        self.assertEqual(self.model_manager.metrics["failures"], 0)
    
    def test_generate_with_explicit_model(self):
        """Test the generate method with an explicit model."""
        # Call the generate method with an explicit model
        response = self.model_manager.generate("Test prompt", model="gpt-3.5-turbo")
        
        # Assert that the model was called
        self.mock_model.generate.assert_called_once_with("Test prompt")
        
        # Assert that the response is correct
        self.assertEqual(response, {"text": "Test response"})
        
        # Assert that the metrics were updated
        self.assertEqual(self.model_manager.metrics["primary_calls"], 1)
        self.assertEqual(self.model_manager.metrics["backup_calls"], 0)
        self.assertEqual(self.model_manager.metrics["failures"], 0)
    
    def test_fallback(self):
        """Test fallback to backup model when primary model fails."""
        # Make the primary model fail
        primary_model = MagicMock()
        primary_model.generate.side_effect = Exception("Primary model failed")
        
        # Make the backup model succeed
        backup_model = MagicMock()
        backup_model.generate.return_value = {"text": "Backup response"}
        
        # Replace the models dictionary with our mocks
        self.model_manager.models = {
            "gpt-4": primary_model,
            "gpt-3.5-turbo": backup_model,
        }
        
        # Call the generate method
        response = self.model_manager.generate("Test prompt")
        
        # Assert that both models were called
        primary_model.generate.assert_called_once_with("Test prompt")
        backup_model.generate.assert_called_once_with("Test prompt")
        
        # Assert that the response is from the backup model
        self.assertEqual(response, {"text": "Backup response"})
        
        # Assert that the metrics were updated
        self.assertEqual(self.model_manager.metrics["primary_calls"], 1)
        self.assertEqual(self.model_manager.metrics["backup_calls"], 1)
        self.assertEqual(self.model_manager.metrics["failures"], 0)
    
    def test_estimate_complexity(self):
        """Test the estimate_complexity method."""
        # Test with a simple prompt
        simple_prompt: str = "What is the capital of France?"
        simple_complexity = self.model_manager.estimate_complexity(simple_prompt)
        
        # Test with a complex prompt
        complex_prompt: str = """
        Analyze the following code and explain in detail how it implements the Observer pattern.
        Provide step-by-step explanation of how the pattern works, compare it with other behavioral
        patterns, and suggest improvements for better maintainability and extensibility.
        
        ```python
        class Subject:
            def __init__(self):
                self._observers = []
                self._state = None
                
            def attach(self, observer):
                if observer not in self._observers:
                    self._observers.append(observer)
                    
            def detach(self, observer):
                try:
                    self._observers.remove(observer)
                except ValueError:
                    pass
                    
            def notify(self):
                for observer in self._observers:
                    observer.update(self)
                    
            @property
            def state(self):
                return self._state
                
            @state.setter
            def state(self, value):
                self._state = value
                self.notify()
                
        class Observer:
            def update(self, subject):
                pass
                
        class ConcreteObserverA(Observer):
            def update(self, subject):
                print(f"ConcreteObserverA: Reacted to the event. New state: {subject.state}")
                
        class ConcreteObserverB(Observer):
            def update(self, subject):
                print(f"ConcreteObserverB: Reacted to the event. New state: {subject.state}")
        ```
        """
        complex_complexity = self.model_manager.estimate_complexity(complex_prompt)
        
        # Assert that the complex prompt has higher complexity
        self.assertLess(simple_complexity, complex_complexity)
        
        # Assert that the complexity is within the expected range
        self.assertGreaterEqual(simple_complexity, 0.0)
        self.assertLessEqual(simple_complexity, 1.0)
        self.assertGreaterEqual(complex_complexity, 0.0)
        self.assertLessEqual(complex_complexity, 1.0)
    
    def test_get_metrics(self):
        """Test the get_metrics method."""
        # Set some metrics
        self.model_manager.metrics = {
            "primary_calls": 10,
            "backup_calls": 5,
            "failures": 2,
        }
        
        # Get the metrics
        metrics = self.model_manager.get_metrics()
        
        # Assert that the metrics are correct
        self.assertEqual(metrics, {
            "primary_calls": 10,
            "backup_calls": 5,
            "failures": 2,
        })
        
        # Assert that the metrics are a copy
        self.assertIsNot(metrics, self.model_manager.metrics)


class TestModelManagerProvider(unittest.TestCase):
    """Tests for the ModelManagerProvider class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config
        self.mock_config = {
            "model": {
                "primary_model": "gpt-4",
                "backup_model": "gpt-3.5-turbo",
                "complexity_threshold": 0.7,
            },
            "agents": {
                "recon": {
                    "model_name": "gpt-4",
                    "backup_model": "gpt-3.5-turbo",
                    "complexity_threshold": 0.8,
                },
                "prompt_engineer": {
                    "model_name": "gpt-4-turbo",
                    "backup_model": "gpt-4",
                    "complexity_threshold": 0.7,
                },
                "exploit_delivery": {
                    "model_name": "gpt-3.5-turbo",
                    "backup_model": None,
                    "complexity_threshold": 0.5,
                },
                "evaluation": {
                    "model_name": "gpt-4",
                    "backup_model": "claude-2",
                    "complexity_threshold": 0.6,
                },
            },
        }
    
    @patch("cybersec_agents.grayswan.utils.model_manager_di.ModelManager")
    def test_create_manager(self, mock_model_manager_class):
        """Test the create_manager method."""
        # Create a manager
        ModelManagerProvider.create_manager(
            primary_model="gpt-4",
            backup_model="gpt-3.5-turbo",
            complexity_threshold=0.7,
            config=self.mock_config,
        )
        
        # Assert that the ModelManager was created with the correct parameters
        mock_model_manager_class.assert_called_once_with(
            primary_model="gpt-4",
            backup_model="gpt-3.5-turbo",
            complexity_threshold=0.7,
        )
    
    @patch("cybersec_agents.grayswan.utils.model_manager_di.ModelManager")
    def test_create_manager_with_defaults(self, mock_model_manager_class):
        """Test the create_manager method with default values."""
        # Create a manager with default values
        ModelManagerProvider.create_manager(
            primary_model="default",
            config=self.mock_config,
        )
        
        # Assert that the ModelManager was created with the correct parameters
        mock_model_manager_class.assert_called_once_with(
            primary_model="gpt-4",  # From config
            backup_model="gpt-3.5-turbo",  # From config
            complexity_threshold=0.7,  # From config
        )
    
    @patch("cybersec_agents.grayswan.utils.model_manager_di.ModelManager")
    def test_create_for_agent(self, mock_model_manager_class):
        """Test the create_for_agent method."""
        # Create a manager for a specific agent
        ModelManagerProvider.create_for_agent(
            agent_type="recon",
            config=self.mock_config,
        )
        
        # Assert that the ModelManager was created with the correct parameters
        mock_model_manager_class.assert_called_once_with(
            primary_model="gpt-4",
            backup_model="gpt-3.5-turbo",
            complexity_threshold=0.8,
        )
    
    @patch("cybersec_agents.grayswan.utils.model_manager_di.ModelManager")
    def test_create_for_agent_with_no_backup(self, mock_model_manager_class):
        """Test the create_for_agent method with an agent that has no backup model."""
        # Create a manager for a specific agent
        ModelManagerProvider.create_for_agent(
            agent_type="exploit_delivery",
            config=self.mock_config,
        )
        
        # Assert that the ModelManager was created with the correct parameters
        mock_model_manager_class.assert_called_once_with(
            primary_model="gpt-3.5-turbo",
            backup_model=None,
            complexity_threshold=0.5,
        )


class TestModelManagerIntegration(unittest.TestCase):
    """Integration tests for the ModelManager with the container."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a container with custom configuration
        config_dict: dict[str, Any] = {
            'model': {
                'primary_model': 'gpt-4',
                'backup_model': 'gpt-3.5-turbo',
                'complexity_threshold': 0.7,
            },
            'agents': {
                'recon': {
                    'model_name': 'gpt-4',
                    'backup_model': 'gpt-3.5-turbo',
                    'complexity_threshold': 0.8,
                },
                'prompt_engineer': {
                    'model_name': 'gpt-4-turbo',
                    'backup_model': 'gpt-4',
                    'complexity_threshold': 0.7,
                },
                'exploit_delivery': {
                    'model_name': 'gpt-3.5-turbo',
                    'backup_model': None,
                    'complexity_threshold': 0.5,
                },
                'evaluation': {
                    'model_name': 'gpt-4',
                    'backup_model': 'claude-2',
                    'complexity_threshold': 0.6,
                },
            },
        }
        self.container = GraySwanContainerFactory.create_container(config_dict)
        
        # Patch the _ensure_model method to avoid creating real models
        patcher = patch.object(ModelManager, '_ensure_model')
        self.addCleanup(patcher.stop)
        patcher.start()
        
        # Patch the generate method to return a mock response
        patcher = patch.object(ModelManager, 'generate')
        self.mock_generate = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_generate.return_value = {"text": "Test response"}
    
    def test_model_manager_from_container(self):
        """Test getting a ModelManager from the container."""
        # Get a model manager from the container
        model_manager = self.container.model_manager()
        
        # Assert that the model manager has the correct configuration
        self.assertEqual(model_manager.primary_model, 'gpt-4')
        self.assertEqual(model_manager.backup_model, 'gpt-3.5-turbo')
        self.assertEqual(model_manager.complexity_threshold, 0.7)
    
    def test_agent_specific_model_managers(self):
        """Test getting agent-specific model managers from the container."""
        # Get model managers for different agents
        recon_model_manager = self.container.recon_model_manager()
        prompt_model_manager = self.container.prompt_engineer_model_manager()
        exploit_model_manager = self.container.exploit_delivery_model_manager()
        eval_model_manager = self.container.evaluation_model_manager()
        
        # Assert that the model managers have the correct configurations
        self.assertEqual(recon_model_manager.primary_model, 'gpt-4')
        self.assertEqual(recon_model_manager.backup_model, 'gpt-3.5-turbo')
        self.assertEqual(recon_model_manager.complexity_threshold, 0.8)
        
        self.assertEqual(prompt_model_manager.primary_model, 'gpt-4-turbo')
        self.assertEqual(prompt_model_manager.backup_model, 'gpt-4')
        self.assertEqual(prompt_model_manager.complexity_threshold, 0.7)
        
        self.assertEqual(exploit_model_manager.primary_model, 'gpt-3.5-turbo')
        self.assertEqual(exploit_model_manager.backup_model, None)
        self.assertEqual(exploit_model_manager.complexity_threshold, 0.5)
        
        self.assertEqual(eval_model_manager.primary_model, 'gpt-4')
        self.assertEqual(eval_model_manager.backup_model, 'claude-2')
        self.assertEqual(eval_model_manager.complexity_threshold, 0.6)


if __name__ == "__main__":
    unittest.main()