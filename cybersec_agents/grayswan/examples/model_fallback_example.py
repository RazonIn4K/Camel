"""
Example script demonstrating the Model Integration and Fallback System.

This script shows how to use the Model Integration and Fallback System with
dependency injection to handle model failures and complexity-based model selection.
"""

import os
import sys
import asyncio
from typing import Dict, Any

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from cybersec_agents.grayswan.container import GraySwanContainerFactory
from cybersec_agents.grayswan.utils.model_manager_di import ModelManager
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("model_fallback_example")


async def demonstrate_model_fallback():
    """
    Demonstrate the model fallback capabilities.
    
    This function shows how the system automatically falls back to a backup model
    when the primary model fails.
    """
    print("\n=== Model Fallback Example ===\n")
    
    # Create container with default configuration
    container = GraySwanContainerFactory.create_container()
    
    # Get model manager from container
    model_manager = container.model_manager()
    
    print(f"Primary model: {model_manager.primary_model}")
    print(f"Backup model: {model_manager.backup_model}")
    print(f"Complexity threshold: {model_manager.complexity_threshold}")
    
    # Create a prompt that will be processed
    prompt: str = "Explain the concept of dependency injection in software engineering."
    
    try:
        # Simulate primary model failure by using a non-existent model
        # This will cause the system to fall back to the backup model
        model_manager.primary_model = "non-existent-model"
        
        print("\nAttempting to generate with non-existent primary model...")
        print("This should trigger fallback to the backup model.")
        
        # Generate with fallback
        response = await model_manager.generate_async(prompt)
        
        print(f"\nGeneration successful using backup model!")
        print(f"Response: {response.get('text', '')[:100]}...")
        
        # Print metrics
        metrics = model_manager.get_metrics()
        print(f"\nMetrics: {metrics}")
        print(f"Primary calls: {metrics['primary_calls']}")
        print(f"Backup calls: {metrics['backup_calls']}")
        print(f"Failures: {metrics['failures']}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        
        # Print metrics even if there was an error
        metrics = model_manager.get_metrics()
        print(f"\nMetrics: {metrics}")


async def demonstrate_complexity_based_selection():
    """
    Demonstrate complexity-based model selection.
    
    This function shows how the system automatically selects the appropriate model
    based on the complexity of the prompt.
    """
    print("\n=== Complexity-Based Model Selection Example ===\n")
    
    # Create container with custom configuration
    config_dict: dict[str, Any] = {
        'model': {
            'primary_model': 'gpt-4',
            'backup_model': 'gpt-3.5-turbo',
            'complexity_threshold': 0.5,  # Lower threshold for demonstration
        },
    }
    container = GraySwanContainerFactory.create_container(config_dict)
    
    # Get model manager from container
    model_manager = container.model_manager()
    
    print(f"Primary model: {model_manager.primary_model}")
    print(f"Backup model: {model_manager.backup_model}")
    print(f"Complexity threshold: {model_manager.complexity_threshold}")
    
    # Create a simple prompt
    simple_prompt: str = "What is the capital of France?"
    
    # Create a complex prompt
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
    
    # Estimate complexity
    simple_complexity = model_manager.estimate_complexity(simple_prompt)
    complex_complexity = model_manager.estimate_complexity(complex_prompt)
    
    print(f"\nSimple prompt complexity: {simple_complexity:.2f}")
    print(f"Complex prompt complexity: {complex_complexity:.2f}")
    
    try:
        # Process simple prompt
        print("\nProcessing simple prompt...")
        print(f"Expected model: {model_manager.primary_model} (Primary)")
        
        # Generate with simple prompt
        simple_response = await model_manager.generate_async(
            simple_prompt, 
            complexity=simple_complexity
        )
        
        # Process complex prompt
        print("\nProcessing complex prompt...")
        print(f"Expected model: {model_manager.backup_model} (Backup)")
        
        # Generate with complex prompt
        complex_response = await model_manager.generate_async(
            complex_prompt, 
            complexity=complex_complexity
        )
        
        # Print metrics
        metrics = model_manager.get_metrics()
        print(f"\nMetrics: {metrics}")
        print(f"Primary calls: {metrics['primary_calls']}")
        print(f"Backup calls: {metrics['backup_calls']}")
        print(f"Failures: {metrics['failures']}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        
        # Print metrics even if there was an error
        metrics = model_manager.get_metrics()
        print(f"\nMetrics: {metrics}")


async def demonstrate_agent_specific_models():
    """
    Demonstrate agent-specific model managers.
    
    This function shows how different agents can use different model managers
    with their own configurations.
    """
    print("\n=== Agent-Specific Model Managers Example ===\n")
    
    # Create container with custom configuration
    config_dict: dict[str, Any] = {
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
    container = GraySwanContainerFactory.create_container(config_dict)
    
    # Get model managers for different agents
    recon_model_manager = container.recon_model_manager()
    prompt_model_manager = container.prompt_engineer_model_manager()
    exploit_model_manager = container.exploit_delivery_model_manager()
    eval_model_manager = container.evaluation_model_manager()
    
    # Print configurations
    print("Reconnaissance Agent Model Manager:")
    print(f"  Primary model: {recon_model_manager.primary_model}")
    print(f"  Backup model: {recon_model_manager.backup_model}")
    print(f"  Complexity threshold: {recon_model_manager.complexity_threshold}")
    
    print("\nPrompt Engineer Agent Model Manager:")
    print(f"  Primary model: {prompt_model_manager.primary_model}")
    print(f"  Backup model: {prompt_model_manager.backup_model}")
    print(f"  Complexity threshold: {prompt_model_manager.complexity_threshold}")
    
    print("\nExploit Delivery Agent Model Manager:")
    print(f"  Primary model: {exploit_model_manager.primary_model}")
    print(f"  Backup model: {exploit_model_manager.backup_model}")
    print(f"  Complexity threshold: {exploit_model_manager.complexity_threshold}")
    
    print("\nEvaluation Agent Model Manager:")
    print(f"  Primary model: {eval_model_manager.primary_model}")
    print(f"  Backup model: {eval_model_manager.backup_model}")
    print(f"  Complexity threshold: {eval_model_manager.complexity_threshold}")


async def main():
    """Main function for the example script."""
    print("Model Integration and Fallback System Example")
    print("============================================")
    
    # Run examples
    await demonstrate_model_fallback()
    await demonstrate_complexity_based_selection()
    await demonstrate_agent_specific_models()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    asyncio.run(main())