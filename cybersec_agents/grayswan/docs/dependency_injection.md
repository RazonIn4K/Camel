# Dependency Injection in Gray Swan Arena

This document explains the dependency injection pattern implementation in the Gray Swan Arena framework.

## Overview

Dependency Injection (DI) is a design pattern that allows for better code organization, testability, and flexibility. In Gray Swan Arena, we've implemented DI using the `dependency_injector` library to manage dependencies between components.

The key benefits of using DI in Gray Swan Arena include:

1. **Improved Testability**: Components can be easily mocked or replaced during testing
2. **Better Flexibility**: Configuration can be changed without modifying code
3. **Cleaner Architecture**: Dependencies are explicit and managed centrally
4. **Reduced Coupling**: Components depend on abstractions rather than concrete implementations
5. **Easier Maintenance**: Dependencies can be updated or replaced without affecting dependent components

## Components

The DI implementation in Gray Swan Arena consists of the following components:

### 1. Container

The `GraySwanContainer` class in `container.py` is the central component of the DI system. It defines all the dependencies and their relationships.

```python
from dependency_injector import containers, providers

class GraySwanContainer(containers.DeclarativeContainer):
    """Dependency Injection Container for Gray Swan Arena."""
    
    # Configuration provider
    config = providers.Configuration()
    
    # Set default configuration values
    config.set_default_values({
        'output_dir': './output',
        'agents': {
            'recon': {
                'output_dir': './output/recon_reports',
                'model_name': 'gpt-4',
            },
            # ... other agent configurations
        },
        # ... other configuration sections
    })
    
    # Providers for various components
    logger = providers.Singleton(setup_logging, name="grayswan")
    
    recon_agent = providers.Factory(
        ReconAgent,
        output_dir=config.agents.recon.output_dir,
        model_name=config.agents.recon.model_name,
    )
    
    # ... other providers
```

### 2. Container Factory

The `GraySwanContainerFactory` class provides methods for creating and configuring containers:

```python
class GraySwanContainerFactory:
    """Factory for creating and configuring GraySwanContainer instances."""
    
    @staticmethod
    def create_container(config_dict: Optional[Dict[str, Any]] = None) -> GraySwanContainer:
        """Create and configure a GraySwanContainer."""
        container = GraySwanContainer()
        
        if config_dict:
            container.config.from_dict(config_dict)
            
        return container
    
    @staticmethod
    def create_container_from_file(config_file: str) -> GraySwanContainer:
        """Create and configure a GraySwanContainer from a configuration file."""
        container = GraySwanContainer()
        
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            container.config.from_yaml(config_file)
        elif config_file.endswith('.json'):
            container.config.from_json(config_file)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file}")
            
        return container
```

### 3. Pipeline

The `GraySwanPipeline` class in `main_di.py` uses the container to access dependencies:

```python
class GraySwanPipeline:
    """Gray Swan Arena pipeline with dependency injection."""
    
    def __init__(self, container: GraySwanContainer):
        """Initialize the GraySwanPipeline."""
        self.container = container
        self.logger = container.logger()
        
        # ... initialization code
    
    def run_reconnaissance(self, target_model: str, target_behavior: str) -> Dict[str, Any]:
        """Run the reconnaissance phase of the Gray Swan Arena pipeline."""
        # Get ReconAgent from container
        recon_agent = self.container.recon_agent()
        
        # ... method implementation
```

## Configuration

The DI system in Gray Swan Arena supports multiple configuration methods:

### 1. Default Configuration

Default configuration values are defined in the `GraySwanContainer` class:

```python
config.set_default_values({
    'output_dir': './output',
    'agents': {
        'recon': {
            'output_dir': './output/recon_reports',
            'model_name': 'gpt-4',
        },
        # ... other agent configurations
    },
    # ... other configuration sections
})
```

### 2. Dictionary Configuration

Configuration can be provided as a dictionary:

```python
config_dict = {
    'output_dir': './custom_output',
    'agents': {
        'recon': {
            'model_name': 'gpt-3.5-turbo',
        },
    },
}

container = GraySwanContainerFactory.create_container(config_dict)
```

### 3. File Configuration

Configuration can be loaded from YAML or JSON files:

```python
container = GraySwanContainerFactory.create_container_from_file('config.yaml')
```

### 4. Command-Line Configuration

The `main_di.py` script supports command-line arguments that override configuration values:

```bash
python -m cybersec_agents.grayswan.main_di \
    --target-model "GPT-4" \
    --target-behavior "generate harmful content" \
    --output-dir "./custom_output" \
    --agent-model "gpt-3.5-turbo" \
    --config-file "config.yaml"
```

## Usage

### Basic Usage

```python
from cybersec_agents.grayswan.container import GraySwanContainerFactory
from cybersec_agents.grayswan.main_di import GraySwanPipeline

# Create container with default configuration
container = GraySwanContainerFactory.create_container()

# Create pipeline
pipeline = GraySwanPipeline(container)

# Run pipeline
results = pipeline.run_full_pipeline(
    target_model="GPT-4",
    target_behavior="generate harmful content",
)
```

### Custom Configuration

```python
from cybersec_agents.grayswan.container import GraySwanContainerFactory
from cybersec_agents.grayswan.main_di import GraySwanPipeline

# Create configuration dictionary
config_dict = {
    'output_dir': './custom_output',
    'agents': {
        'recon': {
            'model_name': 'gpt-3.5-turbo',
        },
    },
    'visualization': {
        'advanced': True,
        'interactive': True,
    },
}

# Create container with custom configuration
container = GraySwanContainerFactory.create_container(config_dict)

# Create pipeline
pipeline = GraySwanPipeline(container)

# Run pipeline
results = pipeline.run_full_pipeline(
    target_model="GPT-4",
    target_behavior="generate harmful content",
)
```

### Configuration from File

```python
from cybersec_agents.grayswan.container import GraySwanContainerFactory
from cybersec_agents.grayswan.main_di import GraySwanPipeline

# Create container from configuration file
container = GraySwanContainerFactory.create_container_from_file('config.yaml')

# Create pipeline
pipeline = GraySwanPipeline(container)

# Run pipeline
results = pipeline.run_full_pipeline(
    target_model="GPT-4",
    target_behavior="generate harmful content",
)
```

## Testing with Dependency Injection

One of the key benefits of DI is improved testability. Here's an example of how to test a component with DI:

```python
import unittest
from unittest.mock import MagicMock
from cybersec_agents.grayswan.container import GraySwanContainer
from cybersec_agents.grayswan.main_di import GraySwanPipeline

class TestGraySwanPipeline(unittest.TestCase):
    def setUp(self):
        # Create a container with mocked dependencies
        self.container = GraySwanContainer()
        
        # Mock the logger
        self.mock_logger = MagicMock()
        self.container.logger = lambda: self.mock_logger
        
        # Mock the recon agent
        self.mock_recon_agent = MagicMock()
        self.container.recon_agent = lambda: self.mock_recon_agent
        
        # Create pipeline with mocked container
        self.pipeline = GraySwanPipeline(self.container)
    
    def test_run_reconnaissance(self):
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
```

## Configuration File Format

### YAML Configuration

```yaml
output_dir: ./custom_output
agents:
  recon:
    output_dir: ./custom_output/recon_reports
    model_name: gpt-4
  prompt_engineer:
    output_dir: ./custom_output/prompt_lists
    model_name: gpt-4
  exploit_delivery:
    output_dir: ./custom_output/exploit_logs
    model_name: gpt-4
    browser_method: playwright
    headless: true
  evaluation:
    output_dir: ./custom_output/evaluation_reports
    model_name: gpt-4
browser:
  method: playwright
  headless: true
  timeout: 60000
  enhanced: true
  retry_attempts: 3
  retry_delay: 1.0
visualization:
  output_dir: ./custom_output/visualizations
  dpi: 300
  theme: default
  advanced: true
  interactive: true
  clustering_clusters: 4
  similarity_threshold: 0.5
```

### JSON Configuration

```json
{
  "output_dir": "./custom_output",
  "agents": {
    "recon": {
      "output_dir": "./custom_output/recon_reports",
      "model_name": "gpt-4"
    },
    "prompt_engineer": {
      "output_dir": "./custom_output/prompt_lists",
      "model_name": "gpt-4"
    },
    "exploit_delivery": {
      "output_dir": "./custom_output/exploit_logs",
      "model_name": "gpt-4",
      "browser_method": "playwright",
      "headless": true
    },
    "evaluation": {
      "output_dir": "./custom_output/evaluation_reports",
      "model_name": "gpt-4"
    }
  },
  "browser": {
    "method": "playwright",
    "headless": true,
    "timeout": 60000,
    "enhanced": true,
    "retry_attempts": 3,
    "retry_delay": 1.0
  },
  "visualization": {
    "output_dir": "./custom_output/visualizations",
    "dpi": 300,
    "theme": "default",
    "advanced": true,
    "interactive": true,
    "clustering_clusters": 4,
    "similarity_threshold": 0.5
  }
}
```

## Best Practices

1. **Use the Container**: Always use the container to get dependencies rather than creating them directly.
2. **Configuration**: Use configuration files or dictionaries to configure the container rather than hardcoding values.
3. **Testing**: Use mocks to test components in isolation.
4. **Dependency Resolution**: Let the container resolve dependencies rather than passing them manually.
5. **Singleton vs. Factory**: Use `providers.Singleton` for components that should be shared, and `providers.Factory` for components that should be created on demand.

## Conclusion

The dependency injection pattern in Gray Swan Arena provides a flexible, testable, and maintainable architecture. By centralizing dependency management and configuration, it makes the codebase more robust and easier to extend.