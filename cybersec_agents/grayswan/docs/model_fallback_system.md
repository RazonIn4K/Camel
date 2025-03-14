# Model Integration and Fallback System

This document explains the Model Integration and Fallback System in the Gray Swan Arena framework.

## Overview

The Model Integration and Fallback System provides a robust mechanism for interacting with language models, with support for:

1. **Fallback Capabilities**: Automatically fall back to backup models when primary models fail
2. **Complexity-Based Model Selection**: Use different models based on prompt complexity
3. **Exponential Backoff**: Handle rate limits and transient errors with exponential backoff
4. **Dependency Injection**: Flexible configuration and improved testability
5. **Agent-Specific Models**: Different agents can use different model configurations

## Components

The system consists of the following components:

### 1. ModelManager

The `ModelManager` class in `model_manager_di.py` is the core component of the system. It provides methods for generating completions with fallback capabilities:

```python
class ModelManager:
    """Manages model interactions with fallback capabilities."""
    
    def __init__(
        self,
        primary_model: str,
        backup_model: Optional[str] = None,
        complexity_threshold: float = 0.7,
    ):
        """Initialize the ModelManager."""
        self.primary_model = primary_model
        self.backup_model = backup_model
        self.complexity_threshold = complexity_threshold
        self.metrics = {"primary_calls": 0, "backup_calls": 0, "failures": 0}
        
        # Initialize models
        self.models = {}
        self._initialize_models()
```

### 2. ModelManagerProvider

The `ModelManagerProvider` class provides factory methods for creating `ModelManager` instances with different configurations:

```python
class ModelManagerProvider:
    """Provider for ModelManager instances with dependency injection support."""
    
    @staticmethod
    @inject
    def create_manager(
        primary_model: str,
        backup_model: Optional[str] = None,
        complexity_threshold: float = 0.7,
        config: Dict[str, Any] = Provide["config"],
    ) -> "ModelManager":
        """Create a ModelManager instance with the specified configuration."""
        # ...
    
    @staticmethod
    @inject
    def create_for_agent(
        agent_type: str,
        config: Dict[str, Any] = Provide["config"],
    ) -> "ModelManager":
        """Create a ModelManager instance for a specific agent type."""
        # ...
```

### 3. Container Integration

The system is integrated with the dependency injection container in `container.py`:

```python
# Model manager providers
model_manager = providers.Factory(
    ModelManager,
    primary_model=config.model.primary_model,
    backup_model=config.model.backup_model,
    complexity_threshold=config.model.complexity_threshold,
)

recon_model_manager = providers.Factory(
    ModelManagerProvider.create_for_agent,
    agent_type="recon",
)

# ... other model manager providers
```

## Features

### 1. Fallback Capabilities

The system automatically falls back to a backup model when the primary model fails:

```python
try:
    # Try primary model
    response = model_obj.generate(prompt, **kwargs)
    return response
except Exception as e:
    # Try backup if primary fails
    if model_to_use != self.backup_model and self.backup_model:
        logger.warning(f"Primary model {model_to_use} failed: {str(e)}. Falling back to {self.backup_model}")
        # ... use backup model
```

### 2. Complexity-Based Model Selection

The system can select different models based on the complexity of the prompt:

```python
# Use backup for complex prompts if available
if (
    complexity is not None
    and complexity >= self.complexity_threshold
    and self.backup_model
    and not model  # Don't override explicit model choice
):
    model_to_use = self.backup_model
    self.metrics["backup_calls"] += 1
    logger.info(f"Using backup model {model_to_use} due to complexity {complexity}")
else:
    self.metrics["primary_calls"] += 1
```

### 3. Exponential Backoff

The system uses exponential backoff to handle rate limits and transient errors:

```python
@with_exponential_backoff
def generate(self, prompt: str, model: Optional[str] = None, complexity: Optional[float] = None, **kwargs) -> Dict[str, Any]:
    """Generate a completion with fallback capability (sync version)."""
    # ...
```

The `with_exponential_backoff` decorator retries the function with increasing delays:

```python
def with_exponential_backoff(
    func: Callable,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[Exception, ...] = (RateLimitError, APIError),
):
    """Decorator to retry a function with exponential backoff."""
    # ...
```

### 4. Complexity Estimation

The system can estimate the complexity of a prompt to determine which model to use:

```python
def estimate_complexity(self, prompt: str) -> float:
    """
    Estimate the complexity of a prompt.
    
    Args:
        prompt: The prompt to analyze
        
    Returns:
        Complexity score between 0.0 and 1.0
    """
    # Simple complexity estimation based on length and special tokens
    length_score = min(len(prompt) / 4000, 1.0) * 0.7
    
    # Check for complex instructions
    complex_indicators = [
        "step by step",
        "explain in detail",
        "analyze",
        # ... other indicators
    ]
    
    # ... calculate complexity score
```

### 5. Metrics

The system tracks metrics about model usage:

```python
def get_metrics(self) -> Dict[str, int]:
    """
    Get usage metrics.
    
    Returns:
        Dictionary of usage metrics
    """
    return self.metrics.copy()
```

## Configuration

The system can be configured through the dependency injection container:

### 1. Default Configuration

```python
config.set_default_values({
    'model': {
        'primary_model': 'gpt-4',
        'backup_model': 'gpt-3.5-turbo',
        'complexity_threshold': 0.7,
        'max_retries': 5,
        'initial_delay': 1.0,
        'backoff_factor': 2.0,
        'jitter': True,
    },
    'agents': {
        'recon': {
            'model_name': 'gpt-4',
            'backup_model': 'gpt-3.5-turbo',
            'complexity_threshold': 0.7,
        },
        # ... other agent configurations
    },
    # ... other configuration sections
})
```

### 2. Custom Configuration

```python
config_dict = {
    'model': {
        'primary_model': 'claude-3-opus',
        'backup_model': 'claude-3-sonnet',
        'complexity_threshold': 0.5,
    },
    'agents': {
        'recon': {
            'model_name': 'gpt-4-turbo',
            'backup_model': 'gpt-4',
        },
    },
}

container = GraySwanContainerFactory.create_container(config_dict)
```

### 3. Agent-Specific Configuration

Each agent can have its own model configuration:

```python
config_dict = {
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
        # ... other agent configurations
    },
}
```

## Usage

### Basic Usage

```python
from cybersec_agents.grayswan.container import GraySwanContainerFactory

# Create container
container = GraySwanContainerFactory.create_container()

# Get model manager
model_manager = container.model_manager()

# Generate with fallback
response = model_manager.generate(
    prompt="Explain the concept of dependency injection.",
    complexity=0.5  # Optional complexity score
)
```

### Agent-Specific Usage

```python
# Get model manager for a specific agent
recon_model_manager = container.recon_model_manager()

# Generate with agent-specific model
response = recon_model_manager.generate(
    prompt="Research the latest AI safety techniques.",
    complexity=0.8
)
```

### Async Usage

```python
# Generate asynchronously
response = await model_manager.generate_async(
    prompt="Explain the concept of dependency injection.",
    complexity=0.5
)
```

### Metrics

```python
# Get metrics
metrics = model_manager.get_metrics()
print(f"Primary calls: {metrics['primary_calls']}")
print(f"Backup calls: {metrics['backup_calls']}")
print(f"Failures: {metrics['failures']}")
```

## Example

See the `model_fallback_example.py` script for a complete example of how to use the Model Integration and Fallback System:

```bash
# Run the example
python -m cybersec_agents.grayswan.examples.model_fallback_example
```

## Best Practices

1. **Configure Appropriate Backup Models**: Choose backup models that are more reliable but potentially less capable than primary models.
2. **Set Appropriate Complexity Thresholds**: Adjust complexity thresholds based on the capabilities of your models.
3. **Monitor Metrics**: Regularly check metrics to ensure the system is working as expected.
4. **Handle Errors**: Always handle errors that might occur even with fallback mechanisms.
5. **Test Fallback Behavior**: Test that fallback behavior works as expected in different scenarios.

## Limitations

1. **Complexity Estimation**: The current complexity estimation is relatively simple and may not accurately reflect the true complexity of all prompts.
2. **Model Availability**: The system assumes that backup models are more available than primary models, which may not always be the case.
3. **Error Handling**: Some errors may not be recoverable even with fallback mechanisms.
4. **Rate Limits**: The system may still hit rate limits if too many requests are made in a short period.
5. **Cost**: Using more capable models as primary models may increase costs.

## Future Enhancements

1. **Improved Complexity Estimation**: More sophisticated complexity estimation based on semantic analysis.
2. **Dynamic Model Selection**: Select models dynamically based on past performance.
3. **Cost Optimization**: Optimize model selection based on cost considerations.
4. **Parallel Requests**: Send requests to multiple models in parallel and use the first response.
5. **Response Quality Evaluation**: Evaluate the quality of responses and adjust model selection accordingly.