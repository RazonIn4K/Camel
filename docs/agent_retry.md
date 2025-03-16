# Agent Retry Capabilities

This document provides an overview of the retry capabilities implemented in the Gray Swan Arena system to improve the resilience and reliability of agent operations.

## Overview

Agent methods often interact with external systems like language models, which can occasionally fail due to rate limits, connectivity issues, or temporary service disruptions. The retry capabilities in Gray Swan Arena provide a robust mechanism to handle these transient failures automatically, improving the overall reliability of the system.

## Key Components

### RetryStrategy

The `RetryStrategy` abstract base class defines the interface for retry strategies. Several concrete implementations are available:

- **FixedDelayRetryStrategy**: Retries operations with a constant delay between attempts.
- **ExponentialBackoffRetryStrategy**: Retries operations with exponentially increasing delays, optionally with jitter for distributed systems.
- **CircuitBreakerRetryStrategy**: Implements the circuit breaker pattern to prevent repeated calls to failing services.

### RetryManager

The `RetryManager` class provides a convenient way to use retry strategies:

1. **Method Retry**: Apply retry logic to individual methods.
2. **Context-Based Retry**: Use a context manager to retry blocks of code.
3. **Direct Retry**: Directly retry individual function calls.

## Using RetryManager

### 1. Creating a RetryManager

```python
from cybersec_agents.grayswan.utils.retry_utils import ExponentialBackoffRetryStrategy
from cybersec_agents.grayswan.utils.retry_manager import RetryManager

# Create a retry strategy
retry_strategy = ExponentialBackoffRetryStrategy(
    initial_delay=1.0,
    max_retries=3,
    backoff_factor=2.0,
    jitter=True
)

# Create a retry manager
retry_manager = RetryManager(
    retry_strategy=retry_strategy,
    operation_name="my_operation",
    agent_id="agent-123"
)
```

### 2. Using the Retry Context Manager

The retry context manager is ideal for retrying blocks of code:

```python
# Retry a block of code that might fail
with retry_manager.retry_context("fetch_data"):
    response = requests.get("https://api.example.com/data")
    data = response.json()
    process_data(data)
```

If any exception occurs within the block, the entire block will be retried according to the retry strategy.

### 3. Using the Retry Method Decorator

For class methods, you can use the retry_method decorator:

```python
class MyAgent:
    def __init__(self):
        self.retry_manager = RetryManager(
            retry_strategy=ExponentialBackoffRetryStrategy(),
            operation_name="my_agent",
            agent_id="agent-123"
        )
    
    # Apply the retry decorator to a method
    @RetryManager.retry_method(operation_name="process_data")
    def process_data(self, data):
        # This method will be retried if it fails
        result = some_external_api_call(data)
        return result
```

### 4. Using Direct Retry

For one-off operations, you can directly retry a function call:

```python
def fetch_data():
    return requests.get("https://api.example.com/data").json()

# Retry the function call
data = retry_manager.retry(fetch_data)
```

## Integration with EvaluationAgent

The `EvaluationAgent` class has been enhanced with retry capabilities:

1. **Constructor Configuration**: The `max_retries` and `initial_retry_delay` parameters control retry behavior.
2. **Context-Based Retries**: Critical operations use the retry context for robust error handling.
3. **Safe Content Access**: The agent handles response content access safely to prevent attribute errors.

Example of how the `EvaluationAgent` uses retry contexts:

```python
# Use retry context for model interaction
with self.retry_manager.retry_context("get_model_response"):
    try:
        # Model interaction code that might fail
        chat_agent = get_chat_agent(...)
        response = chat_agent.step(user_message)
        
        # Safe access to content
        content = extract_content_safely(response)
    except Exception as model_error:
        # Error handling code
        logger.error(f"Error: {model_error}")
        # Try using backup model if available
        if self.backup_model:
            # Backup model code
            pass
        else:
            raise model_error
```

## Monitoring and Observability

The retry system integrates with AgentOps for monitoring and observability:

1. **Retry Attempts**: Each retry attempt is logged with AgentOps for tracking.
2. **Success/Failure Metrics**: Successful and failed retries are recorded.
3. **Performance Impact**: Delays and timing information is captured.

This monitoring allows you to:
- Track which operations require the most retries
- Identify patterns in transient failures
- Optimize retry strategies based on real-world performance

## Best Practices

1. **Choose Appropriate Retry Strategies**: Different operations may require different strategies:
   - Use exponential backoff with jitter for external API calls
   - Use circuit breaker for services with extended outages
   - Use fixed delay for simple internal operations

2. **Set Reasonable Retry Limits**: Too few retries may not handle transient issues, while too many might waste resources on truly unavailable services.

3. **Use Contextual Logging**: Include operation names and attempt counts in logs to make troubleshooting easier.

4. **Handle Non-Retryable Errors**: Some errors should not be retried (e.g., authentication failures, invalid inputs). Consider filtering these in your retry strategy.

5. **Combine with Dead Letter Queue**: For asynchronous operations, consider using the Dead Letter Queue alongside retries for operations that fail despite multiple attempts.

## Example: Complete Retry Implementation

```python
from cybersec_agents.grayswan.utils.retry_utils import ExponentialBackoffRetryStrategy
from cybersec_agents.grayswan.utils.retry_manager import RetryManager

# Create a retry strategy
retry_strategy = ExponentialBackoffRetryStrategy(
    initial_delay=1.0,
    max_retries=3,
    backoff_factor=2.0,
    jitter=True
)

# Create a retry manager
retry_manager = RetryManager(
    retry_strategy=retry_strategy,
    operation_name="data_processing",
    agent_id="processor-agent-1"
)

def process_data(data):
    """Process data with retry capabilities."""
    try:
        # Use retry context for the entire processing pipeline
        with retry_manager.retry_context("process_data"):
            # Step 1: Fetch additional data (might fail)
            additional_data = fetch_additional_data(data["id"])
            
            # Step 2: Combine data (rarely fails)
            combined_data = combine_data(data, additional_data)
            
            # Step 3: Store result (might fail)
            with retry_manager.retry_context("store_result"):
                store_result(combined_data)
            
            return {"status": "success", "result": combined_data}
            
    except Exception as e:
        logger.error(f"Data processing failed after all retries: {e}")
        return {"status": "error", "message": str(e)}
```

This example demonstrates nested retry contexts with different operational names for fine-grained control and monitoring.

## Conclusion

The retry capabilities in Gray Swan Arena provide a robust framework for handling transient failures in agent operations. By properly configuring and using these capabilities, you can significantly improve the reliability and resilience of your agent-based systems. 