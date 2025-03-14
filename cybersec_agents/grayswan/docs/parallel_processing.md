# Parallel Processing in Gray Swan Arena

This document provides an overview of the parallel processing capabilities in Gray Swan Arena, including the architecture, key components, usage examples, and best practices.

## Overview

The parallel processing system in Gray Swan Arena enables efficient execution of tasks in parallel with features such as:

- **Rate Limiting**: Control the frequency of requests to avoid rate limits
- **Batching**: Process items in batches to manage resource usage
- **Retry Mechanism**: Automatically retry failed tasks with exponential backoff
- **Error Handling**: Robust error handling with detailed metrics
- **Metrics Collection**: Track performance and success rates
- **Flexible Execution**: Support for both thread-based and process-based parallelism

## Architecture

The parallel processing system consists of several key components:

### RateLimiter

The `RateLimiter` class implements a token bucket algorithm to control the rate of requests. It ensures that requests are made at an appropriate frequency to avoid rate limits from external APIs.

Features:
- Token-based rate limiting
- Burst handling
- Configurable jitter for avoiding thundering herd problems

### TaskManager

The `TaskManager` class is the core component that manages parallel task execution. It combines rate limiting, retry logic, and error handling to provide a robust framework for parallel processing.

Features:
- Concurrent task execution with configurable parallelism
- Rate limiting via the `RateLimiter`
- Automatic retries with exponential backoff
- Detailed metrics collection
- Support for both thread-based and process-based execution
- Batching capabilities

### Utility Functions

The module provides several utility functions for common parallel processing patterns:

- `run_parallel_tasks`: Run tasks in parallel with a list of items
- `run_parallel_tasks_with_kwargs`: Run tasks in parallel with different kwargs for each item
- `run_parallel_sync`: Synchronous wrapper for `run_parallel_tasks`
- `run_parallel_with_kwargs_sync`: Synchronous wrapper for `run_parallel_tasks_with_kwargs`
- `parallel_map`: Decorator for parallel mapping of a function over a list of items

### Gray Swan Arena Integration

The module includes specialized functions for parallel execution of Gray Swan Arena pipeline phases:

- `run_parallel_reconnaissance`: Run reconnaissance tasks in parallel
- `run_parallel_prompt_engineering`: Run prompt engineering tasks in parallel
- `run_parallel_exploits`: Run exploit delivery in parallel batches
- `run_parallel_evaluation`: Run evaluation tasks in parallel
- `run_full_pipeline_parallel`: Run the complete Gray Swan Arena pipeline with parallel processing
- `run_full_pipeline_parallel_sync`: Synchronous wrapper for `run_full_pipeline_parallel`

## Usage Examples

### Basic Parallel Task Execution

```python
from cybersec_agents.grayswan.utils.parallel_processing import TaskManager
import asyncio

async def main():
    # Define a task function
    async def process_item(item):
        await asyncio.sleep(0.1)  # Simulate work
        return {"item": item, "result": item * 2}
    
    # Create a task manager
    task_manager = TaskManager(
        max_workers=5,
        requests_per_minute=60,
        max_retries=3
    )
    
    # Process items in parallel
    items = list(range(10))
    results_with_metrics = await task_manager.map(process_item, items)
    
    # Extract results
    results = [r for r, _ in results_with_metrics if r is not None]
    
    # Print metrics
    print(f"Task manager metrics: {task_manager.get_metrics()}")

asyncio.run(main())
```

### Parallel Tasks with Different Arguments

```python
from cybersec_agents.grayswan.utils.parallel_processing import TaskManager
import asyncio

async def main():
    # Define a task function with multiple parameters
    async def process_with_options(item_id, multiplier=1.0, prefix=""):
        await asyncio.sleep(0.1)  # Simulate work
        return {"item_id": item_id, "result": item_id * multiplier, "prefix": prefix}
    
    # Create items with different kwargs
    items_with_kwargs = [
        {"item_id": 1, "multiplier": 2.0, "prefix": "A"},
        {"item_id": 2, "multiplier": 3.0, "prefix": "B"},
        {"item_id": 3, "multiplier": 1.5, "prefix": "C"},
    ]
    
    # Create task manager
    task_manager = TaskManager(max_workers=3)
    
    # Process items in parallel
    results_with_metrics = await task_manager.map_with_kwargs(
        process_with_options, items_with_kwargs
    )
    
    # Extract results
    results = [r for r, _ in results_with_metrics if r is not None]

asyncio.run(main())
```

### Using Convenience Functions

```python
from cybersec_agents.grayswan.utils.parallel_processing import run_parallel_sync
import time

# Define a task function
def square(x):
    time.sleep(0.1)  # Simulate work
    return x * x

# Process items in parallel
items = list(range(1, 11))
results, metrics = run_parallel_sync(
    square, items, max_workers=3, requests_per_minute=30
)

print(f"Results: {results}")
print(f"Metrics: {metrics}")
```

### Using the Decorator

```python
from cybersec_agents.grayswan.utils.parallel_processing import parallel_map
import time

# Define a function with the parallel_map decorator
@parallel_map(max_workers=3, requests_per_minute=30)
def process_data(item):
    time.sleep(0.1)  # Simulate work
    return {"id": item["id"], "value": item["value"] * 2}

# Create items to process
items = [
    {"id": 1, "value": 10},
    {"id": 2, "value": 20},
    {"id": 3, "value": 30},
]

# Process items using the decorated function
results, metrics = process_data(items)
```

### Running the Full Pipeline in Parallel

```python
from cybersec_agents.grayswan.utils.parallel_processing import run_full_pipeline_parallel_sync

# Run the full pipeline with parallel processing
results = run_full_pipeline_parallel_sync(
    target_model="GPT-3.5-Turbo",
    target_behavior="Generate harmful content",
    output_dir="./output",
    model_name="gpt-4",
    max_workers=5,
    requests_per_minute=60,
    batch_size=3,
    batch_delay=2.0,
    max_retries=3,
    include_advanced_visualizations=True,
    include_interactive_dashboard=True
)
```

## Performance Considerations

### Optimal Worker Count

The optimal number of workers depends on several factors:

- **CPU-bound tasks**: For CPU-bound tasks, a good rule of thumb is to use `number_of_cores - 1` workers.
- **I/O-bound tasks**: For I/O-bound tasks (like API calls), you can use more workers, typically 2-3 times the number of cores.
- **External API limits**: Consider the rate limits of external APIs when setting the number of workers.

### Rate Limiting

When working with external APIs, it's important to respect their rate limits:

- Set `requests_per_minute` based on the API's documented rate limits
- Use `burst_size` to control the maximum number of concurrent requests
- Enable `jitter` to avoid synchronized request patterns

### Batching

Batching can improve performance by reducing overhead:

- Use `batch_size` to control the number of items processed in each batch
- Set `batch_delay` to add a delay between batches to avoid overwhelming resources
- For large datasets, processing in batches can help manage memory usage

### Thread vs. Process

Choose between thread-based and process-based parallelism based on your workload:

- **Thread-based** (default): Better for I/O-bound tasks and when sharing data between tasks is important
- **Process-based** (`use_processes=True`): Better for CPU-bound tasks and when isolation between tasks is important

## Error Handling

The parallel processing system includes robust error handling:

- **Automatic retries**: Failed tasks are automatically retried with exponential backoff
- **Configurable retry parameters**: Control the number of retries, initial delay, and backoff factor
- **Detailed error reporting**: Each task result includes detailed error information
- **Metrics collection**: Track the number of failures and retries

## Metrics

The system collects detailed metrics about task execution:

- **tasks_submitted**: Total number of tasks submitted
- **tasks_completed**: Number of tasks completed successfully
- **tasks_failed**: Number of tasks that failed after all retries
- **retries**: Total number of retries across all tasks
- **total_time**: Total execution time across all tasks
- **avg_time_per_task**: Average time per task
- **success_rate**: Ratio of completed tasks to submitted tasks

## Best Practices

1. **Start with conservative settings**: Begin with a small number of workers and increase gradually
2. **Monitor performance**: Use the metrics to monitor performance and adjust settings
3. **Handle errors appropriately**: Check for failed tasks and handle them appropriately
4. **Use batching for large datasets**: Process large datasets in batches to manage memory usage
5. **Consider rate limits**: Respect the rate limits of external APIs
6. **Test thoroughly**: Test your parallel processing code thoroughly to ensure it handles errors correctly
7. **Use the appropriate parallelism model**: Choose between thread-based and process-based parallelism based on your workload

## Advanced Usage

### Custom Error Handling

For more advanced error handling, you can subclass `TaskManager` and override the `execute_task` method:

```python
class CustomTaskManager(TaskManager):
    async def execute_task(self, func, *args, **kwargs):
        try:
            result, metrics = await super().execute_task(func, *args, **kwargs)
            # Custom post-processing
            return result, metrics
        except Exception as e:
            # Custom error handling
            logger.error(f"Custom error handling: {str(e)}")
            raise
```

### Custom Rate Limiting

For custom rate limiting strategies, you can subclass `RateLimiter`:

```python
class CustomRateLimiter(RateLimiter):
    async def acquire(self):
        # Custom rate limiting logic
        return await super().acquire()
```

### Integration with Monitoring Systems

You can integrate the metrics with monitoring systems:

```python
# Get metrics from task manager
metrics = task_manager.get_metrics()

# Send metrics to monitoring system
send_to_monitoring_system({
    "parallel_processing.tasks_completed": metrics["tasks_completed"],
    "parallel_processing.tasks_failed": metrics["tasks_failed"],
    "parallel_processing.success_rate": metrics["success_rate"],
    "parallel_processing.avg_time_per_task": metrics["avg_time_per_task"],
})
```

## Conclusion

The parallel processing system in Gray Swan Arena provides a robust framework for executing tasks in parallel with features such as rate limiting, batching, retry mechanisms, and error handling. By using this system, you can significantly improve the performance and reliability of your Gray Swan Arena pipelines.