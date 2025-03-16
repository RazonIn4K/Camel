# Gray Swan Arena Utilities

This directory contains utility modules for the Gray Swan Arena system, providing functionality for reliable message handling, retry strategies, logging, and more.

## Message Handling

The `message_handling.py` module provides utilities for reliable message handling, including a Dead Letter Queue (DLQ) implementation to manage failed messages.

### DeadLetterQueue

The `DeadLetterQueue` class stores messages that could not be processed successfully, allowing for later inspection, analysis, and potential reprocessing.

```python
from cybersec_agents.grayswan.utils.message_handling import DeadLetterQueue

# Create a DLQ with in-memory storage
dlq = DeadLetterQueue(max_size=1000)

# Create a DLQ with persistent storage
dlq = DeadLetterQueue(
    max_size=1000,
    persistent_storage_path="./data/dead_letter_queue.json"
)

# Add a failed message to the queue
dlq.add_message(
    message_content={"type": "example", "content": "Message content"},
    error=ValueError("Processing failed"),
    sender_id="sender_agent",
    receiver_id="receiver_agent",
    context={"operation": "process"}
)

# Get messages from the queue
messages = dlq.get_messages()

# Get messages with filtering
failed_messages = dlq.get_messages(
    error_type="ValueError",
    sender_id="sender_agent",
    time_range=(start_time, end_time)
)

# Reprocess messages
processed, successful = dlq.reprocess_messages(
    process_func=my_processing_function,
    retry_strategy=ExponentialBackoffRetryStrategy(max_retries=3),
    remove_on_success=True
)

# Clear the queue
count = dlq.clear()
```

### MessageProcessor

The `MessageProcessor` class provides a wrapper around message processing functions to handle errors and automatically add failed messages to a Dead Letter Queue.

```python
from cybersec_agents.grayswan.utils.message_handling import MessageProcessor

# Create a message processor with a DLQ
processor = MessageProcessor(dlq)

# Process a message with DLQ error handling
result = processor.process_with_dlq(
    process_func=my_processing_function,
    message={"type": "example", "content": "Message content"},
    sender_id="sender_agent",
    receiver_id="receiver_agent",
    retry_strategy=ExponentialBackoffRetryStrategy(max_retries=3)
)
```

## Retry Strategies

The `retry_utils.py` module provides utilities for implementing various retry strategies, which can be used to enhance reliability of agent operations.

### RetryStrategy

The `RetryStrategy` abstract base class defines the interface for retry strategies, which determine how and when retries should be performed after failures.

```python
from cybersec_agents.grayswan.utils.retry_utils import RetryStrategy

# This is an abstract class, use one of the concrete implementations
```

### FixedDelayRetryStrategy

The `FixedDelayRetryStrategy` class implements a retry strategy with a fixed delay between attempts.

```python
from cybersec_agents.grayswan.utils.retry_utils import FixedDelayRetryStrategy

# Create a fixed delay retry strategy
strategy = FixedDelayRetryStrategy(
    delay=1.0,
    max_retries=3,
    retry_exceptions={ValueError, ConnectionError}
)

# Execute a function with retry
result = strategy.execute_with_retry(my_function, *args, **kwargs)
```

### ExponentialBackoffRetryStrategy

The `ExponentialBackoffRetryStrategy` class implements a retry strategy with exponential backoff between attempts. The delay between attempts increases exponentially with each attempt, optionally with jitter to prevent synchronized retries in distributed systems.

```python
from cybersec_agents.grayswan.utils.retry_utils import ExponentialBackoffRetryStrategy

# Create an exponential backoff retry strategy
strategy = ExponentialBackoffRetryStrategy(
    initial_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=True,
    max_retries=3,
    retry_exceptions={ValueError, ConnectionError}
)

# Execute a function with retry
result = strategy.execute_with_retry(my_function, *args, **kwargs)
```

### CircuitBreakerRetryStrategy

The `CircuitBreakerRetryStrategy` class implements a retry strategy that implements the Circuit Breaker pattern. The circuit breaker prevents repeated retries when a system is failing consistently, helping to prevent cascading failures and allowing the system time to recover.

```python
from cybersec_agents.grayswan.utils.retry_utils import CircuitBreakerRetryStrategy

# Create a circuit breaker retry strategy
strategy = CircuitBreakerRetryStrategy(
    service_name="my_service",
    failure_threshold=5,
    reset_timeout=60.0,
    half_open_max_calls=1,
    base_retry_strategy=ExponentialBackoffRetryStrategy(max_retries=3),
    max_retries=3,
    retry_exceptions={ValueError, ConnectionError}
)

# Execute a function with retry
result = strategy.execute_with_retry(my_function, *args, **kwargs)
```

### with_retry Decorator

The `with_retry` decorator adds retry logic to a function.

```python
from cybersec_agents.grayswan.utils.retry_utils import with_retry

# Decorate a function with retry logic
@with_retry(max_retries=3, retry_exceptions={ValueError, ConnectionError})
def my_function(arg1, arg2):
    # Function implementation
    pass

# Use a specific retry strategy
@with_retry(retry_strategy=ExponentialBackoffRetryStrategy(max_retries=3))
def another_function(arg1, arg2):
    # Function implementation
    pass
```

## Logging

The `logging_utils.py` module provides utilities for setting up and configuring logging throughout the Gray Swan system.

### setup_logging

The `setup_logging` function sets up a logger with the specified configuration.

```python
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up a logger with console output only
logger = setup_logging("my_logger")

# Set up a logger with file output
logger = setup_logging(
    logger_name="my_logger",
    log_level=logging.DEBUG,
    log_file="./logs/my_log.log"
)

# Use the logger
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
```

## Integration with CommunicationChannel

The `CommunicationChannel` class in the `camel_integration.py` module integrates with the Dead Letter Queue to handle failed messages.

```python
from cybersec_agents.grayswan.camel_integration import CommunicationChannel

# Create a communication channel with a dead-letter queue
channel = CommunicationChannel(
    dlq_storage_path="./data/dead_letter_queue.json",
    max_dlq_size=1000
)

# Send a message
channel.send_message(
    message={"type": "example", "content": "Message content"},
    sender_id="sender_agent",
    receiver_id="receiver_agent"
)

# Receive a message
message = channel.receive_message(receiver_id="receiver_agent")

# Get the dead-letter queue
dlq = channel.get_dead_letter_queue()

# Reprocess failed messages
results = channel.reprocess_failed_messages()

# Clear the dead-letter queue
count = channel.clear_dead_letter_queue()
```

## Example Usage

See the `examples/retry_and_dlq_example.py` script for a complete example of how to use the retry strategies and dead-letter queue functionality. 