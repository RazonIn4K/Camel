#!/usr/bin/env python3
"""
Example script demonstrating the use of retry strategies and dead-letter queue.

This script shows how to use the retry strategies and dead-letter queue
functionality in the Gray Swan Arena system.
"""

import logging
import os
import random
import time
from typing import Dict, Any, Optional

# Set up the Python path to include the project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cybersec_agents.grayswan.camel_integration import (
    AgentFactory, 
    CommunicationChannel,
    ModelConfigManager
)
from cybersec_agents.grayswan.utils.retry_utils import (
    ExponentialBackoffRetryStrategy,
    CircuitBreakerRetryStrategy,
    with_retry
)
from cybersec_agents.grayswan.utils.message_handling import (
    DeadLetterQueue,
    MessageProcessor
)
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("retry_example", log_file="./examples/logs/retry_example.log")

def create_unreliable_service(failure_rate: float = 0.5):
    """
    Create a service that fails randomly with the given failure rate.
    
    Args:
        failure_rate: Probability of failure (0.0 to 1.0)
        
    Returns:
        A function that simulates an unreliable service
    """
    def unreliable_service(message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message with a chance of failure.
        
        Args:
            message: The message to process
            
        Returns:
            The processed message
            
        Raises:
            ValueError: Randomly based on failure_rate
        """
        if random.random() < failure_rate:
            logger.warning(f"Service failed to process message: {message}")
            raise ValueError("Service temporarily unavailable")
        
        # Simulate processing delay
        time.sleep(0.1)
        
        # Return processed message
        return {
            "status": "processed",
            "original_message": message,
            "timestamp": time.time()
        }
    
    return unreliable_service

def main():
    """
    Main function demonstrating retry strategies and dead-letter queue.
    """
    logger.info("Starting retry and dead-letter queue example")
    
    # Create a model config manager
    model_manager = ModelConfigManager()
    
    # Create an agent factory
    agent_factory = AgentFactory(model_manager)
    
    # Create a communication channel with a dead-letter queue
    channel = CommunicationChannel(
        dlq_storage_path="./examples/data/dead_letter_queue.json",
        max_dlq_size=100
    )
    
    # Get the dead-letter queue
    dlq = channel.get_dead_letter_queue()
    
    # Create an unreliable service with a 70% failure rate
    unreliable_service = create_unreliable_service(0.7)
    
    # Example 1: Using retry with exponential backoff
    logger.info("Example 1: Using retry with exponential backoff")
    
    # Create a retry strategy
    retry_strategy = ExponentialBackoffRetryStrategy(
        initial_delay=0.5,
        max_delay=5.0,
        backoff_factor=2.0,
        jitter=True,
        max_retries=3
    )
    
    # Create a message processor with the dead-letter queue
    message_processor = MessageProcessor(dlq)
    
    # Process messages with retry
    for i in range(5):
        message = {
            "id": f"msg_{i}",
            "content": f"Test message {i}",
            "timestamp": time.time()
        }
        
        logger.info(f"Processing message {i} with exponential backoff retry")
        result = message_processor.process_with_dlq(
            unreliable_service,
            message,
            sender_id="example_sender",
            receiver_id="example_receiver",
            retry_strategy=retry_strategy
        )
        
        if result:
            logger.info(f"Message {i} processed successfully: {result}")
        else:
            logger.warning(f"Message {i} processing failed and was added to DLQ")
    
    # Example 2: Using circuit breaker
    logger.info("\nExample 2: Using circuit breaker")
    
    # Create a circuit breaker strategy
    circuit_strategy = CircuitBreakerRetryStrategy(
        service_name="example_service",
        failure_threshold=3,
        reset_timeout=5.0,
        half_open_max_calls=1,
        max_retries=2
    )
    
    # Process messages with circuit breaker
    for i in range(10):
        message = {
            "id": f"circuit_msg_{i}",
            "content": f"Circuit breaker test message {i}",
            "timestamp": time.time()
        }
        
        logger.info(f"Processing message {i} with circuit breaker")
        try:
            result = message_processor.process_with_dlq(
                unreliable_service,
                message,
                sender_id="circuit_sender",
                receiver_id="circuit_receiver",
                retry_strategy=circuit_strategy
            )
            
            if result:
                logger.info(f"Message {i} processed successfully with circuit breaker: {result}")
            else:
                logger.warning(f"Message {i} processing failed and was added to DLQ")
        except Exception as e:
            logger.error(f"Circuit breaker error: {str(e)}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Example 3: Using the with_retry decorator
    logger.info("\nExample 3: Using the with_retry decorator")
    
    @with_retry(max_retries=3, retry_exceptions={ValueError})
    def process_with_decorator(message):
        return unreliable_service(message)
    
    for i in range(5):
        message = {
            "id": f"decorator_msg_{i}",
            "content": f"Decorator test message {i}",
            "timestamp": time.time()
        }
        
        logger.info(f"Processing message {i} with decorator")
        try:
            result = process_with_decorator(message)
            logger.info(f"Message {i} processed successfully with decorator: {result}")
        except Exception as e:
            logger.error(f"Decorator retry failed: {str(e)}")
            
            # Add to DLQ manually since we're not using MessageProcessor
            dlq.add_message(
                message_content=message,
                error=e,
                sender_id="decorator_sender",
                receiver_id="decorator_receiver",
                context={"operation": "decorator_example"}
            )
    
    # Example 4: Reprocessing messages from the dead-letter queue
    logger.info("\nExample 4: Reprocessing messages from the dead-letter queue")
    
    # Get current DLQ messages
    messages = dlq.get_messages()
    logger.info(f"Dead-letter queue has {len(messages)} messages")
    
    # Create a more reliable service for reprocessing (20% failure rate)
    more_reliable_service = create_unreliable_service(0.2)
    
    # Wrapper function to adapt the service to the expected signature
    def reprocessing_wrapper(message: Dict[str, Any]) -> None:
        # The wrapper ignores the return value to match the expected signature
        more_reliable_service(message)
    
    # Reprocess messages
    processed, successful = dlq.reprocess_messages(
        process_func=reprocessing_wrapper,
        retry_strategy=ExponentialBackoffRetryStrategy(max_retries=2),
        remove_on_success=True
    )
    
    logger.info(f"Reprocessed {processed} messages, {successful} were successful")
    
    # Check remaining messages
    remaining = dlq.get_messages()
    logger.info(f"Dead-letter queue now has {len(remaining)} messages")
    
    logger.info("Retry and dead-letter queue example completed")

if __name__ == "__main__":
    main() 