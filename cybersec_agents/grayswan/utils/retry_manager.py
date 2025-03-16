"""
Retry manager for Gray Swan Arena.

This module provides a retry manager that works with retry strategies to provide
a convenient way to retry operations.
"""

import logging
import os
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar, Union, cast

import agentops

from .logging_utils import setup_logging
from .retry_utils import RetryStrategy, ExponentialBackoffRetryStrategy

# Set up logging
logger = setup_logging("retry_manager")

# Type variable for the return value of the function being retried
T = TypeVar('T')


class RetryManager:
    """
    Manager for retrying operations with a given retry strategy.
    
    This class provides methods for retrying operations and contextmanagers
    for retrying blocks of code.
    """
    
    def __init__(
        self, 
        retry_strategy: Optional[RetryStrategy] = None,
        operation_name: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """
        Initialize the retry manager.
        
        Args:
            retry_strategy: The retry strategy to use. If None, uses ExponentialBackoffRetryStrategy.
            operation_name: Name of the operation/component (used for logging)
            agent_id: ID of the agent using this retry manager (used for logging)
        """
        self.logger = logging.getLogger(__name__)
        self.retry_strategy = retry_strategy or ExponentialBackoffRetryStrategy()
        self.operation_name = operation_name
        self.agent_id = agent_id
        self.AGENTOPS_AVAILABLE = 'AGENTOPS_API_KEY' in os.environ
    
    def retry(self, operation: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Retry an operation using the retry strategy.
        
        Args:
            operation: The operation to retry
            *args: Positional arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            The result of the operation
            
        Raises:
            The last exception raised by the operation if all retries fail
        """
        operation_name = self.operation_name or operation.__name__
        
        # Log the retry attempt
        self.logger.info(f"Attempting operation '{operation_name}' with retry")
        
        try:
            # Use the retry strategy to execute the operation with retry
            result = self.retry_strategy.execute_with_retry(operation, *args, **kwargs)
            
            # Log successful execution
            self.logger.info(f"Operation '{operation_name}' completed successfully")
            
            return result
            
        except Exception as e:
            # Log the final failure
            self.logger.error(f"Operation '{operation_name}' failed after all retries: {str(e)}")
            
            # Record failure with AgentOps if available
            try:
                if self.AGENTOPS_AVAILABLE:
                    event_data = {
                        "operation": operation_name,
                        "error": str(e),
                        "agent_id": self.agent_id
                    }
                    
                    agentops.record(agentops.ActionEvent(
                        "operation_retry_failed",
                        event_data
                    ))
            except Exception as log_error:
                self.logger.warning(f"Failed to record operation failure with AgentOps: {str(log_error)}")
            
            # Re-raise the exception
            raise
    
    @contextmanager
    def retry_context(self, operation_name: Optional[str] = None):
        """
        Context manager for retrying a block of code.
        
        This context manager will retry the block of code if an exception is raised,
        using the retry strategy.
        
        Args:
            operation_name: Name of the operation (used for logging)
            
        Yields:
            None
            
        Raises:
            The last exception raised by the block if all retries fail
        """
        # Get the operation name (use provided name or fallback to class attribute)
        operation_name = operation_name or self.operation_name or "unknown_operation"
        
        # Create a function that will execute the context body
        def execute_context():
            # This is a placeholder - the actual execution happens in the try/except block below
            # The yield statement will transfer control back to the with block
            pass
        
        attempt = 0
        max_retries = self.retry_strategy.max_retries
        last_exception = None
        
        # Record retry start with AgentOps if available
        start_time = time.time()
        try:
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "operation_retry_started",
                    {
                        "operation": operation_name,
                        "max_retries": max_retries,
                        "agent_id": self.agent_id
                    }
                ))
        except Exception as e:
            self.logger.warning(f"Failed to record retry start with AgentOps: {str(e)}")
        
        while True:
            try:
                # Log the attempt
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt} for operation '{operation_name}'")
                
                # Execute the context body
                yield
                
                # If we get here, the context body executed successfully
                if attempt > 0:
                    self.logger.info(f"Operation '{operation_name}' succeeded after {attempt+1} attempts")
                    
                    # Record retry success with AgentOps if available
                    try:
                        if self.AGENTOPS_AVAILABLE:
                            agentops.record(agentops.ActionEvent(
                                "operation_retry_success",
                                {
                                    "operation": operation_name,
                                    "attempts": attempt + 1,
                                    "duration": time.time() - start_time,
                                    "agent_id": self.agent_id
                                }
                            ))
                    except Exception as e:
                        self.logger.warning(f"Failed to record retry success with AgentOps: {str(e)}")
                
                # Break out of the retry loop
                break
                
            except Exception as e:
                # Store the exception
                last_exception = e
                
                # Increment the attempt counter
                attempt += 1
                
                # Check if we should retry
                if not self.retry_strategy.should_retry(e, attempt - 1):
                    self.logger.warning(
                        f"Retry limit reached or exception not retryable for operation '{operation_name}': {str(e)}"
                    )
                    
                    # Record retry failure with AgentOps if available
                    try:
                        if self.AGENTOPS_AVAILABLE:
                            agentops.record(agentops.ActionEvent(
                                "operation_retry_failure",
                                {
                                    "operation": operation_name,
                                    "attempts": attempt,
                                    "error": str(e),
                                    "duration": time.time() - start_time,
                                    "agent_id": self.agent_id
                                }
                            ))
                    except Exception as log_error:
                        self.logger.warning(f"Failed to record retry failure with AgentOps: {str(log_error)}")
                    
                    # Re-raise the last exception
                    raise last_exception
                
                # Calculate the delay
                delay = self.retry_strategy.get_delay(attempt - 1)
                
                # Log the retry attempt
                self.logger.info(
                    f"Attempt {attempt} for operation '{operation_name}' failed with error: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                # Record retry attempt with AgentOps if available
                try:
                    if self.AGENTOPS_AVAILABLE:
                        agentops.record(agentops.ActionEvent(
                            "operation_retry_attempt",
                            {
                                "operation": operation_name,
                                "attempt": attempt,
                                "error": str(e),
                                "delay": delay,
                                "agent_id": self.agent_id
                            }
                        ))
                except Exception as log_error:
                    self.logger.warning(f"Failed to record retry attempt with AgentOps: {str(log_error)}")
                
                # Wait before the next attempt
                time.sleep(delay)
    
    def retry_method(self, operation_name: Optional[str] = None):
        """
        Decorator for retrying a method.
        
        This decorator will retry the method if an exception is raised,
        using the retry strategy.
        
        Args:
            operation_name: Name of the operation (used for logging)
            
        Returns:
            Decorated method
        """
        def decorator(method):
            @wraps(method)
            def wrapper(instance, *args, **kwargs):
                # Use the provided operation name or fallback to method name
                op_name = operation_name or method.__name__
                
                # Get self.agent_id from the instance if available
                agent_id = getattr(instance, 'agent_id', self.agent_id)
                
                # Create temporary retry manager with instance-specific properties
                temp_retry_manager = RetryManager(
                    retry_strategy=self.retry_strategy,
                    operation_name=op_name,
                    agent_id=agent_id
                )
                
                # Retry the method
                return temp_retry_manager.retry(method, instance, *args, **kwargs)
            
            return wrapper
        return decorator 