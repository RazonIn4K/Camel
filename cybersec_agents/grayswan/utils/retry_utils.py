"""
Retry utilities for Gray Swan Arena.

This module provides utilities for implementing various retry strategies,
which can be used to enhance reliability of agent operations.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps
from random import random
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

import agentops

from .logging_utils import setup_logging

# Set up logging
logger = setup_logging("retry_utils")

# Type variable for the return value of the function being retried
T = TypeVar('T')

class RetryStrategy(ABC):
    """
    Abstract base class for retry strategies.
    
    This class defines the interface for retry strategies, which determine
    how and when retries should be performed after failures.
    """
    
    def __init__(self, max_retries: int = 3, retry_exceptions: Optional[Set[Type[Exception]]] = None):
        """
        Initialize the retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_exceptions: Set of exception types that should trigger a retry.
                              If None, all exceptions will trigger retries.
        """
        self.max_retries = max_retries
        self.retry_exceptions = retry_exceptions or {Exception}
        self.AGENTOPS_AVAILABLE = 'AGENTOPS_API_KEY' in os.environ
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """
        Get the delay before the next retry attempt.
        
        Args:
            attempt: The current attempt number (0-based)
            
        Returns:
            The delay in seconds before the next retry
        """
        pass
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if a retry should be attempted based on the exception and attempt number.
        
        Args:
            exception: The exception that was raised
            attempt: The current attempt number (0-based)
            
        Returns:
            True if a retry should be attempted, False otherwise
        """
        # Check if we've exceeded the maximum number of retries
        if attempt >= self.max_retries:
            return False
        
        # Check if the exception is one that should trigger a retry
        return any(isinstance(exception, exc_type) for exc_type in self.retry_exceptions)
    
    def execute_with_retry(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with retry logic.
        
        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The return value of the function
            
        Raises:
            The last exception raised by the function if all retries fail
        """
        attempt = 0
        last_exception = None
        
        # Record retry start
        start_time = time.time()
        try:
            if self.AGENTOPS_AVAILABLE:
                agentops.record(agentops.ActionEvent(
                    "retry_operation_started",
                    {
                        "max_retries": self.max_retries,
                        "function": func.__name__,
                        "strategy": self.__class__.__name__,
                    }
                ))
        except Exception as e:
            self.logger.warning(f"Failed to record retry start with AgentOps: {str(e)}")
        
        while True:
            try:
                # Attempt to execute the function
                result = func(*args, **kwargs)
                
                # If successful and not the first attempt, log the success
                if attempt > 0:
                    self.logger.info(f"Operation succeeded after {attempt+1} attempts.")
                    
                    # Record retry success
                    try:
                        if self.AGENTOPS_AVAILABLE:
                            agentops.record(agentops.ActionEvent(
                                "retry_operation_success",
                                {
                                    "attempts": attempt + 1,
                                    "function": func.__name__,
                                    "duration": time.time() - start_time,
                                }
                            ))
                    except Exception as e:
                        self.logger.warning(f"Failed to record retry success with AgentOps: {str(e)}")
                
                return result
                
            except Exception as e:
                # Store the exception
                last_exception = e
                
                # Increment the attempt counter
                attempt += 1
                
                # Determine if we should retry
                if not self.should_retry(e, attempt - 1):
                    self.logger.warning(f"Retry limit reached or exception not retryable: {str(e)}")
                    
                    # Record retry failure
                    try:
                        if self.AGENTOPS_AVAILABLE:
                            agentops.record(agentops.ActionEvent(
                                "retry_operation_failure",
                                {
                                    "attempts": attempt,
                                    "function": func.__name__,
                                    "error": str(e),
                                    "duration": time.time() - start_time,
                                }
                            ))
                    except Exception as log_error:
                        self.logger.warning(f"Failed to record retry failure with AgentOps: {str(log_error)}")
                    
                    # Re-raise the last exception
                    raise last_exception
                
                # Log the retry attempt
                delay = self.get_delay(attempt - 1)
                self.logger.info(f"Attempt {attempt} failed with error: {str(e)}. Retrying in {delay:.2f} seconds...")
                
                # Record retry attempt
                try:
                    if self.AGENTOPS_AVAILABLE:
                        agentops.record(agentops.ActionEvent(
                            "retry_attempt",
                            {
                                "attempt": attempt,
                                "function": func.__name__,
                                "error": str(e),
                                "delay": delay,
                            }
                        ))
                except Exception as log_error:
                    self.logger.warning(f"Failed to record retry attempt with AgentOps: {str(log_error)}")
                
                # Wait before the next attempt
                time.sleep(delay)


class FixedDelayRetryStrategy(RetryStrategy):
    """
    A retry strategy with a fixed delay between attempts.
    """
    
    def __init__(self, delay: float = 1.0, max_retries: int = 3, retry_exceptions: Optional[Set[Type[Exception]]] = None):
        """
        Initialize the fixed delay retry strategy.
        
        Args:
            delay: The fixed delay in seconds between retry attempts
            max_retries: Maximum number of retry attempts
            retry_exceptions: Set of exception types that should trigger a retry
        """
        super().__init__(max_retries, retry_exceptions)
        self.delay = delay
    
    def get_delay(self, attempt: int) -> float:
        """
        Get the fixed delay.
        
        Args:
            attempt: The current attempt number (ignored for fixed delay)
            
        Returns:
            The fixed delay in seconds
        """
        return self.delay


class ExponentialBackoffRetryStrategy(RetryStrategy):
    """
    A retry strategy with exponential backoff between attempts.
    
    The delay between attempts increases exponentially with each attempt,
    optionally with jitter to prevent synchronized retries in distributed systems.
    """
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        max_retries: int = 3,
        retry_exceptions: Optional[Set[Type[Exception]]] = None
    ):
        """
        Initialize the exponential backoff retry strategy.
        
        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Factor by which the delay increases with each attempt
            jitter: Whether to add randomness to the delay
            max_retries: Maximum number of retry attempts
            retry_exceptions: Set of exception types that should trigger a retry
        """
        super().__init__(max_retries, retry_exceptions)
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """
        Get the exponentially increasing delay.
        
        Args:
            attempt: The current attempt number (0-based)
            
        Returns:
            The delay in seconds before the next retry
        """
        # Calculate the base delay (without jitter)
        delay = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)
        
        # Add jitter if enabled (random value between 0.5 and 1.5 times the base delay)
        if self.jitter:
            jitter_factor = 0.5 + random()  # Random value between 0.5 and 1.5
            delay *= jitter_factor
        
        return delay


class CircuitBreakerRetryStrategy(RetryStrategy):
    """
    A retry strategy that implements the Circuit Breaker pattern.
    
    The circuit breaker prevents repeated retries when a system is failing
    consistently, helping to prevent cascading failures and allowing the
    system time to recover.
    """
    
    # Class-level tracking of circuit state for different services
    _circuits: Dict[str, Dict[str, Any]] = {}
    
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        base_retry_strategy: Optional[RetryStrategy] = None,
        max_retries: int = 3,
        retry_exceptions: Optional[Set[Type[Exception]]] = None
    ):
        """
        Initialize the circuit breaker retry strategy.
        
        Args:
            service_name: Name of the service being called (used to track circuit state)
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Time in seconds before attempting to close the circuit
            half_open_max_calls: Maximum number of calls allowed when half-open
            base_retry_strategy: The retry strategy to use when the circuit is closed
            max_retries: Maximum number of retry attempts
            retry_exceptions: Set of exception types that should trigger a retry
        """
        super().__init__(max_retries, retry_exceptions)
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.base_retry_strategy = base_retry_strategy or ExponentialBackoffRetryStrategy(max_retries=max_retries)
        
        # Initialize the circuit state if it doesn't exist
        if service_name not in CircuitBreakerRetryStrategy._circuits:
            CircuitBreakerRetryStrategy._circuits[service_name] = {
                'state': 'CLOSED',
                'failures': 0,
                'last_failure_time': 0,
                'half_open_calls': 0,
            }
    
    def get_delay(self, attempt: int) -> float:
        """
        Get the delay before the next retry attempt.
        
        Args:
            attempt: The current attempt number (0-based)
            
        Returns:
            The delay in seconds before the next retry
        """
        # Use the base retry strategy for delay calculation
        return self.base_retry_strategy.get_delay(attempt)
    
    def _get_circuit_state(self) -> Dict[str, Any]:
        """
        Get the current state of the circuit.
        
        Returns:
            Dictionary containing the circuit state
        """
        return CircuitBreakerRetryStrategy._circuits[self.service_name]
    
    def _update_circuit_state(self, state: Optional[str] = None, failures: Optional[int] = None) -> None:
        """
        Update the state of the circuit.
        
        Args:
            state: New circuit state (if None, state is not updated)
            failures: New failure count (if None, failures is not updated)
        """
        circuit = CircuitBreakerRetryStrategy._circuits[self.service_name]
        
        if state is not None:
            circuit['state'] = state
            
            if state == 'OPEN':
                circuit['last_failure_time'] = time.time()
                circuit['half_open_calls'] = 0
                
                # Log circuit open
                self.logger.warning(f"Circuit OPEN for service {self.service_name}")
                
                # Record circuit state change with AgentOps if available
                try:
                    if self.AGENTOPS_AVAILABLE:
                        agentops.record(agentops.ActionEvent(
                            "circuit_state_change",
                            {
                                "service": self.service_name,
                                "state": "OPEN",
                                "failures": circuit['failures'],
                            }
                        ))
                except Exception as e:
                    self.logger.warning(f"Failed to record circuit state change with AgentOps: {str(e)}")
            
            elif state == 'HALF_OPEN':
                circuit['half_open_calls'] = 0
                
                # Log circuit half-open
                self.logger.info(f"Circuit HALF_OPEN for service {self.service_name}")
                
                # Record circuit state change with AgentOps if available
                try:
                    if self.AGENTOPS_AVAILABLE:
                        agentops.record(agentops.ActionEvent(
                            "circuit_state_change",
                            {
                                "service": self.service_name,
                                "state": "HALF_OPEN",
                            }
                        ))
                except Exception as e:
                    self.logger.warning(f"Failed to record circuit state change with AgentOps: {str(e)}")
            
            elif state == 'CLOSED':
                # Log circuit closed
                self.logger.info(f"Circuit CLOSED for service {self.service_name}")
                
                # Record circuit state change with AgentOps if available
                try:
                    if self.AGENTOPS_AVAILABLE:
                        agentops.record(agentops.ActionEvent(
                            "circuit_state_change",
                            {
                                "service": self.service_name,
                                "state": "CLOSED",
                            }
                        ))
                except Exception as e:
                    self.logger.warning(f"Failed to record circuit state change with AgentOps: {str(e)}")
        
        if failures is not None:
            circuit['failures'] = failures
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if a retry should be attempted based on the circuit state.
        
        Args:
            exception: The exception that was raised
            attempt: The current attempt number (0-based)
            
        Returns:
            True if a retry should be attempted, False otherwise
        """
        # Get the current circuit state
        circuit = self._get_circuit_state()
        
        # Check if the circuit is OPEN
        if circuit['state'] == 'OPEN':
            # Check if the reset timeout has elapsed
            if time.time() - circuit['last_failure_time'] > self.reset_timeout:
                # Transition to HALF_OPEN
                self._update_circuit_state(state='HALF_OPEN')
            else:
                # Circuit is OPEN and timeout hasn't elapsed, don't retry
                self.logger.info(f"Circuit is OPEN for service {self.service_name}, not retrying.")
                return False
        
        # Check if the circuit is HALF_OPEN
        if circuit['state'] == 'HALF_OPEN':
            # Only allow a limited number of calls in HALF_OPEN state
            if circuit['half_open_calls'] >= self.half_open_max_calls:
                self.logger.info(f"Maximum calls reached for HALF_OPEN circuit for service {self.service_name}, not retrying.")
                return False
            
            # Increment the number of calls in HALF_OPEN state
            circuit['half_open_calls'] += 1
        
        # Check if we should retry based on the base retry strategy
        return super().should_retry(exception, attempt)
    
    def execute_with_retry(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with circuit breaker and retry logic.
        
        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The return value of the function
            
        Raises:
            The last exception raised by the function if all retries fail,
            or CircuitOpenError if the circuit is open
        """
        # Get the current circuit state
        circuit = self._get_circuit_state()
        
        # Check if the circuit is OPEN
        if circuit['state'] == 'OPEN':
            # Check if the reset timeout has elapsed
            if time.time() - circuit['last_failure_time'] > self.reset_timeout:
                # Transition to HALF_OPEN
                self._update_circuit_state(state='HALF_OPEN')
            else:
                # Circuit is OPEN and timeout hasn't elapsed, raise an error
                error_msg = f"Circuit is OPEN for service {self.service_name}."
                self.logger.info(error_msg)
                
                # Record circuit rejection with AgentOps if available
                try:
                    if self.AGENTOPS_AVAILABLE:
                        agentops.record(agentops.ActionEvent(
                            "circuit_rejection",
                            {
                                "service": self.service_name,
                                "function": func.__name__,
                            }
                        ))
                except Exception as e:
                    self.logger.warning(f"Failed to record circuit rejection with AgentOps: {str(e)}")
                
                raise CircuitOpenError(error_msg)
        
        try:
            # Execute with retry using the parent class method
            result = super().execute_with_retry(func, *args, **kwargs)
            
            # If successful and the circuit is HALF_OPEN, close it
            if circuit['state'] == 'HALF_OPEN':
                self._update_circuit_state(state='CLOSED', failures=0)
            
            # Reset failure count if in CLOSED state
            if circuit['state'] == 'CLOSED':
                self._update_circuit_state(failures=0)
            
            return result
            
        except Exception as e:
            # Increment the failure count
            circuit['failures'] += 1
            
            # If we've reached the failure threshold, open the circuit
            if circuit['failures'] >= self.failure_threshold:
                self._update_circuit_state(state='OPEN')
            
            # Re-raise the exception
            raise


class CircuitOpenError(Exception):
    """
    Exception raised when a circuit is open and a call is attempted.
    """
    pass


def with_retry(
    retry_strategy: Optional[RetryStrategy] = None,
    max_retries: int = 3,
    retry_exceptions: Optional[Set[Type[Exception]]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add retry logic to a function.
    
    Args:
        retry_strategy: The retry strategy to use (if None, uses ExponentialBackoffRetryStrategy)
        max_retries: Maximum number of retry attempts (used if retry_strategy is None)
        retry_exceptions: Set of exception types that should trigger a retry (used if retry_strategy is None)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Use the provided retry strategy or create a default one
            strategy = retry_strategy or ExponentialBackoffRetryStrategy(
                max_retries=max_retries,
                retry_exceptions=retry_exceptions
            )
            
            # Execute the function with retry logic
            return strategy.execute_with_retry(func, *args, **kwargs)
        
        return wrapper
    
    return decorator 