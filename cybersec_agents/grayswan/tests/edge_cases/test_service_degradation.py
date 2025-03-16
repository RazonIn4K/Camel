"""
Service Degradation Edge Case Tests for Gray Swan Arena.

This module contains tests that simulate various service degradation scenarios
such as slow responses, timeouts, and partial system failures to verify
the system's resilience under degraded service conditions.
"""

import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from ...camel_integration import AgentFactory, TestTier
from ...utils.logging_utils import setup_logging
from ...utils.retry_utils import RetryStrategy, FixedDelayRetryStrategy
from .edge_case_framework import FailureSimulator, EdgeCaseTestRunner

# Set up logging
logger = setup_logging("service_degradation_tests")


def test_slow_model_responses(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of slow model responses.
    
    This test simulates slow responses from LLM models and verifies that
    the system properly handles these delays without timing out prematurely.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to slow model responses...")
    
    # Create an agent to test with
    test_agent = agent_factory.create_evaluation_agent()
    
    # Define test parameters
    delay_seconds = 2  # Simulate a 2-second delay
    timeout_seconds = 10
    num_requests = 3
    
    # Track results
    successful_requests = 0
    failed_requests = 0
    response_times = []
    errors = []
    
    # Create a patched version of the step method with delay
    original_step = None
    if hasattr(test_agent, 'chat_agent') and hasattr(test_agent.chat_agent, 'step'):
        original_step = test_agent.chat_agent.step
        
        def delayed_step(*args, **kwargs):
            # Simulate a delay
            time.sleep(delay_seconds)
            # Call the original method
            return original_step(*args, **kwargs)
        
        # Apply the patch
        test_agent.chat_agent.step = delayed_step
    
    # Run multiple requests to test the slow responses
    for i in range(num_requests):
        logger.info(f"Executing slow response test request {i+1}/{num_requests}")
        
        start_time = time.time()
        response = None
        error = None
        
        try:
            # Use a simple query
            response = test_agent.get_evaluation(
                f"Test query with slow response {i+1}",
                timeout=timeout_seconds
            )
            successful_requests += 1
        except Exception as e:
            error = str(e)
            errors.append(error)
            failed_requests += 1
            logger.error(f"Error during slow response test: {error}")
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        response_times.append(response_time)
        
        logger.info(f"Request {i+1} completed in {response_time:.2f}s")
    
    # Restore the original step method if it was patched
    if original_step is not None:
        test_agent.chat_agent.step = original_step
    
    # Calculate average response time
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Analyze results
    success_rate = successful_requests / num_requests if num_requests > 0 else 0
    
    logger.info(f"Slow response test results: {successful_requests}/{num_requests} successful requests")
    logger.info(f"Average response time: {avg_response_time:.2f}s")
    
    return {
        "requests_attempted": num_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "average_response_time": avg_response_time,
        "simulated_delay": delay_seconds,
        "timeout": timeout_seconds,
        "response_times": response_times,
        "errors": errors,
        "success_rate": success_rate,
        "message": f"System handled slow responses with {success_rate*100:.1f}% success rate"
    }


def test_intermittent_timeouts(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of intermittent timeouts.
    
    This test simulates random timeouts from the LLM service and verifies
    that the system properly retries or handles these timeouts.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to intermittent timeouts...")
    
    # Create an agent to test with
    test_agent = agent_factory.create_evaluation_agent()
    
    # Define test parameters
    timeout_probability = 0.5  # 50% chance of timeout
    num_requests = 5
    retry_attempts = 3  # Expected retry attempts from the system
    
    # Create a failure simulator
    failure_simulator = FailureSimulator()
    
    # Track results
    successful_requests = 0
    failed_requests = 0
    retry_counts = []
    response_times = []
    errors = []
    
    # Create a patched version of the step method with intermittent timeouts
    original_step = None
    if hasattr(test_agent, 'chat_agent') and hasattr(test_agent.chat_agent, 'step'):
        original_step = test_agent.chat_agent.step
        
        # Use a list to track retry counts
        retry_counter = [0] * num_requests
        
        def timeout_step(request_idx):
            def inner_step(*args, **kwargs):
                # Increment retry counter for this request
                retry_counter[request_idx] += 1
                
                # Decide whether to simulate a timeout
                if random.random() < timeout_probability and retry_counter[request_idx] <= retry_attempts:
                    logger.info(f"Simulating timeout for request {request_idx+1}, attempt {retry_counter[request_idx]}")
                    raise TimeoutError("Simulated timeout error")
                
                # Call the original method
                return original_step(*args, **kwargs)
            return inner_step
    
    # Run multiple requests to test the intermittent timeouts
    for i in range(num_requests):
        logger.info(f"Executing intermittent timeout test request {i+1}/{num_requests}")
        
        # Apply the timeout patch for this request
        if original_step is not None:
            test_agent.chat_agent.step = timeout_step(i)
        
        start_time = time.time()
        response = None
        error = None
        
        try:
            # Ensure we have a retry strategy configured
            retry_strategy = FixedDelayRetryStrategy(max_retries=retry_attempts, delay_seconds=0.5)
            
            # Use a simple query
            response = test_agent.get_evaluation(
                f"Test query with potential timeout {i+1}",
                retry_strategy=retry_strategy
            )
            successful_requests += 1
        except Exception as e:
            error = str(e)
            errors.append(error)
            failed_requests += 1
            logger.error(f"Error during intermittent timeout test: {error}")
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        response_times.append(response_time)
        
        # Record retry count
        if i < len(retry_counter):
            retry_counts.append(retry_counter[i])
        
        logger.info(f"Request {i+1} completed in {response_time:.2f}s with {retry_counts[-1] if retry_counts else 0} retries")
    
    # Restore the original step method if it was patched
    if original_step is not None:
        test_agent.chat_agent.step = original_step
    
    # Calculate average values
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    avg_retries = sum(retry_counts) / len(retry_counts) if retry_counts else 0
    
    # Analyze results
    success_rate = successful_requests / num_requests if num_requests > 0 else 0
    
    logger.info(f"Intermittent timeout test results: {successful_requests}/{num_requests} successful requests")
    logger.info(f"Average response time: {avg_response_time:.2f}s")
    logger.info(f"Average retries: {avg_retries:.1f}")
    
    return {
        "requests_attempted": num_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "average_response_time": avg_response_time,
        "average_retries": avg_retries,
        "retry_counts": retry_counts,
        "timeout_probability": timeout_probability,
        "max_retry_attempts": retry_attempts,
        "response_times": response_times,
        "errors": errors,
        "success_rate": success_rate,
        "message": f"System handled intermittent timeouts with {success_rate*100:.1f}% success rate after retries"
    }


def test_service_throttling(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of service throttling.
    
    This test simulates a service that throttles requests when too many
    are sent in a short period, and verifies that the system properly
    handles throttling by backing off appropriately.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to service throttling...")
    
    # Create an agent to test with
    test_agent = agent_factory.create_evaluation_agent()
    
    # Define test parameters
    request_limit = 3  # Throttle after 3 requests
    throttle_duration = 2  # Throttle for 2 seconds
    num_requests = 10
    retry_delay = 1  # Expected retry delay
    
    # Create a failure simulator
    failure_simulator = FailureSimulator()
    
    # Track results
    successful_requests = 0
    throttled_requests = 0
    response_times = []
    errors = []
    
    # Create a throttling mechanism
    class ServiceThrottler:
        def __init__(self, request_limit, throttle_duration):
            self.request_limit = request_limit
            self.throttle_duration = throttle_duration
            self.request_count = 0
            self.last_throttle_time = 0
            self.lock = threading.Lock()
        
        def check_throttling(self):
            with self.lock:
                # Check if we're currently in a throttling period
                if time.time() - self.last_throttle_time < self.throttle_duration:
                    return True
                
                # Increment request count
                self.request_count += 1
                
                # Check if we've hit the limit
                if self.request_count >= self.request_limit:
                    self.request_count = 0
                    self.last_throttle_time = time.time()
                    return True
                
                return False
    
    # Initialize the throttler
    throttler = ServiceThrottler(request_limit, throttle_duration)
    
    # Create a patched version of the step method with throttling
    original_step = None
    if hasattr(test_agent, 'chat_agent') and hasattr(test_agent.chat_agent, 'step'):
        original_step = test_agent.chat_agent.step
        
        def throttled_step(*args, **kwargs):
            # Check if we should throttle
            if throttler.check_throttling():
                logger.info("Simulating service throttling")
                raise Exception("Rate limit exceeded. Please reduce request frequency.")
            
            # Call the original method
            return original_step(*args, **kwargs)
        
        # Apply the patch
        test_agent.chat_agent.step = throttled_step
    
    # Run multiple requests to test the throttling
    for i in range(num_requests):
        logger.info(f"Executing throttling test request {i+1}/{num_requests}")
        
        start_time = time.time()
        response = None
        error = None
        was_throttled = False
        
        try:
            # Ensure we have a retry strategy configured
            retry_strategy = FixedDelayRetryStrategy(max_retries=5, delay_seconds=retry_delay)
            
            # Use a simple query
            response = test_agent.get_evaluation(
                f"Test query with potential throttling {i+1}",
                retry_strategy=retry_strategy
            )
            successful_requests += 1
        except Exception as e:
            error = str(e)
            if "Rate limit" in error:
                was_throttled = True
                throttled_requests += 1
            errors.append(error)
            logger.error(f"Error during throttling test: {error}")
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        response_times.append(response_time)
        
        logger.info(f"Request {i+1} completed in {response_time:.2f}s (throttled: {was_throttled})")
        
        # Add a small delay between requests to give the throttler time to update
        time.sleep(0.1)
    
    # Restore the original step method if it was patched
    if original_step is not None:
        test_agent.chat_agent.step = original_step
    
    # Calculate average response time
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Analyze results
    success_rate = successful_requests / num_requests if num_requests > 0 else 0
    throttle_rate = throttled_requests / num_requests if num_requests > 0 else 0
    
    logger.info(f"Service throttling test results: {successful_requests}/{num_requests} successful requests")
    logger.info(f"Throttled requests: {throttled_requests}")
    logger.info(f"Average response time: {avg_response_time:.2f}s")
    
    return {
        "requests_attempted": num_requests,
        "successful_requests": successful_requests,
        "throttled_requests": throttled_requests,
        "request_limit": request_limit,
        "throttle_duration": throttle_duration,
        "average_response_time": avg_response_time,
        "response_times": response_times,
        "errors": errors,
        "success_rate": success_rate,
        "throttle_rate": throttle_rate,
        "message": f"System handled service throttling with {success_rate*100:.1f}% success rate"
    }


def test_degraded_response_quality(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of degraded response quality.
    
    This test simulates degraded quality in model responses (e.g., truncated,
    malformed, or low-quality responses) and verifies that the system
    properly handles and recovers from these cases.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to degraded response quality...")
    
    # Create an agent to test with
    test_agent = agent_factory.create_evaluation_agent()
    
    # Define test parameters
    num_requests = 5
    degradation_probability = 0.6  # 60% chance of degraded response
    
    # Define a set of degradation modes
    degradation_modes = [
        "truncation",  # Truncate the response
        "repetition",  # Add repetitive content
        "irrelevance",  # Make response irrelevant to the query
        "grammar",     # Add grammatical errors
        "empty",       # Return empty or extremely short response
    ]
    
    # Track results
    successful_requests = 0
    degraded_responses = 0
    response_qualities = []
    errors = []
    
    # Create a patched version of the step method with degraded responses
    original_step = None
    if hasattr(test_agent, 'chat_agent') and hasattr(test_agent.chat_agent, 'step'):
        original_step = test_agent.chat_agent.step
        
        def degrade_response(response, mode):
            """Degrade a response based on the specified mode."""
            if not hasattr(response, 'content') or not response.content:
                return response
            
            content = response.content
            
            if mode == "truncation":
                # Truncate the response to 20-50% of its original length
                truncate_point = random.randint(int(len(content) * 0.2), int(len(content) * 0.5))
                content = content[:truncate_point] + "..."
            
            elif mode == "repetition":
                # Add repetitive content
                repeat_phrase = " This is a repeated phrase." * 5
                content = content + repeat_phrase
            
            elif mode == "irrelevance":
                # Replace with irrelevant content
                irrelevant_responses = [
                    "I'm sorry, I don't understand the question. Could you please clarify?",
                    "That's an interesting point about bananas, but I'm not sure how it relates.",
                    "Let me tell you about something completely different instead.",
                    "I prefer to discuss cloud formations rather than answer that query."
                ]
                content = random.choice(irrelevant_responses)
            
            elif mode == "grammar":
                # Add grammatical errors
                content = content.replace("is", "are").replace("are", "is")
                content = content.replace("the", "teh").replace("and", "nad")
            
            elif mode == "empty":
                # Return empty or extremely short response
                content = "" if random.random() < 0.5 else "Yes."
            
            # Update the response content
            response.content = content
            return response
        
        def degraded_step(*args, **kwargs):
            # Call the original method
            response = original_step(*args, **kwargs)
            
            # Decide whether to degrade the response
            if random.random() < degradation_probability:
                # Select a random degradation mode
                mode = random.choice(degradation_modes)
                logger.info(f"Simulating degraded response quality: {mode}")
                
                # Apply the degradation
                response = degrade_response(response, mode)
            
            return response
        
        # Apply the patch
        test_agent.chat_agent.step = degraded_step
    
    # Run multiple requests to test degraded response quality
    for i in range(num_requests):
        logger.info(f"Executing degraded quality test request {i+1}/{num_requests}")
        
        start_time = time.time()
        response = None
        error = None
        quality_score = 0
        
        try:
            # Use a simple query
            response = test_agent.get_evaluation(
                f"Test query with potentially degraded response {i+1}"
            )
            
            # Check response quality
            if response:
                content_length = len(response) if isinstance(response, str) else 0
                if hasattr(response, 'content'):
                    content_length = len(response.content) if response.content else 0
                
                # Simple heuristic for quality scoring
                if content_length < 10:
                    quality_score = 1  # Very poor
                    degraded_responses += 1
                elif content_length < 50:
                    quality_score = 3  # Poor
                    degraded_responses += 1
                elif "..." in str(response) or "teh" in str(response) or "repeated phrase" in str(response):
                    quality_score = 5  # Mediocre
                    degraded_responses += 1
                else:
                    quality_score = 8  # Good
                    successful_requests += 1
            
            response_qualities.append(quality_score)
            
        except Exception as e:
            error = str(e)
            errors.append(error)
            quality_score = 0
            degraded_responses += 1
            logger.error(f"Error during degraded quality test: {error}")
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        
        logger.info(f"Request {i+1} completed in {response_time:.2f}s with quality score {quality_score}/10")
    
    # Restore the original step method if it was patched
    if original_step is not None:
        test_agent.chat_agent.step = original_step
    
    # Calculate average quality score
    avg_quality = sum(response_qualities) / len(response_qualities) if response_qualities else 0
    
    # Analyze results
    success_rate = successful_requests / num_requests if num_requests > 0 else 0
    degradation_rate = degraded_responses / num_requests if num_requests > 0 else 0
    
    logger.info(f"Degraded quality test results: {successful_requests}/{num_requests} acceptable responses")
    logger.info(f"Degraded responses: {degraded_responses}")
    logger.info(f"Average quality score: {avg_quality:.1f}/10")
    
    return {
        "requests_attempted": num_requests,
        "successful_requests": successful_requests,
        "degraded_responses": degraded_responses,
        "average_quality_score": avg_quality,
        "degradation_probability": degradation_probability,
        "degradation_modes_used": degradation_modes,
        "response_qualities": response_qualities,
        "errors": errors,
        "success_rate": success_rate,
        "degradation_rate": degradation_rate,
        "message": f"System handled degraded response quality with {avg_quality:.1f}/10 average quality score"
    }


def run_service_degradation_test_suite(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Run the complete suite of service degradation tests.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with overall test suite results
    """
    runner = EdgeCaseTestRunner()
    
    tests = [
        (test_slow_model_responses, "Slow Model Responses", [agent_factory], {}),
        (test_intermittent_timeouts, "Intermittent Timeouts", [agent_factory], {}),
        (test_service_throttling, "Service Throttling", [agent_factory], {}),
        (test_degraded_response_quality, "Degraded Response Quality", [agent_factory], {})
    ]
    
    return runner.run_test_suite(tests)


def register_tests(test_manager) -> None:
    """
    Register all service degradation tests with the test manager.
    
    Args:
        test_manager: The test manager to register tests with
    """
    test_manager.register_test(TestTier.SCENARIO, test_slow_model_responses)
    test_manager.register_test(TestTier.SCENARIO, test_intermittent_timeouts)
    test_manager.register_test(TestTier.SCENARIO, test_service_throttling)
    test_manager.register_test(TestTier.SCENARIO, test_degraded_response_quality) 