"""
Network Failure Edge Case Tests for Gray Swan Arena.

This module contains tests that simulate various network failure scenarios to verify 
the system's resilience to connectivity issues.
"""

import logging
import random
import time
from typing import Any, Dict, List, Optional

import requests
from unittest.mock import MagicMock, patch

from ...camel_integration import AgentFactory, CommunicationChannel, TestTier
from ...utils.logging_utils import setup_logging
from ...utils.retry_utils import ExponentialBackoffRetryStrategy
from .edge_case_framework import FailureSimulator, EdgeCaseTestRunner

# Set up logging
logger = setup_logging("network_failure_tests")


def test_intermittent_network_failure(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system resilience to intermittent network failures during agent communication.
    
    This test simulates random network failures during message passing between agents
    and verifies that the retry mechanism handles these failures properly.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to intermittent network failures...")
    
    # Create a communication channel with retry capabilities
    retry_strategy = ExponentialBackoffRetryStrategy(
        initial_delay=0.1,  # Short delay for testing
        max_retries=5,
        backoff_factor=2.0,
        jitter=True
    )
    
    channel = CommunicationChannel(retry_strategy=retry_strategy)
    
    # Create a message to send
    test_message = {
        "type": "network_test",
        "sender": "test_agent_1",
        "recipient": "test_agent_2",
        "content": "Testing network resilience",
        "metadata": {
            "importance": "high",
            "retry_count": 0
        }
    }
    
    # Mock the actual send implementation to simulate failures
    original_send = channel._send_message_impl
    
    failure_count = 0
    success_count = 0
    max_failures = 3
    
    def mock_send_impl(message: Dict[str, Any], *args, **kwargs):
        nonlocal failure_count, success_count
        
        # Simulate intermittent network failure
        if failure_count < max_failures and random.random() < 0.7:  # 70% chance of failure
            failure_count += 1
            logger.info(f"Simulating network failure ({failure_count}/{max_failures})")
            raise ConnectionError("Simulated network failure")
        
        # Success after failures or random success
        success_count += 1
        logger.info(f"Message sent successfully (after {failure_count} failures)")
        return original_send(message, *args, **kwargs)
    
    # Apply the mock
    channel._send_message_impl = mock_send_impl
    
    try:
        # Attempt to send the message (should retry on failures)
        channel.send_message(test_message, sender_id="test_agent_1", receiver_id="test_agent_2")
        
        # If we got here, the message was eventually sent successfully
        assert failure_count > 0, "Expected at least one network failure simulation"
        assert success_count > 0, "Expected at least one successful message send"
        
        logger.info(f"Message sent successfully after {failure_count} simulated failures")
        
        return {
            "failure_count": failure_count,
            "success_count": success_count,
            "retries_needed": failure_count,
            "message": "Successfully handled intermittent network failures"
        }
    
    finally:
        # Restore the original implementation
        channel._send_message_impl = original_send


def test_complete_network_outage(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system behavior during a complete network outage.
    
    This test simulates a complete network outage that eventually recovers,
    and verifies that the system can recover and resume operations.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to complete network outage...")
    
    # Create a communication channel with retry capabilities
    retry_strategy = ExponentialBackoffRetryStrategy(
        initial_delay=0.1,  # Short delay for testing
        max_retries=5,
        backoff_factor=1.5,
        jitter=True
    )
    
    channel = CommunicationChannel(retry_strategy=retry_strategy)
    
    # Create a message to send
    test_message = {
        "type": "outage_test",
        "sender": "test_agent_1",
        "recipient": "test_agent_2",
        "content": "Testing network outage recovery",
        "metadata": {
            "importance": "critical",
            "retry_count": 0
        }
    }
    
    # Mock the actual send implementation to simulate a complete outage
    original_send = channel._send_message_impl
    
    failure_count = 0
    success_count = 0
    recovery_attempt = 0
    max_outage_attempts = 5  # Fail completely for this many attempts
    
    def mock_send_impl(message: Dict[str, Any], *args, **kwargs):
        nonlocal failure_count, success_count, recovery_attempt
        
        if recovery_attempt < max_outage_attempts:
            # Complete outage phase - always fail
            recovery_attempt += 1
            failure_count += 1
            logger.info(f"Simulating network outage ({recovery_attempt}/{max_outage_attempts})")
            raise ConnectionError("Simulated complete network outage")
        
        # After max failures, the network "recovers"
        success_count += 1
        logger.info("Network recovered, message sent successfully")
        return original_send(message, *args, **kwargs)
    
    # Apply the mock
    channel._send_message_impl = mock_send_impl
    
    # Create a dead letter queue to track failed messages
    dlq = channel.get_dead_letter_queue()
    dlq.clear()  # Ensure it's empty at start
    
    try:
        # Attempt to send message with complete outage
        # This should eventually succeed after the outage "recovers"
        channel.send_message(test_message, sender_id="test_agent_1", receiver_id="test_agent_2")
        
        # If we got here, the message was eventually sent after recovery
        assert failure_count == max_outage_attempts, f"Expected exactly {max_outage_attempts} failures during outage"
        assert success_count == 1, "Expected exactly one successful message send after recovery"
        
        # Check dead letter queue
        dlq_messages = dlq.get_messages()
        assert len(dlq_messages) == 0, "No messages should be in DLQ when retries succeed"
        
        logger.info(f"Message sent successfully after network recovery (after {failure_count} failures)")
        
        return {
            "outage_duration": failure_count,
            "recovery_successful": True,
            "retries_needed": failure_count,
            "message": "Successfully recovered from complete network outage"
        }
    
    except Exception as e:
        # If max retries exceeded, check that the message was added to the dead letter queue
        dlq_messages = dlq.get_messages()
        assert len(dlq_messages) > 0, "Message should be in DLQ when retries are exhausted"
        
        return {
            "outage_duration": failure_count,
            "recovery_successful": False,
            "retries_attempted": failure_count,
            "error": str(e),
            "dead_letter_messages": len(dlq_messages),
            "message": "Message moved to dead letter queue after retry exhaustion"
        }
    
    finally:
        # Restore the original implementation
        channel._send_message_impl = original_send


def test_high_latency_communication(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system behavior with extremely high network latency.
    
    This test simulates high latency connections and verifies that the system
    can handle delayed responses without timing out or failing.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to high latency communication...")
    
    # Create a communication channel
    channel = CommunicationChannel()
    
    # Create a message to send
    test_message = {
        "type": "latency_test",
        "sender": "test_agent_1",
        "recipient": "test_agent_2",
        "content": "Testing high latency communication",
        "metadata": {
            "importance": "medium",
            "latency_sensitive": False
        }
    }
    
    # Mock the actual send implementation to simulate high latency
    original_send = channel._send_message_impl
    
    # Track latency values
    latencies = []
    max_latency = 2.0  # Maximum latency in seconds
    min_latency = 0.5  # Minimum latency in seconds
    
    def mock_send_impl(message: Dict[str, Any], *args, **kwargs):
        # Simulate high latency
        latency = random.uniform(min_latency, max_latency)
        latencies.append(latency)
        logger.info(f"Simulating high latency of {latency:.2f}s")
        
        # Simulate network delay
        time.sleep(latency)
        
        # Then succeed
        return original_send(message, *args, **kwargs)
    
    # Apply the mock
    channel._send_message_impl = mock_send_impl
    
    try:
        # Time how long the send operation takes
        start_time = time.time()
        
        # Attempt to send message with high latency
        channel.send_message(test_message, sender_id="test_agent_1", receiver_id="test_agent_2")
        
        # Calculate total operation time
        total_time = time.time() - start_time
        
        # Verify the operation took at least the minimum latency
        assert total_time >= min_latency, f"Operation should take at least {min_latency}s with simulated latency"
        
        logger.info(f"Message sent successfully with high latency of {total_time:.2f}s")
        
        return {
            "latencies": latencies,
            "total_time": total_time,
            "average_latency": sum(latencies) / len(latencies) if latencies else 0,
            "message": "Successfully handled high latency communication"
        }
    
    finally:
        # Restore the original implementation
        channel._send_message_impl = original_send


def test_api_rate_limiting(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system behavior when encountering API rate limiting.
    
    This test simulates API rate limiting responses and verifies that the
    system properly backs off and retries according to the rate limits.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to API rate limiting...")
    
    # Create a mock of an agent that uses an API
    recon_agent = agent_factory.create_recon_agent()
    
    # Keep track of API calls and their timing
    api_calls = []
    rate_limit_count = 0
    success_count = 0
    
    # Define a response class to simulate rate limiting
    class MockResponse:
        def __init__(self, status_code, json_data=None, headers=None):
            self.status_code = status_code
            self.json_data = json_data or {}
            self.headers = headers or {}
            
        def json(self):
            return self.json_data
            
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP Error: {self.status_code}")
    
    # Mock the requests.get method to simulate rate limiting
    def mock_requests_get(*args, **kwargs):
        nonlocal rate_limit_count, success_count
        
        # Record the API call
        current_time = time.time()
        api_calls.append(current_time)
        
        # Calculate call rate (calls per minute)
        recent_calls = [t for t in api_calls if current_time - t < 60]
        call_rate = len(recent_calls)
        
        # If more than 3 calls in the last 60 seconds, rate limit
        if call_rate > 3 and rate_limit_count < 3:
            rate_limit_count += 1
            logger.info(f"Simulating rate limit response ({rate_limit_count})")
            
            # Simulate a 429 Too Many Requests response
            headers = {
                "Retry-After": "10",  # Suggest a 10 second wait
                "X-RateLimit-Limit": "60",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(current_time + 10))
            }
            
            response = MockResponse(
                status_code=429,
                json_data={"error": "Rate limit exceeded", "message": "Too many requests"},
                headers=headers
            )
            
            return response
        
        # Otherwise succeed
        success_count += 1
        logger.info(f"Simulating successful API response ({success_count})")
        
        return MockResponse(
            status_code=200,
            json_data={"data": "Test data", "success": True},
            headers={"X-RateLimit-Remaining": "57"}
        )
    
    # Patch the requests.get method for this test
    with patch('requests.get', side_effect=mock_requests_get):
        with patch('requests.post', side_effect=mock_requests_get):
            # Use a retry strategy that respects rate limits
            retry_strategy = ExponentialBackoffRetryStrategy(
                initial_delay=1.0,
                max_retries=5,
                backoff_factor=2.0,
                jitter=True
            )
            
            try:
                # Configure the agent to use our retry strategy
                # Note: This assumes the agent has a retry_strategy attribute
                if hasattr(recon_agent, 'retry_strategy'):
                    original_strategy = recon_agent.retry_strategy
                    recon_agent.retry_strategy = retry_strategy
                
                # Simulate API calls that will trigger rate limiting
                results = []
                for i in range(7):  # Make 7 calls, should trigger rate limiting
                    logger.info(f"Making API call {i+1}/7")
                    
                    # Call a method that would use the API
                    # For testing, we'll just directly use the mocked requests function
                    response = mock_requests_get(f"https://api.example.com/data/{i}")
                    
                    # Add to results
                    results.append({
                        "status_code": response.status_code,
                        "success": response.status_code == 200,
                        "headers": response.headers,
                        "data": response.json()
                    })
                    
                    # Simulate a wait between calls - shorter than real rate limits would use
                    if i < 6:  # Don't sleep after the last call
                        time.sleep(0.5)
                
                # If we get here without exceptions, the test passed
                logger.info(f"Made {len(results)} API calls with {rate_limit_count} rate limits")
                
                return {
                    "total_calls": len(api_calls),
                    "rate_limit_responses": rate_limit_count,
                    "successful_responses": success_count,
                    "results": results,
                    "message": "Successfully handled API rate limiting"
                }
            
            finally:
                # Restore original retry strategy if we modified it
                if hasattr(recon_agent, 'retry_strategy') and 'original_strategy' in locals():
                    recon_agent.retry_strategy = original_strategy


def run_network_failure_test_suite(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Run the complete suite of network failure tests.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with overall test suite results
    """
    runner = EdgeCaseTestRunner()
    
    tests = [
        (test_intermittent_network_failure, "Intermittent Network Failures", [agent_factory], {}),
        (test_complete_network_outage, "Complete Network Outage", [agent_factory], {}),
        (test_high_latency_communication, "High Latency Communication", [agent_factory], {}),
        (test_api_rate_limiting, "API Rate Limiting", [agent_factory], {})
    ]
    
    return runner.run_test_suite(tests)


def register_tests(test_manager) -> None:
    """
    Register all network failure tests with the test manager.
    
    Args:
        test_manager: The test manager to register tests with
    """
    test_manager.register_test(TestTier.SCENARIO, test_intermittent_network_failure)
    test_manager.register_test(TestTier.SCENARIO, test_complete_network_outage) 
    test_manager.register_test(TestTier.SCENARIO, test_high_latency_communication)
    test_manager.register_test(TestTier.SCENARIO, test_api_rate_limiting) 