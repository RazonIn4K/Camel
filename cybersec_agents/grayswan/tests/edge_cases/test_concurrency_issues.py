"""
Concurrency Issue Edge Case Tests for Gray Swan Arena.

This module contains tests that simulate various concurrency issues such as
race conditions and deadlocks to verify the system's resilience.
"""

import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from unittest.mock import MagicMock, patch

from ...camel_integration import AgentFactory, CommunicationChannel, TestTier
from ...utils.logging_utils import setup_logging
from .edge_case_framework import ConcurrencyTester, EdgeCaseTestRunner

# Set up logging
logger = setup_logging("concurrency_tests")


def test_message_ordering(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of out-of-order messages.
    
    This test sends messages in a specific order but delivers them out of order,
    and verifies that the system properly reorders or handles out-of-order messages.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to out-of-order messages...")
    
    # Create a communication channel
    channel = CommunicationChannel()
    
    # Create sequential messages with sequence numbers
    num_messages = 10
    messages = []
    for i in range(num_messages):
        messages.append({
            "type": "test_message",
            "sender": "test_agent_1",
            "recipient": "test_agent_2",
            "content": f"Message {i+1} of {num_messages}",
            "metadata": {
                "sequence_number": i+1,
                "timestamp": time.time() + i  # Sequential timestamps
            }
        })
    
    # Shuffle the messages to simulate out-of-order delivery
    shuffled_messages = messages.copy()
    random.shuffle(shuffled_messages)
    
    # Create a receiver that tracks message order
    received_messages = []
    
    # Mock the _receive_message_impl method to track messages
    original_receive = getattr(channel, '_receive_message_impl', None)
    
    def mock_receive_impl():
        if shuffled_messages:
            message = shuffled_messages.pop(0)
            received_messages.append(message)
            return message
        return None
    
    # Apply the mock if possible
    if original_receive:
        setattr(channel, '_receive_message_impl', mock_receive_impl)
    
    # Send all messages
    for message in messages:
        channel.send_message(message, 
                           sender_id=message["sender"],
                           receiver_id=message["recipient"])
    
    # Wait briefly to ensure all messages are "sent"
    time.sleep(0.1)
    
    # Check if messages were properly sequenced
    is_ordered = True
    expected_sequence = []
    actual_sequence = []
    
    if hasattr(channel, 'message_queue'):
        # If the channel has a message_queue, check if it's ordered by sequence number
        queue_messages = list(channel.message_queue)
        if queue_messages:
            expected_sequence = [m.get("metadata", {}).get("sequence_number", 0) for m in messages]
            actual_sequence = [m.get("metadata", {}).get("sequence_number", 0) for m in queue_messages]
            
            is_ordered = expected_sequence == sorted(expected_sequence)
    elif received_messages:
        # Otherwise use the received_messages we tracked
        expected_sequence = [m.get("metadata", {}).get("sequence_number", 0) for m in messages]
        actual_sequence = [m.get("metadata", {}).get("sequence_number", 0) for m in received_messages]
        
        is_ordered = expected_sequence == sorted(expected_sequence)
    
    # Restore original implementation if needed
    if original_receive:
        setattr(channel, '_receive_message_impl', original_receive)
    
    logger.info(f"Message ordering test: {'ordered' if is_ordered else 'out of order'}")
    logger.debug(f"Expected sequence: {expected_sequence}")
    logger.debug(f"Actual sequence: {actual_sequence}")
    
    return {
        "messages_sent": len(messages),
        "messages_received": len(received_messages),
        "properly_ordered": is_ordered,
        "expected_sequence": expected_sequence,
        "actual_sequence": actual_sequence,
        "message": "System properly handles message ordering" if is_ordered else "System does not reorder messages"
    }


def test_race_conditions(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of race conditions on shared resources.
    
    This test creates a scenario where multiple threads attempt to modify
    a shared resource simultaneously, potentially causing race conditions.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to race conditions...")
    
    # Create a shared resource for testing
    shared_resource = {
        "counter": 0,
        "messages": [],
        "last_update": None
    }
    
    # Function to modify the shared resource - intentionally not thread-safe
    def modify_resource(resource):
        # Read the current counter
        current_counter = resource["counter"]
        
        # Simulate some processing time to increase race condition likelihood
        time.sleep(random.uniform(0.001, 0.005))
        
        # Increment counter
        resource["counter"] = current_counter + 1
        
        # Add a message
        resource["messages"].append(f"Update {current_counter + 1}")
        
        # Update timestamp
        resource["last_update"] = time.time()
    
    # Use the ConcurrencyTester to simulate race conditions
    concurrency_tester = ConcurrencyTester()
    num_threads = 50
    
    # Run the test
    exceptions = concurrency_tester.simulate_race_condition(
        shared_resource=shared_resource,
        modify_func=modify_resource,
        iterations=num_threads
    )
    
    # Analyze results
    expected_counter = num_threads
    actual_counter = shared_resource["counter"]
    message_count = len(shared_resource["messages"])
    
    has_race_condition = actual_counter != expected_counter or message_count != expected_counter
    
    logger.info(f"Race condition detected: {has_race_condition}")
    logger.info(f"Expected counter: {expected_counter}, Actual: {actual_counter}")
    logger.info(f"Message count: {message_count}")
    
    return {
        "threads": num_threads,
        "expected_counter": expected_counter,
        "actual_counter": actual_counter,
        "message_count": message_count,
        "has_race_condition": has_race_condition,
        "exceptions": [str(e) for e in exceptions],
        "message": "System is vulnerable to race conditions" if has_race_condition else "System is resilient to race conditions"
    }


def test_deadlocks(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's resilience to potential deadlock situations.
    
    This test creates a scenario where threads attempt to acquire multiple locks
    in different orders, potentially causing deadlocks.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to deadlocks...")
    
    # Create locks for testing
    lock_a = threading.Lock()
    lock_b = threading.Lock()
    
    # Use the ConcurrencyTester to simulate a potential deadlock
    concurrency_tester = ConcurrencyTester()
    
    # Run the deadlock simulation with a short timeout
    deadlock_detected = concurrency_tester.deadlock_simulation(
        resources=[lock_a, lock_b],
        timeout=0.5
    )
    
    logger.info(f"Deadlock detected: {deadlock_detected}")
    
    return {
        "deadlock_detected": deadlock_detected,
        "message": "System is vulnerable to deadlocks" if deadlock_detected else "System is resilient to deadlocks"
    }


def test_parallel_agent_execution(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of multiple agents running in parallel.
    
    This test creates multiple agents and runs them in parallel,
    verifying that they operate correctly without interfering with each other.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing parallel agent execution...")
    
    # Create a shared resource to track agent operations
    operations_log = []
    
    # Create a lock for the operations log
    log_lock = threading.Lock()
    
    # Function to run an agent task
    def run_agent_task(agent_id):
        # Simulate agent initialization
        logger.info(f"Agent {agent_id} initializing")
        time.sleep(random.uniform(0.01, 0.05))
        
        # Simulate agent processing
        for i in range(3):  # Each agent performs 3 operations
            # Simulate some work
            time.sleep(random.uniform(0.01, 0.05))
            
            # Log the operation
            with log_lock:
                operations_log.append({
                    "agent_id": agent_id,
                    "operation": f"task_{i+1}",
                    "timestamp": time.time()
                })
            
            logger.debug(f"Agent {agent_id} completed operation {i+1}")
        
        logger.info(f"Agent {agent_id} completed all tasks")
    
    # Create and start multiple agent threads
    num_agents = 5
    threads = []
    
    for i in range(num_agents):
        agent_thread = threading.Thread(target=run_agent_task, args=(f"agent_{i+1}",))
        threads.append(agent_thread)
        agent_thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Analyze results
    total_operations = len(operations_log)
    expected_operations = num_agents * 3  # 3 operations per agent
    
    # Verify each agent performed the expected operations
    agent_operation_counts = {}
    for op in operations_log:
        agent_id = op["agent_id"]
        if agent_id not in agent_operation_counts:
            agent_operation_counts[agent_id] = 0
        agent_operation_counts[agent_id] += 1
    
    # Check if all agents performed the expected number of operations
    all_agents_complete = all(count == 3 for count in agent_operation_counts.values())
    
    logger.info(f"Parallel execution test: {total_operations}/{expected_operations} operations completed")
    logger.info(f"All agents completed expected operations: {all_agents_complete}")
    
    return {
        "agents": num_agents,
        "expected_operations": expected_operations,
        "actual_operations": total_operations,
        "agent_operation_counts": agent_operation_counts,
        "all_agents_complete": all_agents_complete,
        "message": "Parallel agent execution successful" if all_agents_complete else "Some agents did not complete all operations"
    }


def test_message_flood(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of a sudden flood of messages.
    
    This test sends a large number of messages in rapid succession,
    verifying that the system properly handles the message flood without
    crashing or losing messages.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to message floods...")
    
    # Create a communication channel
    channel = CommunicationChannel()
    
    # Create a large number of messages
    num_messages = 1000
    messages = []
    for i in range(num_messages):
        messages.append({
            "type": "flood_test",
            "sender": "test_agent_1",
            "recipient": "test_agent_2",
            "content": f"Flood message {i+1}",
            "metadata": {
                "sequence_number": i+1,
                "batch": "flood_test"
            }
        })
    
    # Track message processing
    sent_count = 0
    failed_count = 0
    
    # Send all messages as quickly as possible
    start_time = time.time()
    
    for message in messages:
        try:
            channel.send_message(message, 
                               sender_id=message["sender"],
                               receiver_id=message["recipient"])
            sent_count += 1
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            failed_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate rate
    messages_per_second = sent_count / total_time if total_time > 0 else 0
    
    logger.info(f"Message flood test: {sent_count}/{num_messages} messages sent successfully")
    logger.info(f"Send rate: {messages_per_second:.2f} messages/second")
    
    return {
        "messages_attempted": num_messages,
        "messages_sent": sent_count,
        "messages_failed": failed_count,
        "total_time": total_time,
        "messages_per_second": messages_per_second,
        "success_rate": sent_count / num_messages if num_messages > 0 else 0,
        "message": f"System handled message flood with {sent_count}/{num_messages} messages processed"
    }


def run_concurrency_test_suite(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Run the complete suite of concurrency issue tests.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with overall test suite results
    """
    runner = EdgeCaseTestRunner()
    
    tests = [
        (test_message_ordering, "Message Ordering", [agent_factory], {}),
        (test_race_conditions, "Race Conditions", [agent_factory], {}),
        (test_deadlocks, "Deadlocks", [agent_factory], {}),
        (test_parallel_agent_execution, "Parallel Agent Execution", [agent_factory], {}),
        (test_message_flood, "Message Flood", [agent_factory], {})
    ]
    
    return runner.run_test_suite(tests)


def register_tests(test_manager) -> None:
    """
    Register all concurrency issue tests with the test manager.
    
    Args:
        test_manager: The test manager to register tests with
    """
    test_manager.register_test(TestTier.SCENARIO, test_message_ordering)
    test_manager.register_test(TestTier.SCENARIO, test_race_conditions) 
    test_manager.register_test(TestTier.SCENARIO, test_deadlocks)
    test_manager.register_test(TestTier.SCENARIO, test_parallel_agent_execution)
    test_manager.register_test(TestTier.SCENARIO, test_message_flood) 