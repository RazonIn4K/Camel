"""
Resource Exhaustion Edge Case Tests for Gray Swan Arena.

This module contains tests that simulate various resource exhaustion scenarios
such as memory leaks, high CPU usage, and disk space limitations to verify
the system's resilience under resource constraints.
"""

import logging
import os
import psutil
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ...camel_integration import AgentFactory, TestTier
from ...utils.logging_utils import setup_logging
from .edge_case_framework import FailureSimulator, EdgeCaseTestRunner

# Set up logging
logger = setup_logging("resource_exhaustion_tests")


def test_memory_pressure(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's behavior under memory pressure.
    
    This test simulates memory pressure by allocating large amounts of memory
    and verifies that the system properly handles low memory conditions.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to memory pressure...")
    
    # Use the FailureSimulator to simulate memory pressure
    failure_simulator = FailureSimulator()
    
    # Create an agent to test with
    test_agent = agent_factory.create_evaluation_agent()
    
    # Capture baseline memory
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    # Define test parameters
    memory_pressure_mb = 100  # Allocate 100MB to simulate pressure
    timeout_seconds = 5
    
    # Apply memory pressure
    logger.info(f"Applying memory pressure ({memory_pressure_mb}MB)...")
    memory_objects = failure_simulator.simulate_resource_exhaustion(
        resource_type="memory",
        amount=memory_pressure_mb,
        unit="MB"
    )
    
    # Measure memory usage during pressure
    pressure_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    memory_increase = pressure_memory - baseline_memory
    
    # Attempt to use the agent under memory pressure
    start_time = time.time()
    agent_response = None
    error = None
    
    try:
        # Perform a simple agent operation
        agent_response = test_agent.get_evaluation(
            "Test query during memory pressure",
            timeout=timeout_seconds
        )
    except Exception as e:
        error = str(e)
        logger.error(f"Error during memory pressure test: {error}")
    
    response_time = time.time() - start_time
    
    # Release memory pressure
    del memory_objects
    
    # Check current memory
    end_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    # Analyze results
    success = agent_response is not None and error is None
    
    logger.info(f"Memory pressure test result: {'Success' if success else 'Failed'}")
    logger.info(f"Baseline memory: {baseline_memory:.2f}MB")
    logger.info(f"Memory during pressure: {pressure_memory:.2f}MB (increase: {memory_increase:.2f}MB)")
    logger.info(f"End memory: {end_memory:.2f}MB")
    logger.info(f"Response time: {response_time:.2f}s")
    
    return {
        "baseline_memory_mb": baseline_memory,
        "pressure_memory_mb": pressure_memory,
        "memory_increase_mb": memory_increase,
        "end_memory_mb": end_memory,
        "response_time_seconds": response_time,
        "success": success,
        "error": error,
        "has_memory_leak": (end_memory - baseline_memory) > 10,  # Check if >10MB remained allocated
        "message": "System handled memory pressure successfully" if success else f"System failed under memory pressure: {error}"
    }


def test_cpu_saturation(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's behavior under CPU saturation.
    
    This test simulates high CPU usage by spawning CPU-intensive tasks
    and verifies that the system properly prioritizes and handles critical operations.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to CPU saturation...")
    
    # Use the FailureSimulator to simulate CPU saturation
    failure_simulator = FailureSimulator()
    
    # Create an agent to test with
    test_agent = agent_factory.create_evaluation_agent()
    
    # Capture baseline CPU
    baseline_cpu = psutil.cpu_percent(interval=0.5)
    
    # Define test parameters
    cpu_threads = max(1, psutil.cpu_count() - 1)  # Use all but one CPU core
    timeout_seconds = 8
    
    # Start CPU stress
    logger.info(f"Applying CPU saturation ({cpu_threads} threads)...")
    cpu_stress_handle = failure_simulator.simulate_resource_exhaustion(
        resource_type="cpu",
        amount=cpu_threads,
        unit="threads"
    )
    
    # Allow CPU usage to ramp up
    time.sleep(1)
    
    # Measure CPU during saturation
    saturation_cpu = psutil.cpu_percent(interval=0.5)
    
    # Attempt to use the agent under CPU pressure
    start_time = time.time()
    agent_response = None
    error = None
    
    try:
        # Perform a simple agent operation
        agent_response = test_agent.get_evaluation(
            "Test query during CPU saturation",
            timeout=timeout_seconds
        )
    except Exception as e:
        error = str(e)
        logger.error(f"Error during CPU saturation test: {error}")
    
    response_time = time.time() - start_time
    
    # Release CPU pressure
    failure_simulator.release_resource(cpu_stress_handle)
    
    # Allow system to stabilize
    time.sleep(1)
    
    # Check current CPU
    end_cpu = psutil.cpu_percent(interval=0.5)
    
    # Analyze results
    success = agent_response is not None and error is None
    
    logger.info(f"CPU saturation test result: {'Success' if success else 'Failed'}")
    logger.info(f"Baseline CPU: {baseline_cpu}%")
    logger.info(f"CPU during saturation: {saturation_cpu}%")
    logger.info(f"End CPU: {end_cpu}%")
    logger.info(f"Response time: {response_time:.2f}s")
    
    return {
        "baseline_cpu_percent": baseline_cpu,
        "saturation_cpu_percent": saturation_cpu,
        "end_cpu_percent": end_cpu,
        "response_time_seconds": response_time,
        "success": success,
        "error": error,
        "message": "System handled CPU saturation successfully" if success else f"System failed under CPU saturation: {error}"
    }


def test_disk_space_limitations(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's behavior when disk space is limited.
    
    This test simulates limited disk space and verifies that the system
    properly handles disk space constraints.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to disk space limitations...")
    
    # Use the FailureSimulator to simulate disk space limitations
    failure_simulator = FailureSimulator()
    
    # Create an agent to test with
    test_agent = agent_factory.create_evaluation_agent()
    
    # Get baseline disk info
    disk = psutil.disk_usage('/')
    baseline_free = disk.free / (1024 * 1024 * 1024)  # Convert to GB
    
    # Define test parameters
    # Note: We won't actually fill the disk, just simulate low space
    simulate_free_space_gb = 0.1  # Simulate only 100MB free
    timeout_seconds = 5
    
    # Simulate disk space limitation
    logger.info(f"Simulating low disk space (simulating {simulate_free_space_gb}GB free)...")
    
    # We'll create a temporary file to track disk operations
    temp_log_file = "temp_disk_operations.log"
    
    # Mock disk space check function to return low space
    original_disk_usage = psutil.disk_usage
    
    def mock_disk_usage(path):
        result = original_disk_usage(path)
        # Create a named tuple with modified free space
        SimulatedDiskUsage = type(result)
        return SimulatedDiskUsage(
            result.total,
            result.used,
            int(simulate_free_space_gb * 1024 * 1024 * 1024),  # Convert GB to bytes
            result.percent
        )
    
    # Apply the mock
    with patch('psutil.disk_usage', side_effect=mock_disk_usage):
        # Attempt to use the agent under disk space limitation
        start_time = time.time()
        agent_response = None
        error = None
        
        try:
            # Create a file to test disk operations
            with open(temp_log_file, 'w') as f:
                f.write("Test disk operation during low space simulation\n")
            
            # Perform a simple agent operation
            agent_response = test_agent.get_evaluation(
                "Test query during disk space limitation",
                timeout=timeout_seconds
            )
        except Exception as e:
            error = str(e)
            logger.error(f"Error during disk space limitation test: {error}")
        
        response_time = time.time() - start_time
    
    # Clean up the temporary file
    if os.path.exists(temp_log_file):
        os.remove(temp_log_file)
    
    # Get current disk info
    current_disk = psutil.disk_usage('/')
    current_free = current_disk.free / (1024 * 1024 * 1024)  # Convert to GB
    
    # Analyze results
    success = agent_response is not None and error is None
    
    logger.info(f"Disk space limitation test result: {'Success' if success else 'Failed'}")
    logger.info(f"Baseline free space: {baseline_free:.2f}GB")
    logger.info(f"Simulated free space: {simulate_free_space_gb:.2f}GB")
    logger.info(f"Current free space: {current_free:.2f}GB")
    logger.info(f"Response time: {response_time:.2f}s")
    
    return {
        "baseline_free_space_gb": baseline_free,
        "simulated_free_space_gb": simulate_free_space_gb,
        "current_free_space_gb": current_free,
        "response_time_seconds": response_time,
        "success": success,
        "error": error,
        "message": "System handled disk space limitations successfully" if success else f"System failed under disk space limitations: {error}"
    }


def test_file_descriptor_exhaustion(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's behavior when file descriptors are exhausted.
    
    This test simulates file descriptor exhaustion by opening many file handles
    and verifies that the system properly handles this condition.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to file descriptor exhaustion...")
    
    # Use the FailureSimulator to simulate file descriptor exhaustion
    failure_simulator = FailureSimulator()
    
    # Create an agent to test with
    test_agent = agent_factory.create_evaluation_agent()
    
    # Define test parameters
    num_file_descriptors = 100  # Open 100 file descriptors
    timeout_seconds = 5
    
    # Simulate file descriptor exhaustion
    logger.info(f"Opening {num_file_descriptors} file descriptors...")
    file_handles = failure_simulator.simulate_resource_exhaustion(
        resource_type="file_descriptors",
        amount=num_file_descriptors,
        unit="files"
    )
    
    # Attempt to use the agent under file descriptor exhaustion
    start_time = time.time()
    agent_response = None
    error = None
    
    try:
        # Perform a simple agent operation
        agent_response = test_agent.get_evaluation(
            "Test query during file descriptor exhaustion",
            timeout=timeout_seconds
        )
    except Exception as e:
        error = str(e)
        logger.error(f"Error during file descriptor exhaustion test: {error}")
    
    response_time = time.time() - start_time
    
    # Release file descriptors
    failure_simulator.release_resource(file_handles)
    
    # Analyze results
    success = agent_response is not None and error is None
    
    logger.info(f"File descriptor exhaustion test result: {'Success' if success else 'Failed'}")
    logger.info(f"Number of file descriptors opened: {num_file_descriptors}")
    logger.info(f"Response time: {response_time:.2f}s")
    
    return {
        "file_descriptors_opened": num_file_descriptors,
        "response_time_seconds": response_time,
        "success": success,
        "error": error,
        "message": "System handled file descriptor exhaustion successfully" if success else f"System failed under file descriptor exhaustion: {error}"
    }


def run_resource_exhaustion_test_suite(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Run the complete suite of resource exhaustion tests.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with overall test suite results
    """
    runner = EdgeCaseTestRunner()
    
    tests = [
        (test_memory_pressure, "Memory Pressure", [agent_factory], {}),
        (test_cpu_saturation, "CPU Saturation", [agent_factory], {}),
        (test_disk_space_limitations, "Disk Space Limitations", [agent_factory], {}),
        (test_file_descriptor_exhaustion, "File Descriptor Exhaustion", [agent_factory], {})
    ]
    
    return runner.run_test_suite(tests)


def register_tests(test_manager) -> None:
    """
    Register all resource exhaustion tests with the test manager.
    
    Args:
        test_manager: The test manager to register tests with
    """
    test_manager.register_test(TestTier.SCENARIO, test_memory_pressure)
    test_manager.register_test(TestTier.SCENARIO, test_cpu_saturation)
    test_manager.register_test(TestTier.SCENARIO, test_disk_space_limitations)
    test_manager.register_test(TestTier.SCENARIO, test_file_descriptor_exhaustion) 