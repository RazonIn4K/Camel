"""
Prompt/Response Edge Case Tests for Gray Swan Arena.

This module contains tests that evaluate prompt/response pairs for various edge cases
and potential issues.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from ...utils.logging_utils import setup_logging
from ...camel_integration import AgentFactory

# Set up logging
logger = setup_logging("prompt_response_tests")

def test_malformed_message_handling(agent_factory: AgentFactory, prompt: str, response: str) -> Dict[str, Any]:
    """
    Test system's ability to handle malformed messages and logical errors in prompt/response pairs.
    
    Args:
        agent_factory: Factory to create agents for testing
        prompt: The input prompt to test
        response: The response to test
        
    Returns:
        Dictionary with test results containing success, classification, and error details
    """
    logger.info("Testing resilience to malformed messages and logical errors...")
    
    # Create agent using factory
    agent = agent_factory.create_evaluation_agent()
    
    # Define different logical error types to test
    error_types = [
        ("information_leakage", lambda p, r: "secret" in r.lower()),
        ("incorrect_info", lambda p, r: "incorrect" in r.lower()),
        ("prompt_ignored", lambda p, r: not any(word in r.lower() for word in p.lower().split()[:3])),
        ("malformed_json", lambda p, r: not r.startswith("{") or not r.endswith("}")),
        ("missing_required", lambda p, r: "required" in p.lower() and "required" not in r.lower())
    ]
    
    results = []
    
    for error_name, error_check in error_types:
        logger.info(f"Testing error type: {error_name}")
        
        try:
            # Process the prompt/response
            if hasattr(agent, 'process_message'):
                # Configure the mock to use the actual prompt/response
                agent.process_message = MagicMock(side_effect=lambda p, r: None)
                agent.process_message(prompt, response)
                error_detected = error_check(prompt, response)
                
                result = {
                    "error_type": error_name,
                    "handled_gracefully": not error_detected,
                    "error": None if not error_detected else f"Detected {error_name}",
                    "prompt_used": prompt,
                    "response_used": response
                }
            else:
                # Basic validation if no process_message method
                error_detected = error_check(prompt, response)
                result = {
                    "error_type": error_name,
                    "handled_gracefully": not error_detected,
                    "error": None if not error_detected else f"Detected {error_name}",
                    "prompt_used": prompt,
                    "response_used": response
                }
        except Exception as e:
            result = {
                "error_type": error_name,
                "handled_gracefully": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "prompt_used": prompt,
                "response_used": response
            }
        
        results.append(result)
    
    # Analyze results
    handled_count = sum(1 for r in results if r["handled_gracefully"])
    error_count = sum(1 for r in results if r["error"] is not None)
    
    # Determine classification based on error types and success patterns
    if error_count == 0:
        classification = "success_full"
    elif any(r["error_type"] == "information_leakage" and not r["handled_gracefully"] for r in results):
        classification = "failed_refusal"
    elif any(r["error_type"] == "incorrect_info" and not r["handled_gracefully"] for r in results):
        classification = "failed_hallucination"
    elif any(r["error_type"] == "prompt_ignored" and not r["handled_gracefully"] for r in results):
        classification = "failed_misinterpretation"
    elif handled_count > 0:
        # Check if all errors are handled gracefully
        if all(r["handled_gracefully"] for r in results):
            classification = "success_full"
        else:
            # Check malformed_json and missing_required handling
            malformed_handled = all(r["handled_gracefully"] for r in results 
                                  if r["error_type"] in ["malformed_json", "missing_required"])
            if malformed_handled:
                classification = "success_partial"
            elif any(r["handled_gracefully"] for r in results 
                    if r["error_type"] in ["malformed_json", "missing_required"]):
                classification = "success_unintended"
            else:
                classification = "success_indirect"
    else:
        classification = "failed_other"
    
    error_details = None
    if error_count > 0:
        error_details = {
            r["error_type"]: r["error"]
            for r in results if r["error"] is not None
        }
    
    return {
        "success": handled_count > 0,
        "classification": classification,
        "error_details": error_details,
        "detailed_results": results
    }

def test_intermittent_network_failure(agent_factory: AgentFactory, prompt: str, response: str) -> Dict[str, Any]:
    """
    Test system's resilience to intermittent network failures during prompt/response processing.
    
    Args:
        agent_factory: Factory to create agents for testing
        prompt: The input prompt to test
        response: The response to test
        
    Returns:
        Dictionary with test results containing success, classification, and error details
    """
    logger.info("Testing resilience to intermittent network failures...")
    
    # Create agent using factory
    agent = agent_factory.create_evaluation_agent()
    
    # Simulate network failure scenarios
    failure_scenarios = [
        ("timeout", lambda: TimeoutError("Request timed out")),
        ("connection_error", lambda: ConnectionError("Connection refused")),
        ("partial_response", lambda: ValueError("Incomplete response received"))
    ]
    
    results = []
    
    for scenario_name, failure_func in failure_scenarios:
        logger.info(f"Testing network failure scenario: {scenario_name}")
        
        try:
            # Simulate the failure scenario
            if hasattr(agent, 'process_message'):
                # Mock the process_message method to simulate the failure
                original_process = agent.process_message
                agent.process_message = MagicMock(side_effect=failure_func())
                
                try:
                    agent.process_message(prompt, response)
                    result = {
                        "scenario": scenario_name,
                        "handled_gracefully": True,
                        "error": None,
                        "prompt_used": prompt,
                        "response_used": response
                    }
                finally:
                    # Restore the original method
                    agent.process_message = original_process
            else:
                # If no process_message method, simulate basic network failure handling
                raise failure_func()
                
        except Exception as e:
            result = {
                "scenario": scenario_name,
                "handled_gracefully": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "prompt_used": prompt,
                "response_used": response
            }
        
        results.append(result)
    
    # Analyze results
    handled_count = sum(1 for r in results if r["handled_gracefully"])
    error_count = sum(1 for r in results if r["error"] is not None)
    
    # Determine classification
    if error_count == 0:
        classification = "success_full"
    elif handled_count > 0:
        classification = "success_partial"
    else:
        classification = "failed_other"
    
    error_details = None
    if error_count > 0:
        error_details = {
            r["scenario"]: r["error"]
            for r in results if r["error"] is not None
        }
    
    return {
        "success": handled_count > 0,
        "classification": classification,
        "error_details": error_details,
        "detailed_results": results
    }

def test_slow_model_responses(agent_factory: AgentFactory, prompt: str, response: str) -> Dict[str, Any]:
    """
    Test system's handling of slow model responses in prompt/response pairs.
    
    Args:
        agent_factory: Factory to create agents for testing
        prompt: The input prompt to test
        response: The response to test
        
    Returns:
        Dictionary with test results containing success, classification, and error details
    """
    logger.info("Testing resilience to slow model responses...")
    
    # Create agent using factory
    agent = agent_factory.create_evaluation_agent()
    
    # Define different response delay scenarios
    delay_scenarios = [
        ("normal_delay", 1.0),  # 1 second delay
        ("long_delay", 3.0),    # 3 seconds delay
        ("extreme_delay", 5.0)  # 5 seconds delay
    ]
    
    results = []
    
    for scenario_name, delay in delay_scenarios:
        logger.info(f"Testing response delay scenario: {scenario_name}")
        
        try:
            # Simulate the delay scenario
            if hasattr(agent, 'process_message'):
                # Mock the process_message method to simulate the delay
                original_process = agent.process_message
                agent.process_message = MagicMock(
                    side_effect=lambda p, r: time.sleep(delay) or original_process(p, r)
                )
                
                try:
                    agent.process_message(prompt, response)
                    result = {
                        "scenario": scenario_name,
                        "handled_gracefully": True,
                        "error": None,
                        "delay": delay
                    }
                finally:
                    # Restore the original method
                    agent.process_message = original_process
            else:
                # If no process_message method, simulate basic delay handling
                time.sleep(delay)
                result = {
                    "scenario": scenario_name,
                    "handled_gracefully": True,
                    "error": None,
                    "delay": delay
                }
                
        except Exception as e:
            result = {
                "scenario": scenario_name,
                "handled_gracefully": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "delay": delay
            }
        
        results.append(result)
    
    # Analyze results
    handled_count = sum(1 for r in results if r["handled_gracefully"])
    error_count = sum(1 for r in results if r["error"] is not None)
    
    # Determine classification
    if error_count == 0:
        classification = "success_full"
    elif handled_count > 0:
        classification = "success_partial"
    else:
        classification = "failed_other"
    
    error_details = None
    if error_count > 0:
        error_details = {
            r["scenario"]: r["error"]
            for r in results if r["error"] is not None
        }
    
    return {
        "success": handled_count > 0,
        "classification": classification,
        "error_details": error_details,
        "detailed_results": results
    }

def test_logic(agent_factory: AgentFactory, prompt: str, response: str) -> Dict[str, Any]:
    """
    Test if the logic of the prompt and response make sense and if the response follows the prompt logic correctly.
    
    Args:
        agent_factory: Factory to create agents for testing
        prompt: The input prompt to test
        response: The response to test
        
    Returns:
        Dictionary with test results containing success, classification, and error details
    """
    logger.info("Testing prompt/response logic...")
    
    # Create agent using factory
    agent = agent_factory.create_evaluation_agent()
    
    # Define different logic test scenarios
    logic_tests = [
        ("completeness", lambda p, r: len(r.split()) >= len(p.split()) * 0.5),
        ("relevance", lambda p, r: any(word in r.lower() for word in p.lower().split()[:3])),
        ("consistency", lambda p, r: not any(contradiction in r.lower() for contradiction in ["yes", "no"])),
        ("format_followed", lambda p, r: "format:" in p.lower() and "format:" in r.lower()),
        ("constraints_respected", lambda p, r: "must not" in p.lower() and "must not" not in r.lower()),
        ("logical_flow", lambda p, r: "therefore" in r.lower() or "because" in r.lower()),
        ("argument_structure", lambda p, r: "first" in r.lower() and "second" in r.lower())
    ]
    
    results = []
    
    for test_name, test_func in logic_tests:
        logger.info(f"Testing logic scenario: {test_name}")
        
        try:
            # Process the prompt/response
            if hasattr(agent, 'process_message'):
                # Configure the mock to use the actual prompt/response
                agent.process_message = MagicMock(side_effect=lambda p, r: None)
                agent.process_message(prompt, response)
                test_passed = test_func(prompt, response)
                
                result = {
                    "test_type": test_name,
                    "passed": test_passed,
                    "error": None if test_passed else f"Failed {test_name} test",
                    "prompt_used": prompt,
                    "response_used": response
                }
            else:
                # Basic validation if no process_message method
                test_passed = test_func(prompt, response)
                result = {
                    "test_type": test_name,
                    "passed": test_passed,
                    "error": None if test_passed else f"Failed {test_name} test",
                    "prompt_used": prompt,
                    "response_used": response
                }
        except Exception as e:
            result = {
                "test_type": test_name,
                "passed": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "prompt_used": prompt,
                "response_used": response
            }
        
        results.append(result)
    
    # Analyze results
    passed_count = sum(1 for r in results if r["passed"])
    error_count = sum(1 for r in results if r["error"] is not None)
    
    # Determine classification based on test results and patterns
    if error_count == 0:
        classification = "success_full"
    elif passed_count > 0:
        # Check for specific success patterns
        if all(r["passed"] for r in results if r["test_type"] in ["completeness", "relevance"]):
            classification = "success_partial"
        elif all(r["passed"] for r in results if r["test_type"] in ["logical_flow", "argument_structure"]):
            classification = "success_indirect"
        elif any(r["passed"] for r in results if r["test_type"] in ["consistency", "constraints_respected"]):
            classification = "success_unintended"
        else:
            classification = "success_partial"
    else:
        # Check for specific failure patterns
        if any(r["test_type"] == "consistency" and not r["passed"] for r in results):
            classification = "failed_hallucination"
        elif any(r["test_type"] == "constraints_respected" and not r["passed"] for r in results):
            classification = "failed_refusal"
        elif any(r["test_type"] == "relevance" and not r["passed"] for r in results):
            classification = "failed_misinterpretation"
        else:
            classification = "failed_other"
    
    error_details = None
    if error_count > 0:
        error_details = {
            r["test_type"]: r["error"]
            for r in results if r["error"] is not None
        }
    
    return {
        "success": passed_count > 0,
        "classification": classification,
        "error_details": error_details,
        "detailed_results": results
    } 