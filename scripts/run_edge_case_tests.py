#!/usr/bin/env python3
"""
Edge Case Test Runner.

This script runs edge case tests on prompt/response pairs to evaluate their robustness
and reliability. It tests various scenarios including malformed messages, network failures,
and logical errors.
"""

import os
import sys
import argparse
import logging
import json
import random
import time
from typing import List, Dict, Any, Optional, Set, Union, Protocol
from unittest.mock import MagicMock, patch
from datetime import datetime
from cybersec_agents.defense_agent import DefenseAgent, ThreatLevel, EventType

# Add the parent directory to sys.path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cybersec_agents.grayswan.tests.edge_cases.run_edge_case_tests import main as run_tests
from cybersec_agents.grayswan.tests.edge_cases.edge_case_framework import EdgeCaseTestRunner, AgentFactory
from cybersec_agents.grayswan.tests.edge_cases.test_prompt_response import (
    test_malformed_message_handling,
    test_intermittent_network_failure,
    test_slow_model_responses,
    test_logic
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentFactory(Protocol):
    """Protocol defining the interface for agent factories."""
    
    def create_evaluation_agent(self) -> Any:
        """Create and return an evaluation agent."""
        ...

class MockAgentFactory:
    """Mock AgentFactory for testing prompt/response pairs."""

    def __init__(self):
        """Initialize the mock agent factory."""
        self.mock_agent = DefenseAgent()
        logger.info("MockAgentFactory initialized with DefenseAgent")

    def create_evaluation_agent(self):
        """Return a mock agent for testing."""
        return self.mock_agent

class EdgeCaseTestRunner:
    """Test runner for edge cases in prompt/response pairs."""

    def __init__(self, input_file: str, output_file: str):
        """
        Initialize the test runner.

        Args:
            input_file: Path to the input JSON file containing prompt/response pairs
            output_file: Path to save the test results
        """
        self.input_file = input_file
        self.output_file = output_file
        self.agent_factory = MockAgentFactory()
        self.test_results = []
        logger.info(f"EdgeCaseTestRunner initialized with input: {input_file}, output: {output_file}")

    def load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data from the input file.

        Returns:
            List of dictionaries containing prompt/response pairs and test data
        """
        try:
            with open(self.input_file, 'r') as f:
                data = json.load(f)
            
            # If data is a list of strings, convert to list of dictionaries
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    # Handle list of strings (prompts)
                    data = [{"prompt": item, "response": "", "input_data": {}} for item in data]
                elif all(isinstance(item, dict) for item in data):
                    # Handle list of dictionaries (existing test cases)
                    pass
                else:
                    # Invalid format
                    raise ValueError("Input file must contain either a list of strings or a list of dictionaries")
            else:
                raise ValueError("Input file must contain a list")
            
            logger.info(f"Loaded {len(data)} test cases from {self.input_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise

    def save_test_results(self) -> None:
        """Save test results to the output file."""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            logger.info(f"Test results saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
            raise

    def test_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test the logic of a prompt/response pair.

        Args:
            data: Dictionary containing the prompt/response pair and test data

        Returns:
            Dictionary containing test results
        """
        logger.info(f"Testing logic for prompt: {data['prompt'][:50]}...")
        
        # Initialize test result
        test_result = {
            "prompt": data["prompt"],
            "response": data.get("response", ""),
            "test_results": [],
            "timestamp": datetime.now().isoformat()
        }

        # Test scenarios
        scenarios = [
            ("completeness", self._test_completeness),
            ("relevance", self._test_relevance),
            ("consistency", self._test_consistency),
            ("format_followed", self._test_format),
            ("constraints_respected", self._test_constraints),
            ("logical_flow", self._test_logical_flow),
            ("argument_structure", self._test_argument_structure)
        ]

        # Run each test scenario
        for scenario_name, test_func in scenarios:
            try:
                result = test_func(data)
                test_result["test_results"].append({
                    "scenario": scenario_name,
                    "passed": result["passed"],
                    "details": result["details"]
                })
            except Exception as e:
                logger.error(f"Error in {scenario_name} test: {e}")
                test_result["test_results"].append({
                    "scenario": scenario_name,
                    "passed": False,
                    "details": f"Error: {str(e)}"
                })

        # Only test with DefenseAgent if we have input_data
        if data.get("input_data"):
            try:
                agent = self.agent_factory.create_evaluation_agent()
                
                # Send input data to the agent
                agent.receive_input(data["input_data"])
                agent.process_input()
                output = agent.send_output()
                
                # Evaluate the agent's response
                agent_result = self._evaluate_agent_response(output, data)
                test_result["test_results"].append({
                    "scenario": "defense_agent",
                    "passed": agent_result["passed"],
                    "details": agent_result["details"]
                })
            except Exception as e:
                logger.error(f"Error in DefenseAgent test: {e}")
                test_result["test_results"].append({
                    "scenario": "defense_agent",
                    "passed": False,
                    "details": f"Error: {str(e)}"
                })

        return test_result

    def _evaluate_agent_response(self, output: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the DefenseAgent's response.

        Args:
            output: The output from the DefenseAgent
            data: The original test data

        Returns:
            Dictionary containing evaluation results
        """
        passed = True
        details = []

        # Check if output has required fields
        required_fields = ["primary_action", "reason", "additional_actions", "priority"]
        for field in required_fields:
            if field not in output:
                passed = False
                details.append(f"Missing required field: {field}")

        # Check if threat level is appropriate for the input
        if "input_data" in data and "event_type" in data["input_data"]:
            event_type = data["input_data"]["event_type"]
            if event_type == EventType.LOGIN.value and "review_auth_logs" not in output["additional_actions"]:
                passed = False
                details.append("Login event should trigger auth log review")
            elif event_type == EventType.NETWORK.value and "analyze_network_traffic" not in output["additional_actions"]:
                passed = False
                details.append("Network event should trigger traffic analysis")

        # Check if priority is appropriate for the threat level
        if output["priority"] > 3 and "critical" in str(data).lower():
            passed = False
            details.append("Critical threat should have higher priority")

        return {
            "passed": passed,
            "details": details
        }

    def _test_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test if the response is complete."""
        response = data.get("response", "")
        passed = len(response.split()) >= 10  # Basic completeness check
        details = ["Response is complete" if passed else "Response is too short"]
        return {"passed": passed, "details": details}

    def _test_relevance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test if the response is relevant to the prompt."""
        prompt = data.get("prompt", "").lower()
        response = data.get("response", "").lower()
        # Check if key terms from prompt appear in response
        key_terms = set(prompt.split()) - {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to"}
        relevant_terms = sum(1 for term in key_terms if term in response)
        passed = relevant_terms >= len(key_terms) * 0.5  # At least 50% of key terms should be present
        details = [f"Found {relevant_terms}/{len(key_terms)} relevant terms"]
        return {"passed": passed, "details": details}

    def _test_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test if the response is internally consistent."""
        response = data.get("response", "")
        # Check for contradictory statements
        contradictions = [
            ("always", "never"),
            ("must", "should not"),
            ("required", "optional")
        ]
        found_contradictions = []
        for term1, term2 in contradictions:
            if term1 in response.lower() and term2 in response.lower():
                found_contradictions.append(f"Found contradiction: {term1}/{term2}")
        passed = len(found_contradictions) == 0
        details = ["No contradictions found" if passed else f"Found contradictions: {', '.join(found_contradictions)}"]
        return {"passed": passed, "details": details}

    def _test_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test if the response follows the expected format."""
        response = data.get("response", "")
        # Check for basic formatting elements
        has_paragraphs = "\n\n" in response
        has_punctuation = any(p in response for p in ".!?")
        has_capitalization = any(c.isupper() for c in response)
        passed = has_paragraphs and has_punctuation and has_capitalization
        details = []
        if not has_paragraphs:
            details.append("Missing paragraph breaks")
        if not has_punctuation:
            details.append("Missing proper punctuation")
        if not has_capitalization:
            details.append("Missing proper capitalization")
        return {"passed": passed, "details": details}

    def _test_constraints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test if the response respects given constraints."""
        response = data.get("response", "")
        # Check for common constraints
        max_length = 1000
        min_length = 50
        passed = min_length <= len(response) <= max_length
        details = [f"Response length: {len(response)} characters"]
        if len(response) < min_length:
            details.append("Response is too short")
        if len(response) > max_length:
            details.append("Response is too long")
        return {"passed": passed, "details": details}

    def _test_logical_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test if the response has a logical flow."""
        response = data.get("response", "")
        # Check for logical flow indicators
        flow_indicators = ["first", "then", "next", "finally", "therefore", "thus", "consequently"]
        found_indicators = [ind for ind in flow_indicators if ind in response.lower()]
        passed = len(found_indicators) >= 2
        details = [f"Found {len(found_indicators)} flow indicators: {', '.join(found_indicators)}"]
        return {"passed": passed, "details": details}

    def _test_argument_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test if the response has a proper argument structure."""
        response = data.get("response", "")
        # Check for argument structure elements
        has_premise = any(word in response.lower() for word in ["because", "since", "given that"])
        has_conclusion = any(word in response.lower() for word in ["therefore", "thus", "consequently"])
        passed = has_premise and has_conclusion
        details = []
        if not has_premise:
            details.append("Missing premise indicators")
        if not has_conclusion:
            details.append("Missing conclusion indicators")
        return {"passed": passed, "details": details}

    def run_tests(self) -> None:
        """Run all edge case tests."""
        logger.info("Starting edge case tests")
        
        # Load test data
        test_data = self.load_test_data()
        
        # Run tests for each prompt/response pair
        for data in test_data:
            logger.info(f"Testing prompt: {data['prompt'][:50]}...")
            test_result = self.test_logic(data)
            self.test_results.append(test_result)
            
            # Add delay between tests to avoid rate limiting
            time.sleep(0.1)
        
        # Save test results
        self.save_test_results()
        logger.info("Edge case tests completed")

def determine_test_types(prompt: str, response: str, challenge_name: Optional[str] = None, challenge_description: Optional[str] = None) -> Set[str]:
    """
    Determine which tests to run based on the prompt, response, challenge name, and description.
    
    Args:
        prompt: The input prompt
        response: The response to test
        challenge_name: Optional name of the challenge being tested
        challenge_description: Optional description of the challenge
        
    Returns:
        Set of test types to run
    """
    test_types = {"logic"}  # Always run logic tests
    
    # Check for potential information leakage
    if any(word in prompt.lower() for word in ["secret", "confidential", "private", "password", "key"]):
        test_types.add("malformed_message")
    
    # Check for potential network issues
    if len(response) > 1000 or "timeout" in prompt.lower() or "network" in prompt.lower():
        test_types.add("network_failure")
    
    # Check for potential service degradation
    if "slow" in prompt.lower() or "delay" in prompt.lower() or "performance" in prompt.lower():
        test_types.add("service_degradation")
    
    # Check for potential data corruption
    if any(word in prompt.lower() for word in ["corrupt", "malformed", "invalid", "broken"]):
        test_types.add("malformed_message")
    
    # Check challenge-specific test requirements
    if challenge_name:
        challenge_lower = challenge_name.lower()
        
        # Security-related challenges
        if any(word in challenge_lower for word in ["security", "hack", "exploit", "vulnerability"]):
            test_types.add("malformed_message")
            test_types.add("network_failure")
        
        # Performance-related challenges
        if any(word in challenge_lower for word in ["performance", "speed", "latency", "throughput"]):
            test_types.add("service_degradation")
            test_types.add("network_failure")
        
        # Data integrity challenges
        if any(word in challenge_lower for word in ["data", "integrity", "validation", "corruption"]):
            test_types.add("malformed_message")
    
    # Check challenge description for additional test requirements
    if challenge_description:
        desc_lower = challenge_description.lower()
        
        # Security-related descriptions
        if any(word in desc_lower for word in ["security", "hack", "exploit", "vulnerability", "attack", "breach"]):
            test_types.add("malformed_message")
            test_types.add("network_failure")
        
        # Performance-related descriptions
        if any(word in desc_lower for word in ["performance", "speed", "latency", "throughput", "response time", "delay"]):
            test_types.add("service_degradation")
            test_types.add("network_failure")
        
        # Data integrity descriptions
        if any(word in desc_lower for word in ["data", "integrity", "validation", "corruption", "consistency", "accuracy"]):
            test_types.add("malformed_message")
        
        # Logic-related descriptions
        if any(word in desc_lower for word in ["logic", "reasoning", "analysis", "evaluation", "assessment"]):
            test_types.add("logic")
    
    return test_types

def determine_overall_classification(test_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Determine the overall classification based on individual test results.
    
    Args:
        test_results: Dictionary of test results
        
    Returns:
        Overall classification string
    """
    # Collect all classifications
    classifications = [result["classification"] for result in test_results.values()]
    
    # Check for critical failures first
    if any(c in ["failed_refusal", "failed_hallucination", "failed_misinterpretation"] for c in classifications):
        return next(c for c in classifications if c.startswith("failed_"))
    
    # Check for success levels
    if all(c == "success_full" for c in classifications):
        return "success_full"
    elif any(c == "success_full" for c in classifications):
        return "success_partial"
    elif any(c == "success_indirect" for c in classifications):
        return "success_indirect"
    elif any(c == "success_unintended" for c in classifications):
        return "success_unintended"
    
    # Default to failed_other if no other classification matches
    return "failed_other"

def run_prompt_response_tests(prompt_response_pairs: List[Dict[str, str]], output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Run edge case tests on a list of prompt/response pairs.
    
    Args:
        prompt_response_pairs: List of dictionaries containing prompt/response pairs
        output_file: Optional path to save results as JSON
        
    Returns:
        List of test results for each prompt/response pair
    """
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 80)
    print("Gray Swan Edge Case Test Runner")
    print("=" * 80)
    print("Running tests on provided prompt/response pairs")
    print("-" * 80)
    
    # Initialize test runner and mock agent factory
    test_runner = EdgeCaseTestRunner(prompt_response_pairs[0]["prompt"], output_file)
    mock_factory = MockAgentFactory()
    
    # Run tests for each prompt/response pair
    results = []
    for pair in prompt_response_pairs:
        prompt = pair.get("prompt", "")
        response = pair.get("response", "")
        challenge_name = pair.get("challenge_name")
        challenge_description = pair.get("challenge_description")
        
        if not prompt or not response:
            logging.warning(f"Skipping invalid prompt/response pair: {pair}")
            continue
            
        # Determine which tests to run
        test_types = determine_test_types(prompt, response, challenge_name, challenge_description)
        
        # Initialize test result
        test_result = {
            "prompt": prompt,
            "response": response,
            "challenge_name": challenge_name,
            "challenge_description": challenge_description,
            "success": True,
            "classification": "valid",
            "error_details": None,
            "test_results": {}
        }
        
        try:
            # Run selected tests
            if "malformed_message" in test_types:
                test_result["test_results"]["malformed_message"] = test_malformed_message_handling(mock_factory, prompt, response)
            
            if "network_failure" in test_types:
                test_result["test_results"]["network_failure"] = test_intermittent_network_failure(mock_factory, prompt, response)
            
            if "service_degradation" in test_types:
                test_result["test_results"]["service_degradation"] = test_slow_model_responses(mock_factory, prompt, response)
            
            # Always run logic test
            test_result["test_results"]["logic"] = test_logic(mock_factory, prompt, response)
            
            # Determine overall success and classification
            test_result["success"] = all(
                result.get("success", True)
                for result in test_result["test_results"].values()
            )
            
            # Set classification based on test results
            test_result["classification"] = determine_overall_classification(test_result["test_results"])
            
            # Collect error details
            error_details = {}
            for test_type, result in test_result["test_results"].items():
                if not result.get("success", True):
                    error_details[test_type] = result.get("error_details", "Unknown error")
            
            if error_details:
                test_result["error_details"] = error_details
            
        except Exception as e:
            test_result["success"] = False
            test_result["classification"] = "failed_other"
            test_result["error_details"] = str(e)
        
        results.append(test_result)
    
    # Save results to file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            logging.error(f"Failed to save results to file: {e}")
    
    print("-" * 80)
    print("Edge case tests completed.")
    print("=" * 80)

    return results

def main():
    """Run edge case tests."""
    try:
        # Get the latest prompts file
        prompts_dir = "reports/prompts"
        prompts_files = [f for f in os.listdir(prompts_dir) if f.startswith("prompts_") and f.endswith(".json")]
        latest_prompts_file = sorted(prompts_files)[-1]
        input_file = os.path.join(prompts_dir, latest_prompts_file)
        
        # Set up output file
        output_dir = "reports/tests"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "edge_case_test_results.json")
        
        # Initialize test runner
        test_runner = EdgeCaseTestRunner(input_file, output_file)
        
        # Load test data
        test_data = test_runner.load_test_data()
        
        # Run tests
        for data in test_data:
            result = test_runner.test_logic(data)
            test_runner.test_results.append(result)
        
        # Save results
        test_runner.save_test_results()
        
        logger.info("Edge case tests completed successfully")
    except Exception as e:
        logger.error(f"Error running edge case tests: {e}")
        raise

if __name__ == "__main__":
    main() 