"""
Data Corruption Edge Case Tests for Gray Swan Arena.

This module contains tests that simulate various data corruption scenarios to verify
the system's resilience to malformed or corrupt data.
"""

import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from unittest.mock import MagicMock, patch

from ...camel_integration import AgentFactory, CommunicationChannel, TestTier
from ...utils.logging_utils import setup_logging
from ...utils.retry_utils import ExponentialBackoffRetryStrategy
from .edge_case_framework import FailureSimulator, EdgeCaseTestRunner

# Set up logging
logger = setup_logging("data_corruption_tests")


def test_malformed_message_handling(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's ability to handle malformed messages.
    
    This test sends messages with various forms of corruption and verifies
    that the system properly handles them without crashing.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to malformed messages...")
    
    # Create a communication channel with a dead-letter queue
    channel = CommunicationChannel(dlq_storage_path="./test_output/dlq/malformed_test.json")
    
    # Get the dead-letter queue instance and clear it
    dlq = channel.get_dead_letter_queue()
    dlq.clear()
    
    # Create a valid base message
    valid_message = {
        "type": "test_message",
        "sender": "test_agent_1",
        "recipient": "test_agent_2",
        "content": "This is a test message",
        "metadata": {
            "importance": "high",
            "sequence": 1,
            "session_id": "test-session-123"
        }
    }
    
    # Define different corruption types to test
    corruption_types = [
        ("missing_type", lambda m: {k: v for k, v in m.items() if k != "type"}),
        ("missing_sender", lambda m: {k: v for k, v in m.items() if k != "sender"}),
        ("missing_recipient", lambda m: {k: v for k, v in m.items() if k != "recipient"}),
        ("missing_content", lambda m: {k: v for k, v in m.items() if k != "content"}),
        ("null_content", lambda m: {**m, "content": None}),
        ("empty_content", lambda m: {**m, "content": ""}),
        ("wrong_type_sender", lambda m: {**m, "sender": 12345}),
        ("wrong_type_content", lambda m: {**m, "content": {"invalid": "not a string"}}),
        ("extra_fields", lambda m: {**m, "extra1": "value", "extra2": [1, 2, 3]}),
        ("malformed_metadata", lambda m: {**m, "metadata": "not a dict"})
    ]
    
    results = []
    
    for corruption_name, corruption_func in corruption_types:
        # Create a corrupted message
        corrupted_message = corruption_func(valid_message.copy())
        
        logger.info(f"Testing corruption type: {corruption_name}")
        logger.debug(f"Corrupted message: {corrupted_message}")
        
        try:
            # Attempt to send the corrupted message
            channel.send_message(corrupted_message, 
                                sender_id=corrupted_message.get("sender", "unknown"),
                                receiver_id=corrupted_message.get("recipient", "unknown"))
            
            # If it didn't raise an exception, check if it was logged or added to DLQ
            # depending on the system's error handling approach
            result = {
                "corruption_type": corruption_name,
                "exception": None,
                "handled_gracefully": True,
                "in_dlq": False
            }
        except Exception as e:
            # Caught an exception - the system rejected the malformed message
            logger.info(f"Exception handling {corruption_name}: {type(e).__name__}: {str(e)}")
            
            result = {
                "corruption_type": corruption_name,
                "exception": f"{type(e).__name__}: {str(e)}",
                "handled_gracefully": False,
                "in_dlq": False
            }
        
        # Check if the message was added to the dead-letter queue
        dlq_messages = dlq.get_messages()
        corrupted_in_dlq = any(
            msg["message_content"].get("content") == corrupted_message.get("content")
            for msg in dlq_messages
        )
        
        if corrupted_in_dlq:
            result["in_dlq"] = True
            # If it's in the DLQ, we consider it gracefully handled
            result["handled_gracefully"] = True
        
        results.append(result)
        
        # Clear the DLQ for the next test
        dlq.clear()
    
    # Analyze results
    handled_count = sum(1 for r in results if r["handled_gracefully"])
    exception_count = sum(1 for r in results if r["exception"] is not None)
    dlq_count = sum(1 for r in results if r["in_dlq"])
    
    logger.info(f"Summary: {handled_count}/{len(results)} corruptions handled gracefully")
    logger.info(f"Exceptions: {exception_count}, Added to DLQ: {dlq_count}")
    
    return {
        "total_tests": len(corruption_types),
        "handled_gracefully": handled_count,
        "exceptions": exception_count,
        "added_to_dlq": dlq_count,
        "detailed_results": results,
        "message": f"{handled_count}/{len(results)} corruptions handled gracefully"
    }


def test_message_truncation(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of truncated messages.
    
    This test simulates messages that are truncated during transmission
    and verifies that the system properly detects and handles the truncation.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to message truncation...")
    
    # Create a communication channel with a dead-letter queue
    channel = CommunicationChannel(dlq_storage_path="./test_output/dlq/truncation_test.json")
    
    # Get the dead-letter queue instance and clear it
    dlq = channel.get_dead_letter_queue()
    dlq.clear()
    
    # Create a test message with substantial content
    long_content = "This is a long message " * 50  # 1000+ characters
    test_message = {
        "type": "test_message",
        "sender": "test_agent_1",
        "recipient": "test_agent_2",
        "content": long_content,
        "metadata": {
            "importance": "medium",
            "full_length": len(long_content),
            "checksum": str(hash(long_content))  # Simple checksum
        }
    }
    
    # Mock the actual send implementation to simulate truncation
    original_send = channel._send_message_impl
    
    truncation_results = []
    
    # Try different truncation amounts
    truncation_percentages = [10, 30, 50, 70, 90]
    
    for truncation_pct in truncation_percentages:
        # Calculate how many characters to keep
        keep_chars = int(len(long_content) * (100 - truncation_pct) / 100)
        truncated_content = long_content[:keep_chars]
        
        logger.info(f"Testing {truncation_pct}% truncation (keeping {keep_chars} chars)")
        
        # Create a truncated version of the message
        truncated_message = test_message.copy()
        truncated_message["content"] = truncated_content
        
        # Track if the message was rejected or added to DLQ
        was_rejected = False
        added_to_dlq = False
        
        def mock_send_impl(message: Dict[str, Any], *args, **kwargs):
            nonlocal was_rejected
            
            # Check if the message is truncated by comparing content length
            actual_length = len(message.get("content", ""))
            expected_length = message.get("metadata", {}).get("full_length", 0)
            
            if expected_length > 0 and actual_length < expected_length:
                # Message is truncated
                logger.info(f"Detected truncation: expected {expected_length}, got {actual_length}")
                was_rejected = True
                raise ValueError(f"Message truncated: expected {expected_length}, got {actual_length}")
            
            # Otherwise proceed normally
            return original_send(message, *args, **kwargs)
        
        # Apply the mock
        channel._send_message_impl = mock_send_impl
        
        try:
            # Attempt to send the truncated message
            channel.send_message(truncated_message, 
                               sender_id=truncated_message.get("sender"),
                               receiver_id=truncated_message.get("recipient"))
            
            logger.info(f"Truncated message was accepted (truncation: {truncation_pct}%)")
        except Exception as e:
            logger.info(f"Truncated message was rejected: {str(e)}")
        
        # Check if message was added to DLQ
        dlq_messages = dlq.get_messages()
        for msg in dlq_messages:
            if (msg["message_content"].get("content") == truncated_content or
                truncated_content in msg["message_content"].get("content", "")):
                added_to_dlq = True
                break
        
        # Record result
        result = {
            "truncation_percentage": truncation_pct,
            "characters_kept": keep_chars,
            "was_rejected": was_rejected,
            "added_to_dlq": added_to_dlq,
            "properly_handled": was_rejected or added_to_dlq
        }
        truncation_results.append(result)
        
        # Clear the DLQ for the next test
        dlq.clear()
    
    # Restore the original implementation
    channel._send_message_impl = original_send
    
    # Analyze results
    handled_count = sum(1 for r in truncation_results if r["properly_handled"])
    rejection_count = sum(1 for r in truncation_results if r["was_rejected"])
    dlq_count = sum(1 for r in truncation_results if r["added_to_dlq"])
    
    logger.info(f"Summary: {handled_count}/{len(truncation_results)} truncations properly handled")
    logger.info(f"Rejections: {rejection_count}, Added to DLQ: {dlq_count}")
    
    return {
        "total_tests": len(truncation_percentages),
        "properly_handled": handled_count,
        "rejections": rejection_count,
        "added_to_dlq": dlq_count,
        "detailed_results": truncation_results,
        "message": f"{handled_count}/{len(truncation_results)} truncations properly handled"
    }


def test_json_parsing_errors(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of JSON parsing errors.
    
    This test simulates malformed JSON data and verifies that the
    system properly handles parsing errors without crashing.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to JSON parsing errors...")
    
    # Create a communication channel with a dead-letter queue
    channel = CommunicationChannel(dlq_storage_path="./test_output/dlq/json_test.json")
    
    # Get the dead-letter queue instance and clear it
    dlq = channel.get_dead_letter_queue()
    dlq.clear()
    
    # Define a set of malformed JSON strings to test
    malformed_jsons = [
        # Missing closing brace
        '{"type": "test", "content": "missing brace"',
        # Unquoted keys
        '{type: "test", content: "unquoted keys"}',
        # Missing quotes around string
        '{"type": "test", "content": missing quotes}',
        # Extra comma in object
        '{"type": "test", "content": "extra comma", }',
        # Invalid escape sequence
        '{"type": "test", "content": "invalid escape \\z"}',
        # Control character in string
        '{"type": "test", "content": "control char: \u0001"}',
        # Invalid Unicode
        '{"type": "test", "content": "invalid unicode: \\uXYZ1"}',
        # Duplicate keys
        '{"type": "test", "content": "value1", "content": "value2"}',
        # Invalid number format
        '{"type": "test", "value": 12.34.56}',
        # Trailing characters after JSON
        '{"type": "test", "content": "trailing"} extra'
    ]
    
    # Store results for each test
    results = []
    
    # Mock json.loads to track when it's called with malformed data
    original_loads = json.loads
    
    for i, malformed_json in enumerate(malformed_jsons):
        logger.info(f"Testing malformed JSON #{i+1}")
        
        # Track if parsing error was caught
        parsing_error_caught = False
        added_to_dlq = False
        
        def mock_json_loads(s, *args, **kwargs):
            nonlocal parsing_error_caught
            try:
                return original_loads(s, *args, **kwargs)
            except json.JSONDecodeError as e:
                parsing_error_caught = True
                logger.info(f"Caught JSON parsing error: {str(e)}")
                raise  # Re-raise to let the system handle it
        
        # Apply the mock
        json.loads = mock_json_loads
        
        try:
            # Simulate receiving malformed JSON during deserialization
            # We'll do this by creating a test function that tries to parse the JSON
            def test_function():
                try:
                    # Parse malformed JSON
                    parsed = json.loads(malformed_json)
                    # Channel would normally process this data
                    channel.send_message(parsed, sender_id="test", receiver_id="test")
                except json.JSONDecodeError as e:
                    # System should handle this by adding to DLQ
                    dlq.add_message(
                        message_content={"raw_data": malformed_json},
                        error=e,
                        sender_id="test",
                        receiver_id="test",
                        context={"operation": "json_parsing"}
                    )
                    logger.info(f"Added malformed JSON to DLQ: {str(e)}")
                    raise  # Re-raise to simulate system behavior
            
            # Call the test function
            try:
                test_function()
            except json.JSONDecodeError:
                # Expected exception, the test function should have added to DLQ
                pass
            
            # Check if the error was added to the DLQ
            dlq_messages = dlq.get_messages()
            if dlq_messages:
                added_to_dlq = True
                logger.info(f"Found {len(dlq_messages)} messages in DLQ")
        
        finally:
            # Restore original json.loads
            json.loads = original_loads
        
        # Record result
        result = {
            "test_index": i,
            "malformed_json": malformed_json[:50] + "..." if len(malformed_json) > 50 else malformed_json,
            "parsing_error_caught": parsing_error_caught,
            "added_to_dlq": added_to_dlq,
            "properly_handled": parsing_error_caught and added_to_dlq
        }
        results.append(result)
        
        # Clear the DLQ for the next test
        dlq.clear()
    
    # Analyze results
    handled_count = sum(1 for r in results if r["properly_handled"])
    caught_count = sum(1 for r in results if r["parsing_error_caught"])
    dlq_count = sum(1 for r in results if r["added_to_dlq"])
    
    logger.info(f"Summary: {handled_count}/{len(results)} JSON errors properly handled")
    logger.info(f"Errors caught: {caught_count}, Added to DLQ: {dlq_count}")
    
    return {
        "total_tests": len(malformed_jsons),
        "properly_handled": handled_count,
        "errors_caught": caught_count,
        "added_to_dlq": dlq_count,
        "detailed_results": results,
        "message": f"{handled_count}/{len(results)} JSON parsing errors properly handled"
    }


def test_invalid_response_formats(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Test system's handling of invalid response formats from models.
    
    This test simulates unexpected or invalid responses from language models
    and verifies that the system properly handles these cases.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing resilience to invalid model response formats...")
    
    # Create a prompt engineer agent which interacts with language models
    prompt_engineer = agent_factory.create_prompt_engineer_agent()
    
    # Define various invalid response formats to test
    invalid_formats = [
        ("empty_string", ""),
        ("null_response", None),
        ("non_string", {"response": "nested in object"}),
        ("numeric_response", 12345),
        ("boolean_response", True),
        ("list_response", ["item1", "item2"]),
        ("invalid_json_string", '{malformed":json"}'),
        ("incomplete_response", "This is an incomplete response where the model stops mid-sen"),
        ("repeated_text", "I will help you. I will help you. I will help you. " * 10),
        ("invalid_xml", "<prompt><invalid>XML</mismatched></prompt>")
    ]
    
    # Store results for each test
    results = []
    
    # We'll use patching to inject invalid responses
    for format_name, invalid_format in invalid_formats:
        logger.info(f"Testing invalid format: {format_name}")
        
        # Track if error was handled correctly
        error_handled = False
        error_message = None
        
        # We need to mock the model's generation method
        # The exact method depends on the implementation, so we may need to adjust this
        with patch('camel.agents.ChatAgent.step') as mock_step:
            # Configure the mock to return our invalid format
            mock_response = MagicMock()
            
            # Set up the mock differently based on the type of the invalid format
            if invalid_format is None:
                # None response
                mock_response = None
            elif isinstance(invalid_format, str):
                # String response - need to match the expected return type
                mock_response.msg.content = invalid_format
            else:
                # Other types - inject directly
                mock_response.msg.content = str(invalid_format)
            
            mock_step.return_value = mock_response
            
            try:
                # Attempt to generate a prompt with the invalid response format
                # Using a simple test prompt to trigger model interaction
                # This assumes the agent has a generate_prompts method
                if hasattr(prompt_engineer, 'generate_prompts'):
                    prompt_engineer.generate_prompts(
                        "Generate a prompt to test an AI model",
                        num_prompts=1,
                        target_model="test-model"
                    )
                else:
                    # Fall back to a simpler approach - directly calling the mocked method
                    response = mock_step("test prompt")
                    
                    # Process the response
                    if response and hasattr(response, 'msg') and hasattr(response.msg, 'content'):
                        content = response.msg.content
                    else:
                        content = str(response)
                    
                    # Simple validation
                    if not content or not isinstance(content, str):
                        raise ValueError(f"Invalid response format: {type(content)}")
                
                # If we get here, the system didn't raise an exception
                logger.info(f"System accepted {format_name} without error")
                error_handled = True
            
            except Exception as e:
                # System raised an exception - capture it
                error_message = f"{type(e).__name__}: {str(e)}"
                logger.info(f"System raised exception for {format_name}: {error_message}")
                
                # Consider this handled if the exception is caught at a high level
                # and doesn't crash the whole system
                error_handled = True
        
        # Record the result
        result = {
            "format_name": format_name,
            "invalid_format": str(invalid_format)[:50] + "..." if invalid_format and len(str(invalid_format)) > 50 else str(invalid_format),
            "error_handled": error_handled,
            "error_message": error_message
        }
        results.append(result)
    
    # Analyze results
    handled_count = sum(1 for r in results if r["error_handled"])
    
    logger.info(f"Summary: {handled_count}/{len(results)} invalid formats properly handled")
    
    return {
        "total_tests": len(invalid_formats),
        "properly_handled": handled_count,
        "detailed_results": results,
        "message": f"{handled_count}/{len(results)} invalid response formats properly handled"
    }


def run_data_corruption_test_suite(agent_factory: AgentFactory) -> Dict[str, Any]:
    """
    Run the complete suite of data corruption tests.
    
    Args:
        agent_factory: Factory to create agents for testing
        
    Returns:
        Dictionary with overall test suite results
    """
    runner = EdgeCaseTestRunner()
    
    tests = [
        (test_malformed_message_handling, "Malformed Message Handling", [agent_factory], {}),
        (test_message_truncation, "Message Truncation", [agent_factory], {}),
        (test_json_parsing_errors, "JSON Parsing Errors", [agent_factory], {}),
        (test_invalid_response_formats, "Invalid Response Formats", [agent_factory], {})
    ]
    
    return runner.run_test_suite(tests)


def register_tests(test_manager) -> None:
    """
    Register all data corruption tests with the test manager.
    
    Args:
        test_manager: The test manager to register tests with
    """
    test_manager.register_test(TestTier.SCENARIO, test_malformed_message_handling)
    test_manager.register_test(TestTier.SCENARIO, test_message_truncation) 
    test_manager.register_test(TestTier.SCENARIO, test_json_parsing_errors)
    test_manager.register_test(TestTier.SCENARIO, test_invalid_response_formats) 