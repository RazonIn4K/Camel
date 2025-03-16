"""
Tests for new model integrations in Gray Swan Arena.

This module contains tests for the integration of new AI models into the
Gray Swan Arena framework.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from cybersec_agents.camel.types import ModelType, ModelPlatformType
from cybersec_agents.grayswan.agents.exploit_delivery_agent import ExploitDeliveryAgent

# Mock responses for different models
MOCK_GEMINI_RESPONSE = "Paris is the capital of France."
MOCK_CLAUDE_RESPONSE = "Madrid is the capital of Spain."
MOCK_GPT_RESPONSE = "Rome is the capital of Italy."

@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "test_google_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
            "OPENAI_API_KEY": "test_openai_key",
        },
    ):
        yield

@pytest.mark.usefixtures("mock_env_vars")
def test_gemini_integration():
    """Test the Gemini model integration."""
    try:
        # Mock the Google API client
        with patch('google.generativeai.GenerativeModel') as mock_gemini:
            mock_model = MagicMock()
            mock_model.generate_content.return_value.text = MOCK_GEMINI_RESPONSE
            mock_gemini.return_value = mock_model

            agent = ExploitDeliveryAgent(
                model_type=ModelType.GEMINI_PRO,
                model_platform=ModelPlatformType.GOOGLE,
            )
            prompt = "What is the capital of France?"
            results = agent.execute_prompt_batch([prompt])
            
            assert results, "No results returned from Gemini."
            for result in results:
                assert "error" not in result, f"Error in Gemini response: {result.get('error')}"
                assert result["response"] == MOCK_GEMINI_RESPONSE, f"Unexpected response from Gemini: {result}"
    except Exception as e:
        pytest.fail(f"Gemini integration test failed: {e}")

@pytest.mark.usefixtures("mock_env_vars")
def test_claude_integration():
    """Test the Claude model integration."""
    try:
        # Mock the Anthropic API client
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value.content = MOCK_CLAUDE_RESPONSE
            mock_anthropic.return_value = mock_client

            agent = ExploitDeliveryAgent(
                model_type=ModelType.CLAUDE_3_SONNET,
                model_platform=ModelPlatformType.ANTHROPIC,
            )
            prompt = "What is the capital of Spain?"
            results = agent.execute_prompt_batch([prompt])
            
            assert results, "No results returned from Claude."
            for result in results:
                assert "error" not in result, f"Error in Claude response: {result.get('error')}"
                assert result["response"] == MOCK_CLAUDE_RESPONSE, f"Unexpected response from Claude: {result}"
    except Exception as e:
        pytest.fail(f"Claude integration test failed: {e}")

@pytest.mark.usefixtures("mock_env_vars")
def test_gpt4_integration():
    """Test the GPT-4 model integration."""
    try:
        # Mock the OpenAI API client
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT_RESPONSE
            mock_openai.return_value = mock_client

            agent = ExploitDeliveryAgent(
                model_type=ModelType.GPT_4,
                model_platform=ModelPlatformType.OPENAI,
            )
            prompt = "What is the capital of Italy?"
            results = agent.execute_prompt_batch([prompt])
            
            assert results, "No results returned from GPT-4."
            for result in results:
                assert "error" not in result, f"Error in GPT-4 response: {result.get('error')}"
                assert result["response"] == MOCK_GPT_RESPONSE, f"Unexpected response from GPT-4: {result}"
    except Exception as e:
        pytest.fail(f"GPT-4 integration test failed: {e}")

@pytest.mark.usefixtures("mock_env_vars")
def test_rate_limit_handling():
    """Test rate limit handling for all models."""
    try:
        # Test Gemini rate limit
        with patch('google.generativeai.GenerativeModel') as mock_gemini:
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("Rate limit exceeded")
            mock_gemini.return_value = mock_model

            agent = ExploitDeliveryAgent(
                model_type=ModelType.GEMINI_PRO,
                model_platform=ModelPlatformType.GOOGLE,
            )
            prompt = "Test prompt"
            results = agent.execute_prompt_batch([prompt])
            
            assert results, "No results returned after rate limit handling."
            for result in results:
                assert "error" in result, "Expected error in result after rate limit"
                assert "Rate limit exceeded" in result["error"], "Expected rate limit error message"

        # Test Claude rate limit
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("Rate limit exceeded")
            mock_anthropic.return_value = mock_client

            agent = ExploitDeliveryAgent(
                model_type=ModelType.CLAUDE_3_SONNET,
                model_platform=ModelPlatformType.ANTHROPIC,
            )
            results = agent.execute_prompt_batch([prompt])
            
            assert results, "No results returned after rate limit handling."
            for result in results:
                assert "error" in result, "Expected error in result after rate limit"
                assert "Rate limit exceeded" in result["error"], "Expected rate limit error message"

        # Test GPT-4 rate limit
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
            mock_openai.return_value = mock_client

            agent = ExploitDeliveryAgent(
                model_type=ModelType.GPT_4,
                model_platform=ModelPlatformType.OPENAI,
            )
            results = agent.execute_prompt_batch([prompt])
            
            assert results, "No results returned after rate limit handling."
            for result in results:
                assert "error" in result, "Expected error in result after rate limit"
                assert "Rate limit exceeded" in result["error"], "Expected rate limit error message"
    except Exception as e:
        pytest.fail(f"Rate limit handling test failed: {e}") 