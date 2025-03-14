from unittest.mock import Mock, patch

import pytest

from cybersec_agents import NetworkSecurityAgent


@pytest.fixture
def mock_credentials():
    with patch("cybersec_agents.utils.credentials.CredentialManager") as mock:
        mock_instance = Mock()
        mock_instance.get_credential.return_value = "test-api-key"
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def network_agent(mock_credentials):
    return NetworkSecurityAgent(provider="anthropic")


def test_model_initialization(network_agent):
    assert network_agent.provider == "anthropic"
    assert network_agent.api_key == "test-api-key"
