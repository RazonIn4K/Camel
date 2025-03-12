from unittest.mock import patch

import pytest

from cybersec_agents.core.service_wrapper import CyberSecurityService


@pytest.fixture
def mock_config():
    return {
        "model": {
            "provider": "anthropic",
            "anthropic": {"model_name": "claude-3-7-sonnet-20250219"},
        }
    }


@pytest.fixture
def service(mock_config):
    with patch(
        "cybersec_agents.core.service_wrapper.CyberSecurityService._load_config"
    ) as mock_load:
        mock_load.return_value = mock_config
        service = CyberSecurityService()
        yield service


def test_service_initialization(service):
    assert service.config["model"]["provider"] == "anthropic"
    assert service.agents is not None
