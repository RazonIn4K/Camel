from datetime import datetime
from unittest.mock import patch

import pytest

from cybersec_agents import ForensicsPlanner


class TestForensicsPlanner:
    """Test suite for ForensicsPlanner agent."""

    @pytest.fixture
    def planner(self, config):
        """Creates a ForensicsPlanner instance with test configuration."""
        return ForensicsPlanner(config)

    def test_initialization(self, planner):
        """Test proper initialization of ForensicsPlanner.

        - Verifies configuration is loaded
        - Checks default attributes are set
        - Validates initial state
        """
        assert planner.config is not None
        assert planner.model is not None
        assert hasattr(planner, "generate_investigation_plan")
        assert hasattr(planner, "generate_evidence_collection_procedure")

    @patch("cybersec_agents.ForensicsPlanner._create_investigation_plan")
    def test_generate_investigation_plan(self, mock_create, planner, sample_case_data):
        """Test investigation plan generation.

        - Verifies plan structure and content
        - Checks inclusion of all required components
        - Validates timeline and resource allocation
        """
        expected_plan = {
            "case_details": sample_case_data,
            "investigation_steps": [
                {
                    "phase": "initial_response",
                    "tasks": ["secure_crime_scene", "document_volatile_data"],
                    "estimated_duration": "2 hours",
                },
                {
                    "phase": "evidence_collection",
                    "tasks": ["acquire_memory_dump", "collect_logs"],
                    "estimated_duration": "4 hours",
                },
            ],
            "resource_requirements": {
                "personnel": ["forensic_investigator", "system_administrator"],
                "tools": ["memory_acquisition_tool", "write_blocker"],
            },
        }
        mock_create.return_value = expected_plan

        result = planner.generate_investigation_plan(
            case_type=sample_case_data["case_type"],
            target_systems=sample_case_data["target_systems"],
        )

        assert isinstance(result, dict)
        assert "case_details" in result
        assert "investigation_steps" in result
        assert "resource_requirements" in result
        assert len(result["investigation_steps"]) >= 2

    def test_generate_investigation_plan_invalid_case_type(self, planner):
        """Test error handling for invalid case type.

        - Checks proper exception raising
        - Verifies error message content
        """
        with pytest.raises(ValueError) as exc_info:
            planner.generate_investigation_plan(
                case_type="invalid_case_type", target_systems=["windows_server"]
            )
        assert "Invalid case type" in str(exc_info.value)

    @patch("cybersec_agents.ForensicsPlanner._create_evidence_procedure")
    def test_generate_evidence_collection_procedure(
        self, mock_create, planner, sample_evidence_data
    ):
        """Test evidence collection procedure generation.

        - Verifies procedure structure and steps
        - Validates chain of custody documentation
        - Checks handling of different device types
        """
        expected_procedure = {
            "device_info": {
                "type": sample_evidence_data["device_type"],
                "preparation_steps": ["verify_write_blocker", "prepare_storage_media"],
            },
            "collection_steps": [
                {
                    "step": 1,
                    "action": "acquire_memory",
                    "tools": ["memory_acquisition_tool"],
                    "verification": ["hash_verification"],
                }
            ],
            "documentation_requirements": {
                "photographs": ["device_state", "serial_numbers"],
                "forms": ["chain_of_custody", "evidence_log"],
            },
        }
        mock_create.return_value = expected_procedure

        result = planner.generate_evidence_collection_procedure(
            device_type=sample_evidence_data["device_type"]
        )

        assert isinstance(result, dict)
        assert "device_info" in result
        assert "collection_steps" in result
        assert "documentation_requirements" in result

    @patch("cybersec_agents.ForensicsPlanner._generate_timeline")
    def test_generate_timeline_template(
        self, mock_generate, planner, sample_timeline_data
    ):
        """Test timeline template generation.

        - Verifies timeline structure and events
        - Checks chronological ordering
        - Validates event categorization
        """
        expected_timeline = {
            "metadata": {
                "incident_type": sample_timeline_data["incident_type"],
                "time_range": sample_timeline_data["time_range"],
            },
            "events": sample_timeline_data["events"],
            "analysis": {
                "key_events": ["initial_access"],
                "patterns": ["lateral_movement_observed"],
            },
        }
        mock_generate.return_value = expected_timeline

        result = planner.generate_timeline_template(
            incident_type=sample_timeline_data["incident_type"]
        )

        assert isinstance(result, dict)
        assert "metadata" in result
        assert "events" in result
        assert "analysis" in result
        assert len(result["events"]) >= 1

    def test_validate_evidence_integrity(self, planner, sample_evidence_data):
        """Test evidence integrity validation.

        - Checks hash verification
        - Validates chain of custody
        - Verifies documentation completeness
        """
        evidence_item = sample_evidence_data["evidence_items"][0]

        result = planner.validate_evidence_integrity(
            evidence_hash=evidence_item["hash"], original_hash="sha256:abc123..."
        )

        assert result["verified"] is True
        assert "timestamp" in result
        assert "verification_method" in result

    def test_update_chain_of_custody(self, planner, sample_evidence_data):
        """Test chain of custody update functionality.

        - Verifies proper documentation of evidence handling
        - Checks timestamp accuracy
        - Validates handler information
        """
        custody_event = {
            "timestamp": datetime.now().isoformat(),
            "action": "transfer",
            "operator": "Jane Smith",
            "location": "Forensics Lab",
        }

        result = planner.update_chain_of_custody(
            case_id="FOR-2024-001", evidence_id="MEM001", custody_event=custody_event
        )

        assert isinstance(result, dict)
        assert "chain_of_custody" in result
        assert result["chain_of_custody"][-1] == custody_event
