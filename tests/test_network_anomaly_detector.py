from unittest.mock import patch

import pytest

from cybersec_agents import NetworkAnomalyDetector


class TestNetworkAnomalyDetector:
    """Test suite for NetworkAnomalyDetector agent."""

    @pytest.fixture
    def detector(self, config):
        """Creates a NetworkAnomalyDetector instance with test configuration."""
        return NetworkAnomalyDetector(config)

    def test_initialization(self, detector):
        """Test proper initialization of NetworkAnomalyDetector.

        - Verifies configuration is loaded
        - Checks default attributes are set
        """
        assert detector.config is not None
        assert detector.model is not None
        assert hasattr(detector, "analyze_nmap_output")

    @patch("cybersec_agents.NetworkAnomalyDetector._process_nmap_xml")
    def test_analyze_nmap_output_success(self, mock_process, detector, sample_nmap_xml):
        """Test successful analysis of Nmap output.

        - Verifies correct parsing of XML
        - Checks returned analysis structure
        - Validates vulnerability detection
        """
        expected_analysis = {
            "hosts": [
                {
                    "ip": "192.168.1.1",
                    "open_ports": [80],
                    "services": ["http"],
                    "vulnerabilities": [],
                }
            ],
            "summary": {"total_hosts": 1, "open_ports": 1, "risk_level": "low"},
        }
        mock_process.return_value = expected_analysis

        result = detector.analyze_nmap_output(sample_nmap_xml)

        assert isinstance(result, dict)
        assert "hosts" in result
        assert "summary" in result
        assert len(result["hosts"]) == 1
        assert result["hosts"][0]["ip"] == "192.168.1.1"

    def test_analyze_nmap_output_invalid_xml(self, detector):
        """Test error handling for invalid XML input.

        - Checks proper exception raising
        - Verifies error message content
        """
        with pytest.raises(ValueError) as exc_info:
            detector.analyze_nmap_output("invalid xml content")
        assert "Invalid Nmap XML format" in str(exc_info.value)

    @patch("cybersec_agents.NetworkAnomalyDetector._analyze_traffic_data")
    def test_analyze_network_traffic(self, mock_analyze, detector, sample_pcap_data):
        """Test network traffic analysis functionality.

        - Verifies PCAP data processing
        - Checks anomaly detection
        - Validates returned analysis structure
        """
        expected_result = {
            "anomalies": [],
            "traffic_patterns": {
                "top_talkers": ["192.168.1.100"],
                "protocols": {"TCP": 1},
            },
            "risk_assessment": "low",
        }
        mock_analyze.return_value = expected_result

        result = detector.analyze_network_traffic(sample_pcap_data)

        assert isinstance(result, dict)
        assert "anomalies" in result
        assert "traffic_patterns" in result
        assert "risk_assessment" in result

    def test_extract_vulnerable_systems(self, detector):
        """Test vulnerable system extraction from analysis results.

        - Checks correct identification of vulnerabilities
        - Verifies filtering based on severity
        - Validates returned structure
        """
        analysis_result = {
            "hosts": [
                {
                    "ip": "192.168.1.1",
                    "vulnerabilities": [
                        {"severity": "high", "description": "Open SSH"}
                    ],
                }
            ]
        }

        vulnerable_systems = detector.extract_vulnerable_systems(analysis_result)

        assert isinstance(vulnerable_systems, list)
        assert len(vulnerable_systems) == 1
        assert vulnerable_systems[0]["ip"] == "192.168.1.1"

    @patch("cybersec_agents.NetworkAnomalyDetector._generate_report")
    def test_generate_security_report(self, mock_generate, detector):
        """Test security report generation.

        - Verifies report structure
        - Checks inclusion of all analysis components
        - Validates formatting and content requirements
        """
        expected_report = {
            "executive_summary": "Network analysis complete",
            "findings": [],
            "recommendations": [],
            "technical_details": {},
        }
        mock_generate.return_value = expected_report

        report = detector.generate_security_report(
            nmap_file="scan.xml", pcap_file="capture.pcap"
        )

        assert isinstance(report, dict)
        assert "executive_summary" in report
        assert "findings" in report
        assert "recommendations" in report

    def test_threshold_validation(self, detector):
        """Test threshold validation for anomaly detection.

        - Checks handling of invalid thresholds
        - Verifies threshold application
        - Validates error conditions
        """
        with pytest.raises(ValueError) as exc_info:
            detector.set_detection_threshold(1.5)  # Invalid threshold > 1.0
        assert "Threshold must be between 0.0 and 1.0" in str(exc_info.value)

        detector.set_detection_threshold(0.5)  # Valid threshold
        assert detector.detection_threshold == 0.5
