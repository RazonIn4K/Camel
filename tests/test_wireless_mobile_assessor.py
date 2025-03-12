from datetime import datetime
from unittest.mock import patch

import pytest

from cybersec_agents import WirelessMobileSecurityAssessor


class TestWirelessMobileSecurityAssessor:
    """Test suite for WirelessMobileSecurityAssessor agent."""

    @pytest.fixture
    def wifi_scan_data(self):
        """Sample WiFi network scan data."""
        return {
            "networks": [
                {
                    "ssid": "Corp-Network",
                    "bssid": "00:11:22:33:44:55",
                    "encryption": "WPA2-Enterprise",
                    "signal_strength": -65,
                    "channel": 6,
                    "clients": ["aa:bb:cc:dd:ee:ff"],
                },
                {
                    "ssid": "Guest-WiFi",
                    "bssid": "00:11:22:33:44:56",
                    "encryption": "WPA2-Personal",
                    "signal_strength": -70,
                    "channel": 11,
                    "clients": [],
                },
            ],
            "scan_time": datetime.now().isoformat(),
            "location": "Main Office",
        }

    @pytest.fixture
    def mobile_app_data(self):
        """Sample mobile application security data."""
        return {
            "app_name": "SecureChat",
            "version": "1.2.3",
            "platform": "ios",
            "permissions": ["camera", "microphone", "location", "contacts"],
            "network_access": True,
            "data_storage": {"encryption": True, "backup": False, "cloud_sync": True},
        }

    @pytest.fixture
    def assessor(self, config):
        """Creates a WirelessMobileSecurityAssessor instance."""
        return WirelessMobileSecurityAssessor(config)

    def test_initialization(self, assessor):
        """Test proper initialization of the assessor."""
        assert assessor.config is not None
        assert assessor.model is not None
        assert hasattr(assessor, "analyze_wireless_network")
        assert hasattr(assessor, "assess_mobile_app_security")

    @patch("cybersec_agents.WirelessMobileSecurityAssessor._analyze_network")
    def test_analyze_wireless_network(self, mock_analyze, assessor, wifi_scan_data):
        """Test wireless network security analysis."""
        expected_analysis = {
            "network_security": {
                "encryption_assessment": {
                    "Corp-Network": "STRONG",
                    "Guest-WiFi": "MEDIUM",
                },
                "vulnerabilities": [
                    {
                        "network": "Guest-WiFi",
                        "type": "weak_password_policy",
                        "severity": "MEDIUM",
                    }
                ],
                "recommendations": [
                    "Enable WPA3 on supported devices",
                    "Implement network segmentation",
                ],
            },
            "rogue_ap_detection": {
                "suspicious_networks": [],
                "last_scan": wifi_scan_data["scan_time"],
            },
        }
        mock_analyze.return_value = expected_analysis

        result = assessor.analyze_wireless_network(
            scan_data=wifi_scan_data, network_type="corporate"
        )

        assert isinstance(result, dict)
        assert "network_security" in result
        assert "rogue_ap_detection" in result
        assert len(result["network_security"]["recommendations"]) > 0

    def test_analyze_wireless_network_invalid_data(self, assessor):
        """Test error handling for invalid wireless scan data."""
        with pytest.raises(ValueError) as exc_info:
            assessor.analyze_wireless_network(scan_data={}, network_type="corporate")
        assert "Invalid scan data" in str(exc_info.value)

    @patch("cybersec_agents.WirelessMobileSecurityAssessor._assess_app")
    def test_assess_mobile_app_security(self, mock_assess, assessor, mobile_app_data):
        """Test mobile application security assessment."""
        expected_assessment = {
            "app_security_score": 85,
            "permission_analysis": {
                "high_risk": ["location", "contacts"],
                "medium_risk": ["camera", "microphone"],
                "low_risk": [],
            },
            "data_security": {
                "encryption_status": "COMPLIANT",
                "backup_security": "ATTENTION_NEEDED",
                "cloud_security": "REVIEW_RECOMMENDED",
            },
            "recommendations": [
                {
                    "category": "permissions",
                    "action": "Review location access necessity",
                    "priority": "HIGH",
                },
                {
                    "category": "data_storage",
                    "action": "Enable encrypted backups",
                    "priority": "MEDIUM",
                },
            ],
        }
        mock_assess.return_value = expected_assessment

        result = assessor.assess_mobile_app_security(app_data=mobile_app_data)

        assert isinstance(result, dict)
        assert "app_security_score" in result
        assert "permission_analysis" in result
        assert "recommendations" in result
        assert result["app_security_score"] >= 0 and result["app_security_score"] <= 100

    def test_generate_security_recommendations(self, assessor):
        """Test security recommendation generation."""
        network_type = "enterprise"
        device_types = ["byod", "iot"]
        compliance_requirements = ["pci-dss", "hipaa"]

        result = assessor.generate_security_recommendations(
            network_type=network_type,
            device_types=device_types,
            compliance_requirements=compliance_requirements,
        )

        assert isinstance(result, dict)
        assert "network_recommendations" in result
        assert "device_recommendations" in result
        assert "compliance_gaps" in result

    def test_detect_rogue_access_points(self, assessor, wifi_scan_data):
        """Test rogue access point detection."""
        trusted_networks = ["Corp-Network"]

        result = assessor.detect_rogue_access_points(
            scan_data=wifi_scan_data, trusted_networks=trusted_networks
        )

        assert isinstance(result, dict)
        assert "rogue_aps" in result
        assert "suspicious_aps" in result
        assert "timestamp" in result

    def test_analyze_network_traffic(self, assessor):
        """Test network traffic analysis."""
        traffic_data = {
            "source": "00:11:22:33:44:55",
            "destination": "cloud.service.com",
            "protocol": "https",
            "bytes_sent": 1024,
            "timestamp": datetime.now().isoformat(),
        }

        result = assessor.analyze_network_traffic(traffic_data)

        assert isinstance(result, dict)
        assert "traffic_analysis" in result
        assert "security_concerns" in result
        assert "recommendations" in result

    def test_validate_security_controls(self, assessor):
        """Test security control validation."""
        controls = {
            "wpa3_enabled": True,
            "mac_filtering": True,
            "guest_network_isolation": True,
            "ids_enabled": False,
        }

        result = assessor.validate_security_controls(controls)

        assert isinstance(result, dict)
        assert "passed_controls" in result
        assert "failed_controls" in result
        assert "missing_controls" in result
        assert len(result["passed_controls"]) + len(result["failed_controls"]) == len(
            controls
        )
