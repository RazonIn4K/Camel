from datetime import datetime

import pytest


@pytest.fixture
def sample_nmap_xml():
    """Provides a minimal valid Nmap XML output for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE nmaprun>
<nmaprun scanner="nmap">
    <host>
        <address addr="192.168.1.1" addrtype="ipv4"/>
        <ports>
            <port protocol="tcp" portid="80">
                <state state="open"/>
                <service name="http"/>
            </port>
        </ports>
    </host>
</nmaprun>"""


@pytest.fixture
def sample_pcap_data():
    """Provides sample PCAP data structure for testing."""
    return {
        "packets": [
            {
                "source": "192.168.1.100",
                "destination": "192.168.1.1",
                "protocol": "TCP",
                "port": 80,
                "length": 64,
            }
        ],
        "summary": {"total_packets": 1, "protocols": ["TCP"], "duration": 1.0},
    }


@pytest.fixture
def config():
    """Provides sample configuration for agents."""
    return {
        "api_key": "test_key",
        "model": "gpt-4-turbo",
        "thresholds": {"anomaly_detection": 0.75, "vulnerability_severity": "medium"},
    }


@pytest.fixture
def sample_case_data():
    """Provides sample forensics case data for testing."""
    return {
        "case_id": "FOR-2024-001",
        "case_type": "ransomware",
        "target_systems": ["windows_server_2019", "windows_10_workstation"],
        "timestamp": datetime.now().isoformat(),
        "priority": "high",
        "artifacts": ["memory_dump", "event_logs", "registry_hives"],
    }


@pytest.fixture
def sample_evidence_data():
    """Provides sample digital evidence collection data."""
    return {
        "device_type": "windows_server",
        "evidence_items": [
            {
                "type": "memory_dump",
                "path": "/evidence/ram.dump",
                "hash": "sha256:abc123...",
                "size": "16GB",
            },
            {
                "type": "event_logs",
                "path": "/evidence/windows/system32/winevt/logs/",
                "hash": "sha256:def456...",
                "size": "500MB",
            },
        ],
        "chain_of_custody": [
            {
                "timestamp": datetime.now().isoformat(),
                "action": "acquisition",
                "operator": "John Doe",
                "location": "Server Room A",
            }
        ],
    }


@pytest.fixture
def sample_timeline_data():
    """Provides sample incident timeline data."""
    return {
        "incident_type": "data_breach",
        "events": [
            {
                "timestamp": "2024-02-01T10:00:00Z",
                "category": "initial_access",
                "description": "First suspicious login attempt detected",
            },
            {
                "timestamp": "2024-02-01T10:15:00Z",
                "category": "lateral_movement",
                "description": "Unauthorized access to internal network share",
            },
        ],
        "time_range": {"start": "2024-02-01T00:00:00Z", "end": "2024-02-02T00:00:00Z"},
    }


@pytest.fixture
def network_traffic_data():
    """Sample network traffic data for testing."""
    return {
        "sessions": [
            {
                "source_mac": "00:11:22:33:44:55",
                "destination_mac": "aa:bb:cc:dd:ee:ff",
                "protocol": "https",
                "port": 443,
                "bytes_transferred": 1500,
                "timestamp": datetime.now().isoformat(),
                "duration": 120,  # seconds
            }
        ],
        "capture_info": {
            "start_time": datetime.now().isoformat(),
            "duration": 300,  # seconds
            "interface": "wlan0",
        },
    }


@pytest.fixture
def security_control_data():
    """Sample security control configuration data."""
    return {
        "wireless_controls": {
            "wpa3_enabled": True,
            "mac_filtering": True,
            "guest_isolation": True,
            "ids_enabled": False,
        },
        "mobile_controls": {
            "mdm_enabled": True,
            "app_verification": True,
            "remote_wipe": True,
        },
        "compliance": {"pci_dss": True, "hipaa": True},
    }
