# NetworkAnomalyDetector

## Overview

The NetworkAnomalyDetector agent specializes in analyzing network traffic and identifying potential security threats from Nmap and Wireshark outputs.

## Methods

### analyze_nmap_output

Analyzes Nmap XML output for security vulnerabilities and exposed services.

**Parameters:**
- `nmap_file` (str): Path to Nmap XML output file
- `analysis_options` (Dict, optional): Configuration for analysis
  ```python
  {
      "severity_threshold": float,  # 0.0 to 1.0, default: 0.5
      "port_categories": List[str], # ["dangerous", "suspicious", "standard"]
      "service_focus": List[str]    # ["web", "database", "remote_access"]
  }
  ```

**Returns:**
```python
{
    "vulnerabilities": [
        {
            "host": str,            # IP or hostname
            "port": int,            # Port number
            "service": str,         # Service name
            "severity": float,      # 0.0 to 1.0
            "description": str,     # Detailed description
            "recommendations": List[str]
        }
    ],
    "exposed_services": [
        {
            "host": str,
            "port": int,
            "service": str,
            "version": str,
            "risk_level": str      # "high", "medium", "low"
        }
    ],
    "summary": {
        "total_hosts": int,
        "vulnerable_hosts": int,
        "risk_score": float,       # 0.0 to 1.0
        "critical_findings": int
    }
}
```

### analyze_network_traffic

Analyzes Wireshark PCAP files for anomalous traffic patterns.

**Parameters:**
- `pcap_file` (str): Path to PCAP file
- `protocols` (List[str], optional): Protocols to analyze
  - Supported values: ["tcp", "udp", "http", "https", "dns", "smb"]
- `threshold` (float, optional): Anomaly detection threshold (0.0 to 1.0)
- `time_window` (str, optional): Analysis window (e.g., "1h", "30m", "1d")

**Returns:**
```python
{
    "anomalies": [
        {
            "timestamp": str,      # ISO format
            "protocol": str,
            "source": str,         # IP address
            "destination": str,    # IP address
            "confidence": float,   # 0.0 to 1.0
            "type": str,          # "data_exfiltration", "port_scan", etc.
            "details": Dict        # Protocol-specific details
        }
    ],
    "traffic_patterns": {
        "total_packets": int,
        "protocols": Dict[str, int],  # Protocol counts
        "top_talkers": List[Dict],    # Most active IPs
        "time_distribution": Dict[str, int]  # Traffic over time
    },
    "recommendations": List[str]
}
```

### generate_security_report

Generates a comprehensive security report combining Nmap and Wireshark analysis.

**Parameters:**
- `nmap_file` (str, optional): Path to Nmap XML output
- `pcap_file` (str, optional): Path to PCAP file
- `format` (str, optional): Output format
  - Supported values: ["json", "markdown", "html", "pdf"]
- `include_sections` (List[str], optional): Sections to include
  - Supported values: ["executive", "technical", "remediation", "timeline"]

**Returns:**
```python
{
    "report": {
        "summary": {
            "risk_level": str,     # "critical", "high", "medium", "low"
            "key_findings": List[str],
            "urgent_actions": List[str]
        },
        "technical_details": {
            "network_vulnerabilities": List[Dict],  # From analyze_nmap_output
            "traffic_analysis": Dict,               # From analyze_network_traffic
            "attack_vectors": List[Dict]
        },
        "remediation": {
            "immediate_actions": List[Dict],
            "long_term_recommendations": List[Dict]
        },
        "metadata": {
            "generated_at": str,    # ISO timestamp
            "analysis_duration": float,
            "tool_versions": Dict[str, str]
        }
    },
    "format": str,                 # Requested format
    "file_path": str              # Path to generated report file
}
```

## Error Handling

The agent raises specific exceptions:

```python
from cybersec_agents.exceptions import (
    NmapParseError,           # Invalid Nmap XML
    PcapReadError,            # PCAP file issues
    AnalysisConfigError,      # Invalid configuration
    ReportGenerationError     # Report creation failed
)
```

## Usage Examples

### Basic Network Analysis
```python
detector = NetworkAnomalyDetector()

# Analyze Nmap scan
nmap_results = detector.analyze_nmap_output(
    "scan.xml",
    analysis_options={
        "severity_threshold": 0.7,
        "port_categories": ["dangerous"]
    }
)

# Analyze traffic
traffic_results = detector.analyze_network_traffic(
    "capture.pcap",
    protocols=["tcp", "http"],
    threshold=0.8,
    time_window="1h"
)
```

### Comprehensive Report Generation
```python
report = detector.generate_security_report(
    nmap_file="scan.xml",
    pcap_file="capture.pcap",
    format="markdown",
    include_sections=["executive", "technical"]
)
```


