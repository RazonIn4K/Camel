# ForensicsPlanner

## Overview

The ForensicsPlanner agent assists in planning and executing digital forensics investigations by generating procedural templates and evidence collection guidelines.

## Methods

### generate_investigation_plan

Generates a comprehensive forensic investigation plan based on case parameters.

**Parameters:**
- `case_type` (str): Type of forensic investigation
  - Supported values: ["malware", "data_breach", "ransomware", "insider_threat", "email_compromise", "device_theft"]
- `target_systems` (List[str]): Systems to investigate
  - Supported values: ["windows", "linux", "macos", "ios", "android", "network_devices", "cloud"]
- `timeline_constraints` (str, optional): Time constraints for investigation
  - Format: "XhYmZs" (hours, minutes, seconds) or "Xd" (days)
- `priority` (str, optional): Investigation priority
  - Supported values: ["critical", "high", "medium", "low"]

**Returns:**
```python
{
    "plan": {
        "overview": {
            "case_type": str,
            "priority": str,
            "estimated_duration": str,
            "resource_requirements": List[str]
        },
        "phases": [
            {
                "name": str,
                "description": str,
                "tasks": [
                    {
                        "id": str,
                        "description": str,
                        "tools": List[str],
                        "prerequisites": List[str],
                        "estimated_time": str,
                        "technical_notes": str
                    }
                ],
                "deliverables": List[str]
            }
        ],
        "evidence_handling": {
            "collection_points": List[str],
            "chain_of_custody": Dict,
            "storage_requirements": Dict
        }
    },
    "metadata": {
        "generated_at": str,        # ISO timestamp
        "plan_version": str,
        "compliance_frameworks": List[str]
    }
}
```

### generate_evidence_collection_procedure

Creates detailed procedures for collecting digital evidence.

**Parameters:**
- `device_type` (str): Type of device to collect evidence from
  - Supported values: ["windows_workstation", "windows_server", "linux_server", "mobile_ios", "mobile_android", "network_storage", "cloud_instance"]
- `artifact_types` (List[str]): Types of artifacts to collect
  - Supported values: ["filesystem", "memory", "logs", "registry", "network", "database", "email", "chat_logs", "browser_history"]
- `legal_jurisdiction` (str, optional): Legal framework to follow
  - Supported values: ["US", "EU", "UK", "AU", "CA"] or ISO country codes
- `preservation_priority` (List[str], optional): Priority order for evidence collection
  - Supported values: ["volatile_first", "user_data_first", "system_first"]

**Returns:**
```python
{
    "procedure": {
        "preparation": {
            "required_tools": List[str],
            "prerequisites": List[str],
            "safety_checks": List[str]
        },
        "steps": [
            {
                "order": int,
                "action": str,
                "tool": str,
                "command": str,
                "expected_output": str,
                "verification": str,
                "contingency": str
            }
        ],
        "documentation": {
            "templates": List[str],
            "required_photos": List[str],
            "chain_of_custody": Dict
        }
    },
    "legal_compliance": {
        "jurisdiction": str,
        "requirements": List[str],
        "documentation_needs": List[str]
    },
    "validation_checklist": List[Dict]
}
```

### generate_timeline_template

Creates a forensic timeline template for incident investigation.

**Parameters:**
- `incident_type` (str): Type of security incident
  - Supported values: ["data_breach", "malware_infection", "unauthorized_access", "system_compromise", "insider_threat"]
- `time_range` (Dict): Time range for investigation
  ```python
  {
      "start": str,    # ISO timestamp
      "end": str,      # ISO timestamp
      "timezone": str  # IANA timezone
  }
  ```
- `data_sources` (List[str], optional): Sources to include
  - Supported values: ["system_logs", "network_logs", "application_logs", "security_logs", "user_activity"]
- `correlation_level` (str, optional): Detail level for event correlation
  - Supported values: ["basic", "detailed", "comprehensive"]

**Returns:**
```python
{
    "timeline": {
        "metadata": {
            "incident_type": str,
            "time_range": Dict,
            "data_sources": List[str]
        },
        "events": [
            {
                "timestamp": str,   # ISO timestamp
                "source": str,
                "category": str,
                "description": str,
                "significance": str,  # "high", "medium", "low"
                "artifacts": List[str],
                "correlations": List[Dict]
            }
        ],
        "analysis_points": [
            {
                "timestamp": str,
                "observation": str,
                "implications": List[str],
                "required_investigation": List[str]
            }
        ]
    },
    "visualization": {
        "suggested_tools": List[str],
        "key_events": List[Dict],
        "patterns": List[Dict]
    }
}
```

## Error Handling

The agent raises specific exceptions:

```python
from cybersec_agents.exceptions import (
    InvalidCaseTypeError,        # Unsupported case type
    DeviceTypeError,            # Unsupported device type
    JurisdictionError,          # Invalid legal jurisdiction
    TimelineGenerationError,    # Timeline creation failed
    ComplianceError            # Legal compliance issues
)
```

## Usage Examples

### Basic Investigation Plan
```python
planner = ForensicsPlanner()

# Generate investigation plan
plan = planner.generate_investigation_plan(
    case_type="data_breach",
    target_systems=["windows", "linux"],
    timeline_constraints="48h",
    priority="high"
)

# Generate evidence collection procedure
procedure = planner.generate_evidence_collection_procedure(
    device_type="windows_workstation",
    artifact_types=["memory", "filesystem"],
    legal_jurisdiction="US"
)

# Create timeline template
timeline = planner.generate_timeline_template(
    incident_type="unauthorized_access",
    time_range={
        "start": "2024-03-01T00:00:00Z",
        "end": "2024-03-02T00:00:00Z",
        "timezone": "UTC"
    },
    data_sources=["system_logs", "security_logs"]
)
```
