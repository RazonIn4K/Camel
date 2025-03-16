# CyberSecurityService

## Overview

The `CyberSecurityService` acts as a high-level coordinator for all specialized security agents. It provides a unified interface for managing agent interactions and orchestrating complex security workflows.

## Command Processing Architecture

### process_command Internal Logic

The `process_command` method uses a combination of natural language understanding and predefined command mappings:

```python
result = service.process_command(
    command="Analyze network security and generate forensics plan",
    context={
        "network_scan": "scans/network.xml",
        "priority": "high"
    },
    command_mapping={
        "network_analysis": {
            "agent": "network",
            "method": "analyze_vulnerabilities",
            "required_context": ["network_scan"]
        },
        "forensics_plan": {
            "agent": "forensics",
            "method": "generate_investigation_plan",
            "required_context": ["priority"]
        }
    }
)
```

Command resolution follows this sequence:
1. Natural language parsing to identify intent
2. Matching against predefined command mappings
3. Validation of required context
4. Agent method dispatch

### coordinate_agents Dependencies

The `dependencies` parameter controls execution flow between agents. Example:

```python
workflow = [
    {
        "id": "network_scan",
        "agent": "network",
        "action": "analyze_traffic",
        "params": {"pcap_file": "capture.pcap"}
    },
    {
        "id": "forensics_plan",
        "agent": "forensics",
        "action": "generate_plan",
        "depends_on": ["network_scan"],
        "params": {
            "findings": "${network_scan.vulnerabilities}"
        }
    }
]

dependencies = {
    "forensics_plan": {
        "requires": ["network_scan"],
        "condition": "network_scan.status == 'completed'",
        "data_mapping": {
            "findings": "network_scan.vulnerabilities",
            "priority": "network_scan.threat_level"
        }
    }
}

results = service.coordinate_agents(
    workflow=workflow,
    dependencies=dependencies
)
```

## Model Support

The service supports multiple AI model backends:

- OpenAI GPT-4
- Anthropic Claude (Sonnet 3.5 and 3.7)
- Camel AI's hosted service

Configure the model in `config/agent_config.yaml`:

```yaml
model:
  provider: "anthropic"  # or "openai" or "camel"
  version: "claude-3-sonnet-3.7"  # or "claude-3-sonnet-3.5" or "gpt-4"
  temperature: 0.7
  max_tokens: 4096
```

## Initialization

```python
from cybersec_agents import CyberSecurityService

# Initialize with default configuration
service = CyberSecurityService()

# Initialize with custom configuration
service = CyberSecurityService(
    config_path="custom_config.yaml",
    model_provider="anthropic",
    model_version="claude-3-sonnet-3.7"
)
```

## Core Methods

### process_command

Processes high-level security commands by coordinating appropriate agents.

**Parameters:**
- `command` (str): Security command to process
- `context` (Dict, optional): Additional context for command processing
- `output_format` (str, optional): Desired output format ("json", "markdown", "text")

**Returns:**
- Dict containing processed results and agent outputs

**Example:**
```python
result = service.process_command(
    command="Analyze network security and generate forensics plan",
    context={
        "network_scan": "scans/network.xml",
        "priority": "high"
    }
)
```

### coordinate_agents

Coordinates multiple agents for complex security tasks.

**Parameters:**
- `workflow` (List[Dict]): Sequence of agent tasks
- `shared_context` (Dict): Context shared across agents
- `dependencies` (Dict, optional): Task dependencies

**Returns:**
- Dict containing combined agent results

**Example:**
```python
results = service.coordinate_agents(
    workflow=[
        {
            "agent": "network",
            "action": "analyze_vulnerabilities",
            "params": {"scan_file": "scan.xml"}
        },
        {
            "agent": "forensics",
            "action": "create_investigation_plan",
            "params": {"case_type": "vulnerability"}
        }
    ],
    shared_context={"priority": "high"}
)
```

## Agent Management

### get_agent

Retrieves an initialized agent instance.

**Parameters:**
- `agent_type` (str): Type of agent to retrieve
- `config` (Dict, optional): Agent-specific configuration

**Returns:**
- Instance of requested agent

**Example:**
```python
network_agent = service.get_agent("network")
wireless_agent = service.get_agent("wireless")
```

### register_custom_agent

Registers a custom agent with the service.

**Parameters:**
- `agent_name` (str): Name for the custom agent
- `agent_class` (Type): Custom agent class
- `config` (Dict, optional): Agent configuration

**Example:**
```python
service.register_custom_agent(
    agent_name="custom_scanner",
    agent_class=CustomSecurityScanner,
    config={"scan_depth": "deep"}
)
```

## Integration Examples

### Comprehensive Security Analysis

```python
# Initialize service with Claude Sonnet 3.7
service = CyberSecurityService(model_version="claude-3-sonnet-3.7")

# Perform comprehensive analysis
analysis = service.process_command(
    "Perform comprehensive security analysis",
    context={
        "network_scan": "scans/network.xml",
        "wireless_scan": "scans/wireless.csv",
        "mobile_apps": ["app1", "app2"],
        "output_format": "markdown"
    }
)

# Generate reports
service.generate_reports(analysis, output_dir="reports/")
```

### Custom Workflow

```python
workflow = [
    {
        "agent": "network",
        "action": "analyze_traffic",
        "params": {"pcap_file": "capture.pcap"}
    },
    {
        "agent": "wireless",
        "action": "assess_security",
        "params": {"network_type": "enterprise"}
    },
    {
        "agent": "forensics",
        "action": "generate_plan",
        "params": {"findings": "PREVIOUS_STEP"}
    }
]

results = service.coordinate_agents(workflow=workflow)
```

## Configuration

### Environment Variables

Required environment variables:
```bash
export OPENAI_API_KEY=your_key  # if using OpenAI
export ANTHROPIC_API_KEY=your_key  # if using Claude
export CAMEL_AI_API_KEY=your_key  # if using Camel AI
```

### Configuration File

Example `config/agent_config.yaml`:
```yaml
service:
  model:
    provider: "anthropic"
    version: "claude-3-sonnet-3.7"
    temperature: 0.7
  
  agents:
    network:
      enabled: true
      scan_interval: 300
    
    wireless:
      enabled: true
      monitoring: true
    
    forensics:
      enabled: true
      retention_days: 30

  reporting:
    format: "markdown"
    output_dir: "reports/"
    include_metrics: true
```

## Error Handling

Errors are handled through a hierarchical system:

1. **Exception Types:**
```python
from cybersec_agents.exceptions import (
    AgentInitializationError,
    ModelConfigurationError,
    WorkflowExecutionError,
    CoordinationError
)
```

2. **Error Response Format:**
```python
{
    "status": "error",
    "error_type": "WorkflowExecutionError",
    "message": "Failed to execute network analysis",
    "details": {
        "agent": "network",
        "method": "analyze_traffic",
        "error_code": "PCAP_PARSE_ERROR",
        "original_error": "..."
    },
    "recommendations": [
        "Verify PCAP file format",
        "Check file permissions"
    ]
}
```

## Model Configuration

### Claude Model Support

Specific model version strings and configurations:

```yaml
model:
  provider: "anthropic"
  version: "claude-3-sonnet-3.7"  # Latest Sonnet
  # or "claude-3-sonnet-3.5"      # Previous Sonnet
  context_window: 200000          # Maximum context window
  temperature: 0.7
  max_tokens: 4096
  response_format: {
    "type": "json",              # or "text" for unstructured output
    "schema": {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "recommendations": {"type": "array"},
            "confidence": {"type": "number"}
        }
    }
  }
```

### Model-Specific Features

Claude models provide additional capabilities:

1. **Enhanced Context Understanding:**
```python
service.process_command(
    command="Analyze network security",
    context={
        "network_scan": "scans/network.xml",
        "previous_incidents": ["incident1.json", "incident2.json"],
        "compliance_requirements": ["PCI-DSS", "HIPAA"]
    },
    model_features={
        "context_awareness": True,
        "compliance_check": True
    }
)
```

2. **Multi-Modal Analysis:**
```python
service.analyze_security_artifacts({
    "network_diagram": "diagram.png",
    "log_files": ["system.log", "access.log"],
    "configuration": "config.yaml"
})
```

## Performance Optimization

### Caching Strategy

```python
service = CyberSecurityService(
    cache_config={
        "type": "redis",
        "ttl": 3600,
        "invalidation_rules": {
            "network_scan": "1h",
            "vulnerability_report": "24h"
        }
    }
)
```

### Batch Processing

```python
results = service.batch_process([
    {
        "command": "analyze_network",
        "context": {"scan": "scan1.xml"}
    },
    {
        "command": "analyze_network",
        "context": {"scan": "scan2.xml"}
    }
], batch_size=5)
```

## Best Practices

1. Service Configuration:
   - Use environment-specific configuration files
   - Regularly update API keys
   - Monitor model usage and costs

2. Workflow Design:
   - Define clear agent dependencies
   - Include error handling in workflows
   - Document custom workflows

3. Performance Optimization:
   - Cache common results
   - Use appropriate model configurations
   - Implement request batching

4. Security:
   - Secure API key storage
   - Implement access controls
   - Monitor service usage