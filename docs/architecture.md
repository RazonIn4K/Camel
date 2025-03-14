# Architecture Overview

## Core Components

### 1. Service Wrapper (`service_wrapper.py`)
- High-level interface for coordinating agents
- Manages agent lifecycle and interactions
- Handles configuration and resource management

### 2. Agents
- Network Security Agent: Traffic analysis and threat detection
- Forensics Planner: Investigation planning and evidence collection
- Wireless Security Assessor: Network and mobile security analysis
- Code Assistant: Code review and improvement suggestions

### 3. Integration Layer
- Model management
- API integrations
- Resource coordination

## System Design

[Include system architecture diagram]

## Implementation Details

### Agent Communication
```python
async def comprehensive_security_audit(target_network):
    """Example of multi-agent coordination."""
    service = CyberSecurityService()
    
    # Network scanning
    scan_results = await service.agents["network"].execute_async(
        "perform_network_scan",
        {"target": target_network}
    )
    
    # Further processing...
```

For implementation examples, see the [User Guide](user_guide.md).
