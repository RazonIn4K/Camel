# Multi-Agent Interaction Example

```python
from cybersec_agents import (
    NetworkAnomalyDetector,
    ForensicsPlanner,
    WirelessMobileSecurityAssessor
)

# Initialize agents
network_agent = NetworkAnomalyDetector()
forensics_agent = ForensicsPlanner()
wireless_agent = WirelessMobileSecurityAssessor()

# Network analysis informs forensics planning
network_scan = network_agent.analyze_nmap_output("network_scan.xml")
vulnerable_systems = network_agent.extract_vulnerable_systems(network_scan)

# Create targeted forensics plans
for system in vulnerable_systems:
    plan = forensics_agent.generate_investigation_plan(
        case_type="vulnerability_exploitation",
        target_systems=[system['hostname']]
    )
    
# Assess wireless security based on findings
if any(system['service'] == 'wifi' for system in vulnerable_systems):
    wireless_recommendations = wireless_agent.generate_security_recommendations(
        network_type="enterprise",
        vulnerabilities_found=True
    )
```