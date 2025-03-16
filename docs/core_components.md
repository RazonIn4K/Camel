# Core Components

## Agent Types
1. NetworkAnomalyDetector
- Purpose: Network traffic analysis and threat detection
- Key methods:
```python
detector = NetworkAnomalyDetector()
analysis = detector.analyze_nmap_output("scan.xml")
report = detector.generate_security_report()
```

2. ForensicsPlanner
- Purpose: Digital forensics investigation planning
- Key methods:
```python
planner = ForensicsPlanner()
plan = planner.generate_investigation_plan(
    case_type="malware_infection"
)
```

3. WirelessMobileSecurityAssessor
- Purpose: Wireless network and mobile security assessment
- Key methods:
```python
assessor = WirelessMobileSecurityAssessor()
recommendations = assessor.generate_security_recommendations()
```

## Service Integration
The CyberSecurityService wrapper coordinates agent interactions:
```python
from cybersec_agents import CyberSecurityService

service = CyberSecurityService()
combined_analysis = service.perform_comprehensive_analysis()
```