# API Reference

## Network Anomaly Detector

### NetworkAnomalyDetector

```python
from cybersec_agents import NetworkAnomalyDetector

detector = NetworkAnomalyDetector()
```

#### Methods

##### analyze_nmap_output(nmap_file: str) -> dict
Analyzes Nmap scan results for security insights.

Parameters:
- `nmap_file`: Path to the Nmap output file

Returns:
- Dictionary containing:
  - `threats`: List of identified threats
  - `recommendations`: Security recommendations
  - `risk_level`: Overall risk assessment

Example:
```python
results = detector.analyze_nmap_output("scan_results.txt")
print(results["threats"])
```

##### analyze_pcap(pcap_file: str) -> dict
Analyzes Wireshark PCAP file for network anomalies.

Parameters:
- `pcap_file`: Path to the PCAP file

Returns:
- Dictionary containing:
  - `anomalies`: List of detected anomalies
  - `traffic_patterns`: Unusual traffic patterns
  - `security_flags`: Security concerns

## GUI Components

### CyberSecurityGUI

```python
from cybersec_agents.gui import CyberSecurityGUI

gui = CyberSecurityGUI()
```

#### Methods

##### register_agent(name: str, agent: ChatAgent) -> None
Registers a new agent for use in the GUI.

Parameters:
- `name`: Unique identifier for the agent
- `agent`: Instance of ChatAgent

Example:
```python
from camel.agents import ChatAgent
gui.register_agent("network_analyzer", network_agent)
```

##### run() -> None
Starts the GUI application.
```

</augment_code_snippet>

<augment_code_snippet path="docs/cli_guide.md" mode="EDIT">
```markdown
# CLI Usage Guide

## Basic Commands

### Network Analysis

```bash
# Analyze network scan
cyber-agents run "analyze network scan_results.txt"

# Generate security report
cyber-agents run "analyze security --output report.json"
```

### Output Formats

The CLI supports multiple output formats:
- text (default)
- json
- yaml

Example:
```bash
cyber-agents run "analyze network" --format json --output results.json
```

## Command Reference

### run
Execute a cybersecurity analysis command.

Options:
- `--output, -o`: Output file path
- `--format, -f`: Output format (text|json|yaml)

Examples:
```bash
# Basic analysis
cyber-agents run "analyze network"

# Save results to file
cyber-agents run "analyze security" -o report.txt

# JSON output
cyber-agents run "plan forensics" -f json
```
```

</augment_code_snippet>

<augment_code_snippet path="docs/examples.md" mode="EDIT">
```markdown
# Usage Examples

## Network Security Analysis

### Basic Network Scan Analysis
```python
from cybersec_agents import NetworkAnomalyDetector

# Initialize detector
detector = NetworkAnomalyDetector()

# Analyze Nmap results
results = detector.analyze_nmap_output("network_scan.txt")

# Process results
for threat in results["threats"]:
    print(f"Detected threat: {threat['type']}")
    print(f"Severity: {threat['severity']}")
    print(f"Recommendation: {threat['recommendation']}")
```

### GUI-Based Analysis

```python
from cybersec_agents.gui import CyberSecurityGUI
from cybersec_agents import NetworkAnomalyDetector

# Initialize GUI
gui = CyberSecurityGUI()

# Create and register agent
detector = NetworkAnomalyDetector()
gui.register_agent("network", detector)

# Start GUI
gui.run()
```

### CLI Usage Examples

```bash
# Basic network analysis
cyber-agents run "analyze network scan.txt"

# Generate JSON report
cyber-agents run "analyze security" -f json -o security_report.json

# Interactive forensics planning
cyber-agents run "plan forensics" --interactive
```

## Best Practices

1. Always validate input files before analysis
2. Use appropriate output formats for automation
3. Implement error handling for network operations
4. Regular updates of security definitions
```

</augment_code_snippet>

<augment_code_snippet path="docs/troubleshooting.md" mode="EDIT">
```markdown
# Troubleshooting Guide

## Common Issues

### Network Analysis Failures

#### Issue: Invalid Nmap Output Format
```python
# Incorrect
results = detector.analyze_nmap_output("raw_output.txt")

# Correct
# Ensure Nmap output is in normal format (-oN)
nmap -oN scan_results.txt target_network
results = detector.analyze_nmap_output("scan_results.txt")
```

#### Issue: PCAP Analysis Errors
- Verify PCAP file format
- Check file permissions
- Ensure complete capture files

### GUI Issues

#### Issue: Agent Registration Fails
```python
# Common mistake
gui.register_agent(network_agent)  # Missing name parameter

# Correct usage
gui.register_agent("network", network_agent)
```

#### Issue: GUI Not Responding
- Check agent initialization
- Verify resource usage
- Review error logs

## Error Messages

### "Invalid Agent Configuration"
- Verify agent initialization parameters
- Check model configuration
- Ensure proper system message

### "Analysis Failed"
- Validate input file format
- Check file permissions
- Review agent logs
```

</augment_code_snippet>

