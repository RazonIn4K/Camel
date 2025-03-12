# Cybersecurity AI Agents

This repository contains specialized AI agents built on the Camel AI framework for cybersecurity tasks. The agents are designed to assist security professionals in network analysis, forensics planning, and wireless/mobile security assessment.

## Table of Contents

- [Setup](#setup)
  - [Environment Variables](#environment-variables)
  - [Installation](#installation)
- [Specialized Agents](#specialized-agents)
  - [NetworkAnomalyDetector](#networkanomalydetector)
  - [ForensicsPlanner](#forensicsplanner)
  - [WirelessMobileSecurityAssessor](#wirelessmobilesecurityassessor)
- [Using the Updated Cyber Writer](#using-the-updated-cyber-writer)
- [Advanced Usage](#advanced-usage)

## Setup

### Environment Variables

The following environment variables are required for the agents to function properly:

```
OPENAI_API_KEY=your_openai_api_key
CAMEL_AI_API_KEY=your_camel_ai_api_key
LOG_LEVEL=INFO
ENABLE_MONETIZATION=false
MAX_TOKENS=4096
TEMPERATURE=0.7
```

You can set these in your environment or use a `.env` file in the project root.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cybersecurity-agents.git
   cd cybersecurity-agents
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - camelai>=0.2.0
   - openai>=1.0.0
   - python-dotenv>=0.19.0
   - argparse>=1.4.0
   - pandas>=1.3.0
   - pyyaml>=6.0

3. Configure your environment:
   ```bash
   cp config/config.yaml.example config/config.yaml
   # Edit config.yaml with your API keys and preferences
   ```

## Specialized Agents

### NetworkAnomalyDetector

This agent specializes in analyzing network traffic and identifying potential security threats from Nmap and Wireshark outputs.

#### Command Line Usage

```bash
# Analyze Nmap scan results
python cybersecurity_agents.py network --nmap-file path/to/nmap_output.xml

# Analyze Wireshark capture
python cybersecurity_agents.py network --pcap-file path/to/wireshark_capture.pcap

# Perform a comprehensive analysis with both
python cybersecurity_agents.py network --nmap-file path/to/nmap_output.xml --pcap-file path/to/wireshark_capture.pcap --output-format json
```

#### Programmatic Usage

```python
from cybersecurity_agents import NetworkAnomalyDetector

# Initialize the agent
detector = NetworkAnomalyDetector(
    api_key="your_camel_ai_api_key",
    model="gpt-4-turbo"
)

# Analyze Nmap results
nmap_analysis = detector.analyze_nmap_output("path/to/nmap_output.xml")
print(nmap_analysis)

# Analyze network traffic
traffic_analysis = detector.analyze_network_traffic("path/to/wireshark_capture.pcap")
print(traffic_analysis)

# Generate a comprehensive security report
report = detector.generate_security_report(
    nmap_file="path/to/nmap_output.xml",
    pcap_file="path/to/wireshark_capture.pcap"
)
print(report)
```

### ForensicsPlanner

This agent assists in planning and executing digital forensics investigations by generating procedural templates and evidence collection guidelines.

#### Command Line Usage

```bash
# Generate a forensic investigation plan
python cybersecurity_agents.py forensics --generate-plan --case-type "ransomware" --output-file "ransomware_plan.md"

# Create an evidence collection procedure
python cybersecurity_agents.py forensics --evidence-collection --device-type "windows_server" --output-file "windows_server_evidence.md"

# Generate a forensic timeline template
python cybersecurity_agents.py forensics --timeline-template --incident-type "data_breach" --output-file "data_breach_timeline.md"
```

#### Programmatic Usage

```python
from cybersecurity_agents import ForensicsPlanner

# Initialize the agent
planner = ForensicsPlanner(
    api_key="your_camel_ai_api_key",
    model="gpt-4-turbo"
)

# Generate a forensic investigation plan
plan = planner.generate_investigation_plan(
    case_type="malware_infection",
    target_systems=["windows", "linux"],
    timeline_constraints="72 hours"
)
print(plan)

# Create an evidence collection procedure
procedure = planner.generate_evidence_collection_procedure(
    device_type="mobile_ios",
    artifact_types=["sms", "call_logs", "app_data"],
    legal_jurisdiction="US"
)
print(procedure)

# Generate a forensic timeline template
timeline = planner.create_forensic_timeline_template(
    incident_type="unauthorized_access",
    time_frame="30 days",
    critical_systems=["authentication_server", "database"]
)
print(timeline)
```

### WirelessMobileSecurityAssessor

This agent specializes in assessing the security of wireless networks and mobile devices, identifying vulnerabilities and suggesting mitigations.

#### Command Line Usage

```bash
# Analyze wireless network security
python cybersecurity_agents.py wireless --network-scan path/to/wifi_scan.csv --output-file "wifi_security_assessment.md"

# Assess mobile application security
python cybersecurity_agents.py wireless --app-name "example_app" --platform android --output-file "app_security_report.md"

# Generate security best practices
python cybersecurity_agents.py wireless --generate-best-practices --network-type "enterprise" --output-file "enterprise_wifi_best_practices.md"
```

#### Programmatic Usage

```python
from cybersecurity_agents import WirelessMobileSecurityAssessor

# Initialize the agent
assessor = WirelessMobileSecurityAssessor(
    api_key="your_camel_ai_api_key",
    model="gpt-4-turbo"
)

# Analyze wireless network security
wifi_assessment = assessor.analyze_wireless_network(
    scan_file="path/to/wifi_scan.csv",
    network_type="corporate"
)
print(wifi_assessment)

# Assess mobile application security
app_assessment = assessor.assess_mobile_app_security(
    app_name="example_app",
    platform="ios",
    permissions=["camera", "location", "contacts"]
)
print(app_assessment)

# Generate security recommendations
recommendations = assessor.generate_security_recommendations(
    network_type="guest",
    device_types=["byod", "iot"],
    compliance_requirements=["pci-dss", "hipaa"]
)
print(recommendations)
```

## Using the Updated Cyber Writer

The `cyber_writer.py` script has been updated to incorporate the specialized agents while maintaining backward compatibility.

### Command Line Usage

```bash
# Original functionality (general cybersecurity writing)
python cyber_writer.py --topic "zero trust architecture" --output "zero_trust_paper.md"

# Network analysis mode
python cyber_writer.py --mode network --nmap-file scan.xml --pcap-file capture.pcap

# Forensics planning mode
python cyber_writer.py --mode forensics --case-type "ransomware" --output "ransomware_plan.md"

# Wireless/mobile security mode
python cyber_writer.py --mode wireless --network-scan wifi_scan.csv --output "wifi_assessment.md"
```

### Programmatic Usage

```python
from cyber_writer import CyberWriter

# Initialize the writer
writer = CyberWriter(
    api_key="your_camel_ai_api_key",
    model="gpt-4-turbo"
)

# Original functionality
paper = writer.generate_content(
    topic="Blockchain security vulnerabilities",
    format="academic",
    length="medium"
)
print(paper)

# Access specialized agents through the writer
network_agent = writer.get_network_agent()
network_analysis = network_agent.analyze_nmap_output("path/to/nmap_output.xml")

forensics_agent = writer.get_forensics_agent()
investigation_plan = forensics_agent.generate_investigation_plan(
    case_type="data_exfiltration"
)

wireless_agent = writer.get_wireless_agent()
recommendations = wireless_agent.generate_security_recommendations(
    network_type="home"
)
```

## Advanced Usage

### Combining Agent Capabilities

The agents can work together for comprehensive security analysis:

```python
from cybersecurity_agents import (
    NetworkAnomalyDetector,
    ForensicsPlanner,
    WirelessMobileSecurityAssessor
)

# Initialize agents
network_agent = NetworkAnomalyDetector()
forensics_agent = ForensicsPlanner()
wireless_agent = WirelessMobileSecurityAssessor()

# Use network analysis to inform forensics planning
network_scan = network_agent.analyze_nmap_output("network_scan.xml")
vulnerable_systems = network_agent.extract_vulnerable_systems(network_scan)

# Create targeted forensics plans for vulnerable systems
for system in vulnerable_systems:
    plan = forensics_agent.generate_investigation_plan(
        case_type="vulnerability_exploitation",
        target_systems=[system['hostname']],
        ip_address=system['ip']
    )
    print(f"Forensic plan for {system['hostname']}:")
    print(plan)

# Assess wireless security based on network findings
if any(system['service'] == 'wifi' for system in vulnerable_systems):
    wireless_recommendations = wireless_agent.generate_security_recommendations(
        network_type="enterprise",
        vulnerabilities_found=True
    )
    print("Wireless security recommendations:")
    print(wireless_recommendations)
```

### Monetization Features

If monetization is enabled in the configuration, usage will be tracked and billed according to API usage:

```python
from cyber_writer import CyberWriter

# Initialize with monetization enabled
writer = CyberWriter(
    api_key="your_camel_ai_api_key",
    model="gpt-4-turbo",
    enable_monetization=True,
    billing_id="customer123"
)

# Usage will be tracked and billed
content = writer.generate_content(
    topic="Zero-day vulnerability analysis",
    format="report"
)

# Get billing information
usage_stats = writer.get_usage_statistics()
print(f"Tokens used: {usage_stats['tokens']}")
print(f"Cost: ${usage_stats['cost']}")
```

# Camel AI-Powered Cybersecurity Blog

A sophisticated platform leveraging Camel AI to generate high-quality, technical cybersecurity blog content with built-in monetization capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Monetization Strategies](#monetization-strategies)
- [Deployment](#deployment)
  - [Local Deployment](#local-deployment)
  - [Google Cloud Deployment](#google-cloud-deployment)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## 🔭 Overview

This project uses Camel AI to automatically generate expert-level cybersecurity blog content. It's designed for cybersecurity professionals, educators, and companies looking to maintain a consistent publishing schedule of technical content without the typical resource investment.

Key features:
- AI-driven content generation tailored specifically for cybersecurity topics
- Flexible configuration supporting both OpenAI API and local models
- Built-in monetization tools for generating revenue
- Customizable templates for consistent branding
- Google Cloud deployment support

## 📁 Project Structure

```
/Users/davidortiz/Projects/Camel/
├── config/
│   └── config.yaml         # Configuration for API keys and model settings
├── scripts/
│   └── monetization_setup.py  # Scripts for monetization integration
├── templates/
│   └── cybersecurity_blog_template.md  # Blog post templates
├── hosting/
│   └── app.yaml            # Google Cloud App Engine configuration
├── cyber_writer.py         # Main script for generating content
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 📋 Prerequisites

- Python 3.12.9 or higher
- API keys for OpenAI (if using their API)
- Google Cloud account (for cloud deployment)
- Basic understanding of cybersecurity concepts

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd /Users/davidortiz/Projects/Camel/
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your configuration:
   ```bash
   cp config/config.yaml config/config.local.yaml
   # Edit config.local.yaml with your API keys and preferences
   ```

## ⚙️ Configuration

The system can be configured through the `config/config.yaml` file:

1. API Configuration:
   - Set your OpenAI API key
   - Configure model parameters (temperature, max_tokens, etc.)
   - Enable/disable research tools

2. Model Selection:
   - Choose between OpenAI API or local models
   - Configure model-specific parameters

3. Content Settings:
   - Adjust topic preferences
   - Set technical depth level
   - Configure content length parameters

Example configuration:
```yaml
api:
  openai:
    api_key: "your-api-key-here"
    model: "gpt-4-turbo"
    temperature: 0.7
    max_tokens: 4000
  
model:
  use_local: false
  local_model_path: ""
  
content:
  technical_level: "advanced"  # basic, intermediate, advanced
  include_code_examples: true
  target_word_count: 1500
  research_integration: true
```

## 🚀 Usage

### Generating a Blog Post

1. Ensure your configuration is set up correctly
2. Run the main script:
   ```bash
   python cyber_writer.py --topic "Zero-Day Vulnerabilities" --output-file "zero-day-article.md"
   ```

### Available Commands

- Generate a basic post:
  ```bash
  python cyber_writer.py --topic "Topic Name"
  ```

- Specify output format and location:
  ```bash
  python cyber_writer.py --topic "Topic Name" --format html --output-dir ./published/
  ```

- Use a specific template:
  ```bash
  python cyber_writer.py --topic "Topic Name" --template templates/custom_template.md
  ```

- Schedule regular posting:
  ```bash
  python cyber_writer.py --schedule weekly --topics-file topic_list.txt
  ```

## 💰 Monetization Strategies

The system includes several monetization options that can be configured:

1. **Affiliate Link Integration**
   - Automatically inserts relevant affiliate links for cybersecurity tools
   - Configure affiliate programs in `scripts/monetization_setup.py`
   - Control density and placement of affiliate links

2. **Ad Placement**
   - Strategic ad placement markers in generated content
   - Compatible with Google AdSense and other ad networks
   - Configurable ad frequency and position

3. **Premium Content Gating**
   - Tag sections of content as premium/paid
   - Integration with popular payment processors
   - Support for subscription models

4. **Sponsored Content Integration**
   - Tools for transparently marking sponsored content
   - Customizable sponsorship disclosure templates
   - Analytics for sponsor performance

Example usage:
```bash
# Enable affiliate link integration
python cyber_writer.py --topic "Endpoint Protection" --monetize affiliate

# Enable multiple monetization strategies
python cyber_writer.py --topic "Ransomware Protection" --monetize affiliate,ads
```

Configure monetization settings in `config/config.yaml`:
```yaml
monetization:
  affiliate:
    enabled: true
    programs:
      - name: "CyberVendor"
        base_url: "https://example.com/aff/"
        product_mapping:
          antivirus: "av-product"
          firewall: "fw-product"
  
  ads:
    enabled: true
    density: "medium"  # low, medium, high
    positions: ["top", "middle", "bottom"]
```

## 🌐 Deployment

### Local Deployment

For local testing and development:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script locally:
   ```bash
   python cyber_writer.py --topic "Your Topic"
   ```

### Google Cloud Deployment

For production deployment on Google Cloud App Engine:

1. Install Google Cloud SDK:
   ```bash
   # Install as per Google's instructions for your OS
   ```

2. Initialize Google Cloud SDK:
   ```bash
   gcloud init
   ```

3. Deploy to App Engine:
   ```bash
   cd /Users/davidortiz/Projects/Camel/
   gcloud app deploy hosting/app.yaml
   ```

4. Set up scheduled jobs (optional):
   ```bash
   gcloud scheduler jobs create http weekly-post --schedule="every monday 09:00" \
     --uri="https://your-app-url.appspot.com/generate" \
     --http-method=POST \
     --headers="Content-Type=application/json" \
     --message-body='{"topic": "weekly-security-roundup"}'
   ```

## 🔧 Customization

### Creating Custom Templates

1. Create a new template file in the `templates/` directory
2. Use the following placeholders in your template:
   - `{title}`: Article title
   - `{date}`: Publication date
   - `{intro}`: Introduction paragraph
   - `{main_content}`: Main article content
   - `{technical_details}`: Technical details section
   - `{code_examples}`: Code examples if applicable
   - `{mitigation}`: Mitigation strategies section
   - `{conclusion}`: Conclusion paragraph

### Extending Functionality

The project is designed for easy extension:

1. Add new scripts to the `scripts/` directory
2. Modify the configuration in `config/config.yaml`
3. Extend the `cyber_writer.py` script with additional functionality

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

