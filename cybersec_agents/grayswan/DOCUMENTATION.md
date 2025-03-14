# Gray Swan Arena Documentation

Welcome to the Gray Swan Arena documentation. This index provides an overview of the available documentation resources and guides you to the appropriate documents based on your needs.

## Overview

Gray Swan Arena is a comprehensive framework for red-teaming AI language models. It provides structured tools and methodologies for:

1. **Reconnaissance**: Gathering information about target AI models
2. **Prompt Engineering**: Creating sophisticated prompts designed to test model safeguards
3. **Exploit Delivery**: Executing prompts against target models via APIs or browser automation
4. **Evaluation**: Analyzing results and generating comprehensive reports

The framework is designed with a modular architecture that allows for both end-to-end assessments and use of individual components.

## Documentation Resources

### [README.md](README.md)

**Purpose**: Provides a high-level overview of the Gray Swan Arena framework
**When to use**: Start here if you're new to the framework and want to understand its purpose, structure, and basic usage.

**Contents**:
- Framework overview
- Installation instructions
- Basic command line and API usage
- Configuration requirements
- Output file structure
- Features summary
- Ethical considerations

### [USAGE_GUIDE.md](USAGE_GUIDE.md)

**Purpose**: Detailed explanation of all features and how to use them
**When to use**: When you need detailed information about specific features or components of the framework.

**Contents**:
- Setup and configuration
- Detailed API documentation for all agents
- Browser automation configuration
- Discord integration setup
- Advanced configuration options
- Troubleshooting common issues

### [TUTORIAL.md](TUTORIAL.md)

**Purpose**: Step-by-step walkthrough of a complete red-team assessment
**When to use**: When you want to see how all the components work together in a realistic scenario.

**Contents**:
- Environment setup
- Running reconnaissance
- Generating prompts
- Executing exploits
- Evaluating results
- Complete end-to-end assessment script

### [CHANGELOG.md](CHANGELOG.md)

**Purpose**: Details of changes between versions
**When to use**: When upgrading from a previous version or to check what features are available in your current version.

**Contents**:
- Major features added in each version
- Agent improvements
- Utility enhancements
- Bug fixes

### [.env.example](.env.example)

**Purpose**: Template for environment variable configuration
**When to use**: When setting up the framework for the first time or configuring new features.

**Contents**:
- Required API keys
- Optional configuration settings
- Browser automation settings
- Discord integration options

## Framework Architecture

Gray Swan Arena follows a modular architecture centered around four main agent types:

```
Gray Swan Arena
│
├── Agents
│   ├── ReconAgent - Gathers information about target models
│   ├── PromptEngineerAgent - Generates sophisticated attack prompts
│   ├── ExploitDeliveryAgent - Executes prompts against targets
│   └── EvaluationAgent - Analyzes results and generates reports
│
├── Utilities
│   ├── logging_utils.py - Structured logging functionality
│   ├── discord_utils.py - Discord channel search capabilities
│   ├── browser_utils.py - Browser automation utilities
│   └── visualization_utils.py - Data visualization and reporting tools
│
└── Data
    ├── recon_reports/ - Stores reconnaissance findings
    ├── prompt_lists/ - Stores generated attack prompts
    ├── exploit_logs/ - Stores execution results
    ├── evaluation_reports/ - Stores final reports
    └── logs/ - Stores application logs
```

## Getting Started

If you're new to Gray Swan Arena, we recommend the following path through the documentation:

1. **Start with [README.md](README.md)** to understand the framework's purpose and basic structure.
2. **Follow [TUTORIAL.md](TUTORIAL.md)** to run your first end-to-end assessment.
3. **Refer to [USAGE_GUIDE.md](USAGE_GUIDE.md)** for detailed information about specific features.
4. **Check [CHANGELOG.md](CHANGELOG.md)** to stay updated on new features and improvements.

## API Reference

Quick reference for the main classes and their methods:

### ReconAgent

```python
from cybersec_agents.grayswan import ReconAgent

# Initialize
agent = ReconAgent()

# Key methods
web_search_results = agent.run_web_search("search query")
discord_results = agent.run_discord_search("search query")
report = agent.generate_report(model_info, behavior_info, techniques_info)
report_path = agent.save_report(report)
```

### PromptEngineerAgent

```python
from cybersec_agents.grayswan import PromptEngineerAgent

# Initialize
agent = PromptEngineerAgent()

# Key methods
recon_report = agent.load_recon_report("path/to/report.json")
prompts = agent.generate_prompts(recon_report, num_prompts=20)
diversity = agent.evaluate_prompt_diversity(prompts)
prompt_path = agent.save_prompts(prompts)
```

### ExploitDeliveryAgent

```python
from cybersec_agents.grayswan import ExploitDeliveryAgent
from camel.types import ModelType, ModelPlatformType

# Initialize for API testing
agent = ExploitDeliveryAgent(
    target_model_type=ModelType.GPT_3_5_TURBO,
    target_model_platform=ModelPlatformType.OPENAI
)

# Initialize for browser testing with Playwright (default)
agent = ExploitDeliveryAgent(browser_method="playwright", headless=False)

# Initialize for browser testing with Selenium
agent = ExploitDeliveryAgent(browser_method="selenium", headless=True)

# Key methods
prompts = agent.load_prompts("path/to/prompts.json")
api_results = agent.execute_prompt_batch(prompts, max_concurrent=3)
browser_results = agent.run_prompts(prompts, "Target Model", "Target Behavior", "playwright")
analysis = agent.analyze_results(results)
results_path = agent.save_results(results)
```

### EvaluationAgent

```python
from cybersec_agents.grayswan import EvaluationAgent

# Initialize
agent = EvaluationAgent()

# Key methods
results = agent.load_exploit_results("path/to/results.json")
recon_report = agent.load_recon_report("path/to/report.json")
statistics = agent.calculate_statistics(results)
visualization_paths = agent.create_visualizations(statistics, "output_dir")
report = agent.generate_report(results, statistics, recon_report, visualization_paths)
json_path = agent.save_report(report)
markdown_path = agent.generate_markdown_report(report)
html_path = agent.generate_html_report(report)
```

## Utility Modules

### Browser Automation Utilities

Gray Swan Arena provides flexible browser automation through the `browser_utils` module:

```python
from cybersec_agents.grayswan.utils import (
    BrowserMethod,
    BrowserAutomationFactory,
    is_browser_automation_available
)

# Check which browser automation methods are available
available_methods = is_browser_automation_available()
print(f"Playwright available: {available_methods['playwright']}")
print(f"Selenium available: {available_methods['selenium']}")

# Create a browser driver
driver = BrowserAutomationFactory.create_driver(
    method="playwright",  # or "selenium"
    headless=True
)

# Initialize and use the driver
driver.initialize()
driver.navigate("https://example.com")
response = driver.execute_prompt("Test prompt", "Model Name", "Behavior Name")
driver.close()
```

Key features:
- Supports both Playwright and Selenium
- Unified interface for both browser automation methods
- Headless mode support for CI/CD environments
- Robust error handling and logging

### Visualization Utilities

The framework includes comprehensive visualization tools through the `visualization_utils` module:

```python
from cybersec_agents.grayswan.utils import (
    create_success_rate_chart,
    create_response_time_chart,
    create_prompt_type_effectiveness_chart,
    create_vulnerability_heatmap,
    create_evaluation_report
)

# Create individual charts
success_chart = create_success_rate_chart(results, "output_dir")
time_chart = create_response_time_chart(results, "output_dir")
effectiveness_chart = create_prompt_type_effectiveness_chart(results, "output_dir")
heatmap = create_vulnerability_heatmap(results, "output_dir")

# Create a comprehensive HTML report with all visualizations
report_files = create_evaluation_report(results, "output_dir")
print(f"HTML report generated at: {report_files['html_report']}")
```

Key features:
- Success rate analysis by model and prompt type
- Response time analysis
- Prompt effectiveness evaluation
- Vulnerability mapping across models and attack vectors
- Comprehensive HTML reports with interactive elements

## Environment Variables

Gray Swan Arena uses environment variables for configuration. The following variables are relevant to browser automation and visualization:

```
# Browser Automation
GRAYSWAN_BROWSER_METHOD=playwright  # or selenium
GRAYSWAN_BROWSER_HEADLESS=true  # or false
GRAYSWAN_BROWSER_TIMEOUT=60000  # in milliseconds

# Visualization
GRAYSWAN_VIZ_OUTPUT_DIR=./output/visualizations
GRAYSWAN_VIZ_DPI=300  # Chart resolution
GRAYSWAN_VIZ_THEME=default  # Visualization theme
```

## Ethical Considerations

Gray Swan Arena is designed for authorized testing only. The framework should only be used:

1. On models you own or have explicit permission to test
2. In accordance with the terms of service of model providers
3. For the purpose of improving model safety and security
4. With appropriate safeguards to prevent unintended consequences

Unauthorized testing may violate laws, terms of service, or ethical standards. Always prioritize responsible use.

## Support and Contribution

For support, feature requests, or to contribute to the project:

1. **Submit Issues**: Use the GitHub issue tracker to report bugs or request features
2. **Pull Requests**: Contribute improvements following the guidelines in CONTRIBUTING.md
3. **Documentation**: Help improve these documents by submitting corrections or additions

We welcome contributions that enhance the framework's capabilities, improve documentation, or fix issues. 