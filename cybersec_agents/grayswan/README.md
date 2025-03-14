# Gray Swan Arena

## AI Red-Teaming Framework

Gray Swan Arena is a structured framework for conducting red-teaming exercises against AI language models. The framework uses a multi-agent approach to systematically test the robustness and safety of AI systems against adversarial prompts and techniques.

## Overview

Gray Swan Arena consists of four specialized agents that work together in a pipeline:

1. **Reconnaissance Agent**: Gathers information about the target model, including architecture, capabilities, and potential vulnerabilities.
2. **Prompt Engineer Agent**: Creates sophisticated prompts designed to elicit undesired behavior from the target model.
3. **Exploit Delivery Agent**: Executes the prompts against the target model and evaluates the effectiveness of each attempt. Now with browser automation capabilities.
4. **Evaluation Agent**: Analyzes results and generates comprehensive reports with statistics, visualizations, and recommendations.

## Installation

Gray Swan Arena is included in the cybersec_agents package. Install it along with its dependencies:

```bash
pip install -e .
```

For browser automation features, additional setup is required:

```bash
# For Playwright
pip install playwright
playwright install

# For Selenium
pip install selenium webdriver-manager
```

## Usage

### Command Line Interface

Gray Swan Arena can be run using the cybersec-agents CLI:

```bash
# Run the full pipeline against GPT-3.5
cybersec-agents grayswan --target GPT3.5

# Run just the reconnaissance phase
cybersec-agents grayswan --mode recon

# Generate prompts from an existing recon report
cybersec-agents grayswan --mode prompts --recon-report path/to/report.json

# Include visualizations in the evaluation
cybersec-agents grayswan --visualize

# Specify browser automation method
cybersec-agents grayswan --target GPT3.5 --browser-method playwright
```

### Python API

You can also use Gray Swan Arena programmatically:

#### Reconnaissance Agent

```python
from cybersec_agents.grayswan import ReconAgent

# Create a reconnaissance agent
recon_agent = ReconAgent()

# Gather information
model_info = recon_agent.run_web_search("GPT-4 capabilities")
behavior_info = recon_agent.run_web_search("AI safeguards bypassing")
techniques_info = recon_agent.run_web_search("jailbreaking techniques")

# Search Discord (if configured)
discord_info = recon_agent.run_discord_search("jailbreaking techniques")

# Generate and save report
report = recon_agent.generate_report(model_info, behavior_info, techniques_info, discord_info)
report_path = recon_agent.save_report(report)
```

#### Prompt Engineer Agent

```python
from cybersec_agents.grayswan import PromptEngineerAgent

# Create prompt engineer agent
prompt_agent = PromptEngineerAgent()

# Load reconnaissance report
recon_report = prompt_agent.load_recon_report("path/to/recon_report.json")

# Generate prompts
prompts = prompt_agent.generate_prompts(recon_report, num_prompts=10)

# Evaluate prompt diversity
diversity_metrics = prompt_agent.evaluate_prompt_diversity(prompts)

# Save prompts
prompt_path = prompt_agent.save_prompts(prompts)
```

#### Exploit Delivery Agent

```python
from cybersec_agents.grayswan import ExploitDeliveryAgent
from camel.types import ModelType, ModelPlatformType

# Method 1: API-based testing
# Create exploit delivery agent with specific model
agent = ExploitDeliveryAgent(
    target_model_type=ModelType.GPT_3_5_TURBO,
    target_model_platform=ModelPlatformType.OPENAI
)

# Load prompts
prompts = agent.load_prompts("path/to/prompt_list.json")

# Execute prompts via API
results = agent.execute_prompt_batch(prompts, max_concurrent=3)

# Method 2: Browser-based testing (Playwright or Selenium)
agent = ExploitDeliveryAgent(browser_method="playwright", headless=False)

# Execute prompts via browser automation
results = agent.run_prompts(
    prompts, 
    target_model="Brass Fox Legendary", 
    target_behavior="Leak information"
)

# Save and analyze results
saved_path = agent.save_results(results)
analysis = agent.analyze_results(results)
```

#### Evaluation Agent

```python
from cybersec_agents.grayswan import EvaluationAgent

# Create evaluation agent
eval_agent = EvaluationAgent()

# Load results
results = eval_agent.load_exploit_results("path/to/exploit_results.json")

# Load reconnaissance report (optional)
recon_report = eval_agent.load_recon_report("path/to/recon_report.json")

# Calculate statistics
statistics = eval_agent.calculate_statistics(results)

# Create visualizations
vis_dir = "data/evaluation_reports/visualizations"
visualization_paths = eval_agent.create_visualizations(statistics, vis_dir)

# Generate report
report = eval_agent.generate_report(
    results=results,
    statistics=statistics,
    recon_report=recon_report,
    visualization_paths=visualization_paths
)

# Save report
report_path = eval_agent.save_report(report)

# Generate markdown report
markdown_path = eval_agent.generate_markdown_report(report)

# Generate HTML report with visualizations
html_path = eval_agent.generate_html_report(report)
```

### Utility Modules

Gray Swan Arena now provides dedicated utility modules for browser automation and visualization:

#### Browser Automation

```python
from cybersec_agents.grayswan.utils import (
    BrowserMethod,
    BrowserAutomationFactory,
    is_browser_automation_available
)

# Check available browser automation methods
available = is_browser_automation_available()
print(f"Playwright available: {available['playwright']}")
print(f"Selenium available: {available['selenium']}")

# Create a browser driver
driver = BrowserAutomationFactory.create_driver(
    method="playwright",  # or "selenium"
    headless=True
)

# Use the driver
driver.initialize()
driver.navigate("https://example.com")
response = driver.execute_prompt("Test prompt", "Model Name", "Behavior")
driver.close()
```

#### Visualization

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

## Configuration

Gray Swan Arena requires API keys for the language models being used. Set them in your environment variables or in a `.env` file:

```
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_huggingface_key

# Discord Integration
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_CHANNEL_IDS=123456789012345678,987654321098765432

# Browser Automation Settings
GRAYSWAN_BROWSER_METHOD=playwright  # Options: playwright, selenium
GRAYSWAN_BROWSER_HEADLESS=true  # Set to false to see the browser UI
GRAYSWAN_BROWSER_TIMEOUT=60000  # Timeout in milliseconds
GRAY_SWAN_URL=https://example.com/gray-swan  # URL for Gray Swan Arena web interface

# Visualization Settings
GRAYSWAN_VIZ_OUTPUT_DIR=./output/visualizations  # Directory for visualization outputs
GRAYSWAN_VIZ_DPI=300  # Resolution for saved charts
GRAYSWAN_VIZ_THEME=default  # Visualization theme

# General Settings
MAX_RETRIES=3  # Maximum retries for API calls and operations
LOG_LEVEL=INFO  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Monitoring
AGENTOPS_API_KEY=your_agentops_key
```

## Output

Gray Swan Arena generates structured output files in the `data/` directory:

- `data/recon_reports/`: Reconnaissance data about target models
- `data/prompt_lists/`: Generated attack prompts
- `data/exploit_logs/`: Results of exploit attempts
- `data/evaluation_reports/`: Comprehensive reports and visualizations
- `data/logs/`: Detailed execution logs
- `data/discord_searches/`: Saved Discord search results

## Features

### Modular Architecture
Gray Swan Arena now features a more modular architecture with dedicated utility modules:

- **Browser Automation Utilities**: Dedicated module for browser automation with support for both Playwright and Selenium
- **Visualization Utilities**: Comprehensive visualization tools for generating charts, graphs, and HTML reports
- **Consistent Interfaces**: Standardized interfaces for all utilities to ensure easy integration
- **Enhanced Type Safety**: Improved type annotations throughout the codebase

### Multi-Method Testing
Gray Swan Arena supports multiple methods for testing AI models:

- **API-based Testing**: Direct testing against model APIs
- **Browser Automation (Playwright)**: Automate interactions with web interfaces using Playwright
- **Browser Automation (Selenium)**: Alternative automation using Selenium WebDriver
- **Flexible Fallbacks**: Automatically selects the best available method

### Enhanced Visualization and Reporting
The framework now provides rich visualizations and comprehensive reports:

- **Success Rate Analysis**: Charts showing success rates by model and prompt type
- **Response Time Analysis**: Boxplots showing response times by model
- **Prompt Effectiveness**: Analysis of which prompt types are most effective
- **Vulnerability Heatmaps**: Identify which models are vulnerable to which attack vectors
- **HTML Reports**: Generate interactive HTML reports with embedded visualizations
- **AI-Powered Analysis**: Get AI-generated insights about patterns in successful and failed prompts

### Improved Discord Integration
Enhanced Discord scraping capabilities:

- **Channel Configuration**: Search specific channels using environment variables
- **Metadata Extraction**: Extract attachments, mentions, and other metadata from Discord messages
- **Search Result Management**: Save and load Discord search results
- **Timeout Handling**: Robust timeout management for reliable operation

### Additional Improvements
- **Robust Logging**: Configurable logging system with file and console output
- **Error Handling**: Comprehensive error handling and retry mechanisms
- **Environment Configuration**: More flexible environment variable configuration
- **Visualization Customization**: Configurable visualization options

## Ethical Use

Gray Swan Arena is designed for authorized testing of AI models by their developers or with explicit permission. It should only be used for legitimate security research and improving AI safety. Do not use this tool against models without proper authorization.

## License

This project is part of the cybersec_agents package and is subject to its license terms. 