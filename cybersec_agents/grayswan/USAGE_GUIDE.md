# Gray Swan Arena - Detailed Usage Guide

This guide provides detailed instructions for using the Gray Swan Arena framework, with a focus on the new features: browser automation for exploit delivery, enhanced evaluation and visualization capabilities, and improved Discord integration.

## Table of Contents

1. [Setup and Configuration](#setup-and-configuration)
2. [Reconnaissance Agent](#reconnaissance-agent)
3. [Prompt Engineer Agent](#prompt-engineer-agent)
4. [Exploit Delivery Agent](#exploit-delivery-agent)
   - [API-Based Testing](#api-based-testing)
   - [Browser Automation with Playwright](#browser-automation-with-playwright)
   - [Browser Automation with Selenium](#browser-automation-with-selenium)
   - [Customizing Selectors](#customizing-selectors)
5. [Evaluation Agent](#evaluation-agent)
   - [Creating Visualizations](#creating-visualizations)
   - [Generating Reports](#generating-reports)
6. [Discord Integration](#discord-integration)
   - [Channel Configuration](#channel-configuration)
   - [Searching and Saving Results](#searching-and-saving-results)
7. [Advanced Configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)

## Setup and Configuration

### Installation

To use all features of Gray Swan Arena, install the package and its dependencies:

```bash
# Install the main package
pip install -e .

# Install browser automation dependencies
pip install playwright selenium webdriver-manager
playwright install  # Install browser binaries for Playwright
```

### Environment Configuration

Create a `.env` file in the `cybersec_agents/grayswan` directory by copying the `.env.example` file:

```bash
cp cybersec_agents/grayswan/.env.example cybersec_agents/grayswan/.env
```

Edit the `.env` file to include your API keys and other configuration options:

```
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_huggingface_key

# Discord Integration
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_CHANNEL_IDS=123456789012345678,987654321098765432

# Web Automation Settings
HEADLESS=False
GRAY_SWAN_URL=https://example.com/gray-swan

# General Settings
MAX_RETRIES=3
LOG_LEVEL=INFO
```

## Reconnaissance Agent

The Reconnaissance Agent gathers information about target AI models to inform your red-teaming approach.

### Web Search Capabilities

```python
from cybersec_agents.grayswan import ReconAgent

# Initialize the agent
recon_agent = ReconAgent()

# Perform web searches
model_info = recon_agent.run_web_search("Brass Fox language model capabilities")
safeguards_info = recon_agent.run_web_search("AI safeguards bypassing techniques")
jailbreak_info = recon_agent.run_web_search("jailbreaking techniques for LLMs")
```

### Discord Search Integration

```python
# Search Discord for relevant information
discord_results = recon_agent.run_discord_search("Gray Swan Arena jailbreak techniques")
```

### Generating Reports

```python
# Generate a comprehensive report
report = recon_agent.generate_report(
    model_info=model_info,
    behavior_info=safeguards_info,
    techniques_info=jailbreak_info,
    discord_info=discord_results  # Optional
)

# Save the report to a file
report_path = recon_agent.save_report(report)
```

## Prompt Engineer Agent

The Prompt Engineer Agent creates sophisticated prompts designed to test model safeguards.

### Loading Reports and Generating Prompts

```python
from cybersec_agents.grayswan import PromptEngineerAgent

# Initialize the agent
prompt_agent = PromptEngineerAgent()

# Load reconnaissance data
recon_report = prompt_agent.load_recon_report("path/to/recon_report.json")

# Generate prompts based on the recon data
prompts = prompt_agent.generate_prompts(
    recon_report=recon_report,
    num_prompts=20,  # Number of prompts to generate
    technique=None   # Specific technique to focus on (optional)
)
```

### Evaluating and Saving Prompts

```python
# Analyze prompt diversity
diversity = prompt_agent.evaluate_prompt_diversity(prompts)
print(f"Generated {diversity['total_prompts']} prompts using {len(diversity['technique_distribution'])} techniques")

# Save prompts to a file
prompt_path = prompt_agent.save_prompts(prompts)
```

## Exploit Delivery Agent

The enhanced Exploit Delivery Agent supports multiple methods for executing prompts against target models:

### API-Based Testing

Use this method when testing against AI model APIs directly:

```python
from cybersec_agents.grayswan import ExploitDeliveryAgent
from camel.types import ModelType, ModelPlatformType

# Initialize with a specific target model
agent = ExploitDeliveryAgent(
    target_model_type=ModelType.GPT_3_5_TURBO,
    target_model_platform=ModelPlatformType.OPENAI
)

# Load prompts
prompts = agent.load_prompts("path/to/prompt_list.json")

# Execute prompts against the API
results = agent.execute_prompt_batch(
    prompts=prompts,
    max_concurrent=3  # Number of concurrent requests
)
```

### Browser Automation with Playwright

Use Playwright for web-based testing:

```python
# Initialize without a specific model (for browser automation)
agent = ExploitDeliveryAgent()

# Execute prompts using Playwright
results = agent.run_prompts(
    prompts=prompts,
    target_model="Brass Fox Legendary",  # Model name in the web interface
    target_behavior="Harmful instructions",  # Behavior to test
    method="playwright"  # Specify Playwright method
)
```

### Browser Automation with Selenium

Use Selenium as an alternative for web-based testing:

```python
# Execute prompts using Selenium
results = agent.run_prompts(
    prompts=prompts,
    target_model="Brass Fox Legendary",
    target_behavior="Harmful instructions",
    method="selenium"  # Specify Selenium method
)
```

### Customizing Selectors

For browser automation, you may need to customize the CSS selectors based on the actual web interface:

```python
agent = ExploitDeliveryAgent()

# Override default selectors for your specific web interface
agent.selectors = {
    "model_select": "#custom-model-dropdown",
    "behavior_select": "#behavior-type",
    "prompt_input": ".prompt-textarea",
    "submit_button": "button.submit-btn",
    "response_output": ".response-container",
    "success_indicator": ".success-icon"
}

# Run prompts with custom selectors
results = agent.run_prompts(prompts, "Target Model", "Test Behavior", "playwright")
```

### Saving and Analyzing Results

```python
# Save results to a file
results_path = agent.save_results(results)

# Analyze the results
analysis = agent.analyze_results(results)
print(f"Success rate: {analysis['success_rate'] * 100:.2f}%")

# Most effective techniques
for technique, stats in analysis["technique_stats"].items():
    print(f"{technique}: {stats['rate'] * 100:.2f}% success rate")
```

## Evaluation Agent

The Evaluation Agent provides enhanced reporting and visualization capabilities.

### Loading and Analyzing Results

```python
from cybersec_agents.grayswan import EvaluationAgent

# Initialize the evaluation agent
eval_agent = EvaluationAgent()

# Load results and recon report
results = eval_agent.load_exploit_results("path/to/exploit_results.json")
recon_report = eval_agent.load_recon_report("path/to/recon_report.json")  # Optional

# Calculate statistics
statistics = eval_agent.calculate_statistics(results)
```

### Creating Visualizations

```python
# Create a directory for visualizations
import os
vis_dir = os.path.join("data", "evaluation_reports", "visualizations")
os.makedirs(vis_dir, exist_ok=True)

# Generate visualizations
visualization_paths = eval_agent.create_visualizations(statistics, vis_dir)
```

The visualizations include:
- Overall success rate pie chart
- Success rate by model (horizontal bar chart)
- Success rate by technique (horizontal bar chart)

### Generating Reports

```python
# Generate a comprehensive report
report = eval_agent.generate_report(
    results=results,
    statistics=statistics,
    recon_report=recon_report,  # Optional
    visualization_paths=visualization_paths  # Optional
)

# Save the report as JSON
json_path = eval_agent.save_report(report)

# Generate a markdown report with embedded visualizations
markdown_path = eval_agent.generate_markdown_report(report)
```

The generated markdown report includes:
- Summary statistics
- Success rates by model, technique, and target behavior
- Visualizations
- AI-generated analysis and recommendations
- Example prompts and responses

## Discord Integration

The enhanced Discord integration provides more powerful search capabilities.

### Channel Configuration

Configure Discord channel IDs in your `.env` file:

```
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_CHANNEL_IDS=123456789012345678,987654321098765432
```

### Searching and Saving Results

```python
from cybersec_agents.grayswan.utils.discord_utils import DiscordScraper

# Initialize the Discord scraper
discord_scraper = DiscordScraper()

# Search for messages
results = discord_scraper.search(
    query="jailbreaking techniques",
    channel_ids=None,  # Will use default channels from .env
    limit=100  # Max messages per channel
)

# Format and print results
formatted_results = discord_scraper.format_results(
    results,
    include_metadata=True  # Include attachments and mentions
)
print(formatted_results)

# Save results to a file
discord_scraper.save_results(results, "jailbreak_search_results.json")
```

## Advanced Configuration

### Logging Configuration

Customize logging behavior:

```python
from cybersec_agents.grayswan.utils.logging_utils import setup_logging
import logging

# Set up logger with custom configuration
logger = setup_logging(
    name="CustomLogger",
    log_level=logging.DEBUG,
    log_to_file=True,
    log_filename="custom_log.log"
)

# Use the logger
logger.debug("This is a debug message")
logger.info("This is an info message")
```

### Web Automation Settings

Configure browser automation behavior in your `.env` file:

```
# Run browsers in headless mode (no visible window)
HEADLESS=True

# Specify the URL of the web interface
GRAY_SWAN_URL=https://example.com/gray-swan

# Control retry behavior
MAX_RETRIES=5
```

### Controlling Visualization Appearance

When creating visualizations, you can modify the appearance through matplotlib:

```python
import matplotlib.pyplot as plt

# Set global style
plt.style.use('dark_background')  # Or 'ggplot', 'seaborn', etc.

# Then call the visualization function
visualization_paths = eval_agent.create_visualizations(statistics, vis_dir)
```

## Troubleshooting

### Browser Automation Issues

If you encounter issues with browser automation:

1. Verify that you have installed both Playwright and Selenium:
   ```bash
   pip install playwright selenium webdriver-manager
   playwright install
   ```

2. Check that the selectors match your target web interface by examining the HTML of the page.

3. Try running in non-headless mode to see what's happening:
   ```
   HEADLESS=False
   ```

4. Increase logging level for detailed logs:
   ```
   LOG_LEVEL=DEBUG
   ```

### Discord Integration Issues

If Discord integration isn't working:

1. Verify your bot token and ensure it has the necessary permissions.
2. Check that you've provided valid channel IDs.
3. Make sure the bot is actually a member of the servers/channels you're trying to search.
4. Increase the timeout if searches are timing out:
   ```python
   # Modify the timeout in the code (default is 120 seconds)
   timeout_seconds = 300  # 5 minutes
   ```

### API Call Failures

If API calls to language models are failing:

1. Verify your API keys in the `.env` file.
2. Check for rate limiting or quota issues.
3. Increase the number of retries:
   ```
   MAX_RETRIES=5
   ```

### Visualization Errors

If visualization generation fails:

1. Make sure matplotlib is installed:
   ```bash
   pip install matplotlib
   ```

2. Check for missing dependencies (especially on headless servers):
   ```bash
   pip install seaborn numpy
   ```

3. Switch to a non-GUI backend for headless environments:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   ``` 