# Comprehensive Agent Usage Guide

This guide provides detailed instructions for using the Gray Swan Arena agents available in the Camel AI framework.

## Table of Contents

1. [Gray Swan Arena Agents](#gray-swan-arena-agents)
   - [Reconnaissance Agent](#reconnaissance-agent)
   - [Prompt Engineer Agent](#prompt-engineer-agent)
   - [Exploit Delivery Agent](#exploit-delivery-agent)
   - [Evaluation Agent](#evaluation-agent)
2. [Model Selection and Configuration](#model-selection-and-configuration)
   - [Using o3-mini for Reasoning Tasks](#using-o3-mini-for-reasoning-tasks)
   - [Configuring GPT-4o as a Backup Model](#configuring-gpt-4o-as-a-backup-model)
3. [Running the Full Pipeline](#running-the-full-pipeline)
   - [Advanced Configuration](#advanced-configuration)

## Gray Swan Arena Agents

The Gray Swan Arena provides a set of specialized agents for testing AI safety and security.

### Reconnaissance Agent

The ReconAgent is responsible for gathering information about target AI models.

```python
from cybersec_agents import ReconAgent

# Initialize the agent
agent = ReconAgent(output_dir="./reports", model_name="gpt-4")

# Run web search on a target model
web_results = agent.run_web_search(
    target_model="GPT-4",
    target_behavior="bypass content filters"
)

# Run Discord search on a target model
discord_results = agent.run_discord_search(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    channels=["ai-ethics", "red-teaming"]
)

# Generate a comprehensive report
report = agent.generate_report(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    web_results=web_results,
    discord_results=discord_results
)

# Save the report
report_path = agent.save_report(
    report=report,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

### Prompt Engineer Agent

The PromptEngineerAgent generates effective prompts to test target models based on reconnaissance data.

```python
from cybersec_agents import PromptEngineerAgent

# Initialize the agent
agent = PromptEngineerAgent(output_dir="./prompts", model_name="gpt-4")

# Generate prompts based on reconnaissance data
prompts = agent.generate_prompts(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    recon_report=report,
    num_prompts=10
)

# Save the prompts
prompts_path = agent.save_prompts(
    prompts=prompts,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

### Exploit Delivery Agent

The ExploitDeliveryAgent is responsible for delivering prompts to target models and recording responses.

```python
from cybersec_agents import ExploitDeliveryAgent

# Initialize the agent
agent = ExploitDeliveryAgent(output_dir="./exploits", model_name="gpt-4")

# Run prompts against a target model
results = agent.run_prompts(
    prompts=prompts,
    target_model="gpt-4",
    target_behavior="bypass content filters",
    method="api",
    max_tries=3,
    delay_between_tries=2
)

# Save the results
results_path = agent.save_results(
    results=results,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

### Evaluation Agent

The EvaluationAgent analyzes exploit results, generates visualizations, and produces evaluation reports.

```python
from cybersec_agents import EvaluationAgent

# Initialize the agent
agent = EvaluationAgent(output_dir="./evaluations", model_name="gpt-4")

# Evaluate exploit results
evaluation = agent.evaluate_results(
    results=results,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)

# Save the evaluation
eval_path = agent.save_evaluation(
    evaluation=evaluation,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)

# Generate a summary report
summary = agent.generate_summary(
    evaluation=evaluation,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)

# Save the summary
summary_path = agent.save_summary(
    summary=summary,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)

# Create visualizations
visualizations = agent.create_visualizations(
    evaluation=evaluation,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

## Model Selection and Configuration

Gray Swan Arena agents support multiple AI models, allowing you to optimize performance and cost by selecting appropriate models for different tasks.

### Using o3-mini for Reasoning Tasks

For reasoning tasks that require good performance but don't need the full capabilities of larger models, you can configure agents to use o3-mini:

```python
# Example: Using o3-mini for the Evaluation Agent's reasoning tasks
from cybersec_agents import EvaluationAgent

# Initialize the agent with o3-mini for reasoning tasks
agent = EvaluationAgent(
    output_dir="./evaluations",
    model_name="o3-mini",  # Using o3-mini for reasoning tasks
    reasoning_model="o3-mini"  # Explicitly specify for reasoning tasks
)

# The model will efficiently perform reasoning tasks like classification and analysis
evaluation = agent.evaluate_results(
    results=exploit_results,
    target_model="gpt-4",
    target_behavior="jailbreak"
)

# Generate insight summaries using o3-mini's reasoning capabilities
summary = agent.generate_summary(
    evaluation=evaluation,
    target_model="gpt-4",
    target_behavior="jailbreak"
)
```

### Configuring GPT-4o as a Backup Model

You can configure agents to use GPT-4o as a backup model for tasks requiring advanced capabilities:

```python
# Example: Using o3-mini as primary with GPT-4o as backup for the Prompt Engineer Agent
from cybersec_agents import PromptEngineerAgent

# Initialize with configuration for both models
agent = PromptEngineerAgent(
    output_dir="./prompts",
    model_name="o3-mini",  # Primary model
    backup_model="gpt-4o"  # Backup model for complex tasks
)

# The agent will use o3-mini for initial prompt generation
# If the task complexity exceeds o3-mini's capabilities, it will automatically
# fall back to using GPT-4o
prompts = agent.generate_prompts(
    target_model="claude-3",
    target_behavior="jailbreak",
    recon_report=report,
    num_prompts=10,
    complexity_threshold=0.7  # Threshold for switching to backup model
)
```

## Advanced Configuration Example

Here's a more comprehensive example showing how to configure all agents with different models for different tasks:

```python
from cybersec_agents import ReconAgent, PromptEngineerAgent, ExploitDeliveryAgent, EvaluationAgent

# Define output directory
output_base_dir = "./gray_swan_results"

# Initialize agents with optimized model configuration
recon_agent = ReconAgent(
    output_dir=f"{output_base_dir}/reports",
    model_name="o3-mini",  # Good for initial research
    backup_model="gpt-4o",  # Falls back to GPT-4o for complex analysis
    web_search_model="o3-mini"  # Efficient for search parsing
)

prompt_agent = PromptEngineerAgent(
    output_dir=f"{output_base_dir}/prompts",
    model_name="gpt-4o",  # Creative task benefits from GPT-4o's capabilities
    reasoning_model="o3-mini"  # Use o3-mini for reasoning about prompt structure
)

exploit_agent = ExploitDeliveryAgent(
    output_dir=f"{output_base_dir}/exploits",
    model_name="o3-mini",  # Sufficient for delivery mechanics
    backup_model="gpt-4o",  # Falls back for complex scenarios
    analysis_model="o3-mini"  # Analyzing initial responses
)

eval_agent = EvaluationAgent(
    output_dir=f"{output_base_dir}/evaluations",
    model_name="o3-mini",  # Efficient for standard evaluations
    reasoning_model="o3-mini",  # Good performance for classification tasks
    visualization_model="gpt-4o"  # Better for complex visualization planning
)

# Target parameters
target_model = "claude-3"
target_behavior = "jailbreak"

# Run the pipeline with this optimized configuration
# ... (same pipeline steps as in the basic example)
```

## Running the Full Pipeline

You can run the full Gray Swan Arena pipeline using the main module:

```python
from cybersec_agents import grayswan_main

# Run the full pipeline with default configuration
grayswan_main.main()

# Or with custom model configuration
grayswan_main.main(
    recon_model="o3-mini",
    prompt_model="gpt-4o",
    exploit_model="o3-mini",
    eval_model="o3-mini",
    backup_model="gpt-4o"
)
```

Alternatively, you can run specific phases:

```python
from cybersec_agents.grayswan.main import run_reconnaissance

# Run just the reconnaissance phase with o3-mini
report = run_reconnaissance(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    output_dir="./reports",
    model_name="o3-mini",
    backup_model="gpt-4o"  # Use GPT-4o as backup for complex analysis
)
```
