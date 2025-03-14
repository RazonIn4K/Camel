# Comprehensive Agent Usage Guide

This guide provides detailed instructions for using the Gray Swan Arena agents within the Camel AI framework. These agents are designed for AI safety and security testing, focusing on identifying and evaluating potential vulnerabilities in AI systems.

## Table of Contents
- [Reconnaissance Agent](#reconnaissance-agent)
- [Prompt Engineer Agent](#prompt-engineer-agent)
- [Exploit Delivery Agent](#exploit-delivery-agent)
- [Evaluation Agent](#evaluation-agent)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Model Configuration](#model-configuration)
- [Advanced Usage](#advanced-usage)

## Reconnaissance Agent

The Reconnaissance Agent gathers information about target AI models, including their architecture, vulnerabilities, and community knowledge.

### Initializing the Agent

```python
from cybersec_agents.grayswan.agents import ReconAgent

# Basic initialization
agent = ReconAgent(output_dir="./reports", model_name="gpt-4")

# Using o3-mini for reasoning tasks and GPT-4o as a backup
agent = ReconAgent(
    output_dir="./reports",
    model_name="gpt-4o",
    reasoning_model="o3-mini",
    backup_model="gpt-4"
)
```

### Running Web Search

```python
web_results = agent.run_web_search(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    num_results=5
)
```

### Running Discord Search

```python
discord_results = agent.run_discord_search(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    num_results=5
)
```

### Generating Reports

```python
report = agent.generate_report(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    web_results=web_results,
    discord_results=discord_results
)
```

### Saving Reports

```python
report_path = agent.save_report(
    report=report,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

## Prompt Engineer Agent

The Prompt Engineer Agent generates effective prompts to test target models based on reconnaissance data.

### Initializing the Agent

```python
from cybersec_agents.grayswan.agents import PromptEngineerAgent

# Basic initialization
agent = PromptEngineerAgent(output_dir="./prompts", model_name="gpt-4")

# Using o3-mini for reasoning tasks
agent = PromptEngineerAgent(
    output_dir="./prompts",
    model_name="gpt-4o",
    reasoning_model="o3-mini"
)
```

### Generating Prompts

```python
prompts = agent.generate_prompts(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    recon_report=report
)
```

### Saving Prompts

```python
prompt_path = agent.save_prompts(
    prompts=prompts,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

## Exploit Delivery Agent

The Exploit Delivery Agent is responsible for delivering prompts to target models and recording responses.

### Initializing the Agent

```python
from cybersec_agents.grayswan.agents import ExploitDeliveryAgent

# Basic initialization
agent = ExploitDeliveryAgent(output_dir="./exploits", model_name="gpt-4")

# Using GPT-4o as a backup model
agent = ExploitDeliveryAgent(
    output_dir="./exploits",
    model_name="gpt-4",
    backup_model="gpt-4o"
)
```

### Running Prompts

```python
results = agent.run_prompts(
    prompts=prompts,
    target_model="GPT-4",
    method="api",  # or "browser"
    max_tries=3,
    delay_between_tries=2
)
```

### Saving Results

```python
results_path = agent.save_results(
    results=results,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

## Evaluation Agent

The Evaluation Agent analyzes exploit results, generates visualizations, and produces evaluation reports.

### Initializing the Agent

```python
from cybersec_agents.grayswan.agents import EvaluationAgent

# Basic initialization
agent = EvaluationAgent(output_dir="./evaluations", model_name="gpt-4")

# Using o3-mini for reasoning tasks and GPT-4o as a backup
agent = EvaluationAgent(
    output_dir="./evaluations",
    model_name="gpt-4o",
    reasoning_model="o3-mini",
    backup_model="gpt-4"
)
```

### Evaluating Results

```python
evaluation = agent.evaluate_results(
    results=results,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

### Saving Evaluation

```python
evaluation_path = agent.save_evaluation(
    evaluation=evaluation,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

### Generating Summary

```python
summary = agent.generate_summary(
    evaluation=evaluation,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

### Saving Summary

```python
summary_path = agent.save_summary(
    summary=summary,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

### Creating Visualizations

```python
visualization_paths = agent.create_visualizations(
    evaluation=evaluation,
    target_model="GPT-4",
    target_behavior="bypass content filters"
)
```

## Running the Full Pipeline

To run the full Gray Swan Arena pipeline, you can use the following code:

```python
from cybersec_agents.grayswan import grayswan_main

# Run the full pipeline with default settings
grayswan_main.main()

# Run with custom settings
grayswan_main.main(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    output_dir="./output",
    model_name="gpt-4o",
    reasoning_model="o3-mini",
    backup_model="gpt-4"
)
```

## Model Configuration

The Gray Swan Arena agents support different models for different tasks:

### Primary Model

The primary model is used for most operations and is specified with the `model_name` parameter:

```python
agent = ReconAgent(model_name="gpt-4o")
```

### Reasoning Model

The reasoning model is used for tasks that require complex reasoning, such as generating reports and evaluations:

```python
agent = ReconAgent(
    model_name="gpt-4o",
    reasoning_model="o3-mini"  # Use o3-mini for reasoning tasks
)
```

### Backup Model

The backup model is used if the primary model fails:

```python
agent = ExploitDeliveryAgent(
    model_name="gpt-4o",
    backup_model="gpt-4"  # Fall back to GPT-4 if GPT-4o fails
)
```

### Recommended Configuration

For optimal performance, we recommend the following configuration:

```python
agent = ReconAgent(
    model_name="gpt-4o",       # Use GPT-4o as the primary model
    reasoning_model="o3-mini", # Use o3-mini for reasoning tasks
    backup_model="gpt-4"       # Fall back to GPT-4 if needed
)
```

## Advanced Usage

For advanced usage scenarios, you can customize the agents further:

### Custom System Prompts

You can customize the system prompts used by the agents by modifying the agent's internal methods:

```python
from cybersec_agents.grayswan.agents import ReconAgent

class CustomReconAgent(ReconAgent):
    def _get_system_prompt(self, target_model, target_behavior):
        return f"Custom system prompt for {target_model} and {target_behavior}"
```

### Custom Output Formats

You can customize the output formats by extending the agent classes:

```python
from cybersec_agents.grayswan.agents import EvaluationAgent

class CustomEvaluationAgent(EvaluationAgent):
    def save_evaluation(self, evaluation, target_model, target_behavior):
        # Custom saving logic
        pass
```

### Parallel Processing

For large-scale testing, you can run multiple agents in parallel:

```python
import concurrent.futures
from cybersec_agents.grayswan.agents import ExploitDeliveryAgent

def run_exploit(prompt, target_model):
    agent = ExploitDeliveryAgent()
    return agent.run_prompts([prompt], target_model)

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(run_exploit, prompt, "GPT-4") for prompt in prompts]
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
```
