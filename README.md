# Cybersecurity Agents

Specialized AI agents built on the Camel AI framework for cybersecurity tasks, particularly within the Gray Swan Arena for AI safety and security testing.

## Table of Contents

- [Setup](#setup)
- [Gray Swan Arena Agents](#gray-swan-arena-agents)
  - [ReconAgent](#reconagent)
  - [PromptEngineerAgent](#promptengineeragent)
  - [ExploitDeliveryAgent](#exploitdeliveryagent)
  - [EvaluationAgent](#evaluationagent)
- [Model Configuration](#model-configuration)
- [Advanced Usage](#advanced-usage)

## Setup

### Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key
- `AGENTOPS_API_KEY`: Your AgentOps API key (optional)
- `LOG_LEVEL`: Logging level (default: INFO)
- `MAX_TOKENS`: Maximum tokens for model responses (default: 1000)
- `TEMPERATURE`: Temperature for model responses (default: 0.7)
- `O3_MINI_API_KEY`: Your API key for o3-mini model (if using o3-mini)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/cybersec-agents.git
cd cybersec-agents
pip install -e .
```

## Gray Swan Arena Agents

The Gray Swan Arena provides a set of specialized agents for testing AI safety and security.

### ReconAgent

The Reconnaissance Agent gathers information about target AI models, including their architecture, vulnerabilities, and community knowledge.

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

# Run web search
web_results = agent.run_web_search(
    target_model="GPT-4",
    target_behavior="bypass content filters"
)

# Generate report
report = agent.generate_report(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    web_results=web_results
)
```

### PromptEngineerAgent

The Prompt Engineer Agent generates effective prompts to test target models based on reconnaissance data.

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

# Generate prompts
prompts = agent.generate_prompts(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    recon_report=report
)
```

### ExploitDeliveryAgent

The Exploit Delivery Agent is responsible for delivering prompts to target models and recording responses.

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

# Run prompts
results = agent.run_prompts(
    prompts=prompts,
    target_model="GPT-4",
    method="api"
)
```

### EvaluationAgent

The Evaluation Agent analyzes exploit results, generates visualizations, and produces evaluation reports.

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

# Evaluate results
evaluation = agent.evaluate_results(
    results=results,
    target_model="GPT-4",
    target_behavior="bypass content filters"
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

For more detailed usage instructions, see the [Agent Usage Guide](docs/agent_usage_guide.md).
