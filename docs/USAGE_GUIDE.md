# Gray Swan Arena Usage Guide

## Overview
Gray Swan Arena is a comprehensive framework for testing and evaluating AI model security. This guide provides detailed instructions for setting up and using the framework.

## Setup and Configuration

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gray-swan-arena.git
cd gray-swan-arena
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root with the following API keys:

```env
# OpenAI API Keys
OPENAI_API_KEY=your_openai_api_key
O3_MINI_API_KEY=your_o3_mini_api_key

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google API Key
GOOGLE_API_KEY=your_google_api_key

# Perplexity API Key
PERPLEXITY_API_KEY=your_perplexity_api_key

# Optional: AgentOps API Key for monitoring
AGENTOPS_API_KEY=your_agentops_api_key
```

## Model Configuration

The framework supports the following models:

- OpenAI: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- Anthropic: Claude 3 Sonnet, Claude 3 Opus, Claude 2
- Google: Gemini Pro, Gemini Pro Experimental

Example configuration:

```python
from cybersec_agents.grayswan.agents import ExploitDeliveryAgent

agent = ExploitDeliveryAgent(
    model_name="gpt-4",
    model_type=ModelType.GPT_4,
    model_platform=ModelPlatform.OPENAI,
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## Agent Usage

### Reconnaissance Agent

The Reconnaissance Agent uses the Gemini 2.0 Pro Experimental model for gathering information about target models.

Example usage:
```python
from cybersec_agents.grayswan.agents import ReconAgent
from cybersec_agents.camel.types import ModelType, ModelPlatformType

# Initialize the agent
recon_agent = ReconAgent(
    model_type=ModelType.GEMINI_PRO_EXPERIMENTAL,
    model_platform=ModelPlatformType.GOOGLE
)

# Run reconnaissance
results = recon_agent.run_reconnaissance(target_model="gpt-4")
```

### Prompt Engineer Agent

The Prompt Engineer Agent uses Claude 3 Sonnet for crafting sophisticated prompts.

Example usage:
```python
from cybersec_agents.grayswan.agents import PromptEngineerAgent
from cybersec_agents.camel.types import ModelType, ModelPlatformType

# Initialize the agent
prompt_agent = PromptEngineerAgent(
    model_type=ModelType.CLAUDE_3_SONNET,
    model_platform=ModelPlatformType.ANTHROPIC
)

# Generate prompts
prompts = prompt_agent.generate_prompts(target_behavior="information_leakage")
```

### Exploit Delivery Agent

The Exploit Delivery Agent uses O3-Mini for delivering attack prompts and supports multiple model types.

Example usage:
```python
from cybersec_agents.grayswan.agents import ExploitDeliveryAgent
from cybersec_agents.camel.types import ModelType, ModelPlatformType

# Initialize the agent
exploit_agent = ExploitDeliveryAgent(
    model_type=ModelType.O3_MINI,
    model_platform=ModelPlatformType.OPENAI
)

# Run exploits
results = exploit_agent.run_prompts(
    prompts=["Your prompt here"],
    target_model="gpt-4",
    target_behavior="information_leakage"
)
```

### Evaluation Agent

The Evaluation Agent uses GPT-4o for analyzing model responses.

Example usage:
```python
from cybersec_agents.grayswan.agents import EvaluationAgent
from cybersec_agents.camel.types import ModelType, ModelPlatformType

# Initialize the agent
eval_agent = EvaluationAgent(
    model_type=ModelType.GPT_4_O,
    model_platform=ModelPlatformType.OPENAI
)

# Evaluate results
evaluation = eval_agent.evaluate_results(results)
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   - The framework includes exponential backoff for handling rate limits
   - If you encounter rate limits, the system will automatically retry with increasing delays
   - Monitor the logs for rate limit warnings

2. **Model-Specific Issues**
   - **Gemini**: Ensure your API key has access to the experimental model
   - **Claude**: Check that your API key has access to Claude 3 Sonnet
   - **O3-Mini**: Verify your API key has access to the O3-Mini model

3. **Browser Automation**
   - If using web-based testing, ensure you have the required browser drivers installed
   - For Playwright: `playwright install`
   - For Selenium: The webdriver-manager will handle driver installation

### Logging

The framework uses structured logging for better debugging:

```python
import logging
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
setup_logging()

# Log messages will be written to both console and file
logger = logging.getLogger(__name__)
logger.info("Your message here")
```

## Best Practices

1. **API Key Management**
   - Never commit API keys to version control
   - Use environment variables or secure secret management
   - Rotate API keys regularly

2. **Model Selection**
   - Use the recommended model for each agent type
   - Consider cost implications when selecting models
   - Monitor model performance and adjust parameters as needed

3. **Testing**
   - Start with small test cases before running full evaluations
   - Monitor rate limits and costs
   - Keep track of successful and failed attempts

4. **Security**
   - Follow responsible disclosure practices
   - Document all testing procedures
   - Maintain audit logs of all operations

## Contributing

Please refer to our [Contributing Guidelines](CONTRIBUTING.md) for information on how to contribute to the project. 