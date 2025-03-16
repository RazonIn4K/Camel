# Perplexity API Integration Guide

This guide explains how to integrate the Perplexity API with the Gray Swan Arena framework.

## Overview

The Perplexity API provides access to powerful AI models through an OpenAI-compatible interface. This integration allows you to use Perplexity models seamlessly within the Gray Swan Arena framework.

## Prerequisites

1. A Perplexity API key (sign up at [perplexity.ai](https://www.perplexity.ai))
2. Gray Swan Arena framework installed

## Configuration

### 1. Set Up Environment Variable

Set up your Perplexity API key as an environment variable:

```bash
# Linux/macOS
export PERPLEXITY_API_KEY="your_actual_perplexity_api_key"

# Windows (Command Prompt)
set PERPLEXITY_API_KEY="your_actual_perplexity_api_key"

# Windows (PowerShell)
$env:PERPLEXITY_API_KEY = "your_actual_perplexity_api_key"
```

### 2. Update Configuration File

Update your configuration file (e.g., `config/development.yml` or `config/production.yml`) to include the Perplexity model:

```yaml
models:
  - type: SONA_PRO
    platform: PERPLEXITY
    api_key_env: PERPLEXITY_API_KEY
    base_url: "https://api.perplexity.ai"
    timeout: 60
    max_retries: 5
    temperature: 0.8
    max_tokens: 4096

agents:
  my_agent:
    output_dir: "outputs/my_agent"
    model_type: SONA_PRO
    model_platform: PERPLEXITY
    backup_model_type: GPT_4  # Optional backup model
    backup_model_platform: OPENAI  # Optional backup platform
    complexity_threshold: 0.8
```

#### Important Configuration Fields:

- `type`: Use `SONA_PRO` for Perplexity's SONA model
- `platform`: Must be `PERPLEXITY`
- `api_key_env`: Environment variable name containing your API key
- `base_url`: Must be set to `"https://api.perplexity.ai"` for Perplexity

### 3. Available Perplexity Models

When making API calls, use the correct model name in the `model` parameter:

- `sonar-small-online`: A smaller, faster model
- `sonar-medium-online`: A medium-sized model balancing speed and capability
- `sonar-large-online`: The most capable model

## Usage Example

Here's how to use the Perplexity API in your code:

```python
from camel.types import ModelType, ModelPlatformType
from cybersec_agents.grayswan.config import Config
from openai import OpenAI
import os

# Load configuration
config = Config()

# Get Perplexity model configuration
perplexity_config = config.get_model_config(
    model_type=ModelType.SONA_PRO, 
    model_platform=ModelPlatformType.PERPLEXITY
)

# Create OpenAI client with Perplexity configuration
client = OpenAI(
    api_key=os.getenv(perplexity_config.api_key_env),
    base_url=perplexity_config.base_url
)

# Make an API call
response = client.chat.completions.create(
    model="sonar-small-online",  # Perplexity model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the difference between XSS and CSRF?"}
    ],
    temperature=perplexity_config.temperature,
    max_tokens=perplexity_config.max_tokens
)

print(response.choices[0].message.content)
```

## Full Example

See the complete example in `examples/perplexity_example.py`.

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure the `PERPLEXITY_API_KEY` environment variable is set.
2. **Invalid Base URL**: Ensure the `base_url` is set to `"https://api.perplexity.ai"`.
3. **Rate Limiting**: If you encounter rate limit errors, increase the `max_retries` value in your configuration.

### Errors and Solutions

- **Authentication Error**: Verify your API key is correct and properly set in the environment.
- **Connection Error**: Check your internet connection and that the base URL is correct.
- **Validation Error**: Ensure all required configuration fields are properly set. 