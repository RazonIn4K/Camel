# Camel AI Cybersecurity Documentation

## Table of Contents

1. [Quick Start Guide](#quick-start)
2. [Architecture Overview](#architecture)
3. [User Guide](#user-guide)
4. [Best Practices & Troubleshooting](#best-practices)
5. [API Reference](#api-reference)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify Camel AI installation:
```python
from camel.agents import ChatAgent
from camel.messages import BaseMessage

# Test Camel AI installation
agent = ChatAgent(BaseMessage(
    role_name="Assistant",
    content="You are a cybersecurity expert."
))
```

## Environment Setup

Only OpenAI API key is required for the language model:
```bash
export OPENAI_API_KEY=your_openai_api_key
export LOG_LEVEL=INFO  # Optional
```

## Quick Usage

```python
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelType

# Initialize agent with local Camel AI
model = ModelFactory.create(model_type=ModelType.GPT_4)
agent = ChatAgent(
    system_message="You are a cybersecurity expert.",
    model=model
)

# Run analysis
response = agent.chat("Analyze this network configuration...")
```
