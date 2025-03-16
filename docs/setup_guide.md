# Setup Guide

## Environment Setup
1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package with dependencies:
```bash
pip install -e ".[dev]"
```

## Configuration
1. Create config/config.yaml:
```yaml
api:
  openai:
    api_key: "your-api-key"
    model: "gpt-4-turbo"
    temperature: 0.7
    max_tokens: 4000
```

2. Set required environment variables:
```bash
export OPENAI_API_KEY=your_key
export CAMEL_AI_API_KEY=your_key
export LOG_LEVEL=INFO
```