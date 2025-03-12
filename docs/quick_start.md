# Quick Start Guide

## Installation
```bash
# Install package
pip install -e .

# Set up configuration
cp config/config.yaml.example config/config.yaml
# Edit config.yaml with your API keys
```

## Basic Usage
```bash
# Get help
cyber-agents --help

# Run a command
cyber-agents run "your command here"

# Create a new agent
cyber-agents create-agent "your_agent_name"
```

## Common Tasks
1. Analyze Code:
```bash
cyber-agents run "analyze code quality in src/"
```

2. Improve Prompts:
```bash
cyber-agents run "improve prompt 'explain how this works'"
```

3. Database Tasks:
```bash
cyber-agents run "suggest database optimizations"
```

## Next Steps
1. Read the comprehensive guide
2. Explore available agents
3. Create custom agents
4. Monitor performance