# Quick Start Guide

## Basic Usage

### CLI Interface
```bash
# Analyze code
cyber-agents run "analyze code quality in src/"

# Improve prompts
cyber-agents run "improve prompt 'explain how this works'"

# Security analysis
cyber-agents run "suggest database optimizations"
```

### Python API
```python
from cybersec_agents import DocumentAnalysisAgent

# Initialize agent
doc_agent = DocumentAnalysisAgent()

# Analyze code
analysis = doc_agent.analyze_code("src/main.py")
```

## Available Agents

1. Document Analysis Agent
2. Network Security Agent
3. Forensics Planner
4. Wireless Security Assessor
5. Code Assistant Agent

For detailed usage, see the [User Guide](user_guide.md).