# Creating Custom Agents

1. Create a new agent class in cybersec_agents/:
```python
from camel.agents import ChatAgent
from camel.models import ModelFactory

class CustomSecurityAgent:
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4
        )
        
        self.agent = ChatAgent(
            system_message=self._get_system_message(),
            model=self.model
        )

    def _get_system_message(self) -> str:
        return """Define agent behavior here"""
```

2. Update cybersec_agents/__init__.py:
```python
from .custom_agent import CustomSecurityAgent

__all__ += ['CustomSecurityAgent']
```