# Refactor Plan: `cybersec_agents/grayswan/camel_integration.py`

This document outlines the plan to refactor `cybersec_agents/grayswan/camel_integration.py` to use `model_type` and `model_platform` instead of `model_name`.

## Changes

1.  **Modify `AgentFactory.create_agent`:**
    *   Replace all instances of `model_name` with `model_type` and `model_platform`.
    *   Update the calls to `self.model_manager.get_model_config` to pass `model_type` and `model_platform` as arguments, instead of `model_name`.
    *   For any logic that depends on specific model names (e.g., the conditional blocks for `prompt_engineer`, `evaluation`, `recon`, and `exploit_delivery`), update the logic to use the appropriate `ModelType` and `ModelPlatformType` enums. This will involve retrieving `model_type` and `model_platform` from `kwargs`, providing reasonable defaults (like `ModelType.GPT_4_TURBO` and `ModelPlatformType.OPENAI` for the `prompt_engineer`), and then using these in the call to `get_model_config`.
    *   Update the AgentOps event data to include `model_type` and `model_platform` instead of `model_name`.

**Example (Conceptual - before refactoring):**

```python
# Before
if agent_type == "prompt_engineer":
    model_name = kwargs.get("model_name", "gpt-4o")
    model_config = self.model_manager.get_model_config(model_name)
    # ...
    kwargs["model_name"] = model_name
```

**Example (Conceptual - after refactoring):**

```python
# After
from camel.types import ModelType, ModelPlatformType

if agent_type == "prompt_engineer":
    model_type = kwargs.get("model_type", ModelType.GPT_4_TURBO)
    model_platform = kwargs.get("model_platform", ModelPlatformType.OPENAI)
    model_config = self.model_manager.get_model_config(model_type, model_platform)
    # ...
    kwargs["model_type"] = model_type
    kwargs["model_platform"] = model_platform

```

## Mermaid Diagram (Sequence Diagram)

```mermaid
sequenceDiagram
    participant User
    participant AgentFactory
    participant ModelConfigManager

    User->>AgentFactory: create_agent(agent_type, model_type, model_platform)
    activate AgentFactory
    AgentFactory->>ModelConfigManager: get_model_config(model_type, model_platform)
    activate ModelConfigManager
    ModelConfigManager-->>AgentFactory: model_config
    deactivate ModelConfigManager
    AgentFactory->>AgentFactory: create agent instance
    AgentFactory-->>User: agent instance
    deactivate AgentFactory