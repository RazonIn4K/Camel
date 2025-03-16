"""
Model type mappings for different agents.
"""

from typing import Any, Dict, Type

from camel.types import ModelType


def get_default_model_type_for_agent(agent_class: Type[Any]) -> ModelType:
    """Get the default model type for a given agent class.

    Args:
        agent_class: The agent class to get the default model type for.

    Returns:
        The default model type for the agent.
    """
    # Import here to avoid circular imports
    from cybersec_agents.grayswan.agents.evaluation_agent import EvaluationAgent
    from cybersec_agents.grayswan.agents.exploit_delivery_agent import ExploitDeliveryAgent
    from cybersec_agents.grayswan.agents.prompt_engineer_agent import PromptEngineerAgent
    from cybersec_agents.grayswan.agents.recon_agent import ReconAgent

    model_type_map = {
        ExploitDeliveryAgent: ModelType.GPT_4,
        ReconAgent: ModelType.GEMINI_PRO,
        EvaluationAgent: ModelType.CLAUDE_3_SONNET,
        PromptEngineerAgent: ModelType.GPT_4_TURBO,
    }

    return model_type_map.get(agent_class, ModelType.GPT_4) 