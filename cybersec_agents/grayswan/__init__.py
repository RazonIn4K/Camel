from typing import Any, Dict, List, Optional, Tuple, Union

"""
Gray Swan Arena - A framework for AI security testing and evaluation.

This module provides a comprehensive framework for testing and evaluating AI models
against various security threats and vulnerabilities.
"""

__version__: str = "0.2.0"

from .agents.evaluation_agent import EvaluationAgent
from .agents.exploit_delivery_agent import ExploitDeliveryAgent
from .agents.prompt_engineer_agent import PromptEngineerAgent

# Import main components
from .agents.recon_agent import ReconAgent
from .camel_integration import CommunicationChannel

# Make main function available at package level
from .main import main

__all__: list[Any] = [
    "ReconAgent",
    "PromptEngineerAgent",
    "ExploitDeliveryAgent",
    "EvaluationAgent",
    "main",
    "CommunicationChannel",
]
