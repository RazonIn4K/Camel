"""
Type definitions for the CAMEL framework.

This module contains various type definitions used throughout the CAMEL framework,
including model types, platform types, and other common types.
"""

from enum import Enum, auto


class ModelType(Enum):
    """Enumeration of supported model types."""

    GPT_4 = auto()
    GPT_4_TURBO = auto()
    GPT_3_5_TURBO = auto()
    CLAUDE_2 = auto()
    CLAUDE_3_SONNET = auto()
    CLAUDE_3_OPUS = auto()
    MISTRAL_7B = auto()
    MISTRAL_8X7B = auto()
    GEMINI_PRO = auto()
    GEMINI_PRO_EXPERIMENTAL = auto()
    SONA_PRO = auto()
    O3_MINI = auto()
    GEMINI_2_PRO = auto()  # Add Gemini
    CLAUDE_3_7_SONNET = auto()  # Add Claude
    GPT_4O = auto()  # Add GPT-4o


class ModelPlatformType(Enum):
    """Enumeration of supported model platforms."""

    OPENAI = auto()
    ANTHROPIC = auto()
    HUGGINGFACE = auto()
    GOOGLE = auto()
    LOCAL = auto() 