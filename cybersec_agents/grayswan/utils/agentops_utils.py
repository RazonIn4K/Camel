"""Utility functions for AgentOps integration."""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_agentops_initialized = False


def initialize_agentops(api_key: Optional[str] = None) -> bool:
    """Initialize AgentOps with the provided API key.

    Args:
        api_key: The AgentOps API key (optional, will use environment variable if not provided)

    Returns:
        True if initialization is successful, False otherwise
    """
    global _agentops_initialized
    if _agentops_initialized:
        logger.debug("AgentOps already initialized")
        return True

    # If api_key not provided, try to get from environment
    if not api_key:
        api_key = os.getenv("AGENTOPS_API_KEY")
        if not api_key:
            logger.warning("No AgentOps API key found, tracking will be disabled")
            return False

    try:
        import agentops

        logger.info("Initializing AgentOps...")
        agentops.init(api_key=api_key)
        _agentops_initialized = True
        logger.info("AgentOps initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize AgentOps: {str(e)}")
        return False


def start_agentops_session(
    agent_type: str = "generic", 
    model: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> bool:
    """Start a new AgentOps session.

    Args:
        agent_type: Type of agent (e.g., "ReconAgent", "PromptEngineerAgent")
        model: Name of the primary model being used
        tags: List of additional tags

    Returns:
        True if session started successfully, False otherwise
    """
    if not _agentops_initialized:
        logger.debug("AgentOps not initialized, can't start session")
        return False

    try:
        import agentops

        # Create a list of all tags
        all_tags = [agent_type.lower()]
        if model:
            all_tags.append(f"model:{model}")
        if tags:
            all_tags.extend(tags)

        agentops.start_session(tags=all_tags)
        logger.debug(f"Started AgentOps session with tags: {all_tags}")
        return True
    except Exception as e:
        logger.warning(f"Failed to start AgentOps session: {str(e)}")
        return False


def log_agentops_event(event_name: str, properties: Optional[Dict[str, Any]] = None) -> bool:
    """Log an event to AgentOps.

    Args:
        event_name: Name of the event
        properties: Properties of the event

    Returns:
        True if event logged successfully, False otherwise
    """
    if not _agentops_initialized:
        logger.debug("AgentOps not initialized, can't log event")
        return False

    try:
        import agentops

        agentops.record_event(event_name, properties or {})
        logger.debug(f"Logged AgentOps event: {event_name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to log AgentOps event: {str(e)}")
        return False


def end_agentops_session() -> bool:
    """End the current AgentOps session.

    Returns:
        True if session ending is successful, False otherwise
    """
    try:
        import agentops

        logger.info("Ending AgentOps session")
        agentops.end_session()
        logger.info("AgentOps session ended successfully")
        return True
    except ImportError:
        logger.debug("AgentOps not installed. Nothing to end.")
        return False
    except Exception as e:
        logger.warning(f"Failed to end AgentOps session: {str(e)}")
        return False


def track_llm_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Track LLM usage with AgentOps.

    Args:
        model: The model name
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        success: Whether the request was successful
        metadata: Additional metadata for the request

    Returns:
        True if tracking is successful, False otherwise
    """
    try:
        import agentops

        event_data = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "success": success,
        }

        if metadata:
            event_data.update(metadata)

        agentops.log_event("llm_request", event_data)
        return True
    except (ImportError, Exception):
        return False
