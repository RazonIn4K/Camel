"""Utility functions for AgentOps integration."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_agentops_initialized = False


def initialize_agentops(api_key: str) -> bool:
    """Initialize AgentOps with the provided API key.

    Args:
        api_key: The AgentOps API key

    Returns:
        True if initialization is successful, False otherwise
    """
    global _agentops_initialized
    if _agentops_initialized:
        logger.debug("AgentOps already initialized")
        return True

    try:
        import agentops

        logger.info("Initializing AgentOps...")
        agentops.init(api_key=api_key)
        _agentops_initialized = True
        logger.info("AgentOps initialized successfully")
        return True
    except ImportError:
        logger.warning("AgentOps not installed. Install with 'pip install agentops'")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize AgentOps: {str(e)}")
        return False


def start_agentops_session(
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    inherited_session_id: Optional[str] = None,
) -> bool:
    """Start an AgentOps session.

    Args:
        session_id: Optional session ID
        tags: Optional list of tags for the session
        inherited_session_id: Optional ID of a parent session to inherit from

    Returns:
        True if session start is successful, False otherwise
    """
    try:
        import agentops

        logger.info(f"Starting AgentOps session (ID: {session_id}, Tags: {tags})")

        # Start the session with the appropriate parameters based on the AgentOps API
        agentops.start_session(tags=tags, inherited_session_id=inherited_session_id)

        logger.info("AgentOps session started successfully")
        return True
    except ImportError:
        logger.warning("AgentOps not installed. Install with 'pip install agentops'")
        return False
    except Exception as e:
        logger.warning(f"Failed to start AgentOps session: {str(e)}")
        return False


def log_agentops_event(
    event_name: str, event_data: Optional[Dict[str, Any]] = None
) -> bool:
    """Log an event to AgentOps.

    Args:
        event_name: The name of the event
        event_data: Optional data for the event

    Returns:
        True if event logging is successful, False otherwise
    """
    try:
        import agentops

        agentops.log_event(event_name, event_data or {})
        logger.debug(f"Logged AgentOps event: {event_name}")
        return True
    except ImportError:
        # Don't log a warning for every event, just silently fail
        return False
    except Exception as e:
        logger.debug(f"Failed to log AgentOps event: {str(e)}")
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
