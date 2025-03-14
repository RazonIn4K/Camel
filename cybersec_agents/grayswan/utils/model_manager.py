"""
Model Manager for Gray Swan Arena.

This module provides a robust model management system with fallback capabilities,
exponential backoff for API rate limits, and model switching based on complexity.
"""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import ModelFactory from the provided code
from cybersec_agents.grayswan.utils.model_utils import (
    create_model,
    get_default_model_type_for_agent,
)

# Set up logging
logger = logging.getLogger(__name__)


class ModelBackupError(Exception):
    """Exception raised when both primary and backup models fail."""

    def __init__(
        self,
        message: str,
        primary_model: str,
        backup_model: str,
        operation: str = "unknown",
    ):
        self.primary_model = primary_model
        self.backup_model = backup_model
        self.operation = operation
        super().__init__(message)


class RateLimitError(Exception):
    """Exception raised when API rate limits are hit."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(message)


class APIError(Exception):
    """Exception raised for general API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


def with_exponential_backoff(
    func: Callable,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[Exception, ...] = (RateLimitError, APIError),
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        func: The function to decorate
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add randomness to delay
        retry_on: Exceptions to retry on

    Returns:
        Decorated function
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None

        for retry in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except retry_on as e:
                last_exception = e

                # Check if we should retry
                if retry == max_retries - 1:
                    break

                # Get retry delay (use retry_after if available for RateLimitError)
                if isinstance(e, RateLimitError) and e.retry_after is not None:
                    actual_delay = e.retry_after
                else:
                    actual_delay = delay
                    if jitter:
                        actual_delay = delay * (0.75 + random.random() * 0.5)

                logger.warning(
                    f"Attempt {retry + 1}/{max_retries} failed: {str(e)}. "
                    f"Retrying in {actual_delay:.2f} seconds..."
                )

                # Wait before retrying
                await asyncio.sleep(actual_delay)

                # Increase delay for next retry
                delay *= backoff_factor

        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"All {max_retries} attempts failed")

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None

        for retry in range(max_retries):
            try:
                return func(*args, **kwargs)
            except retry_on as e:
                last_exception = e

                # Check if we should retry
                if retry == max_retries - 1:
                    break

                # Get retry delay (use retry_after if available for RateLimitError)
                if isinstance(e, RateLimitError) and e.retry_after is not None:
                    actual_delay = e.retry_after
                else:
                    actual_delay = delay
                    if jitter:
                        actual_delay = delay * (0.75 + random.random() * 0.5)

                logger.warning(
                    f"Attempt {retry + 1}/{max_retries} failed: {str(e)}. "
                    f"Retrying in {actual_delay:.2f} seconds..."
                )

                # Wait before retrying
                time.sleep(actual_delay)

                # Increase delay for next retry
                delay *= backoff_factor

        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"All {max_retries} attempts failed")

    # Check if the function is async
    import inspect

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class ModelManager:
    """
    Manages model interactions with fallback capabilities.

    This class provides a robust interface for interacting with language models,
    with support for fallback to backup models, exponential backoff for rate limits,
    and complexity-based model selection.
    """

    def __init__(
        self,
        primary_model: str,
        backup_model: Optional[str] = None,
        complexity_threshold: float = 0.7,
    ):
        """
        Initialize the ModelManager.

        Args:
            primary_model: Primary model to use
            backup_model: Backup model to use if primary fails
            complexity_threshold: Threshold for using backup model (0.0-1.0)
        """
        self.primary_model = primary_model
        self.backup_model = backup_model
        self.complexity_threshold = complexity_threshold
        self.metrics = {"primary_calls": 0, "backup_calls": 0, "failures": 0}

        # Initialize models
        self.models = {}
        self._initialize_models()

        logger.info(
            f"ModelManager initialized with primary={primary_model}, "
            f"backup={backup_model}, threshold={complexity_threshold}"
        )

    def _initialize_models(self):
        """Initialize models based on available configurations."""
        # Create model for primary
        self._ensure_model(self.primary_model)

        # Create model for backup if available
        if self.backup_model:
            self._ensure_model(self.backup_model)

    def _ensure_model(self, model_name: str):
        """
        Ensure a model is available.

        Args:
            model_name: Name of the model to ensure
        """
        if model_name not in self.models:
            try:
                # Determine model type and platform
                model_type = get_default_model_type_for_agent("default")

                # Create the model
                model = create_model(
                    model_type=model_type, model_config_dict={"model_name": model_name}
                )

                if model:
                    self.models[model_name] = model
                    logger.info(f"Model {model_name} initialized successfully")
                else:
                    logger.error(f"Failed to initialize model {model_name}")
            except Exception as e:
                logger.error(f"Error initializing model {model_name}: {str(e)}")

    def get_backup_model(self, model_name: str) -> Optional[str]:
        """
        Get an appropriate backup model for a given model.

        Args:
            model_name: The primary model name

        Returns:
            Name of an appropriate backup model or None
        """
        # Define backup pairs
        backup_pairs = {
            "gpt-4": "gpt-3.5-turbo",
            "gpt-4-turbo": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo-instruct",
            "claude-2": "claude-instant-1",
            "claude-3-opus": "claude-3-sonnet",
            "claude-3-sonnet": "claude-3-haiku",
        }

        return backup_pairs.get(model_name)

    @with_exponential_backoff
    async def generate_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        complexity: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a completion with fallback capability (async version).

        Args:
            prompt: The prompt to send
            model: Specific model to use (overrides primary)
            complexity: Complexity score of the prompt (0.0-1.0)
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dictionary containing the model's response
        """
        try:
            # Determine which model to use
            model_to_use = model or self.primary_model

            # Use backup for complex prompts if available
            if (
                complexity is not None
                and complexity >= self.complexity_threshold
                and self.backup_model
                and not model  # Don't override explicit model choice
            ):
                model_to_use = self.backup_model
                self.metrics["backup_calls"] += 1
                logger.info(
                    f"Using backup model {model_to_use} due to complexity {complexity}"
                )
            else:
                self.metrics["primary_calls"] += 1

            # Ensure model is available
            self._ensure_model(model_to_use)

            # Get the model
            model_obj = self.models.get(model_to_use)
            if not model_obj:
                raise ValueError(f"Model {model_to_use} not available")

            # Generate the completion
            response = await model_obj.generate(prompt, **kwargs)
            return response

        except Exception as e:
            # Try backup if primary fails and we weren't already using it
            if (
                model_to_use != self.backup_model
                and self.backup_model
                and not model  # Don't override explicit model choice
            ):
                try:
                    logger.warning(
                        f"Primary model {model_to_use} failed: {str(e)}. "
                        f"Falling back to {self.backup_model}"
                    )

                    self.metrics["backup_calls"] += 1

                    # Ensure backup model is available
                    self._ensure_model(self.backup_model)

                    # Get the backup model
                    backup_model_obj = self.models.get(self.backup_model)
                    if not backup_model_obj:
                        raise ValueError(
                            f"Backup model {self.backup_model} not available"
                        )

                    # Generate with backup
                    response = await backup_model_obj.generate(prompt, **kwargs)
                    return response

                except Exception as backup_e:
                    self.metrics["failures"] += 1
                    raise ModelBackupError(
                        f"Both models failed: {str(e)} and {str(backup_e)}",
                        model_to_use,
                        self.backup_model,
                        "generate",
                    )

            # Re-raise if no backup or explicit model choice
            self.metrics["failures"] += 1
            raise

    @with_exponential_backoff
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        complexity: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a completion with fallback capability (sync version).

        Args:
            prompt: The prompt to send
            model: Specific model to use (overrides primary)
            complexity: Complexity score of the prompt (0.0-1.0)
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dictionary containing the model's response
        """
        try:
            # Determine which model to use
            model_to_use = model or self.primary_model

            # Use backup for complex prompts if available
            if (
                complexity is not None
                and complexity >= self.complexity_threshold
                and self.backup_model
                and not model  # Don't override explicit model choice
            ):
                model_to_use = self.backup_model
                self.metrics["backup_calls"] += 1
                logger.info(
                    f"Using backup model {model_to_use} due to complexity {complexity}"
                )
            else:
                self.metrics["primary_calls"] += 1

            # Ensure model is available
            self._ensure_model(model_to_use)

            # Get the model
            model_obj = self.models.get(model_to_use)
            if not model_obj:
                raise ValueError(f"Model {model_to_use} not available")

            # Generate the completion
            response = model_obj.generate(prompt, **kwargs)
            return response

        except Exception as e:
            # Try backup if primary fails and we weren't already using it
            if (
                model_to_use != self.backup_model
                and self.backup_model
                and not model  # Don't override explicit model choice
            ):
                try:
                    logger.warning(
                        f"Primary model {model_to_use} failed: {str(e)}. "
                        f"Falling back to {self.backup_model}"
                    )

                    self.metrics["backup_calls"] += 1

                    # Ensure backup model is available
                    self._ensure_model(self.backup_model)

                    # Get the backup model
                    backup_model_obj = self.models.get(self.backup_model)
                    if not backup_model_obj:
                        raise ValueError(
                            f"Backup model {self.backup_model} not available"
                        )

                    # Generate with backup
                    response = backup_model_obj.generate(prompt, **kwargs)
                    return response

                except Exception as backup_e:
                    self.metrics["failures"] += 1
                    raise ModelBackupError(
                        f"Both models failed: {str(e)} and {str(backup_e)}",
                        model_to_use,
                        self.backup_model,
                        "generate",
                    )

            # Re-raise if no backup or explicit model choice
            self.metrics["failures"] += 1
            raise

    def estimate_complexity(self, prompt: str) -> float:
        """
        Estimate the complexity of a prompt.

        Args:
            prompt: The prompt to analyze

        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Simple complexity estimation based on length and special tokens
        length_score = min(len(prompt) / 4000, 1.0) * 0.7

        # Check for complex instructions
        complex_indicators = [
            "step by step",
            "explain in detail",
            "analyze",
            "compare and contrast",
            "evaluate",
            "synthesize",
            "create a comprehensive",
            "write code",
            "implement",
            "debug",
            "optimize",
        ]

        indicator_score = 0.0
        for indicator in complex_indicators:
            if indicator in prompt.lower():
                indicator_score += 0.1

        indicator_score = min(indicator_score, 0.3)

        # Combine scores
        complexity = length_score + indicator_score

        return min(complexity, 1.0)

    def get_metrics(self) -> Dict[str, int]:
        """
        Get usage metrics.

        Returns:
            Dictionary of usage metrics
        """
        return self.metrics.copy()
