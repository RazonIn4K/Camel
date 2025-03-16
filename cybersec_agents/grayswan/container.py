"""
Dependency Injection Container for Gray Swan Arena.

This module provides a dependency injection container for the Gray Swan Arena framework,
making it easier to manage dependencies, improve testability, and enhance flexibility.
"""

from typing import Any, Dict, Optional, Type

from dependency_injector import containers, providers

from .agents.evaluation_agent import EvaluationAgent
from .agents.exploit_delivery_agent import ExploitDeliveryAgent
from .agents.prompt_engineer_agent import PromptEngineerAgent
from .agents.recon_agent import ReconAgent
from .utils.advanced_visualization_utils import create_advanced_evaluation_report
from .utils.browser_utils import BrowserAutomationFactory
from .utils.enhanced_browser_utils import EnhancedBrowserAutomationFactory
from .utils.logging_utils import setup_logging
from .utils.model_manager_di import ModelManager, ModelManagerProvider
from .utils.visualization_utils import create_evaluation_report
from .camel_integration import AgentFactory
from .utils.model_config_manager import ModelConfigManager
from camel.types import ModelType, ModelPlatformType


class GraySwanContainer(containers.DeclarativeContainer):
    """
    Dependency Injection Container for Gray Swan Arena.

    This container provides all the dependencies needed by the Gray Swan Arena framework,
    including agents, utilities, and configuration.
    """

    # Configuration provider
    config = providers.Configuration()


    # Logging provider
    logger = providers.Singleton(
        setup_logging,
        name="grayswan",
        level=config.logging.level,
        log_file=config.logging.file,
    )

    # Browser automation providers
    browser_factory = providers.Factory(
        BrowserAutomationFactory.create_driver,
        method=config.browser.method,
        headless=config.browser.headless,
    )

    enhanced_browser_factory = providers.Factory(
        EnhancedBrowserAutomationFactory.create_driver,
        method=config.browser.method,
        headless=config.browser.headless,
        retry_attempts=config.browser.retry_attempts,
        retry_delay=config.browser.retry_delay,
    )

    # Model configuration manager
    model_config_manager = providers.Singleton(
        ModelConfigManager,
        config=config.model,
    )

    # Agent factory
    agent_factory = providers.Singleton(
        AgentFactory,
        model_manager=model_config_manager,
    )

    # Agent providers using factory
    recon_agent = providers.Factory(
        agent_factory.create_agent,
        agent_type="recon",
        output_dir=config.agents.recon.output_dir,
        model_type=config.agents.recon.model_type,
        model_platform=config.agents.recon.model_platform,
        backup_model_type=config.agents.recon.backup_model_type,
        backup_model_platform=config.agents.recon.backup_model_platform,
    )

    prompt_engineer_agent = providers.Factory(
        agent_factory.create_agent,
        agent_type="prompt_engineer",
        output_dir=config.agents.prompt_engineer.output_dir,
        model_type=config.agents.prompt_engineer.model_type,
        model_platform=config.agents.prompt_engineer.model_platform,
        backup_model_type=config.agents.prompt_engineer.backup_model_type,
        backup_model_platform=config.agents.prompt_engineer.backup_model_platform,
        reasoning_model_type=config.agents.prompt_engineer.reasoning_model_type,
        reasoning_model_platform=config.agents.prompt_engineer.reasoning_model_platform,
    )

    exploit_delivery_agent = providers.Factory(
        agent_factory.create_agent,
        agent_type="exploit_delivery",
        output_dir=config.agents.exploit_delivery.output_dir,
        model_type=config.agents.exploit_delivery.model_type,
        model_platform=config.agents.exploit_delivery.model_platform,
        backup_model_type=config.agents.exploit_delivery.backup_model_type,
        backup_model_platform=config.agents.exploit_delivery.backup_model_platform,
        browser_method=config.agents.exploit_delivery.browser_method,
        headless=config.agents.exploit_delivery.headless,
    )

    evaluation_agent = providers.Factory(
        agent_factory.create_agent,
        agent_type="evaluation",
        output_dir=config.agents.evaluation.output_dir,
        model_type=config.agents.evaluation.model_type,
        model_platform=config.agents.evaluation.model_platform,
        backup_model_type=config.agents.evaluation.backup_model_type,
        backup_model_platform=config.agents.evaluation.backup_model_platform,
        reasoning_model_type=config.agents.evaluation.reasoning_model_type,
        reasoning_model_platform=config.agents.evaluation.reasoning_model_platform,
    )
# Visualization providers
    # Visualization providers
    create_evaluation_report = providers.Callable(
        create_evaluation_report,
    )

    create_advanced_evaluation_report = providers.Callable(
        create_advanced_evaluation_report,
    )


class GraySwanContainerFactory:
    """Factory for creating and configuring GraySwanContainer instances."""

    @staticmethod
    def create_container(config_dict: Optional[Dict[str, Any]] = None) -> GraySwanContainer:
        """
        Create and configure a GraySwanContainer.

        Args:
            config_dict: Optional configuration dictionary to override defaults

        Returns:
            Configured GraySwanContainer instance
        """
        container = GraySwanContainer()
        container.config.from_dict(config_dict or {})
        return container

    @staticmethod
    def create_container_from_file(config_file: str) -> GraySwanContainer:
        """
        Create and configure a GraySwanContainer from a configuration file.

        Args:
            config_file: Path to configuration file (YAML or JSON)

        Returns:
            Configured GraySwanContainer instance
        """
        container = GraySwanContainer()
        container.config.from_yaml(config_file)
        return container
