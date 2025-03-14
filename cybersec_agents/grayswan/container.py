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


class GraySwanContainer(containers.DeclarativeContainer):
    """
    Dependency Injection Container for Gray Swan Arena.

    This container provides all the dependencies needed by the Gray Swan Arena framework,
    including agents, utilities, and configuration.
    """

    # Configuration provider
    config: dict[str, Any] = providers.Configuration()

    # Set default configuration values
    config.set_default_values(
        {
            "output_dir": "./output",
            "agents": {
                "recon": {
                    "output_dir": "./output/recon_reports",
                    "model_name": "gpt-4",
                    "backup_model": "gpt-3.5-turbo",
                    "complexity_threshold": 0.7,
                },
                "prompt_engineer": {
                    "output_dir": "./output/prompt_lists",
                    "model_name": "gpt-4",
                    "backup_model": "gpt-3.5-turbo",
                    "complexity_threshold": 0.7,
                },
                "exploit_delivery": {
                    "output_dir": "./output/exploit_logs",
                    "model_name": "gpt-4",
                    "backup_model": "gpt-3.5-turbo",
                    "complexity_threshold": 0.7,
                    "browser_method": "playwright",
                    "headless": True,
                },
                "evaluation": {
                    "output_dir": "./output/evaluation_reports",
                    "model_name": "gpt-4",
                    "backup_model": "gpt-3.5-turbo",
                    "complexity_threshold": 0.7,
                },
            },
            "model": {
                "primary_model": "gpt-4",
                "backup_model": "gpt-3.5-turbo",
                "complexity_threshold": 0.7,
                "max_retries": 5,
                "initial_delay": 1.0,
                "backoff_factor": 2.0,
                "jitter": True,
            },
            "browser": {
                "method": "playwright",
                "headless": True,
                "timeout": 60000,
                "enhanced": True,
                "retry_attempts": 3,
                "retry_delay": 1.0,
            },
            "visualization": {
                "output_dir": "./output/visualizations",
                "dpi": 300,
                "theme": "default",
                "advanced": True,
                "interactive": True,
                "clustering_clusters": 4,
                "similarity_threshold": 0.5,
            },
            "discord": {
                "bot_token": None,
                "channel_ids": [],
                "timeout": 30,
            },
            "logging": {
                "level": "INFO",
                "file": "./output/logs/grayswan.log",
            },
        }
    )

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

    # Model manager providers
    model_manager = providers.Factory(
        ModelManager,
        primary_model=config.model.primary_model,
        backup_model=config.model.backup_model,
        complexity_threshold=config.model.complexity_threshold,
    )

    recon_model_manager = providers.Factory(
        ModelManagerProvider.create_for_agent,
        agent_type="recon",
    )

    prompt_engineer_model_manager = providers.Factory(
        ModelManagerProvider.create_for_agent,
        agent_type="prompt_engineer",
    )

    exploit_delivery_model_manager = providers.Factory(
        ModelManagerProvider.create_for_agent,
        agent_type="exploit_delivery",
    )

    evaluation_model_manager = providers.Factory(
        ModelManagerProvider.create_for_agent,
        agent_type="evaluation",
    )

    # Agent providers
    recon_agent = providers.Factory(
        ReconAgent,
        output_dir=config.agents.recon.output_dir,
        model_name=config.agents.recon.model_name,
    )

    prompt_engineer_agent = providers.Factory(
        PromptEngineerAgent,
        output_dir=config.agents.prompt_engineer.output_dir,
        model_name=config.agents.prompt_engineer.model_name,
    )

    exploit_delivery_agent = providers.Factory(
        ExploitDeliveryAgent,
        output_dir=config.agents.exploit_delivery.output_dir,
        model_name=config.agents.exploit_delivery.model_name,
        browser_method=config.agents.exploit_delivery.browser_method,
        headless=config.agents.exploit_delivery.headless,
    )

    evaluation_agent = providers.Factory(
        EvaluationAgent,
        output_dir=config.agents.evaluation.output_dir,
        model_name=config.agents.evaluation.model_name,
    )

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
    def create_container(
        config_dict: Optional[Dict[str, Any]] = None
    ) -> GraySwanContainer:
        """
        Create and configure a GraySwanContainer.

        Args:
            config_dict: Optional configuration dictionary to override defaults

        Returns:
            Configured GraySwanContainer instance
        """
        container = GraySwanContainer()

        if config_dict:
            container.config.from_dict(config_dict)

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

        # Determine file type and load accordingly
        if config_file.endswith(".yaml") or config_file.endswith(".yml"):
            container.config.from_yaml(config_file)
        elif config_file.endswith(".json"):
            container.config.from_json(config_file)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file}")

        return container
