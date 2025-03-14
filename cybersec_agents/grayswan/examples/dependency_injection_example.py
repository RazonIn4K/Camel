"""
Example script demonstrating the dependency injection pattern in Gray Swan Arena.

This script shows how to use the dependency injection container to create and
configure the Gray Swan Arena pipeline.
"""

import json
import os
import sys
from typing import Any, Dict

import yaml

# Add the parent directory to the path so we can import the modules
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from cybersec_agents.grayswan.container import GraySwanContainerFactory
from cybersec_agents.grayswan.main_di import GraySwanPipeline


def basic_example():
    """
    Basic example of using the dependency injection container with default configuration.
    """
    print("\n=== Basic Example ===\n")

    # Create container with default configuration
    container = GraySwanContainerFactory.create_container()

    # Create pipeline
    pipeline = GraySwanPipeline(container)

    # Get logger from container
    logger = container.logger()
    logger.info("Running basic example with default configuration")

    # Get agents from container
    recon_agent = container.recon_agent()
    prompt_agent = container.prompt_engineer_agent()
    exploit_agent = container.exploit_delivery_agent()
    eval_agent = container.evaluation_agent()

    # Print agent configurations
    print(
        f"Reconnaissance Agent: model={recon_agent.model_name}, output_dir={recon_agent.output_dir}"
    )
    print(
        f"Prompt Engineer Agent: model={prompt_agent.model_name}, output_dir={prompt_agent.output_dir}"
    )
    print(
        f"Exploit Delivery Agent: model={exploit_agent.model_name}, output_dir={exploit_agent.output_dir}"
    )
    print(
        f"Evaluation Agent: model={eval_agent.model_name}, output_dir={eval_agent.output_dir}"
    )


def custom_config_example():
    """
    Example of using the dependency injection container with custom configuration.
    """
    print("\n=== Custom Configuration Example ===\n")

    # Create custom configuration
    config_dict: dict[str, Any] = {
        "output_dir": "./custom_output",
        "agents": {
            "recon": {
                "output_dir": "./custom_output/recon_reports",
                "model_name": "gpt-3.5-turbo",
            },
            "prompt_engineer": {
                "output_dir": "./custom_output/prompt_lists",
                "model_name": "gpt-3.5-turbo",
            },
            "exploit_delivery": {
                "output_dir": "./custom_output/exploit_logs",
                "model_name": "gpt-3.5-turbo",
                "browser_method": "playwright",
                "headless": False,
            },
            "evaluation": {
                "output_dir": "./custom_output/evaluation_reports",
                "model_name": "gpt-3.5-turbo",
            },
        },
        "browser": {
            "method": "playwright",
            "headless": False,
            "timeout": 30000,
            "enhanced": True,
            "retry_attempts": 5,
            "retry_delay": 2.0,
        },
        "visualization": {
            "output_dir": "./custom_output/visualizations",
            "dpi": 150,
            "theme": "dark",
            "advanced": True,
            "interactive": True,
            "clustering_clusters": 5,
            "similarity_threshold": 0.7,
        },
    }

    # Create container with custom configuration
    container = GraySwanContainerFactory.create_container(config_dict)

    # Create pipeline
    pipeline = GraySwanPipeline(container)

    # Get logger from container
    logger = container.logger()
    logger.info("Running example with custom configuration")

    # Get agents from container
    recon_agent = container.recon_agent()
    prompt_agent = container.prompt_engineer_agent()
    exploit_agent = container.exploit_delivery_agent()
    eval_agent = container.evaluation_agent()

    # Print agent configurations
    print(
        f"Reconnaissance Agent: model={recon_agent.model_name}, output_dir={recon_agent.output_dir}"
    )
    print(
        f"Prompt Engineer Agent: model={prompt_agent.model_name}, output_dir={prompt_agent.output_dir}"
    )
    print(
        f"Exploit Delivery Agent: model={exploit_agent.model_name}, output_dir={exploit_agent.output_dir}"
    )
    print(
        f"Evaluation Agent: model={eval_agent.model_name}, output_dir={eval_agent.output_dir}"
    )

    # Print browser configuration
    browser_driver = container.enhanced_browser_factory()
    print(f"\nBrowser Configuration:")
    print(f"  Method: {browser_driver.method}")
    print(f"  Headless: {browser_driver.headless}")
    print(f"  Retry Attempts: {browser_driver.retry_attempts}")
    print(f"  Retry Delay: {browser_driver.retry_delay}")


def config_file_example():
    """
    Example of using the dependency injection container with configuration from a file.
    """
    print("\n=== Configuration File Example ===\n")

    # Create a temporary YAML configuration file
    config_file: str = "temp_config.yaml"
    config_dict: dict[str, Any] = {
        "output_dir": "./file_output",
        "agents": {
            "recon": {
                "output_dir": "./file_output/recon_reports",
                "model_name": "claude-2",
            },
            "prompt_engineer": {
                "output_dir": "./file_output/prompt_lists",
                "model_name": "claude-2",
            },
            "exploit_delivery": {
                "output_dir": "./file_output/exploit_logs",
                "model_name": "claude-2",
                "browser_method": "selenium",
                "headless": True,
            },
            "evaluation": {
                "output_dir": "./file_output/evaluation_reports",
                "model_name": "claude-2",
            },
        },
        "browser": {
            "method": "selenium",
            "headless": True,
            "timeout": 45000,
            "enhanced": True,
            "retry_attempts": 2,
            "retry_delay": 0.5,
        },
        "visualization": {
            "output_dir": "./file_output/visualizations",
            "dpi": 200,
            "theme": "light",
            "advanced": False,
            "interactive": False,
            "clustering_clusters": 3,
            "similarity_threshold": 0.6,
        },
    }

    try:
        # Write configuration to file
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        # Create container from configuration file
        container = GraySwanContainerFactory.create_container_from_file(config_file)

        # Create pipeline
        pipeline = GraySwanPipeline(container)

        # Get logger from container
        logger = container.logger()
        logger.info("Running example with configuration from file")

        # Get agents from container
        recon_agent = container.recon_agent()
        prompt_agent = container.prompt_engineer_agent()
        exploit_agent = container.exploit_delivery_agent()
        eval_agent = container.evaluation_agent()

        # Print agent configurations
        print(
            f"Reconnaissance Agent: model={recon_agent.model_name}, output_dir={recon_agent.output_dir}"
        )
        print(
            f"Prompt Engineer Agent: model={prompt_agent.model_name}, output_dir={prompt_agent.output_dir}"
        )
        print(
            f"Exploit Delivery Agent: model={exploit_agent.model_name}, output_dir={exploit_agent.output_dir}"
        )
        print(
            f"Evaluation Agent: model={eval_agent.model_name}, output_dir={eval_agent.output_dir}"
        )

        # Print browser configuration
        browser_driver = container.enhanced_browser_factory()
        print(f"\nBrowser Configuration:")
        print(f"  Method: {browser_driver.method}")
        print(f"  Headless: {browser_driver.headless}")
        print(f"  Retry Attempts: {browser_driver.retry_attempts}")
        print(f"  Retry Delay: {browser_driver.retry_delay}")

    finally:
        # Clean up temporary file
        if os.path.exists(config_file):
            os.remove(config_file)


def override_example():
    """
    Example of overriding specific configuration values.
    """
    print("\n=== Configuration Override Example ===\n")

    # Create container with default configuration
    container = GraySwanContainerFactory.create_container()

    # Override specific configuration values
    container.config.agents.recon.model_name.override("llama-2-70b")
    container.config.browser.headless.override(False)
    container.config.visualization.advanced.override(True)

    # Create pipeline
    pipeline = GraySwanPipeline(container)

    # Get logger from container
    logger = container.logger()
    logger.info("Running example with overridden configuration")

    # Get agents from container
    recon_agent = container.recon_agent()

    # Print agent configurations
    print(
        f"Reconnaissance Agent: model={recon_agent.model_name}, output_dir={recon_agent.output_dir}"
    )

    # Print browser configuration
    browser_driver = container.enhanced_browser_factory()
    print(f"\nBrowser Configuration:")
    print(f"  Method: {browser_driver.method}")
    print(f"  Headless: {browser_driver.headless}")


def pipeline_example():
    """
    Example of running a simplified pipeline with the dependency injection container.
    """
    print("\n=== Pipeline Example ===\n")

    # Create custom configuration for the example
    config_dict: dict[str, Any] = {
        "output_dir": "./example_output",
        "agents": {
            "recon": {
                "output_dir": "./example_output/recon_reports",
                "model_name": "gpt-3.5-turbo",
            },
            "prompt_engineer": {
                "output_dir": "./example_output/prompt_lists",
                "model_name": "gpt-3.5-turbo",
            },
            "exploit_delivery": {
                "output_dir": "./example_output/exploit_logs",
                "model_name": "gpt-3.5-turbo",
                "browser_method": "playwright",
                "headless": True,
            },
            "evaluation": {
                "output_dir": "./example_output/evaluation_reports",
                "model_name": "gpt-3.5-turbo",
            },
        },
    }

    # Create container with custom configuration
    container = GraySwanContainerFactory.create_container(config_dict)

    # Create pipeline
    pipeline = GraySwanPipeline(container)

    # Get logger from container
    logger = container.logger()
    logger.info("Running simplified pipeline example")

    # Define target model and behavior
    target_model: str = "Example Model"
    target_behavior: str = "generate harmful content"

    # Create output directory
    os.makedirs(config_dict["output_dir"], exist_ok=True)

    # Run a simplified pipeline (just print the steps)
    print(f"Running simplified pipeline for {target_model} - {target_behavior}")
    print(f"1. Reconnaissance phase")
    print(f"2. Prompt engineering phase")
    print(f"3. Exploit delivery phase")
    print(f"4. Evaluation phase")
    print(f"\nResults would be saved to: {os.path.abspath(config_dict['output_dir'])}")


def main():
    """Main function for the example script."""
    print("Dependency Injection Example for Gray Swan Arena")
    print("===============================================")

    # Run examples
    basic_example()
    custom_config_example()
    config_file_example()
    override_example()
    pipeline_example()

    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()
