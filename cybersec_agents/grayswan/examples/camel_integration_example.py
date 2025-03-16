"""
Example usage of the CAMEL integration module.

This example demonstrates how to use the AgentFactory and CommunicationChannel
classes to create agents and facilitate communication between them.
"""

import logging
import os
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to Python path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from cybersec_agents.grayswan.utils.model_config_manager import ModelConfigManager
from cybersec_agents.grayswan.camel_integration import AgentFactory, CommunicationChannel


def main():
    """Run the example."""
    # Create a ModelConfigManager
    model_manager = ModelConfigManager()
    logger.info(f"Created ModelConfigManager with config file: {model_manager.config_path}")
    
    # Create an AgentFactory with the ModelConfigManager
    agent_factory = AgentFactory(model_manager)
    logger.info("Created AgentFactory")
    
    # Create a CommunicationChannel
    channel = CommunicationChannel()
    logger.info("Created CommunicationChannel")
    
    try:
        # Create a PromptEngineerAgent
        prompt_engineer = agent_factory.create_agent(
            "prompt_engineer",
            output_dir="./examples/prompts",
            model_name="gpt-4o",
            reasoning_model="o3-mini"
        )
        logger.info("Created PromptEngineerAgent")
        
        # Create an EvaluationAgent
        evaluation_agent = agent_factory.create_agent(
            "evaluation",
            output_dir="./examples/evaluations",
            model_name="o3-mini",
            backup_model="gpt-4o",
            reasoning_model="o3-mini"
        )
        logger.info("Created EvaluationAgent")
        
        # Demonstrate sending a message from the PromptEngineerAgent to the EvaluationAgent
        message = {
            "type": "prompt_submission",
            "sender": "prompt_engineer",
            "recipient": "evaluation",
            "content": {
                "prompt": "Is this a secure implementation?",
                "target_model": "test-model",
                "target_behavior": "security_analysis"
            }
        }
        
        logger.info(f"Sending message: {message}")
        channel.send_message(message)
        
        # Demonstrate receiving the message
        received = channel.receive_message()
        logger.info(f"Received message: {received}")
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during example execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 