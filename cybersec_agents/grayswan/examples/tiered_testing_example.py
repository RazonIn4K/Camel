"""
Example usage of the tiered testing framework in the CAMEL integration module.

This example demonstrates how to set up and run tests in different tiers
using the TestManager and TestTier classes.
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
from cybersec_agents.grayswan.camel_integration import (
    AgentFactory, 
    TestManager, 
    TestTier, 
    setup_test_suite
)

def main():
    """Run the tiered testing example."""
    # Create output directories for tests
    os.makedirs("./test_output/prompts", exist_ok=True)
    os.makedirs("./test_output/recon", exist_ok=True)
    os.makedirs("./test_output/exploits", exist_ok=True)
    os.makedirs("./test_output/evaluations", exist_ok=True)
    
    logger.info("Starting tiered testing example...")
    
    try:
        # Create a ModelConfigManager
        model_manager = ModelConfigManager()
        logger.info(f"Created ModelConfigManager with config file: {model_manager.config_path}")
        
        # Create an AgentFactory with the ModelConfigManager
        agent_factory = AgentFactory(model_manager)
        logger.info("Created AgentFactory")
        
        # Set up the test suite with all tests registered
        test_manager = setup_test_suite(agent_factory)
        logger.info("Set up test suite with registered tests")
        
        # Run unit tests
        logger.info("Running unit tests...")
        try:
            unit_results = test_manager.run_tests(TestTier.UNIT)
            logger.info(f"Unit tests completed. Passed: {sum(1 for r in unit_results if r['success'])}/{len(unit_results)}")
        except Exception as e:
            logger.error(f"Unit tests failed: {e}")
        
        # Run integration tests
        logger.info("\nRunning integration tests...")
        try:
            integration_results = test_manager.run_tests(TestTier.INTEGRATION)
            logger.info(f"Integration tests completed. Passed: {sum(1 for r in integration_results if r['success'])}/{len(integration_results)}")
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
        
        # Run scenario tests
        logger.info("\nRunning scenario tests...")
        try:
            scenario_results = test_manager.run_tests(TestTier.SCENARIO)
            logger.info(f"Scenario tests completed. Passed: {sum(1 for r in scenario_results if r['success'])}/{len(scenario_results)}")
        except Exception as e:
            logger.error(f"Scenario tests failed: {e}")
        
        logger.info("\nTiered testing example completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during tiered testing example: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 