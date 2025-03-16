#!/usr/bin/env python3
"""
Initialize Agents for Gray Swan Arena.

This script initializes the agents for the Gray Swan Arena multi-agent system,
sets up telemetry and monitoring, and runs tests to verify agent functionality.

Usage:
    python initialize_agents.py [options]

Options:
    -c, --config CONFIG      Path to the model configuration file (default: config/model_config.yaml)
    -o, --output_dir DIR     Directory for storing agent outputs and artifacts (default: output)
    -t, --run_tests          Run unit tests during initialization
    -e, --edge_case_tests    Run edge case tests
    --edge_test_category CAT Category of edge case tests to run (choices: network, data, concurrency, resource, service, all)
    --report_dir DIR         Directory for test reports (default: reports)
    -v, --verbose            Enable verbose logging
    -d, --debug              Enable debug mode
    -m, --model MODEL        Override the default model specified in the config
"""

import os
import sys
import logging
import time
from typing import Dict, Any, Set, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_initialization.log')
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)

def import_modules():
    """Import required modules for agent initialization."""
    try:
        # Import internal modules
        from modules import telemetry_setup, agent_setup, test_management, argument_parsing
        
        # Import camel_integration
        from cybersec_agents.grayswan.camel_integration import TestManager, TestTier
        
        return {
            "telemetry_setup": telemetry_setup,
            "agent_setup": agent_setup,
            "test_management": test_management,
            "argument_parsing": argument_parsing,
            "TestManager": TestManager,
            "TestTier": TestTier
        }
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        raise ImportError(f"Required module not found: {str(e)}")

def main():
    """Main function for agent initialization."""
    start_time = time.time()
    success = False
    
    try:
        logger.info("Starting agent initialization...")
        
        # Import modules
        modules = import_modules()
        telemetry_setup = modules["telemetry_setup"]
        agent_setup = modules["agent_setup"]
        test_management = modules["test_management"]
        argument_parsing = modules["argument_parsing"]
        TestManager = modules["TestManager"]
        TestTier = modules["TestTier"]
        
        # Parse command-line arguments
        args = argument_parsing.parse_args()
        
        # Enable debug logging if requested
        if args.get("debug"):
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Initialize telemetry
        tracer, metrics_counters = telemetry_setup.initialize_telemetry()
        
        # Initialize AgentOps for tracking
        agentops_available = telemetry_setup.initialize_agentops(tracer)
        
        # Create output directory
        output_dir = agent_setup.create_output_directory(args.get("output_dir", "output"))
        
        # Initialize model config manager
        config_file = args.get("config", "config/model_config.yaml")
        model_config_manager = agent_setup.ModelConfigManager(config_file=config_file)
        
        # Override model if specified in arguments
        if args.get("model"):
            logger.info(f"Overriding default model with: {args['model']}")
            # No direct way to override, but we can use the model in the AgentFactory creation
        
        # Setup agent factory and create agents
        agent_factory, agent_instances = agent_setup.setup_agent_factory(
            model_config_manager,
            output_dir,
            tracer,
            metrics_counters
        )
        
        # Initialize test manager
        test_manager = TestManager(agent_factory)
        
        # Register tests
        test_management.register_tests(test_manager, tracer, agent_instances)
        
        # Run unit tests if requested
        if args.get("run_tests"):
            unit_test_results = test_management.run_unit_tests(
                test_manager,
                tracer,
                metrics_counters
            )
            logger.info(f"Unit test results: {unit_test_results.get('passed', 0)} passed, " +
                       f"{unit_test_results.get('failed', 0)} failed")
        
        # Run edge case tests if requested
        if args.get("edge_case_tests"):
            # Convert edge test category to a set if specified
            edge_categories = None
            if args.get("edge_test_category"):
                edge_categories = {args["edge_test_category"]}
            
            # Run edge case tests
            edge_case_results = test_management.run_edge_case_tests(
                agent_factory,
                tracer,
                metrics_counters,
                categories=edge_categories,
                report_dir=args.get("report_dir")
            )
            
            # Log summary results
            summary = edge_case_results.get("summary", {})
            logger.info(f"Edge case test summary: {summary.get('passed', 0)} passed, " +
                       f"{summary.get('failed', 0)} failed")
        
        logger.info("Agent initialization completed successfully")
        success = True
        
    except Exception as e:
        logger.error(f"Agent initialization failed: {str(e)}", exc_info=True)
        success = False
        
    finally:
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        # End AgentOps session if available
        try:
            if 'telemetry_setup' in locals():
                telemetry_setup.end_agentops_session(
                    tracer if 'tracer' in locals() else None,
                    success=success
                )
        except Exception as e:
            logger.error(f"Failed to end AgentOps session: {str(e)}")
        
        # Return success status as exit code
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

