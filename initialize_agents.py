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

# Add the directory to the Python path if necessary
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the install_nltk_dependencies module
try:
    from scripts.install_nltk_dependencies import run_installation
except ImportError:
    # More descriptive error handling
    def run_installation(**kwargs):
        logger.error("Critical dependency missing: install_nltk_dependencies.py")
        logger.error("Some functionality may not work correctly without NLTK")
        return False

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
    modules = {}
    missing_modules = []
    
    # Try each import separately for better error tracking
    try:
        from modules import telemetry_setup
        modules["telemetry_setup"] = telemetry_setup
    except ImportError as e:
        logger.error(f"Failed to import telemetry_setup module: {str(e)}")
        missing_modules.append(("modules.telemetry_setup", str(e)))
        
    # Similar blocks for other required modules...
    
    if missing_modules:
        error_msg = "Failed to import required modules:\n"
        for module_path, error in missing_modules:
            error_msg += f"  - {module_path}: {error}\n"
        raise ImportError(error_msg)
    
    return modules

def ensure_nltk_dependencies():
    """Ensure NLTK dependencies are properly installed."""
    nltk_installed = run_installation(quiet=True)
    
    if not nltk_installed:
        logger.error("Failed to install NLTK dependencies.")
        
        # Only prompt in interactive mode
        if sys.stdin.isatty() and sys.stdout.isatty():
            if input("Continue anyway? (y/N): ").lower() != 'y':
                logger.info("Exiting due to NLTK installation failure.")
                return False
        
        logger.warning("Continuing without NLTK. Note that sentiment analysis and text processing features may not work correctly.")
    
    return True

def initialize_agents(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Initialize all agents based on configuration.
    
    Args:
        config: Configuration dictionary for the agents
        
    Returns:
        Dictionary containing initialized agents
    """
    # Make sure NLTK dependencies are installed
    nltk_installed = run_installation(quiet=True)
    if not nltk_installed:
        print("Warning: NLTK dependencies may not be properly installed.")
    
    # Initialize agents based on configuration
    # (existing code would go here)
    
    # Return the initialized agents
    return {}  # Replace with actual initialized agents

def setup_environment(args):
    """Set up the environment for agents, including telemetry and metrics.
    
    Args:
        args: Dictionary containing command-line arguments
        
    Returns:
        Tuple containing (tracer, metrics_counters, agentops_available)
    """
    # Enable debug logging if requested
    if args.get("debug"):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize telemetry
    tracer, metrics_counters = telemetry_setup.initialize_telemetry()
    
    # Initialize AgentOps for tracking
    agentops_available = telemetry_setup.initialize_agentops(tracer)
    
    return tracer, metrics_counters, agentops_available

def setup_agents_and_testing(args, tracer, metrics_counters):
    """Setup agent factory and testing infrastructure.
    
    Args:
        args: Dictionary containing command-line arguments
        tracer: Telemetry tracer object
        metrics_counters: Metrics counter object
        
    Returns:
        Tuple containing (model_config_manager, agent_factory, agent_instances, 
                         test_manager, output_dir)
    """
    # Create output directory
    output_dir = agent_setup.create_output_directory(args.get("output_dir", "output"))
    
    # Initialize model config manager and agents
    config_file = args.get("config", "config/model_config.yaml")
    model_config_manager = agent_setup.ModelConfigManager(config_file=config_file)
    
    # Apply model override if specified
    if args.get("model"):
        override_model = args["model"]
        logger.info(f"Overriding default model with: {override_model}")
        # Actually implement the override
        model_config_manager.set_default_model(override_model)
    
    # Initialize agents
    agent_factory, agent_instances = agent_setup.setup_agent_factory(
        model_config_manager,
        output_dir,
        tracer,
        metrics_counters
    )
    
    # Initialize test manager
    test_manager = TestManager(agent_factory)
    test_management.register_tests(test_manager, tracer, agent_instances)
    
    return model_config_manager, agent_factory, agent_instances, test_manager, output_dir

def main():
    """Main function for agent initialization."""
    start_time = time.time()
    success = False  # Set default value
    
    try:
        logger.info("Starting agent initialization...")
        
        # Ensure NLTK dependencies are installed
        if not ensure_nltk_dependencies():
            sys.exit(1)
        
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
        
        # Setup environment
        tracer, metrics_counters, agentops_available = setup_environment(args)
        
        # Setup agents and testing
        model_config_manager, agent_factory, agent_instances, test_manager, output_dir = setup_agents_and_testing(args, tracer, metrics_counters)
        
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
        
        # Make sure NLTK dependencies are installed
        nltk_installed = run_installation(quiet=True)
        if not nltk_installed:
            logger.error("Failed to install NLTK dependencies.")
            if input("Continue anyway? (y/N): ").lower() != 'y':
                logger.info("Exiting due to NLTK installation failure.")
                sys.exit(1)
            else:
                logger.warning("Continuing without NLTK. Some features may not work.")
        
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
        telemetry_setup_available = 'telemetry_setup' in locals() and telemetry_setup is not None
        tracer_available = 'tracer' in locals() and tracer is not None
        
        if telemetry_setup_available:
            try:
                telemetry_setup.end_agentops_session(
                    tracer if tracer_available else None,
                    success=success
                )
                logger.info("AgentOps session ended successfully")
            except Exception as e:
                logger.error(f"Failed to end AgentOps session: {str(e)}")
        
        # Return success status as exit code
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()  # No need for sys.exit(main()) since main already calls sys.exit

