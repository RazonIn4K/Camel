# Gray Swan Arena

Gray Swan Arena is a multi-agent cybersecurity simulation system that leverages large language models to create a dynamic environment for testing and training security capabilities.

## System Architecture

The Gray Swan Arena system is composed of several key components:

- **Agent Factory**: Creates and manages specialized agents for different tasks
- **Testing Framework**: Handles unit tests and edge case tests for system verification
- **Telemetry and Monitoring**: Provides real-time visibility into agent operations
- **Configuration Management**: Allows customization of agent behavior and models

## Directory Structure

```
├── config/                      # Configuration files
│   ├── model_config.yaml        # Model settings and parameters
│   └── agent_definitions.yaml   # Agent types and capabilities
├── modules/                     # Modular components
│   ├── agent_setup.py           # Agent initialization
│   ├── argument_parsing.py      # Command-line argument handling
│   ├── telemetry_setup.py       # OpenTelemetry and AgentOps integration
│   └── test_management.py       # Test registration and execution
├── cybersec_agents/             # Core agent implementations
│   └── grayswan/                # Gray Swan specific components
│       ├── tests/               # Test suites
│       │   ├── unit/            # Unit tests
│       │   └── edge_cases/      # Edge case tests
│       └── camel_integration.py # Integration with CAMEL framework
├── initialize_agents.py         # Main entry point
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenTelemetry Collector (optional, for advanced telemetry)
- AgentOps API key (optional, for agent tracking)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/gray-swan-arena.git
   cd gray-swan-arena
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # API keys for language models
   export OPENAI_API_KEY=your_openai_api_key
   
   # AgentOps tracking (optional)
   export AGENTOPS_API_KEY=your_agentops_api_key
   ```

## Usage

### Basic Usage

To initialize agents with default settings:

```bash
python initialize_agents.py
```

### Command-Line Options

The script supports the following options:

```
-c, --config CONFIG      Path to the model configuration file (default: config/model_config.yaml)
-o, --output_dir DIR     Directory for storing agent outputs and artifacts (default: output)
-t, --run_tests          Run unit tests during initialization
-e, --edge_case_tests    Run edge case tests
--edge_test_category CAT Category of edge case tests to run (choices: network, data, 
                         concurrency, resource, service, all)
--report_dir DIR         Directory for test reports (default: reports)
-v, --verbose            Enable verbose logging
-d, --debug              Enable debug mode
-m, --model MODEL        Override the default model specified in the config
```

### Examples

Run with unit tests and verbose logging:
```bash
python initialize_agents.py -t -v
```

Run edge case tests for network failures:
```bash
python initialize_agents.py -e --edge_test_category network
```

Use a custom configuration file:
```bash
python initialize_agents.py -c custom_config.yaml
```

## Configuration

### Model Configuration

The `config/model_config.yaml` file contains settings for language models used by the agents, including:

- Default model selection
- Temperature and token settings
- Retry policies
- Task-specific model assignments

### Agent Definitions

The `config/agent_definitions.yaml` file defines the agents in the system:

- Agent types and descriptions
- Capabilities for each agent
- Model assignments and backup models
- Agent-specific parameters
- Interaction rules between agents

## Edge Case Testing

The edge case testing framework tests system resilience against:

- Network failures
- Data corruption
- Concurrency issues
- Resource exhaustion
- Service degradation

Run all edge case tests with:
```bash
python initialize_agents.py -e
```

## Telemetry and Monitoring

The system includes comprehensive telemetry via OpenTelemetry and AgentOps:

- Agent activity tracking
- Test execution metrics
- Performance monitoring
- Error tracking and analysis

## Development

### Adding New Agents

1. Define the agent in `config/agent_definitions.yaml`
2. Implement the agent class in the appropriate module
3. Register the agent type with the AgentFactory

### Adding New Tests

1. Create a new test module in the appropriate test directory
2. Implement the tests following the existing patterns
3. Add a `register_tests` function to register with TestManager

## License

This project is licensed under the MIT License - see the LICENSE file for details.
