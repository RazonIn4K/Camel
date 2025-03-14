# Gray Swan Arena Test Suite

This directory contains various test scripts for validating, benchmarking, and integrating Gray Swan Arena with external services.

## Prerequisites

Before running the tests, make sure you have:

1. Installed all required dependencies:
   ```bash
   pip install -e .
   pip install -e ".[dev]"  # For development dependencies
   ```

2. Set up necessary environment variables in a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_key_here
   AGENTOPS_API_KEY=your_agentops_key_here  # Optional, for AgentOps integration
   DISCORD_TOKEN=your_discord_token_here  # Optional, for Discord integration
   ```

## Running Tests

### Running All Tests

To run all available tests:

```bash
python tests/run_all_tests.py
```

### Running Specific Test Modules

To run only specific test modules:

```bash
python tests/run_all_tests.py --modules camel e2e browser
```

Available modules:
- `camel`: Tests integration with Camel AI framework
- `agentops`: Tests integration with AgentOps monitoring
- `e2e`: Runs an end-to-end test of Gray Swan Arena
- `browser`: Tests browser automation features
- `discord`: Tests Discord integration features
- `benchmark`: Benchmarks performance of Gray Swan Arena agents

### Running Benchmarks

To run only the benchmark tests:

```bash
python tests/run_all_tests.py --benchmark-only
```

You can customize the benchmark with additional options:

```bash
python tests/run_all_tests.py --benchmark-only \
  --benchmark-topics "network security" "password policies" \
  --benchmark-models "gpt-3.5-turbo" "gpt-4"
```

Alternatively, you can run the benchmark script directly with more options:

```bash
python tests/benchmark_agents.py --help
```

## Test Descriptions

### Camel Integration Test (`test_camel_integration.py`)
Tests the integration with Camel AI framework and OpenAI API. Verifies that Gray Swan Arena can properly use Camel's agents.

### AgentOps Integration Test (`test_agentops_integration.py`)
Tests the integration with AgentOps for monitoring agent performance. Verifies that Gray Swan Arena can send telemetry data to AgentOps.

### End-to-End Test (`test_grayswan_e2e.py`)
Tests the complete Gray Swan Arena pipeline:
1. Recon: gathering information about the target
2. Prompt Engineering: creating prompts based on recon data
3. Exploit Delivery: running prompts against the target model
4. Evaluation: analyzing results and generating reports

### Browser Automation Test (`test_browser_automation.py`)
Tests browser automation features using Playwright and Selenium. Verifies that Gray Swan Arena can interact with web-based AI models.

### Discord Integration Test (`test_discord_integration.py`)
Tests integration with Discord for data collection. Verifies that Gray Swan Arena can scrape and analyze Discord messages.

### Benchmark (`benchmark_agents.py`)
Benchmarks the performance of Gray Swan Arena agents with different models and topics. Measures execution time, throughput, and success rates.

## Test Output

All test output is saved to the `tests/output` directory. This includes:
- Test reports
- Benchmark results
- Generated visualizations
- Response data from agents

## Troubleshooting

If tests are failing, check the following:

1. **API Keys**: Ensure all necessary API keys are properly set in your `.env` file.
2. **Dependencies**: Make sure all dependencies are installed correctly.
3. **Network**: Some tests require internet access for API calls and web searches.
4. **Browser Automation**: For browser tests, ensure that at least one of Playwright or Selenium is installed:
   ```bash
   pip install playwright selenium
   playwright install
   ```
5. **Logs**: Check the test logs in the console output and files in `tests/output` for specific error messages.

## Adding New Tests

When adding new tests to this directory:

1. Create a new test file with a descriptive name.
2. Add the test to the `TEST_MODULES` dictionary in `run_all_tests.py`.
3. Follow the pattern of existing tests: setup, test functions, and a main function to run the tests.
4. Add appropriate error handling and output messages.
5. Update this README with information about the new test.
