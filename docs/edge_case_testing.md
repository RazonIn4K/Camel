# Edge Case Testing Framework

## Overview

The Edge Case Testing Framework is a comprehensive testing solution designed to verify the resilience and robustness of the Gray Swan Arena system under various adverse conditions. This framework enables systematic testing of failure scenarios, concurrency issues, resource constraints, and other edge cases that may impact system behavior.

## Key Components

The framework consists of several key components:

1. **FailureSimulator**: Provides utilities for simulating various types of failures in tests:
   - Network failures (connection errors, timeouts)
   - Service degradation (increased latency, jitter)
   - Data corruption (malformed data, truncated content)
   - Resource exhaustion (memory, CPU, disk space)

2. **ConcurrencyTester**: Enables testing of concurrency-related issues:
   - Race conditions when multiple threads access shared resources
   - Deadlock scenarios with multiple locks acquired in different orders

3. **EdgeCaseTestRunner**: Manages the execution of edge case tests with detailed reporting:
   - Provides consistent test execution patterns
   - Collects detailed metrics about test performance
   - Generates comprehensive test reports

4. **Specialized Test Modules**: Various pre-implemented test modules targeting specific failure modes:
   - Network failure tests
   - Data corruption tests
   - Concurrency issue tests
   - Resource exhaustion tests
   - Service degradation tests

5. **Telemetry Integration**: Built-in support for monitoring and observability:
   - OpenTelemetry tracing for detailed debugging
   - Metrics collection for test performance analysis
   - AgentOps integration for test result tracking

## Test Categories

The framework includes tests in the following categories:

### Network Failures
- Intermittent network outages
- Complete network disconnections
- High-latency connections
- API rate limiting

### Data Corruption
- Malformed message handling
- Message truncation
- JSON parsing errors
- Invalid response formats

### Concurrency Issues
- Race conditions between multiple agents
- Deadlocks and livelocks
- Thread exhaustion
- Inconsistent state handling

### Resource Exhaustion
- Memory constraints
- CPU saturation
- Disk space limitations
- Thread pool exhaustion

### Service Degradation
- Slow API responses
- Gradual performance decline
- Intermittent timeouts
- Backend service unavailability

## Usage

### Running Edge Case Tests

To run edge case tests, use the provided command-line script:

```bash
python -m cybersec_agents.grayswan.tests.edge_cases.run_edge_case_tests
```

#### Command-Line Arguments

- `--categories` / `-c`: Specify test categories to run (default: all)
  ```bash
  python -m cybersec_agents.grayswan.tests.edge_cases.run_edge_case_tests -c network data
  ```

- `--report-dir` / `-r`: Set the directory for test reports (default: ./reports)
  ```bash
  python -m cybersec_agents.grayswan.tests.edge_cases.run_edge_case_tests -r /path/to/reports
  ```

- `--verbose` / `-v`: Enable verbose logging
  ```bash
  python -m cybersec_agents.grayswan.tests.edge_cases.run_edge_case_tests -v
  ```

### Integration with Initialize Agents

Edge case tests can also be run as part of the agent initialization process:

```bash
python initialize_agents.py --edge_case_tests --edge_test_category network
```

#### Command-Line Arguments

- `--edge_case_tests` / `-e`: Run edge case tests during initialization
- `--edge_test_category`: Specify which category of edge case tests to run (options: network, data, concurrency, resource, service, all)
- `--report_dir`: Define the directory for edge case test reports

### Writing Custom Edge Case Tests

You can create custom edge case tests by utilizing the provided framework classes and implementing specialized test functions. Here's a basic example of a custom test:

```python
from cybersec_agents.grayswan.tests.edge_cases.edge_case_framework import FailureSimulator

def test_my_custom_edge_case():
    """Test custom edge case scenario."""
    
    # Set up test requirements
    resource = {}
    
    # Use the failure simulator to create adverse conditions
    with FailureSimulator.network_failure(probability=0.5):
        # Perform operations that should be resilient to network failures
        result = perform_operation(resource)
        
    # Verify expectations
    assert result.status == "success", "Operation should succeed despite failures"
    
    return {
        "status": result.status,
        "retry_count": result.retry_count
    }

# Register the test with the EdgeCaseTestManager
def register_tests(test_manager):
    test_manager.register_test(test_my_custom_edge_case)
```

## Understanding Test Reports

Edge case test reports are generated in HTML format and include:

1. A summary section with overall pass/fail statistics
2. System metrics showing resource usage during tests
3. Detailed results for each individual test, including:
   - Test name and description
   - Pass/fail status
   - Duration
   - Error messages (if any)
   - Test-specific details and measurements

## Telemetry and Debugging

The Edge Case Testing Framework integrates with OpenTelemetry to provide detailed tracing information:

- Each test execution creates a trace with spans for each operation
- Spans include attributes like test parameters, success/failure status, and timing information
- Error conditions are recorded with detailed exception information
- Key events within each test are recorded with timestamps

## Best Practices

When working with the Edge Case Testing Framework:

1. **Start with existing test modules** before writing custom tests
2. **Run specific categories** during development to focus on relevant edge cases
3. **Review HTML reports** to identify patterns in failures
4. **Use the verbose flag** for detailed logging during troubleshooting
5. **Incorporate edge case testing** into your CI/CD pipeline
6. **Gradually increase test complexity** from simple network failures to more complex concurrency issues

## Advanced Configuration

For advanced use cases, you can configure:

- Custom failure probabilities and patterns
- System-specific resource thresholds
- Test repetition counts for detecting inconsistent failures
- Integration with external monitoring systems
- Custom metrics collection for specialized tests

## Troubleshooting

If you encounter issues with the Edge Case Testing Framework:

1. Check that all required dependencies are installed
2. Verify that your testing environment has sufficient resources
3. Review the logs with verbose mode enabled
4. Ensure that test implementations match the expected interface
5. Check for conflicts with other systems that may be using the same resources

For further assistance, consult the API documentation or contact the Gray Swan Arena development team. 