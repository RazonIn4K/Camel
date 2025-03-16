#!/usr/bin/env python
"""
Edge Case Test Runner for Gray Swan Arena.

This script provides a command-line interface for running edge case tests
in the Gray Swan Arena. It includes options for selecting specific test
categories, generating reports, and tracking metrics.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, Any

# Set up path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Try to import monitoring and metrics libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from opentelemetry import metrics
    from opentelemetry.metrics import get_meter_provider
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Import edge case test modules
from cybersec_agents.grayswan.tests.edge_cases.edge_case_framework import EdgeCaseTestRunner
from cybersec_agents.grayswan.utils.logging_utils import setup_logging
from cybersec_agents.grayswan.camel_integration import TestTier

# Define the edge case test tier
# Extend TestTier class with EDGE_CASE
setattr(TestTier, 'EDGE_CASE', 'edge_case')

# Try to import all test modules
try:
    from cybersec_agents.grayswan.tests.edge_cases.test_network_failures import register_tests as register_network_tests
    NETWORK_TESTS_AVAILABLE = True
except ImportError:
    NETWORK_TESTS_AVAILABLE = False

try:
    from cybersec_agents.grayswan.tests.edge_cases.test_data_corruption import register_tests as register_data_tests
    DATA_TESTS_AVAILABLE = True
except ImportError:
    DATA_TESTS_AVAILABLE = False

try:
    from cybersec_agents.grayswan.tests.edge_cases.test_concurrency_issues import register_tests as register_concurrency_tests
    CONCURRENCY_TESTS_AVAILABLE = True
except ImportError:
    CONCURRENCY_TESTS_AVAILABLE = False

try:
    from cybersec_agents.grayswan.tests.edge_cases.test_resource_exhaustion import register_tests as register_resource_tests
    RESOURCE_TESTS_AVAILABLE = True
except ImportError:
    RESOURCE_TESTS_AVAILABLE = False

try:
    from cybersec_agents.grayswan.tests.edge_cases.test_service_degradation import register_tests as register_service_tests
    SERVICE_TESTS_AVAILABLE = True
except ImportError:
    SERVICE_TESTS_AVAILABLE = False

# Set up logger
logger = setup_logging("edge_case_runner")

# Create a modified version of TestManager that doesn't require AgentFactory
class EdgeCaseTestManager:
    """
    Manages and runs edge case tests.
    
    This is a simplified version of TestManager that doesn't require
    an AgentFactory, since edge case tests are more focused on system
    behaviors rather than specific agent implementations.
    """
    
    def __init__(self):
        """Initialize the EdgeCaseTestManager."""
        self.logger = logging.getLogger(__name__)
        self.tests: Dict[str, List[Callable]] = {
            TestTier.EDGE_CASE: []
        }
        self.test_results = []
        
    def get_test_results(self):
        """
        Get all test results collected during test runs.
        
        Returns:
            A list of test result dictionaries
        """
        return self.test_results

    def register_test(self, test_func: Callable):
        """
        Register a test function in the edge case tier.
        
        Args:
            test_func: The test function to register
        """
        self.tests[TestTier.EDGE_CASE].append(test_func)
        self.logger.info(f"Registered edge case test: {test_func.__name__}")

    def run_tests(self):
        """
        Run all edge case tests.
        
        Returns:
            List of test result dictionaries for this run
        """
        tier = TestTier.EDGE_CASE
        self.logger.info(f"Starting {tier} tests...")
        
        success = True
        results = []
        start_time = time.time()
        
        for test_func in self.tests[tier]:
            test_start_time = time.time()
            test_result = {
                "name": test_func.__name__,
                "tier": tier,
                "success": False,
                "status": "failed",
                "error": None,
                "duration": 0,
                "details": {}
            }
            
            try:
                self.logger.info(f"Running test: {test_func.__name__}")
                test_details = test_func()  # Edge case tests don't need agent_factory
                test_result["status"] = "passed"
                test_result["success"] = True
                test_result["details"] = test_details or {}
                self.logger.info(f"Test {test_func.__name__} passed.")
                
            except AssertionError as e:
                self.logger.error(f"Test {test_func.__name__} assertion failed: {e}")
                test_result["status"] = "failed"
                test_result["success"] = False
                test_result["error"] = str(e)
                success = False
            except Exception as e:
                self.logger.error(f"Test {test_func.__name__} error: {e}")
                test_result["status"] = "error"
                test_result["success"] = False
                test_result["error"] = f"{type(e).__name__}: {str(e)}"
                success = False
            
            test_result["duration"] = time.time() - test_start_time
            results.append(test_result)
            self.test_results.append(test_result)
        
        total_duration = time.time() - start_time
        
        # Record overall test run results
        run_result = {
            "name": "test_run_completed",
            "tier": tier,
            "result": "success" if success else "failure",
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r["success"]),
            "failed_tests": sum(1 for r in results if not r["success"]),
            "duration": total_duration
        }
        
        self.test_results.append(run_result)

        if success:
            self.logger.info(f"{tier} tests completed successfully.")
        else:
            self.logger.warning(f"{tier} tests had failures or errors.")
        
        return results


# Set up metrics if available
if METRICS_AVAILABLE:
    meter = get_meter_provider().get_meter("edge_case_testing")
    test_duration_histogram = meter.create_histogram(
        name="edge_case_test_duration",
        description="Duration of edge case tests in seconds",
        unit="s"
    )
    test_success_counter = meter.create_counter(
        name="edge_case_test_success_count",
        description="Count of successful edge case tests",
        unit="1"
    )
    test_failure_counter = meter.create_counter(
        name="edge_case_test_failure_count", 
        description="Count of failed edge case tests",
        unit="1"
    )


def setup_metrics_recording() -> Optional[Dict]:
    """Set up metrics recording for system resource usage during tests."""
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available, system metrics will not be recorded")
        return None
    
    # Initial metrics snapshot
    metrics_data = {
        "start_time": time.time(),
        "start_cpu_percent": psutil.cpu_percent(interval=0.1),
        "start_memory_percent": psutil.virtual_memory().percent,
        "start_memory_used": psutil.virtual_memory().used,
        "process": psutil.Process(os.getpid()),
        "start_process_cpu_percent": 0,
        "start_process_memory_info": None,
    }
    
    # Get process-specific metrics
    try:
        metrics_data["start_process_cpu_percent"] = metrics_data["process"].cpu_percent(interval=0.1)
        metrics_data["start_process_memory_info"] = metrics_data["process"].memory_info()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
        logger.warning(f"Could not get process metrics: {str(e)}")
    
    return metrics_data


def record_metrics(metrics_data: Optional[Dict]) -> Dict:
    """Record current metrics and calculate differences from the start."""
    if not metrics_data:
        return {}
    
    results = {
        "duration": time.time() - metrics_data["start_time"],
        "cpu_percent_change": psutil.cpu_percent(interval=0.1) - metrics_data["start_cpu_percent"],
        "memory_percent_change": psutil.virtual_memory().percent - metrics_data["start_memory_percent"],
        "memory_used_change": psutil.virtual_memory().used - metrics_data["start_memory_used"],
    }
    
    # Get process-specific metrics
    try:
        process = metrics_data["process"]
        current_process_cpu = process.cpu_percent(interval=0.1)
        current_process_memory = process.memory_info()
        
        results["process_cpu_percent_change"] = current_process_cpu - metrics_data["start_process_cpu_percent"]
        
        if metrics_data["start_process_memory_info"]:
            results["process_rss_change"] = current_process_memory.rss - metrics_data["start_process_memory_info"].rss
            results["process_vms_change"] = current_process_memory.vms - metrics_data["start_process_memory_info"].vms
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
        logger.warning(f"Could not get process metrics: {str(e)}")
    
    return results


def generate_report(results: Dict, output_dir: str) -> str:
    """Generate a detailed HTML report from test results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for the report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(output_dir, f"edge_case_report_{timestamp}.html")
    
    # Start building the HTML report
    html_content = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Edge Case Test Report</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        .header { background-color: #f4f4f4; padding: 10px; border-radius: 5px; }",
        "        .summary { margin: 20px 0; }",
        "        .test-result { margin-bottom: 15px; padding: 10px; border-radius: 5px; }",
        "        .passed { background-color: #e6ffe6; }",
        "        .failed { background-color: #ffe6e6; }",
        "        .error { background-color: #fff0e6; }",
        "        .details { margin-top: 10px; }",
        "        table { border-collapse: collapse; width: 100%; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "        th { background-color: #f2f2f2; }",
        "        .metrics { margin-top: 20px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='header'>",
        f"        <h1>Edge Case Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h1>",
        "    </div>",
        "    <div class='summary'>",
        f"        <h2>Summary</h2>",
        f"        <p>Total Tests: {results.get('total_tests', 0)}</p>",
        f"        <p>Passed: {results.get('passed', 0)}</p>",
        f"        <p>Failed: {results.get('failed', 0)}</p>",
        f"        <p>Errors: {results.get('errors', 0)}</p>",
        f"        <p>Total Duration: {results.get('duration', 0):.2f} seconds</p>",
        f"        <p>Pass Rate: {results.get('pass_rate', 0)*100:.2f}%</p>",
        "    </div>",
    ]
    
    # Add system metrics if available
    if 'system_metrics' in results:
        metrics = results['system_metrics']
        html_content.extend([
            "    <div class='metrics'>",
            "        <h2>System Metrics</h2>",
            "        <table>",
            "            <tr><th>Metric</th><th>Value</th></tr>",
            f"            <tr><td>CPU Usage Change</td><td>{metrics.get('cpu_percent_change', 'N/A'):.2f}%</td></tr>",
            f"            <tr><td>Memory Usage Change</td><td>{metrics.get('memory_percent_change', 'N/A'):.2f}%</td></tr>",
            f"            <tr><td>Memory Used Change</td><td>{metrics.get('memory_used_change', 'N/A') / (1024*1024):.2f} MB</td></tr>",
            f"            <tr><td>Process CPU Change</td><td>{metrics.get('process_cpu_percent_change', 'N/A'):.2f}%</td></tr>",
        ])
        if 'process_rss_change' in metrics:
            html_content.append(
                f"            <tr><td>Process RSS Change</td><td>{metrics.get('process_rss_change', 'N/A') / (1024*1024):.2f} MB</td></tr>"
            )
        if 'process_vms_change' in metrics:
            html_content.append(
                f"            <tr><td>Process VMS Change</td><td>{metrics.get('process_vms_change', 'N/A') / (1024*1024):.2f} MB</td></tr>"
            )
        html_content.extend([
            "        </table>",
            "    </div>",
        ])
    
    # Add detailed test results
    html_content.extend([
        "    <div class='test-results'>",
        "        <h2>Detailed Test Results</h2>",
    ])
    
    for test_result in results.get('test_results', []):
        status_class = 'passed' if test_result.get('success', False) else ('failed' if test_result.get('status') == 'failed' else 'error')
        html_content.extend([
            f"        <div class='test-result {status_class}'>",
            f"            <h3>{test_result.get('name', 'Unknown Test')}</h3>",
            f"            <p><strong>Description:</strong> {test_result.get('description', 'No description')}</p>",
            f"            <p><strong>Status:</strong> {test_result.get('status', 'Unknown')}</p>",
            f"            <p><strong>Duration:</strong> {test_result.get('duration', 0):.2f} seconds</p>",
        ])
        
        if 'error' in test_result and test_result['error']:
            html_content.append(f"            <p><strong>Error:</strong> {test_result['error']}</p>")
        
        if 'details' in test_result and test_result['details']:
            html_content.extend([
                "            <div class='details'>",
                "                <h4>Test Details</h4>",
                "                <table>",
                "                    <tr><th>Key</th><th>Value</th></tr>",
            ])
            
            for key, value in test_result['details'].items():
                # Handle different types of values
                if isinstance(value, dict):
                    formatted_value = "<pre>" + str(value) + "</pre>"
                elif isinstance(value, list):
                    formatted_value = "<pre>" + str(value) + "</pre>"
                else:
                    formatted_value = str(value)
                
                html_content.append(f"                    <tr><td>{key}</td><td>{formatted_value}</td></tr>")
            
            html_content.extend([
                "                </table>",
                "            </div>",
            ])
        
        html_content.append("        </div>")
    
    # Close the HTML tags
    html_content.extend([
        "    </div>",
        "</body>",
        "</html>",
    ])
    
    # Write the HTML content to the file
    with open(report_filename, 'w') as f:
        f.write('\n'.join(html_content))
    
    logger.info(f"Report generated at {report_filename}")
    return report_filename


def register_all_edge_case_tests(test_manager: EdgeCaseTestManager, categories: Optional[Set[str]] = None) -> int:
    """
    Register all available edge case tests with the test manager.
    
    Args:
        test_manager: The test manager to register tests with
        categories: Set of categories to include, or None for all
        
    Returns:
        Number of test categories registered
    """
    categories_set = categories if categories is not None else {'all'}
    registered_count = 0
    
    # Helper function to check if category should be registered
    def should_register(category):
        return 'all' in categories_set or category in categories_set
    
    # Register network failure tests
    if NETWORK_TESTS_AVAILABLE and should_register('network'):
        logger.info("Registering network failure tests")
        register_network_tests(test_manager)
        registered_count += 1
    
    # Register data corruption tests
    if DATA_TESTS_AVAILABLE and should_register('data'):
        logger.info("Registering data corruption tests")
        register_data_tests(test_manager)
        registered_count += 1
    
    # Register concurrency issue tests
    if CONCURRENCY_TESTS_AVAILABLE and should_register('concurrency'):
        logger.info("Registering concurrency issue tests")
        register_concurrency_tests(test_manager)
        registered_count += 1
    
    # Register resource exhaustion tests
    if RESOURCE_TESTS_AVAILABLE and should_register('resource'):
        logger.info("Registering resource exhaustion tests")
        register_resource_tests(test_manager)
        registered_count += 1
    
    # Register service degradation tests
    if SERVICE_TESTS_AVAILABLE and should_register('service'):
        logger.info("Registering service degradation tests")
        register_service_tests(test_manager)
        registered_count += 1
    
    return registered_count


def run_edge_case_tests(categories: Optional[Set[str]] = None, report_dir: Optional[str] = None) -> Dict:
    """
    Run edge case tests and return results.
    
    Args:
        categories: Set of test categories to run, or None for all
        report_dir: Directory to save test reports to, or None for no reports
        
    Returns:
        Dictionary containing test results
    """
    # Initialize our custom test manager that doesn't require AgentFactory
    test_manager = EdgeCaseTestManager()
    
    # Register tests
    categories_set = categories if categories is not None else {'all'}
    num_categories = register_all_edge_case_tests(test_manager, categories_set)
    logger.info(f"Registered {num_categories} test categories")
    
    # Initialize the edge case test runner
    test_runner = EdgeCaseTestRunner()
    
    # Set up metrics recording
    metrics_data = setup_metrics_recording()
    
    # Run tests
    logger.info(f"Running edge case tests for categories: {', '.join(categories_set)}")
    test_results = test_manager.run_tests()
    
    # Process results
    processed_results = {
        'total_tests': len(test_results),
        'passed': sum(1 for r in test_results if r.get('success', False)),
        'failed': sum(1 for r in test_results if not r.get('success', False) and r.get('status') == 'failed'),
        'errors': sum(1 for r in test_results if not r.get('success', False) and r.get('status') == 'error'),
        'test_results': test_results,
        'duration': sum(r.get('duration', 0) for r in test_results),
    }
    
    processed_results['pass_rate'] = (
        processed_results['passed'] / processed_results['total_tests'] 
        if processed_results['total_tests'] > 0 else 0
    )
    
    # Record system metrics
    if metrics_data:
        processed_results['system_metrics'] = record_metrics(metrics_data)
    
    # Generate report if requested
    if report_dir:
        report_path = generate_report(processed_results, report_dir)
        processed_results['report_path'] = report_path
    
    # Record metrics if available
    if METRICS_AVAILABLE:
        # Record test durations
        for result in test_results:
            test_duration_histogram.record(
                result.get('duration', 0),
                {"test_name": result.get('name', 'unknown'), "status": result.get('status', 'unknown')}
            )
        
        # Record success and failure counts
        test_success_counter.add(processed_results['passed'])
        test_failure_counter.add(processed_results['failed'] + processed_results['errors'])
    
    return processed_results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run edge case tests for Gray Swan Arena")
    parser.add_argument(
        "--categories", "-c", 
        nargs="+", 
        choices=["all", "network", "data", "concurrency", "resource", "service"],
        default=["all"],
        help="Categories of edge case tests to run"
    )
    parser.add_argument(
        "--report-dir", "-r",
        type=str,
        default="./reports",
        help="Directory to save test reports to"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert categories to a set
    categories = set(args.categories)
    
    # Run tests
    start_time = time.time()
    results = run_edge_case_tests(categories, args.report_dir)
    total_time = time.time() - start_time
    
    # Print summary results
    print("\n" + "="*50)
    print("EDGE CASE TEST RESULTS")
    print("="*50)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Errors: {results['errors']}")
    print(f"Pass Rate: {results['pass_rate']*100:.2f}%")
    print(f"Total Duration: {results['duration']:.2f} seconds")
    print(f"Total Run Time: {total_time:.2f} seconds")
    
    if 'report_path' in results:
        print(f"\nDetailed report saved to: {results['report_path']}")
    
    # Return exit code based on test results
    return 0 if results['failed'] == 0 and results['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main()) 