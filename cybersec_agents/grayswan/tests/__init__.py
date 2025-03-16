"""
Tests package for Gray Swan Arena.

This package contains test modules and utilities for testing the Gray Swan Arena
functionality, including unit tests, integration tests, and edge case tests.
"""

# Import the edge case test framework
from .edge_cases.edge_case_framework import FailureSimulator, ConcurrencyTester, EdgeCaseTestRunner

# Import test registration functions
from .edge_cases.test_network_failures import register_tests as register_network_tests
from .edge_cases.test_data_corruption import register_tests as register_data_corruption_tests
from .edge_cases.test_concurrency_issues import register_tests as register_concurrency_tests
from .edge_cases.test_resource_exhaustion import register_tests as register_resource_tests
from .edge_cases.test_service_degradation import register_tests as register_service_tests
from .edge_cases.run_edge_case_tests import register_all_edge_case_tests

# Export the registration functions
__all__ = [
    'FailureSimulator',
    'ConcurrencyTester',
    'EdgeCaseTestRunner',
    'register_network_tests',
    'register_data_corruption_tests',
    'register_concurrency_tests',
    'register_resource_tests',
    'register_service_tests',
    'register_all_edge_case_tests'
] 