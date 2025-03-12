#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

import pytest


def run_tests(providers: List[str], verbose: bool = False) -> bool:
    """Run integration tests for specified providers."""
    test_args = [
        "tests/test_model_integration.py",
        "-v" if verbose else "",
        "-m",
        "integration",
    ]

    if providers:
        provider_params = [f"provider=={p}" for p in providers]
        test_args.extend(["-k", " or ".join(provider_params)])

    return pytest.main(test_args) == 0


def validate_environment() -> List[str]:
    """Validate environment variables and return available providers."""
    available_providers = []
    required_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": ["GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT_ID"],
    }

    for provider, vars in required_vars.items():
        if isinstance(vars, str):
            vars = [vars]
        if all(os.environ.get(var) for var in vars):
            available_providers.append(provider)

    return available_providers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI model integration tests")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["anthropic", "openai", "google"],
        help="Specific providers to test",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    # Validate environment
    available_providers = validate_environment()
    if not available_providers:
        print(
            "No provider credentials found. Please set required environment variables."
        )
        sys.exit(1)

    # Run tests for specified or all available providers
    test_providers = args.providers or available_providers
    if run_tests(test_providers, args.verbose):
        print("All integration tests passed successfully!")
        sys.exit(0)
    else:
        print("Some tests failed. Check the output above for details.")
        sys.exit(1)
