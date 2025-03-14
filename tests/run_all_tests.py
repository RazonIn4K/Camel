#!/usr/bin/env python3
"""Script to run all Gray Swan Arena tests."""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).parent.parent))

# Define available test modules
TEST_MODULES: dict[str, Any] = {
    "camel": {
        "script": "test_camel_integration.py",
        "description": "Test integration with Camel AI framework",
        "requires_api_key": ["OPENAI_API_KEY"],
    },
    "agentops": {
        "script": "test_agentops_integration.py",
        "description": "Test integration with AgentOps monitoring",
        "requires_api_key": ["AGENTOPS_API_KEY", "OPENAI_API_KEY"],
    },
    "e2e": {
        "script": "test_grayswan_e2e.py",
        "description": "Run end-to-end test of Gray Swan Arena",
        "requires_api_key": ["OPENAI_API_KEY"],
    },
    "browser": {
        "script": "test_browser_automation.py",
        "description": "Test browser automation features",
        "requires_api_key": [],
    },
    "discord": {
        "script": "test_discord_integration.py",
        "description": "Test Discord integration features",
        "requires_api_key": ["DISCORD_TOKEN"],
    },
    "benchmark": {
        "script": "benchmark_agents.py",
        "description": "Benchmark Gray Swan Arena agents",
        "requires_api_key": ["OPENAI_API_KEY"],
        "is_benchmark": True,
    },
}


def setup_environment() -> Dict[str, bool]:
    """Set up the test environment and check API keys."""
    print("Setting up test environment...")

    # Create output directory
    output_dir = Path("tests/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for required API keys
    api_keys: dict[str, Any] = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") is not None,
        "AGENTOPS_API_KEY": os.environ.get("AGENTOPS_API_KEY") is not None,
        "DISCORD_TOKEN": os.environ.get("DISCORD_TOKEN") is not None,
    }

    # Print available API keys
    print("\nAPI Key Status:")
    for key, available in api_keys.items():
        status: str = "✅ Available" if available else "❌ Not Available"
        print(f"- {key}: {status}")

    return api_keys


def validate_test_modules(modules: List[str], api_keys: Dict[str, bool]) -> List[str]:
    """Validate the test modules and filter out those that can't run."""
    valid_modules: list[Any] = []
    invalid_modules: list[Any] = []

    for module in modules:
        if module not in TEST_MODULES:
            print(f"⚠️ Unknown test module: {module}")
            continue

        # Check if required API keys are available
        missing_keys: list[Any] = []
        for key in TEST_MODULES[module]["requires_api_key"]:
            if not api_keys.get(key, False):
                missing_keys.append(key)

        if missing_keys:
            print(
                f"⚠️ Cannot run '{module}' test: Missing API keys: {', '.join(missing_keys)}"
            )
            invalid_modules.append(module)
        else:
            valid_modules.append(module)

    return valid_modules


def run_test_module(module: str) -> bool:
    """Run a specific test module."""
    module_info = TEST_MODULES[module]
    script_path = Path(__file__).parent / module_info["script"]

    print("\n" + "=" * 80)
    print(f"RUNNING TEST MODULE: {module}")
    print(f"Description: {module_info['description']}")
    print("=" * 80)

    if not script_path.exists():
        print(f"❌ Test script not found: {script_path}")
        return False

    try:
        result: Any = subprocess.run(
            [sys.executable, str(script_path)], check=False, capture_output=False
        )

        success = result.returncode == 0
        status: str = "✅ PASSED" if success else "❌ FAILED"

        print("\n" + "=" * 80)
        print(f"TEST MODULE RESULT: {module} - {status}")
        print(f"Exit code: {result.returncode}")
        print("=" * 80)

        return success

    except Exception as e:
        print(f"\n❌ Error running test module '{module}': {e}")
        return False


def run_benchmark(
    topics: Optional[List[str]] = None, models: Optional[List[str]] = None
) -> bool:
    """Run the benchmark script with specified options."""
    script_path = Path(__file__).parent / TEST_MODULES["benchmark"]["script"]

    if not script_path.exists():
        print(f"❌ Benchmark script not found: {script_path}")
        return False

    # Build command with arguments
    command: list[Any] = [sys.executable, str(script_path)]

    if topics:
        command.extend(["--topics"] + topics)

    if models:
        command.extend(["--eval-models"] + models)
        command.extend(["--target-models"] + models)

    print("\n" + "=" * 80)
    print("RUNNING BENCHMARK")
    print(f"Topics: {topics if topics else 'Default'}")
    print(f"Models: {models if models else 'Default'}")
    print("=" * 80)

    try:
        result: Any = subprocess.run(command, check=False, capture_output=False)

        success = result.returncode == 0
        status: str = "✅ COMPLETED" if success else "❌ FAILED"

        print("\n" + "=" * 80)
        print(f"BENCHMARK RESULT: {status}")
        print(f"Exit code: {result.returncode}")
        print("=" * 80)

        return success

    except Exception as e:
        print(f"\n❌ Error running benchmark: {e}")
        return False


def run_all_tests(modules: List[str]) -> Dict[str, bool]:
    """Run all specified test modules and return results."""
    results: list[Any] = {}

    for module in modules:
        results[module] = run_test_module(module)

    return results


def generate_report(results: Dict[str, bool]) -> str:
    """Generate a summary report of test results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate statistics
    total: int = len(results)
    passed = sum(1 for success in results.values() if success)
    failed = total - passed

    # Build report
    report: list[Any] = [
        "# Gray Swan Arena Test Report",
        f"\nGenerated: {timestamp}",
        f"\n## Summary",
        f"\n- Total Tests: {total}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Success Rate: {(passed/total)*100:.1f}%",
        f"\n## Test Results",
    ]

    for module, success in results.items():
        status: str = "✅ PASSED" if success else "❌ FAILED"
        description = TEST_MODULES[module]["description"]
        report.append(f"\n- {module}: {status}")
        report.append(f"  - {description}")

    return "\n".join(report)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Gray Swan Arena tests")

    parser.add_argument(
        "--modules",
        nargs="+",
        choices=list(TEST_MODULES.keys()) + ["all"],
        default=["all"],
        help="Test modules to run",
    )

    parser.add_argument(
        "--benchmark-only", action="store_true", help="Run only the benchmark test"
    )

    parser.add_argument(
        "--benchmark-topics", nargs="+", help="Topics to use for benchmarking"
    )

    parser.add_argument(
        "--benchmark-models", nargs="+", help="Models to use for benchmarking"
    )

    return parser.parse_args()


def main():
    """Main entry point for the test runner."""
    args = parse_args()

    print("=" * 80)
    print("GRAY SWAN ARENA TEST RUNNER")
    print("=" * 80)

    # Set up environment
    api_keys = setup_environment()

    # Determine which modules to run
    if args.benchmark_only:
        modules: list[Any] = ["benchmark"]
    elif "all" in args.modules:
        modules = list(TEST_MODULES.keys())
    else:
        modules = args.modules

    # Validate modules
    valid_modules = validate_test_modules(modules, api_keys)

    if not valid_modules:
        print("\n❌ No valid test modules to run.")
        sys.exit(1)

    print(
        f"\nPreparing to run {len(valid_modules)} test modules: {', '.join(valid_modules)}"
    )

    # Run tests
    if args.benchmark_only:
        success = run_benchmark(args.benchmark_topics, args.benchmark_models)
        results = {"benchmark": success}
    else:
        results: list[Any] = run_all_tests(valid_modules)

    # Generate and save report
    report = generate_report(results)

    report_file = Path("tests/output/test_report.md")
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\nTest report saved to {report_file}")

    # Print overall result
    if all(results.values()):
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED. See report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
