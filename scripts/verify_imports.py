#!/usr/bin/env python3
"""Verification script to test all imports and class availability for Gray Swan Arena."""
import importlib
import sys
from typing import Any, Dict


def test_import(module_path: str) -> Dict[str, Any]:
    """Test importing a specific module."""
    try:
        module = importlib.import_module(module_path)
        return {"success": True, "module": module}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    # Core imports to test
    imports_to_test: list[Any] = [
        # Main package
        "cybersec_agents",
        # Gray Swan Arena components
        "cybersec_agents.grayswan",
        "cybersec_agents.grayswan.agents.recon_agent",
        "cybersec_agents.grayswan.agents.prompt_engineer_agent",
        "cybersec_agents.grayswan.agents.exploit_delivery_agent",
        "cybersec_agents.grayswan.agents.evaluation_agent",
        # Gray Swan Arena utilities
        "cybersec_agents.grayswan.utils.logging_utils",
        "cybersec_agents.grayswan.utils.agentops_utils",
        "cybersec_agents.grayswan.utils.browser_utils",
        "cybersec_agents.grayswan.utils.discord_utils",
        "cybersec_agents.grayswan.utils.visualization_utils",
        "cybersec_agents.grayswan.utils.model_utils",
    ]

    # Test each import
    failed_imports: list[Any] = []
    successful_imports: list[Any] = []

    print("Testing imports...")
    print("-" * 50)

    for import_path in imports_to_test:
        result: Any = test_import(import_path)
        if result["success"]:
            successful_imports.append(import_path)
            print(f"✓ {import_path}")
        else:
            failed_imports.append((import_path, result["error"]))
            print(f"✗ {import_path} - {result['error']}")

    print("-" * 50)
    print(
        f"Summary: {len(successful_imports)} successful, {len(failed_imports)} failed"
    )

    # Test class imports
    print("\nTesting class imports...")
    print("-" * 50)

    class_imports_to_test: list[Any] = [
        # Gray Swan Arena agents
        ("cybersec_agents", "ReconAgent"),
        ("cybersec_agents", "PromptEngineerAgent"),
        ("cybersec_agents", "ExploitDeliveryAgent"),
        ("cybersec_agents", "EvaluationAgent"),
        # Direct imports from grayswan
        ("cybersec_agents.grayswan", "ReconAgent"),
        ("cybersec_agents.grayswan", "PromptEngineerAgent"),
        ("cybersec_agents.grayswan", "ExploitDeliveryAgent"),
        ("cybersec_agents.grayswan", "EvaluationAgent"),
        # Main function
        ("cybersec_agents.grayswan", "main"),
    ]

    failed_class_imports: list[Any] = []
    successful_class_imports: list[Any] = []

    for module_path, class_name in class_imports_to_test:
        try:
            module = importlib.import_module(module_path)
            getattr(module, class_name)
            successful_class_imports.append((module_path, class_name))
            print(f"✓ from {module_path} import {class_name}")
        except (ImportError, AttributeError) as e:
            failed_class_imports.append((module_path, class_name, str(e)))
            print(f"✗ from {module_path} import {class_name} - {str(e)}")

    print("-" * 50)
    print(
        f"Summary: {len(successful_class_imports)} successful, {len(failed_class_imports)} failed"
    )

    # Exit with error code if any imports failed
    if failed_imports or failed_class_imports:
        sys.exit(1)
    else:
        print("\nAll imports verified successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
