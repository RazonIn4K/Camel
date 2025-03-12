#!/usr/bin/env python3
"""Verification script to test all imports and class availability."""
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
    imports_to_test = [
        # Main package
        "cybersec_agents",
        # Agents
        "cybersec_agents.agents.network_security_agent",
        "cybersec_agents.agents.forensics_planner",
        "cybersec_agents.agents.network_anomaly_detector",
        "cybersec_agents.agents.wireless_mobile_assessor",
        # Analyzers
        "cybersec_agents.analyzers.codebase_analyzer",
        "cybersec_agents.analyzers.forensics_engine",
        "cybersec_agents.analyzers.wireless_scanner",
        # Core
        "cybersec_agents.core.service_wrapper",
        # Generators
        "cybersec_agents.generators.blog_generator",
        # Utils
        "cybersec_agents.utils.credentials",
    ]

    # Test each import
    failed_imports = []
    successful_imports = []

    print("Testing imports...")
    print("-" * 50)

    for import_path in imports_to_test:
        result = test_import(import_path)
        if result["success"]:
            successful_imports.append(import_path)
            print(f"✓ {import_path}")
        else:
            failed_imports.append((import_path, result["error"]))
            print(f"✗ {import_path} - Error: {result['error']}")

    print("\nTesting class instantiation...")
    print("-" * 50)

    # Test class instantiation if imports successful
    if "cybersec_agents" in successful_imports:
        try:
            from cybersec_agents import (
                CodebaseAnalyzerAgent,
                CyberSecurityBlogGenerator,
                CyberSecurityService,
                ForensicsPlanner,
                NetworkAnomalyDetector,
                NetworkSecurityAgent,
                WirelessMobileSecurityAssessor,
            )

            # Try instantiating each class
            classes_to_test = [
                ("NetworkSecurityAgent", NetworkSecurityAgent),
                ("CyberSecurityService", CyberSecurityService),
                ("ForensicsPlanner", ForensicsPlanner),
                ("NetworkAnomalyDetector", NetworkAnomalyDetector),
                ("WirelessMobileSecurityAssessor", WirelessMobileSecurityAssessor),
                ("CodebaseAnalyzerAgent", CodebaseAnalyzerAgent),
                ("CyberSecurityBlogGenerator", CyberSecurityBlogGenerator),
            ]

            for class_name, class_type in classes_to_test:
                try:
                    class_type()
                    print(f"✓ {class_name} successfully instantiated")
                except Exception as e:
                    print(f"✗ {class_name} instantiation failed - Error: {str(e)}")

        except ImportError as e:
            print(f"Failed to import classes from cybersec_agents: {str(e)}")

    print("\nSummary:")
    print("-" * 50)
    print(f"Successful imports: {len(successful_imports)}")
    print(f"Failed imports: {len(failed_imports)}")

    if failed_imports:
        print("\nFailed imports details:")
        for import_path, error in failed_imports:
            print(f"- {import_path}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
