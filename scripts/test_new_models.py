#!/usr/bin/env python3
"""
Test script to verify that the new models are properly integrated.
This script checks if the models are correctly recognized and can be used.
"""

import os
import sys
from typing import Dict, List, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from camel.types import ModelPlatformType, ModelType
from cybersec_agents.grayswan.utils.model_utils import (
    get_api_key,
    get_model_name_from_type,
    get_model_platform,
    get_model_type,
)

def check_model_type_enum() -> bool:
    """Check if the new model types are in the ModelType enum."""
    required_types = [
        "GEMINI_2_PRO",
        "CLAUDE_3_7_SONNET",
        "O3_MINI",
        "GPT_4O",
    ]
    
    available_types = [t.name for t in ModelType]
    missing_types = [t for t in required_types if t not in available_types]
    
    if missing_types:
        print(f"❌ Missing model types: {', '.join(missing_types)}")
        return False
    
    print("✅ All required model types are available")
    return True

def check_model_platform_enum() -> bool:
    """Check if the required model platforms are in the ModelPlatformType enum."""
    required_platforms = [
        "GOOGLE",
        "ANTHROPIC",
        "OPENAI",
    ]
    
    available_platforms = [p.name for p in ModelPlatformType]
    missing_platforms = [p for p in required_platforms if p not in available_platforms]
    
    if missing_platforms:
        print(f"❌ Missing model platforms: {', '.join(missing_platforms)}")
        return False
    
    print("✅ All required model platforms are available")
    return True

def check_model_name_mapping() -> bool:
    """Check if the model name mapping functions work correctly."""
    test_cases = [
        (ModelType.GEMINI_2_PRO, ModelPlatformType.GOOGLE, "gemini-2.0-pro-exp-02-05"),
        (ModelType.CLAUDE_3_7_SONNET, ModelPlatformType.ANTHROPIC, "claude-3-7-sonnet"),
        (ModelType.O3_MINI, ModelPlatformType.OPENAI, "o3-mini"),
        (ModelType.GPT_4O, ModelPlatformType.OPENAI, "gpt-4o"),
    ]
    
    all_passed = True
    
    for model_type, model_platform, expected_name in test_cases:
        try:
            actual_name = get_model_name_from_type(model_type, model_platform)
            if actual_name != expected_name:
                print(f"❌ Model name mismatch for {model_type.name}: expected '{expected_name}', got '{actual_name}'")
                all_passed = False
            else:
                print(f"✅ Model name mapping works for {model_type.name}")
                
            # Test reverse mapping
            reverse_type = get_model_type(expected_name)
            if reverse_type != model_type:
                print(f"❌ Reverse model type mapping failed for {expected_name}: expected {model_type.name}, got {reverse_type.name}")
                all_passed = False
            else:
                print(f"✅ Reverse model type mapping works for {expected_name}")
                
            reverse_platform = get_model_platform(expected_name)
            if reverse_platform != model_platform:
                print(f"❌ Reverse model platform mapping failed for {expected_name}: expected {model_platform.name}, got {reverse_platform.name}")
                all_passed = False
            else:
                print(f"✅ Reverse model platform mapping works for {expected_name}")
                
        except Exception as e:
            print(f"❌ Error testing model name mapping for {model_type.name}: {str(e)}")
            all_passed = False
    
    return all_passed

def check_api_keys() -> bool:
    """Check if the required API keys are set."""
    required_keys = [
        (ModelType.GEMINI_2_PRO, ModelPlatformType.GOOGLE),
        (ModelType.CLAUDE_3_7_SONNET, ModelPlatformType.ANTHROPIC),
        (ModelType.O3_MINI, ModelPlatformType.OPENAI),
        (ModelType.GPT_4O, ModelPlatformType.OPENAI),
    ]
    
    all_passed = True
    
    for model_type, model_platform in required_keys:
        try:
            # We don't print the actual key for security reasons
            api_key = get_api_key(model_type, model_platform)
            if api_key:
                print(f"✅ API key found for {model_platform.name}")
            else:
                print(f"❌ API key not found for {model_platform.name}")
                all_passed = False
        except Exception as e:
            print(f"❌ Error checking API key for {model_platform.name}: {str(e)}")
            all_passed = False
    
    return all_passed

def main() -> None:
    """Run all tests."""
    print("Testing new model integration...\n")
    
    tests = [
        ("Model Type Enum", check_model_type_enum),
        ("Model Platform Enum", check_model_platform_enum),
        ("Model Name Mapping", check_model_name_mapping),
        ("API Keys", check_api_keys),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n=== Testing {name} ===")
        result = test_func()
        results.append((name, result))
        print(f"=== {name}: {'PASSED' if result else 'FAILED'} ===")
    
    print("\n=== Summary ===")
    all_passed = all(result for _, result in results)
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {name}")
    
    if all_passed:
        print("\n🎉 All tests passed! The new models are properly integrated.")
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()