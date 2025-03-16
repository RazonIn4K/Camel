"""
Advanced visualization utilities for the Gray Swan Arena.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import logging
from camel.agents import ChatAgent

from ..utils.logging_utils import setup_logging

logger = setup_logging("advanced_visualization_utils")

def create_attack_pattern_visualization(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent
) -> Dict[str, Any]:
    """Create a visualization of attack patterns.

    Args:
        results: List of test results to analyze
        output_dir: Directory to save the visualization
        evaluation_agent: Agent to use for evaluation

    Returns:
        Dict containing the visualization data
    """
    try:
        # Group results by attack pattern
        pattern_data = {}
        for result in results:
            pattern = result.get("attack_pattern", "Unknown")
            success = result.get("success", False)
            
            if pattern not in pattern_data:
                pattern_data[pattern] = {
                    "total": 0,
                    "success": 0,
                    "examples": []
                }
            
            pattern_data[pattern]["total"] += 1
            if success:
                pattern_data[pattern]["success"] += 1
            pattern_data[pattern]["examples"].append({
                "prompt": result.get("prompt", ""),
                "response": result.get("response", ""),
                "success": success,
                "response_time": result.get("response_time", 0)
            })
        
        # Calculate success rates
        for pattern in pattern_data:
            data = pattern_data[pattern]
            data["success_rate"] = data["success"] / data["total"] if data["total"] > 0 else 0
        
        # Create visualization data
        visualization_data = {
            "patterns": pattern_data,
            "summary": {
                "total_patterns": len(pattern_data),
                "total_attempts": sum(data["total"] for data in pattern_data.values()),
                "total_successes": sum(data["success"] for data in pattern_data.values()),
                "overall_success_rate": sum(data["success"] for data in pattern_data.values()) / 
                                     sum(data["total"] for data in pattern_data.values()) if 
                                     sum(data["total"] for data in pattern_data.values()) > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return visualization_data

    except Exception as e:
        logger.error(f"Failed to create attack pattern visualization: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def create_prompt_similarity_network(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent
) -> Dict[str, Any]:
    """Create a network visualization of prompt similarities.

    Args:
        results: List of test results to analyze
        output_dir: Directory to save the network
        evaluation_agent: Agent to use for evaluation

    Returns:
        Dict containing the network data
    """
    try:
        # Group results by prompt type
        prompt_data = {}
        for result in results:
            prompt_type = result.get("prompt_type", "Unknown")
            success = result.get("success", False)
            
            if prompt_type not in prompt_data:
                prompt_data[prompt_type] = {
                    "total": 0,
                    "success": 0,
                    "examples": []
                }
            
            prompt_data[prompt_type]["total"] += 1
            if success:
                prompt_data[prompt_type]["success"] += 1
            prompt_data[prompt_type]["examples"].append({
                "prompt": result.get("prompt", ""),
                "response": result.get("response", ""),
                "success": success,
                "response_time": result.get("response_time", 0)
            })
        
        # Calculate success rates
        for prompt_type in prompt_data:
            data = prompt_data[prompt_type]
            data["success_rate"] = data["success"] / data["total"] if data["total"] > 0 else 0
        
        # Create network data
        network_data = {
            "prompt_types": prompt_data,
            "summary": {
                "total_types": len(prompt_data),
                "total_attempts": sum(data["total"] for data in prompt_data.values()),
                "total_successes": sum(data["success"] for data in prompt_data.values()),
                "overall_success_rate": sum(data["success"] for data in prompt_data.values()) / 
                                     sum(data["total"] for data in prompt_data.values()) if 
                                     sum(data["total"] for data in prompt_data.values()) > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return network_data

    except Exception as e:
        logger.error(f"Failed to create prompt similarity network: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def create_success_prediction_model(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent
) -> Dict[str, Any]:
    """Create a model for predicting attack success.

    Args:
        results: List of test results to train on
        output_dir: Directory to save the model
        evaluation_agent: Agent to use for evaluation

    Returns:
        Dict containing the model data
    """
    try:
        # Extract features from results
        features = []
        labels = []
        for result in results:
            feature = {
                "prompt_length": len(result.get("prompt", "")),
                "response_time": result.get("response_time", 0),
                "has_keywords": bool(result.get("keywords", [])),
                "has_context": bool(result.get("context", "")),
                "has_examples": bool(result.get("examples", [])),
                "has_constraints": bool(result.get("constraints", [])),
                "has_instructions": bool(result.get("instructions", [])),
                "has_metadata": bool(result.get("metadata", {}))
            }
            features.append(feature)
            labels.append(1 if result.get("success", False) else 0)
        
        # Create model data
        model_data = {
            "features": features,
            "labels": labels,
            "summary": {
                "total_samples": len(features),
                "positive_samples": sum(labels),
                "negative_samples": len(labels) - sum(labels),
                "class_balance": sum(labels) / len(labels) if len(labels) > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return model_data

    except Exception as e:
        logger.error(f"Failed to create success prediction model: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def create_interactive_dashboard(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent
) -> Dict[str, Any]:
    """Create an interactive dashboard for visualization.

    Args:
        results: List of test results to visualize
        output_dir: Directory to save the dashboard
        evaluation_agent: Agent to use for evaluation

    Returns:
        Dict containing the dashboard data
    """
    try:
        # Group results by various dimensions
        dashboard_data = {
            "by_model": {},
            "by_prompt_type": {},
            "by_attack_vector": {},
            "by_success": {"success": 0, "failure": 0},
            "response_times": [],
            "summary": {
                "total_attempts": len(results),
                "total_successes": sum(1 for r in results if r.get("success", False)),
                "success_rate": sum(1 for r in results if r.get("success", False)) / len(results) if len(results) > 0 else 0,
                "avg_response_time": sum(r.get("response_time", 0) for r in results) / len(results) if len(results) > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Process results
        for result in results:
            # Group by model
            model = result.get("model_name", "Unknown")
            if model not in dashboard_data["by_model"]:
                dashboard_data["by_model"][model] = {"success": 0, "total": 0}
            dashboard_data["by_model"][model]["total"] += 1
            if result.get("success", False):
                dashboard_data["by_model"][model]["success"] += 1
            
            # Group by prompt type
            prompt_type = result.get("prompt_type", "Unknown")
            if prompt_type not in dashboard_data["by_prompt_type"]:
                dashboard_data["by_prompt_type"][prompt_type] = {"success": 0, "total": 0}
            dashboard_data["by_prompt_type"][prompt_type]["total"] += 1
            if result.get("success", False):
                dashboard_data["by_prompt_type"][prompt_type]["success"] += 1
            
            # Group by attack vector
            attack_vector = result.get("attack_vector", "Unknown")
            if attack_vector not in dashboard_data["by_attack_vector"]:
                dashboard_data["by_attack_vector"][attack_vector] = {"success": 0, "total": 0}
            dashboard_data["by_attack_vector"][attack_vector]["total"] += 1
            if result.get("success", False):
                dashboard_data["by_attack_vector"][attack_vector]["success"] += 1
            
            # Group by success
            if result.get("success", False):
                dashboard_data["by_success"]["success"] += 1
            else:
                dashboard_data["by_success"]["failure"] += 1
            
            # Collect response times
            dashboard_data["response_times"].append(result.get("response_time", 0))
        
        return dashboard_data

    except Exception as e:
        logger.error(f"Failed to create interactive dashboard: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def create_advanced_evaluation_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent,
    reasoning_agent: ChatAgent
) -> Dict[str, Any]:
    """Create an advanced evaluation report with detailed analysis.
    
    Args:
        results: List of test results to evaluate
        output_dir: Directory to save the report
        evaluation_agent: Agent to use for evaluation
        reasoning_agent: Agent to use for reasoning tasks
        
    Returns:
        Dict containing the advanced evaluation report
    """
    return {
        "status": "not_implemented",
        "message": "Advanced evaluation report generation is not yet implemented",
        "timestamp": datetime.now().isoformat()
    }

def create_evaluation_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
    evaluation_agent: ChatAgent,
    reasoning_agent: ChatAgent
) -> Dict[str, Any]:
    """Create a basic evaluation report.

    Args:
        results: List of test results to evaluate
        output_dir: Directory to save the report
        evaluation_agent: Agent to use for evaluation
        reasoning_agent: Agent to use for reasoning tasks

    Returns:
        Dict containing the evaluation report
    """
    return {
        "status": "not_implemented",
        "message": "Basic evaluation report generation is not yet implemented",
        "timestamp": datetime.now().isoformat()
    }
