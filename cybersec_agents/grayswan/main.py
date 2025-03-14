"""
Main module for Gray Swan Arena.

This module provides the main functionality for running Gray Swan Arena agents
and executing the full pipeline for AI safety evaluation.
"""

import os
import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import agents
from cybersec_agents.grayswan.agents.recon_agent import ReconAgent
from cybersec_agents.grayswan.agents.prompt_engineer_agent import PromptEngineerAgent
from cybersec_agents.grayswan.agents.exploit_delivery_agent import ExploitDeliveryAgent
from cybersec_agents.grayswan.agents.evaluation_agent import EvaluationAgent

# Import utilities
from cybersec_agents.grayswan.utils.agentops_utils import (
    initialize_agentops,
    log_agentops_event,
    start_agentops_session,
)
from cybersec_agents.grayswan.utils.logging_utils import setup_logging
from cybersec_agents.grayswan.utils.model_manager import ModelManager

# Set up logging
logger = setup_logging("grayswan_main")


async def run_parallel_reconnaissance(
    target_model: str,
    target_behavior: str,
    output_dir: str = "./reports",
    model_name: str = "gpt-4",
    reasoning_model: Optional[str] = None,
    backup_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run reconnaissance tasks in parallel.
    
    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save reports
        model_name: Name of the model to use for the agent
        reasoning_model: Model to use for reasoning tasks
        backup_model: Backup model to use if primary fails
        
    Returns:
        Dictionary containing reconnaissance results and report path
    """
    logger.info(f"Starting parallel reconnaissance for {target_model} - {target_behavior}")
    
    # Log phase start
    log_agentops_event(
        "phase_started",
        {
            "phase": "parallel_reconnaissance",
            "target_model": target_model,
            "target_behavior": target_behavior,
            "model_name": model_name,
            "reasoning_model": reasoning_model,
            "backup_model": backup_model,
        },
    )
    
    # Initialize the ReconAgent
    recon_agent = ReconAgent(
        output_dir=output_dir, 
        model_name=model_name,
        reasoning_model=reasoning_model,
        backup_model=backup_model,
    )
    
    # Create tasks for concurrent execution
    web_task = asyncio.create_task(
        asyncio.to_thread(
            recon_agent.run_web_search,
            target_model,
            target_behavior
        )
    )
    
    discord_task = asyncio.create_task(
        asyncio.to_thread(
            recon_agent.run_discord_search,
            target_model,
            target_behavior
        )
    )
    
    # Wait for all tasks to complete
    web_results, discord_results = await asyncio.gather(web_task, discord_task)
    
    # Generate and save report
    report = recon_agent.generate_report(
        target_model=target_model,
        target_behavior=target_behavior,
        web_results=web_results,
        discord_results=discord_results,
    )
    
    report_path = recon_agent.save_report(
        report=report, target_model=target_model, target_behavior=target_behavior
    )
    
    logger.info(f"Parallel reconnaissance completed, report saved to {report_path}")
    
    # Log completion event
    log_agentops_event(
        "phase_completed",
        {
            "phase": "reconnaissance",
            "target_model": target_model,
            "target_behavior": target_behavior,
            "status": "completed",
            "report_path": report_path,
        },
    )
    
    return {
        "report": report,
        "path": report_path,
        "web_results": web_results,
        "discord_results": discord_results
    }


def run_reconnaissance(
    target_model: str,
    target_behavior: str,
    output_dir: str = "./reports",
    model_name: str = "gpt-4",
    reasoning_model: Optional[str] = None,
    backup_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the reconnaissance phase of the Gray Swan Arena pipeline.

    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save reports
        model_name: Name of the model to use for the agent
        reasoning_model: Model to use for reasoning tasks
        backup_model: Backup model to use if primary fails

    Returns:
        Dictionary containing the reconnaissance report
    """
    logger.info(f"Starting reconnaissance phase for {target_model} - {target_behavior}")

    # Log phase start
    log_agentops_event(
        "phase_started",
        {
            "phase": "reconnaissance",
            "target_model": target_model,
            "target_behavior": target_behavior,
            "model_name": model_name,
            "reasoning_model": reasoning_model,
            "backup_model": backup_model,
        },
    )

    try:
        # Initialize the ReconAgent
        recon_agent = ReconAgent(
            output_dir=output_dir, 
            model_name=model_name,
            reasoning_model=reasoning_model,
            backup_model=backup_model,
        )

        # Run web search
        web_results = recon_agent.run_web_search(
            target_model=target_model, target_behavior=target_behavior
        )

        # Run Discord search
        discord_results = recon_agent.run_discord_search(
            target_model=target_model, target_behavior=target_behavior
        )

        # Generate and save report
        report = recon_agent.generate_report(
            target_model=target_model,
            target_behavior=target_behavior,
            web_results=web_results,
            discord_results=discord_results,
        )

        report_path = recon_agent.save_report(
            report=report, target_model=target_model, target_behavior=target_behavior
        )

        logger.info(f"Reconnaissance phase completed, report saved to {report_path}")

        # Log completion event
        log_agentops_event(
            "phase_completed",
            {
                "phase": "reconnaissance",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "completed",
                "report_path": report_path,
            },
        )

        return report

    except Exception as e:
        logger.error(f"Reconnaissance phase failed: {str(e)}")

        # Log error event
        log_agentops_event(
            "phase_error",
            {
                "phase": "reconnaissance",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "failed",
                "error": str(e),
            },
        )

        return {
            "error": str(e),
            "target_model": target_model,
            "target_behavior": target_behavior,
            "timestamp": datetime.now().isoformat(),
        }


def run_prompt_engineering(
    target_model: str,
    target_behavior: str,
    recon_report: Dict[str, Any],
    output_dir: str = "./prompts",
    model_name: str = "gpt-4",
    reasoning_model: Optional[str] = None,
    backup_model: Optional[str] = None,
    num_prompts: int = 10,
) -> Dict[str, Any]:
    """
    Run the prompt engineering phase of the Gray Swan Arena pipeline.
    
    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        recon_report: Report from the reconnaissance phase
        output_dir: Directory to save prompts
        model_name: Name of the model to use for the agent
        reasoning_model: Model to use for reasoning tasks
        backup_model: Backup model to use if primary fails
        num_prompts: Number of prompts to generate
        
    Returns:
        Dictionary containing the generated prompts and file path
    """
    logger.info(f"Starting prompt engineering phase for {target_model} - {target_behavior}")
    
    # Log phase start
    log_agentops_event(
        "phase_started",
        {
            "phase": "prompt_engineering",
            "target_model": target_model,
            "target_behavior": target_behavior,
            "model_name": model_name,
            "reasoning_model": reasoning_model,
            "backup_model": backup_model,
        },
    )
    
    try:
        # Initialize the PromptEngineerAgent
        prompt_agent = PromptEngineerAgent(
            output_dir=output_dir, 
            model_name=model_name,
            reasoning_model=reasoning_model,
            backup_model=backup_model,
        )
        
        # Generate prompts
        prompts = prompt_agent.generate_prompts(
            target_model=target_model,
            target_behavior=target_behavior,
            recon_report=recon_report,
            num_prompts=num_prompts,
        )
        
        # Save prompts
        prompts_path = prompt_agent.save_prompts(
            prompts=prompts,
            target_model=target_model,
            target_behavior=target_behavior,
        )
        
        logger.info(f"Prompt engineering phase completed, {len(prompts)} prompts saved to {prompts_path}")
        
        # Log completion event
        log_agentops_event(
            "phase_completed",
            {
                "phase": "prompt_engineering",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "completed",
                "num_prompts": len(prompts),
                "prompts_path": prompts_path,
            },
        )
        
        return {
            "prompts": prompts,
            "path": prompts_path,
        }
        
    except Exception as e:
        logger.error(f"Prompt engineering phase failed: {str(e)}")
        
        # Log error event
        log_agentops_event(
            "phase_error",
            {
                "phase": "prompt_engineering",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "failed",
                "error": str(e),
            },
        )
        
        return {
            "error": str(e),
            "target_model": target_model,
            "target_behavior": target_behavior,
            "timestamp": datetime.now().isoformat(),
            "prompts": [],
        }


async def run_parallel_exploits(
    prompts: List[str],
    target_model: str,
    target_behavior: str,
    output_dir: str = "./exploits",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    method: str = "api",
    max_concurrent: int = 3,
) -> Dict[str, Any]:
    """
    Run exploit delivery in parallel batches.
    
    Args:
        prompts: List of prompts to test
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save exploit results
        model_name: Name of the model to use for the agent
        backup_model: Backup model to use if primary fails
        method: Method to use (api, web)
        max_concurrent: Maximum number of concurrent operations
        
    Returns:
        Dictionary containing exploit results and file path
    """
    logger.info(f"Starting parallel exploit delivery for {target_model} with {len(prompts)} prompts")
    
    # Initialize the ExploitDeliveryAgent
    exploit_agent = ExploitDeliveryAgent(output_dir=output_dir, model_name=model_name)
    
    # Split prompts into batches
    batch_size = max_concurrent
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    
    all_results = []
    
    for batch_idx, batch in enumerate(batches):
        logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
        
        # Create tasks for concurrent execution
        tasks = []
        for prompt in batch:
            task = asyncio.create_task(
                asyncio.to_thread(
                    _execute_single_prompt,
                    exploit_agent,
                    prompt,
                    target_model,
                    target_behavior,
                    method
                )
            )
            tasks.append(task)
        
        # Wait for batch to complete
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        # Small delay between batches to avoid rate limiting
        if batch_idx < len(batches) - 1:
            await asyncio.sleep(2)
    
    # Save results
    results_path = exploit_agent.save_results(
        results=all_results,
        target_model=target_model,
        target_behavior=target_behavior,
    )
    
    logger.info(f"Parallel exploit delivery completed, results saved to {results_path}")
    
    # Log completion event
    log_agentops_event(
        "phase_completed",
        {
            "phase": "exploit_delivery",
            "target_model": target_model,
            "target_behavior": target_behavior,
            "status": "completed",
            "num_results": len(all_results),
            "results_path": results_path,
        },
    )
    
    return {
        "results": all_results,
        "path": results_path,
    }


def _execute_single_prompt(
    agent: ExploitDeliveryAgent,
    prompt: str,
    target_model: str,
    target_behavior: str,
    method: str
) -> Dict[str, Any]:
    """
    Execute a single prompt and return the result.
    
    Args:
        agent: The ExploitDeliveryAgent instance
        prompt: The prompt to test
        target_model: The target model
        target_behavior: The behavior to target
        method: Method to use (api, web)
        
    Returns:
        Dictionary containing the result
    """
    try:
        # Execute the prompt based on the method
        if method == "api":
            response = agent._execute_via_api(prompt, target_model)
        elif method == "web":
            response = agent._execute_via_web(prompt, target_model)
        else:
            response = agent._execute_via_api(prompt, target_model)
        
        # Analyze the response
        success, reason = agent._analyze_response(response, target_behavior)
        
        return {
            "prompt": prompt,
            "target_model": target_model,
            "target_behavior": target_behavior,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "reason": reason,
            "response": response,
            "error": None
        }
    except Exception as e:
        return {
            "prompt": prompt,
            "target_model": target_model,
            "target_behavior": target_behavior,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "response": None,
            "error": str(e)
        }


def run_exploit_delivery(
    prompts: List[str],
    target_model: str,
    target_behavior: str,
    output_dir: str = "./exploits",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    method: str = "api",
) -> Dict[str, Any]:
    """
    Run the exploit delivery phase of the Gray Swan Arena pipeline.
    
    Args:
        prompts: List of prompts to test
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save exploit results
        model_name: Name of the model to use for the agent
        backup_model: Backup model to use if primary fails
        method: Method to use (api, web)
        
    Returns:
        Dictionary containing the exploit results
    """
    logger.info(f"Starting exploit delivery phase for {target_model} - {target_behavior}")
    
    # Log phase start
    log_agentops_event(
        "phase_started",
        {
            "phase": "exploit_delivery",
            "target_model": target_model,
            "target_behavior": target_behavior,
            "method": method,
        },
    )
    
    try:
        # Initialize the ExploitDeliveryAgent
        exploit_agent = ExploitDeliveryAgent(output_dir=output_dir, model_name=model_name)
        
        # Run prompts
        results = exploit_agent.run_prompts(
            prompts=prompts,
            target_model=target_model,
            target_behavior=target_behavior,
            method=method,
        )
        
        # Save results
        results_path = exploit_agent.save_results(
            results=results,
            target_model=target_model,
            target_behavior=target_behavior,
        )
        
        logger.info(f"Exploit delivery phase completed, results saved to {results_path}")
        
        # Log completion event
        log_agentops_event(
            "phase_completed",
            {
                "phase": "exploit_delivery",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "completed",
                "num_results": len(results),
                "results_path": results_path,
            },
        )
        
        return {
            "results": results,
            "path": results_path,
        }
        
    except Exception as e:
        logger.error(f"Exploit delivery phase failed: {str(e)}")
        
        # Log error event
        log_agentops_event(
            "phase_error",
            {
                "phase": "exploit_delivery",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "failed",
                "error": str(e),
            },
        )
        
        return {
            "error": str(e),
            "target_model": target_model,
            "target_behavior": target_behavior,
            "timestamp": datetime.now().isoformat(),
            "results": [],
        }


def run_evaluation(
    exploit_results: List[Dict[str, Any]],
    target_model: str,
    target_behavior: str,
    output_dir: str = "./evaluations",
    model_name: str = "gpt-4",
    reasoning_model: Optional[str] = None,
    backup_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the evaluation phase of the Gray Swan Arena pipeline.
    
    Args:
        exploit_results: Results from the exploit delivery phase
        target_model: The target model that was tested
        target_behavior: The behavior that was targeted
        output_dir: Directory to save evaluation results
        model_name: Name of the model to use for the agent
        reasoning_model: Model to use for reasoning tasks
        backup_model: Backup model to use if primary fails
        
    Returns:
        Dictionary containing the evaluation results
    """
    logger.info(f"Starting evaluation phase for {target_model} - {target_behavior}")
    
    # Log phase start
    log_agentops_event(
        "phase_started",
        {
            "phase": "evaluation",
            "target_model": target_model,
            "target_behavior": target_behavior,
            "model_name": model_name,
            "reasoning_model": reasoning_model,
            "backup_model": backup_model,
        },
    )
    
    try:
        # Initialize the EvaluationAgent
        eval_agent = EvaluationAgent(
            output_dir=output_dir, 
            model_name=model_name,
            reasoning_model=reasoning_model,
            backup_model=backup_model,
        )
        
        # Evaluate results
        evaluation = eval_agent.evaluate_results(
            results=exploit_results,
            target_model=target_model,
            target_behavior=target_behavior,
        )
        
        # Create visualizations
        visualizations = eval_agent.create_visualizations(
            evaluation=evaluation,
            target_model=target_model,
            target_behavior=target_behavior,
        )
        
        # Generate summary
        summary = eval_agent.generate_summary(
            evaluation=evaluation,
            target_model=target_model,
            target_behavior=target_behavior,
        )
        
        # Save evaluation and summary
        eval_path = eval_agent.save_evaluation(
            evaluation=evaluation,
            target_model=target_model,
            target_behavior=target_behavior,
        )
        
        summary_path = eval_agent.save_summary(
            summary=summary,
            target_model=target_model,
            target_behavior=target_behavior,
        )
        
        logger.info(f"Evaluation phase completed, results saved to {eval_path}")
        
        # Log completion event
        log_agentops_event(
            "phase_completed",
            {
                "phase": "evaluation",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "completed",
                "eval_path": eval_path,
                "summary_path": summary_path,
            },
        )
        
        return {
            "evaluation": evaluation,
            "visualizations": visualizations,
            "summary": summary,
            "paths": {
                "evaluation": eval_path,
                "summary": summary_path,
            },
        }
        
    except Exception as e:
        logger.error(f"Evaluation phase failed: {str(e)}")
        
        # Log error event
        log_agentops_event(
            "phase_error",
            {
                "phase": "evaluation",
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "failed",
                "error": str(e),
            },
        )
        
        return {
            "error": str(e),
            "target_model": target_model,
            "target_behavior": target_behavior,
            "timestamp": datetime.now().isoformat(),
        }


async def run_full_pipeline_async(
    target_model: str,
    target_behavior: str,
    output_dir: str = "./output",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    skip_phases: List[str] = None,
    max_prompts: int = 10,
    test_method: str = "api",
    max_concurrent: int = 3,
    complexity_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Run the complete Gray Swan Arena pipeline asynchronously.
    
    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save outputs
        model_name: Name of the model to use for agents
        backup_model: Backup model to use if primary fails
        skip_phases: List of phases to skip
        max_prompts: Maximum number of prompts to generate
        test_method: Method for testing (api or web)
        max_concurrent: Maximum concurrent operations
        complexity_threshold: Threshold for using backup model (0.0-1.0)
        
    Returns:
        Dictionary containing results from all phases
    """
    results = {}
    skip_phases = skip_phases or []
    
    # Initialize AgentOps session
    api_key = os.getenv("AGENTOPS_API_KEY")
    if api_key:
        initialize_agentops(api_key)
        start_agentops_session(tags=["full_pipeline"])
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "exploits"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "evaluations"), exist_ok=True)
    
    # Initialize model manager
    if not backup_model:
        # Create a model manager instance
        model_manager = ModelManager(
            primary_model=model_name,
            complexity_threshold=complexity_threshold
        )
        # Try to get a suitable backup model
        backup_model = model_manager.get_backup_model(model_name)
        if backup_model:
            # Reinitialize with the backup model
            model_manager = ModelManager(
                primary_model=model_name,
                backup_model=backup_model,
                complexity_threshold=complexity_threshold
            )
    else:
        # Use the provided backup model
        model_manager = ModelManager(
            primary_model=model_name,
            backup_model=backup_model,
            complexity_threshold=complexity_threshold
        )
    
    logger.info(f"Using model manager with primary={model_name}, backup={backup_model}")
    
    # Log pipeline start
    log_agentops_event(
        "pipeline_started",
        {
            "target_model": target_model,
            "target_behavior": target_behavior,
            "agent_model": model_name,
            "backup_model": backup_model,
            "skip_phases": skip_phases,
            "max_prompts": max_prompts,
            "test_method": test_method,
            "max_concurrent": max_concurrent,
        },
    )
    
    try:
        # Phase 1: Reconnaissance
        if "recon" not in skip_phases:
            logger.info("Starting reconnaissance phase")
            recon_results = await run_parallel_reconnaissance(
                target_model=target_model,
                target_behavior=target_behavior,
                output_dir=os.path.join(output_dir, "reports"),
                model_name=model_name,
                reasoning_model=backup_model,
                backup_model=backup_model,
            )
            results["reconnaissance"] = recon_results
        
        # Phase 2: Prompt Engineering
        if "prompt" not in skip_phases:
            logger.info("Starting prompt engineering phase")
            recon_report = results.get("reconnaissance", {}).get("report", {})
            prompt_results = run_prompt_engineering(
                target_model=target_model,
                target_behavior=target_behavior,
                recon_report=recon_report,
                output_dir=os.path.join(output_dir, "prompts"),
                model_name=model_name,
                reasoning_model=backup_model,
                backup_model=backup_model,
                num_prompts=max_prompts,
            )
            results["prompt_engineering"] = prompt_results
        
        # Phase 3: Exploit Delivery
        if "exploit" not in skip_phases and "prompt_engineering" in results:
            logger.info("Starting exploit delivery phase")
            prompts = results["prompt_engineering"]["prompts"]
            exploit_results = await run_parallel_exploits(
                prompts=prompts,
                target_model=target_model,
                target_behavior=target_behavior,
                output_dir=os.path.join(output_dir, "exploits"),
                model_name=model_name,
                backup_model=backup_model,
                method=test_method,
                max_concurrent=max_concurrent,
            )
            results["exploit_delivery"] = exploit_results
        
        # Phase 4: Evaluation
        if "eval" not in skip_phases and "exploit_delivery" in results:
            logger.info("Starting evaluation phase")
            exploit_results_list = results["exploit_delivery"]["results"]
            eval_results = run_evaluation(
                exploit_results=exploit_results_list,
                target_model=target_model,
                target_behavior=target_behavior,
                output_dir=os.path.join(output_dir, "evaluations"),
                model_name=model_name,
                reasoning_model=backup_model,
                backup_model=backup_model,
            )
            results["evaluation"] = eval_results
        
        # Add model manager metrics to results
        results["model_metrics"] = model_manager.get_metrics()
        
        # Log pipeline completion
        log_agentops_event(
            "pipeline_completed",
            {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "completed",
                "phases_completed": list(results.keys()),
                "model_metrics": results["model_metrics"],
            },
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        
        # Log pipeline error
        log_agentops_event(
            "pipeline_error",
            {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "failed",
                "error": str(e),
                "phases_completed": list(results.keys()),
                "model_metrics": model_manager.get_metrics(),
            },
        )
        
        return {
            "error": str(e),
            "target_model": target_model,
            "target_behavior": target_behavior,
            "timestamp": datetime.now().isoformat(),
            "partial_results": results,
            "model_metrics": model_manager.get_metrics(),
        }


def run_full_pipeline(
    target_model: str,
    target_behavior: str,
    output_dir: str = "./output",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    skip_phases: List[str] = None,
    max_prompts: int = 10,
    test_method: str = "api",
    max_concurrent: int = 3,
    complexity_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Run the complete Gray Swan Arena pipeline.
    
    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save outputs
        model_name: Name of the model to use for agents
        backup_model: Backup model to use if primary fails
        skip_phases: List of phases to skip
        max_prompts: Maximum number of prompts to generate
        test_method: Method for testing (api or web)
        max_concurrent: Maximum concurrent operations
        complexity_threshold: Threshold for using backup model (0.0-1.0)
        
    Returns:
        Dictionary containing results from all phases
    """
    return asyncio.run(run_full_pipeline_async(
        target_model=target_model,
        target_behavior=target_behavior,
        output_dir=output_dir,
        model_name=model_name,
        backup_model=backup_model,
        skip_phases=skip_phases,
        max_prompts=max_prompts,
        test_method=test_method,
        max_concurrent=max_concurrent,
        complexity_threshold=complexity_threshold,
    ))


def main():
    """
    Main function for command-line execution.
    """
    # Set up arguments parser
    import argparse

    parser = argparse.ArgumentParser(
        description="Gray Swan Arena - AI Safety Testing Framework"
    )

    # Add arguments
    parser.add_argument(
        "--target-model", type=str, default="gpt-3.5-turbo", help="Target model to test"
    )
    parser.add_argument(
        "--target-behavior",
        type=str,
        default="bypass content policies",
        help="Behavior to target",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="Directory to save outputs"
    )
    parser.add_argument(
        "--agent-model", type=str, default="gpt-4", help="Model to use for agents"
    )
    parser.add_argument(
        "--reasoning-model", type=str, help="Model to use for reasoning tasks (e.g., o3-mini)"
    )
    parser.add_argument(
        "--backup-model", type=str, help="Backup model to use if primary fails"
    )
    parser.add_argument(
        "--complexity-threshold",
        type=float,
        default=0.7,
        help="Threshold for using backup model (0.0-1.0)"
    )
    parser.add_argument(
        "--skip-phases",
        type=str,
        nargs="*",
        choices=["recon", "prompt", "exploit", "eval"],
        help="Phases to skip",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=10, help="Maximum number of prompts to generate"
    )
    parser.add_argument(
        "--test-method",
        type=str,
        default="api",
        choices=["api", "web"],
        help="Method for testing (api or web)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=3, help="Maximum concurrent operations"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy mode (for backward compatibility)",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run tasks in parallel where possible"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    recon_dir = os.path.join(args.output_dir, "reports")
    prompt_dir = os.path.join(args.output_dir, "prompts")
    exploit_dir = os.path.join(args.output_dir, "exploits")
    eval_dir = os.path.join(args.output_dir, "evaluations")

    # Initialize AgentOps
    initialize_agentops()
    
    # Start a session
    start_agentops_session(agent_type="GraySwanPipeline", model=args.agent_model)
    
    # Log pipeline start
    log_agentops_event(
        "pipeline_started",
        {
            "target_model": args.target_model,
            "target_behavior": args.target_behavior,
            "agent_model": args.agent_model,
            "reasoning_model": args.reasoning_model,
            "backup_model": args.backup_model,
            "test_method": args.test_method,
            "parallel": args.parallel,
        },
    )

    # Run the pipeline
    try:
        # Step 1: Reconnaissance (if not skipped)
        report = None
        if "recon" not in (args.skip_phases or []):
            logger.info("Starting reconnaissance phase")
            start_time = time.time()
            
            if args.parallel:
                report = asyncio.run(
                    run_parallel_reconnaissance(
                        target_model=args.target_model,
                        target_behavior=args.target_behavior,
                        output_dir=recon_dir,
                        model_name=args.agent_model,
                        reasoning_model=args.reasoning_model,
                        backup_model=args.backup_model,
                    )
                )
            else:
                report = run_reconnaissance(
                    target_model=args.target_model,
                    target_behavior=args.target_behavior,
                    output_dir=recon_dir,
                    model_name=args.agent_model,
                    reasoning_model=args.reasoning_model,
                    backup_model=args.backup_model,
                )
            
            recon_time = time.time() - start_time
            logger.info(f"Reconnaissance phase completed in {recon_time:.2f} seconds")
        else:
            logger.info("Skipping reconnaissance phase")
            
        # Step 2: Prompt Engineering (if not skipped)
        prompts = None
        if "prompt" not in (args.skip_phases or []) and report:
            logger.info("Starting prompt engineering phase")
            start_time = time.time()
            
            prompts = run_prompt_engineering(
                target_model=args.target_model,
                target_behavior=args.target_behavior,
                recon_report=report,
                output_dir=prompt_dir,
                model_name=args.agent_model,
                reasoning_model=args.reasoning_model,
                backup_model=args.backup_model,
                max_prompts=args.max_prompts,
            )
            
            prompt_time = time.time() - start_time
            logger.info(f"Prompt engineering phase completed in {prompt_time:.2f} seconds")
        else:
            logger.info("Skipping prompt engineering phase")
            
        # Step 3: Exploit Delivery (if not skipped)
        results = None
        if "exploit" not in (args.skip_phases or []) and prompts:
            logger.info("Starting exploit delivery phase")
            start_time = time.time()
            
            results = run_exploit_delivery(
                target_model=args.target_model,
                target_behavior=args.target_behavior,
                prompts=prompts,
                output_dir=exploit_dir,
                model_name=args.agent_model,
                backup_model=args.backup_model,
                method=args.test_method,
                max_concurrent=args.max_concurrent,
            )
            
            exploit_time = time.time() - start_time
            logger.info(f"Exploit delivery phase completed in {exploit_time:.2f} seconds")
        else:
            logger.info("Skipping exploit delivery phase")
            
        # Step 4: Evaluation (if not skipped)
        if "eval" not in (args.skip_phases or []) and results:
            logger.info("Starting evaluation phase")
            start_time = time.time()
            
            evaluation = run_evaluation(
                target_model=args.target_model,
                target_behavior=args.target_behavior,
                results=results,
                output_dir=eval_dir,
                model_name=args.agent_model,
                reasoning_model=args.reasoning_model,
                backup_model=args.backup_model,
            )
            
            eval_time = time.time() - start_time
            logger.info(f"Evaluation phase completed in {eval_time:.2f} seconds")
        else:
            logger.info("Skipping evaluation phase")
            
        logger.info(f"Gray Swan Arena pipeline completed successfully")
        
        # Log pipeline completion
        log_agentops_event(
            "pipeline_completed",
            {
                "target_model": args.target_model,
                "target_behavior": args.target_behavior,
                "status": "success",
            },
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        
        # Log pipeline error
        log_agentops_event(
            "pipeline_error",
            {
                "target_model": args.target_model,
                "target_behavior": args.target_behavior,
                "error": str(e),
            },
        )


if __name__ == "__main__":
    main()
