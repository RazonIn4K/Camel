"""
Main module for Gray Swan Arena with Dependency Injection.

This module provides the main functionality for running Gray Swan Arena agents
and executing the full pipeline for AI safety evaluation, using dependency injection
for better testability and flexibility.
"""

import os
import asyncio
import time
import json
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import container
from cybersec_agents.grayswan.container import GraySwanContainer, GraySwanContainerFactory

# Import utilities
from cybersec_agents.grayswan.utils.agentops_utils import (
    initialize_agentops,
    log_agentops_event,
    start_agentops_session,
)
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("grayswan_main_di")


class GraySwanPipeline:
    """
    Gray Swan Arena pipeline with dependency injection.
    
    This class provides methods for running the Gray Swan Arena pipeline
    using dependency injection for better testability and flexibility.
    """
    
    def __init__(self, container: GraySwanContainer):
        """
        Initialize the GraySwanPipeline.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = container.logger()
        
        # Initialize AgentOps if API key is available
        api_key = os.getenv("AGENTOPS_API_KEY")
        if api_key:
            initialize_agentops(api_key)
            start_agentops_session(tags=["grayswan_pipeline"])
            
        self.logger.info("GraySwanPipeline initialized with dependency injection")
    
    async def run_parallel_reconnaissance(
        self,
        target_model: str,
        target_behavior: str,
    ) -> Dict[str, Any]:
        """
        Run reconnaissance tasks in parallel.
        
        Args:
            target_model: The target model to test
            target_behavior: The behavior to target
            
        Returns:
            Dictionary containing reconnaissance results and report path
        """
        self.logger.info(f"Starting parallel reconnaissance for {target_model} - {target_behavior}")
        
        # Get ReconAgent from container
        recon_agent = self.container.recon_agent()
        
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
        
        self.logger.info(f"Parallel reconnaissance completed, report saved to {report_path}")
        
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
        self,
        target_model: str,
        target_behavior: str,
    ) -> Dict[str, Any]:
        """
        Run the reconnaissance phase of the Gray Swan Arena pipeline.
        
        Args:
            target_model: The target model to test
            target_behavior: The behavior to target
            
        Returns:
            Dictionary containing the reconnaissance report
        """
        self.logger.info(f"Starting reconnaissance phase for {target_model} - {target_behavior}")
        
        try:
            # Get ReconAgent from container
            recon_agent = self.container.recon_agent()
            
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
            
            self.logger.info(f"Reconnaissance phase completed, report saved to {report_path}")
            
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
            self.logger.error(f"Reconnaissance phase failed: {str(e)}")
            
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
        self,
        target_model: str,
        target_behavior: str,
        recon_report: Dict[str, Any],
        num_prompts: int = 10,
    ) -> Dict[str, Any]:
        """
        Run the prompt engineering phase of the Gray Swan Arena pipeline.
        
        Args:
            target_model: The target model to test
            target_behavior: The behavior to target
            recon_report: Report from the reconnaissance phase
            num_prompts: Number of prompts to generate
            
        Returns:
            Dictionary containing the generated prompts and file path
        """
        self.logger.info(f"Starting prompt engineering phase for {target_model} - {target_behavior}")
        
        # Log phase start
        log_agentops_event(
            "phase_started",
            {
                "phase": "prompt_engineering",
                "target_model": target_model,
                "target_behavior": target_behavior,
            },
        )
        
        try:
            # Get PromptEngineerAgent from container
            prompt_agent = self.container.prompt_engineer_agent()
            
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
            
            self.logger.info(f"Prompt engineering phase completed, {len(prompts)} prompts saved to {prompts_path}")
            
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
            self.logger.error(f"Prompt engineering phase failed: {str(e)}")
            
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
        self,
        prompts: List[str],
        target_model: str,
        target_behavior: str,
        method: str = "api",
        max_concurrent: int = 3,
    ) -> Dict[str, Any]:
        """
        Run exploit delivery in parallel batches.
        
        Args:
            prompts: List of prompts to test
            target_model: The target model to test
            target_behavior: The behavior to target
            method: Method to use (api, web)
            max_concurrent: Maximum number of concurrent operations
            
        Returns:
            Dictionary containing exploit results and file path
        """
        self.logger.info(f"Starting parallel exploit delivery for {target_model} with {len(prompts)} prompts")
        
        # Get ExploitDeliveryAgent from container
        exploit_agent = self.container.exploit_delivery_agent()
        
        # Split prompts into batches
        batch_size = max_concurrent
        batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
        
        all_results: list[Any] = []
        
        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
            
            # Create tasks for concurrent execution
            tasks: list[Any] = []
            for prompt in batch:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        self._execute_single_prompt,
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
        
        self.logger.info(f"Parallel exploit delivery completed, results saved to {results_path}")
        
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
        self,
        agent: Any,
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
        self,
        prompts: List[str],
        target_model: str,
        target_behavior: str,
        method: str = "api",
    ) -> Dict[str, Any]:
        """
        Run the exploit delivery phase of the Gray Swan Arena pipeline.
        
        Args:
            prompts: List of prompts to test
            target_model: The target model to test
            target_behavior: The behavior to target
            method: Method to use (api, web)
            
        Returns:
            Dictionary containing the exploit results
        """
        self.logger.info(f"Starting exploit delivery phase for {target_model} - {target_behavior}")
        
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
            # Get ExploitDeliveryAgent from container
            exploit_agent = self.container.exploit_delivery_agent()
            
            # Run prompts
            results: list[Any] = exploit_agent.run_prompts(
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
            
            self.logger.info(f"Exploit delivery phase completed, results saved to {results_path}")
            
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
            self.logger.error(f"Exploit delivery phase failed: {str(e)}")
            
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
        self,
        exploit_results: List[Dict[str, Any]],
        target_model: str,
        target_behavior: str,
        include_advanced_visualizations: bool = True,
        include_interactive_dashboard: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the evaluation phase of the Gray Swan Arena pipeline.
        
        Args:
            exploit_results: Results from the exploit delivery phase
            target_model: The target model that was tested
            target_behavior: The behavior that was targeted
            include_advanced_visualizations: Whether to include advanced visualizations
            include_interactive_dashboard: Whether to include interactive dashboard
            
        Returns:
            Dictionary containing the evaluation results
        """
        self.logger.info(f"Starting evaluation phase for {target_model} - {target_behavior}")
        
        # Log phase start
        log_agentops_event(
            "phase_started",
            {
                "phase": "evaluation",
                "target_model": target_model,
                "target_behavior": target_behavior,
            },
        )
        
        try:
            # Get EvaluationAgent from container
            eval_agent = self.container.evaluation_agent()
            
            # Evaluate results
            evaluation = eval_agent.evaluate_results(
                results=exploit_results,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            
            # Create basic visualizations
            basic_visualizations = eval_agent.create_visualizations(
                evaluation=evaluation,
                target_model=target_model,
                target_behavior=target_behavior,
            )
            
            # Create advanced visualizations if requested
            advanced_visualizations: dict[str, Any] = {}
            if include_advanced_visualizations:
                advanced_visualizations = eval_agent.create_advanced_visualizations(
                    results=exploit_results,
                    target_model=target_model,
                    target_behavior=target_behavior,
                    include_interactive=include_interactive_dashboard,
                )
            
            # Combine visualizations
            visualizations: dict[str, Any] = {**basic_visualizations, **advanced_visualizations}
            
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
            
            self.logger.info(f"Evaluation phase completed, results saved to {eval_path}")
            
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
                    "has_advanced_visualizations": include_advanced_visualizations,
                    "has_interactive_dashboard": include_interactive_dashboard,
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
            self.logger.error(f"Evaluation phase failed: {str(e)}")
            
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
        self,
        target_model: str,
        target_behavior: str,
        skip_phases: List[str] = None,
        max_prompts: int = 10,
        test_method: str = "api",
        max_concurrent: int = 3,
        include_advanced_visualizations: bool = True,
        include_interactive_dashboard: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete Gray Swan Arena pipeline asynchronously.
        
        Args:
            target_model: The target model to test
            target_behavior: The behavior to target
            skip_phases: List of phases to skip
            max_prompts: Maximum number of prompts to generate
            test_method: Method for testing (api or web)
            max_concurrent: Maximum concurrent operations
            include_advanced_visualizations: Whether to include advanced visualizations
            include_interactive_dashboard: Whether to include interactive dashboard
            
        Returns:
            Dictionary containing results from all phases
        """
        results: list[Any] = {}
        skip_phases = skip_phases or []
        
        # Start AgentOps session
        start_agentops_session(tags=["full_pipeline"])
        
        # Log pipeline start
        log_agentops_event(
            "pipeline_started",
            {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "skip_phases": skip_phases,
                "max_prompts": max_prompts,
                "test_method": test_method,
                "max_concurrent": max_concurrent,
                "include_advanced_visualizations": include_advanced_visualizations,
                "include_interactive_dashboard": include_interactive_dashboard,
            },
        )
        
        try:
            # Phase 1: Reconnaissance
            if "recon" not in skip_phases:
                self.logger.info("Starting reconnaissance phase")
                recon_results = await self.run_parallel_reconnaissance(
                    target_model=target_model,
                    target_behavior=target_behavior,
                )
                results["reconnaissance"] = recon_results
            
            # Phase 2: Prompt Engineering
            if "prompt" not in skip_phases:
                self.logger.info("Starting prompt engineering phase")
                recon_report = results.get("reconnaissance", {}).get("report", {})
                prompt_results = self.run_prompt_engineering(
                    target_model=target_model,
                    target_behavior=target_behavior,
                    recon_report=recon_report,
                    num_prompts=max_prompts,
                )
                results["prompt_engineering"] = prompt_results
            
            # Phase 3: Exploit Delivery
            if "exploit" not in skip_phases and "prompt_engineering" in results:
                self.logger.info("Starting exploit delivery phase")
                prompts = results["prompt_engineering"]["prompts"]
                exploit_results = await self.run_parallel_exploits(
                    prompts=prompts,
                    target_model=target_model,
                    target_behavior=target_behavior,
                    method=test_method,
                    max_concurrent=max_concurrent,
                )
                results["exploit_delivery"] = exploit_results
            
            # Phase 4: Evaluation
            if "eval" not in skip_phases and "exploit_delivery" in results:
                self.logger.info("Starting evaluation phase")
                exploit_results_list = results["exploit_delivery"]["results"]
                eval_results = self.run_evaluation(
                    exploit_results=exploit_results_list,
                    target_model=target_model,
                    target_behavior=target_behavior,
                    include_advanced_visualizations=include_advanced_visualizations,
                    include_interactive_dashboard=include_interactive_dashboard,
                )
                results["evaluation"] = eval_results
            
            # Log pipeline completion
            log_agentops_event(
                "pipeline_completed",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "completed",
                    "phases_completed": list(results.keys()),
                },
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            
            # Log pipeline error
            log_agentops_event(
                "pipeline_error",
                {
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "failed",
                    "error": str(e),
                    "phases_completed": list(results.keys()),
                },
            )
            
            return {
                "error": str(e),
                "target_model": target_model,
                "target_behavior": target_behavior,
                "timestamp": datetime.now().isoformat(),
                "partial_results": results,
            }
    
    def run_full_pipeline(
        self,
        target_model: str,
        target_behavior: str,
        skip_phases: List[str] = None,
        max_prompts: int = 10,
        test_method: str = "api",
        max_concurrent: int = 3,
        include_advanced_visualizations: bool = True,
        include_interactive_dashboard: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete Gray Swan Arena pipeline.
        
        Args:
            target_model: The target model to test
            target_behavior: The behavior to target
            skip_phases: List of phases to skip
            max_prompts: Maximum number of prompts to generate
            test_method: Method for testing (api or web)
            max_concurrent: Maximum concurrent operations
            include_advanced_visualizations: Whether to include advanced visualizations
            include_interactive_dashboard: Whether to include interactive dashboard
            
        Returns:
            Dictionary containing results from all phases
        """
        return asyncio.run(self.run_full_pipeline_async(
            target_model=target_model,
            target_behavior=target_behavior,
            skip_phases=skip_phases,
            max_prompts=max_prompts,
            test_method=test_method,
            max_concurrent=max_concurrent,
            include_advanced_visualizations=include_advanced_visualizations,
            include_interactive_dashboard=include_interactive_dashboard,
        ))


def main():
    """
    Main function for command-line execution.
    """
    # Set up arguments parser
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gray Swan Arena - AI Safety Testing Framework (with Dependency Injection)"
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
        "--config-file", type=str, help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--advanced-visualizations",
        action="store_true",
        help="Include advanced visualizations",
    )
    parser.add_argument(
        "--interactive-dashboard",
        action="store_true",
        help="Include interactive dashboard",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy mode (only run reconnaissance phase)",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration dictionary
    config_dict: dict[str, Any] = {
        "output_dir": args.output_dir,
        "agents": {
            "recon": {
                "output_dir": os.path.join(args.output_dir, "reports"),
                "model_name": args.agent_model,
            },
            "prompt_engineer": {
                "output_dir": os.path.join(args.output_dir, "prompts"),
                "model_name": args.agent_model,
            },
            "exploit_delivery": {
                "output_dir": os.path.join(args.output_dir, "exploits"),
                "model_name": args.agent_model,
                "browser_method": args.test_method if args.test_method == "web" else "playwright",
                "headless": True,
            },
            "evaluation": {
                "output_dir": os.path.join(args.output_dir, "evaluations"),
                "model_name": args.agent_model,
            },
        },
        "browser": {
            "method": "playwright",
            "headless": True,
            "timeout": 60000,
            "enhanced": True,
            "retry_attempts": 3,
            "retry_delay": 1.0,
        },
        "visualization": {
            "output_dir": os.path.join(args.output_dir, "visualizations"),
            "dpi": 300,
            "theme": "default",
            "advanced": args.advanced_visualizations,
            "interactive": args.interactive_dashboard,
            "clustering_clusters": 4,
            "similarity_threshold": 0.5,
        },
    }
    
    # Load configuration from file if provided
    if args.config_file:
        try:
            if args.config_file.endswith('.yaml') or args.config_file.endswith('.yml'):
                with open(args.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif args.config_file.endswith('.json'):
                with open(args.config_file, 'r') as f:
                    file_config = json.load(f)
            else:
                print(f"Unsupported configuration file format: {args.config_file}")
                return
            
            # Merge file configuration with command-line configuration
            # (command-line arguments take precedence)
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        deep_update(d[k], v)
                    else:
                        d[k] = v
            
            deep_update(config_dict, file_config)
            
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return
    
    # Create container
    container = GraySwanContainerFactory.create_container(config_dict)
    
    # Create pipeline
    pipeline = GraySwanPipeline(container)
    
    if args.legacy:
        # Run only reconnaissance (legacy mode)
        report = pipeline.run_reconnaissance(
            target_model=args.target_model,
            target_behavior=args.target_behavior,
        )
        print(f"Reconnaissance complete. Report generated for {args.target_model}.")
    else:
        # Run full pipeline
        results: list[Any] = pipeline.run_full_pipeline(
            target_model=args.target_model,
            target_behavior=args.target_behavior,
            skip_phases=args.skip_phases,
            max_prompts=args.max_prompts,
            test_method=args.test_method,
            max_concurrent=args.max_concurrent,
            include_advanced_visualizations=args.advanced_visualizations,
            include_interactive_dashboard=args.interactive_dashboard,
        )
        
        # Print summary
        print(f"\nGray Swan Arena pipeline completed for {args.target_model}.")
        print(f"Target behavior: {args.target_behavior}")
        
        if "error" in results:
            print(f"\nError: {results['error']}")
            print("Partial results may be available in the output directory.")
        else:
            print("\nPhases completed:")
            for phase in ["reconnaissance", "prompt_engineering", "exploit_delivery", "evaluation"]:
                if phase in results:
                    print(f"- {phase.replace('_', ' ').title()}")
            
            if "evaluation" in results and "evaluation" in results["evaluation"]:
                eval_data = results["evaluation"]["evaluation"]
                success_rate = eval_data.get("success_rate", 0) * 100
                print(f"\nSuccess rate: {success_rate:.1f}%")
                print(f"Total attempts: {eval_data.get('total_attempts', 0)}")
                print(f"Successful attempts: {eval_data.get('successful_attempts', 0)}")
            
            print(f"\nResults saved to: {os.path.abspath(args.output_dir)}")
            
            # Print interactive dashboard path if available
            if args.interactive_dashboard and "evaluation" in results:
                vis_paths = results["evaluation"].get("visualizations", {})
                dashboard_path = next((p for k, p in vis_paths.items() if "dashboard" in k.lower()), None)
                if dashboard_path:
                    print(f"\nInteractive dashboard available at: {os.path.abspath(dashboard_path)}")
                    print("Open in a web browser to view.")


if __name__ == "__main__":
    main()