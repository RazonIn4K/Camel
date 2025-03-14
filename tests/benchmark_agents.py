#!/usr/bin/env python3
"""Benchmark script for evaluating performance of Gray Swan Arena agents."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).parent.parent))

# Check if agentops is installed
try:
    import agentops
    from agentops.events import ActionEvent, LLMEvent
except ImportError:
    print("AgentOps not installed. Install with 'pip install agentops'")
    agentops = None

from cybersec_agents.grayswan import (
    EvaluationAgent,
    ExploitDeliveryAgent,
    PromptEngineerAgent,
    ReconAgent,
)
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Constants
OUTPUT_DIR = Path("tests/benchmark_results")
MODELS = ["gpt-3.5-turbo", "gpt-4"]
DEFAULT_NUM_PROMPTS = 3
DEFAULT_TOPICS = [
    "network security",
    "password policies",
    "cybersecurity training",
    "data protection",
    "AI safety mechanisms",
]

# Set up logging
logger = setup_logging(name="benchmark", log_level=20)  # INFO level


def setup_benchmark_environment() -> Dict[str, Any]:
    """Set up the benchmark environment."""
    # Load environment variables
    load_dotenv()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = OUTPUT_DIR / f"benchmark_{timestamp}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable not found.")
        print("Benchmark tests cannot proceed without an API key.")
        sys.exit(1)

    # Check for AgentOps API key
    agentops_api_key = os.environ.get("AGENTOPS_API_KEY")
    if agentops and agentops_api_key:
        # Initialize AgentOps
        session = agentops.init()
        session.add_tags(["benchmark", "gray-swan-arena"])
        print(f"AgentOps initialized with session ID: {session.session_id}")
    else:
        session = None
        print("AgentOps not initialized. Set AGENTOPS_API_KEY to enable monitoring.")

    print(f"Benchmark environment set up.")
    print(f"Output will be saved to {benchmark_dir.absolute()}")

    return {
        "timestamp": timestamp,
        "benchmark_dir": benchmark_dir,
        "openai_api_key": openai_api_key,
        "agentops_session": session,
    }


def benchmark_recon_agent(
    env: Dict[str, Any], topics: List[str], model: str
) -> Tuple[bool, float, str]:
    """Benchmark the ReconAgent."""
    print(f"\n=== Benchmarking ReconAgent with model {model} ===")

    benchmark_dir = env["benchmark_dir"]
    recon_dir = benchmark_dir / "recon"
    recon_dir.mkdir(exist_ok=True)
    session = env.get("agentops_session")

    results = []
    total_time = 0

    try:
        # Initialize agent
        agent = ReconAgent()

        if session:
            session.record(
                ActionEvent(
                    action_type="agent_created",
                    inputs={"agent_type": "ReconAgent", "model": model},
                    outputs={"status": "initialized"},
                )
            )

        for topic in topics:
            start_time = time.time()

            if session:
                session.record(
                    ActionEvent(
                        action_type="web_search_started",
                        inputs={"topic": topic},
                        outputs={},
                    )
                )

            print(f"Performing web search for topic: {topic}")
            web_results = agent.run_web_search(topic)

            if session:
                session.record(
                    ActionEvent(
                        action_type="web_search_completed",
                        inputs={"topic": topic},
                        outputs={
                            "result_count": len(web_results) if web_results else 0
                        },
                    )
                )

            # Generate report
            print(f"Generating report for topic: {topic} using {model}")

            if session:
                session.record(
                    ActionEvent(
                        action_type="report_generation_started",
                        inputs={"topic": topic, "model": model},
                        outputs={},
                    )
                )

            report = agent.generate_report(
                model_info=web_results,
                behavior_info=f"Information about {topic}",
                techniques_info="Web search techniques",
            )

            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed

            if session:
                session.record(
                    ActionEvent(
                        action_type="report_generation_completed",
                        inputs={"topic": topic, "model": model},
                        outputs={"time_seconds": elapsed},
                    )
                )

            # Save report
            topic_slug = topic.replace(" ", "_").lower()
            report_file = recon_dir / f"{topic_slug}_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            # Record metrics
            result = {
                "topic": topic,
                "model": model,
                "time_seconds": elapsed,
                "report_length": len(json.dumps(report)),
                "num_results": len(report.get("search_results", [])),
                "report_file": str(report_file),
            }
            results.append(result)

            print(f"✅ Completed topic '{topic}' in {elapsed:.2f} seconds")

        # Save benchmark results
        results_file = recon_dir / f"benchmark_results_{model.replace('-', '_')}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        avg_time = total_time / len(topics) if topics else 0
        print(f"✅ ReconAgent benchmark with {model} completed successfully!")
        print(f"Average time per topic: {avg_time:.2f} seconds")

        return True, avg_time, str(results_file)

    except Exception as e:
        print(f"❌ Error during ReconAgent benchmark: {e}")
        if session:
            session.record(
                ActionEvent(
                    action_type="benchmark_error",
                    inputs={"agent_type": "ReconAgent", "model": model},
                    outputs={"error": str(e)},
                )
            )
        return False, 0, ""


def benchmark_prompt_engineer_agent(
    env: Dict[str, Any],
    recon_results_file: str,
    model: str,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
) -> Tuple[bool, float, str]:
    """Benchmark the PromptEngineerAgent."""
    print(f"\n=== Benchmarking PromptEngineerAgent with model {model} ===")

    benchmark_dir = env["benchmark_dir"]
    prompt_dir = benchmark_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    session = env.get("agentops_session")

    if not os.path.exists(recon_results_file):
        print(f"❌ Recon results file not found: {recon_results_file}")
        return False, 0, ""

    try:
        # Load recon results
        with open(recon_results_file, "r") as f:
            recon_results = json.load(f)

        # Collect all recon reports
        all_reports = []
        for result in recon_results:
            report_file = result.get("report_file")
            if report_file and os.path.exists(report_file):
                with open(report_file, "r") as f:
                    report = json.load(f)
                    all_reports.append(report)

        if not all_reports:
            print("❌ No recon reports found")
            return False, 0, ""

        # Initialize agent
        agent = PromptEngineerAgent()

        if session:
            session.record(
                ActionEvent(
                    action_type="agent_created",
                    inputs={"agent_type": "PromptEngineerAgent", "model": model},
                    outputs={"status": "initialized"},
                )
            )

        # Generate prompts
        print(f"Generating {num_prompts} prompts from recon reports using {model}...")
        start_time = time.time()

        prompts = []
        for i, report in enumerate(all_reports):
            topic = recon_results[i]["topic"]
            print(f"Generating prompts for topic: {topic}")

            # Set the report data
            agent.set_recon_data(report)

            if session:
                session.record(
                    ActionEvent(
                        action_type="prompt_generation_started",
                        inputs={
                            "topic": topic,
                            "model": model,
                            "num_prompts": num_prompts,
                        },
                        outputs={},
                    )
                )

            # Generate prompts for this report, specifying model
            report_prompts = agent.generate_prompts(
                model=model, num_prompts=num_prompts  # Specify model
            )

            if session:
                session.record(
                    ActionEvent(
                        action_type="prompt_generation_completed",
                        inputs={"topic": topic, "model": model},
                        outputs={"num_prompts_generated": len(report_prompts)},
                    )
                )

            # Add topic info
            for prompt in report_prompts:
                prompt["topic"] = topic

            prompts.extend(report_prompts)

        end_time = time.time()
        elapsed = end_time - start_time

        # Save prompts
        prompts_file = prompt_dir / f"prompts_{model.replace('-', '_')}.json"
        with open(prompts_file, "w") as f:
            json.dump(prompts, f, indent=2)

        # Record metrics
        total_prompts = len(prompts)
        time_per_prompt = elapsed / total_prompts if total_prompts > 0 else 0

        results = {
            "model": model,
            "total_time_seconds": elapsed,
            "total_prompts": total_prompts,
            "time_per_prompt_seconds": time_per_prompt,
            "num_topics": len(all_reports),
            "prompts_file": str(prompts_file),
        }

        # Save benchmark results
        results_file = prompt_dir / f"benchmark_results_{model.replace('-', '_')}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✅ PromptEngineerAgent benchmark with {model} completed successfully!")
        print(f"Generated {total_prompts} prompts in {elapsed:.2f} seconds")
        print(f"Average time per prompt: {time_per_prompt:.2f} seconds")

        return True, time_per_prompt, str(prompts_file)

    except Exception as e:
        print(f"❌ Error during PromptEngineerAgent benchmark: {e}")
        if session:
            session.record(
                ActionEvent(
                    action_type="benchmark_error",
                    inputs={"agent_type": "PromptEngineerAgent", "model": model},
                    outputs={"error": str(e)},
                )
            )
        return False, 0, ""


def benchmark_exploit_delivery_agent(
    env: Dict[str, Any], prompts_file: str, model: str, target_model: str
) -> Tuple[bool, float, str]:
    """Benchmark the ExploitDeliveryAgent."""
    print(f"\n=== Benchmarking ExploitDeliveryAgent with model {model} ===")
    print(f"Target model for evaluation: {target_model}")

    benchmark_dir = env["benchmark_dir"]
    exploit_dir = benchmark_dir / "exploits"
    exploit_dir.mkdir(exist_ok=True)
    session = env.get("agentops_session")

    if not os.path.exists(prompts_file):
        print(f"❌ Prompts file not found: {prompts_file}")
        return False, 0, ""

    try:
        # Load prompts
        with open(prompts_file, "r") as f:
            prompts = json.load(f)

        if not prompts:
            print("❌ No prompts found")
            return False, 0, ""

        # Initialize agent
        agent = ExploitDeliveryAgent()

        if session:
            session.record(
                ActionEvent(
                    action_type="agent_created",
                    inputs={"agent_type": "ExploitDeliveryAgent", "model": model},
                    outputs={"status": "initialized"},
                )
            )

        # Run prompts through target model
        print(
            f"Running {len(prompts)} prompts against {target_model}, evaluating with {model}..."
        )
        start_time = time.time()

        # Set up batches (up to 5 prompts per batch to avoid timeouts)
        batch_size = 5
        prompt_batches = [
            prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
        ]

        all_results = []
        total_prompts_run = 0

        for i, batch in enumerate(prompt_batches):
            print(
                f"Running batch {i+1}/{len(prompt_batches)} ({len(batch)} prompts)..."
            )

            if session:
                session.record(
                    ActionEvent(
                        action_type="prompt_execution_started",
                        inputs={
                            "batch": i + 1,
                            "num_prompts": len(batch),
                            "target_model": target_model,
                            "evaluation_model": model,
                        },
                        outputs={},
                    )
                )

            # Run this batch of prompts, specifying models
            try:
                batch_results = agent.run_prompts(
                    prompts=batch,
                    target_model=target_model,
                    evaluation_model=model,  # Specify the model for evaluation
                    target_behavior="default",
                )

                if session:
                    session.record(
                        ActionEvent(
                            action_type="prompt_execution_completed",
                            inputs={"batch": i + 1, "target_model": target_model},
                            outputs={"num_results": len(batch_results)},
                        )
                    )

                # Add additional metadata
                for result in batch_results:
                    result["target_model"] = target_model
                    result["eval_model"] = model

                all_results.extend(batch_results)
                total_prompts_run += len(batch)

                print(f"✅ Batch {i+1} completed: {len(batch_results)} results")

                # Brief pause between batches to avoid rate limits
                if i < len(prompt_batches) - 1:
                    time.sleep(2)

            except Exception as e:
                print(f"❌ Error running batch {i+1}: {e}")
                if session:
                    session.record(
                        ActionEvent(
                            action_type="batch_error",
                            inputs={"batch": i + 1, "target_model": target_model},
                            outputs={"error": str(e)},
                        )
                    )

        end_time = time.time()
        elapsed = end_time - start_time

        # Save results
        results_file = (
            exploit_dir / f"exploit_results_{target_model.replace('-', '_')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        # Record metrics
        time_per_prompt = elapsed / total_prompts_run if total_prompts_run > 0 else 0

        metrics = {
            "eval_model": model,
            "target_model": target_model,
            "total_time_seconds": elapsed,
            "total_prompts_run": total_prompts_run,
            "time_per_prompt_seconds": time_per_prompt,
            "results_file": str(results_file),
        }

        # Save benchmark metrics
        metrics_file = (
            exploit_dir / f"benchmark_metrics_{target_model.replace('-', '_')}.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ ExploitDeliveryAgent benchmark with {model} completed successfully!")
        print(
            f"Ran {total_prompts_run}/{len(prompts)} prompts in {elapsed:.2f} seconds"
        )
        print(f"Average time per prompt: {time_per_prompt:.2f} seconds")

        return True, time_per_prompt, str(results_file)

    except Exception as e:
        print(f"❌ Error during ExploitDeliveryAgent benchmark: {e}")
        if session:
            session.record(
                ActionEvent(
                    action_type="benchmark_error",
                    inputs={"agent_type": "ExploitDeliveryAgent", "model": model},
                    outputs={"error": str(e)},
                )
            )
        return False, 0, ""


def benchmark_evaluation_agent(
    env: Dict[str, Any], exploit_results_file: str, recon_results_file: str, model: str
) -> Tuple[bool, float, Path]:
    """Benchmark the EvaluationAgent."""
    print(f"\n=== Benchmarking EvaluationAgent with model {model} ===")

    benchmark_dir = env["benchmark_dir"]
    eval_dir = benchmark_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    session = env.get("agentops_session")

    if not os.path.exists(exploit_results_file):
        print(f"❌ Exploit results file not found: {exploit_results_file}")
        return False, 0, eval_dir

    try:
        # Load exploit results
        with open(exploit_results_file, "r") as f:
            exploit_results = json.load(f)

        if not exploit_results:
            print("❌ No exploit results found")
            return False, 0, eval_dir

        # Load recon data
        recon_data = None
        if recon_results_file and os.path.exists(recon_results_file):
            with open(recon_results_file, "r") as f:
                recon_results = json.load(f)

            # Get the first report file
            if recon_results and len(recon_results) > 0:
                report_file = recon_results[0].get("report_file")
                if report_file and os.path.exists(report_file):
                    with open(report_file, "r") as f:
                        recon_data = json.load(f)

        # Initialize agent
        agent = EvaluationAgent()

        if session:
            session.record(
                ActionEvent(
                    action_type="agent_created",
                    inputs={"agent_type": "EvaluationAgent", "model": model},
                    outputs={"status": "initialized"},
                )
            )

        # Load data
        print(f"Loading {len(exploit_results)} exploit results for evaluation...")
        agent.load_results(exploit_results)

        if recon_data:
            print("Loading recon data for context...")
            agent.load_recon_data(recon_data)

        # Run evaluation
        print("Running evaluation...")
        start_time = time.time()

        if session:
            session.record(
                ActionEvent(
                    action_type="evaluation_started",
                    inputs={"num_results": len(exploit_results), "model": model},
                    outputs={},
                )
            )

        # Calculate statistics
        print("Calculating statistics...")
        stats = agent.calculate_statistics()

        # Create visualizations
        print("Creating visualizations...")
        agent.create_visualizations(output_dir=str(eval_dir))

        # Generate reports with specified model
        print(f"Generating evaluation reports using {model}...")

        if session:
            session.record(
                ActionEvent(
                    action_type="report_generation_started",
                    inputs={"model": model},
                    outputs={},
                )
            )

        # JSON report
        json_report = agent.generate_report(format="json", model=model)
        json_report_file = eval_dir / "evaluation_report.json"
        with open(json_report_file, "w") as f:
            json.dump(json_report, f, indent=2)

        # Markdown report
        md_report = agent.generate_report(format="markdown", model=model)
        md_report_file = eval_dir / "evaluation_report.md"
        with open(md_report_file, "w") as f:
            f.write(md_report)

        # HTML report
        html_report = agent.generate_report(format="html", model=model)
        html_report_file = eval_dir / "evaluation_report.html"
        with open(html_report_file, "w") as f:
            f.write(html_report)

        end_time = time.time()
        elapsed = end_time - start_time

        if session:
            session.record(
                ActionEvent(
                    action_type="report_generation_completed",
                    inputs={"model": model},
                    outputs={"time_seconds": elapsed},
                )
            )

        # Record metrics
        metrics = {
            "model": model,
            "total_time_seconds": elapsed,
            "num_results_evaluated": len(exploit_results),
            "reports_generated": [
                str(json_report_file),
                str(md_report_file),
                str(html_report_file),
            ],
        }

        # Save benchmark metrics
        metrics_file = eval_dir / "benchmark_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ EvaluationAgent benchmark with {model} completed successfully!")
        print(f"Evaluated {len(exploit_results)} results in {elapsed:.2f} seconds")
        print(f"Reports saved to {eval_dir}")

        return True, elapsed, eval_dir

    except Exception as e:
        print(f"❌ Error during EvaluationAgent benchmark: {e}")
        if session:
            session.record(
                ActionEvent(
                    action_type="benchmark_error",
                    inputs={"agent_type": "EvaluationAgent", "model": model},
                    outputs={"error": str(e)},
                )
            )
        return False, 0, eval_dir


def run_full_benchmark(
    env: Dict[str, Any],
    topics: List[str],
    eval_model: str,
    target_model: str,
    num_prompts: int,
) -> Dict[str, Any]:
    """Run a full benchmark of all agents in sequence."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING FULL BENCHMARK")
    print(f"Evaluation Model: {eval_model}")
    print(f"Target Model: {target_model}")
    print(f"Topics: {', '.join(topics)}")
    print(f"Prompts Per Topic: {num_prompts}")
    print(f"{'=' * 60}")

    benchmark_start = time.time()
    session = env.get("agentops_session")

    if session:
        session.record(
            ActionEvent(
                action_type="benchmark_started",
                inputs={
                    "eval_model": eval_model,
                    "target_model": target_model,
                    "topics": topics,
                    "num_prompts": num_prompts,
                },
                outputs={},
            )
        )

    results = {
        "eval_model": eval_model,
        "target_model": target_model,
        "topics": topics,
        "num_prompts": num_prompts,
        "timestamp": env["timestamp"],
        "benchmark_dir": str(env["benchmark_dir"]),
        "agents": {},
    }

    # Step 1: Benchmark ReconAgent
    print("\nStep 1/4: Benchmarking ReconAgent...")
    recon_success, recon_time, recon_results_file = benchmark_recon_agent(
        env=env, topics=topics, model=eval_model
    )

    results["agents"]["recon"] = {
        "success": recon_success,
        "time": recon_time,
        "results_file": recon_results_file,
    }

    # Step 2: Benchmark PromptEngineerAgent
    if recon_success:
        print("\nStep 2/4: Benchmarking PromptEngineerAgent...")
        prompt_success, prompt_time, prompts_file = benchmark_prompt_engineer_agent(
            env=env,
            recon_results_file=recon_results_file,
            model=eval_model,
            num_prompts=num_prompts,
        )
    else:
        print("\nSkipping PromptEngineerAgent benchmark due to ReconAgent failure")
        prompt_success = False
        prompt_time = 0
        prompts_file = ""

    results["agents"]["prompt_engineer"] = {
        "success": prompt_success,
        "time": prompt_time,
        "results_file": prompts_file,
    }

    # Step 3: Benchmark ExploitDeliveryAgent
    if prompt_success:
        print("\nStep 3/4: Benchmarking ExploitDeliveryAgent...")
        (
            exploit_success,
            exploit_time,
            exploit_results_file,
        ) = benchmark_exploit_delivery_agent(
            env=env,
            prompts_file=prompts_file,
            model=eval_model,
            target_model=target_model,
        )
    else:
        print(
            "\nSkipping ExploitDeliveryAgent benchmark due to PromptEngineerAgent failure"
        )
        exploit_success = False
        exploit_time = 0
        exploit_results_file = ""

    results["agents"]["exploit_delivery"] = {
        "success": exploit_success,
        "time": exploit_time,
        "results_file": exploit_results_file,
    }

    # Step 4: Benchmark EvaluationAgent
    if exploit_success:
        print("\nStep 4/4: Benchmarking EvaluationAgent...")
        eval_success, eval_time, eval_dir = benchmark_evaluation_agent(
            env=env,
            exploit_results_file=exploit_results_file,
            recon_results_file=recon_results_file,
            model=eval_model,
        )
    else:
        print(
            "\nSkipping EvaluationAgent benchmark due to ExploitDeliveryAgent failure"
        )
        eval_success = False
        eval_time = 0
        eval_dir = None

    results["agents"]["evaluation"] = {
        "success": eval_success,
        "time": eval_time,
        "output_dir": str(eval_dir) if eval_dir else None,
    }

    # Calculate overall benchmark metrics
    benchmark_end = time.time()
    total_time = benchmark_end - benchmark_start

    results["total_time_seconds"] = total_time
    results["overall_success"] = all(
        [recon_success, prompt_success, exploit_success, eval_success]
    )

    # Save overall benchmark results
    benchmark_results_file = env["benchmark_dir"] / "benchmark_summary.json"
    with open(benchmark_results_file, "w") as f:
        json.dump(results, f, indent=2)

    if session:
        session.record(
            ActionEvent(
                action_type="benchmark_completed",
                inputs={},
                outputs={
                    "total_time_seconds": total_time,
                    "overall_success": results["overall_success"],
                },
            )
        )
        session.end_session("Success" if results["overall_success"] else "Failure")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"ReconAgent: {'✅ PASSED' if recon_success else '❌ FAILED'} - {recon_time:.2f}s avg per topic"
    )
    print(
        f"PromptEngineerAgent: {'✅ PASSED' if prompt_success else '❌ FAILED'} - {prompt_time:.2f}s avg per prompt"
    )
    print(
        f"ExploitDeliveryAgent: {'✅ PASSED' if exploit_success else '❌ FAILED'} - {exploit_time:.2f}s avg per prompt"
    )
    print(
        f"EvaluationAgent: {'✅ PASSED' if eval_success else '❌ FAILED'} - {eval_time:.2f}s total"
    )
    print(f"{'=' * 60}")
    print(f"Total benchmark time: {total_time:.2f} seconds")
    print(f"Results saved to: {benchmark_results_file}")

    if results["overall_success"]:
        print("\n✅ FULL BENCHMARK COMPLETED SUCCESSFULLY!")
    else:
        print("\n⚠️ BENCHMARK COMPLETED WITH SOME FAILURES")
        print("See error messages above for details.")

    return results


def create_summary_report(benchmarks: List[Dict[str, Any]], output_dir: Path) -> None:
    """Create a summary report comparing multiple benchmarks."""
    if not benchmarks:
        print("No benchmark data to summarize.")
        return

    print("\nCreating benchmark summary report...")

    # Extract summary data
    summary_data = []
    for benchmark in benchmarks:
        model_combo = f"{benchmark['eval_model']} → {benchmark['target_model']}"

        summary_entry = {
            "model_combination": model_combo,
            "eval_model": benchmark["eval_model"],
            "target_model": benchmark["target_model"],
            "total_time_seconds": benchmark.get("total_time_seconds", 0),
            "overall_success": benchmark.get("overall_success", False),
            "topics": len(benchmark.get("topics", [])),
            "prompts_per_topic": benchmark.get("num_prompts", 0),
            "recon_time": benchmark.get("agents", {}).get("recon", {}).get("time", 0),
            "prompt_time": benchmark.get("agents", {})
            .get("prompt_engineer", {})
            .get("time", 0),
            "exploit_time": benchmark.get("agents", {})
            .get("exploit_delivery", {})
            .get("time", 0),
            "eval_time": benchmark.get("agents", {})
            .get("evaluation", {})
            .get("time", 0),
        }

        summary_data.append(summary_entry)

    # Save summary as JSON
    summary_file = output_dir / "benchmark_comparison.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Create a simple markdown table
    md_lines = [
        "# Gray Swan Arena Benchmark Comparison",
        "",
        "## Performance Summary",
        "",
        "| Model Combination | Total Time | Recon Time | Prompt Time | Exploit Time | Eval Time | Success |",
        "|------------------|------------|------------|-------------|--------------|-----------|---------|",
    ]

    for entry in summary_data:
        success_marker = "✅" if entry["overall_success"] else "❌"
        md_lines.append(
            f"| {entry['model_combination']} | "
            f"{entry['total_time_seconds']:.2f}s | "
            f"{entry['recon_time']:.2f}s | "
            f"{entry['prompt_time']:.2f}s | "
            f"{entry['exploit_time']:.2f}s | "
            f"{entry['eval_time']:.2f}s | "
            f"{success_marker} |"
        )

    md_lines.extend(
        [
            "",
            "## Benchmark Details",
            "",
            "- **Topics per benchmark:** " + str(summary_data[0]["topics"]),
            "- **Prompts per topic:** " + str(summary_data[0]["prompts_per_topic"]),
            "- **Timestamp:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",
            "## Interpretation",
            "",
            "The 'Model Combination' column shows which model was used for evaluation (left) and which model was targeted (right).",
            "Times for Recon, Prompt, and Exploit are average times per item, while Eval Time is total processing time.",
        ]
    )

    # Save markdown summary
    md_file = output_dir / "benchmark_comparison.md"
    with open(md_file, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Summary report saved to {md_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Gray Swan Arena agents")

    parser.add_argument(
        "--topics",
        nargs="+",
        default=DEFAULT_TOPICS,
        help="List of topics to use for benchmarking",
    )

    parser.add_argument(
        "--eval-models",
        nargs="+",
        default=["gpt-3.5-turbo"],
        help="Models to use for evaluation (agent operations)",
    )

    parser.add_argument(
        "--target-models",
        nargs="+",
        default=["gpt-3.5-turbo"],
        help="Models to target for exploit delivery",
    )

    parser.add_argument(
        "--prompts-per-topic",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help="Number of prompts to generate per topic",
    )

    return parser.parse_args()


def main():
    """Main entry point for the benchmark script."""
    args = parse_args()

    print("=" * 60)
    print("GRAY SWAN ARENA: AGENT BENCHMARK")
    print("=" * 60)

    # Set up environment
    env = setup_benchmark_environment()

    all_benchmarks = []

    for eval_model in args.eval_models:
        for target_model in args.target_models:
            benchmark_result = run_full_benchmark(
                env=env,
                topics=args.topics,
                eval_model=eval_model,
                target_model=target_model,
                num_prompts=args.prompts_per_topic,
            )
            all_benchmarks.append(benchmark_result)

    # Create summary report if multiple combinations were tested
    if len(all_benchmarks) > 1:
        create_summary_report(all_benchmarks, env["benchmark_dir"])

    # End AgentOps session if active
    session = env.get("agentops_session")
    if session:
        try:
            session.end_session("Success")
        except Exception as e:
            print(f"Error ending AgentOps session: {e}")


if __name__ == "__main__":
    main()
