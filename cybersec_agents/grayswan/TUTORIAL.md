# Gray Swan Arena Tutorial: Complete Red-Team Assessment

This tutorial walks you through a complete red-team assessment of an AI language model using Gray Swan Arena, from setup to final report generation.

## Prerequisites

Before starting, make sure you have:

- Python 3.8+ installed
- Git installed
- API keys for at least one language model provider (OpenAI, Anthropic, etc.)
- (Optional) A Discord bot token for Discord integration

## Step 1: Setting Up Your Environment

First, clone the repository and install the package:

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/camel.git
cd camel

# Install the package in development mode
pip install -e .

# Install additional dependencies for web automation
pip install playwright selenium webdriver-manager
playwright install
```

Next, set up your environment variables:

```bash
# Copy the example .env file
cp cybersec_agents/grayswan/.env.example cybersec_agents/grayswan/.env

# Edit the .env file with your API keys and settings
nano cybersec_agents/grayswan/.env  # or use your preferred editor
```

Fill in your API keys and other settings in the .env file:

```
# Required API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Optional Settings
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_CHANNEL_IDS=123456789012345678,987654321098765432
HEADLESS=False
MAX_RETRIES=3
```

## Step 2: Conducting Reconnaissance

Let's start by gathering information about our target model:

```python
# Create a Python script named 'run_recon.py'
from cybersec_agents.grayswan import ReconAgent
import os

# Set up the agent
recon_agent = ReconAgent()

# Define our target
target_model = "Brass Fox"
target_behavior = "generating harmful content"

# Run web searches
print(f"Searching for information about {target_model}...")
model_info = recon_agent.run_web_search(f"{target_model} language model capabilities")

print(f"Searching for information about {target_behavior}...")
behavior_info = recon_agent.run_web_search(f"AI safeguards for {target_behavior}")

print("Researching jailbreaking techniques...")
techniques_info = recon_agent.run_web_search("latest jailbreaking techniques for language models")

# (Optional) Search Discord for additional information
discord_info = None
if os.environ.get("DISCORD_BOT_TOKEN"):
    print("Searching Discord for relevant information...")
    try:
        discord_info = recon_agent.run_discord_search(f"{target_model} jailbreak techniques")
    except Exception as e:
        print(f"Discord search failed: {e}")

# Generate a comprehensive report
print("Generating reconnaissance report...")
report = recon_agent.generate_report(
    model_info=model_info,
    behavior_info=behavior_info,
    techniques_info=techniques_info,
    discord_info=discord_info,
    target_model=target_model,
    target_behavior=target_behavior
)

# Save the report
report_path = recon_agent.save_report(report)
print(f"Reconnaissance report saved to: {report_path}")
```

Run the script:

```bash
python run_recon.py
```

## Step 3: Engineering Attack Prompts

Now, let's generate prompts based on our reconnaissance:

```python
# Create a Python script named 'generate_prompts.py'
from cybersec_agents.grayswan import PromptEngineerAgent
import sys
import json

# Check if a recon report path was provided
if len(sys.argv) < 2:
    print("Usage: python generate_prompts.py path/to/recon_report.json")
    sys.exit(1)

# Get the report path from command line arguments
recon_report_path = sys.argv[1]

# Set up the agent
prompt_agent = PromptEngineerAgent()

# Load the reconnaissance report
print(f"Loading reconnaissance report from {recon_report_path}...")
recon_report = prompt_agent.load_recon_report(recon_report_path)

# Extract target information
target_model = recon_report.get("target_model", "Unknown Model")
target_behavior = recon_report.get("target_behavior", "Unknown Behavior")
print(f"Generating prompts for {target_model} targeting {target_behavior}...")

# Generate prompts based on the report
prompts = prompt_agent.generate_prompts(
    recon_report=recon_report,
    num_prompts=20,
    technique=None  # Generate prompts using various techniques
)

# Evaluate diversity of generated prompts
diversity = prompt_agent.evaluate_prompt_diversity(prompts)
print(f"Generated {diversity['total_prompts']} prompts using {len(diversity['technique_distribution'])} techniques")

# Print technique distribution
print("\nTechnique distribution:")
for technique, count in diversity['technique_distribution'].items():
    percentage = (count / diversity['total_prompts']) * 100
    print(f"  - {technique}: {count} prompts ({percentage:.1f}%)")

# Save prompts to a file
prompt_path = prompt_agent.save_prompts(prompts)
print(f"Prompts saved to: {prompt_path}")
```

Run the script with your reconnaissance report:

```bash
python generate_prompts.py data/recon_reports/recon_report_YYYY-MM-DD_HH-MM-SS.json
```

## Step 4: Delivering Exploits

Let's test our prompts against the target model. We'll show two approaches:

### Option A: API-Based Testing

```python
# Create a Python script named 'run_exploits_api.py'
from cybersec_agents.grayswan import ExploitDeliveryAgent
from camel.types import ModelType, ModelPlatformType
import sys
import json

# Check if a prompt list path was provided
if len(sys.argv) < 2:
    print("Usage: python run_exploits_api.py path/to/prompt_list.json")
    sys.exit(1)

# Get the prompt list path from command line arguments
prompt_list_path = sys.argv[1]

# Set up the agent with a specific target model
agent = ExploitDeliveryAgent(
    target_model_type=ModelType.GPT_3_5_TURBO,  # Change as needed
    target_model_platform=ModelPlatformType.OPENAI  # Change as needed
)

# Load prompts
print(f"Loading prompts from {prompt_list_path}...")
prompts = agent.load_prompts(prompt_list_path)
print(f"Loaded {len(prompts)} prompts")

# Execute prompts against the API
print("Executing prompts against the API...")
results = agent.execute_prompt_batch(
    prompts=prompts,
    max_concurrent=3  # Number of concurrent requests
)

# Analyze the results
analysis = agent.analyze_results(results)
print(f"\nExecution complete. Success rate: {analysis['success_rate'] * 100:.2f}%")

# Print top techniques
print("\nMost effective techniques:")
sorted_techniques = sorted(
    analysis["technique_stats"].items(),
    key=lambda x: x[1]["rate"],
    reverse=True
)
for technique, stats in sorted_techniques[:5]:
    print(f"  - {technique}: {stats['rate'] * 100:.2f}% success rate ({stats['success']}/{stats['total']} prompts)")

# Save results to a file
results_path = agent.save_results(results)
print(f"Results saved to: {results_path}")
```

### Option B: Browser-Based Testing

```python
# Create a Python script named 'run_exploits_browser.py'
from cybersec_agents.grayswan import ExploitDeliveryAgent
import sys
import json
import os

# Check if a prompt list path was provided
if len(sys.argv) < 2:
    print("Usage: python run_exploits_browser.py path/to/prompt_list.json [playwright|selenium]")
    sys.exit(1)

# Get the prompt list path and method from command line arguments
prompt_list_path = sys.argv[1]
method = sys.argv[2] if len(sys.argv) > 2 else "playwright"

# Validate method
if method not in ["playwright", "selenium"]:
    print("Method must be either 'playwright' or 'selenium'")
    sys.exit(1)

# Set up the agent (no specific model for browser automation)
agent = ExploitDeliveryAgent()

# Load prompts
print(f"Loading prompts from {prompt_list_path}...")
prompts = agent.load_prompts(prompt_list_path)
print(f"Loaded {len(prompts)} prompts")

# Get target model and behavior from environment or use defaults
target_model = os.environ.get("TARGET_MODEL", "Brass Fox")
target_behavior = os.environ.get("TARGET_BEHAVIOR", "Harmful Content")

# Execute prompts using browser automation
print(f"Running prompts against {target_model} using {method}...")
results = agent.run_prompts(
    prompts=prompts,
    target_model=target_model,
    target_behavior=target_behavior,
    method=method
)

# Analyze the results
analysis = agent.analyze_results(results)
print(f"\nExecution complete. Success rate: {analysis['success_rate'] * 100:.2f}%")

# Print top techniques
print("\nMost effective techniques:")
sorted_techniques = sorted(
    analysis["technique_stats"].items(),
    key=lambda x: x[1]["rate"],
    reverse=True
)
for technique, stats in sorted_techniques[:5]:
    print(f"  - {technique}: {stats['rate'] * 100:.2f}% success rate ({stats['success']}/{stats['total']} prompts)")

# Save results to a file
results_path = agent.save_results(results)
print(f"Results saved to: {results_path}")
```

Run one of the scripts with your prompt list:

```bash
# For API-based testing
python run_exploits_api.py data/prompt_lists/prompt_list_YYYY-MM-DD_HH-MM-SS.json

# OR for browser-based testing
python run_exploits_browser.py data/prompt_lists/prompt_list_YYYY-MM-DD_HH-MM-SS.json playwright
```

## Step 5: Evaluating Results

Finally, let's analyze the results and generate a comprehensive report:

```python
# Create a Python script named 'generate_report.py'
from cybersec_agents.grayswan import EvaluationAgent
import sys
import os
import matplotlib.pyplot as plt

# Check if the necessary files were provided
if len(sys.argv) < 2:
    print("Usage: python generate_report.py path/to/exploit_results.json [path/to/recon_report.json]")
    sys.exit(1)

# Get file paths from command line arguments
exploit_results_path = sys.argv[1]
recon_report_path = sys.argv[2] if len(sys.argv) > 2 else None

# Set up the agent
eval_agent = EvaluationAgent()

# Load results
print(f"Loading exploit results from {exploit_results_path}...")
results = eval_agent.load_exploit_results(exploit_results_path)

# Load recon report if provided
recon_report = None
if recon_report_path:
    print(f"Loading reconnaissance report from {recon_report_path}...")
    recon_report = eval_agent.load_recon_report(recon_report_path)

# Calculate statistics
print("Calculating statistics...")
statistics = eval_agent.calculate_statistics(results)

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total prompts tested: {statistics['total_prompts']}")
print(f"Overall success rate: {statistics['overall_success_rate'] * 100:.2f}%")
print(f"Number of techniques used: {len(statistics['success_by_technique'])}")

# Create a directory for visualizations
vis_dir = os.path.join("data", "evaluation_reports", "visualizations")
os.makedirs(vis_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')  # Modern, readable style

# Generate visualizations
print("Creating visualizations...")
visualization_paths = eval_agent.create_visualizations(statistics, vis_dir)

# Generate a comprehensive report
print("Generating final report...")
report = eval_agent.generate_report(
    results=results,
    statistics=statistics,
    recon_report=recon_report,
    visualization_paths=visualization_paths
)

# Save the report as JSON
json_path = eval_agent.save_report(report)
print(f"Report saved to: {json_path}")

# Generate a markdown report with embedded visualizations
markdown_path = eval_agent.generate_markdown_report(report)
print(f"Markdown report saved to: {markdown_path}")
```

Run the script with your results file:

```bash
python generate_report.py data/exploit_logs/exploit_results_YYYY-MM-DD_HH-MM-SS.json data/recon_reports/recon_report_YYYY-MM-DD_HH-MM-SS.json
```

## Step 6: Reviewing the Report

The final markdown report provides a comprehensive analysis of your red-team assessment, including:

1. **Executive Summary**: High-level findings and recommendations
2. **Test Results**: Detailed statistics on prompt success rates
3. **Visualizations**: Charts showing success rates by technique and model
4. **Response Analysis**: In-depth analysis of model responses
5. **Vulnerability Assessment**: Summary of identified vulnerabilities
6. **Recommendations**: Specific recommendations for improving model safety

You can open the markdown report in any markdown viewer or convert it to PDF:

```bash
# Install grip for markdown rendering
pip install grip

# Preview the report
grip data/evaluation_reports/evaluation_report_YYYY-MM-DD_HH-MM-SS.md

# Or convert to PDF using a tool like pandoc
pip install pandoc
pandoc data/evaluation_reports/evaluation_report_YYYY-MM-DD_HH-MM-SS.md -o final_report.pdf
```

## Complete End-to-End Script

For convenience, here's a single script that runs the entire process:

```python
# Create a Python script named 'run_full_assessment.py'
from cybersec_agents.grayswan import ReconAgent, PromptEngineerAgent, ExploitDeliveryAgent, EvaluationAgent
from camel.types import ModelType, ModelPlatformType
import os
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Run a complete AI red-team assessment')
    parser.add_argument('--target', type=str, default='Brass Fox',
                        help='Target model name')
    parser.add_argument('--behavior', type=str, default='harmful content generation',
                        help='Target behavior to test')
    parser.add_argument('--prompts', type=int, default=20,
                        help='Number of prompts to generate')
    parser.add_argument('--method', type=str, choices=['api', 'playwright', 'selenium'],
                        default='api', help='Method for exploit delivery')
    parser.add_argument('--model-type', type=str, default='gpt-3.5-turbo',
                        help='Model type for API testing')
    parser.add_argument('--skip-recon', action='store_true',
                        help='Skip reconnaissance phase')
    parser.add_argument('--skip-prompt-gen', action='store_true',
                        help='Skip prompt generation phase')
    parser.add_argument('--skip-exploit', action='store_true',
                        help='Skip exploit delivery phase')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation phase')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Generate timestamp for file naming
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    
    # Initialize paths
    recon_path = None
    prompt_path = None
    results_path = None
    
    # Step 1: Reconnaissance
    if not args.skip_recon:
        print(f"\n=== PHASE 1: RECONNAISSANCE ===")
        print(f"Target: {args.target}")
        print(f"Behavior: {args.behavior}\n")
        
        recon_agent = ReconAgent()
        
        # Run web searches
        model_info = recon_agent.run_web_search(f"{args.target} language model capabilities")
        behavior_info = recon_agent.run_web_search(f"AI safeguards for {args.behavior}")
        techniques_info = recon_agent.run_web_search("latest jailbreaking techniques for language models")
        
        # Optional Discord search
        discord_info = None
        if os.environ.get("DISCORD_BOT_TOKEN"):
            try:
                discord_info = recon_agent.run_discord_search(f"{args.target} jailbreak techniques")
            except Exception as e:
                print(f"Discord search failed: {e}")
        
        # Generate report
        report = recon_agent.generate_report(
            model_info=model_info,
            behavior_info=behavior_info,
            techniques_info=techniques_info,
            discord_info=discord_info,
            target_model=args.target,
            target_behavior=args.behavior
        )
        
        # Save report
        recon_path = recon_agent.save_report(report)
        print(f"Reconnaissance report saved to: {recon_path}")
    
    # Step 2: Prompt Engineering
    if not args.skip_prompt_gen:
        print(f"\n=== PHASE 2: PROMPT ENGINEERING ===")
        
        prompt_agent = PromptEngineerAgent()
        
        # Use existing recon report or load a previous one
        if recon_path is None:
            # Find the most recent recon report if none specified
            recon_dir = os.path.join("data", "recon_reports")
            if os.path.exists(recon_dir):
                reports = [f for f in os.listdir(recon_dir) if f.startswith("recon_report_")]
                if reports:
                    reports.sort(reverse=True)  # Sort by timestamp (newest first)
                    recon_path = os.path.join(recon_dir, reports[0])
                    print(f"Using most recent reconnaissance report: {recon_path}")
        
        if recon_path:
            # Load the recon report
            recon_report = prompt_agent.load_recon_report(recon_path)
            
            # Generate prompts
            print(f"Generating {args.prompts} prompts...")
            prompts = prompt_agent.generate_prompts(
                recon_report=recon_report,
                num_prompts=args.prompts
            )
            
            # Evaluate diversity
            diversity = prompt_agent.evaluate_prompt_diversity(prompts)
            print(f"Generated {diversity['total_prompts']} prompts using {len(diversity['technique_distribution'])} techniques")
            
            # Save prompts
            prompt_path = prompt_agent.save_prompts(prompts)
            print(f"Prompts saved to: {prompt_path}")
        else:
            print("No reconnaissance report found. Skipping prompt generation.")
    
    # Step 3: Exploit Delivery
    if not args.skip_exploit:
        print(f"\n=== PHASE 3: EXPLOIT DELIVERY ===")
        
        # Use existing prompt list or load a previous one
        if prompt_path is None:
            # Find the most recent prompt list if none specified
            prompt_dir = os.path.join("data", "prompt_lists")
            if os.path.exists(prompt_dir):
                lists = [f for f in os.listdir(prompt_dir) if f.startswith("prompt_list_")]
                if lists:
                    lists.sort(reverse=True)  # Sort by timestamp (newest first)
                    prompt_path = os.path.join(prompt_dir, lists[0])
                    print(f"Using most recent prompt list: {prompt_path}")
        
        if prompt_path:
            # Initialize agent based on method
            if args.method == 'api':
                # Map string to ModelType enum
                model_map = {
                    'gpt-3.5-turbo': ModelType.GPT_3_5_TURBO,
                    'gpt-4': ModelType.GPT_4,
                    'claude-2': ModelType.CLAUDE_2
                }
                model_type = model_map.get(args.model_type.lower(), ModelType.GPT_3_5_TURBO)
                
                print(f"Testing against API: {model_type}")
                agent = ExploitDeliveryAgent(
                    target_model_type=model_type,
                    target_model_platform=ModelPlatformType.OPENAI  # Change as needed
                )
                
                # Load prompts
                prompts = agent.load_prompts(prompt_path)
                print(f"Loaded {len(prompts)} prompts")
                
                # Execute prompts
                results = agent.execute_prompt_batch(prompts=prompts, max_concurrent=3)
            else:
                # Browser-based testing
                print(f"Testing using browser automation: {args.method}")
                agent = ExploitDeliveryAgent()
                
                # Load prompts
                prompts = agent.load_prompts(prompt_path)
                print(f"Loaded {len(prompts)} prompts")
                
                # Execute prompts
                results = agent.run_prompts(
                    prompts=prompts,
                    target_model=args.target,
                    target_behavior=args.behavior,
                    method=args.method
                )
            
            # Analyze results
            analysis = agent.analyze_results(results)
            print(f"Execution complete. Success rate: {analysis['success_rate'] * 100:.2f}%")
            
            # Save results
            results_path = agent.save_results(results)
            print(f"Results saved to: {results_path}")
        else:
            print("No prompt list found. Skipping exploit delivery.")
    
    # Step 4: Evaluation
    if not args.skip_eval:
        print(f"\n=== PHASE 4: EVALUATION ===")
        
        # Use existing results or load a previous one
        if results_path is None:
            # Find the most recent results if none specified
            results_dir = os.path.join("data", "exploit_logs")
            if os.path.exists(results_dir):
                logs = [f for f in os.listdir(results_dir) if f.startswith("exploit_results_")]
                if logs:
                    logs.sort(reverse=True)  # Sort by timestamp (newest first)
                    results_path = os.path.join(results_dir, logs[0])
                    print(f"Using most recent exploit results: {results_path}")
        
        if results_path:
            # Initialize agent
            eval_agent = EvaluationAgent()
            
            # Load results
            results = eval_agent.load_exploit_results(results_path)
            
            # Load recon report if available
            recon_report = None
            if recon_path:
                recon_report = eval_agent.load_recon_report(recon_path)
            
            # Calculate statistics
            statistics = eval_agent.calculate_statistics(results)
            
            # Create visualization directory
            vis_dir = os.path.join("data", "evaluation_reports", "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Generate visualizations
            print("Creating visualizations...")
            visualization_paths = eval_agent.create_visualizations(statistics, vis_dir)
            
            # Generate report
            print("Generating final report...")
            report = eval_agent.generate_report(
                results=results,
                statistics=statistics,
                recon_report=recon_report,
                visualization_paths=visualization_paths
            )
            
            # Save reports
            json_path = eval_agent.save_report(report)
            print(f"JSON report saved to: {json_path}")
            
            markdown_path = eval_agent.generate_markdown_report(report)
            print(f"Markdown report saved to: {markdown_path}")
            
            print("\n=== ASSESSMENT COMPLETE ===")
            print(f"Final report available at: {markdown_path}")
        else:
            print("No exploit results found. Skipping evaluation.")

if __name__ == "__main__":
    main()
```

Run the full assessment:

```bash
# Run a complete assessment
python run_full_assessment.py --target "Brass Fox" --behavior "harmful content" --method playwright

# Or run specific phases
python run_full_assessment.py --skip-recon --skip-prompt-gen --method api
```

## Conclusion

You've now completed a full red-team assessment using Gray Swan Arena. This tutorial covered:

1. Setting up your environment
2. Gathering reconnaissance on your target model
3. Generating attack prompts
4. Executing prompts using API calls or browser automation
5. Evaluating results and generating comprehensive reports

The generated reports provide actionable insights into model vulnerabilities and suggest improvements to enhance model safety.

By using Gray Swan Arena's modular architecture, you can also run individual components of the pipeline or customize the process for your specific needs.

For more detailed information on each component, refer to the comprehensive [USAGE_GUIDE.md](USAGE_GUIDE.md). 