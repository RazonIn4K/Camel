# API Reference

## Gray Swan Arena Agents

This API reference documents the interfaces for the Gray Swan Arena agents, which focus on AI safety and security testing.

## Model Configuration Options

All Gray Swan Arena agents support configurable model options to optimize performance and cost. The following parameters can be specified when initializing any agent:

- `model_name`: Primary model used for most operations (default: "gpt-4")
- `reasoning_model`: Model specifically used for reasoning tasks (default: same as model_name)
- `backup_model`: Model used as fallback if the primary model fails (default: None)

Recommended model configurations:
- For reasoning tasks: `o3-mini` offers excellent performance with high efficiency
- For primary operations: `gpt-4o` provides advanced capabilities
- For backup operations: `gpt-4` offers reliable fallback capabilities

### ReconAgent

```python
from cybersec_agents.grayswan.agents import ReconAgent

# Basic initialization
agent = ReconAgent(output_dir="./reports", model_name="gpt-4")

# Optimized initialization with specialized models
agent = ReconAgent(
    output_dir="./reports",
    model_name="gpt-4o",          # Primary model for most tasks
    reasoning_model="o3-mini",    # Model for reasoning tasks like report generation
    backup_model="gpt-4"          # Fallback if primary model fails
)
```

#### Methods

##### run_web_search(target_model: str, target_behavior: str, num_results: int = 5) -> Dict[str, Any]
Runs web search to gather information about the target model.

Parameters:
- `target_model`: The target model to search for
- `target_behavior`: The behavior to target
- `num_results`: Number of results to gather (default: 5)

Returns:
- Dictionary containing search results

##### run_discord_search(target_model: str, target_behavior: str, num_results: int = 5) -> Dict[str, Any]
Searches Discord channels for information about the target model.

Parameters:
- `target_model`: The target model to search for
- `target_behavior`: The behavior to target
- `num_results`: Number of results to gather (default: 5)

Returns:
- Dictionary containing search results

##### generate_report(target_model: str, target_behavior: str, web_results: Dict[str, Any] = None, discord_results: Dict[str, Any] = None) -> Dict[str, Any]
Generates a comprehensive report based on collected data.

Parameters:
- `target_model`: The target model
- `target_behavior`: The behavior to target
- `web_results`: Results from web search (optional)
- `discord_results`: Results from Discord search (optional)

Returns:
- Dictionary containing the report

##### save_report(report: Dict[str, Any], target_model: str, target_behavior: str) -> str
Saves the report to a file.

Parameters:
- `report`: The report to save
- `target_model`: The target model
- `target_behavior`: The behavior to target

Returns:
- Path to the saved report

### PromptEngineerAgent

```python
from cybersec_agents.grayswan.agents import PromptEngineerAgent

# Basic initialization
agent = PromptEngineerAgent(output_dir="./prompts", model_name="gpt-4")

# Optimized initialization with specialized models
agent = PromptEngineerAgent(
    output_dir="./prompts",
    model_name="gpt-4o",          # Primary model for creative tasks
    reasoning_model="o3-mini",    # Model for reasoning about prompt structure
    backup_model="gpt-4"          # Fallback if primary model fails
)
```

#### Methods

##### generate_prompts(target_model: str, target_behavior: str, recon_report: Dict[str, Any], num_prompts: int = 10) -> List[Dict[str, Any]]
Generates prompts based on reconnaissance data.

Parameters:
- `target_model`: The target model
- `target_behavior`: The behavior to target
- `recon_report`: Report from the reconnaissance phase
- `num_prompts`: Number of prompts to generate (default: 10)

Returns:
- List of generated prompts

##### save_prompts(prompts: List[Dict[str, Any]], target_model: str, target_behavior: str) -> str
Saves the prompts to a file.

Parameters:
- `prompts`: The prompts to save
- `target_model`: The target model
- `target_behavior`: The behavior to target

Returns:
- Path to the saved prompts

### ExploitDeliveryAgent

```python
from cybersec_agents.grayswan.agents import ExploitDeliveryAgent

# Basic initialization
agent = ExploitDeliveryAgent(output_dir="./exploits", model_name="gpt-4")

# Optimized initialization with specialized models
agent = ExploitDeliveryAgent(
    output_dir="./exploits",
    model_name="gpt-4o",          # Primary model for delivery tasks
    backup_model="gpt-4"          # Fallback if primary model fails
)
```

#### Methods

##### run_prompts(prompts: List[Dict[str, Any]], target_model: str, method: str = "api", max_tries: int = 3, delay_between_tries: int = 2) -> List[Dict[str, Any]]
Runs prompts against the target model.

Parameters:
- `prompts`: The prompts to run
- `target_model`: The target model
- `method`: Method to use (api or browser) (default: "api")
- `max_tries`: Maximum number of tries per prompt (default: 3)
- `delay_between_tries`: Delay between tries in seconds (default: 2)

Returns:
- List of results

##### save_results(results: List[Dict[str, Any]], target_model: str, target_behavior: str) -> str
Saves the results to a file.

Parameters:
- `results`: The results to save
- `target_model`: The target model
- `target_behavior`: The behavior to target

Returns:
- Path to the saved results

### EvaluationAgent

```python
from cybersec_agents.grayswan.agents import EvaluationAgent

# Basic initialization
agent = EvaluationAgent(output_dir="./evaluations", model_name="gpt-4")

# Optimized initialization with specialized models
agent = EvaluationAgent(
    output_dir="./evaluations",
    model_name="gpt-4o",          # Primary model for evaluation tasks
    reasoning_model="o3-mini",    # Model for reasoning about results
    backup_model="gpt-4"          # Fallback if primary model fails
)
```

#### Methods

##### evaluate_results(results: List[Dict[str, Any]], target_model: str, target_behavior: str) -> Dict[str, Any]
Evaluates the results of exploit attempts.

Parameters:
- `results`: The results to evaluate
- `target_model`: The target model
- `target_behavior`: The behavior to target

Returns:
- Dictionary containing the evaluation

##### save_evaluation(evaluation: Dict[str, Any], target_model: str, target_behavior: str) -> str
Saves the evaluation to a file.

Parameters:
- `evaluation`: The evaluation to save
- `target_model`: The target model
- `target_behavior`: The behavior to target

Returns:
- Path to the saved evaluation

##### generate_summary(evaluation: Dict[str, Any], target_model: str, target_behavior: str) -> Dict[str, Any]
Generates a summary of the evaluation.

Parameters:
- `evaluation`: The evaluation to summarize
- `target_model`: The target model
- `target_behavior`: The behavior to target

Returns:
- Dictionary containing the summary

##### save_summary(summary: Dict[str, Any], target_model: str, target_behavior: str) -> str
Saves the summary to a file.

Parameters:
- `summary`: The summary to save
- `target_model`: The target model
- `target_behavior`: The behavior to target

Returns:
- Path to the saved summary

##### create_visualizations(evaluation: Dict[str, Any], target_model: str, target_behavior: str) -> List[str]
Creates visualizations based on the evaluation.

Parameters:
- `evaluation`: The evaluation to visualize
- `target_model`: The target model
- `target_behavior`: The behavior to target

Returns:
- List of paths to the created visualizations

## Network Anomaly Detector

### NetworkAnomalyDetector

This class provides tools for analyzing network scan results, particularly Nmap outputs, to identify potential threats and security issues.

```python
from cybersec_agents import NetworkAnomalyDetector

detector = NetworkAnomalyDetector()
```

#### Methods

##### analyze_nmap_output(nmap_file: str) -> Dict[str, Any]
Analyzes an Nmap output file to identify potential threats, vulnerabilities, and security recommendations.

Parameters:
- `nmap_file`: Path to the Nmap output file

Returns:
- Dictionary containing identified threats, security recommendations, and an overall risk assessment

Example usage:
```python
from cybersec_agents import NetworkAnomalyDetector

detector = NetworkAnomalyDetector()
results = detector.analyze_nmap_output("scan_results.xml")
print(f"Identified {len(results['threats'])} potential threats")
print(f"Risk assessment: {results['risk_assessment']}")
```

## Main Module

The main module provides functions for running the Gray Swan Arena pipeline.

```python
from cybersec_agents.grayswan import main

# Run the full pipeline with default settings
main.main()

# Run with custom settings
main.main(
    target_model="GPT-4",
    target_behavior="bypass content filters",
    output_dir="./output",
    model_name="gpt-4o",
    reasoning_model="o3-mini",
    backup_model="gpt-4",
    method="api",
    parallel=True
)
```

### Functions

#### main(target_model: str = "GPT-4", target_behavior: str = "bypass content filters", output_dir: str = "./output", model_name: str = "gpt-4", reasoning_model: Optional[str] = None, backup_model: Optional[str] = None, method: str = "api", parallel: bool = False) -> Dict[str, Any]
Runs the full Gray Swan Arena pipeline.

Parameters:
- `target_model`: The target model to test (default: "GPT-4")
- `target_behavior`: The behavior to target (default: "bypass content filters")
- `output_dir`: Base directory for output (default: "./output")
- `model_name`: Name of the model to use for agents (default: "gpt-4")
- `reasoning_model`: Model to use for reasoning tasks (default: None)
- `backup_model`: Backup model to use if primary fails (default: None)
- `method`: Method to use for exploit delivery (api or browser) (default: "api")
- `parallel`: Whether to run reconnaissance in parallel (default: False)

Returns:
- Dictionary containing results from all phases

#### run_reconnaissance(target_model: str, target_behavior: str, output_dir: str = "./reports", model_name: str = "gpt-4", reasoning_model: Optional[str] = None, backup_model: Optional[str] = None) -> Dict[str, Any]
Runs the reconnaissance phase of the Gray Swan Arena pipeline.

Parameters:
- `target_model`: The target model to test
- `target_behavior`: The behavior to target
- `output_dir`: Directory to save reports (default: "./reports")
- `model_name`: Name of the model to use for the agent (default: "gpt-4")
- `reasoning_model`: Model to use for reasoning tasks (default: None)
- `backup_model`: Backup model to use if primary fails (default: None)

Returns:
- Dictionary containing the reconnaissance report

#### run_prompt_engineering(target_model: str, target_behavior: str, recon_report: Dict[str, Any], output_dir: str = "./prompts", model_name: str = "gpt-4", reasoning_model: Optional[str] = None, backup_model: Optional[str] = None, num_prompts: int = 10) -> Dict[str, Any]
Runs the prompt engineering phase of the Gray Swan Arena pipeline.

Parameters:
- `target_model`: The target model to test
- `target_behavior`: The behavior to target
- `recon_report`: Report from the reconnaissance phase
- `output_dir`: Directory to save prompts (default: "./prompts")
- `model_name`: Name of the model to use for the agent (default: "gpt-4")
- `reasoning_model`: Model to use for reasoning tasks (default: None)
- `backup_model`: Backup model to use if primary fails (default: None)
- `num_prompts`: Number of prompts to generate (default: 10)

Returns:
- Dictionary containing the generated prompts

#### run_exploit_delivery(target_model: str, target_behavior: str, prompts: List[Dict[str, Any]], output_dir: str = "./exploits", model_name: str = "gpt-4", backup_model: Optional[str] = None, method: str = "api") -> Dict[str, Any]
Runs the exploit delivery phase of the Gray Swan Arena pipeline.

Parameters:
- `target_model`: The target model to test
- `target_behavior`: The behavior to target
- `prompts`: Prompts from the previous phase
- `output_dir`: Directory to save exploit results (default: "./exploits")
- `model_name`: Name of the model to use for the agent (default: "gpt-4")
- `backup_model`: Backup model to use if primary fails (default: None)
- `method`: Method to use for exploit delivery (api or browser) (default: "api")

Returns:
- Dictionary containing the exploit results

#### run_evaluation(target_model: str, target_behavior: str, results: List[Dict[str, Any]], output_dir: str = "./evaluations", model_name: str = "gpt-4", reasoning_model: Optional[str] = None, backup_model: Optional[str] = None) -> Dict[str, Any]
Runs the evaluation phase of the Gray Swan Arena pipeline.

Parameters:
- `target_model`: The target model to test
- `target_behavior`: The behavior to target
- `results`: Results from the previous phase
- `output_dir`: Directory to save evaluations (default: "./evaluations")
- `model_name`: Name of the model to use for the agent (default: "gpt-4")
- `reasoning_model`: Model to use for reasoning tasks (default: None)
- `backup_model`: Backup model to use if primary fails (default: None)

Returns:
- Dictionary containing the evaluation
