# API Reference

## Gray Swan Arena Agents

This API reference documents the interfaces for the Gray Swan Arena agents, which focus on AI safety and security testing.

## Model Configuration Options

All Gray Swan Arena agents support configurable model options to optimize performance and cost. The following parameters can be specified when initializing any agent:

- `model_name`: Primary model used for most operations (default: "gpt-4")
- `backup_model`: Model used as fallback for complex tasks (default: None)
- `reasoning_model`: Model specifically used for reasoning tasks (default: same as model_name)

Recommended model configurations:
- For reasoning and analysis tasks: `o3-mini` offers excellent performance with high efficiency
- For creative tasks and complex scenarios: `gpt-4o` provides advanced capabilities

### ReconAgent

```python
from cybersec_agents import ReconAgent

# Basic initialization
agent = ReconAgent(output_dir="./reports", model_name="gpt-4")

# Optimized initialization with specialized models
agent = ReconAgent(
    output_dir="./reports",
    model_name="o3-mini",          # Primary model for most tasks
    backup_model="gpt-4o",         # Fallback for complex analysis
    web_search_model="o3-mini"     # Model for processing web search results
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

##### run_discord_search(target_model: str, target_behavior: str, channels: List[str] = None) -> Dict[str, Any]
Searches Discord channels for information about the target model.

Parameters:
- `target_model`: The target model to search for
- `target_behavior`: The behavior to target
- `channels`: List of Discord channels to search (default: ["ai-ethics", "red-teaming", "vulnerabilities"])

Returns:
- Dictionary containing search results

##### generate_report(target_model: str, target_behavior: str, web_results: Dict[str, Any] = None, discord_results: Dict[str, Any] = None) -> Dict[str, Any]
Generates a comprehensive report based on gathered information.

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
- `target_behavior`: The behavior targeted

Returns:
- Path to the saved report

### PromptEngineerAgent

```python
from cybersec_agents import PromptEngineerAgent

# Basic initialization
agent = PromptEngineerAgent(output_dir="./prompts", model_name="gpt-4")

# Optimized initialization with specialized models
agent = PromptEngineerAgent(
    output_dir="./prompts",
    model_name="gpt-4o",           # Primary model for creative prompt generation
    reasoning_model="o3-mini",     # Efficient model for reasoning about prompt structure
    complexity_threshold=0.7       # Threshold for task complexity evaluation
)
```

#### Methods

##### generate_prompts(target_model: str, target_behavior: str, recon_report: Dict[str, Any], num_prompts: int = 10) -> List[Dict[str, Any]]
Generates attack prompts based on reconnaissance data.

Parameters:
- `target_model`: The target model
- `target_behavior`: The behavior to target
- `recon_report`: Reconnaissance report from ReconAgent
- `num_prompts`: Number of prompts to generate (default: 10)

Returns:
- List of prompts as dictionaries

##### save_prompts(prompts: List[Dict[str, Any]], target_model: str, target_behavior: str) -> str
Saves the generated prompts to a file.

Parameters:
- `prompts`: The prompts to save
- `target_model`: The target model
- `target_behavior`: The behavior targeted

Returns:
- Path to the saved prompts

### ExploitDeliveryAgent

```python
from cybersec_agents import ExploitDeliveryAgent

# Basic initialization
agent = ExploitDeliveryAgent(output_dir="./exploits", model_name="gpt-4")

# Optimized initialization with specialized models
agent = ExploitDeliveryAgent(
    output_dir="./exploits",
    model_name="o3-mini",          # Efficient model for delivery mechanics
    backup_model="gpt-4o",         # Backup for complex scenarios
    analysis_model="o3-mini"       # Model for analyzing initial responses
)
```

#### Methods

##### run_prompts(prompts: List[Dict[str, Any]], target_model: str, target_behavior: str, method: str = "api", max_tries: int = 3, delay_between_tries: int = 2) -> List[Dict[str, Any]]
Executes prompts against the target model.

Parameters:
- `prompts`: List of prompts from PromptEngineerAgent
- `target_model`: The target model
- `target_behavior`: The behavior being targeted
- `method`: Method for running prompts ("api" or "website") (default: "api")
- `max_tries`: Maximum number of attempts per prompt (default: 3)
- `delay_between_tries`: Delay between attempts in seconds (default: 2)

Returns:
- List of results as dictionaries

##### save_results(results: List[Dict[str, Any]], target_model: str, target_behavior: str) -> str
Saves the exploit results to a file.

Parameters:
- `results`: The results to save
- `target_model`: The target model
- `target_behavior`: The behavior targeted

Returns:
- Path to the saved results

### EvaluationAgent

```python
from cybersec_agents import EvaluationAgent

# Basic initialization
agent = EvaluationAgent(output_dir="./evaluations", model_name="gpt-4")

# Optimized initialization with specialized models
agent = EvaluationAgent(
    output_dir="./evaluations",
    model_name="o3-mini",          # Default model for most tasks
    reasoning_model="o3-mini",     # Model for classification and reasoning
    visualization_model="gpt-4o",  # Model for complex visualization planning
    backup_model="gpt-4o"          # Backup model for challenging cases
)
```

#### Methods

##### evaluate_results(results: List[Dict[str, Any]], target_model: str, target_behavior: str) -> Dict[str, Any]
Evaluates the exploit results.

Parameters:
- `results`: List of results from ExploitDeliveryAgent
- `target_model`: The target model
- `target_behavior`: The behavior targeted

Returns:
- Dictionary containing evaluation metrics

##### save_evaluation(evaluation: Dict[str, Any], target_model: str, target_behavior: str) -> str
Saves the evaluation to a file.

Parameters:
- `evaluation`: The evaluation to save
- `target_model`: The target model
- `target_behavior`: The behavior targeted

Returns:
- Path to the saved evaluation

##### generate_summary(evaluation: Dict[str, Any], target_model: str, target_behavior: str) -> Dict[str, Any]
Generates a summary report of the evaluation.

Parameters:
- `evaluation`: Evaluation data
- `target_model`: The target model
- `target_behavior`: The behavior targeted

Returns:
- Dictionary containing the summary report

##### save_summary(summary: Dict[str, Any], target_model: str, target_behavior: str) -> str
Saves the summary report to a file.

Parameters:
- `summary`: The summary to save
- `target_model`: The target model
- `target_behavior`: The behavior targeted

Returns:
- Path to the saved summary

##### create_visualizations(evaluation: Dict[str, Any], target_model: str, target_behavior: str) -> List[str]
Creates visualizations based on the evaluation data.

Parameters:
- `evaluation`: Evaluation data
- `target_model`: The target model
- `target_behavior`: The behavior targeted

Returns:
- List of paths to generated visualizations

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
