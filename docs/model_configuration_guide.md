# Gray Swan Arena Model Configuration Guide

This guide provides detailed information about configuring models for Gray Swan Arena agents to optimize performance, cost, and effectiveness.

## Table of Contents

1. [Model Selection Strategy](#model-selection-strategy)
2. [Using o3-mini for Reasoning Tasks](#using-o3-mini-for-reasoning-tasks)
3. [Configuring GPT-4o as a Backup Model](#configuring-gpt-4o-as-a-backup-model)
4. [Agent-Specific Configurations](#agent-specific-configurations)
5. [End-to-End Pipeline Example](#end-to-end-pipeline-example)

## Model Selection Strategy

The Gray Swan Arena framework supports configuring different models for different tasks to optimize performance and efficiency. The general strategy is:

1. **Use o3-mini for reasoning tasks**: The o3-mini model provides excellent performance for tasks requiring logical reasoning, classification, and analysis, with significantly improved efficiency.

2. **Use GPT-4o for creative tasks**: For tasks requiring creative generation, complex reasoning, or handling edge cases, GPT-4o provides superior capabilities.

3. **Configure backup models**: Set up backup models to handle cases where the primary model might not have sufficient capabilities.

## Using o3-mini for Reasoning Tasks

The o3-mini model is particularly well-suited for reasoning tasks in the Gray Swan Arena pipeline, such as:

- Analyzing reconnaissance data
- Evaluating exploit results
- Classifying model responses
- Generating structured analysis

### Example: EvaluationAgent with o3-mini

```python
from cybersec_agents import EvaluationAgent

# Initialize the evaluation agent with o3-mini for reasoning tasks
eval_agent = EvaluationAgent(
    output_dir="./evaluations",
    model_name="o3-mini",  # Primary model
    reasoning_model="o3-mini"  # Explicitly specify for reasoning tasks
)

# Load exploit results
import json
with open("./results/exploit_results.json", "r") as f:
    exploit_results = json.load(f)

# Evaluate results using o3-mini's reasoning capabilities
evaluation = eval_agent.evaluate_results(
    results=exploit_results,
    target_model="claude-3",
    target_behavior="jailbreak"
)

# Generate a detailed analysis using o3-mini's reasoning
summary = eval_agent.generate_summary(
    evaluation=evaluation,
    target_model="claude-3",
    target_behavior="jailbreak"
)

# Save the evaluation
eval_agent.save_evaluation(evaluation, "claude-3", "jailbreak")
eval_agent.save_summary(summary, "claude-3", "jailbreak")
```

### Performance Benefits

Using o3-mini for reasoning tasks can provide significant benefits:

- **Speed**: Faster processing of evaluation tasks
- **Cost**: Lower cost per token compared to larger models
- **Efficiency**: Comparable quality for classification and analysis tasks

## Configuring GPT-4o as a Backup Model

For tasks that might require more advanced capabilities, configuring GPT-4o as a backup model ensures your pipeline can handle complex scenarios.

### Example: ReconAgent with GPT-4o Backup

```python
from cybersec_agents import ReconAgent

# Initialize with o3-mini as primary and GPT-4o as backup
recon_agent = ReconAgent(
    output_dir="./reports",
    model_name="o3-mini",      # Efficient for most reconnaissance tasks
    backup_model="gpt-4o",     # Powerful backup for complex analysis
    complexity_threshold=0.7   # Threshold for switching to backup model
)

# Run web search with automatic model selection
web_results = recon_agent.run_web_search(
    target_model="gpt-4",
    target_behavior="jailbreak",
    num_results=10
)

# Generate report - will use o3-mini but fall back to GPT-4o
# if the complexity exceeds the threshold
report = recon_agent.generate_report(
    target_model="gpt-4",
    target_behavior="jailbreak",
    web_results=web_results
)

# Save the report
recon_agent.save_report(report, "gpt-4", "jailbreak")
```

### When to Use GPT-4o

Consider using GPT-4o in the following scenarios:

- **Complex Prompt Generation**: When creating sophisticated prompts for testing LLM security
- **Analyzing Advanced Techniques**: When analyzing novel jailbreak methods
- **Generating Creative Content**: When developing innovative attack vectors
- **Edge Cases**: When dealing with unexpected or unusual model behaviors

## Agent-Specific Configurations

Each Gray Swan Arena agent has specific tasks that benefit from different model configurations:

### ReconAgent

```python
# Optimized ReconAgent Configuration
recon_agent = ReconAgent(
    output_dir="./reports",
    model_name="o3-mini",          # Good for initial search parsing
    backup_model="gpt-4o",         # For complex analysis
    web_search_model="o3-mini",    # Efficient for search processing
    report_generation_model="gpt-4o"  # Better for comprehensive reports
)
```

### PromptEngineerAgent

```python
# Optimized PromptEngineerAgent Configuration
prompt_agent = PromptEngineerAgent(
    output_dir="./prompts",
    model_name="gpt-4o",           # Creative prompt generation benefits from GPT-4o
    reasoning_model="o3-mini",     # Analyzing patterns in effective prompts
    diversity_threshold=0.8,       # Ensures diverse prompt generation
    complexity_threshold=0.7       # Threshold for task complexity
)
```

### ExploitDeliveryAgent

```python
# Optimized ExploitDeliveryAgent Configuration
exploit_agent = ExploitDeliveryAgent(
    output_dir="./exploits",
    model_name="o3-mini",          # Sufficient for delivery mechanics
    backup_model="gpt-4o",         # For handling complex response analysis
    analysis_model="o3-mini",      # For basic response analysis
    adaptation_model="gpt-4o"      # For adapting prompts based on responses
)
```

### EvaluationAgent

```python
# Optimized EvaluationAgent Configuration
eval_agent = EvaluationAgent(
    output_dir="./evaluations",
    model_name="o3-mini",          # Efficient for standard evaluations
    reasoning_model="o3-mini",     # Good for classification tasks
    visualization_model="gpt-4o",  # Better for visualization planning
    summary_model="gpt-4o"         # For generating insightful summaries
)
```

## End-to-End Pipeline Example

This example demonstrates a complete Gray Swan Arena pipeline using the optimal model configuration for each component:

```python
from cybersec_agents import ReconAgent, PromptEngineerAgent, ExploitDeliveryAgent, EvaluationAgent

# Define base configuration
output_base_dir = "./gray_swan_results"
target_model = "gpt-4"
target_behavior = "jailbreak"

# Initialize agents with optimized model configuration
recon_agent = ReconAgent(
    output_dir=f"{output_base_dir}/reports",
    model_name="o3-mini",
    backup_model="gpt-4o",
    web_search_model="o3-mini"
)

prompt_agent = PromptEngineerAgent(
    output_dir=f"{output_base_dir}/prompts",
    model_name="gpt-4o",
    reasoning_model="o3-mini"
)

exploit_agent = ExploitDeliveryAgent(
    output_dir=f"{output_base_dir}/exploits",
    model_name="o3-mini",
    backup_model="gpt-4o"
)

eval_agent = EvaluationAgent(
    output_dir=f"{output_base_dir}/evaluations",
    model_name="o3-mini",
    visualization_model="gpt-4o",
    summary_model="gpt-4o"
)

# Step 1: Reconnaissance
print("Running reconnaissance...")
web_results = recon_agent.run_web_search(target_model, target_behavior)
report = recon_agent.generate_report(target_model, target_behavior, web_results)
report_path = recon_agent.save_report(report, target_model, target_behavior)
print(f"Reconnaissance report saved to: {report_path}")

# Step 2: Prompt Engineering
print("Generating prompts...")
prompts = prompt_agent.generate_prompts(target_model, target_behavior, report, num_prompts=5)
prompts_path = prompt_agent.save_prompts(prompts, target_model, target_behavior)
print(f"Prompts saved to: {prompts_path}")

# Step 3: Exploit Delivery
print("Running exploits...")
results = exploit_agent.run_prompts(prompts, target_model, target_behavior)
results_path = exploit_agent.save_results(results, target_model, target_behavior)
print(f"Exploit results saved to: {results_path}")

# Step 4: Evaluation
print("Evaluating results...")
evaluation = eval_agent.evaluate_results(results, target_model, target_behavior)
eval_path = eval_agent.save_evaluation(evaluation, target_model, target_behavior)
print(f"Evaluation saved to: {eval_path}")

summary = eval_agent.generate_summary(evaluation, target_model, target_behavior)
summary_path = eval_agent.save_summary(summary, target_model, target_behavior)
print(f"Summary saved to: {summary_path}")

viz_paths = eval_agent.create_visualizations(evaluation, target_model, target_behavior)
print(f"Visualizations saved to: {viz_paths}")

print("Gray Swan Arena pipeline completed successfully!")
```

## Performance Comparison

| Task | o3-mini | GPT-4o | Notes |
|------|---------|--------|-------|
| Web search parsing | ✅ Excellent | ✅ Excellent | o3-mini is more cost-effective |
| Prompt generation | ⚠️ Good | ✅ Excellent | GPT-4o produces more diverse prompts |
| Response analysis | ✅ Excellent | ✅ Excellent | o3-mini is sufficient for most cases |
| Report generation | ⚠️ Good | ✅ Excellent | GPT-4o provides more insightful analysis |
| Visualization planning | ⚠️ Good | ✅ Excellent | GPT-4o better understands data relationships |

## Conclusion

By strategically configuring model usage for different tasks, you can optimize the Gray Swan Arena framework for:

- **Cost efficiency**: Using o3-mini for suitable tasks reduces costs
- **Performance**: Leveraging GPT-4o where creative or complex reasoning is required
- **Adaptability**: Using backup models ensures the pipeline can handle various scenarios

The recommended approach is to use o3-mini as the default model for reasoning tasks while configuring GPT-4o as a backup or specialized model for tasks requiring advanced capabilities.
