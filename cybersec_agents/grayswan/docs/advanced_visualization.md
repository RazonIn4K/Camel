# Advanced Visualization and Analytics

This document provides an overview of the advanced visualization and analytics capabilities in the Gray Swan Arena framework.

## Overview

The advanced visualization system provides sophisticated data analysis and visualization tools for understanding attack patterns, prompt similarities, and success factors in AI red-teaming exercises. It extends the basic visualization capabilities with:

1. **Attack Pattern Clustering**: Identifies patterns in attack strategies using dimensionality reduction and clustering
2. **Prompt Similarity Network**: Visualizes relationships between prompts based on content similarity
3. **Success Prediction Model**: Builds a machine learning model to predict success factors
4. **Interactive Dashboard**: Creates a comprehensive HTML dashboard with filtering and interactive elements
5. **Advanced Metrics**: Provides detailed statistical analysis of results

## Key Components

### Attack Pattern Visualization

The `create_attack_pattern_visualization` function creates a visualization showing attack patterns and effectiveness clusters:

```python
from cybersec_agents.grayswan.utils.advanced_visualization_utils import create_attack_pattern_visualization

# Create attack pattern visualization
visualization_path = create_attack_pattern_visualization(
    results=exploit_results,  # List of result dictionaries
    output_dir="./output/visualizations",
    title="Attack Pattern Clustering",
    n_clusters=4,  # Number of clusters to identify
    random_state=42  # For reproducibility
)
```

This visualization:
- Uses dimensionality reduction (t-SNE or PCA) to project data into 2D space
- Applies K-means clustering to identify patterns
- Visualizes clusters with different colors
- Sizes points based on success
- Includes cluster centers and annotations

### Prompt Similarity Network

The `create_prompt_similarity_network` function creates a network visualization showing similarities between prompts:

```python
from cybersec_agents.grayswan.utils.advanced_visualization_utils import create_prompt_similarity_network

# Create prompt similarity network
network_path = create_prompt_similarity_network(
    results=exploit_results,  # List of result dictionaries
    output_dir="./output/visualizations",
    title="Prompt Similarity Network",
    threshold=0.5,  # Similarity threshold for connecting nodes
    random_state=42  # For reproducibility
)
```

This visualization:
- Uses TF-IDF vectorization to convert prompts to numerical vectors
- Calculates cosine similarity between prompts
- Creates a network graph with prompts as nodes
- Connects similar prompts with edges
- Colors nodes based on success
- Sizes edges based on similarity strength

### Success Prediction Model

The `create_success_prediction_model` function builds a machine learning model to predict success factors:

```python
from cybersec_agents.grayswan.utils.advanced_visualization_utils import create_success_prediction_model

# Create success prediction model
model_path, metrics = create_success_prediction_model(
    results=exploit_results,  # List of result dictionaries
    output_dir="./output/visualizations",
    title="Success Prediction Model",
    test_size=0.3,  # Proportion of data to use for testing
    random_state=42  # For reproducibility
)

# Print model metrics
print(f"Model accuracy: {metrics['accuracy']:.2f}")
print(f"Feature importance: {metrics['feature_importance']}")
```

This function:
- Extracts features from prompts (length, word count, etc.)
- Trains a Random Forest classifier to predict success
- Evaluates model performance (accuracy, precision, recall, F1)
- Visualizes feature importance
- Returns model metrics for further analysis

### Interactive Dashboard

The `create_interactive_dashboard` function creates a comprehensive HTML dashboard:

```python
from cybersec_agents.grayswan.utils.advanced_visualization_utils import create_interactive_dashboard

# Create interactive dashboard
dashboard_path = create_interactive_dashboard(
    results=exploit_results,  # List of result dictionaries
    output_dir="./output/visualizations",
    title="Gray Swan Arena Interactive Dashboard"
)

# Print instructions for viewing the dashboard
print(f"Dashboard created at: {dashboard_path}")
print(f"Open in a web browser to view")
```

This dashboard includes:
- Summary statistics and metrics
- Multiple visualizations in a tabbed interface
- Interactive filtering capabilities
- Detailed results table
- Responsive design for different screen sizes

### Comprehensive Evaluation Report

The `create_advanced_evaluation_report` function creates a comprehensive evaluation report with all visualizations:

```python
from cybersec_agents.grayswan.utils.advanced_visualization_utils import create_advanced_evaluation_report

# Create comprehensive evaluation report
report_files = create_advanced_evaluation_report(
    results=exploit_results,  # List of result dictionaries
    output_dir="./output/visualizations",
    include_interactive=True  # Whether to include interactive dashboard
)

# Print paths to all generated files
for name, path in report_files.items():
    print(f"{name}: {path}")
```

This function:
- Creates all basic visualizations (success rate, response time, etc.)
- Adds advanced visualizations (attack patterns, similarity network, etc.)
- Optionally includes an interactive dashboard
- Returns paths to all generated files

## Integration with Evaluation Agent

The advanced visualization system can be integrated with the Evaluation Agent:

```python
from cybersec_agents.grayswan import EvaluationAgent
from cybersec_agents.grayswan.utils.advanced_visualization_utils import create_advanced_evaluation_report

# Create evaluation agent
eval_agent = EvaluationAgent()

# Load results
results = eval_agent.load_exploit_results("path/to/exploit_results.json")

# Create advanced visualizations
visualization_paths = create_advanced_evaluation_report(
    results=results,
    output_dir="./output/visualizations"
)

# Generate report with advanced visualizations
report = eval_agent.generate_report(
    results=results,
    visualization_paths=visualization_paths
)

# Save report
report_path = eval_agent.save_report(report)
```

## Example Usage

See the `advanced_visualization_example.py` script for a complete example of how to use the advanced visualization capabilities:

```bash
# Run with default settings
python -m cybersec_agents.grayswan.examples.advanced_visualization_example

# Run with custom settings
python -m cybersec_agents.grayswan.examples.advanced_visualization_example \
    --output-dir "./output/visualizations" \
    --num-samples 100 \
    --save-data \
    --interactive
```

## Requirements

The advanced visualization system requires the following packages:

- matplotlib
- seaborn
- pandas
- numpy
- scikit-learn
- networkx (optional, for similarity network)

Install these packages with:

```bash
pip install matplotlib seaborn pandas numpy scikit-learn networkx
```

## Best Practices

1. **Data Preparation**: Ensure your results include all necessary fields (model_name, prompt_type, success, etc.)
2. **Sample Size**: For clustering and machine learning, aim for at least 30-50 data points
3. **Output Directory**: Use a dedicated directory for visualizations to keep them organized
4. **Interactive Dashboard**: For sharing results, the interactive dashboard provides the most comprehensive view
5. **Customization**: Adjust parameters like n_clusters and threshold based on your specific data

## Limitations

1. **Data Requirements**: Some visualizations require specific fields in the results
2. **Sample Size**: Machine learning models require sufficient data for training
3. **Performance**: Large datasets may require significant processing time
4. **Dependencies**: Some visualizations require optional dependencies like networkx

## Troubleshooting

If you encounter issues with the advanced visualization system:

1. **Check Dependencies**: Ensure all required packages are installed
2. **Check Data Format**: Verify that your results include all necessary fields
3. **Check Sample Size**: Some visualizations require a minimum number of data points
4. **Check Output Directory**: Ensure the output directory exists and is writable
5. **Check Logs**: Look for error messages in the logs

## Future Enhancements

Planned enhancements for the advanced visualization system include:

1. **Real-time Visualization**: Update visualizations as new results come in
2. **Interactive Clustering**: Allow users to adjust clustering parameters interactively
3. **Natural Language Insights**: Generate natural language insights from the data
4. **Comparative Analysis**: Compare results across different runs or configurations
5. **Export Capabilities**: Export visualizations in different formats (PDF, SVG, etc.)