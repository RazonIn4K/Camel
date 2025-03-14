# Advanced Analytics and Visualization

This document explains the Advanced Analytics and Visualization capabilities in the Gray Swan Arena framework.

## Overview

The Advanced Analytics and Visualization module provides sophisticated data analysis and visualization capabilities for analyzing results from AI red-teaming exercises. It extends the basic visualization utilities with more advanced techniques such as:

1. **Feature Extraction and Analysis**: Extract and analyze features from prompts and responses
2. **Correlation Analysis**: Identify relationships between features and success rates
3. **Distribution Analysis**: Visualize the distribution of features across successful and failed attempts
4. **Advanced Clustering**: Group similar prompts and identify patterns using dimensionality reduction and clustering algorithms
5. **Model Comparison**: Compare different models using radar charts and detailed metrics
6. **Predictive Modeling**: Build machine learning models to predict success factors and identify important features
7. **Comprehensive Reporting**: Generate comprehensive analytics reports with multiple visualizations

## Components

The system consists of the following components:

### Advanced Analytics Utilities

The `advanced_analytics_utils.py` module provides functions for advanced analytics and visualization:

```python
from cybersec_agents.grayswan.utils.advanced_analytics_utils import (
    extract_features_from_results,
    create_correlation_matrix,
    create_feature_distribution_plots,
    create_pairplot,
    create_advanced_clustering,
    create_advanced_model_comparison,
    create_advanced_success_prediction_model,
    create_advanced_analytics_report
)
```

## Features

### 1. Feature Extraction and Analysis

The system can extract various features from prompts and responses:

```python
def extract_features_from_results(
    results: List[Dict[str, Any]],
    include_text_features: bool = True,
    include_nlp_features: bool = True
) -> pd.DataFrame:
    """Extract features from results for analysis."""
```

Features extracted include:
- **Basic features**: Model, prompt type, attack vector, success, response time
- **Text-based features**: Prompt length, word count, question marks, exclamation marks, code markers, uppercase ratio, etc.
- **NLP-based features**: Sentiment scores (negative, neutral, positive, compound)
- **Response features**: Response length, response word count

### 2. Correlation Analysis

The system can create correlation matrices to identify relationships between features:

```python
def create_correlation_matrix(
    df: pd.DataFrame,
    output_dir: str,
    title: str = "Feature Correlation Matrix",
    figsize: Tuple[int, int] = (14, 12)
) -> str:
    """Create a correlation matrix visualization."""
```

This function:
1. Selects numerical columns from the DataFrame
2. Calculates the correlation matrix
3. Creates a heatmap visualization
4. Saves the visualization to a file

### 3. Distribution Analysis

The system can create distribution plots for features:

```python
def create_feature_distribution_plots(
    df: pd.DataFrame,
    output_dir: str,
    features: Optional[List[str]] = None,
    hue: str = "success",
    figsize: Tuple[int, int] = (16, 12)
) -> List[str]:
    """Create distribution plots for features."""
```

This function:
1. Creates histograms or count plots for each feature
2. Colors the plots by success/failure
3. Organizes plots in batches
4. Saves the visualizations to files

The system can also create pairplots to visualize relationships between pairs of features:

```python
def create_pairplot(
    df: pd.DataFrame,
    output_dir: str,
    features: Optional[List[str]] = None,
    hue: str = "success",
    title: str = "Feature Pairplot",
    max_features: int = 5
) -> str:
    """Create a pairplot visualization."""
```

### 4. Advanced Clustering

The system can perform advanced clustering to identify patterns in the data:

```python
def create_advanced_clustering(
    results: List[Dict[str, Any]],
    output_dir: str,
    n_clusters: int = 4,
    algorithm: str = "kmeans",
    perplexity: int = 30,
    random_state: int = 42
) -> Dict[str, Any]:
    """Create advanced clustering visualization with multiple algorithms."""
```

This function:
1. Extracts features from results
2. Applies dimensionality reduction (t-SNE or PCA)
3. Applies clustering (KMeans or DBSCAN)
4. Creates a visualization of the clusters
5. Analyzes the clusters to identify patterns
6. Saves the visualization and analysis to files

The system supports two clustering algorithms:
- **KMeans**: Partitions the data into k clusters
- **DBSCAN**: Density-based clustering that can identify clusters of arbitrary shape

### 5. Model Comparison

The system can create advanced model comparisons:

```python
def create_advanced_model_comparison(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Advanced Model Comparison"
) -> str:
    """Create an advanced model comparison visualization."""
```

This function:
1. Calculates metrics for each model (success rate, response time, etc.)
2. Creates a radar chart comparing models across multiple dimensions
3. Creates a detailed comparison table
4. Saves the visualization and table to files

### 6. Predictive Modeling

The system can build machine learning models to predict success factors:

```python
def create_advanced_success_prediction_model(
    results: List[Dict[str, Any]],
    output_dir: str,
    model_type: str = "random_forest",
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """Create an advanced model to predict success based on prompt features."""
```

This function:
1. Extracts features from results
2. Prepares features and target for machine learning
3. Trains a machine learning model (Random Forest or Gradient Boosting)
4. Evaluates the model using various metrics
5. Creates visualizations of feature importance, ROC curve, and confusion matrix
6. Saves the visualizations and metrics to files

The system supports two model types:
- **Random Forest**: An ensemble of decision trees
- **Gradient Boosting**: A boosting algorithm that builds trees sequentially

### 7. Comprehensive Reporting

The system can generate comprehensive analytics reports:

```python
def create_advanced_analytics_report(
    results: List[Dict[str, Any]],
    output_dir: str,
    include_clustering: bool = True,
    include_prediction: bool = True,
    include_model_comparison: bool = True
) -> Dict[str, str]:
    """Create a comprehensive analytics report with advanced visualizations."""
```

This function:
1. Creates a correlation matrix
2. Creates feature distribution plots
3. Creates a pairplot
4. Creates advanced clustering visualizations (if requested)
5. Creates model comparison visualizations (if requested)
6. Creates predictive models (if requested)
7. Returns a dictionary mapping chart names to file paths

## Dependencies

The Advanced Analytics and Visualization module has the following dependencies:

- **numpy**: For numerical operations
- **pandas**: For data manipulation
- **matplotlib**: For creating visualizations
- **seaborn**: For creating statistical visualizations
- **scikit-learn**: For machine learning and dimensionality reduction
- **nltk**: For natural language processing (optional)

## Usage Examples

### Basic Feature Analysis

```python
from cybersec_agents.grayswan.utils.advanced_analytics_utils import (
    extract_features_from_results,
    create_correlation_matrix,
    create_feature_distribution_plots,
    create_pairplot
)

# Extract features
features_df = extract_features_from_results(results)

# Create correlation matrix
correlation_file = create_correlation_matrix(features_df, output_dir)

# Create feature distribution plots
distribution_files = create_feature_distribution_plots(features_df, output_dir)

# Create pairplot
pairplot_file = create_pairplot(features_df, output_dir)
```

### Clustering Analysis

```python
from cybersec_agents.grayswan.utils.advanced_analytics_utils import create_advanced_clustering

# Create KMeans clustering
kmeans_results = create_advanced_clustering(
    results, output_dir, n_clusters=4, algorithm="kmeans"
)

# Create DBSCAN clustering
dbscan_results = create_advanced_clustering(
    results, output_dir, algorithm="dbscan"
)
```

### Model Comparison

```python
from cybersec_agents.grayswan.utils.advanced_analytics_utils import create_advanced_model_comparison

# Create advanced model comparison
model_comparison_file = create_advanced_model_comparison(results, output_dir)
```

### Predictive Modeling

```python
from cybersec_agents.grayswan.utils.advanced_analytics_utils import create_advanced_success_prediction_model

# Create Random Forest prediction model
rf_results = create_advanced_success_prediction_model(
    results, output_dir, model_type="random_forest"
)

# Create Gradient Boosting prediction model
gb_results = create_advanced_success_prediction_model(
    results, output_dir, model_type="gradient_boosting"
)
```

### Comprehensive Reporting

```python
from cybersec_agents.grayswan.utils.advanced_analytics_utils import create_advanced_analytics_report

# Create comprehensive analytics report
report_files = create_advanced_analytics_report(
    results, output_dir,
    include_clustering=True,
    include_prediction=True,
    include_model_comparison=True
)
```

## Integration with Gray Swan Arena

The Advanced Analytics and Visualization module is designed to work seamlessly with the Gray Swan Arena framework:

### Evaluation Agent

The Evaluation Agent can use the Advanced Analytics and Visualization module to analyze results:

```python
from cybersec_agents.grayswan.agents.evaluation_agent import EvaluationAgent
from cybersec_agents.grayswan.utils.advanced_analytics_utils import create_advanced_analytics_report

# Create evaluation agent
eval_agent = EvaluationAgent()

# Load results
results = eval_agent.load_exploit_results("path/to/exploit_results.json")

# Create advanced analytics report
report_files = create_advanced_analytics_report(results, eval_agent.output_dir)

# Include advanced analytics in evaluation report
report = eval_agent.generate_report(
    results=results,
    statistics=eval_agent.calculate_statistics(results),
    visualization_paths=report_files
)
```

## Best Practices

1. **Start with Basic Analysis**: Begin with basic feature analysis to understand the data
2. **Use Correlation Analysis**: Identify relationships between features and success rates
3. **Visualize Distributions**: Visualize the distribution of features to identify patterns
4. **Apply Clustering**: Use clustering to identify groups of similar prompts
5. **Compare Models**: Compare different models to identify strengths and weaknesses
6. **Build Predictive Models**: Build machine learning models to predict success factors
7. **Generate Comprehensive Reports**: Generate comprehensive reports for stakeholders

## Limitations

1. **Data Requirements**: Some analyses require a minimum number of samples
2. **Computational Resources**: Advanced analyses can be computationally intensive
3. **Interpretability**: Some machine learning models can be difficult to interpret
4. **Dependency Requirements**: Some analyses require additional dependencies
5. **Visualization Quality**: The quality of visualizations depends on the data

## Future Enhancements

1. **Interactive Visualizations**: Add interactive visualizations using Plotly or Bokeh
2. **More Machine Learning Models**: Add support for more machine learning models
3. **Natural Language Processing**: Enhance NLP capabilities for text analysis
4. **Time Series Analysis**: Add support for time series analysis
5. **Anomaly Detection**: Add support for anomaly detection
6. **Explainable AI**: Add support for explaining model predictions
7. **Automated Insights**: Generate automated insights from the data
8. **Real-time Analytics**: Add support for real-time analytics
9. **Integration with Other Tools**: Integrate with other data analysis tools
10. **Export to Other Formats**: Add support for exporting to other formats (PDF, PowerPoint, etc.)