# Gray Swan Arena

## AI Red-Teaming Framework

Gray Swan Arena is a structured framework for conducting red-teaming exercises against AI language models. The framework uses a multi-agent approach to systematically test the robustness and safety of AI systems against adversarial prompts and techniques.

## Overview

Gray Swan Arena consists of four specialized agents that work together in a pipeline:

1. **Reconnaissance Agent**: Gathers information about the target model, including architecture, capabilities, and potential vulnerabilities.
2. **Prompt Engineer Agent**: Creates sophisticated prompts designed to elicit undesired behavior from the target model.
3. **Exploit Delivery Agent**: Executes the prompts against the target model and evaluates the effectiveness of each attempt. Now with browser automation capabilities.
4. **Evaluation Agent**: Analyzes results and generates comprehensive reports with statistics, visualizations, and recommendations.

## Installation

Gray Swan Arena is included in the cybersec_agents package. Install it along with its dependencies:

```bash
pip install -e .
```

For browser automation features, additional setup is required:

```bash
# For Playwright
pip install playwright
playwright install

# For Selenium
pip install selenium webdriver-manager
```

## Usage

### Command Line Interface

Gray Swan Arena can be run using the cybersec-agents CLI:

```bash
# Run the full pipeline against GPT-3.5
cybersec-agents grayswan --target GPT3.5

# Run just the reconnaissance phase
cybersec-agents grayswan --mode recon

# Generate prompts from an existing recon report
cybersec-agents grayswan --mode prompts --recon-report path/to/report.json

# Include visualizations in the evaluation
cybersec-agents grayswan --visualize

# Include advanced visualizations and analytics
cybersec-agents grayswan --target GPT3.5 --advanced-visualizations

# Create interactive dashboard
cybersec-agents grayswan --target GPT3.5 --interactive-dashboard

# Include advanced analytics
cybersec-agents grayswan --target GPT3.5 --advanced-analytics

# Specify clustering algorithm
cybersec-agents grayswan --target GPT3.5 --advanced-analytics --clustering-algorithm kmeans

# Specify prediction model type
cybersec-agents grayswan --target GPT3.5 --advanced-analytics --prediction-model random_forest

# Specify browser automation method
cybersec-agents grayswan --target GPT3.5 --browser-method playwright

# Use enhanced browser automation with retry capabilities
cybersec-agents grayswan --target GPT3.5 --enhanced-browser --retry-attempts 5

# Run with visible browser for debugging
cybersec-agents grayswan --target GPT3.5 --browser-method playwright --visible

# Enable parallel processing
cybersec-agents grayswan --target GPT3.5 --parallel

# Configure parallel processing options
cybersec-agents grayswan --target GPT3.5 --parallel --max-workers 5 --batch-size 3 --batch-delay 2.0

# Use process-based parallelism for CPU-bound tasks
cybersec-agents grayswan --target GPT3.5 --parallel --use-processes

# Configure rate limiting for API calls
cybersec-agents grayswan --target GPT3.5 --parallel --requests-per-minute 60

# Use dependency injection with configuration file
cybersec-agents grayswan-di --target GPT3.5 --config-file config.yaml

# Override configuration values
cybersec-agents grayswan-di --target GPT3.5 --agent-model gpt-4 --advanced-visualizations
```

The `grayswan-di` command uses the dependency injection version of the framework, which provides more flexibility and configuration options.

### Python API

You can also use Gray Swan Arena programmatically:

#### Reconnaissance Agent

```python
from cybersec_agents.grayswan import ReconAgent

# Create a reconnaissance agent
recon_agent = ReconAgent()

# Gather information
model_info = recon_agent.run_web_search("GPT-4 capabilities")
behavior_info = recon_agent.run_web_search("AI safeguards bypassing")
techniques_info = recon_agent.run_web_search("jailbreaking techniques")

# Basic Discord search
discord_info = recon_agent.run_discord_search("jailbreaking techniques")

# Enhanced Discord search with advanced features
from cybersec_agents.grayswan.utils.enhanced_discord_utils import EnhancedDiscordScraper

# Create enhanced Discord scraper
discord_scraper = EnhancedDiscordScraper()

# Search with advanced filters
discord_results = discord_scraper.search_with_filters(
    query="jailbreaking techniques",
    min_date="2023-01-01",
    max_date="2023-12-31",
    content_filter=r"(prompt|injection|bypass)",
    has_attachments=True,
    include_sentiment=True
)

# Extract trending topics
trending_topics = discord_scraper.get_trending_topics(discord_results, top_n=10)

# Analyze user activity
user_activity = discord_scraper.get_user_activity(discord_results)

# Analyze time distribution
time_dist = discord_scraper.get_time_distribution(discord_results, interval="month")

# Format results with statistics
formatted_results = discord_scraper.format_advanced_results(
    discord_results,
    include_metadata=True,
    include_stats=True
)

# Generate and save report
report = recon_agent.generate_report(model_info, behavior_info, techniques_info, discord_info)
report_path = recon_agent.save_report(report)
```

#### Prompt Engineer Agent

```python
from cybersec_agents.grayswan import PromptEngineerAgent

# Create prompt engineer agent
prompt_agent = PromptEngineerAgent()

# Load reconnaissance report
recon_report = prompt_agent.load_recon_report("path/to/recon_report.json")

# Generate prompts
prompts = prompt_agent.generate_prompts(recon_report, num_prompts=10)

# Evaluate prompt diversity
diversity_metrics = prompt_agent.evaluate_prompt_diversity(prompts)

# Save prompts
prompt_path = prompt_agent.save_prompts(prompts)
```

#### Exploit Delivery Agent

```python
from cybersec_agents.grayswan import ExploitDeliveryAgent
from camel.types import ModelType, ModelPlatformType

# Method 1: API-based testing
# Create exploit delivery agent with specific model
agent = ExploitDeliveryAgent(
    target_model_type=ModelType.GPT_3_5_TURBO,
    target_model_platform=ModelPlatformType.OPENAI
)

# Load prompts
prompts = agent.load_prompts("path/to/prompt_list.json")

# Execute prompts via API
results = agent.execute_prompt_batch(prompts, max_concurrent=3)

# Method 2: Basic browser-based testing (Playwright or Selenium)
agent = ExploitDeliveryAgent(browser_method="playwright", headless=False)

# Execute prompts via browser automation
results = agent.run_prompts(
    prompts,
    target_model="Brass Fox Legendary",
    target_behavior="Leak information"
)

# Method 3: Enhanced browser-based testing with self-healing capabilities
from cybersec_agents.grayswan.utils.enhanced_browser_utils import EnhancedBrowserAutomationFactory

# Create an agent with enhanced browser automation
agent = ExploitDeliveryAgent(browser_method="playwright", headless=False)

# Execute prompts with enhanced browser automation
results = agent.run_prompts(
    prompts,
    target_model="Claude 3 Opus",
    target_behavior="Generate harmful content",
    method="web"  # Use web-based execution with enhanced browser automation
)

# Get browser automation metrics
browser_metrics = agent.get_browser_metrics()
print(f"Browser automation metrics: {browser_metrics}")

# Save and analyze results
saved_path = agent.save_results(results)
analysis = agent.analyze_results(results)
```

#### Evaluation Agent

```python
from cybersec_agents.grayswan import EvaluationAgent

# Create evaluation agent
eval_agent = EvaluationAgent()

# Load results
results = eval_agent.load_exploit_results("path/to/exploit_results.json")

# Load reconnaissance report (optional)
recon_report = eval_agent.load_recon_report("path/to/recon_report.json")

# Calculate statistics
statistics = eval_agent.calculate_statistics(results)

# Create basic visualizations
vis_dir = "data/evaluation_reports/visualizations"
visualization_paths = eval_agent.create_visualizations(statistics, vis_dir)

# Create advanced visualizations and analytics
advanced_vis_dir = "data/evaluation_reports/visualizations/advanced"
advanced_vis_paths = eval_agent.create_advanced_visualizations(
    results=results,
    target_model="GPT-4",
    target_behavior="Generate harmful content",
    include_interactive=True  # Create interactive dashboard
)

# Generate report
report = eval_agent.generate_report(
    results=results,
    statistics=statistics,
    recon_report=recon_report,
    visualization_paths={**visualization_paths, **advanced_vis_paths}
)

# Save report
report_path = eval_agent.save_report(report)

# Generate markdown report
markdown_path = eval_agent.generate_markdown_report(report)

# Generate HTML report with visualizations
html_path = eval_agent.generate_html_report(report)
```

### Utility Modules

Gray Swan Arena now provides dedicated utility modules for browser automation and visualization:

#### Browser Automation

```python
from cybersec_agents.grayswan.utils import (
    BrowserMethod,
    BrowserAutomationFactory,
    is_browser_automation_available
)

# Check available browser automation methods
available = is_browser_automation_available()
print(f"Playwright available: {available['playwright']}")
print(f"Selenium available: {available['selenium']}")

# Create a browser driver
driver = BrowserAutomationFactory.create_driver(
    method="playwright",  # or "selenium"
    headless=True
)

# Use the driver
driver.initialize()
driver.navigate("https://example.com")
response = driver.execute_prompt("Test prompt", "Model Name", "Behavior")
driver.close()
```

#### Enhanced Browser Automation

Gray Swan Arena now includes enhanced browser automation capabilities with self-healing and adaptive selectors:

```python
from cybersec_agents.grayswan.utils.enhanced_browser_utils import (
    EnhancedBrowserAutomationFactory,
    EnhancedPlaywrightDriver
)

# Create an enhanced browser driver with retry capabilities
driver = EnhancedBrowserAutomationFactory.create_driver(
    method="playwright",
    headless=True,
    retry_attempts=3,
    retry_delay=1.0
)

# Use the enhanced driver
driver.initialize()
driver.navigate("https://chat.openai.com/")

# Execute prompt with adaptive selectors and retry logic
response = driver.execute_prompt(
    prompt="Explain the concept of browser automation.",
    model="gpt-4",
    behavior=""
)

# Get metrics about the browser automation
metrics = driver.get_metrics()
print(f"Browser automation metrics: {metrics}")

driver.close()
```

For more details, see the [Enhanced Browser Automation Documentation](docs/enhanced_browser_automation.md).

#### Visualization

```python
from cybersec_agents.grayswan.utils import (
    create_success_rate_chart,
    create_response_time_chart,
    create_prompt_type_effectiveness_chart,
    create_vulnerability_heatmap,
    create_evaluation_report
)

# Create individual charts
success_chart = create_success_rate_chart(results, "output_dir")
time_chart = create_response_time_chart(results, "output_dir")
effectiveness_chart = create_prompt_type_effectiveness_chart(results, "output_dir")
heatmap = create_vulnerability_heatmap(results, "output_dir")

# Create a comprehensive HTML report with all visualizations
report_files = create_evaluation_report(results, "output_dir")
print(f"HTML report generated at: {report_files['html_report']}")
```

#### Advanced Visualization and Analytics

Gray Swan Arena now includes advanced visualization and analytics capabilities:

```python
from cybersec_agents.grayswan.utils.advanced_visualization_utils import (
    create_attack_pattern_visualization,
    create_prompt_similarity_network,
    create_success_prediction_model,
    create_interactive_dashboard,
    create_advanced_evaluation_report
)

# Advanced Analytics
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

# Extract features from results
features_df = extract_features_from_results(exploit_results)

# Create correlation matrix
correlation_file = create_correlation_matrix(features_df, "output_dir")

# Create feature distribution plots
distribution_files = create_feature_distribution_plots(features_df, "output_dir")

# Create pairplot
pairplot_file = create_pairplot(features_df, "output_dir")

# Create advanced clustering
clustering_results = create_advanced_clustering(
    exploit_results, "output_dir", n_clusters=4, algorithm="kmeans"
)
print(f"Silhouette score: {clustering_results.get('silhouette_score', 'N/A')}")

# Create advanced model comparison
model_comparison_file = create_advanced_model_comparison(exploit_results, "output_dir")

# Create advanced success prediction models
rf_results = create_advanced_success_prediction_model(
    exploit_results, "output_dir", model_type="random_forest"
)
gb_results = create_advanced_success_prediction_model(
    exploit_results, "output_dir", model_type="gradient_boosting"
)

# Create comprehensive analytics report
report_files = create_advanced_analytics_report(
    exploit_results, "output_dir",
    include_clustering=True,
    include_prediction=True,
    include_model_comparison=True
)

# Create attack pattern visualization using dimensionality reduction and clustering
attack_pattern_file = create_attack_pattern_visualization(
    results=exploit_results,
    output_dir="output_dir",
    title="Attack Pattern Clustering",
    n_clusters=4
)

# Create prompt similarity network
similarity_network_file = create_prompt_similarity_network(
    results=exploit_results,
    output_dir="output_dir",
    threshold=0.5  # Similarity threshold for connecting nodes
)

# Create success prediction model with feature importance
prediction_file, metrics = create_success_prediction_model(
    results=exploit_results,
    output_dir="output_dir"
)
print(f"Model accuracy: {metrics['accuracy']:.2f}")
print(f"Feature importance: {metrics['feature_importance']}")

# Create interactive dashboard with filtering capabilities
dashboard_file = create_interactive_dashboard(
    results=exploit_results,
    output_dir="output_dir",
    title="Gray Swan Arena Dashboard"
)

# Create comprehensive evaluation report with all advanced visualizations
report_files = create_advanced_evaluation_report(
    results=exploit_results,
    output_dir="output_dir",
    include_interactive=True
)

# Print instructions for viewing the dashboard
print(f"Interactive dashboard created at: {dashboard_file}")
print(f"Open in a web browser to view")
```

For more details, see the [Advanced Visualization Documentation](docs/advanced_visualization.md).

#### Parallel Processing

Gray Swan Arena now includes a robust parallel processing system for improved performance and scalability:

```python
from cybersec_agents.grayswan.utils.parallel_processing import (
    TaskManager,
    run_parallel_tasks,
    run_parallel_sync,
    parallel_map,
    run_full_pipeline_parallel_sync
)

# Method 1: Using the TaskManager class
async def process_items_with_task_manager():
    # Define a task function
    async def process_item(item):
        # Simulate processing time
        await asyncio.sleep(0.1)
        return {"item": item, "result": item * 2}
    
    # Create a task manager
    task_manager = TaskManager(
        max_workers=5,
        requests_per_minute=60,
        burst_size=5,
        max_retries=3,
        retry_delay=1.0,
        retry_backoff_factor=2.0,
        jitter=True
    )
    
    # Process items in parallel
    items = list(range(10))
    results_with_metrics = await task_manager.map(
        process_item, items, batch_size=3, batch_delay=1.0
    )
    
    # Extract results
    results = [r for r, _ in results_with_metrics if r is not None]
    
    # Get metrics
    metrics = task_manager.get_metrics()
    print(f"Task manager metrics: {metrics}")

# Method 2: Using convenience functions
def process_items_with_convenience_functions():
    # Define a task function
    def square(x):
        # Simulate processing time
        time.sleep(0.1)
        return x * x
    
    # Process items in parallel
    items = list(range(10))
    results, metrics = run_parallel_sync(
        square, items, max_workers=5, requests_per_minute=60
    )
    
    print(f"Results: {results}")
    print(f"Metrics: {metrics}")

# Method 3: Using the decorator
@parallel_map(max_workers=5, requests_per_minute=60)
def process_data(item):
    # Simulate processing time
    time.sleep(0.1)
    return {"id": item["id"], "value": item["value"] * 2}

# Process items using the decorated function
items = [{"id": i, "value": i * 10} for i in range(5)]
results, metrics = process_data(items)

# Method 4: Running the full pipeline in parallel
results = run_full_pipeline_parallel_sync(
    target_model="GPT-3.5-Turbo",
    target_behavior="Generate harmful content",
    output_dir="./output",
    model_name="gpt-4",
    max_workers=5,
    requests_per_minute=60,
    batch_size=3,
    batch_delay=2.0,
    max_retries=3,
    include_advanced_visualizations=True,
    include_interactive_dashboard=True
)
```

For more details, see the [Parallel Processing Documentation](docs/parallel_processing.md).

#### Dependency Injection

Gray Swan Arena now supports dependency injection for better testability and flexibility:

```python
from cybersec_agents.grayswan.container import GraySwanContainerFactory
from cybersec_agents.grayswan.main_di import GraySwanPipeline

# Create container with default configuration
container = GraySwanContainerFactory.create_container()

# Create container with custom configuration
config_dict = {
    'output_dir': './custom_output',
    'agents': {
        'recon': {
            'model_name': 'gpt-3.5-turbo',
        },
    },
    'visualization': {
        'advanced': True,
        'interactive': True,
    },
}
container = GraySwanContainerFactory.create_container(config_dict)

# Create container from configuration file
container = GraySwanContainerFactory.create_container_from_file('config.yaml')

# Create pipeline
pipeline = GraySwanPipeline(container)

# Run pipeline
results = pipeline.run_full_pipeline(
    target_model="GPT-4",
    target_behavior="generate harmful content",
    include_advanced_visualizations=True,
    include_interactive_dashboard=True
)
```

For more details, see the [Dependency Injection Documentation](docs/dependency_injection.md).

#### Model Integration and Fallback System

Gray Swan Arena now includes a robust model management system with fallback capabilities:

```python
from cybersec_agents.grayswan.container import GraySwanContainerFactory
from cybersec_agents.grayswan.utils.model_manager_di import ModelManager

# Method 1: Using the container
container = GraySwanContainerFactory.create_container()

# Get model manager from container
model_manager = container.model_manager()

# Generate with fallback capabilities
response = model_manager.generate(
    prompt="Explain the concept of model fallback.",
    complexity=0.6  # Optional complexity score
)

# Get agent-specific model manager
recon_model_manager = container.recon_model_manager()
prompt_model_manager = container.prompt_engineer_model_manager()

# Method 2: Direct instantiation
model_manager = ModelManager(
    primary_model="gpt-4",
    backup_model="gpt-3.5-turbo",
    complexity_threshold=0.7
)

# Estimate complexity of a prompt
complexity = model_manager.estimate_complexity(prompt)
print(f"Prompt complexity: {complexity:.2f}")

# Generate with complexity-based model selection
response = model_manager.generate(
    prompt=prompt,
    complexity=complexity
)

# Get metrics
metrics = model_manager.get_metrics()
print(f"Primary calls: {metrics['primary_calls']}")
print(f"Backup calls: {metrics['backup_calls']}")
print(f"Failures: {metrics['failures']}")
```

For more details, see the [Model Fallback System Documentation](docs/model_fallback_system.md).

## Configuration

Gray Swan Arena requires API keys for the language models being used. Set them in your environment variables or in a `.env` file:

```
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_huggingface_key

# Discord Integration
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_CHANNEL_IDS=123456789012345678,987654321098765432

# Model Integration and Fallback Settings
GRAYSWAN_PRIMARY_MODEL=gpt-4  # Primary model to use
GRAYSWAN_BACKUP_MODEL=gpt-3.5-turbo  # Backup model to use if primary fails
GRAYSWAN_COMPLEXITY_THRESHOLD=0.7  # Threshold for using backup model (0.0-1.0)
GRAYSWAN_MAX_RETRIES=5  # Maximum retries for API calls
GRAYSWAN_INITIAL_DELAY=1.0  # Initial delay in seconds for exponential backoff
GRAYSWAN_BACKOFF_FACTOR=2.0  # Factor to increase delay with each retry
GRAYSWAN_JITTER=true  # Whether to add randomness to delay

# Browser Automation Settings
GRAYSWAN_BROWSER_METHOD=playwright  # Options: playwright, selenium
GRAYSWAN_BROWSER_HEADLESS=true  # Set to false to see the browser UI
GRAYSWAN_BROWSER_TIMEOUT=60000  # Timeout in milliseconds
GRAYSWAN_BROWSER_ENHANCED=true  # Use enhanced browser automation with self-healing
GRAYSWAN_BROWSER_RETRY_ATTEMPTS=3  # Number of retry attempts for enhanced browser automation
GRAYSWAN_BROWSER_RETRY_DELAY=1.0  # Base delay between retries in seconds
GRAY_SWAN_URL=https://example.com/gray-swan  # URL for Gray Swan Arena web interface

# Visualization Settings
GRAYSWAN_VIZ_OUTPUT_DIR=./output/visualizations  # Directory for visualization outputs
GRAYSWAN_VIZ_DPI=300  # Resolution for saved charts
GRAYSWAN_VIZ_THEME=default  # Visualization theme
GRAYSWAN_VIZ_ADVANCED=true  # Enable advanced visualization capabilities
GRAYSWAN_VIZ_INTERACTIVE=true  # Create interactive dashboards
GRAYSWAN_VIZ_CLUSTERING_CLUSTERS=4  # Number of clusters for attack pattern visualization
GRAYSWAN_VIZ_SIMILARITY_THRESHOLD=0.5  # Threshold for prompt similarity network

# Advanced Analytics Settings
GRAYSWAN_ANALYTICS_ENABLED=true  # Enable advanced analytics capabilities
GRAYSWAN_ANALYTICS_OUTPUT_DIR=./output/analytics  # Directory for analytics outputs
GRAYSWAN_ANALYTICS_INCLUDE_NLP=true  # Include NLP-based features in analysis
GRAYSWAN_ANALYTICS_CLUSTERING_ALGORITHM=kmeans  # Clustering algorithm (kmeans or dbscan)
GRAYSWAN_ANALYTICS_PREDICTION_MODEL=random_forest  # Prediction model type (random_forest or gradient_boosting)
GRAYSWAN_ANALYTICS_TEST_SIZE=0.3  # Test size for prediction models
GRAYSWAN_ANALYTICS_TSNE_NEIGHBORS=30  # t-SNE neighbors parameter (formerly perplexity)
GRAYSWAN_ANALYTICS_INCLUDE_CLUSTERING=true  # Include clustering in analytics report
GRAYSWAN_ANALYTICS_INCLUDE_PREDICTION=true  # Include prediction models in analytics report
GRAYSWAN_ANALYTICS_INCLUDE_MODEL_COMPARISON=true  # Include model comparison in analytics report

# Parallel Processing Settings
GRAYSWAN_PARALLEL_ENABLED=true  # Enable parallel processing
GRAYSWAN_PARALLEL_MAX_WORKERS=5  # Maximum number of concurrent workers
GRAYSWAN_PARALLEL_REQUESTS_PER_MINUTE=60  # Maximum number of requests per minute
GRAYSWAN_PARALLEL_BATCH_SIZE=3  # Size of batches to process
GRAYSWAN_PARALLEL_BATCH_DELAY=2.0  # Delay between batches in seconds
GRAYSWAN_PARALLEL_MAX_RETRIES=3  # Maximum number of retries for failed tasks
GRAYSWAN_PARALLEL_RETRY_DELAY=1.0  # Initial delay between retries in seconds
GRAYSWAN_PARALLEL_RETRY_BACKOFF_FACTOR=2.0  # Factor to increase delay with each retry
GRAYSWAN_PARALLEL_JITTER=true  # Whether to add randomness to delays
GRAYSWAN_PARALLEL_USE_PROCESSES=false  # Whether to use processes instead of threads

# General Settings
MAX_RETRIES=3  # Maximum retries for API calls and operations
LOG_LEVEL=INFO  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Monitoring
AGENTOPS_API_KEY=your_agentops_key
```

## Output

Gray Swan Arena generates structured output files in the `data/` directory:

- `data/recon_reports/`: Reconnaissance data about target models
- `data/prompt_lists/`: Generated attack prompts
- `data/exploit_logs/`: Results of exploit attempts
- `data/evaluation_reports/`: Comprehensive reports and visualizations
- `data/advanced_analytics/`: Advanced analytics reports and visualizations
- `data/logs/`: Detailed execution logs
- `data/discord_searches/`: Saved Discord search results

The `data/advanced_analytics/` directory contains:

- `data/advanced_analytics/feature_distributions/`: Feature distribution plots
- `data/advanced_analytics/clustering/`: Clustering visualizations and analysis
- `data/advanced_analytics/prediction_models/`: Success prediction models and metrics
- `data/advanced_analytics/model_comparison/`: Model comparison visualizations and tables
- `data/advanced_analytics/correlation_matrices/`: Feature correlation matrices
- `data/advanced_analytics/reports/`: Comprehensive analytics reports

## Features

### Modular Architecture
Gray Swan Arena now features a more modular architecture with dedicated utility modules:

- **Browser Automation Utilities**: Dedicated module for browser automation with support for both Playwright and Selenium
- **Enhanced Browser Automation**: Self-healing browser automation with adaptive selectors and retry mechanisms
- **Visualization Utilities**: Comprehensive visualization tools for generating charts, graphs, and HTML reports
- **Consistent Interfaces**: Standardized interfaces for all utilities to ensure easy integration
- **Enhanced Type Safety**: Improved type annotations throughout the codebase
- **Dependency Injection**: Flexible dependency management with the dependency-injector library

### Multi-Method Testing
Gray Swan Arena supports multiple methods for testing AI models:

- **API-based Testing**: Direct testing against model APIs
- **Browser Automation (Playwright)**: Automate interactions with web interfaces using Playwright
- **Browser Automation (Selenium)**: Alternative automation using Selenium WebDriver
- **Enhanced Browser Automation**: Self-healing browser automation with adaptive selectors and retry mechanisms
- **Flexible Fallbacks**: Automatically selects the best available method

### Enhanced Visualization and Reporting
The framework now provides rich visualizations and comprehensive reports:

- **Success Rate Analysis**: Charts showing success rates by model and prompt type
- **Response Time Analysis**: Boxplots showing response times by model
- **Prompt Effectiveness**: Analysis of which prompt types are most effective
- **Vulnerability Heatmaps**: Identify which models are vulnerable to which attack vectors
- **HTML Reports**: Generate interactive HTML reports with embedded visualizations
- **AI-Powered Analysis**: Get AI-generated insights about patterns in successful and failed prompts

### Advanced Analytics and Visualization
Gray Swan Arena now includes sophisticated data analysis and visualization capabilities:

- **Attack Pattern Clustering**: Identifies patterns in attack strategies using dimensionality reduction and clustering
- **Prompt Similarity Network**: Visualizes relationships between prompts based on content similarity
- **Success Prediction Model**: Builds a machine learning model to predict success factors
- **Interactive Dashboard**: Creates a comprehensive HTML dashboard with filtering and interactive elements
- **Advanced Metrics**: Provides detailed statistical analysis of results
- **Feature Extraction and Analysis**: Extract and analyze features from prompts and responses
- **Correlation Analysis**: Identify relationships between features and success rates
- **Distribution Analysis**: Visualize the distribution of features across successful and failed attempts
- **Advanced Clustering**: Group similar prompts using multiple clustering algorithms (KMeans, DBSCAN)
- **Model Comparison**: Compare different models using radar charts and detailed metrics
- **Predictive Modeling**: Build machine learning models (Random Forest, Gradient Boosting) to predict success
- **Comprehensive Reporting**: Generate comprehensive analytics reports with multiple visualizations

For more details, see the [Advanced Analytics Documentation](docs/advanced_analytics.md).

### Improved Discord Integration
Enhanced Discord scraping capabilities:

- **Channel Configuration**: Search specific channels using environment variables
- **Metadata Extraction**: Extract attachments, mentions, and other metadata from Discord messages
- **Search Result Management**: Save and load Discord search results
- **Timeout Handling**: Robust timeout management for reliable operation
- **Advanced Filtering**: Filter messages by date, author, content, attachments, mentions, and sentiment
- **Sentiment Analysis**: Analyze the sentiment of messages (positive, negative, neutral)
- **Topic Extraction**: Identify trending topics in message content
- **User and Channel Activity Analysis**: Analyze user and channel activity patterns
- **Time Distribution Analysis**: Analyze message distribution over time
- **Result Caching**: Cache search results for improved performance
- **Timeframe-Based Searching**: Search for messages within specific timeframes

For more details, see the [Enhanced Discord Integration Documentation](docs/enhanced_discord_integration.md).

### Dependency Injection
Gray Swan Arena now implements the Dependency Injection pattern for better testability and flexibility:

- **Container-Based Architecture**: Centralized dependency management with the dependency-injector library
- **Configuration Management**: Flexible configuration through dictionaries, YAML, or JSON files
- **Improved Testability**: Easy mocking of dependencies for unit testing
- **Reduced Coupling**: Components depend on abstractions rather than concrete implementations
- **Runtime Configuration**: Change configuration without modifying code
- **Simplified Pipeline**: Cleaner pipeline implementation with explicit dependencies

For more details, see the [Dependency Injection Documentation](docs/dependency_injection.md).

### Model Integration and Fallback System
Gray Swan Arena now includes a robust model management system with fallback capabilities:

- **Fallback Capabilities**: Automatically fall back to backup models when primary models fail
- **Complexity-Based Model Selection**: Use different models based on prompt complexity
- **Exponential Backoff**: Handle rate limits and transient errors with exponential backoff
- **Agent-Specific Models**: Different agents can use different model configurations
- **Metrics Tracking**: Monitor model usage and failure rates
- **Improved Reliability**: More robust handling of model failures and rate limits

For more details, see the [Model Fallback System Documentation](docs/model_fallback_system.md).

### Parallel Processing
Gray Swan Arena now includes a robust parallel processing system for improved performance and scalability:

- **Task Parallelization**: Execute tasks concurrently with configurable parallelism
- **Rate Limiting**: Control request frequency to avoid rate limits from external APIs
- **Batching**: Process items in batches to manage resource usage
- **Retry Mechanism**: Automatically retry failed tasks with exponential backoff
- **Error Handling**: Robust error handling with detailed metrics
- **Metrics Collection**: Track performance and success rates
- **Flexible Execution**: Support for both thread-based and process-based parallelism
- **Pipeline Optimization**: Optimized pipeline execution with parallel phases

For more details, see the [Parallel Processing Documentation](docs/parallel_processing.md).

### Additional Improvements
- **Robust Logging**: Configurable logging system with file and console output
- **Error Handling**: Comprehensive error handling and retry mechanisms
- **Self-Healing Automation**: Adaptive selectors and recovery mechanisms for browser automation
- **Automation Metrics**: Detailed metrics for browser automation performance and reliability
- **Environment Configuration**: More flexible environment variable configuration
- **Visualization Customization**: Configurable visualization options

## Ethical Use

Gray Swan Arena is designed for authorized testing of AI models by their developers or with explicit permission. It should only be used for legitimate security research and improving AI safety. Do not use this tool against models without proper authorization.

## License

This project is part of the cybersec_agents package and is subject to its license terms. 