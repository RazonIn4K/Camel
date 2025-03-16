# Gray Swan Arena Examples

This directory contains example scripts demonstrating how to use various features of the Gray Swan Arena framework.

## Browser Automation Example

The `browser_automation_example.py` script demonstrates how to use the browser automation utilities to interact with web-based AI model interfaces.

### Features Demonstrated

- Checking available browser automation methods (Playwright and Selenium)
- Creating and initializing a browser driver
- Navigating to a target URL
- Executing a prompt and retrieving the response
- Proper error handling and cleanup

### Running the Example

```bash
# Make sure you have the required dependencies installed
pip install playwright selenium webdriver-manager

# Install Playwright browsers
playwright install

# Run the example
python browser_automation_example.py
```

## Visualization Example

The `visualization_example.py` script demonstrates how to use the visualization utilities to create charts and reports from test results.

### Features Demonstrated

- Generating sample test results
- Creating various types of charts:
  - Success rate charts by model and prompt type
  - Response time analysis with boxplots
  - Prompt type effectiveness analysis
  - Vulnerability heatmaps
- Generating a comprehensive HTML report with all visualizations

### Running the Example

```bash
# Make sure you have the required dependencies installed
pip install matplotlib seaborn pandas numpy

# Run the example
python visualization_example.py
```

The example will generate visualization files in the `data/visualization_example` directory.

## Full Pipeline Example

For a complete example of running the entire Gray Swan Arena pipeline, please refer to the tutorial in the main documentation:

- [TUTORIAL.md](../TUTORIAL.md) - Step-by-step walkthrough of a complete red-team assessment

## Additional Resources

- [README.md](../README.md) - Overview of the Gray Swan Arena framework
- [DOCUMENTATION.md](../DOCUMENTATION.md) - Comprehensive documentation of all features
- [USAGE_GUIDE.md](../USAGE_GUIDE.md) - Detailed usage instructions 