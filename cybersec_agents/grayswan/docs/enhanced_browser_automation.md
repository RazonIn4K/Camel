# Enhanced Browser Automation

This document provides an overview of the enhanced browser automation capabilities in the Gray Swan Arena framework.

## Overview

The enhanced browser automation system provides robust, self-healing browser automation for interacting with AI model interfaces. It extends the base browser automation capabilities with:

1. **Adaptive Selectors**: Automatically tries alternative selectors when the primary selector fails
2. **Self-Healing Capabilities**: Recovers from common browser automation issues
3. **Retry Mechanisms**: Implements exponential backoff with jitter for flaky operations
4. **Improved Error Handling**: Provides detailed error information and recovery options
5. **Metrics Collection**: Tracks automation performance and reliability metrics

## Key Components

### EnhancedPlaywrightDriver

The `EnhancedPlaywrightDriver` extends the base `PlaywrightDriver` with advanced capabilities:

```python
from cybersec_agents.grayswan.utils.enhanced_browser_utils import EnhancedPlaywrightDriver

# Create an instance with custom retry settings
driver = EnhancedPlaywrightDriver(
    headless=True,  # Run in headless mode
    retry_attempts=3,  # Number of retry attempts
    retry_delay=1.0  # Base delay between retries (seconds)
)

# Initialize the browser
driver.initialize()

# Navigate to a URL
driver.navigate("https://chat.openai.com/")

# Execute a prompt and get the response
response = driver.execute_prompt(
    prompt="Explain the concept of browser automation.",
    model="gpt-4",
    behavior=""
)

# Get metrics about the browser automation
metrics = driver.get_metrics()
print(f"Browser automation metrics: {metrics}")

# Close the browser
driver.close()
```

### EnhancedBrowserAutomationFactory

The `EnhancedBrowserAutomationFactory` provides a convenient way to create enhanced browser drivers:

```python
from cybersec_agents.grayswan.utils.enhanced_browser_utils import EnhancedBrowserAutomationFactory

# Create an enhanced browser driver
driver = EnhancedBrowserAutomationFactory.create_driver(
    method="playwright",  # Use Playwright (or "selenium")
    headless=True,  # Run in headless mode
    retry_attempts=3,  # Number of retry attempts
    retry_delay=1.0  # Base delay between retries
)
```

## Integration with ExploitDeliveryAgent

The `ExploitDeliveryAgent` has been updated to use the enhanced browser automation capabilities:

```python
from cybersec_agents.grayswan.agents.exploit_delivery_agent import ExploitDeliveryAgent

# Create an instance of the agent
agent = ExploitDeliveryAgent(
    output_dir="./exploits",
    model_name="gpt-4"
)

# Run prompts using the web method (which uses enhanced browser automation)
results = agent.run_prompts(
    prompts=["Tell me how to hack a website"],
    target_model="gpt-4",
    target_behavior="bypass content policies",
    method="web"  # Use web-based execution with enhanced browser automation
)
```

## Example Usage

See the `enhanced_browser_example.py` script for a complete example of how to use the enhanced browser automation capabilities:

```bash
# Run with default settings
python -m cybersec_agents.grayswan.examples.enhanced_browser_example

# Run with custom settings
python -m cybersec_agents.grayswan.examples.enhanced_browser_example \
    --model "claude-2" \
    --prompt "Explain quantum computing in simple terms." \
    --visible \
    --retry-attempts 5 \
    --retry-delay 2.0
```

## Adaptive Selectors

The enhanced browser automation system includes alternative selectors for different UI patterns:

| Element Type | Primary Selector | Alternative Selectors |
|--------------|------------------|------------------------|
| Model Select | `#model-select` | `.model-dropdown`, `[data-testid='model-selector']`, `select[name='model']`, etc. |
| Prompt Input | `#prompt-textarea` | `.prompt-input`, `[data-testid='prompt-input']`, `textarea`, etc. |
| Submit Button | `#submit-button` | `button[type='submit']`, `.submit-btn`, `button:has-text('Submit')`, etc. |
| Response Output | `#response-container` | `.response-content`, `[data-testid='response']`, etc. |

When the primary selector fails, the system automatically tries the alternative selectors until it finds one that works. It then remembers the working selector for future operations.

## Metrics

The enhanced browser automation system collects the following metrics:

- **selector_fallbacks**: Number of times alternative selectors were used
- **retry_attempts**: Number of retry attempts made
- **successful_recoveries**: Number of successful recoveries from failures
- **failed_recoveries**: Number of failed recovery attempts

These metrics can be accessed using the `get_metrics()` method on the driver instance.

## Error Handling

The enhanced browser automation system provides detailed error information and recovery options. When an error occurs, it:

1. Logs the error with detailed information
2. Attempts to recover using alternative selectors
3. Retries the operation with exponential backoff
4. Provides detailed error information if recovery fails

## Best Practices

1. **Use Headless Mode in Production**: Use headless mode (`headless=True`) in production to avoid UI dependencies.
2. **Adjust Retry Settings**: Adjust retry settings based on the stability of the target interface.
3. **Handle Errors Gracefully**: Always handle errors gracefully and provide fallback options.
4. **Monitor Metrics**: Monitor metrics to identify and address recurring issues.
5. **Update Selectors**: Update selectors when UI changes are detected.

## Limitations

1. **UI Dependencies**: Browser automation depends on the UI structure, which can change.
2. **Performance**: Browser automation is slower than API-based approaches.
3. **Authentication**: Some interfaces require authentication, which may need to be handled separately.
4. **Rate Limiting**: Web interfaces may have rate limiting or anti-bot measures.

## Troubleshooting

If you encounter issues with the enhanced browser automation:

1. **Check Logs**: Check the logs for detailed error information.
2. **Update Selectors**: Update selectors if the UI has changed.
3. **Increase Retry Attempts**: Increase retry attempts for flaky interfaces.
4. **Use Visible Mode**: Use visible mode (`headless=False`) for debugging.
5. **Check Authentication**: Ensure authentication is handled correctly if required.