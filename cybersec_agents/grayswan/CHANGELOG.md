# Gray Swan Arena Changelog

## Version 0.2.1 (Latest)

### Major Features

#### Modular Browser Automation Utilities

- **Dedicated Browser Utilities**: Extracted browser automation into a dedicated utility module:
  - `BrowserDriver` interface for consistent browser interaction
  - `PlaywrightDriver` and `SeleniumDriver` implementations
  - `BrowserAutomationFactory` for easy driver creation
- **Improved Configuration**: Enhanced environment variable support for browser settings
- **Availability Detection**: Automatic detection of available browser automation methods
- **Simplified Integration**: Easier integration with the ExploitDeliveryAgent

#### Comprehensive Visualization Utilities

- **Dedicated Visualization Module**: Created a dedicated module for data visualization:
  - Success rate charts by model and prompt type
  - Response time analysis with boxplots
  - Prompt type effectiveness analysis
  - Vulnerability heatmaps by model and attack vector
- **HTML Report Generation**: Generate comprehensive HTML reports with interactive elements
- **Customizable Output**: Configurable output directories and visualization settings
- **Enhanced Integration**: Seamless integration with the EvaluationAgent

### Code Improvements

- **Modular Architecture**: Improved code organization with dedicated utility modules
- **Consistent Interfaces**: Standardized interfaces for browser automation and visualization
- **Enhanced Documentation**: Updated documentation to reflect new utilities and features
- **Environment Variables**: Added new environment variables for browser and visualization configuration
- **Type Safety**: Improved type annotations throughout the codebase

### Documentation Updates

- **Updated DOCUMENTATION.md**: Comprehensive documentation of new utilities
- **Updated README.md**: Revised overview and usage instructions
- **Updated .env.example**: Added new environment variables
- **API Examples**: Added examples for using the new utility modules

### Bug Fixes

- Fixed version inconsistency between code and documentation
- Addressed missing dependencies in setup.py
- Improved error handling in browser automation
- Enhanced logging throughout the framework

## Version 0.2.0

### Major Features

#### Browser Automation for Exploit Delivery

- **Multi-Method Testing**: Added support for two browser automation approaches:
  - **Playwright**: Fast, modern browser automation for chromium, firefox, and webkit
  - **Selenium**: Industry-standard browser automation with wide compatibility
- **Customizable Selectors**: Configurable CSS selectors to adapt to different web interfaces
- **Headless Operation**: Optional headless mode for running tests without UI
- **Error Handling**: Robust retry mechanism and error handling for browser interactions

#### Enhanced Visualization & Reporting

- **Advanced Statistics**: More comprehensive statistics calculation including:
  - Success rates by technique, model, and target behavior
  - Confidence intervals for success rates
  - Time-based analysis of model responses
- **Data Visualization**: Generate visual representations of test results:
  - Overall success rate pie charts
  - Success rate by technique (horizontal bar charts)
  - Success rate by model (horizontal bar charts)
- **Markdown Reports**: Generate comprehensive markdown reports with embedded visualizations
- **Actionable Insights**: AI-assisted analysis of results with concrete recommendations

#### Improved Discord Integration

- **Enhanced Search**: More powerful Discord message search capabilities
- **Metadata Extraction**: Support for extracting attachments, mentions, and other message metadata
- **Results Storage**: Save Discord search results for future reference
- **Error Recovery**: Improved error handling and connection management

### Agent Improvements

#### Reconnaissance Agent

- **Web Search Capabilities**: Improved web search functionality with better error handling
- **Discord Integration**: Better integration with Discord scraping utility
- **Enhanced Report Generation**: More comprehensive reconnaissance reports

#### Prompt Engineer Agent

- **Technique Diversity**: Support for more jailbreaking techniques
- **Prompt Diversity Analysis**: Tools to analyze the diversity of generated prompts
- **Improved Report Loading**: Better handling of different report formats

#### Exploit Delivery Agent

- **Browser Automation**: Added support for browser-based testing (see major features)
- **Concurrent API Testing**: Improved batch execution with configurable concurrency
- **Enhanced Results Analysis**: More detailed analysis of test results
- **Flexible Initialization**: Support for both API-based and browser-based testing

#### Evaluation Agent

- **Visualization Generation**: Create visual representations of test results (see major features)
- **Enhanced Statistics**: More comprehensive statistical analysis
- **Markdown Report Generation**: Generate well-formatted markdown reports with embedded visualizations

### Utility Improvements

- **Logging**: More robust and configurable logging with both file and console output
- **Discord Scraping**: Improved Discord scraping with better error handling
- **Configuration Management**: Better handling of environment variables and configuration
- **Error Handling**: More comprehensive error handling throughout the framework

### Developer Experience

- **Improved Documentation**: More comprehensive documentation with usage examples
- **Type Hints**: Better type annotations for improved IDE experience
- **Consistent API**: More consistent API design across components
- **Example Scripts**: Example scripts for common use cases

### Bug Fixes

- Fixed issue with response parsing in API-based testing
- Addressed timeout issues in Discord scraping
- Improved error handling for when API keys are missing
- Fixed path handling in report generation
- Addressed concurrency issues in batch execution

## Version 0.1.0 (Initial Release)

- Initial framework implementation with basic red-teaming capabilities
- Support for reconnaissance, prompt engineering, exploit delivery, and evaluation
- API-based testing against language models
- Basic report generation
- Command-line interface support 