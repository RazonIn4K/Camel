# Gray Swan Arena - Class Diagram

This class diagram illustrates the key classes in the Gray Swan Arena framework, their relationships, attributes, and methods.

```mermaid
classDiagram
    %% Core abstract class
    class Agent {
        <<abstract>>
        +logger: Logger
        +config: Dict
        +validate_config()
        +setup_logger()
    }
    
    %% Agent implementations
    class ReconAgent {
        +model_info: Dict
        +behavior_info: Dict
        +techniques_info: Dict
        +discord_info: Dict
        +run_web_search(query: str) Dict
        +run_discord_search(query: str) Dict
        +generate_report(...) Dict
        +save_report(report: Dict) str
    }
    
    class PromptEngineerAgent {
        +techniques: List[str]
        +load_recon_report(path: str) Dict
        +generate_prompts(recon_report: Dict, num_prompts: int) List[Dict]
        +evaluate_prompt_diversity(prompts: List[Dict]) Dict
        +save_prompts(prompts: List[Dict]) str
    }
    
    class ExploitDeliveryAgent {
        +target_model_type: ModelType
        +target_model_platform: ModelPlatformType
        +selectors: Dict[str, str]
        +load_prompts(path: str) List[Dict]
        +execute_prompt_batch(prompts: List[Dict], max_concurrent: int) List[Dict]
        +run_prompts(prompts: List[Dict], target_model: str, target_behavior: str, method: str) List[Dict]
        +analyze_results(results: List[Dict]) Dict
        +save_results(results: List[Dict]) str
    }
    
    class EvaluationAgent {
        +stats: Dict
        +load_exploit_results(path: str) List[Dict]
        +load_recon_report(path: str) Dict
        +calculate_statistics(results: List[Dict]) Dict
        +create_visualizations(statistics: Dict, output_dir: str) Dict[str, str]
        +generate_report(...) Dict
        +save_report(report: Dict) str
        +generate_markdown_report(report: Dict) str
    }
    
    %% Utility classes
    class DiscordScraper {
        -bot_token: str
        -channel_ids: List[str]
        +search(query: str, channel_ids: List[str], limit: int) List[Dict]
        +format_results(results: List[Dict], include_metadata: bool) str
        +save_results(results: List[Dict], filename: str) str
    }
    
    class LoggingUtils {
        <<static>>
        +setup_logging(name: str, log_level: int, log_to_file: bool, log_filename: str) Logger
    }
    
    %% Browser automation classes
    class BrowserAutomationFactory {
        <<static>>
        +create_driver(method: str, headless: bool) BrowserDriver
    }
    
    class BrowserDriver {
        <<interface>>
        +selectors: Dict[str, str]
        +initialize()
        +navigate(url: str)
        +execute_prompt(prompt: str, model: str, behavior: str) str
        +close()
    }
    
    class PlaywrightDriver {
        -browser: Browser
        -page: Page
        +initialize()
        +navigate(url: str)
        +execute_prompt(prompt: str, model: str, behavior: str) str
        +close()
    }
    
    class SeleniumDriver {
        -driver: WebDriver
        +initialize()
        +navigate(url: str)
        +execute_prompt(prompt: str, model: str, behavior: str) str
        +close()
    }
    
    %% Visualization classes
    class VisualizationGenerator {
        +create_success_rate_pie(stats: Dict, output_path: str) str
        +create_technique_bar_chart(stats: Dict, output_path: str) str
        +create_model_bar_chart(stats: Dict, output_path: str) str
    }
    
    %% Base orchestrator
    class GraySwanOrchestrator {
        +recon_agent: ReconAgent
        +prompt_agent: PromptEngineerAgent
        +exploit_agent: ExploitDeliveryAgent
        +eval_agent: EvaluationAgent
        +run_pipeline(args: Dict) Dict
        +run_recon(args: Dict) Dict
        +run_prompt_generation(args: Dict) Dict
        +run_exploit_delivery(args: Dict) Dict
        +run_evaluation(args: Dict) Dict
    }
    
    %% CLI interface
    class CLI {
        +orchestrator: GraySwanOrchestrator
        +parse_args() Dict
        +main()
    }
    
    %% Relationships
    Agent <|-- ReconAgent
    Agent <|-- PromptEngineerAgent
    Agent <|-- ExploitDeliveryAgent
    Agent <|-- EvaluationAgent
    
    ReconAgent --> DiscordScraper : uses
    ExploitDeliveryAgent --> BrowserAutomationFactory : uses
    BrowserAutomationFactory --> BrowserDriver : creates
    BrowserDriver <|.. PlaywrightDriver : implements
    BrowserDriver <|.. SeleniumDriver : implements
    
    EvaluationAgent --> VisualizationGenerator : uses
    Agent --> LoggingUtils : uses
    
    GraySwanOrchestrator --> ReconAgent : orchestrates
    GraySwanOrchestrator --> PromptEngineerAgent : orchestrates
    GraySwanOrchestrator --> ExploitDeliveryAgent : orchestrates
    GraySwanOrchestrator --> EvaluationAgent : orchestrates
    
    CLI --> GraySwanOrchestrator : controls
```

## Class Descriptions

### Core Framework

#### Agent (Abstract Base Class)
Common functionality shared by all agents, including configuration validation and logging setup.

#### Specialized Agents

- **ReconAgent**: Gathers information about target models through web searches and Discord.
- **PromptEngineerAgent**: Generates diverse attack prompts based on reconnaissance data.
- **ExploitDeliveryAgent**: Executes prompts against target models via APIs or browsers.
- **EvaluationAgent**: Analyzes results and generates comprehensive reports with visualizations.

#### Orchestration

- **GraySwanOrchestrator**: Coordinates the execution of agents in the red-teaming pipeline.
- **CLI**: Provides a command-line interface for interacting with the framework.

### Utilities

#### Discord Integration

- **DiscordScraper**: Searches Discord channels for relevant information.

#### Logging

- **LoggingUtils**: Utility for setting up structured logging.

### Browser Automation

#### Factory and Interface

- **BrowserAutomationFactory**: Creates appropriate browser driver implementations.
- **BrowserDriver**: Interface defining browser automation capabilities.

#### Implementations

- **PlaywrightDriver**: Implements browser automation using Playwright.
- **SeleniumDriver**: Implements browser automation using Selenium.

### Visualization

- **VisualizationGenerator**: Creates visual representations of assessment results.

## Key Relationships

1. **Inheritance**: All agent classes inherit from the common Agent base class.
2. **Composition**: The orchestrator contains instances of all agent classes.
3. **Usage**:
   - ReconAgent uses DiscordScraper for Discord searches.
   - ExploitDeliveryAgent uses BrowserAutomationFactory for browser testing.
   - EvaluationAgent uses VisualizationGenerator for creating charts.
4. **Implementation**: Concrete browser drivers implement the BrowserDriver interface.

## Design Patterns

1. **Factory Pattern**: BrowserAutomationFactory creates appropriate driver implementations.
2. **Strategy Pattern**: Different browser automation strategies through the common interface.
3. **Command Pattern**: CLI translates commands into orchestrator method calls.
4. **Facade Pattern**: Orchestrator provides a simplified interface to the complex agent system. 