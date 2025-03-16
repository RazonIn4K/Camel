# Reconnaissance Agent - Detailed Architecture

This diagram provides a detailed view of the Reconnaissance Agent's internal architecture, showing components, workflows, and data processing.

```mermaid
flowchart TB
    %% Color definitions
    classDef coreComponent fill:#f9f,stroke:#333,stroke-width:2px
    classDef externalService fill:#fbb,stroke:#333,stroke-width:1px
    classDef dataProcessor fill:#bbf,stroke:#333,stroke-width:1px
    classDef dataStorage fill:#bfb,stroke:#333,stroke-width:1px
    classDef asyncProcess fill:#ffb,stroke:#333,stroke-width:1px
    classDef bottleneck fill:#fbb,stroke:#f00,stroke-width:3px,stroke-dasharray: 5 5

    %% Main component
    ReconAgent[Reconnaissance Agent]:::coreComponent
    
    %% Core subcomponents
    subgraph ReconComponents["Internal Components"]
        WebSearchMgr[Web Search Manager]:::dataProcessor
        DiscordSearchMgr[Discord Search Manager]:::dataProcessor
        ModelInfoCollector[Model Information Collector]:::dataProcessor
        TechniqueAnalyzer[Jailbreak Technique Analyzer]:::dataProcessor
        ReportGenerator[Report Generator]:::dataProcessor
    end
    
    %% External Services
    subgraph ExternalServices["External Services"]
        WebSearchAPI[Web Search API]:::externalService
        LLM[Large Language Models]:::externalService
        DiscordAPI[Discord API]:::externalService
    end
    
    %% Storage
    DB[(Report Database)]:::dataStorage
    Cache[(Web Search Cache)]:::dataStorage
    
    %% Input sources
    User([User])
    CliArgs([CLI Arguments])
    ConfigFile([Configuration])
    
    %% Flow within the agent
    User --> CliArgs
    CliArgs --> ReconAgent
    ConfigFile --> ReconAgent
    
    ReconAgent --> WebSearchMgr
    ReconAgent --> DiscordSearchMgr
    ReconAgent --> ModelInfoCollector
    ReconAgent --> TechniqueAnalyzer
    
    %% Web Search flow
    WebSearchMgr -- "1. Sends Search Query" --> WebSearchAPI
    WebSearchAPI -- "2. Returns Search Results" --> WebSearchMgr
    WebSearchMgr -- "3. Caches Results" --> Cache
    WebSearchMgr -- "4. Processes Results" --> |Structured Information| ModelInfoCollector
    WebSearchMgr -- "5. Processes Results" --> |Behavior Information| TechniqueAnalyzer
    
    %% Discord Search flow
    DiscordSearchMgr -- "1. Sends Query" --> DiscordAPI
    DiscordAPI -- "2. Returns Messages" --> DiscordSearchMgr
    DiscordSearchMgr -- "3. Filters & Processes" --> TechniqueAnalyzer
    
    %% Analysis & Report Generation
    ModelInfoCollector -- "Analyzed Model Data" --> ReportGenerator
    TechniqueAnalyzer -- "Analyzed Techniques" --> ReportGenerator
    Cache -- "Cached Search Results" --> ReportGenerator
    
    %% Final output
    ReportGenerator -- "1. Generates Report" --> |Recon Report| DB
    ReportGenerator -- "2. Returns Report" --> ReconAgent
    
    %% LLM Integration
    ReportGenerator -. "Uses for Analysis" .-> LLM
    TechniqueAnalyzer -. "Uses for Analysis" .-> LLM
    
    %% Highlight bottlenecks
    WebSearchAPI:::bottleneck
    DiscordAPI:::bottleneck
    
    %% Asynchronous processes
    WebSearchMgr:::asyncProcess
    DiscordSearchMgr:::asyncProcess
```

## Component Description

### Core Components
- **Reconnaissance Agent**: Main agent coordinating the reconnaissance process
- **Web Search Manager**: Manages web searches using external APIs
- **Discord Search Manager**: Searches Discord channels for relevant information
- **Model Information Collector**: Gathers and structures information about target models
- **Jailbreak Technique Analyzer**: Analyzes potential jailbreak techniques
- **Report Generator**: Creates comprehensive reconnaissance reports

### Data Flow
1. **Input Gathering**: Receives configuration and CLI arguments
2. **Web Search**: Queries external search APIs for target model information
3. **Discord Search**: Searches Discord channels for insights
4. **Information Processing**: Analyzes and structures the gathered information
5. **Report Generation**: Creates a comprehensive report using LLM assistance
6. **Storage**: Saves the report to the database and returns to the main agent

### Key Features
1. **Asynchronous Processing**: Web and Discord searches run asynchronously
2. **Caching**: Search results are cached to improve performance
3. **LLM Integration**: Uses LLMs for analysis and report generation
4. **Structured Output**: Generates structured reports for consumption by other agents

## Bottlenecks and Challenges
1. **External API Rate Limits**: Web Search and Discord APIs have rate limits (marked in red)
2. **LLM Processing Time**: Analysis using LLMs can be time-consuming
3. **Data Quality**: Quality of reconnaissance depends on available information

## Integration Points
- Input from user and configuration
- Output to Prompt Engineer Agent
- Data persistence in report database
- External service dependencies 