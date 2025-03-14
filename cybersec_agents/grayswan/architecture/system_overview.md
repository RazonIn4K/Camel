# Gray Swan Arena System Architecture Overview

The following diagram provides a high-level overview of the Gray Swan Arena architecture, showing the main components, data flows, and integration points.

```mermaid
flowchart TB
    %% Color definitions
    classDef agentClass fill:#f9f,stroke:#333,stroke-width:2px
    classDef utilityClass fill:#bbf,stroke:#333,stroke-width:1px
    classDef storageClass fill:#bfb,stroke:#333,stroke-width:1px
    classDef externalClass fill:#fbb,stroke:#333,stroke-width:1px
    classDef serviceClass fill:#ffb,stroke:#333,stroke-width:1px

    %% Main components
    User[User/Operator] --> CLI
    CLI[Command Line Interface] --> Orchestrator
    
    %% Core Orchestration
    subgraph Core["Gray Swan Arena Core"]
        Orchestrator[Orchestrator] --> ReconAgent
        Orchestrator --> PromptAgent
        Orchestrator --> ExploitAgent
        Orchestrator --> EvalAgent
    end
    
    %% Agents
    subgraph Agents["Agent Pipeline"]
        ReconAgent[Reconnaissance Agent]
        PromptAgent[Prompt Engineer Agent]
        ExploitAgent[Exploit Delivery Agent]
        EvalAgent[Evaluation Agent]
        
        ReconAgent --> |Recon Report| PromptAgent
        PromptAgent --> |Attack Prompts| ExploitAgent
        ExploitAgent --> |Exploit Results| EvalAgent
    end
    
    %% Utilities
    subgraph Utilities["Utility Services"]
        LoggingUtil[Logging Service]
        DiscordUtil[Discord Scraper]
        ConfigUtil[Configuration Manager]
    end
    
    %% External Services
    subgraph External["External Services"]
        WebSearch[Web Search API]
        LLMApis[LLM Model APIs]
        BrowserAuto[Browser Automation]
        Discord[Discord API]
    end
    
    %% Storage
    subgraph Storage["Data Storage"]
        ReconStorage[Recon Reports]
        PromptStorage[Prompt Lists]
        ExploitStorage[Exploit Logs]
        EvalStorage[Evaluation Reports]
        LogStorage[System Logs]
        DiscordStorage[Discord Results]
    end
    
    %% Connections from Agents to Utilities
    ReconAgent -.-> LoggingUtil
    PromptAgent -.-> LoggingUtil
    ExploitAgent -.-> LoggingUtil
    EvalAgent -.-> LoggingUtil
    ReconAgent -.-> ConfigUtil
    PromptAgent -.-> ConfigUtil
    ExploitAgent -.-> ConfigUtil
    EvalAgent -.-> ConfigUtil
    ReconAgent --> DiscordUtil
    
    %% Connections to external services
    ReconAgent --> WebSearch
    ReconAgent --> LLMApis
    DiscordUtil --> Discord
    ExploitAgent --> LLMApis
    ExploitAgent --> BrowserAuto
    PromptAgent --> LLMApis
    EvalAgent --> LLMApis
    
    %% Data storage connections
    ReconAgent --> ReconStorage
    PromptAgent --> PromptStorage
    ExploitAgent --> ExploitStorage
    EvalAgent --> EvalStorage
    LoggingUtil --> LogStorage
    DiscordUtil --> DiscordStorage
    
    %% Feedback loops
    EvalAgent -.-> |Improvement Suggestions| ReconAgent
    
    %% Apply classes
    class ReconAgent,PromptAgent,ExploitAgent,EvalAgent agentClass
    class LoggingUtil,DiscordUtil,ConfigUtil utilityClass
    class ReconStorage,PromptStorage,ExploitStorage,EvalStorage,LogStorage,DiscordStorage storageClass
    class WebSearch,LLMApis,BrowserAuto,Discord externalClass
    class CLI,Orchestrator serviceClass
```

## Legend

- **Purple Boxes**: Agent components that form the core pipeline
- **Blue Boxes**: Utility services that support the agents
- **Green Boxes**: Storage components for persistent data
- **Red Boxes**: External services and APIs
- **Yellow Boxes**: Service components like the CLI and orchestrator

## Interaction Types

- **Solid Lines**: Direct data flow and dependencies
- **Dotted Lines**: Service usage and utility connections

## Key Integration Points

1. **CLI to Orchestrator**: Entry point for user commands
2. **Agent Pipeline**: Sequential flow of information through the red-team assessment process
3. **External Service Integration**: Connections to web search, Discord, and AI model APIs
4. **Data Persistence**: Storage of reports, prompts, results, and logs
5. **Cross-cutting Concerns**: Logging and configuration services used by all components 