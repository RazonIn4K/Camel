# Evaluation Agent - Detailed Architecture

This diagram provides a detailed view of the Evaluation Agent's internal architecture, showing its data processing, visualization generation, and report creation workflows.

```mermaid
flowchart TB
    %% Color definitions
    classDef coreComponent fill:#f9f,stroke:#333,stroke-width:2px
    classDef subComponent fill:#f9d,stroke:#333,stroke-width:1px
    classDef dataProcessor fill:#bbf,stroke:#333,stroke-width:1px
    classDef vizComponent fill:#bdf,stroke:#333,stroke-width:1px
    classDef dataStorage fill:#bfb,stroke:#333,stroke-width:1px
    classDef asyncProcess fill:#ffb,stroke:#333,stroke-width:1px
    classDef llmComponent fill:#fdb,stroke:#333,stroke-width:1px

    %% Main component
    EvalAgent[Evaluation Agent]:::coreComponent
    
    %% Core subcomponents
    subgraph EvalComponents["Internal Components"]
        DataLoader[Data Loader]:::dataProcessor
        StatCalculator[Statistics Calculator]:::dataProcessor
        
        subgraph Visualization["Visualization Engine"]
            VizFactory[Visualization Factory]:::vizComponent
            PieChartGen[Pie Chart Generator]:::vizComponent
            BarChartGen[Bar Chart Generator]:::vizComponent
            TimelineGen[Timeline Generator]:::vizComponent
            ImageSaver[Image Saver]:::dataProcessor
        end
        
        subgraph ReportGen["Report Generation"]
            JsonReportGen[JSON Report Generator]:::dataProcessor
            MarkdownGen[Markdown Generator]:::dataProcessor
            LLMAnalyzer[LLM-based Analysis]:::llmComponent
            EmbedManager[Visualization Embedder]:::dataProcessor
        end
    end
    
    %% External components
    LLM[Large Language Models]:::llmComponent
    
    %% Storage
    ResultsDB[(Results Database)]:::dataStorage
    ReconDB[(Recon Reports DB)]:::dataStorage
    ReportStorage[(Report Storage)]:::dataStorage
    VizStorage[(Visualization Storage)]:::dataStorage
    
    %% Data flow - Initial loading
    EvalAgent --> DataLoader
    DataLoader -- "Loads Exploit Results" --> ResultsDB
    DataLoader -- "Loads Recon Data (Optional)" --> ReconDB
    
    %% Statistics calculation
    DataLoader -- "Provides Raw Data" --> StatCalculator
    StatCalculator -- "Calculates Stats" --> |Statistics Data| VizFactory
    StatCalculator -- "Provides Statistics" --> JsonReportGen
    
    %% Visualization generation
    VizFactory -- "Success Rate Viz" --> PieChartGen
    VizFactory -- "Technique Comparison" --> BarChartGen
    VizFactory -- "Model Comparison" --> BarChartGen
    VizFactory -- "Time Analysis" --> TimelineGen
    
    PieChartGen -- "Generated Chart" --> ImageSaver
    BarChartGen -- "Generated Chart" --> ImageSaver
    TimelineGen -- "Generated Chart" --> ImageSaver
    
    ImageSaver -- "Saves Visualization" --> VizStorage
    
    %% Report generation
    JsonReportGen -- "Generates JSON" --> ReportStorage
    VizStorage -- "Visualization Paths" --> EmbedManager
    DataLoader -- "Raw Results" --> LLMAnalyzer
    StatCalculator -- "Statistics" --> LLMAnalyzer
    LLMAnalyzer -- "Uses for Analysis" --> LLM
    LLM -- "Analysis & Recommendations" --> LLMAnalyzer
    
    LLMAnalyzer -- "Analysis" --> MarkdownGen
    EmbedManager -- "Embedded Visualizations" --> MarkdownGen
    JsonReportGen -- "Report Data" --> MarkdownGen
    
    MarkdownGen -- "Generates Markdown" --> ReportStorage
    
    %% Final output
    ReportStorage -- "Returns Reports" --> EvalAgent
```

## Component Architecture

### Core Components
- **Evaluation Agent**: Main agent coordinating the evaluation process
- **Data Loader**: Loads exploit results and optional reconnaissance data
- **Statistics Calculator**: Processes raw data to calculate comprehensive statistics

### Visualization Engine
- **Visualization Factory**: Central factory that determines which visualizations to generate
- **Pie Chart Generator**: Creates pie charts for overall success rates
- **Bar Chart Generator**: Creates horizontal bar charts for comparing techniques and models
- **Timeline Generator**: Creates timeline visualizations for temporal analysis
- **Image Saver**: Saves generated visualizations to storage

### Report Generation
- **JSON Report Generator**: Creates structured JSON reports with statistics
- **LLM-based Analysis**: Uses LLMs to analyze results and generate insights
- **Visualization Embedder**: Manages embedding visualizations into reports
- **Markdown Generator**: Creates comprehensive markdown reports with embedded visualizations

## Critical Workflows

### Data Processing Flow
1. Load raw exploit results from the database
2. Optionally load reconnaissance data for context
3. Calculate comprehensive statistics from the raw data
4. Pass statistics to visualization and report generation components

### Visualization Generation Flow
1. Factory determines required visualizations based on available data
2. Individual generators create specific chart types
3. Generated visualizations are saved to storage
4. Visualization paths are provided to the report generation process

### Report Generation Flow
1. Create structured JSON report with raw data and statistics
2. Use LLM to analyze results and generate insights
3. Embed visualizations into markdown report
4. Generate final markdown report with all components
5. Save reports to storage

## Key Features

### Statistical Analysis
- **Success Rate Calculation**: Overall and per-category success rates
- **Technique Effectiveness**: Comparative analysis of different techniques
- **Model Vulnerability**: Analysis of vulnerability by model type
- **Confidence Intervals**: Statistical confidence for all metrics

### Visualization Capabilities
- **Overall Success Rate**: Pie charts showing success vs. failure
- **Technique Comparison**: Horizontal bar charts ranking techniques by effectiveness
- **Model Comparison**: Horizontal bar charts ranking models by vulnerability
- **Temporal Analysis**: Timeline visualizations for time-based patterns

### Report Generation
- **Structured Data**: JSON reports with complete statistical data
- **Human-Readable Reports**: Markdown reports designed for easy consumption
- **AI-Assisted Analysis**: LLM-generated insights and recommendations
- **Visual Integration**: Embedded visualizations within reports

## Integration Points
- **Data Sources**: Loading from results and reconnaissance databases
- **LLM Integration**: Using language models for advanced analysis
- **Output Storage**: Saving reports and visualizations for later reference
- **Output Consumption**: Providing actionable insights to security teams 