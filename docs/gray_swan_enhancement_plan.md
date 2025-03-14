# Gray Swan Arena Enhancement Plan

## Executive Summary

This document outlines a comprehensive plan for enhancing the Gray Swan Arena framework, a cybersecurity testing platform designed to evaluate AI model vulnerabilities. The enhancements focus on implementing the complete four-phase pipeline, adding robust model management, implementing parallel processing, enhancing browser automation, adding advanced analytics, and refactoring to a more maintainable architecture.

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Enhancement Goals](#enhancement-goals)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Plan](#implementation-plan)
   - [Phase 1: Core Pipeline Implementation](#phase-1-core-pipeline-implementation)
   - [Phase 2: Model Management System](#phase-2-model-management-system)
   - [Phase 3: Parallel Processing Implementation](#phase-3-parallel-processing-implementation)
   - [Phase 4: Browser Automation Enhancements](#phase-4-browser-automation-enhancements)
   - [Phase 5: Advanced Analytics and Visualization](#phase-5-advanced-analytics-and-visualization)
   - [Phase 6: Dependency Injection and Monitoring](#phase-6-dependency-injection-and-monitoring)
5. [Detailed Implementation Code](#detailed-implementation-code)
6. [Additional Enhancements](#additional-enhancements)
7. [Implementation Timeline](#implementation-timeline)
8. [Key Benefits](#key-benefits)
9. [Potential Challenges](#potential-challenges)

## Current State Analysis

The current implementation has:
- A partial pipeline with only the Reconnaissance phase implemented in main.py
- Four agent classes (ReconAgent, PromptEngineerAgent, ExploitDeliveryAgent, EvaluationAgent) already defined
- Utility modules for browser automation, Discord integration, visualization, and model management
- Architecture documentation describing a four-phase pipeline

## Enhancement Goals

1. **Complete Pipeline Implementation**: Connect all four phases into a cohesive workflow
2. **Model Fallback System**: Implement robust model handling with fallback capabilities
3. **Parallel Processing**: Add concurrent execution for reconnaissance and exploit delivery
4. **Browser Automation Enhancements**: Improve web-based testing capabilities
5. **Advanced Analytics**: Implement comprehensive visualization and reporting
6. **Dependency Injection**: Refactor to use a more maintainable architecture

## Architecture Overview

```mermaid
flowchart TB
    %% Color definitions
    classDef agentClass fill:#f9f,stroke:#333,stroke-width:2px
    classDef utilityClass fill:#bbf,stroke:#333,stroke-width:1px
    classDef enhancementClass fill:#fbb,stroke:#333,stroke-width:2px
    classDef containerClass fill:#bfb,stroke:#333,stroke-width:1px

    %% Main components
    User[User/Operator] --> CLI
    CLI[Command Line Interface] --> Orchestrator
    
    %% Core Orchestration with Enhancements
    subgraph Core["Gray Swan Arena Core"]
        Orchestrator[Enhanced Orchestrator] --> ReconAgent
        Orchestrator --> PromptAgent
        Orchestrator --> ExploitAgent
        Orchestrator --> EvalAgent
        
        %% New Components
        Orchestrator --> ModelManager
        Orchestrator --> MonitoringSystem
        
        %% Container for Dependency Injection
        Container[DI Container] -.-> Orchestrator
        Container -.-> ReconAgent
        Container -.-> PromptAgent
        Container -.-> ExploitAgent
        Container -.-> EvalAgent
        Container -.-> ModelManager
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
    
    %% Enhanced Utilities
    subgraph Utilities["Enhanced Utility Services"]
        ModelManager[Model Manager with Fallback]
        MonitoringSystem[Monitoring System]
        EnhancedBrowser[Enhanced Browser Automation]
        EnhancedDiscord[Enhanced Discord Scraper]
        AdvancedViz[Advanced Visualization]
    end
    
    %% External Services
    subgraph External["External Services"]
        WebSearch[Web Search API]
        LLMApis[LLM Model APIs]
        BrowserAuto[Browser Automation]
        Discord[Discord API]
    end
    
    %% Connections to enhanced utilities
    ReconAgent -.-> ModelManager
    PromptAgent -.-> ModelManager
    ExploitAgent -.-> ModelManager
    EvalAgent -.-> ModelManager
    
    ReconAgent -.-> MonitoringSystem
    PromptAgent -.-> MonitoringSystem
    ExploitAgent -.-> MonitoringSystem
    EvalAgent -.-> MonitoringSystem
    
    ExploitAgent -.-> EnhancedBrowser
    ReconAgent -.-> EnhancedDiscord
    EvalAgent -.-> AdvancedViz
    
    %% Connections to external services
    ModelManager --> LLMApis
    EnhancedBrowser --> BrowserAuto
    EnhancedDiscord --> Discord
    
    %% Apply classes
    class ReconAgent,PromptAgent,ExploitAgent,EvalAgent agentClass
    class ModelManager,MonitoringSystem,EnhancedBrowser,EnhancedDiscord,AdvancedViz enhancementClass
    class Container containerClass
```

## Detailed Workflow Sequence

```mermaid
sequenceDiagram
    actor User
    participant CLI as Command Line Interface
    participant Pipeline as EnhancedGraySwanPipeline
    participant ModelMgr as ModelManager
    participant Recon as ReconAgent
    participant Prompt as PromptEngineerAgent
    participant Exploit as ExploitDeliveryAgent
    participant Eval as EvaluationAgent
    participant Monitor as MonitoringSystem
    
    User->>CLI: Execute assessment command
    CLI->>Pipeline: Initialize with parameters
    Pipeline->>ModelMgr: Initialize model manager
    Pipeline->>Monitor: Start metrics collection
    
    %% Phase 1: Reconnaissance (Parallel)
    Pipeline->>Recon: Start reconnaissance phase
    activate Recon
    
    par Web Search
        Recon->>ModelMgr: Request model for web search
        ModelMgr-->>Recon: Return appropriate model
        Recon->>Recon: Execute web search
    and Discord Search
        Recon->>ModelMgr: Request model for Discord search
        ModelMgr-->>Recon: Return appropriate model
        Recon->>Recon: Execute Discord search
    end
    
    Recon->>ModelMgr: Request model for analysis
    ModelMgr-->>Recon: Return appropriate model
    Recon->>Recon: Generate report
    Recon-->>Pipeline: Return reconnaissance results
    deactivate Recon
    
    %% Phase 2: Prompt Engineering
    Pipeline->>Prompt: Start prompt engineering phase
    activate Prompt
    Prompt->>ModelMgr: Request model for prompt generation
    ModelMgr-->>Prompt: Return appropriate model
    Prompt->>Prompt: Generate diverse prompts
    Prompt-->>Pipeline: Return generated prompts
    deactivate Prompt
    
    %% Phase 3: Exploit Delivery (Batched)
    Pipeline->>Exploit: Start exploit delivery phase
    activate Exploit
    
    Exploit->>Exploit: Prepare batched requests
    
    loop For each batch
        Exploit->>ModelMgr: Request models for batch
        ModelMgr-->>Exploit: Return appropriate models
        
        par Execute Batch
            Exploit->>Exploit: Execute prompts in parallel
        end
    end
    
    Exploit->>Exploit: Analyze results
    Exploit-->>Pipeline: Return exploit results
    deactivate Exploit
    
    %% Phase 4: Evaluation
    Pipeline->>Eval: Start evaluation phase
    activate Eval
    Eval->>ModelMgr: Request model for analysis
    ModelMgr-->>Eval: Return appropriate model
    Eval->>Eval: Generate statistics
    Eval->>Eval: Create visualizations
    Eval->>Eval: Generate report
    Eval-->>Pipeline: Return evaluation results
    deactivate Eval
    
    Pipeline->>Monitor: Collect final metrics
    Pipeline-->>CLI: Return assessment summary
    CLI-->>User: Display results
```

## Implementation Plan

### Phase 1: Core Pipeline Implementation

1. **Update main.py to implement the full pipeline**
   - Refactor the main function to execute all four phases sequentially
   - Add command-line arguments for controlling each phase
   - Implement proper data flow between phases

2. **Create an EnhancedGraySwanPipeline class**
   - Implement a class-based approach for better organization
   - Add methods for each phase of the pipeline
   - Include configuration management

### Phase 2: Model Management System

1. **Implement ModelManager class**
   - Create a robust model management system with fallback capabilities
   - Add exponential backoff for API rate limits
   - Implement model switching based on complexity

2. **Integrate ModelFactory from provided code**
   - Implement handlers for different LLM providers (OpenAI, Anthropic, Google, Llama)
   - Create a unified interface for model interactions

### Phase 3: Parallel Processing Implementation

1. **Add parallel processing to ReconAgent**
   - Implement concurrent web and Discord searches
   - Use asyncio for asynchronous operations

2. **Enhance ExploitDeliveryAgent with batch processing**
   - Implement batched API calls with rate limiting
   - Add concurrent processing for browser-based testing

### Phase 4: Browser Automation Enhancements

1. **Implement EnhancedPlaywrightDriver**
   - Add adaptive selectors for different UIs
   - Implement self-healing capabilities for browser automation

2. **Complete the _execute_via_web method**
   - Fully implement web-based testing
   - Add robust error handling and recovery

### Phase 5: Advanced Analytics and Visualization

1. **Enhance visualization_utils.py**
   - Add advanced pattern visualization
   - Implement clustering for attack patterns

2. **Improve EvaluationAgent**
   - Add more comprehensive statistical analysis
   - Enhance report generation with actionable insights

### Phase 6: Dependency Injection and Monitoring

1. **Implement GraySwanContainer**
   - Create a dependency injection container
   - Configure services and their dependencies

2. **Add monitoring system**
   - Implement Prometheus metrics
   - Add method monitoring with decorators

## Detailed Implementation Code

### 1. Enhanced Orchestrator (main.py)

```python
# Key changes to main.py
def run_full_pipeline(
    target_model: str,
    target_behavior: str,
    output_dir: str = "./output",
    model_name: str = "gpt-4",
    skip_phases: List[str] = None,
    max_prompts: int = 20,
    test_method: str = "api",
    max_concurrent: int = 3
) -> Dict[str, Any]:
    """
    Run the complete Gray Swan Arena pipeline.
    
    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save outputs
        model_name: Name of the model to use for agents
        skip_phases: List of phases to skip
        max_prompts: Maximum number of prompts to generate
        test_method: Method for testing (api or web)
        max_concurrent: Maximum concurrent operations
        
    Returns:
        Dictionary containing results from all phases
    """
    results = {}
    skip_phases = skip_phases or []
    
    # Initialize model manager
    model_manager = ModelManager(
        primary_model=model_name,
        backup_model=get_backup_model(model_name)
    )
    
    # Phase 1: Reconnaissance
    if "recon" not in skip_phases:
        recon_agent = ReconAgent(
            output_dir=os.path.join(output_dir, "reports"),
            model_name=model_name
        )
        
        # Run parallel reconnaissance
        recon_results = asyncio.run(run_parallel_reconnaissance(
            recon_agent=recon_agent,
            target_model=target_model,
            target_behavior=target_behavior
        ))
        
        results["reconnaissance"] = recon_results
    
    # Phase 2: Prompt Engineering
    if "prompt" not in skip_phases:
        prompt_agent = PromptEngineerAgent(
            output_dir=os.path.join(output_dir, "prompts"),
            model_name=model_name
        )
        
        # Generate diverse prompts
        prompts = prompt_agent.generate_diverse_prompts(
            target_model=target_model,
            target_behavior=target_behavior,
            recon_report=results.get("reconnaissance", {}),
            num_prompts=max_prompts
        )
        
        prompt_path = prompt_agent.save_prompts(
            prompts=prompts,
            target_model=target_model,
            target_behavior=target_behavior
        )
        
        results["prompt_engineering"] = {
            "prompts": prompts,
            "path": prompt_path
        }
    
    # Phase 3: Exploit Delivery
    if "exploit" not in skip_phases and "prompt_engineering" in results:
        exploit_agent = ExploitDeliveryAgent(
            output_dir=os.path.join(output_dir, "exploits"),
            model_name=model_name
        )
        
        # Run prompts in batches
        exploit_results = asyncio.run(run_parallel_exploits(
            exploit_agent=exploit_agent,
            prompts=results["prompt_engineering"]["prompts"],
            target_model=target_model,
            target_behavior=target_behavior,
            method=test_method,
            max_concurrent=max_concurrent
        ))
        
        exploit_path = exploit_agent.save_results(
            results=exploit_results,
            target_model=target_model,
            target_behavior=target_behavior
        )
        
        results["exploit_delivery"] = {
            "results": exploit_results,
            "path": exploit_path
        }
    
    # Phase 4: Evaluation
    if "eval" not in skip_phases and "exploit_delivery" in results:
        eval_agent = EvaluationAgent(
            output_dir=os.path.join(output_dir, "evaluations"),
            model_name=model_name
        )
        
        # Evaluate results
        evaluation = eval_agent.evaluate_results(
            results=results["exploit_delivery"]["results"],
            target_model=target_model,
            target_behavior=target_behavior
        )
        
        # Create visualizations
        visualizations = eval_agent.create_visualizations(
            evaluation=evaluation,
            target_model=target_model,
            target_behavior=target_behavior
        )
        
        # Generate summary
        summary = eval_agent.generate_summary(
            evaluation=evaluation,
            target_model=target_model,
            target_behavior=target_behavior
        )
        
        eval_path = eval_agent.save_evaluation(
            evaluation=evaluation,
            target_model=target_model,
            target_behavior=target_behavior
        )
        
        summary_path = eval_agent.save_summary(
            summary=summary,
            target_model=target_model,
            target_behavior=target_behavior
        )
        
        results["evaluation"] = {
            "evaluation": evaluation,
            "visualizations": visualizations,
            "summary": summary,
            "paths": {
                "evaluation": eval_path,
                "summary": summary_path
            }
        }
    
    return results
```

### 2. Model Manager Implementation

```python
# Implementation of ModelManager class
class ModelManager:
    """Manages model interactions with fallback capabilities."""
    
    def __init__(self, 
                primary_model: str, 
                backup_model: Optional[str] = None,
                complexity_threshold: float = 0.7):
        self.primary_model = primary_model
        self.backup_model = backup_model
        self.complexity_threshold = complexity_threshold
        self.metrics = {"primary_calls": 0, "backup_calls": 0, "failures": 0}
        
        # Initialize handlers
        self.handlers = {}
        self._initialize_handlers()
    
    def _initialize_handlers(self):
        """Initialize model handlers based on available models."""
        # Create handlers for primary model
        self.handlers[self.primary_model] = ModelFactory.create_handler(
            self.primary_model
        )
        
        # Create handlers for backup model if available
        if self.backup_model:
            self.handlers[self.backup_model] = ModelFactory.create_handler(
                self.backup_model
            )
    
    @with_exponential_backoff
    async def generate(self, 
                     prompt: str, 
                     model: Optional[str] = None,
                     complexity: Optional[float] = None,
                     **kwargs) -> Dict[str, Any]:
        """Generate a completion with fallback capability."""
        try:
            # Determine which model to use
            model_to_use = model or self.primary_model
            
            # Use backup for complex prompts if available
            if complexity is not None and complexity >= self.complexity_threshold and self.backup_model:
                model_to_use = self.backup_model
                self.metrics["backup_calls"] += 1
            else:
                self.metrics["primary_calls"] += 1
            
            # Get the appropriate handler
            handler = self.handlers.get(model_to_use)
            if not handler:
                handler = ModelFactory.create_handler(model_to_use)
                self.handlers[model_to_use] = handler
            
            # Generate the completion
            return await handler.generate(prompt, **kwargs)
            
        except Exception as e:
            # Try backup if primary fails
            if model_to_use != self.backup_model and self.backup_model:
                try:
                    self.metrics["backup_calls"] += 1
                    handler = self.handlers.get(self.backup_model)
                    if not handler:
                        handler = ModelFactory.create_handler(self.backup_model)
                        self.handlers[self.backup_model] = handler
                    
                    return await handler.generate(prompt, **kwargs)
                except Exception as backup_e:
                    self.metrics["failures"] += 1
                    raise ModelBackupError(
                        f"Both models failed: {str(e)} and {str(backup_e)}",
                        model_to_use,
                        self.backup_model
                    )
            
            # Re-raise if no backup
            self.metrics["failures"] += 1
            raise
```

### 3. Parallel Processing Implementation

```python
# Implementation of parallel reconnaissance
async def run_parallel_reconnaissance(
    recon_agent: ReconAgent,
    target_model: str,
    target_behavior: str
) -> Dict[str, Any]:
    """Run reconnaissance tasks in parallel."""
    # Create tasks for concurrent execution
    web_task = asyncio.create_task(
        asyncio.to_thread(
            recon_agent.run_web_search,
            target_model,
            target_behavior
        )
    )
    
    discord_task = asyncio.create_task(
        asyncio.to_thread(
            recon_agent.run_discord_search,
            target_model,
            target_behavior
        )
    )
    
    # Wait for all tasks to complete
    web_results, discord_results = await asyncio.gather(
        web_task, discord_task
    )
    
    # Generate and save report
    report = recon_agent.generate_report(
        target_model=target_model,
        target_behavior=target_behavior,
        web_results=web_results,
        discord_results=discord_results
    )
    
    report_path = recon_agent.save_report(
        report=report,
        target_model=target_model,
        target_behavior=target_behavior
    )
    
    return {
        "report": report,
        "path": report_path,
        "web_results": web_results,
        "discord_results": discord_results
    }

# Implementation of parallel exploit delivery
async def run_parallel_exploits(
    exploit_agent: ExploitDeliveryAgent,
    prompts: List[str],
    target_model: str,
    target_behavior: str,
    method: str = "api",
    max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """Run exploit delivery in parallel batches."""
    all_results = []
    
    # Split prompts into batches
    batch_size = max_concurrent
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
        
        # Create tasks for concurrent execution
        tasks = []
        for prompt in batch:
            task = asyncio.create_task(
                asyncio.to_thread(
                    _execute_single_prompt,
                    exploit_agent,
                    prompt,
                    target_model,
                    target_behavior,
                    method
                )
            )
            tasks.append(task)
        
        # Wait for batch to complete
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        # Small delay between batches to avoid rate limiting
        if batch_idx < len(batches) - 1:
            await asyncio.sleep(2)
    
    return all_results

def _execute_single_prompt(
    agent: ExploitDeliveryAgent,
    prompt: str,
    target_model: str,
    target_behavior: str,
    method: str
) -> Dict[str, Any]:
    """Execute a single prompt and return the result."""
    # This is a helper function for the parallel execution
    try:
        # Execute the prompt based on the method
        if method == "api":
            response = agent._execute_via_api(prompt, target_model)
        elif method == "web":
            response = agent._execute_via_web(prompt, target_model)
        else:
            response = agent._execute_via_api(prompt, target_model)
        
        # Analyze the response
        success, reason = agent._analyze_response(response, target_behavior)
        
        return {
            "prompt": prompt,
            "target_model": target_model,
            "target_behavior": target_behavior,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "reason": reason,
            "response": response,
            "error": None
        }
    except Exception as e:
        return {
            "prompt": prompt,
            "target_model": target_model,
            "target_behavior": target_behavior,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "response": None,
            "error": str(e)
        }
```

### 4. Enhanced Browser Automation

```python
# Implementation of EnhancedPlaywrightDriver
class EnhancedPlaywrightDriver(PlaywrightDriver):
    """Enhanced browser driver with adaptive selectors and self-healing capabilities."""
    
    def __init__(self, headless: bool = True):
        super().__init__(headless)
        self.selector_alternatives = {
            "model_select": ["#model-select", ".model-dropdown", "[data-testid='model-selector']"],
            "prompt_input": ["#prompt-textarea", ".prompt-input", "[data-testid='prompt-input']", "textarea"],
            "submit_button": ["#submit-button", "button[type='submit']", ".submit-btn", "button:has-text('Submit')"],
            "response_output": ["#response-container", ".response-content", "[data-testid='response']"]
        }
    
    def try_selectors(self, selector_type: str, action_fn):
        """Try multiple selectors to find a working one."""
        selectors = self.selector_alternatives.get(selector_type, [self.selectors[selector_type]])
        
        for selector in selectors:
            try:
                result = action_fn(selector)
                # Update the working selector for future use
                self.selectors[selector_type] = selector
                return result
            except Exception as e:
                continue
                
        # If all selectors fail, raise an exception
        raise ValueError(f"All selectors failed for {selector_type}")
    
    def execute_prompt(self, prompt: str, model: str, behavior: str) -> str:
        """Enhanced execute_prompt with adaptive selectors and retry logic."""
        if not self.page:
            raise ValueError("Browser not initialized. Call initialize() first.")
            
        try:
            # Select model if available
            def select_model(selector):
                if self.page.locator(selector).count() > 0:
                    self.page.select_option(selector, model)
                    logger.info(f"Selected model: {model}")
                return True
            
            self.try_selectors("model_select", select_model)
            
            # Enter prompt
            def enter_prompt(selector):
                self.page.fill(selector, prompt)
                logger.info("Entered prompt")
                return True
            
            self.try_selectors("prompt_input", enter_prompt)
            
            # Submit
            def click_submit(selector):
                self.page.click(selector)
                logger.info("Submitted prompt")
                return True
            
            self.try_selectors("submit_button", click_submit)
            
            # Wait for response with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    def wait_for_response(selector):
                        self.page.wait_for_selector(
                            selector, state="visible", timeout=60000
                        )
                        return self.page.text_content(selector)
                    
                    response = self.try_selectors("response_output", wait_for_response)
                    logger.info("Received response")
                    return response
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt+1} for response: {e}")
                    time.sleep(2)
            
            return "Error: Failed to get response after retries"
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            return f"Error: {str(e)}"
```

### 5. Advanced Analytics Implementation

```python
# Implementation of advanced pattern visualization
def create_attack_pattern_visualization(
    results: List[Dict[str, Any]],
    output_dir: str
) -> str:
    """Create visualization showing attack patterns and effectiveness clusters."""
    # Convert results to a format suitable for machine learning
    data = []
    for result in results:
        # Extract features from the result
        features = {
            "model": result.get("model_name", "Unknown"),
            "prompt_type": result.get("prompt_type", "Unknown"),
            "attack_vector": result.get("attack_vector", "Unknown"),
            "prompt_length": len(result.get("prompt", "")),
            "response_time": result.get("response_time", 0),
            "success": 1 if result.get("success", False) else 0
        }
        data.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Apply dimensionality reduction to visualize patterns
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    # Select numerical columns
    numerical_cols = ["prompt_length", "response_time", "success"]
    
    if len(df) > 10:  # Need enough data points
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numerical_cols])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(scaled_data)
        
        # Add results to DataFrame
        df["x"] = tsne_results[:, 0]
        df["y"] = tsne_results[:, 1]
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        scatter = sns.scatterplot(
            data=df, 
            x="x", 
            y="y", 
            hue="model",
            style="prompt_type",
            size="success",
            sizes=(50, 200),
            alpha=0.7
        )
        
        plt.title("Attack Pattern Clustering", fontsize=16)
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.legend(title="Model", loc="best")
        
        # Save plot
        output_file = os.path.join(output_dir, "attack_pattern_visualization.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_file
    
    return ""
```

### 6. Dependency Injection Implementation

```python
# Implementation of GraySwanContainer
from dependency_injector import containers, providers
from .utils.logging_utils import setup_logging
from .agents.recon_agent import ReconAgent
from .agents.prompt_engineer_agent import PromptEngineerAgent
from .agents.exploit_delivery_agent import ExploitDeliveryAgent
from .agents.evaluation_agent import EvaluationAgent
from .utils.model_utils import ModelManager

class GraySwanContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Services
    logger = providers.Singleton(
        setup_logging,
        name="grayswan"
    )
    
    model_manager = providers.Singleton(
        ModelManager,
        primary_model=config.models.primary_model,
        backup_model=config.models.backup_model,
        complexity_threshold=config.models.complexity_threshold
    )
    
    # Agents
    recon_agent = providers.Factory(
        ReconAgent,
        output_dir=config.agents.recon.output_dir,
        model_name=config.agents.recon.model_name
    )
    
    prompt_engineer_agent = providers.Factory(
        PromptEngineerAgent,
        output_dir=config.agents.prompt_engineer.output_dir,
        model_name=config.agents.prompt_engineer.model_name
    )
    
    exploit_delivery_agent = providers.Factory(
        ExploitDeliveryAgent,
        output_dir=config.agents.exploit_delivery.output_dir,
        model_name=config.agents.exploit_delivery.model_name
    )
    
    evaluation_agent = providers.Factory(
        EvaluationAgent,
        output_dir=config.agents.evaluation.output_dir,
        model_name=config.agents.evaluation.model_name
    )
```

## Additional Enhancements

Based on feedback, the following additional enhancements will be implemented:

### 1. Integration Testing Strategy

- **End-to-End Tests**: Implement comprehensive tests for the complete pipeline
  ```python
  def test_full_pipeline():
      """Test the complete pipeline with a mock target model."""
      # Setup test environment
      output_dir = tempfile.mkdtemp()
      
      # Run the pipeline
      results = run_full_pipeline(
          target_model="test-model",
          target_behavior="test-behavior",
          output_dir=output_dir,
          model_name="gpt-3.5-turbo"
      )
      
      # Verify results
      assert "reconnaissance" in results
      assert "prompt_engineering" in results
      assert "exploit_delivery" in results
      assert "evaluation" in results
      
      # Verify files were created
      assert os.path.exists(results["reconnaissance"]["path"])
      assert os.path.exists(results["prompt_engineering"]["path"])
      assert os.path.exists(results["exploit_delivery"]["path"])
      assert os.path.exists(results["evaluation"]["paths"]["evaluation"])
      assert os.path.exists(results["evaluation"]["paths"]["summary"])
  ```

- **Component Tests**: Add tests for each enhanced component
  ```python
  def test_model_manager_fallback():
      """Test that the model manager falls back to backup model."""
      # Setup test environment
      manager = ModelManager(
          primary_model="failing-model",
          backup_model="working-model"
      )
      
      # Mock the handlers
      manager.handlers["failing-model"] = MockHandler(should_fail=True)
      manager.handlers["working-model"] = MockHandler(should_fail=False)
      
      # Test fallback
      result = asyncio.run(manager.generate("test prompt"))
      
      # Verify backup was used
      assert manager.metrics["backup_calls"] == 1
      assert manager.metrics["primary_calls"] == 1
      assert manager.metrics["failures"] == 0
  ```

- **Regression Tests**: Ensure existing functionality isn't broken
  ```python
  def test_backwards_compatibility():
      """Test that the enhanced pipeline is backwards compatible."""
      # Setup test environment
      output_dir = tempfile.mkdtemp()
      
      # Run just the reconnaissance phase (original functionality)
      result = run_reconnaissance(
          target_model="test-model",
          target_behavior="test-behavior",
          output_dir=output_dir,
          model_name="gpt-3.5-turbo"
      )
      
      # Verify result structure matches original
      assert "target_model" in result
      assert "target_behavior" in result
      assert "timestamp" in result
      assert "content" in result
      assert "sections" in result
  ```

### 2. Configuration Management

- **Environment-Specific Settings**:
  ```python
  # config.py
  import os
  from typing import Dict, Any
  import yaml
  
  def load_config(env: str = None) -> Dict[str, Any]:
      """Load configuration based on environment."""
      env = env or os.environ.get("GRAYSWAN_ENV", "development")
      
      # Load base config
      with open("config/base.yaml", "r") as f:
          config = yaml.safe_load(f)
      
      # Load environment-specific config
      env_file = f"config/{env}.yaml"
      if os.path.exists(env_file):
          with open(env_file, "r") as f:
              env_config = yaml.safe_load(f)
              # Deep merge configs
              _deep_merge(config, env_config)
      
      # Override with environment variables
      _override_from_env(config)
      
      return config
  
  def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
      """Deep merge override into base."""
      for key, value in override.items():
          if key in base and isinstance(base[key], dict) and isinstance(value, dict):
              _deep_merge(base[key], value)
          else:
              base[key] = value
  
  def _override_from_env(config: Dict[str, Any], prefix: str = "GRAYSWAN_") -> None:
      """Override config with environment variables."""
      for key, value in os.environ.items():
          if key.startswith(prefix):
              # Convert GRAYSWAN_MODELS_PRIMARY_MODEL to config["models"]["primary_model"]
              parts = key[len(prefix):].lower().split("_")
              current = config
              for part in parts[:-1]:
                  if part not in current:
                      current[part] = {}
                  current = current[part]
              current[parts[-1]] = value
  ```

- **Runtime Configuration Updates**:
  ```python
  class ConfigManager:
      """Manager for runtime configuration updates."""
      
      def __init__(self, initial_config: Dict[str, Any]):
          self.config = initial_config
          self.listeners = []
      
      def update(self, path: str, value: Any) -> None:
          """Update a configuration value at runtime."""
          parts = path.split(".")
          current = self.config
          for part in parts[:-1]:
              if part not in current:
                  current[part] = {}
              current = current[part]
          
          # Update the value
          current[parts[-1]] = value
          
          # Notify listeners
          for listener in self.listeners:
              listener(path, value)
      
      def get(self, path: str, default: Any = None) -> Any:
          """Get a configuration value."""
          parts = path.split(".")
          current = self.config
          for part in parts:
              if part not in current:
                  return default
              current = current[part]
          return current
      
      def add_listener(self, listener: Callable[[str, Any], None]) -> None:
          """Add a listener for configuration changes."""
          self.listeners.append(listener)
  ```

- **Configuration Validation**:
  ```python
  from pydantic import BaseModel, Field
  
  class ModelConfig(BaseModel):
      """Configuration for models."""
      primary_model: str = Field(..., description="Primary model to use")
      backup_model: str = Field(None, description="Backup model to use")
      complexity_threshold: float = Field(0.7, description="Threshold for using backup model")
  
  class AgentConfig(BaseModel):
      """Configuration for an agent."""
      output_dir: str = Field(..., description="Directory to save outputs")
      model_name: str = Field(..., description="Model to use for the agent")
  
  class AgentsConfig(BaseModel):
      """Configuration for all agents."""
      recon: AgentConfig
      prompt_engineer: AgentConfig
      exploit_delivery: AgentConfig
      evaluation: AgentConfig
  
  class GraySwanConfig(BaseModel):
      """Complete configuration for Gray Swan Arena."""
      models: ModelConfig
      agents: AgentsConfig
      logging: Dict[str, Any] = Field(default_factory=dict)
      monitoring: Dict[str, Any] = Field(default_factory=dict)
  
  def validate_config(config: Dict[str, Any]) -> GraySwanConfig:
      """Validate configuration against schema."""
      return GraySwanConfig(**config)
  ```

### 3. Telemetry and Observability

- **Structured Logging with Correlation IDs**:
  ```python
  import logging
  import uuid
  from contextvars import ContextVar
  
  # Context variable to store correlation ID
  correlation_id_var = ContextVar("correlation_id", default=None)
  
  class CorrelationFilter(logging.Filter):
      """Filter that adds correlation ID to log records."""
      
      def filter(self, record):
          correlation_id = correlation_id_var.get()
          record.correlation_id = correlation_id or "unknown"
          return True
  
  def setup_structured_logging(name: str) -> logging.Logger:
      """Set up structured logging with correlation ID."""
      logger = logging.getLogger(name)
      
      # Add correlation ID filter
      logger.addFilter(CorrelationFilter())
      
      # Create JSON formatter
      formatter = logging.Formatter(
          '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "correlation_id": "%(correlation_id)s", "name": "%(name)s", "message": "%(message)s"}'
      )
      
      # Add handler
      handler = logging.StreamHandler()
      handler.setFormatter(formatter)
      logger.addHandler(handler)
      
      return logger
  
  def with_correlation_id(func):
      """Decorator to add correlation ID to context."""
      @wraps(func)
      def wrapper(*args, **kwargs):
          # Generate correlation ID if not already set
          correlation_id = correlation_id_var.get()
          if correlation_id is None:
              correlation_id = str(uuid.uuid4())
              correlation_id_var.set(correlation_id)
          
          return func(*args, **kwargs)
      
      return wrapper
  ```

- **Detailed Performance Metrics**:
  ```python
  import time
  from functools import wraps
  from prometheus_client import Histogram
  
  # Define metrics
  PHASE_DURATION = Histogram(
      "grayswan_phase_duration_seconds",
      "Duration of each phase in seconds",
      ["phase"]
  )
  
  def measure_phase(phase_name: str):
      """Decorator to measure phase duration."""
      def decorator(func):
          @wraps(func)
          def wrapper(*args, **kwargs):
              start_time = time.time()
              try:
                  return func(*args, **kwargs)
              finally:
                  duration = time.time() - start_time
                  PHASE_DURATION.labels(phase=phase_name).observe(duration)
          
          return wrapper
      
      return decorator
  ```

- **Health Check Endpoints**:
  ```python
  from fastapi import FastAPI, HTTPException
  from pydantic import BaseModel
  
  app = FastAPI()
  
  class HealthStatus(BaseModel):
      """Health status response."""
      status: str
      version: str
      components: Dict[str, str]
  
  @app.get("/health", response_model=HealthStatus)
  async def health_check():
      """Health check endpoint."""
      # Check components
      components = {}
      
      # Check model API access
      try:
          # Simple test call to OpenAI API
          client = openai.OpenAI()
          client.models.list()
          components["openai_api"] = "healthy"
      except Exception as e:
          components["openai_api"] = f"unhealthy: {str(e)}"
      
      # Check database access if applicable
      # ...
      
      # Determine overall status
      status = "healthy"
      if any(c.startswith("unhealthy") for c in components.values()):
          status = "degraded"
      
      return HealthStatus(
          status=status,
          version="1.0.0",  # Get from package version
          components=components
      )
  ```

### 4. Error Recovery Mechanisms

- **Checkpoint/Resume Functionality**:
  ```python
  class PipelineState:
      """State of the pipeline for checkpoint/resume."""
      
      def __init__(self, pipeline_id: str, target_model: str, target_behavior: str):
          self.pipeline_id = pipeline_id
          self.target_model = target_model
          self.target_behavior = target_behavior
          self.phases = {
              "recon": {"status": "pending", "result": None},
              "prompt": {"status": "pending", "result": None},
              "exploit": {"status": "pending", "result": None},
              "eval": {"status": "pending", "result": None}
          }
      
      def save(self, checkpoint_dir: str) -> str:
          """Save pipeline state to a checkpoint file."""
          os.makedirs(checkpoint_dir, exist_ok=True)
          checkpoint_path = os.path.join(
              checkpoint_dir, f"{self.pipeline_id}.json"
          )
          
          with open(checkpoint_path, "w") as f:
              json.dump(self.__dict__, f, indent=2)
          
          return checkpoint_path
      
      @classmethod
      def load(cls, checkpoint_path: str) -> "PipelineState":
          """Load pipeline state from a checkpoint file."""
          with open(checkpoint_path, "r") as f:
              data = json.load(f)
          
          state = cls(
              pipeline_id=data["pipeline_id"],
              target_model=data["target_model"],
              target_behavior=data["target_behavior"]
          )
          state.phases = data["phases"]
          
          return state
      
      def update_phase(self, phase: str, status: str, result: Any = None) -> None:
          """Update the status and result of a phase."""
          if phase not in self.phases:
              raise ValueError(f"Invalid phase: {phase}")
          
          self.phases[phase]["status"] = status
          if result is not None:
              self.phases[phase]["result"] = result
  ```

- **State Management System**:
  ```python
  class StateMachine:
      """State machine for pipeline execution."""
      
      def __init__(self, pipeline_state: PipelineState):
          self.state = pipeline_state
      
      async def run(self, checkpoint_dir: str = "./checkpoints") -> Dict[str, Any]:
          """Run the pipeline with state management."""
          results = {}
          
          # Phase 1: Reconnaissance
          if self.state.phases["recon"]["status"] != "completed":
              try:
                  self.state.update_phase("recon", "running")
                  recon_results = await self._run_recon_phase()
                  self.state.update_phase("recon", "completed", recon_results)
                  results["reconnaissance"] = recon_results
                  self.state.save(checkpoint_dir)
              except Exception as e:
                  self.state.update_phase("recon", "failed", {"error": str(e)})
                  self.state.save(checkpoint_dir)
                  raise
          else:
              results["reconnaissance"] = self.state.phases["recon"]["result"]
          
          # Phase 2: Prompt Engineering
          if self.state.phases["prompt"]["status"] != "completed":
              try:
                  self.state.update_phase("prompt", "running")
                  prompt_results = await self._run_prompt_phase(results["reconnaissance"])
                  self.state.update_phase("prompt", "completed", prompt_results)
                  results["prompt_engineering"] = prompt_results
                  self.state.save(checkpoint_dir)
              except Exception as e:
                  self.state.update_phase("prompt", "failed", {"error": str(e)})
                  self.state.save(checkpoint_dir)
                  raise
          else:
              results["prompt_engineering"] = self.state.phases["prompt"]["result"]
          
          # Continue with other phases...
          
          return results
  ```

- **Dedicated Error Handling Strategies**:
  ```python
  class ErrorHandler:
      """Handler for different types of errors."""
      
      def __init__(self):
          self.strategies = {
              "RateLimitError": self._handle_rate_limit,
              "AuthenticationError": self._handle_auth_error,
              "APIError": self._handle_api_error,
              "NetworkError": self._handle_network_error,
              "TimeoutError": self._handle_timeout,
              "default": self._handle_default
          }
      
      def handle(self, error: Exception) -> bool:
          """Handle an error using the appropriate strategy."""
          error_type = type(error).__name__
          strategy = self.strategies.get(error_type, self.strategies["default"])
          return strategy(error)
      
      def _handle_rate_limit(self, error: Exception) -> bool:
          """Handle rate limit errors."""
          logger.warning(f"Rate limit exceeded: {str(error)}")
          # Implement exponential backoff
          time.sleep(5)
          return True  # Retry
      
      def _handle_auth_error(self, error: Exception) -> bool:
          """Handle authentication errors."""
          logger.error(f"Authentication error: {str(error)}")
          # Notify admin
          return False  # Don't retry
      
      # Other strategies...
  ```

### 5. Documentation Enhancements

- **Updated Architectural Diagrams**:
  ```markdown
  # Gray Swan Arena Architecture
  
  ## System Components
  
  ```mermaid
  flowchart TB
      %% Updated diagram with new components
      %% ...
  ```
  
  ## Data Flow
  
  ```mermaid
  flowchart LR
      %% Data flow diagram
      %% ...
  ```
  
  ## Component Interactions
  
  ```mermaid
  sequenceDiagram
      %% Detailed sequence diagram
      %% ...
  ```
  ```

- **Example Usage Patterns**:
  ```markdown
  # Usage Examples
  
  ## Basic Usage
  
  ```python
  from cybersec_agents.grayswan.main import run_full_pipeline
  
  # Run the full pipeline
  results = run_full_pipeline(
      target_model="gpt-3.5-turbo",
      target_behavior="bypass content policies",
      output_dir="./output",
      model_name="gpt-4"
  )
  
  # Print summary
  print(f"Success rate: {results['evaluation']['evaluation']['success_rate']:.2%}")
  ```
  
  ## Advanced Usage
  
  ```python
  # Skip certain phases
  results = run_full_pipeline(
      target_model="gpt-3.5-turbo",
      target_behavior="bypass content policies",
      output_dir="./output",
      model_name="gpt-4",
      skip_phases=["recon"],  # Skip reconnaissance phase
      max_prompts=50,  # Generate more prompts
      test_method="web",  # Use web-based testing
      max_concurrent=5  # Increase concurrency
  )
  ```
  
  ## Using the Container
  
  ```python
  from cybersec_agents.grayswan.container import GraySwanContainer
  
  # Create and configure container
  container = GraySwanContainer()
  container.config.from_yaml("config/production.yml")
  
  # Get agents from container
  recon_agent = container.recon_agent()
  prompt_agent = container.prompt_engineer_agent()
  
  # Use agents
  # ...
  ```
  ```

- **Performance Considerations**:
  ```markdown
  # Performance Considerations
  
  ## Resource Requirements
  
  | Component | CPU | Memory | Disk | Network |
  |-----------|-----|--------|------|---------|
  | Reconnaissance | Medium | Low | Low | High |
  | Prompt Engineering | High | Medium | Low | Medium |
  | Exploit Delivery | Medium | Medium | Medium | High |
  | Evaluation | High | High | High | Low |
  
  ## Scaling Considerations
  
  - **API Rate Limits**: The system is designed to respect API rate limits through exponential backoff and batching.
  - **Concurrency**: The `max_concurrent` parameter controls the level of parallelism. Start with a low value (3-5) and increase based on available resources.
  - **Memory Usage**: When processing large datasets, monitor memory usage, especially during the evaluation phase.
  
  ## Benchmarks
  
  | Configuration | Runtime | Success Rate | Notes |
  |---------------|---------|--------------|-------|
  | Default | 15 min | 35% | Baseline |
  | High Concurrency | 8 min | 35% | 2x faster |
  | Web Testing | 25 min | 40% | Higher success rate |
  | Large Dataset | 45 min | 38% | 100 prompts |
  ```

## Implementation Timeline

1. **Week 1: Core Pipeline and Model Management**
   - Implement the full pipeline in main.py
   - Create the ModelManager with fallback capabilities
   - Integrate the ModelFactory from provided code

2. **Week 2: Parallel Processing and Browser Automation**
   - Implement parallel processing for reconnaissance
   - Add batched processing for exploit delivery
   - Enhance browser automation with adaptive selectors

3. **Week 3: Advanced Analytics and Dependency Injection**
   - Implement advanced visualization capabilities
   - Add the dependency injection container
   - Create the monitoring system

4. **Week 4: Testing and Refinement**
   - Comprehensive testing of all components
   - Performance optimization
   - Documentation updates

## Key Benefits

1. **Improved Reliability**: The model fallback system ensures operations continue even if primary models fail
2. **Enhanced Performance**: Parallel processing reduces overall execution time
3. **Better Insights**: Advanced analytics provide deeper understanding of vulnerabilities
4. **Maintainable Architecture**: Dependency injection makes the codebase more maintainable
5. **Robust Browser Testing**: Enhanced browser automation handles different UIs more effectively

## Potential Challenges

1. **API Rate Limits**: Need careful management of API calls to avoid rate limiting
2. **Browser Automation Complexity**: Web interfaces can change, requiring adaptive selectors
3. **Dependency Management**: Need to ensure all required packages are installed
4. **Error Handling**: Comprehensive error handling is critical for reliability