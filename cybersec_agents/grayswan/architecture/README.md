# Gray Swan Arena - Architecture Diagrams

This directory contains comprehensive architecture diagrams for the Gray Swan Arena framework, providing visual representations of the system at different levels of abstraction and from different perspectives.

## Overview

The Gray Swan Arena framework is a structured system for conducting red-team assessments against AI language models. The architecture follows a modular, agent-based approach with clear separation of concerns and well-defined data flows.

## Available Diagrams

### [System Overview](system_overview.md)

**Purpose**: Provides a high-level view of the entire system architecture.

**What it Shows**:
- All major components and their relationships
- Data flows between components
- Integration points with external services
- Storage components and persistence mechanisms

**When to Use**: Start here to understand the overall architecture before diving into specific components.

### Agent-Specific Diagrams

#### [Reconnaissance Agent Detail](recon_agent_detail.md)

**Purpose**: Details the internal architecture of the Reconnaissance Agent.

**What it Shows**:
- Internal components of the Reconnaissance Agent
- Data gathering and processing workflows
- Integration with web search and Discord APIs
- Information analysis and report generation process

**When to Use**: When you need to understand how the system gathers information about target models.

#### [Exploit Delivery Agent Detail](exploit_agent_detail.md)

**Purpose**: Details the internal architecture of the Exploit Delivery Agent.

**What it Shows**:
- API-based testing components and workflows
- Browser automation architecture (Playwright and Selenium)
- Decision points for testing method selection
- Result collection and analysis

**When to Use**: When you need to understand how the system executes and evaluates attack prompts.

#### [Evaluation Agent Detail](evaluation_agent_detail.md)

**Purpose**: Details the internal architecture of the Evaluation Agent.

**What it Shows**:
- Data processing components
- Visualization generation engine
- Report creation workflow
- LLM-assisted analysis components

**When to Use**: When you need to understand how the system analyzes results and generates comprehensive reports.

### Process and Interaction Diagrams

#### [Workflow Sequence](workflow_sequence.md)

**Purpose**: Illustrates the end-to-end workflow of a complete red-team assessment.

**What it Shows**:
- Chronological sequence of operations across all agents
- Interactions between components over time
- Parallel and asynchronous operations
- Data exchange between components

**When to Use**: When you need to understand the temporal aspects of the system and how components interact during a complete assessment.

#### [Class Diagram](class_diagram.md)

**Purpose**: Shows the object-oriented structure of the codebase.

**What it Shows**:
- Key classes and their relationships
- Attributes and methods of major classes
- Inheritance hierarchies
- Design patterns used in the implementation

**When to Use**: When you need to understand the code structure from an object-oriented perspective, especially for development or extension.

## How to Use These Diagrams

### Exploring the Architecture

1. **Start with the System Overview** to understand the high-level components and their relationships.
2. **Dive into specific agent diagrams** to understand how each component works internally.
3. **Review the Workflow Sequence** to understand how everything fits together in a complete assessment.
4. **Refer to the Class Diagram** when working with the codebase.

### Common Tasks

- **Adding new capabilities**: Identify the appropriate agent and understand its internal architecture.
- **Troubleshooting issues**: Use the sequence diagram to trace the flow of operations.
- **Extending the framework**: Use the class diagram to understand extension points.
- **Performance optimization**: Identify bottlenecks marked in the diagrams.

## Diagram Notation

All diagrams use Mermaid notation with consistent color coding:

- **Purple**: Agent components
- **Blue**: Utility components and data processors
- **Green**: Data storage components
- **Red**: External services and bottlenecks
- **Yellow**: Asynchronous processes and decision points

Relationships between components are shown with different line styles:

- **Solid lines**: Direct data flow and dependencies
- **Dotted lines**: Service usage and utility connections

## Updating These Diagrams

As the Gray Swan Arena framework evolves, these diagrams should be updated to reflect changes in architecture. When making significant changes to the system:

1. Identify which diagrams are affected
2. Update the relevant diagrams to reflect the new architecture
3. Update this index if new diagrams are added or old ones removed 