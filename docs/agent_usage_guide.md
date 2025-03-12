# Comprehensive Agent Usage Guide

## Available Agents and Their Use Cases

### 1. Document Analysis Agent
```python
from cybersec_agents import DocumentAnalysisAgent

# Initialize agent
doc_agent = DocumentAnalysisAgent()

# Analyze PDF
analysis = doc_agent.analyze_pdf("path/to/document.pdf")

# Analyze source code
code_review = doc_agent.analyze_code("path/to/source.py")

# Analyze multiple files
multi_analysis = doc_agent.analyze_files([
    "docs/spec.pdf",
    "src/main.py",
    "config.yaml"
])
```

Command Line Usage:
```bash
# Analyze single file
cyber-agents analyze-doc --file "path/to/file.pdf"

# Analyze directory
cyber-agents analyze-doc --dir "src/" --type "python"

# Generate summary report
cyber-agents analyze-doc --file "document.pdf" --output "analysis.md"
```

### 2. Prompt Engineering Agent
```python
from cybersec_agents import PromptEngineeringAgent

# Initialize agent
prompt_agent = PromptEngineeringAgent()

# Improve a prompt
better_prompt = prompt_agent.improve_prompt(
    "how does this code work",
    context={
        "language": "python",
        "domain": "web_development"
    }
)

# Generate structured prompts
structured_prompt = prompt_agent.generate_structured_prompt(
    topic="database optimization",
    components=["context", "constraints", "examples"]
)
```

Command Line Usage:
```bash
# Improve prompt
cyber-agents improve-prompt "explain this function"

# Generate domain-specific prompt
cyber-agents generate-prompt --domain "database" --type "analysis"
```

### 3. Code Assistant Agent
```python
from cybersec_agents import CodeAssistantAgent

# Initialize agent
code_agent = CodeAssistantAgent()

# Review code
review = code_agent.review_code(
    file_path="src/main.py",
    focus=["security", "performance"]
)

# Suggest improvements
suggestions = code_agent.suggest_improvements(
    code_snippet="""
    def process_data(data):
        return data.split(',')
    """,
    aspects=["error_handling", "type_hints"]
)

# Generate tests
tests = code_agent.generate_tests(
    file_path="src/utils.py",
    test_framework="pytest"
)
```

Command Line Usage:
```bash
# Review code
cyber-agents review-code --file "src/main.py"

# Generate tests
cyber-agents generate-tests --file "src/utils.py"
```

## GUI Integration Guide

### File Type Support

The GUI supports the following file types:

1. Documents:
   - PDF (.pdf)
   - Word (.doc, .docx)
   - Text (.txt)
   - Markdown (.md)

2. Source Code:
   - Python (.py)
   - JavaScript (.js)
   - TypeScript (.ts)
   - Java (.java)
   - C++ (.cpp)
   - Ruby (.rb)
   - Go (.go)
   - And more...

3. Configuration:
   - YAML (.yaml, .yml)
   - JSON (.json)
   - INI (.ini)
   - TOML (.toml)

4. Data:
   - CSV (.csv)
   - XML (.xml)
   - SQL (.sql)

### GUI Usage Examples

1. Document Analysis:
```python
from cybersec_agents.gui import DocumentViewer

# Initialize viewer
viewer = DocumentViewer()

# Load and analyze document
viewer.load_document("path/to/document.pdf")
analysis = viewer.analyze_current_document()

# Extract specific sections
sections = viewer.extract_sections(
    criteria=["headings", "code_blocks", "tables"]
)
```

2. Code Analysis:
```python
from cybersec_agents.gui import CodeEditor

# Initialize editor
editor = CodeEditor()

# Load and analyze code
editor.load_file("src/main.py")
suggestions = editor.get_improvements()

# Interactive code review
editor.start_review_session(
    focus=["security", "performance"],
    interactive=True
)
```

3. Multi-File Analysis:
```python
from cybersec_agents.gui import ProjectExplorer

# Initialize explorer
explorer = ProjectExplorer()

# Analyze project
explorer.load_project("path/to/project")
analysis = explorer.analyze_project(
    file_types=["python", "javascript"],
    aspects=["security", "quality"]
)
```

### GUI Configuration

```yaml
gui:
  theme: "dark"
  file_types:
    enabled:
      - pdf
      - python
      - javascript
      - markdown
    custom_handlers:
      pdf: "pdf_handler.py"
      python: "python_handler.py"
  
  viewers:
    document:
      enable_syntax_highlight: true
      show_line_numbers: true
    
    code:
      theme: "monokai"
      font_size: 12
      enable_minimap: true
```

### Adding Custom File Type Support

1. Create a custom handler:
```python
from cybersec_agents.gui.handlers import BaseFileHandler

class CustomFileHandler(BaseFileHandler):
    def __init__(self):
        self.supported_extensions = [".custom"]
    
    def read_file(self, file_path: str) -> str:
        # Implement custom file reading logic
        pass
    
    def parse_content(self, content: str) -> Dict:
        # Implement custom parsing logic
        pass
```

2. Register the handler:
```python
from cybersec_agents.gui import GUI
from custom_handler import CustomFileHandler

gui = GUI()
gui.register_handler(".custom", CustomFileHandler())
```

## Integration with Camel AI

The GUI automatically integrates with Camel AI for:
1. Document understanding
2. Code analysis
3. Context management
4. Multi-agent coordination

Example:
```python
from cybersec_agents.gui import GUI
from camel.agents import DocumentAgent

# Initialize GUI with custom Camel agent
gui = GUI()
doc_agent = DocumentAgent(specialization="technical_documents")
gui.register_agent("document_analysis", doc_agent)

# Use in analysis
results = gui.analyze_document(
    "spec.pdf",
    agent="document_analysis"
)
```

## Best Practices

1. File Organization:
   - Group related files
   - Use consistent naming
   - Maintain clear structure

2. Agent Selection:
   - Choose specialized agents for specific tasks
   - Combine agents for complex analysis
   - Use prompt engineering for better results

3. Resource Management:
   - Monitor memory usage
   - Cache frequent operations
   - Clean up temporary files

4. Error Handling:
   - Validate file types
   - Handle parsing errors
   - Provide meaningful feedback