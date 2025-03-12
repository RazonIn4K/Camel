from cybersec_agents.analyzers import CodebaseAnalyzerAgent

# Initialize the agent
analyzer = CodebaseAnalyzerAgent(project_root="./", memory_db_path=".agent_memory.db")

# Ask questions about the codebase
response = analyzer.answer_question(
    "How does the wireless security scanning functionality work?"
)
print(response)

# Get implementation suggestions
new_feature = analyzer.suggest_implementation(
    "Add support for Bluetooth device scanning"
)
print(new_feature)

# Understand code flow
flow = analyzer.explain_code_flow("wireless_scanner.py")
print(flow)
