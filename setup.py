from setuptools import find_packages, setup

setup(
    name="cybersec_agents",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "typing-extensions>=4.0.0",
        "scapy>=2.5.0",
        "pyyaml>=6.0.2",
        "camel-ai>=0.2.26",
        "openai>=1.3.5",
        "langchain>=0.0.337",
        "google-cloud-secret-manager>=2.16.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "autoflake>=2.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "docformatter>=1.7.0",
            "pre-commit>=3.0.0",
        ],
        "gui": [
            "tkinter",
        ],
    },
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "cybersec-agents=cybersec_agents.cli.main:main",
        ],
    },
)
