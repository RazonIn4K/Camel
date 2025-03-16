from typing import Any, Dict, List, Optional, Tuple, Union

from setuptools import find_packages, setup

setup(
    name="cybersec_agents",
    version="0.2.1",
    description="A collection of cybersecurity agents, including Gray Swan Arena for AI red-teaming",
    author="David Ortiz",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pyyaml",
        "requests",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "python-dotenv",
        "openai",
        "flask",
        "discord.py",
        "nltk>=3.8.1",  # Required for sentiment analysis in Gray Swan Arena
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-grpc",
        "opentelemetry-exporter-otlp-proto-http",
        "opentelemetry-instrumentation",
        "opentelemetry-instrumentation-requests",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "mypy",
            "pre-commit",
        ],
        "browser": [
            "playwright",
            "selenium",
            "webdriver-manager",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Security",
    ],
)
