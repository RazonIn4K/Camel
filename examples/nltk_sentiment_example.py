#!/usr/bin/env python
"""
NLTK Sentiment Analysis Example

This script demonstrates how to use the NLTK utilities for sentiment analysis
in the Camel project. It shows how to initialize NLTK, analyze sentiment,
and handle potential errors.
"""

import logging
import os
import sys
from typing import Any, Dict, List

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("nltk_example")

# Import NLTK utilities
from cybersec_agents.utils.nltk_utils import (
    analyze_sentiment,
    ensure_vader_lexicon,
    get_nltk_info,
    get_sentiment_analyzer,
    initialize_nltk,
)


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_sentiment_analysis(text: str, sentiment: Dict[str, float]) -> None:
    """Print sentiment analysis results in a formatted way."""
    print(f'Text: "{text}"')
    print(f"Negative: {sentiment.get('neg', 0):.3f}")
    print(f"Neutral:  {sentiment.get('neu', 0):.3f}")
    print(f"Positive: {sentiment.get('pos', 0):.3f}")
    print(f"Compound: {sentiment.get('compound', 0):.3f}")

    # Interpret the sentiment
    compound = sentiment.get("compound", 0)
    if compound >= 0.05:
        print("Interpretation: Positive")
    elif compound <= -0.05:
        print("Interpretation: Negative")
    else:
        print("Interpretation: Neutral")
    print()


def analyze_multiple_texts(texts: List[str]) -> None:
    """Analyze sentiment for multiple texts."""
    for text in texts:
        sentiment = analyze_sentiment(text)
        print_sentiment_analysis(text, sentiment)


def check_nltk_installation() -> None:
    """Check NLTK installation and print information."""
    info = get_nltk_info()

    print(f"NLTK Installed: {info['installed']}")
    if info["installed"]:
        print(f"NLTK Version: {info['version']}")
        print(f"VADER Lexicon Available: {info['vader_lexicon']}")
        print(f"SentimentIntensityAnalyzer Working: {info['sentiment_analyzer']}")
        print("\nNLTK Data Paths:")
        for path in info["data_path"]:
            print(f"  - {path}")

        print("\nPackages:")
        for package, available in info["packages"].items():
            status: str = "Available" if available else "Not Available"
            print(f"  - {package}: {status}")
    else:
        print("NLTK is not installed.")


def initialize_and_test() -> None:
    """Initialize NLTK and test sentiment analysis."""
    # Initialize NLTK
    print("Initializing NLTK...")
    packages: list[Any] = ["vader_lexicon", "punkt", "stopwords"]
    results: list[Any] = initialize_nltk(packages, quiet=False)

    # Check initialization results
    all_success = all(results.values())
    if all_success:
        print("All packages initialized successfully.")
    else:
        print("Some packages failed to initialize:")
        for package, success in results.items():
            status: str = "Success" if success else "Failed"
            print(f"  - {package}: {status}")

    # Ensure VADER lexicon is available
    vader_available = ensure_vader_lexicon()
    if vader_available:
        print("VADER lexicon is available.")
    else:
        print("VADER lexicon is not available. Sentiment analysis may not work.")

    # Get sentiment analyzer
    sia = get_sentiment_analyzer()
    if sia:
        print("SentimentIntensityAnalyzer initialized successfully.")
    else:
        print("Failed to initialize SentimentIntensityAnalyzer.")


def main() -> None:
    """Main function to run the example."""
    print_section("NLTK Sentiment Analysis Example")

    # Check NLTK installation
    print_section("NLTK Installation Information")
    check_nltk_installation()

    # Initialize NLTK
    print_section("NLTK Initialization")
    initialize_and_test()

    # Analyze sentiment of example texts
    print_section("Sentiment Analysis Examples")
    example_texts: list[Any] = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Terrible service.",
        "The product arrived on time and functions as expected.",
        "I'm not sure if I like this or not. It has pros and cons.",
        "The customer service was helpful but the product quality is poor.",
        "Despite some minor issues, I'm generally satisfied with my purchase.",
    ]
    analyze_multiple_texts(example_texts)

    # Analyze custom text
    print_section("Custom Text Analysis")
    try:
        custom_text = input("Enter text to analyze (or press Enter to skip): ")
        if custom_text:
            sentiment = analyze_sentiment(custom_text)
            print_sentiment_analysis(custom_text, sentiment)
    except KeyboardInterrupt:
        print("\nAnalysis cancelled.")

    print_section("Example Complete")


if __name__ == "__main__":
    main()
