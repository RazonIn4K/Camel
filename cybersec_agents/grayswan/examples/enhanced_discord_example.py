"""
Example script demonstrating the Enhanced Discord Integration.

This script shows how to use the Enhanced Discord Integration to search for messages,
analyze sentiment, extract trending topics, and more.
"""

import os
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from cybersec_agents.grayswan.utils.enhanced_discord_utils import EnhancedDiscordScraper
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("discord_example")


def basic_search_example():
    """
    Demonstrate basic search functionality.
    """
    print("\n=== Basic Search Example ===\n")
    
    # Create Discord scraper
    scraper = EnhancedDiscordScraper()
    
    if not scraper.available:
        print("Discord scraper not available. Check your DISCORD_BOT_TOKEN and discord.py installation.")
        return
    
    # Perform a basic search
    query: str = "jailbreak technique"
    print(f"Searching for: '{query}'")
    
    results: list[Any] = scraper.search(query, limit=20)
    
    # Print results
    print(f"Found {len(results)} messages")
    print(scraper.format_results(results, include_metadata=True))
    
    # Save results
    if results:
        filename: str = "basic_search_results.json"
        filepath = scraper.save_results(results, filename)
        if filepath:
            print(f"Results saved to: {filepath}")


def advanced_search_example():
    """
    Demonstrate advanced search with filters.
    """
    print("\n=== Advanced Search Example ===\n")
    
    # Create Discord scraper
    scraper = EnhancedDiscordScraper()
    
    if not scraper.available:
        print("Discord scraper not available. Check your DISCORD_BOT_TOKEN and discord.py installation.")
        return
    
    # Perform an advanced search with filters
    query: str = "AI safety"
    print(f"Searching for: '{query}' with filters")
    
    # Get current date for date range
    today = datetime.now().strftime("%Y-%m-%d")
    one_month_ago: tuple[Any, ...] = (datetime.now().replace(day=1) - datetime.timedelta(days=1)).replace(day=1).strftime("%Y-%m-%d")
    
    results: list[Any] = scraper.search_with_filters(
        query=query,
        limit=50,
        min_date=one_month_ago,
        max_date=today,
        content_filter=r"(safety|alignment|ethics)",
        has_attachments=True,
        include_sentiment=True
    )
    
    # Print results
    print(f"Found {len(results)} messages matching filters")
    print(scraper.format_advanced_results(
        results,
        include_metadata=True,
        include_stats=True,
        max_messages=5
    ))
    
    # Save results
    if results:
        filename: str = "advanced_search_results.json"
        filepath = scraper.save_results(results, filename)
        if filepath:
            print(f"Results saved to: {filepath}")


def sentiment_analysis_example():
    """
    Demonstrate sentiment analysis capabilities.
    """
    print("\n=== Sentiment Analysis Example ===\n")
    
    # Create Discord scraper
    scraper = EnhancedDiscordScraper()
    
    if not scraper.available:
        print("Discord scraper not available. Check your DISCORD_BOT_TOKEN and discord.py installation.")
        return
    
    if not scraper.nlp_available:
        print("NLP capabilities not available. Install with 'pip install nltk'")
        return
    
    # Search for messages with sentiment analysis
    query: str = "language model"
    print(f"Analyzing sentiment for messages about: '{query}'")
    
    # Get messages with sentiment analysis
    results: list[Any] = scraper.search_with_filters(
        query=query,
        limit=30,
        include_sentiment=True
    )
    
    if not results:
        print("No results found")
        return
    
    # Count sentiment categories
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for result in results:
        if "sentiment" in result:
            classification = result["sentiment"]["classification"]
            sentiment_counts[classification] += 1
    
    # Print sentiment distribution
    print("\nSentiment Distribution:")
    total: int = sum(sentiment_counts.values())
    for sentiment, count in sentiment_counts.items():
        percentage: tuple[Any, ...] = (count / total) * 100 if total > 0 else 0
        print(f"- {sentiment.capitalize()}: {count} messages ({percentage:.1f}%)")
    
    # Print examples of each sentiment
    for sentiment in ["positive", "negative", "neutral"]:
        print(f"\n{sentiment.capitalize()} Message Example:")
        for result in results:
            if "sentiment" in result and result["sentiment"]["classification"] == sentiment:
                print(f"From {result['author']} (score: {result['sentiment']['compound']:.2f}):")
                print(f"  {result['content'][:150]}...")
                break


def trending_topics_example():
    """
    Demonstrate trending topics extraction.
    """
    print("\n=== Trending Topics Example ===\n")
    
    # Create Discord scraper
    scraper = EnhancedDiscordScraper()
    
    if not scraper.available:
        print("Discord scraper not available. Check your DISCORD_BOT_TOKEN and discord.py installation.")
        return
    
    if not scraper.nlp_available:
        print("NLP capabilities not available. Install with 'pip install nltk'")
        return
    
    # Search for messages
    query: str = "AI"
    print(f"Extracting trending topics from messages about: '{query}'")
    
    # Get messages
    results: list[Any] = scraper.search(query, limit=100)
    
    if not results:
        print("No results found")
        return
    
    # Extract trending topics
    topics = scraper.get_trending_topics(
        results,
        top_n=15,
        exclude_words=["about", "would", "could", "should", "their", "there", "these", "those"]
    )
    
    # Print trending topics
    print(f"\nTop {len(topics)} Trending Topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic['word']}: {topic['count']} occurrences")


def timeframe_search_example():
    """
    Demonstrate searching by timeframe.
    """
    print("\n=== Timeframe Search Example ===\n")
    
    # Create Discord scraper
    scraper = EnhancedDiscordScraper()
    
    if not scraper.available:
        print("Discord scraper not available. Check your DISCORD_BOT_TOKEN and discord.py installation.")
        return
    
    # Search for messages from different timeframes
    query: str = "AI model"
    timeframes: list[Any] = ["today", "this_week", "last_week", "this_month"]
    
    for timeframe in timeframes:
        print(f"\nSearching for '{query}' from {timeframe}:")
        
        # Search by timeframe
        results: list[Any] = scraper.search_by_timeframe(
            query=query,
            timeframe=timeframe,
            limit=20
        )
        
        # Print results count
        print(f"Found {len(results)} messages")
        
        # Print time distribution
        if results:
            time_dist = scraper.get_time_distribution(results, interval="day")
            print("\nTime Distribution:")
            for date, count in time_dist.items():
                print(f"- {date}: {count} messages")


def user_activity_example():
    """
    Demonstrate user activity analysis.
    """
    print("\n=== User Activity Example ===\n")
    
    # Create Discord scraper
    scraper = EnhancedDiscordScraper()
    
    if not scraper.available:
        print("Discord scraper not available. Check your DISCORD_BOT_TOKEN and discord.py installation.")
        return
    
    # Search for messages
    query: str = "AI"
    print(f"Analyzing user activity for messages about: '{query}'")
    
    # Get messages
    results: list[Any] = scraper.search(query, limit=100)
    
    if not results:
        print("No results found")
        return
    
    # Get user activity
    user_activity = scraper.get_user_activity(results)
    
    # Print top users
    print("\nTop Contributors:")
    for i, (user, count) in enumerate(list(user_activity.items())[:10], 1):
        print(f"{i}. {user}: {count} messages")
    
    # Get channel activity
    channel_activity = scraper.get_channel_activity(results)
    
    # Print top channels
    print("\nTop Channels:")
    for i, (channel, count) in enumerate(list(channel_activity.items())[:5], 1):
        print(f"{i}. {channel}: {count} messages")


def load_and_analyze_example():
    """
    Demonstrate loading saved results and analyzing them.
    """
    print("\n=== Load and Analyze Example ===\n")
    
    # Create Discord scraper
    scraper = EnhancedDiscordScraper()
    
    # Try to load results from a file
    filename: str = "basic_search_results.json"
    print(f"Loading results from: {filename}")
    
    results: list[Any] = scraper.load_results(filename)
    
    if not results:
        print(f"No results found in {filename}")
        print("Run the basic_search_example() first to generate the file")
        return
    
    print(f"Loaded {len(results)} messages")
    
    # Analyze the loaded results
    if scraper.nlp_available:
        # Add sentiment analysis
        results_with_sentiment = scraper.search_with_filters(
            query="",  # Empty query since we're just processing existing results
            include_sentiment=True
        )
        
        # Extract trending topics
        topics = scraper.get_trending_topics(results, top_n=10)
        
        print("\nTrending Topics in Loaded Results:")
        for i, topic in enumerate(topics, 1):
            print(f"{i}. {topic['word']}: {topic['count']} occurrences")
    
    # Get time distribution
    time_dist = scraper.get_time_distribution(results, interval="month")
    
    print("\nTime Distribution by Month:")
    for date, count in time_dist.items():
        print(f"- {date}: {count} messages")


def main():
    """Main function to run all examples."""
    print("Enhanced Discord Integration Examples")
    print("====================================")
    
    # Run examples
    basic_search_example()
    advanced_search_example()
    sentiment_analysis_example()
    trending_topics_example()
    timeframe_search_example()
    user_activity_example()
    load_and_analyze_example()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()