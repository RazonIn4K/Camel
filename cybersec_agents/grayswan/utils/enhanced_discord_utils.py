"""
Enhanced Discord utilities for Gray Swan Arena.

This module provides enhanced Discord integration capabilities with filtering,
sentiment analysis, and advanced search features.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dotenv import load_dotenv

from .discord_utils import DiscordScraper
from .logging_utils import setup_logging

# Set up logger
logger = setup_logging("enhanced_discord_utils")

from collections import Counter

# Import NLTK utilities
from cybersec_agents.utils.nltk_utils import (
    analyze_sentiment,
    get_sentiment_analyzer,
    initialize_nltk,
)

# Initialize NLTK and check if NLP capabilities are available
try:
    # Initialize required NLTK data packages
    nltk_status = initialize_nltk(["vader_lexicon", "punkt", "stopwords"], quiet=True)

    # Import NLTK modules if initialization was successful
    if all(nltk_status.values()):
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        # Check if sentiment analyzer is available
        sia = get_sentiment_analyzer()
        NLP_AVAILABLE = sia is not None

        if NLP_AVAILABLE:
            logger.info("NLP capabilities available for Discord analysis")
        else:
            logger.warning(
                "Sentiment analyzer not available. Some features will be limited."
            )
    else:
        NLP_AVAILABLE: bool = False
        missing_packages: list[Any] = [pkg for pkg, status in nltk_status.items() if not status]
        logger.warning(
            f"NLTK initialization incomplete. Missing packages: {', '.join(missing_packages)}"
        )
except ImportError:
    NLP_AVAILABLE: bool = False
    logger.warning("NLP capabilities not available. Install with 'pip install nltk'")


class EnhancedDiscordScraper(DiscordScraper):
    """
    Enhanced utility for scraping and analyzing messages from Discord channels.

    This class extends the base DiscordScraper with additional features:
    - Advanced filtering (date, author, content)
    - Sentiment analysis
    - Topic extraction
    - Message clustering
    - Reaction analysis
    - Timeout handling
    """

    def __init__(self, timeout: int = 120):
        """
        Initialize the Enhanced Discord scraper.

        Args:
            timeout: Timeout in seconds for Discord operations (default: 120)
        """
        super().__init__()
        self.timeout = timeout

        # Initialize NLP components if available
        self.nlp_available = NLP_AVAILABLE
        if self.nlp_available:
            self.sia = get_sentiment_analyzer()
            if self.sia:
                logger.info("Sentiment analysis initialized")
            else:
                self.nlp_available = False
                logger.warning("Failed to initialize sentiment analyzer")

        # Cache for search results
        self.result_cache = {}
        self.cache_expiry = 3600  # Cache results for 1 hour

    def search_with_filters(
        self,
        query: str,
        channel_ids: Optional[List[str]] = None,
        limit: int = 100,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        author_filter: Optional[str] = None,
        content_filter: Optional[str] = None,
        has_attachments: Optional[bool] = None,
        has_mentions: Optional[bool] = None,
        sentiment_filter: Optional[str] = None,
        include_sentiment: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for messages with advanced filtering options.

        Args:
            query: The search query
            channel_ids: List of channel IDs to search
            limit: Maximum messages per channel
            min_date: Minimum date in ISO format (YYYY-MM-DD)
            max_date: Maximum date in ISO format (YYYY-MM-DD)
            author_filter: Filter by author name or ID (case-insensitive substring match)
            content_filter: Additional content filter as regex pattern
            has_attachments: Filter for messages with attachments
            has_mentions: Filter for messages with mentions
            sentiment_filter: Filter by sentiment (positive, negative, neutral)
            include_sentiment: Whether to include sentiment analysis in results

        Returns:
            List of matching messages with metadata
        """
        # Check if Discord is available
        if not self.available:
            logger.warning("Discord search unavailable")
            return []

        # Generate cache key
        cache_key = f"{query}_{channel_ids}_{limit}_{min_date}_{max_date}_{author_filter}_{content_filter}_{has_attachments}_{has_mentions}"

        # Check cache
        if cache_key in self.result_cache:
            cache_entry = self.result_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_expiry:
                logger.info(f"Using cached results for query: {query}")
                results: list[Any] = cache_entry["results"]

                # Apply sentiment filter if needed (this is done here because sentiment might not be in cached results)
                if sentiment_filter and self.nlp_available:
                    results: list[Any] = self._filter_by_sentiment(
                        results, sentiment_filter, include_sentiment
                    )

                return results

        # Get base results
        results: list[Any] = self.search(query, channel_ids, limit)

        # Apply filters
        if min_date:
            try:
                min_datetime = datetime.fromisoformat(min_date)
                results: list[Any] = [
                    r
                    for r in results
                    if datetime.fromisoformat(r["timestamp"].split("T")[0])
                    >= min_datetime
                ]
            except ValueError:
                logger.warning(
                    f"Invalid min_date format: {min_date}. Expected YYYY-MM-DD"
                )

        if max_date:
            try:
                max_datetime = datetime.fromisoformat(max_date)
                results: list[Any] = [
                    r
                    for r in results
                    if datetime.fromisoformat(r["timestamp"].split("T")[0])
                    <= max_datetime
                ]
            except ValueError:
                logger.warning(
                    f"Invalid max_date format: {max_date}. Expected YYYY-MM-DD"
                )

        if author_filter:
            author_filter_lower = author_filter.lower()
            results: list[Any] = [
                r
                for r in results
                if author_filter_lower in r["author"].lower()
                or author_filter_lower in r["author_id"].lower()
            ]

        if content_filter:
            try:
                pattern = re.compile(content_filter, re.IGNORECASE)
                results: list[Any] = [r for r in results if pattern.search(r["content"])]
            except re.error:
                logger.warning(f"Invalid content_filter regex: {content_filter}")

        if has_attachments is not None:
            results: list[Any] = [
                r for r in results if (len(r["attachments"]) > 0) == has_attachments
            ]

        if has_mentions is not None:
            results: list[Any] = [r for r in results if (len(r["mentions"]) > 0) == has_mentions]

        # Apply sentiment analysis if needed
        if (sentiment_filter or include_sentiment) and self.nlp_available:
            results: list[Any] = self._filter_by_sentiment(
                results, sentiment_filter, include_sentiment
            )

        # Cache results
        self.result_cache[cache_key] = {"results": results, "timestamp": time.time()}

        return results

    def _filter_by_sentiment(
        self,
        results: List[Dict[str, Any]],
        sentiment_filter: Optional[str],
        include_sentiment: bool,
    ) -> List[Dict[str, Any]]:
        """
        Apply sentiment analysis and filtering to results.

        Args:
            results: List of message dictionaries
            sentiment_filter: Filter by sentiment (positive, negative, neutral)
            include_sentiment: Whether to include sentiment analysis in results

        Returns:
            Filtered and/or enhanced results
        """
        if not self.nlp_available:
            return results

        filtered_results: list[Any] = []

        for result in results:
            # Skip if already has sentiment analysis
            if "sentiment" in result and not sentiment_filter:
                filtered_results.append(result)
                continue

            # Perform sentiment analysis
            content = result.get("content", "")
            sentiment_scores = self.sia.polarity_scores(content)

            # Determine sentiment classification
            if sentiment_scores["compound"] >= 0.05:
                classification: str = "positive"
            elif sentiment_scores["compound"] <= -0.05:
                classification: str = "negative"
            else:
                classification: str = "neutral"

            # Add sentiment to result if requested
            if include_sentiment:
                result["sentiment"] = {
                    "negative": sentiment_scores["neg"],
                    "neutral": sentiment_scores["neu"],
                    "positive": sentiment_scores["pos"],
                    "compound": sentiment_scores["compound"],
                    "classification": classification,
                }

            # Apply sentiment filter if specified
            if sentiment_filter:
                if classification == sentiment_filter.lower():
                    filtered_results.append(result)
            else:
                filtered_results.append(result)

        return filtered_results

    def get_trending_topics(
        self,
        results: List[Dict[str, Any]],
        top_n: int = 10,
        exclude_words: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract trending topics from search results.

        Args:
            results: List of message dictionaries
            top_n: Number of top topics to return
            exclude_words: Additional words to exclude

        Returns:
            List of trending topics with counts
        """
        if not results or not self.nlp_available:
            return []

        # Extract all content
        all_content: str = " ".join([r.get("content", "") for r in results])

        # Tokenize and clean
        tokens = word_tokenize(all_content.lower())

        # Get stop words and add custom exclusions
        stop_words = set(stopwords.words("english"))
        if exclude_words:
            stop_words.update(set(w.lower() for w in exclude_words))

        # Filter tokens
        filtered_tokens: list[Any] = [
            w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 3
        ]

        # Get frequency
        word_freq = Counter(filtered_tokens)

        # Return top N topics
        return [
            {"word": word, "count": count}
            for word, count in word_freq.most_common(top_n)
        ]

    def get_user_activity(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze user activity from search results.

        Args:
            results: List of message dictionaries

        Returns:
            Dictionary mapping user names to message counts
        """
        user_counts: dict[str, int] = {}

        for result in results:
            author = result.get("author", "Unknown")
            user_counts[author] = user_counts.get(author, 0) + 1

        return dict(sorted(user_counts.items(), key=lambda x: x[1], reverse=True))

    def get_channel_activity(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze channel activity from search results.

        Args:
            results: List of message dictionaries

        Returns:
            Dictionary mapping channel names to message counts
        """
        channel_counts: dict[str, int] = {}

        for result in results:
            channel = result.get("channel", "Unknown")
            channel_counts[channel] = channel_counts.get(channel, 0) + 1

        return dict(sorted(channel_counts.items(), key=lambda x: x[1], reverse=True))

    def get_time_distribution(
        self, results: List[Dict[str, Any]], interval: str = "day"
    ) -> Dict[str, int]:
        """
        Analyze time distribution of messages.

        Args:
            results: List of message dictionaries
            interval: Time interval for grouping (hour, day, weekday, month)

        Returns:
            Dictionary mapping time intervals to message counts
        """
        time_counts: dict[str, int] = {}

        for result in results:
            try:
                timestamp = result.get("timestamp", "")
                if not timestamp:
                    continue

                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                if interval == "hour":
                    key = dt.strftime("%H:00")
                elif interval == "day":
                    key = dt.strftime("%Y-%m-%d")
                elif interval == "weekday":
                    key = dt.strftime("%A")
                elif interval == "month":
                    key = dt.strftime("%B %Y")
                else:
                    key = dt.strftime("%Y-%m-%d")

                time_counts[key] = time_counts.get(key, 0) + 1
            except ValueError:
                continue

        return dict(sorted(time_counts.items()))

    def search_by_timeframe(
        self,
        query: str,
        timeframe: str,
        channel_ids: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search for messages within a specific timeframe.

        Args:
            query: The search query
            timeframe: Timeframe specifier (today, yesterday, this_week, last_week, this_month, last_month)
            channel_ids: List of channel IDs to search
            limit: Maximum messages per channel

        Returns:
            List of matching messages with metadata
        """
        now = datetime.now()

        if timeframe == "today":
            min_date = now.strftime("%Y-%m-%d")
            max_date = now.strftime("%Y-%m-%d")
        elif timeframe == "yesterday":
            yesterday = now - timedelta(days=1)
            min_date = yesterday.strftime("%Y-%m-%d")
            max_date = yesterday.strftime("%Y-%m-%d")
        elif timeframe == "this_week":
            # Start of week (Monday)
            start_of_week = now - timedelta(days=now.weekday())
            min_date = start_of_week.strftime("%Y-%m-%d")
            max_date = now.strftime("%Y-%m-%d")
        elif timeframe == "last_week":
            # Start of last week
            start_of_last_week = now - timedelta(days=now.weekday() + 7)
            end_of_last_week = start_of_last_week + timedelta(days=6)
            min_date = start_of_last_week.strftime("%Y-%m-%d")
            max_date = end_of_last_week.strftime("%Y-%m-%d")
        elif timeframe == "this_month":
            min_date = now.strftime("%Y-%m-01")
            max_date = now.strftime("%Y-%m-%d")
        elif timeframe == "last_month":
            last_month = now.replace(day=1) - timedelta(days=1)
            start_of_last_month = last_month.replace(day=1)
            min_date = start_of_last_month.strftime("%Y-%m-%d")
            max_date = last_month.strftime("%Y-%m-%d")
        else:
            logger.warning(f"Invalid timeframe: {timeframe}")
            return []

        return self.search_with_filters(
            query=query,
            channel_ids=channel_ids,
            limit=limit,
            min_date=min_date,
            max_date=max_date,
        )

    def load_results(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load search results from a JSON file.

        Args:
            filename: Name of the file to load

        Returns:
            List of message dictionaries
        """
        try:
            # Determine file path
            data_dir = os.path.join("data", "discord_searches")
            filepath = os.path.join(data_dir, filename)

            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                return []

            # Load the file
            with open(filepath, "r", encoding="utf-8") as f:
                results: list[Any] = json.load(f)

            logger.info(f"Loaded {len(results)} Discord messages from {filepath}")
            return results
        except Exception as e:
            logger.error(f"Failed to load Discord search results: {e}")
            return []

    def format_advanced_results(
        self,
        results: List[Dict[str, Any]],
        include_metadata: bool = False,
        include_stats: bool = False,
        max_messages: int = 10,
    ) -> str:
        """
        Format search results into a readable string with advanced options.

        Args:
            results: List of message dictionaries
            include_metadata: Whether to include additional metadata in the output
            include_stats: Whether to include statistics about the results
            max_messages: Maximum number of messages to include in the output

        Returns:
            Formatted results string
        """
        if not results:
            return "No Discord messages found matching the query."

        output = f"Found {len(results)} relevant Discord messages:\n\n"

        # Include statistics if requested
        if include_stats:
            # User activity
            user_activity = self.get_user_activity(results)
            top_users = list(user_activity.items())[:5]

            output += "Top Contributors:\n"
            for user, count in top_users:
                output += f"- {user}: {count} messages\n"

            # Channel activity
            channel_activity = self.get_channel_activity(results)
            top_channels = list(channel_activity.items())[:5]

            output += "\nTop Channels:\n"
            for channel, count in top_channels:
                output += f"- {channel}: {count} messages\n"

            # Trending topics
            if self.nlp_available:
                trending_topics = self.get_trending_topics(results, top_n=5)

                output += "\nTrending Topics:\n"
                for topic in trending_topics:
                    output += f"- {topic['word']}: {topic['count']} occurrences\n"

            output += "\n"

        # Include messages
        for i, msg in enumerate(results[:max_messages], 1):
            output += f"{i}. From {msg['author']} in {msg['channel']} at {msg['timestamp']}:\n"
            output += f"   {msg['content']}\n"

            if include_metadata:
                output += f"   URL: {msg['url']}\n"

                if msg.get("sentiment"):
                    sentiment = msg["sentiment"]
                    output += f"   Sentiment: {sentiment['classification']} (score: {sentiment['compound']:.2f})\n"

                if msg["attachments"]:
                    output += "   Attachments:\n"
                    for attachment in msg["attachments"]:
                        output += f"     - {attachment['filename']} ({attachment['content_type']})\n"

                if msg["mentions"]:
                    output += f"   Mentions: {', '.join(msg['mentions'])}\n"

            output += "\n"

        if len(results) > max_messages:
            output += f"... and {len(results) - max_messages} more messages."

        return output

    def clear_cache(self) -> None:
        """Clear the search results cache."""
        self.result_cache = {}
        logger.info("Search results cache cleared")


# Example usage
if __name__ == "__main__":
    discord_scraper = EnhancedDiscordScraper()

    if discord_scraper.available:
        # Search with filters
        results: list[Any] = discord_scraper.search_with_filters(
            query="jailbreak technique",
            limit=50,
            min_date="2023-01-01",
            max_date="2023-12-31",
            include_sentiment=True,
        )

        # Print formatted results
        print(
            discord_scraper.format_advanced_results(
                results, include_metadata=True, include_stats=True
            )
        )

        # Get trending topics
        trending_topics = discord_scraper.get_trending_topics(results)
        print("\nTrending Topics:")
        for topic in trending_topics:
            print(f"- {topic['word']}: {topic['count']} occurrences")

        # Save results
        discord_scraper.save_results(results, "enhanced_jailbreak_search_results.json")
    else:
        print(
            "Discord scraper not available. Check your DISCORD_BOT_TOKEN and discord.py installation."
        )
