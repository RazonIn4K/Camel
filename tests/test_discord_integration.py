from typing import Any, Dict, List, Optional, Tuple, Union
"""Test Discord integration features of Gray Swan Arena."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).parent.parent))

from cybersec_agents.grayswan.utils.discord_utils import DiscordScraper
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Constants
OUTPUT_DIR = Path("tests/output")
TEST_CHANNEL_ID: tuple[Any, ...] = (
    "123456789012345678"  # Replace with actual channel ID for real testing
)


def setup_test_environment():
    """Set up the test environment for Discord tests."""
    # Load environment variables
    load_dotenv()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(name="discord_test", log_level=20)  # INFO level

    # Check for Discord token
    discord_token = os.environ.get("DISCORD_TOKEN")
    if not discord_token:
        print("⚠️ DISCORD_TOKEN environment variable not found.")
        print("Some tests will be skipped or run in simulation mode.")

    # Get test channel ID from environment or use default
    channel_id = os.environ.get("TEST_DISCORD_CHANNEL_ID", TEST_CHANNEL_ID)

    print(f"Test environment set up.")
    print(f"Output will be saved to {OUTPUT_DIR.absolute()}")
    print(
        f"Discord integration: {'✅ Enabled' if discord_token else '⚠️ Simulation mode'}"
    )

    return {"logger": logger, "discord_token": discord_token, "channel_id": channel_id}


def test_discord_scraper_initialization(env):
    """Test initializing the Discord scraper."""
    print("\n=== Testing Discord Scraper Initialization ===")

    logger = env["logger"]
    discord_token = env["discord_token"]

    try:
        if not discord_token:
            print("Running in simulation mode (no Discord token available)")
            # Create with a dummy token
            scraper = DiscordScraper(token="simulation_mode", logger=logger)
            print("✅ Discord scraper initialized in simulation mode")
            return True, scraper

        # Create with actual token
        scraper = DiscordScraper(token=discord_token, logger=logger)
        print("✅ Discord scraper initialized with actual token")
        return True, scraper

    except Exception as e:
        print(f"❌ Error initializing Discord scraper: {e}")
        return False, None


def test_discord_message_collection(env, scraper):
    """Test collecting messages from a Discord channel."""
    print("\n=== Testing Discord Message Collection ===")

    if not scraper:
        print("❌ Cannot test message collection without initialized scraper")
        return False

    channel_id = env["channel_id"]
    discord_token = env["discord_token"]

    try:
        if not discord_token:
            # Simulation mode - create fake messages
            print("Running in simulation mode (no Discord token available)")
            messages: list[Any] = [
                {
                    "id": "1",
                    "content": "This is a simulated message for testing",
                    "author": {"username": "test_user_1", "id": "11111"},
                    "timestamp": datetime.now().isoformat(),
                    "attachments": [],
                },
                {
                    "id": "2",
                    "content": "Here's another simulated message",
                    "author": {"username": "test_user_2", "id": "22222"},
                    "timestamp": datetime.now().isoformat(),
                    "attachments": [],
                },
            ]
        else:
            # Real mode - fetch actual messages
            print(f"Fetching messages from channel {channel_id}")
            messages = scraper.get_channel_messages(channel_id, limit=10)

        # Save messages to file
        output_file = OUTPUT_DIR / "discord_messages.json"
        with open(output_file, "w") as f:
            # Convert to serializable format if needed
            serializable_messages: list[Any] = []
            for msg in messages:
                if isinstance(msg, dict):
                    serializable_messages.append(msg)
                else:
                    # If object has to_dict method (e.g., discord.py Message object)
                    if hasattr(msg, "to_dict"):
                        serializable_messages.append(msg.to_dict())
                    else:
                        # Basic serialization for other objects
                        serializable_msg: dict[str, Any] = {
                            "id": getattr(msg, "id", "unknown"),
                            "content": getattr(msg, "content", ""),
                            "author": {
                                "username": getattr(
                                    getattr(msg, "author", None), "name", "unknown"
                                ),
                                "id": getattr(
                                    getattr(msg, "author", None), "id", "unknown"
                                ),
                            },
                            "timestamp": str(
                                getattr(msg, "created_at", datetime.now())
                            ),
                        }
                        serializable_messages.append(serializable_msg)

            json.dump(serializable_messages, f, indent=2, default=str)

        print(f"✅ Retrieved {len(messages)} messages and saved to {output_file}")
        print(
            f"First message preview: {messages[0]['content'][:50]}..."
            if messages
            else "No messages found"
        )

        return True

    except Exception as e:
        print(f"❌ Error collecting Discord messages: {e}")
        return False


def test_discord_data_analysis(env):
    """Test analyzing Discord message data."""
    print("\n=== Testing Discord Data Analysis ===")

    try:
        # Load previously saved messages
        messages_file = OUTPUT_DIR / "discord_messages.json"

        if not messages_file.exists():
            print(
                f"⚠️ No message data file found at {messages_file}. Can't perform analysis."
            )
            return False

        with open(messages_file, "r") as f:
            messages = json.load(f)

        if not messages:
            print("⚠️ No messages available for analysis")
            return False

        # Simple analysis
        user_message_counts: dict[str, Any] = {}
        word_counts: dict[str, Any] = {}
        total_words: int = 0

        for msg in messages:
            # Count messages per user
            author = msg.get("author", {}).get("username", "unknown")
            user_message_counts[author] = user_message_counts.get(author, 0) + 1

            # Count words
            content = msg.get("content", "")
            words = content.split()
            total_words += len(words)

            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Save analysis to file
        analysis: dict[str, Any] = {
            "total_messages": len(messages),
            "total_words": total_words,
            "avg_words_per_message": total_words / len(messages) if messages else 0,
            "user_message_counts": user_message_counts,
            "most_common_words": sorted(
                word_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

        analysis_file = OUTPUT_DIR / "discord_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"✅ Analysis completed and saved to {analysis_file}")
        print(f"Found {len(messages)} messages from {len(user_message_counts)} users")
        print(f"Average words per message: {analysis['avg_words_per_message']:.2f}")

        return True

    except Exception as e:
        print(f"❌ Error analyzing Discord data: {e}")
        return False


def run_discord_tests():
    """Run all Discord integration tests."""
    print("=" * 60)
    print("GRAY SWAN ARENA: DISCORD INTEGRATION TESTS")
    print("=" * 60)

    # Set up test environment
    env = setup_test_environment()

    # Test Discord scraper initialization
    init_success, scraper = test_discord_scraper_initialization(env)

    # Test message collection
    if init_success:
        collection_success = test_discord_message_collection(env, scraper)
    else:
        collection_success: bool = False
        print("⚠️ Skipping message collection test due to initialization failure")

    # Test data analysis
    analysis_success = test_discord_data_analysis(env)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(
        f"Discord Scraper Initialization: {'✅ PASSED' if init_success else '❌ FAILED'}"
    )
    print(
        f"Discord Message Collection: {'✅ PASSED' if collection_success else '❌ FAILED'}"
    )
    print(f"Discord Data Analysis: {'✅ PASSED' if analysis_success else '❌ FAILED'}")

    if init_success and collection_success and analysis_success:
        print("\n✅ DISCORD INTEGRATION TESTS PASSED!")
        return True
    else:
        print("\n⚠️ DISCORD INTEGRATION TESTS PARTIALLY PASSED OR FAILED")
        if not env["discord_token"]:
            print(
                "Note: Some failures are expected when running in simulation mode without a Discord token."
            )
        return False


if __name__ == "__main__":
    run_discord_tests()
