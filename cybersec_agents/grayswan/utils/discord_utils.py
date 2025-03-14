"""Discord utilities for Gray Swan Arena."""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .logging_utils import setup_logging

# Set up logger
logger = setup_logging("discord_utils")

# Try importing discord.py
try:
    import discord

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning("Discord.py not installed. Install with 'pip install discord.py'")


class DiscordScraper:
    """Utility for scraping messages from Discord channels."""

    def __init__(self):
        """Initialize the Discord scraper."""
        load_dotenv()
        self.token = os.getenv("DISCORD_BOT_TOKEN")
        self.available = DISCORD_AVAILABLE and self.token is not None

        # Try to parse channel IDs from environment
        self.default_channel_ids = []
        channel_ids_str = os.getenv("DISCORD_CHANNEL_IDS", "")
        if channel_ids_str:
            try:
                self.default_channel_ids = [
                    channel_id.strip()
                    for channel_id in channel_ids_str.split(",")
                    if channel_id.strip()
                ]
                logger.info(
                    f"Loaded {len(self.default_channel_ids)} default channel IDs from environment"
                )
            except Exception as e:
                logger.warning(f"Error parsing DISCORD_CHANNEL_IDS: {e}")

        if not DISCORD_AVAILABLE:
            logger.warning(
                "Discord functionality unavailable: discord.py not installed"
            )
        elif not self.token:
            logger.warning(
                "Discord functionality unavailable: DISCORD_BOT_TOKEN not set in .env"
            )
        else:
            logger.info("Discord scraper initialized successfully")

    async def _search_messages_async(
        self, query: str, channel_ids: Optional[List[str]] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Asynchronously search for messages in Discord channels that match the query.

        Args:
            query: The search query
            channel_ids: List of channel IDs to search. If None, searches all accessible channels or default channels.
            limit: Maximum number of messages to retrieve per channel

        Returns:
            List of matching messages with metadata
        """
        if not self.available:
            logger.error("Discord functionality not available")
            return []

        # Use default channel IDs if none provided
        channel_ids = channel_ids or self.default_channel_ids

        # Create Discord client with intents
        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        # Store matching messages
        matching_messages: List[Dict[str, Any]] = []
        search_complete = asyncio.Event()

        @client.event
        async def on_ready():
            logger.info(f"Logged in as {client.user}")
            try:
                # If no channel IDs provided, search all accessible channels
                if not channel_ids:
                    channels_to_search = [
                        channel
                        for guild in client.guilds
                        for channel in guild.text_channels
                        if channel.permissions_for(guild.me).read_messages
                    ]
                else:
                    channels_to_search = []
                    for channel_id in channel_ids:
                        try:
                            channel = client.get_channel(int(channel_id))
                            if channel:
                                channels_to_search.append(channel)
                            else:
                                # Try to fetch channel if not immediately accessible
                                try:
                                    channel = await client.fetch_channel(
                                        int(channel_id)
                                    )
                                    channels_to_search.append(channel)
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to fetch channel {channel_id}: {e}"
                                    )
                        except ValueError:
                            logger.warning(f"Invalid channel ID: {channel_id}")

                logger.info(
                    f"Searching {len(channels_to_search)} channels for '{query}'"
                )

                # Search for messages in each channel
                for channel in channels_to_search:
                    logger.info(f"Searching in channel: {channel.name}")
                    try:
                        async for message in channel.history(limit=limit):
                            if query.lower() in message.content.lower():
                                # Extract attachments
                                attachments = []
                                for attachment in message.attachments:
                                    attachments.append(
                                        {
                                            "filename": attachment.filename,
                                            "url": attachment.url,
                                            "content_type": attachment.content_type,
                                            "size": attachment.size,
                                        }
                                    )

                                # Extract mentioned users
                                mentions = [user.name for user in message.mentions]

                                matching_messages.append(
                                    {
                                        "content": message.content,
                                        "author": str(message.author),
                                        "author_id": str(message.author.id),
                                        "channel": channel.name,
                                        "channel_id": str(channel.id),
                                        "guild": (
                                            channel.guild.name
                                            if hasattr(channel, "guild")
                                            else "Unknown"
                                        ),
                                        "timestamp": message.created_at.isoformat(),
                                        "url": message.jump_url,
                                        "attachments": attachments,
                                        "mentions": mentions,
                                        "edited": message.edited_at is not None,
                                    }
                                )
                    except Exception as e:
                        logger.error(
                            f"Error searching channel {channel.name}: {str(e)}"
                        )

                logger.info(f"Found {len(matching_messages)} matching messages")
            except Exception as e:
                logger.error(f"Error during Discord search: {str(e)}")
            finally:
                search_complete.set()

        # Run the client
        try:
            # Start the client
            client_task = asyncio.create_task(client.start(self.token))

            # Wait for search to complete with timeout
            timeout_seconds = 120  # 2 minutes timeout
            try:
                await asyncio.wait_for(search_complete.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Discord search timed out after {timeout_seconds} seconds"
                )

            # Close the client
            await client.close()

            # Cancel the client task if it's still running
            if not client_task.done():
                client_task.cancel()
                try:
                    await client_task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            logger.error(f"Failed to start Discord client: {str(e)}")

        return matching_messages

    def search(
        self, query: str, channel_ids: Optional[List[str]] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for the async Discord search function.

        Args:
            query: The search query
            channel_ids: List of channel IDs to search
            limit: Maximum messages per channel

        Returns:
            List of matching messages
        """
        if not self.available:
            logger.warning("Discord search unavailable")
            return []

        # Use default channel IDs if none provided
        channel_ids = channel_ids or self.default_channel_ids

        try:
            # Create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._search_messages_async(query, channel_ids, limit)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in Discord search: {str(e)}")
            return []

    def format_results(
        self, results: List[Dict[str, Any]], include_metadata: bool = False
    ) -> str:
        """Format search results into a readable string.

        Args:
            results: List of message dictionaries
            include_metadata: Whether to include additional metadata in the output

        Returns:
            Formatted results string
        """
        if not results:
            return "No Discord messages found matching the query."

        output = f"Found {len(results)} relevant Discord messages:\n\n"

        # Include up to 10 messages in the output
        for i, msg in enumerate(results[:10], 1):
            output += f"{i}. From {msg['author']} in {msg['channel']} at {msg['timestamp']}:\n"
            output += f"   {msg['content']}\n"

            if include_metadata:
                output += f"   URL: {msg['url']}\n"

                if msg["attachments"]:
                    output += "   Attachments:\n"
                    for attachment in msg["attachments"]:
                        output += f"     - {attachment['filename']} ({attachment['content_type']})\n"

                if msg["mentions"]:
                    output += f"   Mentions: {', '.join(msg['mentions'])}\n"

            output += "\n"

        if len(results) > 10:
            output += f"... and {len(results) - 10} more messages."

        return output

    def save_results(
        self, results: List[Dict[str, Any]], filename: str
    ) -> Optional[str]:
        """Save search results to a JSON file.

        Args:
            results: List of message dictionaries
            filename: Name of the file to save

        Returns:
            Path to the saved file, or None if saving failed
        """
        if not results:
            logger.warning("No results to save")
            return None

        try:
            # Create the data directory if it doesn't exist
            data_dir = os.path.join("data", "discord_searches")
            os.makedirs(data_dir, exist_ok=True)

            filepath = os.path.join(data_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(results)} Discord messages to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save Discord search results: {e}")
            return None


# Example usage
if __name__ == "__main__":
    discord_scraper = DiscordScraper()

    if discord_scraper.available:
        results = discord_scraper.search("jailbreak technique", limit=20)
        print(discord_scraper.format_results(results, include_metadata=True))

        # Save results
        discord_scraper.save_results(results, "jailbreak_search_results.json")
    else:
        print(
            "Discord scraper not available. Check your DISCORD_BOT_TOKEN and discord.py installation."
        )
