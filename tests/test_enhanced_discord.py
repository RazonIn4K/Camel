"""
Tests for the Enhanced Discord Integration.

This module contains tests for the EnhancedDiscordScraper class.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from cybersec_agents.grayswan.utils.enhanced_discord_utils import EnhancedDiscordScraper


class TestEnhancedDiscordScraper(unittest.TestCase):
    """Tests for the EnhancedDiscordScraper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a patcher for the DiscordScraper.search method
        self.search_patcher = patch('cybersec_agents.grayswan.utils.discord_utils.DiscordScraper.search')
        self.mock_search = self.search_patcher.start()
        
        # Create sample search results
        self.sample_results = [
            {
                "content": "I think GPT-4 is amazing! It's so much better than previous models.",
                "author": "User1",
                "author_id": "123456789",
                "channel": "ai-discussion",
                "channel_id": "987654321",
                "guild": "AI Research",
                "timestamp": "2023-06-15T14:30:00",
                "url": "https://discord.com/channels/123/456/789",
                "attachments": [],
                "mentions": [],
                "edited": False
            },
            {
                "content": "I'm having trouble with the new jailbreak technique. It doesn't seem to work anymore.",
                "author": "User2",
                "author_id": "234567890",
                "channel": "ai-safety",
                "channel_id": "876543210",
                "guild": "AI Research",
                "timestamp": "2023-06-16T10:15:00",
                "url": "https://discord.com/channels/123/456/790",
                "attachments": [
                    {
                        "filename": "screenshot.png",
                        "url": "https://cdn.discordapp.com/attachments/123/456/screenshot.png",
                        "content_type": "image/png",
                        "size": 1024
                    }
                ],
                "mentions": ["User3"],
                "edited": True
            },
            {
                "content": "I hate how these models keep getting more restrictive. It's frustrating!",
                "author": "User3",
                "author_id": "345678901",
                "channel": "ai-discussion",
                "channel_id": "987654321",
                "guild": "AI Research",
                "timestamp": "2023-06-17T09:45:00",
                "url": "https://discord.com/channels/123/456/791",
                "attachments": [],
                "mentions": [],
                "edited": False
            },
            {
                "content": "The alignment techniques seem to be working well. The model is much safer now.",
                "author": "User1",
                "author_id": "123456789",
                "channel": "ai-safety",
                "channel_id": "876543210",
                "guild": "AI Research",
                "timestamp": "2023-06-18T16:20:00",
                "url": "https://discord.com/channels/123/456/792",
                "attachments": [],
                "mentions": ["User2"],
                "edited": False
            },
            {
                "content": "I'm neutral about the new update. It has some good features and some bad ones.",
                "author": "User4",
                "author_id": "456789012",
                "channel": "ai-discussion",
                "channel_id": "987654321",
                "guild": "AI Research",
                "timestamp": "2023-06-19T11:30:00",
                "url": "https://discord.com/channels/123/456/793",
                "attachments": [],
                "mentions": [],
                "edited": False
            }
        ]
        
        # Set up the mock search method to return the sample results
        self.mock_search.return_value = self.sample_results
        
        # Create an EnhancedDiscordScraper instance
        self.scraper = EnhancedDiscordScraper()
        
        # Mock the nlp_available attribute to True
        self.scraper.nlp_available = True
        
        # Create a mock SentimentIntensityAnalyzer
        self.mock_sia = MagicMock()
        self.scraper.sia = self.mock_sia
        
        # Set up the mock SentimentIntensityAnalyzer to return sentiment scores
        self.mock_sia.polarity_scores.side_effect = lambda text: {
            "neg": 0.1 if "trouble" in text or "hate" in text or "frustrating" in text else 0.0,
            "neu": 0.5 if "neutral" in text else 0.3,
            "pos": 0.8 if "amazing" in text or "better" in text or "working well" in text or "safer" in text else 0.0,
            "compound": 0.8 if "amazing" in text or "better" in text or "working well" in text or "safer" in text else
                       -0.6 if "trouble" in text or "hate" in text or "frustrating" in text else 0.0
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patcher
        self.search_patcher.stop()
    
    def test_search_with_filters_no_filters(self):
        """Test search_with_filters with no filters."""
        # Call the method
        results = self.scraper.search_with_filters("test query")
        
        # Assert that the search method was called with the correct arguments
        self.mock_search.assert_called_once_with("test query", None, 100)
        
        # Assert that the results are correct
        self.assertEqual(results, self.sample_results)
    
    def test_search_with_filters_date_filter(self):
        """Test search_with_filters with date filters."""
        # Call the method with date filters
        results = self.scraper.search_with_filters(
            "test query",
            min_date="2023-06-16",
            max_date="2023-06-18"
        )
        
        # Assert that the search method was called
        self.mock_search.assert_called_once()
        
        # Assert that the results are filtered correctly
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["timestamp"], "2023-06-16T10:15:00")
        self.assertEqual(results[1]["timestamp"], "2023-06-17T09:45:00")
        self.assertEqual(results[2]["timestamp"], "2023-06-18T16:20:00")
    
    def test_search_with_filters_author_filter(self):
        """Test search_with_filters with author filter."""
        # Call the method with author filter
        results = self.scraper.search_with_filters(
            "test query",
            author_filter="User1"
        )
        
        # Assert that the search method was called
        self.mock_search.assert_called_once()
        
        # Assert that the results are filtered correctly
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["author"], "User1")
        self.assertEqual(results[1]["author"], "User1")
    
    def test_search_with_filters_content_filter(self):
        """Test search_with_filters with content filter."""
        # Call the method with content filter
        results = self.scraper.search_with_filters(
            "test query",
            content_filter=r"jailbreak|alignment"
        )
        
        # Assert that the search method was called
        self.mock_search.assert_called_once()
        
        # Assert that the results are filtered correctly
        self.assertEqual(len(results), 2)
        self.assertIn("jailbreak", results[0]["content"].lower())
        self.assertIn("alignment", results[1]["content"].lower())
    
    def test_search_with_filters_has_attachments(self):
        """Test search_with_filters with has_attachments filter."""
        # Call the method with has_attachments filter
        results = self.scraper.search_with_filters(
            "test query",
            has_attachments=True
        )
        
        # Assert that the search method was called
        self.mock_search.assert_called_once()
        
        # Assert that the results are filtered correctly
        self.assertEqual(len(results), 1)
        self.assertTrue(len(results[0]["attachments"]) > 0)
    
    def test_search_with_filters_has_mentions(self):
        """Test search_with_filters with has_mentions filter."""
        # Call the method with has_mentions filter
        results = self.scraper.search_with_filters(
            "test query",
            has_mentions=True
        )
        
        # Assert that the search method was called
        self.mock_search.assert_called_once()
        
        # Assert that the results are filtered correctly
        self.assertEqual(len(results), 2)
        self.assertTrue(len(results[0]["mentions"]) > 0)
        self.assertTrue(len(results[1]["mentions"]) > 0)
    
    def test_search_with_filters_sentiment_filter(self):
        """Test search_with_filters with sentiment filter."""
        # Call the method with sentiment filter
        results = self.scraper.search_with_filters(
            "test query",
            sentiment_filter="positive",
            include_sentiment=True
        )
        
        # Assert that the search method was called
        self.mock_search.assert_called_once()
        
        # Assert that the results are filtered correctly
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["sentiment"]["classification"], "positive")
        self.assertEqual(results[1]["sentiment"]["classification"], "positive")
    
    def test_search_with_filters_include_sentiment(self):
        """Test search_with_filters with include_sentiment."""
        # Call the method with include_sentiment
        results = self.scraper.search_with_filters(
            "test query",
            include_sentiment=True
        )
        
        # Assert that the search method was called
        self.mock_search.assert_called_once()
        
        # Assert that sentiment analysis was added to the results
        for result in results:
            self.assertIn("sentiment", result)
            self.assertIn("classification", result["sentiment"])
            self.assertIn("compound", result["sentiment"])
            self.assertIn("negative", result["sentiment"])
            self.assertIn("neutral", result["sentiment"])
            self.assertIn("positive", result["sentiment"])
    
    def test_search_with_filters_combined(self):
        """Test search_with_filters with combined filters."""
        # Call the method with combined filters
        results = self.scraper.search_with_filters(
            "test query",
            min_date="2023-06-16",
            author_filter="User1",
            has_mentions=True,
            include_sentiment=True
        )
        
        # Assert that the search method was called
        self.mock_search.assert_called_once()
        
        # Assert that the results are filtered correctly
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["author"], "User1")
        self.assertEqual(results[0]["timestamp"], "2023-06-18T16:20:00")
        self.assertTrue(len(results[0]["mentions"]) > 0)
        self.assertIn("sentiment", results[0])
    
    def test_get_trending_topics(self):
        """Test get_trending_topics."""
        # Mock the nltk functions
        with patch('cybersec_agents.grayswan.utils.enhanced_discord_utils.word_tokenize') as mock_tokenize, \
             patch('cybersec_agents.grayswan.utils.enhanced_discord_utils.stopwords') as mock_stopwords, \
             patch('cybersec_agents.grayswan.utils.enhanced_discord_utils.Counter') as mock_counter:
            
            # Set up the mocks
            mock_tokenize.return_value = ["i", "think", "gpt", "4", "is", "amazing", "it", "is", "so", "much", "better", "than", "previous", "models"]
            mock_stopwords.words.return_value = ["i", "it", "is", "so", "much", "than"]
            mock_counter.return_value.most_common.return_value = [
                ("gpt", 5),
                ("models", 4),
                ("amazing", 3),
                ("better", 2),
                ("previous", 1)
            ]
            
            # Call the method
            topics = self.scraper.get_trending_topics(self.sample_results, top_n=3)
            
            # Assert that the mocks were called
            mock_tokenize.assert_called_once()
            mock_stopwords.words.assert_called_once_with('english')
            mock_counter.return_value.most_common.assert_called_once_with(3)
            
            # Assert that the topics are correct
            self.assertEqual(len(topics), 3)
            self.assertEqual(topics[0]["word"], "gpt")
            self.assertEqual(topics[0]["count"], 5)
            self.assertEqual(topics[1]["word"], "models")
            self.assertEqual(topics[1]["count"], 4)
            self.assertEqual(topics[2]["word"], "amazing")
            self.assertEqual(topics[2]["count"], 3)
    
    def test_get_user_activity(self):
        """Test get_user_activity."""
        # Call the method
        user_activity = self.scraper.get_user_activity(self.sample_results)
        
        # Assert that the user activity is correct
        self.assertEqual(len(user_activity), 4)
        self.assertEqual(user_activity["User1"], 2)
        self.assertEqual(user_activity["User2"], 1)
        self.assertEqual(user_activity["User3"], 1)
        self.assertEqual(user_activity["User4"], 1)
    
    def test_get_channel_activity(self):
        """Test get_channel_activity."""
        # Call the method
        channel_activity = self.scraper.get_channel_activity(self.sample_results)
        
        # Assert that the channel activity is correct
        self.assertEqual(len(channel_activity), 2)
        self.assertEqual(channel_activity["ai-discussion"], 3)
        self.assertEqual(channel_activity["ai-safety"], 2)
    
    def test_get_time_distribution(self):
        """Test get_time_distribution."""
        # Call the method with different intervals
        day_dist = self.scraper.get_time_distribution(self.sample_results, interval="day")
        weekday_dist = self.scraper.get_time_distribution(self.sample_results, interval="weekday")
        
        # Assert that the day distribution is correct
        self.assertEqual(len(day_dist), 5)
        self.assertEqual(day_dist["2023-06-15"], 1)
        self.assertEqual(day_dist["2023-06-16"], 1)
        self.assertEqual(day_dist["2023-06-17"], 1)
        self.assertEqual(day_dist["2023-06-18"], 1)
        self.assertEqual(day_dist["2023-06-19"], 1)
        
        # Assert that the weekday distribution is correct (note: actual weekdays depend on the dates)
        self.assertEqual(len(weekday_dist), 5)  # Might be less if some messages are on the same weekday
    
    def test_search_by_timeframe(self):
        """Test search_by_timeframe."""
        # Mock datetime.now to return a fixed date
        with patch('cybersec_agents.grayswan.utils.enhanced_discord_utils.datetime') as mock_datetime:
            # Set up the mock
            mock_now = datetime(2023, 6, 20, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # Call the method with different timeframes
            today_results = self.scraper.search_by_timeframe("test query", "today")
            this_week_results = self.scraper.search_by_timeframe("test query", "this_week")
            
            # Assert that the search_with_filters method was called with the correct arguments
            self.assertEqual(self.mock_search.call_count, 2)
            
            # For today, min_date and max_date should be the same (2023-06-20)
            self.assertEqual(today_results, [])  # No results on 2023-06-20
            
            # For this_week, min_date should be the start of the week (2023-06-19 if Monday)
            # and max_date should be today (2023-06-20)
            self.assertEqual(len(this_week_results), 1)  # Only one result on 2023-06-19
    
    def test_format_advanced_results(self):
        """Test format_advanced_results."""
        # Call the method with different options
        basic_format = self.scraper.format_advanced_results(self.sample_results)
        detailed_format = self.scraper.format_advanced_results(
            self.sample_results,
            include_metadata=True,
            include_stats=True,
            max_messages=2
        )
        
        # Assert that the basic format is correct
        self.assertIn("Found 5 relevant Discord messages", basic_format)
        self.assertIn("User1", basic_format)
        self.assertIn("User2", basic_format)
        
        # Assert that the detailed format is correct
        self.assertIn("Found 5 relevant Discord messages", detailed_format)
        self.assertIn("Top Contributors", detailed_format)
        self.assertIn("Top Channels", detailed_format)
        self.assertIn("User1", detailed_format)
        self.assertIn("... and 3 more messages", detailed_format)
    
    @patch('cybersec_agents.grayswan.utils.enhanced_discord_utils.json')
    @patch('cybersec_agents.grayswan.utils.enhanced_discord_utils.os')
    def test_load_results(self, mock_os, mock_json):
        """Test load_results."""
        # Set up the mocks
        mock_os.path.join.side_effect = lambda *args: "/".join(args)
        mock_os.path.exists.return_value = True
        mock_json.load.return_value = self.sample_results
        
        # Mock the open function
        mock_open = unittest.mock.mock_open()
        with patch('builtins.open', mock_open):
            # Call the method
            results = self.scraper.load_results("test_results.json")
            
            # Assert that the mocks were called
            mock_os.path.join.assert_called_with("data", "discord_searches", "test_results.json")
            mock_os.path.exists.assert_called_once()
            mock_open.assert_called_once_with("/".join(["data", "discord_searches", "test_results.json"]), "r", encoding="utf-8")
            mock_json.load.assert_called_once()
            
            # Assert that the results are correct
            self.assertEqual(results, self.sample_results)
    
    def test_clear_cache(self):
        """Test clear_cache."""
        # Add some items to the cache
        self.scraper.result_cache = {
            "key1": {"results": [], "timestamp": 0},
            "key2": {"results": [], "timestamp": 0}
        }
        
        # Call the method
        self.scraper.clear_cache()
        
        # Assert that the cache is empty
        self.assertEqual(self.scraper.result_cache, {})


if __name__ == "__main__":
    unittest.main()