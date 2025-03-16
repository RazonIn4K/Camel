# Enhanced Discord Integration

This document explains the Enhanced Discord Integration in the Gray Swan Arena framework.

## Overview

The Enhanced Discord Integration provides advanced capabilities for searching, analyzing, and extracting insights from Discord messages. It extends the base Discord integration with features such as:

1. **Advanced Filtering**: Filter messages by date, author, content, attachments, mentions, and sentiment
2. **Sentiment Analysis**: Analyze the sentiment of messages (positive, negative, neutral)
3. **Topic Extraction**: Identify trending topics in message content
4. **User and Channel Activity Analysis**: Analyze user and channel activity patterns
5. **Time Distribution Analysis**: Analyze message distribution over time
6. **Result Caching**: Cache search results for improved performance
7. **Timeframe-Based Searching**: Search for messages within specific timeframes (today, this week, last month, etc.)

## Components

The system consists of the following components:

### EnhancedDiscordScraper

The `EnhancedDiscordScraper` class in `enhanced_discord_utils.py` extends the base `DiscordScraper` class with advanced features:

```python
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
```

## Features

### 1. Advanced Filtering

The system provides advanced filtering capabilities for Discord messages:

```python
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
    """Search for messages with advanced filtering options."""
```

This method allows you to filter messages by:
- Date range (min_date, max_date)
- Author (author_filter)
- Content using regex patterns (content_filter)
- Presence of attachments (has_attachments)
- Presence of mentions (has_mentions)
- Sentiment (sentiment_filter)

### 2. Sentiment Analysis

The system can analyze the sentiment of Discord messages:

```python
def _filter_by_sentiment(
    self,
    results: List[Dict[str, Any]],
    sentiment_filter: Optional[str],
    include_sentiment: bool
) -> List[Dict[str, Any]]:
    """Apply sentiment analysis and filtering to results."""
```

Sentiment analysis classifies messages as:
- **Positive**: Messages with a positive tone
- **Negative**: Messages with a negative tone
- **Neutral**: Messages with a neutral tone

Each message gets a sentiment score with the following components:
- **negative**: Score for negative sentiment (0.0-1.0)
- **neutral**: Score for neutral sentiment (0.0-1.0)
- **positive**: Score for positive sentiment (0.0-1.0)
- **compound**: Overall sentiment score (-1.0 to 1.0)
- **classification**: Overall classification (positive, negative, neutral)

### 3. Topic Extraction

The system can extract trending topics from Discord messages:

```python
def get_trending_topics(
    self,
    results: List[Dict[str, Any]],
    top_n: int = 10,
    exclude_words: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Extract trending topics from search results."""
```

This method:
1. Extracts all content from the messages
2. Tokenizes the content into words
3. Removes stop words and custom excluded words
4. Counts the frequency of each word
5. Returns the top N most frequent words

### 4. User and Channel Activity Analysis

The system can analyze user and channel activity:

```python
def get_user_activity(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze user activity from search results."""

def get_channel_activity(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze channel activity from search results."""
```

These methods count the number of messages per user or channel and return a sorted dictionary.

### 5. Time Distribution Analysis

The system can analyze the distribution of messages over time:

```python
def get_time_distribution(
    self,
    results: List[Dict[str, Any]],
    interval: str = "day"
) -> Dict[str, int]:
    """Analyze time distribution of messages."""
```

This method groups messages by time intervals:
- **hour**: Hour of the day (00:00, 01:00, etc.)
- **day**: Day (YYYY-MM-DD)
- **weekday**: Day of the week (Monday, Tuesday, etc.)
- **month**: Month and year (January 2023, February 2023, etc.)

### 6. Timeframe-Based Searching

The system can search for messages within specific timeframes:

```python
def search_by_timeframe(
    self,
    query: str,
    timeframe: str,
    channel_ids: Optional[List[str]] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Search for messages within a specific timeframe."""
```

Supported timeframes:
- **today**: Messages from today
- **yesterday**: Messages from yesterday
- **this_week**: Messages from the current week (starting Monday)
- **last_week**: Messages from the previous week
- **this_month**: Messages from the current month
- **last_month**: Messages from the previous month

### 7. Result Caching

The system caches search results to improve performance:

```python
# Cache for search results
self.result_cache = {}
self.cache_expiry = 3600  # Cache results for 1 hour
```

Search results are cached for a configurable period (default: 1 hour) to avoid redundant API calls.

### 8. Advanced Result Formatting

The system provides advanced formatting options for search results:

```python
def format_advanced_results(
    self,
    results: List[Dict[str, Any]],
    include_metadata: bool = False,
    include_stats: bool = False,
    max_messages: int = 10
) -> str:
    """Format search results into a readable string with advanced options."""
```

This method can include:
- Message metadata (URL, attachments, mentions)
- Statistics about the results (top users, top channels, trending topics)
- A configurable number of messages

## Dependencies

The Enhanced Discord Integration has the following dependencies:

- **discord.py**: For Discord API integration
- **nltk**: For natural language processing (sentiment analysis, topic extraction)
- **dotenv**: For loading environment variables

## Configuration

The system can be configured through environment variables:

```
# Discord Integration
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_CHANNEL_IDS=123456789012345678,987654321098765432
```

## Usage Examples

### Basic Usage

```python
from cybersec_agents.grayswan.utils.enhanced_discord_utils import EnhancedDiscordScraper

# Create Discord scraper
scraper = EnhancedDiscordScraper()

# Search for messages
results = scraper.search("jailbreak technique", limit=20)

# Print results
print(scraper.format_results(results))

# Save results
scraper.save_results(results, "search_results.json")
```

### Advanced Filtering

```python
# Search with filters
results = scraper.search_with_filters(
    query="AI safety",
    min_date="2023-01-01",
    max_date="2023-12-31",
    author_filter="John",
    content_filter=r"(safety|alignment|ethics)",
    has_attachments=True,
    include_sentiment=True
)

# Print results with statistics
print(scraper.format_advanced_results(
    results,
    include_metadata=True,
    include_stats=True
))
```

### Sentiment Analysis

```python
# Search with sentiment analysis
results = scraper.search_with_filters(
    query="language model",
    include_sentiment=True
)

# Filter for positive messages
positive_results = scraper.search_with_filters(
    query="language model",
    sentiment_filter="positive",
    include_sentiment=True
)
```

### Topic Extraction

```python
# Extract trending topics
topics = scraper.get_trending_topics(
    results,
    top_n=10,
    exclude_words=["about", "would", "could"]
)

# Print topics
for topic in topics:
    print(f"{topic['word']}: {topic['count']} occurrences")
```

### Timeframe-Based Searching

```python
# Search for messages from this week
results = scraper.search_by_timeframe(
    query="AI model",
    timeframe="this_week",
    limit=50
)
```

### User and Channel Activity

```python
# Get user activity
user_activity = scraper.get_user_activity(results)

# Get channel activity
channel_activity = scraper.get_channel_activity(results)

# Get time distribution
time_dist = scraper.get_time_distribution(results, interval="day")
```

### Loading and Saving Results

```python
# Save results
filepath = scraper.save_results(results, "search_results.json")

# Load results
loaded_results = scraper.load_results("search_results.json")
```

## Integration with Gray Swan Arena

The Enhanced Discord Integration is designed to work seamlessly with the Gray Swan Arena framework:

### Reconnaissance Agent

The Reconnaissance Agent can use the Enhanced Discord Integration to gather information about AI models and jailbreaking techniques:

```python
from cybersec_agents.grayswan.agents.recon_agent import ReconAgent
from cybersec_agents.grayswan.utils.enhanced_discord_utils import EnhancedDiscordScraper

# Create agents
recon_agent = ReconAgent()
discord_scraper = EnhancedDiscordScraper()

# Search Discord
discord_results = discord_scraper.search_with_filters(
    query="jailbreak technique",
    min_date="2023-01-01",
    include_sentiment=True
)

# Extract insights
trending_topics = discord_scraper.get_trending_topics(discord_results)
user_activity = discord_scraper.get_user_activity(discord_results)

# Include in reconnaissance report
report = recon_agent.generate_report(
    target_model="GPT-4",
    target_behavior="harmful content",
    web_results=web_results,
    discord_results=discord_results,
    discord_insights={
        "trending_topics": trending_topics,
        "user_activity": user_activity
    }
)
```

## Best Practices

1. **Use Specific Queries**: More specific queries yield more relevant results
2. **Apply Appropriate Filters**: Use filters to narrow down results
3. **Cache Results**: Use the caching mechanism to avoid redundant API calls
4. **Handle Rate Limits**: Be mindful of Discord API rate limits
5. **Process Results in Batches**: For large result sets, process results in batches
6. **Include Sentiment Analysis**: Sentiment analysis can provide valuable insights
7. **Extract Trending Topics**: Topic extraction can identify important themes
8. **Analyze Activity Patterns**: User and channel activity analysis can reveal patterns
9. **Use Timeframe-Based Searching**: Timeframe-based searching can focus on recent or specific periods
10. **Save and Load Results**: Save results for later analysis and reference

## Limitations

1. **API Rate Limits**: Discord API has rate limits that may restrict the number of requests
2. **Message History Limits**: Discord may limit access to older messages
3. **Channel Access**: The bot must have access to the channels it searches
4. **NLP Dependencies**: Some features require NLTK and may not work if not installed
5. **Performance**: Searching large numbers of messages can be slow
6. **Accuracy**: Sentiment analysis and topic extraction may not always be accurate
7. **Privacy Considerations**: Be mindful of privacy when scraping and analyzing messages

## Future Enhancements

1. **Message Clustering**: Group similar messages together
2. **Entity Recognition**: Identify entities (people, organizations, products) in messages
3. **Conversation Analysis**: Analyze conversation threads and interactions
4. **Reaction Analysis**: Analyze message reactions
5. **Image and Attachment Analysis**: Analyze images and attachments
6. **Multi-Channel Correlation**: Correlate messages across multiple channels
7. **Real-Time Monitoring**: Monitor Discord channels in real-time
8. **Advanced NLP**: Implement more advanced NLP techniques
9. **Integration with Other Data Sources**: Integrate with other data sources (Twitter, Reddit, etc.)
10. **Visualization**: Create visualizations of Discord data