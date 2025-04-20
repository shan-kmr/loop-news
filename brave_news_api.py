#!/usr/bin/env python3
"""
Brave News API Client with Twitter Integration

This module provides a client for interacting with the Brave News Search API
and Twitter API. It allows users to search for news articles and tweets
with the same keywords and format the results.
"""

import requests
import json
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Import the TwitterAPI from our twitter_api module
from twitter_api import TwitterAPI

class BraveNewsAPI:
    """A class to interact with the Brave News Search API."""

    # API endpoint for Brave News Search
    BASE_URL = "https://api.search.brave.com/res/v1/news/search"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Brave News API client.

        Args:
            api_key: Brave Search API key. If None, will look for BRAVE_API_KEY environment variable.
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")

        if not self.api_key:
            print("Error: Brave API key not provided.")
            print("Please either set the BRAVE_API_KEY environment variable or provide it as a parameter.")
            sys.exit(1)

    def search_news(self,
                   query: str,
                   count: int = 10,
                   offset: int = 0,
                   country: str = "US",
                   search_lang: str = "en",
                   ui_lang: str = "en-US",
                   freshness: str = "pw",
                   spellcheck: bool = True,
                   safesearch: str = "moderate",
                   extra_snippets: bool = True) -> Dict[str, Any]:
        """
        Search for news articles using the Brave News Search API.

        Args:
            query: Search query term
            count: Number of results to return (default: 10, max: 50)
            offset: Zero-based page offset for pagination (default: 0, max: 9)
            country: Country source for the news results (default: "US")
            search_lang: Language code for results (default: "en")
            ui_lang: User interface language preference (default: "en-US")
            freshness: Filter by news discovery date (default: "pw" - past week)
            spellcheck: Whether to apply spellchecking on the search query (default: True)
            safesearch: Filter for adult content (default: "moderate")
            extra_snippets: Whether to return additional alternative excerpts (default: True)

        Returns:
            Dict containing the API response
        """
        # Prepare headers
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }

        # Prepare parameters
        params = {
            "q": query,
            "count": min(count, 50),  # Ensure count doesn't exceed API limit
            "offset": min(offset, 9),  # Ensure offset doesn't exceed API limit
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
            "freshness": freshness,
            "spellcheck": 1 if spellcheck else 0,
            "safesearch": safesearch,
            "extra_snippets": 1 if extra_snippets else 0
        }

        try:
            # Make the API request
            response = requests.get(
                self.BASE_URL,
                headers=headers,
                params=params
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse and return the JSON response
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            return {"error": str(e)}

    def format_news_results(self, response: Dict[str, Any]) -> None:
        """
        Format and print the news search results in a timeline view, sorted by date.

        Args:
            response: Brave News Search API response
        """
        if "error" in response:
            print(f"Error: {response['error']}")
            return

        # Check if we have results
        if "results" not in response or not response["results"]:
            print("No results found.")
            return

        # Print search metadata
        print(f"\n===== News Timeline: {response.get('query', {}).get('original', 'Unknown')} =====")
        print(f"Results as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50 + "\n")

        # Sort articles by age - convert age strings to a sortable format first
        articles = response["results"]
        
        # Function to convert age strings to seconds for sorting
        def extract_age_in_seconds(article):
            age = article.get("age", "Unknown")
            
            # Try to extract time information from the age string
            if isinstance(age, str):
                # Handle common formats like "2 hours ago", "1 day ago", etc.
                if "minute" in age:
                    minutes = int(''.join(filter(str.isdigit, age)))
                    return minutes * 60
                elif "hour" in age:
                    hours = int(''.join(filter(str.isdigit, age)))
                    return hours * 3600
                elif "day" in age:
                    days = int(''.join(filter(str.isdigit, age)))
                    return days * 86400
                elif "week" in age:
                    weeks = int(''.join(filter(str.isdigit, age)))
                    return weeks * 604800
                elif "month" in age:
                    months = int(''.join(filter(str.isdigit, age)))
                    return months * 2592000
            
            # Default to a large number for unknown ages to place them at the end
            return float('inf')
        
        # Sort articles by age (newest first)
        sorted_articles = sorted(articles, key=extract_age_in_seconds)
        
        # Group articles by day for timeline display
        current_day = None

        # Print each news article
        for i, article in enumerate(sorted_articles, 1):
            title = article.get("title", "No title")
            description = article.get("description", "No description available")
            url = article.get("url", "No URL available")
            age = article.get("age", "Unknown")
            source = article.get("meta_url", {}).get("netloc", "Unknown source")

            # Extract the day for timeline grouping
            day_str = "Today"
            if "day" in age:
                days = int(''.join(filter(str.isdigit, age)))
                if days == 1:
                    day_str = "Yesterday"
                else:
                    day_str = f"{days} days ago"
            elif "week" in age or "month" in age:
                day_str = age

            # Print day header if changed
            if day_str != current_day:
                current_day = day_str
                print(f"\n[{day_str.upper()}]")
                print("-" * 20)

            # Print article in terminal format
            print(f"{i}. {title}")
            print(f"   Time: {age} | Source: {source}")
            print(f"   {description}")
            print(f"   URL: {url}")
            print("-" * 50)

class NewsAndTweetSearcher:
    """
    A class that combines Brave News API and Twitter API search functionality.
    """
    
    def __init__(self, brave_api_key: Optional[str] = None, 
                 twitter_bearer_token: Optional[str] = None,
                 twitter_api_key: Optional[str] = None, 
                 twitter_api_secret: Optional[str] = None):
        """
        Initialize with API keys for both services.
        
        Args:
            brave_api_key: Brave Search API key
            twitter_bearer_token: Twitter Bearer token
            twitter_api_key: Twitter API key
            twitter_api_secret: Twitter API secret
        """
        self.brave_api = BraveNewsAPI(api_key=brave_api_key)
        # Store Twitter credentials for lazy initialization
        self.twitter_credentials = {
            'bearer_token': twitter_bearer_token,
            'api_key': twitter_api_key,
            'api_secret': twitter_api_secret
        }
        self._twitter_api = None
    
    @property
    def twitter_api(self):
        """Lazily initialize Twitter API only when needed"""
        if self._twitter_api is None:
            self._twitter_api = TwitterAPI(
                bearer_token=self.twitter_credentials['bearer_token'],
                api_key=self.twitter_credentials['api_key'],
                api_secret=self.twitter_credentials['api_secret']
            )
        return self._twitter_api
    
    def search(self, query: str, news_count: int = 10, tweet_count: int = 10, 
               freshness: str = "pw", include_news: bool = True, include_tweets: bool = False) -> None:
        """
        Search for news and tweets with the same query.
        Default behavior is news only, with results displayed in a chronological timeline view.
        
        Args:
            query: Search query to use for both APIs
            news_count: Number of news results to retrieve
            tweet_count: Number of tweet results to retrieve
            freshness: Freshness parameter for news search
            include_news: Whether to include news search results (default: True)
            include_tweets: Whether to include tweet search results (default: False)
        """
        if include_news:
            # Search for news
            print(f"Searching for news about '{query}'...")
            news_results = self.brave_api.search_news(
                query=query,
                count=news_count,
                freshness=freshness
            )
            # Format and display the news results in a timeline view
            self.brave_api.format_news_results(news_results)
        
        if include_tweets:
            # Search for tweets
            print(f"Searching for tweets about '{query}'...")
            tweet_results = self.twitter_api.search_tweets(
                query=query,
                max_results=tweet_count
            )
            # Format and display the tweet results
            self.twitter_api.format_tweet_results(tweet_results)

def main():
    """Main function to run the news and tweet search from command line"""
    parser = argparse.ArgumentParser(
        description="Search for news and tweets using Brave Search API and Twitter API. By default, only news search is enabled.")
    parser.add_argument("--query", type=str, default="breaking news", 
                        help="Search query (default: 'breaking news')")
    parser.add_argument("--news-count", type=int, default=10, 
                        help="Number of news results to retrieve (default: 10, max: 50)")
    parser.add_argument("--tweet-count", type=int, default=10, 
                        help="Number of tweet results to retrieve (default: 10, max: 100)")
    parser.add_argument("--freshness", type=str, default="pw",
                        choices=["pd", "pw", "pm", "py"],
                        help="Filter for news freshness: pd (past day), pw (past week), pm (past month), py (past year)")
    parser.add_argument("--brave-api-key", type=str, 
                        help="Brave Search API key (optional, can use BRAVE_API_KEY env var)")
    parser.add_argument("--twitter-bearer-token", type=str, 
                        help="Twitter Bearer token (optional, can use TWITTER_BEARER_TOKEN env var)")
    parser.add_argument("--twitter-api-key", type=str, 
                        help="Twitter API key (optional, can use TWITTER_API_KEY env var)")
    parser.add_argument("--twitter-api-secret", type=str, 
                        help="Twitter API secret (optional, can use TWITTER_API_SECRET env var)")
    parser.add_argument("--include-tweets", action="store_true",
                        help="Include Twitter search results (default: news only)")
    parser.add_argument("--news-only", action="store_true",
                        help="Only search for news (skip tweets, default behavior)")
    parser.add_argument("--tweets-only", action="store_true",
                        help="Only search for tweets (skip news)")

    args = parser.parse_args()

    # Determine what to include based on flags
    include_news = not args.tweets_only
    include_tweets = args.include_tweets or args.tweets_only

    # Check if we need Twitter credentials but none are provided
    if include_tweets:
        if not (args.twitter_bearer_token or 
                (args.twitter_api_key and args.twitter_api_secret) or
                os.environ.get("TWITTER_BEARER_TOKEN") or
                (os.environ.get("TWITTER_API_KEY") and os.environ.get("TWITTER_API_SECRET"))):
            print("Warning: Twitter search requested but no Twitter API credentials provided.")
            print("Set TWITTER_BEARER_TOKEN environment variable or provide --twitter-bearer-token.")
            print("Continuing with news search only.")
            include_tweets = False

    # Initialize the combined searcher
    searcher = NewsAndTweetSearcher(
        brave_api_key=args.brave_api_key,
        twitter_bearer_token=args.twitter_bearer_token,
        twitter_api_key=args.twitter_api_key,
        twitter_api_secret=args.twitter_api_secret
    )

    # Search for news and tweets
    searcher.search(
        query=args.query,
        news_count=args.news_count,
        tweet_count=args.tweet_count,
        freshness=args.freshness,
        include_news=include_news,
        include_tweets=include_tweets
    )

if __name__ == "__main__":
    main() 