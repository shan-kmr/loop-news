#!/usr/bin/env python3
"""
Twitter API Client

This module provides a client for interacting with the Twitter (X) API.
It allows users to search for tweets and format the results.
"""

import requests
import os
import sys
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

class TwitterAPI:
    """A class to interact with the Twitter (X) API."""

    # API endpoint for Twitter v2 search
    BASE_URL = "https://api.twitter.com/2/tweets/search/recent"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 bearer_token: Optional[str] = None):
        """
        Initialize the Twitter API client.

        Args:
            api_key: Twitter API key. If None, will look for TWITTER_API_KEY environment variable.
            api_secret: Twitter API secret. If None, will look for TWITTER_API_SECRET environment variable.
            bearer_token: Twitter Bearer token. If None, will look for TWITTER_BEARER_TOKEN environment variable.
        """
        # Get credentials from parameters or environment variables
        self.api_key = api_key or os.environ.get("TWITTER_API_KEY")
        self.api_secret = api_secret or os.environ.get("TWITTER_API_SECRET")
        self.bearer_token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN")
        self.credentials_valid = True

        # Check if we have at least bearer token
        if not self.bearer_token and not (self.api_key and self.api_secret):
            print("Warning: Twitter API credentials not provided. Twitter search will be unavailable.")
            self.credentials_valid = False
            return  # Don't exit, just mark as invalid

        # If we have API key and secret but no bearer token, get bearer token
        if not self.bearer_token and self.api_key and self.api_secret:
            try:
                self.bearer_token = self._get_bearer_token()
            except Exception as e:
                print(f"Warning: Failed to get Twitter bearer token: {e}")
                self.credentials_valid = False

    def _get_bearer_token(self) -> str:
        """
        Get a bearer token using the API key and secret.
        
        Returns:
            str: Bearer token
        """
        url = "https://api.twitter.com/oauth2/token"
        auth = (self.api_key, self.api_secret)
        data = {'grant_type': 'client_credentials'}
        
        try:
            response = requests.post(url, auth=auth, data=data)
            response.raise_for_status()
            return response.json()['access_token']
        except requests.exceptions.RequestException as e:
            print(f"Error getting bearer token: {e}")
            raise

    def search_tweets(self,
                    query: str,
                    max_results: int = 10,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    tweet_fields: List[str] = None,
                    expansions: List[str] = None,
                    user_fields: List[str] = None) -> Dict[str, Any]:
        """
        Search for tweets using the Twitter API.

        Args:
            query: Search query term
            max_results: Maximum number of results to return (default: 10, max: 100)
            start_time: UTC datetime to start search from (default: 7 days ago)
            end_time: UTC datetime to end search at (default: now)
            tweet_fields: Additional tweet fields to include (default: created_at,author_id,text,public_metrics)
            expansions: Expansions to include (default: author_id)
            user_fields: User fields to include (default: name,username,profile_image_url)

        Returns:
            Dict containing the API response
        """
        # Check if credentials are valid
        if not self.credentials_valid:
            return {"error": "Twitter API credentials are missing or invalid"}
            
        # Default fields
        if tweet_fields is None:
            tweet_fields = ["created_at", "author_id", "text", "public_metrics"]
        if expansions is None:
            expansions = ["author_id"]
        if user_fields is None:
            user_fields = ["name", "username", "profile_image_url"]

        # Default time range is last 7 days (limited by Twitter recent search)
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.utcnow()

        # Format dates to ISO 8601 format
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }

        # Prepare parameters
        params = {
            "query": query,
            "max_results": min(max_results, 100),  # Ensure count doesn't exceed API limit
            "start_time": start_time_str,
            "end_time": end_time_str,
            "tweet.fields": ",".join(tweet_fields),
            "expansions": ",".join(expansions),
            "user.fields": ",".join(user_fields)
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
            print(f"Error making Twitter API request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return {"error": str(e)}

    def format_tweet_results(self, response: Dict[str, Any]) -> None:
        """
        Format and print the tweet search results.

        Args:
            response: Twitter API response
        """
        if "error" in response:
            print(f"Error: {response['error']}")
            return

        # Check if we have results
        if "data" not in response or not response["data"]:
            print("No results found.")
            return

        # Build a user lookup dictionary if 'includes' is present
        users = {}
        if "includes" in response and "users" in response["includes"]:
            for user in response["includes"]["users"]:
                users[user["id"]] = user

        # Print search results header
        print(f"\n===== Twitter Search Results =====")
        print("=" * 35 + "\n")

        # Print each tweet
        for i, tweet in enumerate(response["data"], 1):
            tweet_id = tweet.get("id", "Unknown")
            text = tweet.get("text", "No text available")
            created_at = tweet.get("created_at", "Unknown date")
            author_id = tweet.get("author_id", "Unknown")
            
            # Format the date if available
            if created_at != "Unknown date":
                try:
                    created_dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                    created_at = created_dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass  # Keep the original format if parsing fails
            
            # Get user info if available
            user_name = "Unknown"
            user_handle = "Unknown"
            
            if author_id in users:
                user_name = users[author_id].get("name", "Unknown")
                user_handle = users[author_id].get("username", "Unknown")
            
            # Get tweet metrics if available
            metrics = tweet.get("public_metrics", {})
            likes = metrics.get("like_count", 0)
            retweets = metrics.get("retweet_count", 0)
            replies = metrics.get("reply_count", 0)
            
            # Format the tweet URL
            tweet_url = f"https://twitter.com/{user_handle}/status/{tweet_id}"
            
            # Print tweet in terminal format
            print(f"{i}. @{user_handle} ({user_name})")
            print(f"   Date: {created_at}")
            print(f"   {text}")
            print(f"   ❤️ {likes} | 🔄 {retweets} | 💬 {replies}")
            print(f"   URL: {tweet_url}")
            print("-" * 50)

def main():
    """Main function to run the tweet search from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search for tweets using Twitter API.")
    parser.add_argument("--query", type=str, required=True, 
                        help="Search query (required)")
    parser.add_argument("--count", type=int, default=10, 
                        help="Number of results to retrieve (default: 10, max: 100)")
    parser.add_argument("--days", type=int, default=7, 
                        help="Number of days to look back (default: 7, max: 7)")
    parser.add_argument("--bearer-token", type=str, 
                        help="Twitter Bearer token (optional, can use TWITTER_BEARER_TOKEN env var)")
    parser.add_argument("--api-key", type=str, 
                        help="Twitter API key (optional, can use TWITTER_API_KEY env var)")
    parser.add_argument("--api-secret", type=str, 
                        help="Twitter API secret (optional, can use TWITTER_API_SECRET env var)")

    args = parser.parse_args()

    # Initialize the Twitter API client
    twitter_api = TwitterAPI(
        api_key=args.api_key,
        api_secret=args.api_secret,
        bearer_token=args.bearer_token
    )

    # Calculate start time based on days argument
    days = min(args.days, 7)  # Twitter recent search is limited to 7 days
    start_time = datetime.utcnow() - timedelta(days=days)

    # Search for tweets
    print(f"Searching for tweets about '{args.query}'...")
    results = twitter_api.search_tweets(
        query=args.query,
        max_results=args.count,
        start_time=start_time
    )

    # Format and display the results
    twitter_api.format_tweet_results(results)

if __name__ == "__main__":
    main() 