#!/usr/bin/env python3
"""
Reddit API Integration for Loop

This module provides a wrapper around PRAW (Python Reddit API Wrapper)
to fetch Reddit posts in a format compatible with the news timeline app.
"""

import praw
from datetime import datetime, timedelta
import time
from urllib.parse import urlparse
import re

class RedditAPI:
    """A wrapper around PRAW to provide Reddit content in a format compatible with the news app."""
    
    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        """Initialize the Reddit API client with credentials."""
        self.client_id = client_id or "F08_cb8WOEItwgq6z4ZPrw"
        self.client_secret = client_secret or "2cWVtFQANN0US0AlJ7rlulLX_0EVnQ"
        self.user_agent = user_agent or "script:loop:v1.0 (by /u/Inside-Advisor-7840)"
        self.reddit = None
        self.initialized = False
    
    def init(self):
        """Initialize the Reddit API connection."""
        if not self.initialized:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                self.initialized = True
                print("Reddit API initialized successfully")
            except Exception as e:
                print(f"Error initializing Reddit API: {str(e)}")
                raise
    
    def format_age(self, created_utc):
        """Format the age of a post similar to how the news API does it."""
        now = datetime.now()
        post_time = datetime.fromtimestamp(created_utc)
        diff = now - post_time
        
        if diff.days == 0:
            hours = diff.seconds // 3600
            if hours == 0:
                minutes = diff.seconds // 60
                return f"{minutes} minutes ago"
            return f"{hours} hours ago"
        elif diff.days == 1:
            return "1 day ago"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        elif diff.days < 30:
            weeks = diff.days // 7
            return f"{weeks} weeks ago" if weeks > 1 else "1 week ago"
        else:
            months = diff.days // 30
            return f"{months} months ago" if months > 1 else "1 month ago"
    
    def search(self, query, count=10, time_filter="week", freshness=None):
        """
        Search Reddit for posts matching the query.
        
        Args:
            query (str): The search query.
            count (int): Number of results to return.
            time_filter (str): One of 'hour', 'day', 'week', 'month', 'year', 'all'.
            freshness (str): Parameter to maintain compatibility with Brave API. Ignored.
            
        Returns:
            dict: Search results in a format compatible with Brave News API.
        """
        if not self.initialized:
            self.init()
        
        # Map the time_filter to Reddit's time_filter options
        valid_time_filters = ['hour', 'day', 'week', 'month', 'year', 'all']
        if time_filter not in valid_time_filters:
            time_filter = 'week'  # Default to week if invalid
        
        # Perform the search
        try:
            results = []
            posts = self.reddit.subreddit('all').search(
                query, 
                sort='relevance', 
                time_filter=time_filter,
                limit=count
            )
            
            for post in posts:
                # Parse the URL to get netloc for source information
                parsed_url = urlparse(post.url)
                netloc = parsed_url.netloc
                
                # Handle reddit.com URLs better for display
                if 'reddit.com' in netloc:
                    subreddit_name = post.subreddit.display_name
                    netloc = f"r/{subreddit_name} (Reddit)"
                
                # Create an article-like dict that matches the structure used by Brave API
                article = {
                    'title': post.title,
                    'description': post.selftext[:500] if hasattr(post, 'selftext') and post.selftext else f"Reddit post in r/{post.subreddit.display_name} with {post.score} upvotes and {post.num_comments} comments.",
                    'url': post.url,
                    'reddit_permalink': f"https://reddit.com{post.permalink}",
                    'age': self.format_age(post.created_utc),
                    'created_utc': post.created_utc,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'source': f"Reddit - r/{post.subreddit.display_name}",
                    'meta_url': {
                        'netloc': netloc,
                        'path': parsed_url.path
                    },
                    'is_reddit': True,
                    'data_reddit': 'true'
                }
                results.append(article)
            
            return {
                'results': results,
                'query': query,
                'count': len(results),
                'search_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error searching Reddit: {str(e)}")
            return {
                'results': [],
                'query': query,
                'count': 0,
                'error': str(e),
                'search_time': datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    reddit_api = RedditAPI()
    results = reddit_api.search("climate change", count=5)
    
    print(f"Found {len(results['results'])} results:")
    for i, article in enumerate(results['results'], 1):
        print(f"{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   Age: {article['age']}")
        print(f"   URL: {article['url']}")
        print(f"   Reddit: {article['reddit_permalink']}")
        print(f"   Upvotes: {article['score']} | Comments: {article['num_comments']}")
        print() 