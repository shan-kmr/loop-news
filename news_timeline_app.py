#!/usr/bin/env python3
"""
News Timeline Web Application

A Flask-based web app that provides a UI for searching news articles and
displaying them in a timeline view, sorted by recency.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_login import LoginManager, login_required, current_user
import os
import re
import json
import string
import time
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import threading
import queue
import pytz  # Add pytz for timezone handling
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import urlparse
import traceback

# Import configuration and models
from config import *
from models import User
from auth import auth_bp, init_oauth
from brave_news_api import BraveNewsAPI
from reddit_api import RedditAPI  # Import our new Reddit API module

# Add dependencies for LLM models
try:
    # OpenAI is the only supported model
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI package not available. To enable OpenAI summaries run:")
    print("pip install openai")
    OPENAI_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config')

# Register auth blueprint
app.register_blueprint(auth_bp, url_prefix='/auth')

# Set up Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'  # Redirect to login when login_required

# Initialize OAuth
init_oauth(app)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create the Brave News API client
brave_api = None

# Create the Reddit API client
reddit_api = None

@login_manager.user_loader
def load_user(user_id):
    """Load user from storage for Flask-Login"""
    return User.get(user_id)

def init_models():
    """Initialize the OpenAI client"""
    if OPENAI_AVAILABLE:
        init_openai()
    else:
        print("Warning: OpenAI is not available. Please install the openai package.")

def init_openai():
    """Initialize the OpenAI client"""
    if not OPENAI_AVAILABLE:
        return
    
    try:
        # Set the API key
        openai.api_key = OPENAI_API_KEY
        print("OpenAI client initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")

def summarize_daily_news(day_articles, query):
    """Generate a summary of articles for a specific day using OpenAI"""
    return summarize_daily_news_openai(day_articles, query)

def summarize_daily_news_openai(day_articles, query):
    """Generate a summary of articles for a specific day using OpenAI's API"""
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        # Prepare content to summarize
        article_texts = []
        for article in day_articles:
            title = article.get('title', '')
            desc = article.get('description', '')
            source = article.get('meta_url', {}).get('netloc', 'Unknown source')
            article_texts.append(f"- {title}: {desc} (Source: {source})")
        
        all_articles_text = "\n".join(article_texts[:15])  # Limit to 15 articles to avoid token overflow
        
        # Create messages for the OpenAI API
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise news summaries. Summarize the key events and topics from these news articles in about 2-3 sentences without adding any new information. Focus on identifying the main developments and common themes. Start directly with the summary content - do not use phrases like 'Here is the summary' or 'The main news is'."},
            {"role": "user", "content": f"Here are news articles about \"{query}\" from the same day:\n\n{all_articles_text}\n\nPlease provide a brief summary of the main news for this day."}
        ]
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=150,
            top_p=0.9
        )
        
        # Extract the summary from the response
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Error generating summary with OpenAI: {str(e)}")
        return None

# Load search history from JSON file
def load_search_history(user=None):
    history_file = get_history_file(user=user)
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
    return {}

# Save search history to JSON file
def save_search_history(history, user=None):
    history_file = get_history_file(user=user)
    try:
        # Ensure the directory exists if the path contains a directory
        dir_name = os.path.dirname(history_file)
        if dir_name:  # Only try to create directory if there is one
            os.makedirs(dir_name, exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving history: {e}")

# Get search history file path
def get_history_file(user=None):
    """
    Get search history file path
    
    Args:
        user: Optional User object. If not provided and current_user is authenticated,
              will use current_user. For background tasks where current_user is not 
              available, this allows passing None to get the default location.
    """
    try:
        # First try with the provided user parameter
        if user and hasattr(user, 'is_authenticated') and user.is_authenticated:
            history_file = user.get_history_file()
            if history_file:  # Check that we got a valid path
                return history_file
        
        # Then try with current_user if in a request context
        if 'current_user' in globals() and current_user and hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
            history_file = current_user.get_history_file()
            if history_file:  # Check that we got a valid path
                return history_file
    except Exception as e:
        print(f"Error getting user history file: {e}")
    
    # Default location for anonymous users, background threads, or if user path is invalid
    default_dir = 'user_data/shared'
    os.makedirs(default_dir, exist_ok=True)
    return os.path.join(default_dir, 'search_history.json')

def update_current_day_results(query, count, freshness, old_results):
    """
    Refresh all articles from the last search date to the current date,
    filling in any gap periods in between.
    
    Args:
        query: The search query
        count: Number of results to fetch
        freshness: Freshness parameter for the API
        old_results: Previous search results
        
    Returns:
        Updated results dictionary with refreshed current day articles and a list of days with new content
    """
    global brave_api
    global reddit_api
    
    if brave_api is None:
        brave_api_key = app.config.get("BRAVE_API_KEY")
        if not brave_api_key:
            return old_results, []
        brave_api = BraveNewsAPI(api_key=brave_api_key)
    
    try:
        # For frequent refreshes (every 60 minutes), default to past day
        # For longer term refreshes, use the original freshness or at least past week
        refresh_freshness = 'pd'  # Default to past day for 60-minute refreshes
        
        # If the original freshness was longer (pw, pm, py), keep that for non-frequent refreshes
        if freshness in ['pw', 'pm', 'py'] and CACHE_VALIDITY_HOURS > 24:
            refresh_freshness = freshness  # Keep original longer freshness for non-frequent refreshes
        
        print(f"Using freshness '{refresh_freshness}' for refreshing '{query}' (original: {freshness})")
        
        # Get fresh news results
        fresh_results = brave_api.search_news(
            query=query,
            count=max(count * 2, 50),  # Get more results to ensure gap coverage
            freshness=refresh_freshness
        )
        
        # Get Reddit results if available
        reddit_results = None
        if reddit_api is not None:
            try:
                # Map freshness to time_filter for Reddit
                time_filter_map = {
                    'pd': 'day',
                    'pw': 'week',
                    'pm': 'month',
                    'py': 'year',
                }
                reddit_time_filter = time_filter_map.get(refresh_freshness, 'week')
                
                # Get Reddit results
                reddit_results = reddit_api.search(
                    query=query,
                    count=count,
                    time_filter=reddit_time_filter
                )
                print(f"Got {len(reddit_results.get('results', []))} Reddit results for '{query}'")
            except Exception as e:
                print(f"Error getting Reddit results: {str(e)}")
                reddit_results = None
        
        if not fresh_results or 'results' not in fresh_results or not old_results or 'results' not in old_results:
            # If we have Reddit results but no fresh news results, combine them with old results
            if reddit_results and 'results' in reddit_results and old_results and 'results' in old_results:
                # Combine Reddit results with old results
                combined_results = old_results.copy()
                combined_results['results'] = old_results['results'] + reddit_results['results']
                return combined_results, ['Today']
            # Otherwise return fresh results or old results if no fresh ones
            return fresh_results or old_results, []
        
        # Get new and old articles
        new_articles = fresh_results['results']
        old_articles = old_results['results']
        
        # Add Reddit results if available
        if reddit_results and 'results' in reddit_results:
            new_articles.extend(reddit_results['results'])
            print(f"Added {len(reddit_results['results'])} Reddit results to new articles (total: {len(new_articles)})")
        
        # Create a set of URLs from old articles for comparison
        old_urls = {article.get('url') for article in old_articles if 'url' in article}
        
        # Find truly new articles (not already in old results)
        new_article_urls = {article.get('url') for article in new_articles if 'url' in article}
        truly_new_urls = new_article_urls - old_urls
        
        # Track which days have new content added
        days_with_new_articles = []
        
        if truly_new_urls:
            # We have new content, find which days they belong to
            for article in new_articles:
                if article.get('url') in truly_new_urls:
                    day = day_group_filter(article)
                    if day not in days_with_new_articles:
                        days_with_new_articles.append(day)
            
            # Sort old and new articles to ensure consistent order
            old_articles_dict = {article.get('url'): article for article in old_articles if 'url' in article}
            
            # Remove duplicates while merging old and new articles
            merged_articles = []
            
            for article in new_articles:
                if 'url' in article:
                    merged_articles.append(article)
                    # Remove from old_articles_dict to avoid duplicates
                    old_articles_dict.pop(article.get('url'), None)
            
            # Add remaining old articles that weren't in new results
            merged_articles.extend(old_articles_dict.values())
            
            # Sort by most recent first
            merged_articles.sort(key=extract_age_in_seconds)
            
            # Create merged results
            merged_results = fresh_results.copy()
            merged_results['results'] = merged_articles
            
            print(f"Merged {len(new_articles)} new and {len(old_articles)} old articles into {len(merged_articles)} unique articles")
            print(f"Found new content for days: {days_with_new_articles}")
            
            return merged_results, days_with_new_articles
        else:
            print(f"No new content found for '{query}'")
            return old_results, []
    except Exception as e:
        print(f"Error refreshing results: {str(e)}")
        return old_results, []

def collect_day_summaries(history):
    """Collect latest day (Today) summaries from history entries for display on the home page."""
    latest_summaries = {}
    
    # Priority order of days, with "Today" being highest priority
    day_priority = {
        "Today": 1,
        "Yesterday": 2,
        "2 days ago": 3,
        "3 days ago": 4,
        "4 days ago": 5,
        "5 days ago": 6,
        "6 days ago": 7,
        "1 week ago": 8,
        "2 weeks ago": 9,
        "1 month ago": 10
    }
    
    for key, entry in history.items():
        query = entry.get('query', '')
        if not query:
            continue
            
        if 'day_summaries' in entry and entry['day_summaries']:
            best_day = None
            best_priority = float('inf')
            best_summary = None
            
            # Find the most recent day with a valid summary
            for day, summary in entry['day_summaries'].items():
                if is_valid_summary(summary):
                    priority = day_priority.get(day, 999)  # Default to low priority if not in our list
                    if priority < best_priority:
                        best_priority = priority
                        best_day = day
                        best_summary = summary
            
            # If we found a valid summary, store it
            if best_day and best_summary:
                summary_key = f"{best_day}_{query}"
                latest_summaries[summary_key] = {
                    'query': query,
                    'day': best_day,
                    'summary': best_summary,
                }
    
    return latest_summaries

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route that handles both the search form and displaying results."""
    global brave_api
    global reddit_api
    
    # Redirect to raison d'etre page if not logged in
    if not current_user.is_authenticated:
        return redirect(url_for('raison_detre'))
    
    # Initialize the API clients if needed
    if brave_api is None:
        brave_api_key = app.config.get("BRAVE_API_KEY")
        if not brave_api_key:
            return render_template('error.html', 
                                  error="BRAVE_API_KEY environment variable or config not set. Please set it and restart the app.",
                                  user=current_user)
        brave_api = BraveNewsAPI(api_key=brave_api_key)
    
    if reddit_api is None:
        try:
            reddit_api = RedditAPI()
            reddit_api.init()
        except Exception as e:
            print(f"Warning: Could not initialize Reddit API: {str(e)}")
            print("Reddit content will not be available")

    # Default query and results
    query = request.args.get('query', 'breaking news')
    results = None
    search_time = None
    error = None
    sorted_articles = []
    topic_groups = []
    history = load_search_history()
    
    # Check if refresh is being forced via URL parameter
    force_refresh = request.args.get('force_refresh') == 'true'
    
    # Handle forced refresh from URL parameter explicitly
    if force_refresh and query and query != 'breaking news':
        print(f"Force refreshing query from URL parameter: '{query}'")
        # Look for existing entry to refresh
        cache_keys = []
        for key, entry in history.items():
            if entry.get('query') == query:
                cache_keys.append(key)
                
        if cache_keys:
            # Use the first matching entry
            cache_key = cache_keys[0]
            entry = history[cache_key]
            count = entry.get('count', 10)
            freshness = entry.get('freshness', 'pw')
            old_results = entry.get('results')
            
            if old_results:
                try:
                    # Update results while preserving history
                    updated_results, days_with_new_articles = update_current_day_results(query, count, freshness, old_results)
                    
                    # Only update if we got new results
                    if updated_results != old_results:
                        # Update the entry with fresh results
                        entry['results'] = updated_results
                        entry['timestamp'] = datetime.now().isoformat()
                        
                        # Clear summaries that need regeneration
                        if days_with_new_articles and 'day_summaries' in entry:
                            for day in days_with_new_articles:
                                if day in entry['day_summaries']:
                                    entry['day_summaries'][day] = None
                        
                        # Force regeneration of Today's summary
                        if 'day_summaries' in entry:
                            entry['day_summaries']['Today'] = None
                            
                        # Save the updated history
                        save_search_history(history)
                        print(f"Successfully force refreshed '{query}' while preserving history")
                    else:
                        # Even if results are the same, update the timestamp
                        entry['timestamp'] = datetime.now().isoformat()
                        save_search_history(history)
                        print(f"No new results for '{query}', but updated timestamp")
                        
                    # Redirect to detail page with the refreshed data
                    return redirect(url_for('history_item', query=query))
                except Exception as e:
                    print(f"Error during force refresh: {str(e)}")
    
    # History and timing handling
    sorted_history = sorted(history.values(), key=lambda entry: entry.get('search_time', 0), reverse=True)
    
    # Format history timestamps for display
    for entry in sorted_history:
        if 'search_time' in entry:
            entry['formatted_time'] = datetime.fromtimestamp(entry['search_time']).strftime('%Y-%m-%d %H:%M:%S')
    
    # Check for stale entries in history on home page load and refresh if needed
    if force_refresh is False and request.method == 'GET' and not request.args.get('query'):
        refreshed_entries = []
        for key, entry in history.items():
            try:
                # Parse the cached timestamp
                cached_time_str = entry['timestamp']
                cached_time = datetime.fromisoformat(cached_time_str)
                # Remove timezone if present
                if hasattr(cached_time, 'tzinfo') and cached_time.tzinfo is not None:
                    cached_time = cached_time.replace(tzinfo=None)
                
                # Calculate time difference
                time_diff = datetime.now() - cached_time
                
                # Refresh if older than 60 minutes
                if time_diff.total_seconds() > 3600:  # 60 minutes
                    query = entry.get('query')
                    count = entry.get('count', 10)
                    freshness = entry.get('freshness', 'pw')
                    
                    if query:
                        print(f"Home page: Auto-refreshing stale entry '{query}' ({int(time_diff.total_seconds() / 60)} minutes old)")
                        results = entry.get('results')
                        
                        if results:
                            updated_results, days_with_new_articles = update_current_day_results(query, count, freshness, results)
                            
                            if updated_results != results:
                                # Update the entry with fresh results
                                entry['results'] = updated_results
                                entry['timestamp'] = datetime.now().isoformat()
                                
                                # Clear summaries that need regeneration
                                if days_with_new_articles and 'day_summaries' in entry:
                                    for day in days_with_new_articles:
                                        if day in entry['day_summaries']:
                                            entry['day_summaries'][day] = None
                                
                                # Force regeneration of Today's summary
                                if 'day_summaries' in entry:
                                    entry['day_summaries']['Today'] = None
                                
                                refreshed_entries.append(query)
            except Exception as e:
                print(f"Error checking timestamp for auto-refresh: {str(e)}")
        
        # If any entries were refreshed, save the updated history
        if refreshed_entries:
            save_search_history(history)
            print(f"Auto-refreshed {len(refreshed_entries)} stale entries: {', '.join(refreshed_entries)}")
    
    # Collect day summaries regardless of request method
    day_summaries = collect_day_summaries(history)
    print(f"Collected {len(day_summaries)} day summaries for home page display")
    
    # Debug each summary
    for key, summary in day_summaries.items():
        print(f"Home page summary for '{summary['query']}' on {summary['day']}: {summary['summary'][:50]}...")
    
    # Process form submission
    if request.method == 'POST' and current_user.is_authenticated:
        query = request.form.get('query', '').strip()
        count = int(request.form.get('count', 10))
        freshness = request.form.get('freshness', 'pw')
        similarity_threshold = float(request.form.get('similarity_threshold', 0.3))
        force_refresh = request.form.get('force_refresh') == 'true'
        
        # Check if user has reached the limit of 3 topics
        unique_queries = set()
        for entry in history.values():
            if 'query' in entry:
                unique_queries.add(entry['query'])
        
        if len(unique_queries) >= 3 and query not in unique_queries:
            error = "You've reached the maximum of 3 briefs. Please remove one before adding another."
        elif query:
            # Check if we already have cached results that are still valid
            cache_key = f"{query}_{count}_{freshness}"
            cached_entry = history.get(cache_key)
            use_cache = False
            needs_day_refresh = False
            
            if cached_entry and not force_refresh:
                # Store current time for comparison, ensuring naive datetime (no timezone)
                now = datetime.now()
                
                try:
                    # Parse the cached timestamp and ensure it's naive (no timezone)
                    cached_time_str = cached_entry['timestamp']
                    cached_time = datetime.fromisoformat(cached_time_str)
                    # Remove timezone if present
                    if hasattr(cached_time, 'tzinfo') and cached_time.tzinfo is not None:
                        cached_time = cached_time.replace(tzinfo=None)
                    
                    # Calculate time difference
                    time_diff = now - cached_time
                    
                    # Use cache if it's less than CACHE_VALIDITY_HOURS old
                    if time_diff.total_seconds() < CACHE_VALIDITY_HOURS * 3600:
                        use_cache = True
                        
                        # Check if we need to refresh the current day's results (older than 60 minutes)
                        if time_diff.total_seconds() > 3600:  # 60 minutes in seconds
                            needs_day_refresh = True
                        
                        results = cached_entry['results']
                        search_time = cached_time
                        
                        # Check if we have cached summaries
                        if 'day_summaries' in cached_entry:
                            day_summaries = cached_entry['day_summaries']
                        else:
                            day_summaries = {}
                        
                        # If we need to refresh current day results
                        if needs_day_refresh:
                            print(f"Search is {int(time_diff.total_seconds() / 60)} minutes old. Refreshing current day results for '{query}'")
                            updated_results, days_with_new_articles = update_current_day_results(query, count, freshness, results)
                            
                            if updated_results != results:
                                results = updated_results
                                search_time = datetime.now()
                                
                                # If we have days with new articles, clear their summaries to force regeneration
                                if days_with_new_articles and 'day_summaries' in cached_entry:
                                    print(f"Clearing summaries for days with new content: {', '.join(days_with_new_articles)}")
                                    for day in days_with_new_articles:
                                        if day in cached_entry['day_summaries']:
                                            cached_entry['day_summaries'][day] = None
                                
                                # Update the cache with refreshed results
                                cached_entry['results'] = results
                                cached_entry['timestamp'] = search_time.isoformat()
                                history[cache_key] = cached_entry
                                save_search_history(history)
                        else:
                            print(f"Using cached results for '{query}' from {cached_time}, {int(time_diff.total_seconds() / 60)} minutes old")
                        
                        # Extract articles and sort them
                        if results and 'results' in results:
                            sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
                            topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, query, day_summaries)
                            
                            # Ensure we have a Today summary, force generation if needed
                            if 'Today' not in day_summaries or not is_valid_summary(day_summaries.get('Today')):
                                # Find today's articles
                                today_articles = [a for a in sorted_articles if day_group_filter(a) == 'Today']
                                if today_articles:
                                    print(f"Force generating Today's summary for cached results '{query}'")
                                    today_summary = summarize_daily_news(today_articles, query)
                                    if is_valid_summary(today_summary):
                                        day_summaries['Today'] = today_summary
                                        cached_entry['day_summaries'] = day_summaries
                                        save_search_history(history)
                                        print(f"Successfully added Today's summary to cache: {today_summary[:50]}...")
                except Exception as e:
                    print(f"Error parsing cached timestamp: {e}")
                    use_cache = False
            
            # Make a new API call if we don't have valid cached results
            if not use_cache:
                try:
                    print(f"Making new API call for '{query}'")
                    # Search for news
                    results = brave_api.search_news(
                        query=query,
                        count=count,
                        freshness=freshness
                    )
                    search_time = datetime.now()
                    
                    # Get Reddit results if available
                    if reddit_api is not None:
                        try:
                            # Map freshness to time_filter for Reddit
                            time_filter_map = {
                                'pd': 'day',
                                'pw': 'week',
                                'pm': 'month',
                                'py': 'year',
                            }
                            reddit_time_filter = time_filter_map.get(freshness, 'week')
                            
                            # Get Reddit results
                            reddit_results = reddit_api.search(
                                query=query,
                                count=count,
                                time_filter=reddit_time_filter
                            )
                            print(f"Got {len(reddit_results.get('results', []))} Reddit results for '{query}'")
                            
                            # Merge Reddit results with Brave results
                            if reddit_results and 'results' in reddit_results and len(reddit_results['results']) > 0:
                                results['results'].extend(reddit_results['results'])
                                print(f"Added {len(reddit_results['results'])} Reddit results to search results (total: {len(results['results'])})")
                        except Exception as e:
                            print(f"Error getting Reddit results: {str(e)}")
                    
                    # Sort articles by age (newest first)
                    if results and 'results' in results:
                        sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
                        
                        # Generate topic groups and summaries
                        day_summaries = {}
                        topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, query, day_summaries)
                        
                        # Ensure we have a Today summary, force generation if needed
                        if 'Today' not in day_summaries or not is_valid_summary(day_summaries.get('Today')):
                            # Find today's articles
                            today_articles = [a for a in sorted_articles if day_group_filter(a) == 'Today']
                            if today_articles:
                                print(f"Force generating Today's summary for '{query}'")
                                today_summary = summarize_daily_news(today_articles, query)
                                if is_valid_summary(today_summary):
                                    day_summaries['Today'] = today_summary
                                    print(f"Successfully added Today's summary: {today_summary[:50]}...")
                        
                        # Cache the results with summaries
                        cache_key = f"{query}_{count}_{freshness}"
                        history[cache_key] = {
                            'query': query,
                            'count': count,
                            'freshness': freshness,
                            'timestamp': search_time.isoformat(),
                            'results': results,
                            'day_summaries': day_summaries,
                            'topic_groups': topic_groups,  # Store the topic groups for future use
                            'search_time': datetime.now().timestamp()  # Add search_time for proper sorting
                        }
                        save_search_history(history)
                        print(f"Cached new results for query: '{query}' with {len(day_summaries)} summaries")
                except Exception as e:
                    error = f"Error searching for news: {str(e)}"
    
    # Get list of history entries for sidebar
    history_entries = []
    for key, entry in history.items():
        history_entries.append({
            'query': entry['query'],
            'timestamp': entry['timestamp'],
            'key': key
        })
    
    # Sort history by most recent first
    history_entries.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('index.html', 
                          query=query,
                          results=results, 
                          search_time=search_time,
                          history_entries=history_entries,
                          error=error,
                          topic_groups=topic_groups,
                          day_summaries=day_summaries,
                          history=history,
                          active_tab="search",
                          user=current_user)

@app.route('/history', methods=['GET'])
def history():
    """View to display search history with the latest search."""
    # Redirect to raison d'etre page if not logged in
    if not current_user.is_authenticated:
        return redirect(url_for('raison_detre'))
        
    history = load_search_history()
    
    # If there are history entries, redirect to the most recent one
    history_entries = []
    for key, entry in history.items():
        history_entries.append({
            'query': entry['query'],
            'timestamp': entry['timestamp'],
            'key': key
        })
    
    # Sort history by most recent first
    history_entries.sort(key=lambda x: x['timestamp'], reverse=True)
    
    if history_entries:
        # Redirect to the most recent search
        latest_query = history_entries[0]['query']
        return redirect(url_for('history_item', query=latest_query))
    else:
        # No history yet, show empty history page
        return render_template('index.html',
                              query='',
                              results=None,
                              search_time=None,
                              error=None,
                              history_entries=history_entries,
                              topic_groups=[],
                              day_summaries={},
                              active_tab="history",
                              user=current_user)

def repair_history_file():
    """
    Repairs the history file by fixing inconsistencies in the data structure.
    Especially focuses on ensuring yesterday's data is properly preserved.
    """
    try:
        # Load the current history
        history = load_search_history()
        
        # Set to track if we made any changes
        changes_made = False
        
        # Check each entry and fix common issues
        for key, entry in history.items():
            # Ensure day_summaries exists
            if 'day_summaries' not in entry or entry['day_summaries'] is None:
                entry['day_summaries'] = {}
                print(f"Fixed missing day_summaries for entry: {entry.get('query', 'unknown')}")
                changes_made = True
                
            # Ensure results structure is complete
            if 'results' in entry and entry['results'] is not None:
                # Check that all articles have day_group information
                if 'results' in entry['results'] and entry['results']['results'] is not None:
                    articles_fixed = 0
                    articles_by_day = {}
                    
                    # Group articles by day
                    for article in entry['results']['results']:
                        # Make sure each article has an age
                        if 'age' not in article:
                            article['age'] = 'Unknown'
                            articles_fixed += 1
                            
                        # Count articles by day
                        day = day_group_filter(article)
                        if day not in articles_by_day:
                            articles_by_day[day] = 0
                        articles_by_day[day] += 1
                    
                    # Debug info
                    print(f"Entry '{entry.get('query', 'unknown')}' has articles for days: {', '.join(articles_by_day.keys())}")
                    
                    # Check if we have yesterday's data but no summary - this is a common issue
                    if 'Yesterday' in articles_by_day and ('Yesterday' not in entry['day_summaries'] or not is_valid_summary(entry['day_summaries'].get('Yesterday'))):
                        # Find yesterday's articles
                        yesterday_articles = [a for a in entry['results']['results'] if day_group_filter(a) == 'Yesterday']
                        
                        if yesterday_articles:
                            print(f"Fixing missing Yesterday summary for query: '{entry.get('query', 'unknown')}' ({len(yesterday_articles)} articles)")
                            try:
                                yesterday_summary = summarize_daily_news(yesterday_articles, entry.get('query', ''))
                                if is_valid_summary(yesterday_summary):
                                    entry['day_summaries']['Yesterday'] = yesterday_summary
                                    print(f"Generated new Yesterday summary: {yesterday_summary[:50]}...")
                                    changes_made = True
                            except Exception as e:
                                print(f"Error generating Yesterday summary: {str(e)}")
                    
                    # Check if we have any days with articles but no summaries
                    for day in articles_by_day.keys():
                        if day not in entry['day_summaries'] or not is_valid_summary(entry['day_summaries'].get(day)):
                            print(f"Missing summary for day '{day}' with {articles_by_day[day]} articles")
                    
                    if articles_fixed > 0:
                        print(f"Fixed {articles_fixed} articles with missing age data")
                        changes_made = True
        
        # Save changes if needed
        if changes_made:
            save_search_history(history)
            print("Saved repaired history file")
            return True
        else:
            print("No issues found in history file")
            return False
            
    except Exception as e:
        print(f"Error repairing history file: {str(e)}")
        return False

@app.route('/history/<path:query>', methods=['GET'])
@login_required
def history_item(query):
    """View to display a specific history item."""
    # Try to repair any inconsistencies in the history file
    repair_history_file()
    
    # Load history
    history = load_search_history()
    day_summaries = {}  # Initialize day_summaries with a default empty dictionary
    
    # Debug information
    print(f"Looking for history item with query: '{query}'")
    
    # Get list of history entries for sidebar
    history_entries = []
    for key, entry in history.items():
        stored_query = entry['query']
        history_entries.append({
            'query': stored_query,
            'timestamp': entry['timestamp'],
            'key': key
        })
    
    # Sort history by most recent first
    history_entries.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Find the requested history item
    results = None
    search_time = None
    topic_groups = []
    current_query = ''
    similarity_threshold = float(request.args.get('similarity_threshold', 0.3))
    force_refresh = request.args.get('force_refresh') == 'true'
    
    # Try to find an exact match first
    found = False
    if query:
        for key, entry in history.items():
            stored_query = entry['query']
            
            # Check for exact match
            if stored_query == query:
                print(f"Found exact match for query: '{query}'")
                found = True
                # Process the found entry
                current_query = query
                cached_time = datetime.fromisoformat(entry['timestamp'])
                time_diff = datetime.now() - cached_time
                
                # First, preserve the complete historical results
                historical_results = entry.get('results', {})
                results = historical_results
                search_time = cached_time
                count = entry.get('count', 10)
                freshness = entry.get('freshness', 'pw')
                
                # Process day summaries before any refresh to preserve them
                day_summaries = entry.get('day_summaries', {})
                if day_summaries is None:
                    day_summaries = {}
                
                # Check if we have stored topic groupings from previous runs
                stored_topic_groups = entry.get('topic_groups', [])
                
                # Print out day summaries for debugging
                if day_summaries:
                    print(f"Found existing day summaries: {', '.join(day_summaries.keys())}")
                    for day, summary in day_summaries.items():
                        is_valid = is_valid_summary(summary)
                        print(f"  - {day}: {'Valid' if is_valid else 'Invalid'} summary")
                
                # Determine if we need to refresh based on time or force flag
                needs_refresh = force_refresh or (time_diff.total_seconds() > 60 * 60)  # 1 hour
                
                # Only refresh Today's content, preserving historical data
                if needs_refresh:
                    print(f"Refreshing today's results for '{query}'")
                    # This will only update the Today group, preserving older content
                    updated_results, days_with_new_articles = update_current_day_results(query, count, freshness, results)
                    
                    if updated_results != results:
                        # Preserve the updated results while keeping historical content
                        results = updated_results
                        search_time = datetime.now()
                        
                        # If we have days with new articles, clear their summaries to force regeneration
                        if days_with_new_articles and day_summaries:
                            print(f"Clearing summaries for days with new content: {', '.join(days_with_new_articles)}")
                            for day in days_with_new_articles:
                                if day in day_summaries:
                                    day_summaries[day] = None
                    else:
                        # Even if results are the same, update the timestamp
                        search_time = datetime.now()
                    
                    # Update the cache with refreshed results while preserving historical data
                    entry['results'] = results
                    entry['timestamp'] = search_time.isoformat()
                    
                    # Ensure day_summaries is saved back to the entry
                    if 'day_summaries' not in entry or entry['day_summaries'] != day_summaries:
                        entry['day_summaries'] = day_summaries
                    
                    save_search_history(history)
                else:
                    print(f"Using cached results for '{query}' from {cached_time}, {int(time_diff.total_seconds() / 60)} minutes old")
                
                # Validate each summary to ensure we don't have invalid entries
                valid_summaries = 0
                for day, summary in list(day_summaries.items()):
                    if is_valid_summary(summary):
                        valid_summaries += 1
                    else:
                        # Clear invalid summaries so they'll be regenerated
                        day_summaries[day] = None
                
                print(f"Found {valid_summaries} valid day summaries out of {len(day_summaries)} total in cache")
                
                # Process articles - ensuring all historical articles are included
                if results and 'results' in results:
                    # First, organize articles by day to see what we're working with
                    articles_by_day = {}
                    for article in results['results']:
                        day = day_group_filter(article)
                        if day not in articles_by_day:
                            articles_by_day[day] = []
                        articles_by_day[day].append(article)
                    
                    print("Article counts by day:")
                    for day, articles in articles_by_day.items():
                        print(f"  - {day}: {len(articles)} articles")
                    
                    # OPTIMIZATION: Only regroup today's articles if needed
                    final_topic_groups = []
                    
                    # First, extract and keep all stored topic groups for days other than Today
                    if stored_topic_groups:
                        historical_topic_groups = [group for group in stored_topic_groups 
                                                 if group.get('day_group') != 'Today']
                        
                        if historical_topic_groups:
                            print(f"Reusing {len(historical_topic_groups)} stored topic groups for historical days")
                            final_topic_groups.extend(historical_topic_groups)
                    
                    # Now process Today's articles if any exist
                    if 'Today' in articles_by_day and articles_by_day['Today']:
                        print(f"Processing {len(articles_by_day['Today'])} articles for Today")
                        # Sort today's articles by age
                        today_articles = sorted(articles_by_day['Today'], key=extract_age_in_seconds)
                        # Group today's articles into topics
                        today_topic_groups = group_articles_by_topic(today_articles, 
                                                                   similarity_threshold, 
                                                                   current_query, 
                                                                   {k: v for k, v in day_summaries.items() if k == 'Today'})
                        print(f"Generated {len(today_topic_groups)} new topic groups for Today")
                        final_topic_groups.extend(today_topic_groups)
                    
                    # If we have any days that don't have stored topic groups, process them
                    days_with_stored_groups = set([group.get('day_group') for group in stored_topic_groups])
                    days_needing_grouping = [day for day in articles_by_day.keys() 
                                           if day != 'Today' and day not in days_with_stored_groups]
                    
                    if days_needing_grouping:
                        print(f"Processing {len(days_needing_grouping)} days that don't have stored topic groups: {', '.join(days_needing_grouping)}")
                        # Process each day that needs grouping
                        for day in days_needing_grouping:
                            if day in articles_by_day and articles_by_day[day]:
                                print(f"Grouping {len(articles_by_day[day])} articles for {day}")
                                day_articles = sorted(articles_by_day[day], key=extract_age_in_seconds)
                                day_topic_groups = group_articles_by_topic(day_articles, 
                                                                         similarity_threshold, 
                                                                         current_query, 
                                                                         {k: v for k, v in day_summaries.items() if k == day})
                                print(f"Generated {len(day_topic_groups)} new topic groups for {day}")
                                final_topic_groups.extend(day_topic_groups)
                    
                    # If we didn't have any stored groups or had to regenerate some days, 
                    # we need to use the old method to group everything
                    if not final_topic_groups:
                        print("No stored topic groups found, grouping all articles from scratch")
                        sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
                        final_topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, current_query, day_summaries)
                    
                    # Sort topic groups by day priority to ensure consistent ordering in the timeline
                    day_priority = {
                        "Today": 1,
                        "Yesterday": 2,
                        "2 days ago": 3,
                        "3 days ago": 4,
                        "4 days ago": 5,
                        "5 days ago": 6,
                        "6 days ago": 7,
                        "1 week ago": 8,
                        "2 weeks ago": 9,
                        "1 month ago": 10
                    }
                    
                    final_topic_groups = sorted(final_topic_groups, key=lambda x: (
                        day_priority.get(x['day_group'], 999),  # First sort by day priority
                        extract_age_in_seconds(x['articles'][0]) if x.get('articles') else 0  # Then by article age within the day
                    ))
                    
                    # Store the topic groups for future use
                    entry['topic_groups'] = final_topic_groups
                    save_search_history(history)
                    print(f"Saved {len(final_topic_groups)} topic groups to history for future use")
                    
                    # Set the topic groups for rendering
                    topic_groups = final_topic_groups
                    
                    # Save the updated day_summaries back to the entry
                    if 'day_summaries' not in entry or entry['day_summaries'] != day_summaries:
                        entry['day_summaries'] = day_summaries
                        save_search_history(history)
                
                break
    
    # If no exact match was found, perform a new search
    if not found:
        print(f"No exact match found for query: '{query}'. Performing new search.")
        try:
            # Initialize API if needed
            global brave_api
            global reddit_api
            if brave_api is None:
                brave_api_key = app.config.get("BRAVE_API_KEY")
                if brave_api_key:
                    brave_api = BraveNewsAPI(api_key=brave_api_key)
                else:
                    return render_template('error.html', error="BRAVE_API_KEY not set", user=current_user)
            
            # Initialize Reddit API if needed
            if reddit_api is None:
                try:
                    reddit_api = RedditAPI()
                    reddit_api.init()
                except Exception as e:
                    print(f"Warning: Could not initialize Reddit API: {str(e)}")
                    print("Reddit content will not be available")
            
            # Default values
            count = 10
            freshness = 'pw'
            
            # Make a new API call for news
            print(f"Making new API call for '{query}'")
            results = brave_api.search_news(
                query=query,
                count=count,
                freshness=freshness
            )
            search_time = datetime.now()
            current_query = query
            
            # Get Reddit results if available
            if reddit_api is not None:
                try:
                    # Map freshness to time_filter for Reddit
                    time_filter_map = {
                        'pd': 'day',
                        'pw': 'week',
                        'pm': 'month',
                        'py': 'year',
                    }
                    reddit_time_filter = time_filter_map.get(freshness, 'week')
                    
                    # Get Reddit results
                    reddit_results = reddit_api.search(
                        query=query,
                        count=count,
                        time_filter=reddit_time_filter
                    )
                    print(f"Got {len(reddit_results.get('results', []))} Reddit results for '{query}'")
                    
                    # Merge Reddit results with Brave results
                    if reddit_results and 'results' in reddit_results and len(reddit_results['results']) > 0:
                        results['results'].extend(reddit_results['results'])
                        print(f"Added {len(reddit_results['results'])} Reddit results to search results (total: {len(results['results'])})")
                except Exception as e:
                    print(f"Error getting Reddit results: {str(e)}")
            
            # Generate day summaries
            sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
            day_summaries = {}
            
            # Group articles by day
            articles_by_day = {}
            
            # Sort topic groups by day priority to ensure consistent ordering in the timeline
            day_priority = {
                "Today": 1,
                "Yesterday": 2,
                "2 days ago": 3,
                "3 days ago": 4,
                "4 days ago": 5,
                "5 days ago": 6,
                "6 days ago": 7,
                "1 week ago": 8,
                "2 weeks ago": 9,
                "1 month ago": 10
            }
            
            topic_groups = sorted(topic_groups, key=lambda x: (
                day_priority.get(x['day_group'], 999),  # First sort by day priority
                extract_age_in_seconds(x['articles'][0]) if x.get('articles') else 0  # Then by article age within the day
            ))
            
            # Cache the results
            cache_key = f"{query}_{count}_{freshness}"
            history[cache_key] = {
                'query': query,
                'count': count,
                'freshness': freshness,
                'timestamp': search_time.isoformat(),
                'results': results,
                'day_summaries': day_summaries,
                'topic_groups': topic_groups,  # Store the topic groups for future use
                'search_time': datetime.now().timestamp()  # Add search_time for proper sorting
            }
            save_search_history(history)
            print(f"Cached new results for query: '{query}' with {len(day_summaries)} summaries")
        except Exception as e:
            print(f"Error performing search: {str(e)}")
            # Ensure day_summaries is still defined in case of error
            day_summaries = {}
    
    # Ensure we always have a search_time for the ticker to work
    if search_time is None:
        search_time = datetime.now()
    
    return render_template('index.html', 
                          query=current_query, 
                          results=results, 
                          search_time=search_time,
                          error=None,
                          history_entries=history_entries,
                          topic_groups=topic_groups,
                          day_summaries=day_summaries,
                          history=history,
                          active_tab="history",
                          user=current_user)

@app.route('/api/delete_history/<path:query>', methods=['POST'])
@login_required
def delete_history_item(query):
    """API endpoint to delete a history item."""
    history = load_search_history()
    
    # Find and remove the history item
    items_to_remove = []
    for key, entry in history.items():
        if entry['query'] == query:
            items_to_remove.append(key)
    
    for key in items_to_remove:
        history.pop(key, None)
    
    save_search_history(history)
    
    return jsonify({'success': True})

@app.route('/api/clear_history', methods=['POST'])
@login_required
def clear_history():
    """API endpoint to clear all history."""
    save_search_history({})
    return jsonify({'success': True})

def clean_text(text):
    """Clean and preprocess text for similarity comparison."""
    if not text:
        return ""
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_summary(summary):
    """Check if a summary exists and is valid."""
    if summary is None:
        return False
    if not isinstance(summary, str):
        return False
    if len(summary.strip()) == 0:
        return False
    return True

def group_articles_by_topic_openai(articles, query=""):
    """
    Group articles into topics based on content similarity using OpenAI.
    NOTE: This function should only be called with articles from a single day
    to maintain day boundary enforcement.
    
    Args:
        articles: List of news articles (all from the SAME DAY)
        query: The search query used (for contextual grouping)
        
    Returns:
        Dictionary with topic groups and their titles/summaries
    """
    if not OPENAI_AVAILABLE or not articles:
        return None
    
    try:
        # Check that all articles are from the same day
        if len(articles) > 0:
            # Get the day of the first article as reference
            expected_day = day_group_filter(articles[0])
            
            # Verify all articles are from the same day
            for article in articles:
                article_day = day_group_filter(article)
                if article_day != expected_day:
                    print(f"WARNING: group_articles_by_topic_openai received articles from different days: {article_day} vs {expected_day}")
                    # We'll continue processing but enforce the day in the result
        
        # Prepare article data for OpenAI
        article_data = []
        for idx, article in enumerate(articles):
            title = article.get('title', '')
            desc = article.get('description', '')
            source = article.get('meta_url', {}).get('netloc', 'Unknown source')
            age = article.get('age', 'Unknown')
            
            article_data.append({
                "id": idx,
                "title": title, 
                "description": desc,
                "source": source,
                "age": age
            })
        
        # Create the prompt for OpenAI
        messages = [
            {"role": "system", "content": """You are an expert at organizing news articles into coherent topic groups. 
Your task is to:
1. Group similar articles into distinct topics
2. Give each topic a concise, descriptive title (max 10 words)
3. Write a brief 1-sentence summary for each topic group
4. Place each article in exactly one group
5. Create more restrictive themes rather than broad categories"""},
            {"role": "user", "content": f"""Here are news articles about "{query}".
Please group them into topics, with a concise title and brief summary for each group.
Articles: {json.dumps(article_data, indent=2)}

Return JSON in this exact format:
{{
  "topic_groups": [
    {{
      "title": "Concise topic title",
      "summary": "Brief one-sentence summary of what this topic group is about",
      "article_ids": [0, 3, 5]
    }},
    ...
  ]
}}"""}
        ]
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o-mini", # Use gpt-4.1-nano when available
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Extract the response content
        result = json.loads(response.choices[0].message.content)
        
        # Process the groups
        topic_groups = []
        
        # Get the common day for all articles - use the day of the first article
        common_day = day_group_filter(articles[0]) if articles else "Unknown"
        
        for group in result.get("topic_groups", []):
            article_ids = group.get("article_ids", [])
            if not article_ids:
                continue
                
            # Collect the articles for this group
            group_articles = [articles[idx] for idx in article_ids if idx < len(articles)]
            if not group_articles:
                continue
                
            # Sort the articles by age
            group_articles = sorted(group_articles, key=extract_age_in_seconds)
            
            # Create the topic group entry
            newest_article = group_articles[0]
            topic_groups.append({
                'title': group.get("title", "Untitled Topic"),
                'summary': group.get("summary", ""),
                'articles': group_articles,
                'count': len(group_articles),
                'newest_age': newest_article.get('age', 'Unknown'),
                'day_group': common_day  # Explicitly set to maintain day boundary
            })
        
        # Sort topic groups by the age of their newest article
        topic_groups = sorted(topic_groups, key=lambda x: extract_age_in_seconds(x['articles'][0]))
        
        return topic_groups
        
    except Exception as e:
        print(f"Error grouping articles with OpenAI: {str(e)}")
        return None

def group_articles_by_topic(articles, similarity_threshold=0.3, query="", day_summaries=None):
    """
    Group articles into topics based on content similarity.
    CRITICAL: Enforce day boundaries - never mix articles from different days in the same topic.
    
    Args:
        articles: List of news articles or list of lists where each inner list contains articles from one day
        similarity_threshold: Threshold for considering articles as similar (0-1)
        query: The search query used (for summarization)
        day_summaries: Dictionary of existing summaries by day
        
    Returns:
        List of topic groups, each containing related articles
    """
    if not articles:
        return []
    
    # Initialize day_summaries if not provided
    if day_summaries is None:
        day_summaries = {}
    
    # OPTIMIZATION: Handle the case where articles are already grouped by day
    if isinstance(articles, list) and len(articles) > 0 and isinstance(articles[0], list):
        print("Received pre-grouped articles by day, using optimized processing")
        day_articles_list = articles
        # Flatten the list of articles for day detection
        flat_articles = []
        for day_articles in day_articles_list:
            flat_articles.extend(day_articles)
        articles = flat_articles
    else:
        # Original logic: organize articles by day
        day_articles_list = []
    
    # CRITICAL CHANGE: First, organize articles by day to enforce day boundaries
    day_grouped_articles = {}
    for article in articles:
        day = day_group_filter(article)
        if day not in day_grouped_articles:
            day_grouped_articles[day] = []
        day_grouped_articles[day].append(article)
    
    # Sort days in order of recency (Today, Yesterday, etc.)
    day_priority = {
        "Today": 1,
        "Yesterday": 2,
        "2 days ago": 3,
        "3 days ago": 4,
        "4 days ago": 5,
        "5 days ago": 6,
        "6 days ago": 7,
        "1 week ago": 8,
        "2 weeks ago": 9,
        "1 month ago": 10
    }
    sorted_days = sorted(day_grouped_articles.keys(), key=lambda d: day_priority.get(d, 999))
    
    # Print day distribution for debugging
    print(f"Articles by day before grouping:")
    for day in sorted_days:
        print(f"  - {day}: {len(day_grouped_articles[day])} articles")
    
    # Initialize final topic groups list - will contain topics from all days
    final_topic_groups = []
    
    # Try to use OpenAI for more intelligent grouping if available
    # But still enforce day boundaries by processing each day separately
    if OPENAI_AVAILABLE:
        print("Using OpenAI for topic grouping, enforcing day boundaries...")
        
        # Process each day separately with OpenAI to maintain day boundaries
        for day in sorted_days:
            day_articles = day_grouped_articles[day]
            if not day_articles:
                continue
                
            print(f"Processing {len(day_articles)} articles for day: {day}")
            
            # Use OpenAI to group just this day's articles
            day_openai_groups = group_articles_by_topic_openai(day_articles, query)
            
            if day_openai_groups:
                # Set the day group explicitly for all topics from this day
                for topic in day_openai_groups:
                    topic['day_group'] = day
                    
                    # Add the day's summary to all topics from this day
                    existing_summary = day_summaries.get(day)
                    topic['day_summary'] = existing_summary
                
                # Add this day's topics to our final list
                final_topic_groups.extend(day_openai_groups)
                print(f"Added {len(day_openai_groups)} OpenAI-grouped topics for {day}")
            else:
                # Fallback to simple grouping if OpenAI failed for this day
                print(f"OpenAI grouping failed for {day}, falling back to simple grouping")
                
                # Create a single topic for each article as a fallback
                for article in day_articles:
                    final_topic_groups.append({
                        'title': article.get('title', 'Untitled Topic'),
                        'articles': [article],
                        'count': 1,
                        'newest_age': article.get('age', 'Unknown'),
                        'day_group': day,
                        'day_summary': day_summaries.get(day)
                    })
        
        # If we successfully created any topics with OpenAI, process summaries and return
        if final_topic_groups:
            # Process day summaries - generate any that are missing
            days_needing_summaries = set()
            for topic in final_topic_groups:
                day = topic['day_group']
                if not is_valid_summary(topic.get('day_summary')):
                    if day not in days_needing_summaries:
                        days_needing_summaries.add(day)
            
            if days_needing_summaries and OPENAI_AVAILABLE:
                for day in days_needing_summaries:
                    if day in day_summaries and is_valid_summary(day_summaries[day]):
                        print(f"Skipping summary generation for day: {day} (already valid)")
                        continue
                        
                    print(f"Generating new summary for day: {day} using OpenAI")
                    day_articles = day_grouped_articles.get(day, [])
                    if not day_articles:
                        continue
                        
                    summary = summarize_daily_news(day_articles, query)
                    
                    if is_valid_summary(summary):
                        print(f"Successfully generated summary for {day}: {summary[:50]}...")
                        day_summaries[day] = summary
                        
                        # Update all topics from this day with the new summary
                        for topic in final_topic_groups:
                            if topic['day_group'] == day:
                                topic['day_summary'] = summary
                        
                        try:
                            # Save summaries to history
                            history = load_search_history()
                            for key, entry in history.items():
                                if entry.get('query') == query:
                                    entry['day_summaries'] = day_summaries
                                    save_search_history(history)
                                    print(f"Saved new {day} summary to history for {query}")
                                    break
                        except Exception as e:
                            print(f"Error saving summary to history: {e}")
            
            topics_with_summaries = sum(1 for topic in final_topic_groups if is_valid_summary(topic.get('day_summary')))
            print(f"Created {len(final_topic_groups)} total topics across all days, {topics_with_summaries} have summaries.")
            
            # Return the final list with all days' topics
            return final_topic_groups
    
    # Fallback to TF-IDF based approach - processing each day separately
    print("Using TF-IDF for topic grouping, enforcing day boundaries...")
    
    # Process each day separately with TF-IDF to maintain day boundaries
    for day in sorted_days:
        day_articles = day_grouped_articles[day]
        if not day_articles:
            continue
            
        print(f"Processing {len(day_articles)} articles for day: {day}")
        
        # Extract title and description for content comparison - only for this day's articles
        article_contents = []
        for article in day_articles:
            title = article.get('title', '')
            desc = article.get('description', '')
            content = f"{title} {desc}"
            article_contents.append(clean_text(content))
        
        # Compute TF-IDF vectorization for this day's articles
        try:
            day_topic_groups = []
            
            if len(day_articles) > 1:
                vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(article_contents)
                
                # Compute cosine similarity
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Group articles based on similarity - only within this day
                visited = [False] * len(day_articles)
                
                for i in range(len(day_articles)):
                    if visited[i]:
                        continue
                        
                    visited[i] = True
                    group = [day_articles[i]]
                    
                    # Find similar articles - only looking at this day's articles
                    for j in range(i+1, len(day_articles)):
                        if not visited[j] and similarity_matrix[i, j] >= similarity_threshold:
                            group.append(day_articles[j])
                            visited[j] = True
                    
                    # Create a topic for this group - explicitly setting the day
                    group = sorted(group, key=extract_age_in_seconds)
                    newest_article = group[0]
                    topic_title = newest_article.get('title', 'Untitled Topic')
                    
                    day_topic_groups.append({
                        'title': topic_title,
                        'articles': group,
                        'count': len(group),
                        'newest_age': newest_article.get('age', 'Unknown'),
                        'day_group': day  # Explicitly set the day to maintain boundaries
                    })
            else:
                # For a single article, just add it as its own topic
                for article in day_articles:
                    day_topic_groups.append({
                        'title': article.get('title', 'Untitled Topic'),
                        'articles': [article],
                        'count': 1,
                        'newest_age': article.get('age', 'Unknown'),
                        'day_group': day  # Explicitly set the day
                    })
            
            # Add existing summary for this day to all topics from this day
            existing_summary = day_summaries.get(day)
            for topic in day_topic_groups:
                topic['day_summary'] = existing_summary
            
            # Add this day's topics to our final list
            final_topic_groups.extend(day_topic_groups)
            print(f"Added {len(day_topic_groups)} TF-IDF grouped topics for {day}")
            
        except Exception as e:
            print(f"Error during TF-IDF grouping for day {day}: {str(e)}")
            # Fallback to simple grouping without TF-IDF
            for article in day_articles:
                final_topic_groups.append({
                    'title': article.get('title', 'Untitled Topic'),
                    'articles': [article],
                    'count': 1,
                    'newest_age': article.get('age', 'Unknown'),
                    'day_group': day,
                    'day_summary': day_summaries.get(day)
                })
    
    # Process day summaries - generate any that are missing
    days_needing_summaries = set()
    for topic in final_topic_groups:
        day = topic['day_group']
        if not is_valid_summary(topic.get('day_summary')):
            if day not in days_needing_summaries:
                days_needing_summaries.add(day)
    
    if days_needing_summaries and OPENAI_AVAILABLE:
        for day in days_needing_summaries:
            if day in day_summaries and is_valid_summary(day_summaries[day]):
                print(f"Skipping summary generation for day: {day} (already valid)")
                continue
                
            print(f"Generating new summary for day: {day} using OpenAI")
            day_articles = day_grouped_articles.get(day, [])
            if not day_articles:
                continue
                
            summary = summarize_daily_news(day_articles, query)
            
            if is_valid_summary(summary):
                print(f"Successfully generated summary for {day}: {summary[:50]}...")
                day_summaries[day] = summary
                
                # Update all topics from this day with the new summary
                for topic in final_topic_groups:
                    if topic['day_group'] == day:
                        topic['day_summary'] = summary
                
                try:
                    # Save summaries to history
                    history = load_search_history()
                    for key, entry in history.items():
                        if entry.get('query') == query:
                            entry['day_summaries'] = day_summaries
                            save_search_history(history)
                            print(f"Saved new {day} summary to history for {query}")
                            break
                except Exception as e:
                    print(f"Error saving summary to history: {e}")
    
    topics_with_summaries = sum(1 for topic in final_topic_groups if is_valid_summary(topic.get('day_summary')))
    print(f"Created {len(final_topic_groups)} total topics across all days, {topics_with_summaries} have summaries.")
    
    return final_topic_groups

# Function to extract age in seconds for sorting
def extract_age_in_seconds(article):
    """Convert age strings to seconds for sorting"""
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

# Template filter to get the day group for an article
@app.template_filter('day_group')
def day_group_filter(article):
    """Get the day group for an article (Today, Yesterday, etc.)"""
    age = article.get("age", "Unknown")
    
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
        
    return day_str

# Template filter to format dates
@app.template_filter('format_datetime')
def format_datetime_filter(value, format='%Y-%m-%d %H:%M'):
    """Format a datetime object using Eastern Time."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    
    # Convert to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    
    # If datetime is naive (no timezone), assume it's in UTC
    if value.tzinfo is None:
        value = pytz.utc.localize(value)
    
    # Convert to Eastern Time
    value_eastern = value.astimezone(eastern)
    
    # Format with EST suffix if format includes date elements
    if any(x in format for x in ['%Y', '%m', '%d', '%b']):
        return f"{value_eastern.strftime(format)} EST"
    else:
        return value_eastern.strftime(format)

# Make the is_valid_summary function available in templates
app.jinja_env.globals['is_valid_summary'] = is_valid_summary
# Flask automatically makes get_flashed_messages available in templates, no need to add it manually

@app.route('/request-access', methods=['GET', 'POST'])
def request_access_redirect():
    """Handle the request access form submission"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        
        # Very basic validation
        if not email or '@' not in email:
            flash('Please enter a valid email address.', 'error')
            return redirect(url_for('index'))
            
        # Use the function from auth.py to add the email to the waitlist
        from auth import add_email_to_waitlist
        success = add_email_to_waitlist(email)
        
        if success:
            flash('Your access request has been submitted. You will be notified when access is granted.', 'success')
        else:
            flash('Your email is already in our waitlist. Please wait for approval.', 'info')
            
        return redirect(url_for('index'))
    
    # GET requests just redirect back to the homepage
    return redirect(url_for('index'))

@app.route('/raison-detre', methods=['GET'])
def raison_detre():
    """Display the raison d'être page"""
    # Get history entries for sidebar if needed
    history_entries = []
    history = load_search_history()
    
    if current_user.is_authenticated:
        for key, entry in history.items():
            history_entries.append(entry)
        # Sort by timestamp descending
        history_entries.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    
    return render_template('index.html',
        active_tab='raison-detre',
        user=current_user,
        history_entries=history_entries,
        history=history,
        query=None,
        error=None,
        philosophy_content={
            'heading': 'Welcome to Loop',
            'subheading': 'Track all moves in a single timeline',
            'intro': 'Loop is a modern news tracking application designed to help you stay informed on topics that matter to you. It organizes news articles by topic and presents them in a chronological timeline for easy consumption.',
            'sections': [
                {
                    'title': 'Why We Built This',
                    'content': 'In today\'s fast-paced world, staying informed becomes increasingly difficult. Traditional news sources either overwhelm with information or miss critical context. Loop addresses this by focusing on specific topics you care about, organizing news chronologically, and providing AI-generated summaries to give you the big picture at a glance.'
                },
                {
                    'title': 'How It Works',
                    'content': 'Loop lets you track up to 3 topics simultaneously. For each topic, we gather the latest news, organize articles by day, and generate concise summaries. The timeline view shows you how stories evolve over time, helping you understand not just what happened, but how events unfolded and are connected.'
                },
                {
                    'title': 'Limited Access Model',
                    'content': 'We currently operate on an invitation-based model to ensure quality of service. If you\'d like to use Loop, please request access using the link in the navigation bar. We\'ll notify you when your account is approved.'
                }
            ]
        }
    )

def save_notification_settings(user_email, topic, frequency):
    """Save notification settings to the user's history file"""
    print(f"Attempting to save notification settings: {user_email}, {topic}, {frequency}")
    
    # Try to find the topic in any history file
    found = False
    user_dir = 'user_data'
    
    # Generate email-based path for this user
    safe_email = user_email.replace("@", "_at_").replace(".", "_dot_")
    user_specific_dir = os.path.join(user_dir, safe_email)
    os.makedirs(user_specific_dir, exist_ok=True)
    
    user_history_file = os.path.join(user_specific_dir, 'search_history.json')
    
    # First check if this user already has a history file
    if os.path.exists(user_history_file):
        try:
            with open(user_history_file, 'r') as f:
                history = json.load(f)
                
            # Look for the topic in this user's history
            for key, entry in history.items():
                if entry.get('query') == topic:
                    # Create or update notification settings
                    if 'notifications' not in entry:
                        entry['notifications'] = {
                            'recipients': [],
                            'last_sent': {}
                        }
                    
                    # Make sure notifications has the correct structure
                    if 'recipients' not in entry['notifications']:
                        entry['notifications']['recipients'] = []
                        
                    if 'last_sent' not in entry['notifications']:
                        entry['notifications']['last_sent'] = {}
                    
                    # Check if this email is already in recipients
                    recipient_exists = False
                    for recipient in entry['notifications']['recipients']:
                        if recipient['email'] == user_email:
                            # Update existing recipient
                            recipient['frequency'] = frequency
                            recipient_exists = True
                            break
                    
                    # Add new recipient if not found
                    if not recipient_exists:
                        entry['notifications']['recipients'].append({
                            'email': user_email,
                            'frequency': frequency
                        })
                    
                    # Save updated history
                    with open(user_history_file, 'w') as f:
                        json.dump(history, f)
                    
                    print(f"Saved notification settings for '{topic}' to {user_email}: {frequency} in {user_history_file}")
                    return True
                    
            print(f"Topic {topic} not found in user history file: {user_history_file}")
        except Exception as e:
            print(f"Error reading/writing user history file: {e}")
    else:
        print(f"User history file does not exist: {user_history_file}")
    
    # If we get here, either the user has no history file or the topic isn't in their history
    # Look in the shared history file
    shared_history_file = 'user_data/shared/search_history.json'
    if os.path.exists(shared_history_file):
        try:
            with open(shared_history_file, 'r') as f:
                history = json.load(f)
                
            # Look for the topic in shared history
            for key, entry in history.items():
                if entry.get('query') == topic:
                    # Create or update notification settings
                    if 'notifications' not in entry:
                        entry['notifications'] = {
                            'recipients': [],
                            'last_sent': {}
                        }
                    
                    # Make sure notifications has the correct structure
                    if 'recipients' not in entry['notifications']:
                        entry['notifications']['recipients'] = []
                        
                    if 'last_sent' not in entry['notifications']:
                        entry['notifications']['last_sent'] = {}
                    
                    # Check if this email is already in recipients
                    recipient_exists = False
                    for recipient in entry['notifications']['recipients']:
                        if recipient['email'] == user_email:
                            # Update existing recipient
                            recipient['frequency'] = frequency
                            recipient_exists = True
                            break
                    
                    # Add new recipient if not found
                    if not recipient_exists:
                        entry['notifications']['recipients'].append({
                            'email': user_email,
                            'frequency': frequency
                        })
                    
                    # Save updated history to shared file
                    with open(shared_history_file, 'w') as f:
                        json.dump(history, f)
                    
                    print(f"Saved notification settings for '{topic}' to {user_email}: {frequency} in shared history")
                    return True
                    
            print(f"Topic {topic} not found in shared history file")
        except Exception as e:
            print(f"Error reading/writing shared history file: {e}")
    
    # If we couldn't find the topic in any file, return failure
    print(f"Failed to save notification settings: topic {topic} not found in any history file")
    return False

def check_and_send_notifications():
    """Check for due notifications and send emails by scanning all history files"""
    updated_files = {}
    
    # Current time in Eastern Time
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # First, scan the user_data directory for all history files
    user_dir = 'user_data'
    
    # Process the shared history file
    process_history_file('user_data/shared/search_history.json', now, eastern, updated_files)
    
    # Process user-specific history files
    for item in os.listdir(user_dir):
        item_path = os.path.join(user_dir, item)
        # Skip files, we want directories that might contain user data
        if not os.path.isdir(item_path) or item == 'shared' or item == 'anonymous':
            continue
            
        # Look for search_history.json in this user directory
        history_file = os.path.join(item_path, 'search_history.json')
        if os.path.exists(history_file):
            process_history_file(history_file, now, eastern, updated_files)
    
    # Save any updated history files
    for file_path, history in updated_files.items():
        try:
            with open(file_path, 'w') as f:
                json.dump(history, f)
            print(f"Updated history file: {file_path}")
        except Exception as e:
            print(f"Error saving history file {file_path}: {e}")

def process_history_file(file_path, now, eastern, updated_files):
    """Process a single history file for notifications"""
    if not os.path.exists(file_path):
        return
        
    try:
        with open(file_path, 'r') as f:
            history = json.load(f)
    except Exception as e:
        print(f"Error loading history file {file_path}: {e}")
        return
    
    updated = False
    global brave_api
    
    # Initialize Brave API if needed
    if brave_api is None:
        brave_api_key = app.config.get("BRAVE_API_KEY")
        if not brave_api_key:
            print("Error: BRAVE_API_KEY not set, cannot refresh content")
            return
        brave_api = BraveNewsAPI(api_key=brave_api_key)
    
    # Process each entry in this history file
    for key, entry in history.items():
        if 'notifications' not in entry or 'query' not in entry:
            continue
        
        notifications = entry['notifications']
        if 'recipients' not in notifications or not notifications['recipients']:
            continue
        
        # Get topic query, count and freshness
        topic = entry.get('query')
        count = entry.get('count', 10)
        freshness = entry.get('freshness', 'pw')
        
        # Process each recipient
        for recipient in notifications['recipients']:
            email = recipient.get('email')
            frequency = recipient.get('frequency')
            
            if not email or not frequency:
                continue
            
            # Get last sent time for this recipient
            last_sent_dict = notifications.get('last_sent')
            
            # Initialize last_sent_dict if it's None
            if last_sent_dict is None:
                last_sent_dict = {}
                notifications['last_sent'] = last_sent_dict
            
            last_sent_str = last_sent_dict.get(email)
            
            should_send = False
            
            # Check if we should send based on frequency
            if not last_sent_str:
                # Never sent before
                should_send = True
            else:
                try:
                    last_sent = datetime.fromisoformat(last_sent_str)
                    
                    # If datetime is naive (no timezone), assume it's in UTC
                    if last_sent.tzinfo is None:
                        last_sent = pytz.utc.localize(last_sent)
                    
                    # Convert to Eastern Time
                    last_sent = last_sent.astimezone(eastern)
                    
                    if frequency == 'hourly':
                        # Send if it's been more than an hour
                        diff = now - last_sent
                        if diff.total_seconds() >= 3600:  # 1 hour
                            should_send = True
                    elif frequency == 'daily':
                        # Send if it's been more than a day
                        diff = now - last_sent
                        if diff.total_seconds() >= 86400:  # 24 hours
                            should_send = True
                except Exception as e:
                    print(f"Error parsing last sent time: {e}")
                    should_send = True
            
            if should_send:
                print(f"Time to send {frequency} notification for '{topic}' to {email}")
                
                # Variables to track what's new
                has_new_content = False
                new_articles = []
                previous_today_summary = None
                current_today_summary = None
                
                # Store the previous Today summary if it exists
                if 'day_summaries' in entry and 'Today' in entry['day_summaries']:
                    previous_today_summary = entry['day_summaries'].get('Today')
                
                # Refresh the content before sending notification
                print(f"Refreshing content for '{topic}' before sending notification")
                results = entry.get('results')
                
                if results:
                    # Get the current set of URLs to compare later
                    current_urls = set()
                    if 'results' in results:
                        for article in results['results']:
                            if 'url' in article:
                                current_urls.add(article['url'])
                    
                    # Update the current day results to get fresh content
                    updated_results, days_with_new_articles = update_current_day_results(topic, count, freshness, results)
                    
                    # Check if we got new content
                    content_updated = updated_results != results
                    
                    if content_updated:
                        # We got new content, update the entry
                        entry['results'] = updated_results
                        entry['timestamp'] = datetime.now().isoformat()
                        
                        # Find new articles by comparing URLs
                        if 'results' in updated_results:
                            for article in sorted(updated_results['results'], key=extract_age_in_seconds):
                                if 'url' in article and article['url'] not in current_urls:
                                    new_articles.append(article)
                                    has_new_content = True
                            
                            # Cap to top 3 newest articles
                            new_articles = new_articles[:3]
                            
                            if new_articles:
                                print(f"Found {len(new_articles)} new articles for '{topic}'")
                        
                        # Regenerate summaries for days with new content
                        if 'day_summaries' not in entry:
                            entry['day_summaries'] = {}
                            
                        # Process articles
                        if updated_results and 'results' in updated_results:
                            sorted_articles = sorted(updated_results['results'], key=extract_age_in_seconds)
                            
                            # Group articles by day
                            articles_by_day = {}
                            for article in sorted_articles:
                                day = day_group_filter(article)
                                if day not in articles_by_day:
                                    articles_by_day[day] = []
                                articles_by_day[day].append(article)
                            
                            # Generate or refresh summaries for days
                            for day, day_articles in articles_by_day.items():
                                # Always refresh Today's summary, or if this day has new articles
                                if day == 'Today' or day in days_with_new_articles or not is_valid_summary(entry['day_summaries'].get(day)):
                                    print(f"Generating new summary for {day} for topic '{topic}'")
                                    day_summary = summarize_daily_news(day_articles, topic)
                                    if is_valid_summary(day_summary):
                                        entry['day_summaries'][day] = day_summary
                                        print(f"Generated new {day} summary: {day_summary[:50]}...")
                                        
                                        # For Today's summary, check if it's different from previous
                                        if day == 'Today':
                                            current_today_summary = day_summary
                                            if previous_today_summary != current_today_summary:
                                                print("Today's summary has changed")
                                                has_new_content = True
                
                # For hourly emails, only send if there's new content
                should_actually_send = True
                if frequency == 'hourly' and not has_new_content:
                    print(f"No new content for '{topic}', skipping hourly notification to {email}")
                    should_actually_send = False
                
                # Now send the notification with the fresh content if appropriate
                if should_actually_send and send_notification_email(topic, entry, frequency, email, new_articles):
                    # Update last sent timestamp for this recipient
                    current_time = datetime.now().isoformat()
                    print(f"Successfully sent notification for '{topic}' to {email}, updating last_sent to {current_time}")
                    
                    if 'last_sent' not in notifications:
                        notifications['last_sent'] = {}
                    
                    # Update the timestamp to current time
                    notifications['last_sent'][email] = current_time
                    updated = True
    
    # Mark this file for update if changes were made
    if updated:
        updated_files[file_path] = history

def send_notification_email(topic, entry, frequency, recipient_email, new_articles=None):
    """Send notification email for a specific topic"""
    if not recipient_email:
        print(f"Error: No recipient email provided")
        return False
    
    print(f"Preparing to send {frequency} notification email for topic '{topic}' to {recipient_email}")
    
    # Display content timestamp to debug freshness
    content_timestamp = entry.get('timestamp')
    if content_timestamp:
        try:
            content_time = datetime.fromisoformat(content_timestamp)
            age_minutes = (datetime.now() - content_time).total_seconds() / 60
            print(f"Content timestamp: {content_timestamp} (age: {int(age_minutes)} minutes)")
        except Exception as e:
            print(f"Error parsing content timestamp: {e}")
    
    # Check if we have summaries
    day_summaries = entry.get('day_summaries', {})
    if not day_summaries:
        print(f"Error: No day summaries available for topic '{topic}'")
        return False
    
    print(f"Available day summaries: {list(day_summaries.keys())}")
    
    # For hourly, use today's summary; for daily, use yesterday's
    summary_text = None
    summary_date = None
    
    # Get current time in Eastern Time
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    date_str = now.strftime('%b %d, %Y')
    
    # For daily summaries using yesterday's content, use yesterday's date
    yesterday_date = None
    if frequency == 'daily':
        yesterday = now - timedelta(days=1)
        yesterday_date = yesterday.strftime('%b %d, %Y')
    
    if frequency == 'hourly':
        if 'Today' in day_summaries and is_valid_summary(day_summaries['Today']):
            summary_text = day_summaries['Today']
            summary_date = 'Today'
            print(f"Using Today's summary for hourly notification")
    elif frequency == 'daily':
        if 'Yesterday' in day_summaries and is_valid_summary(day_summaries['Yesterday']):
            summary_text = day_summaries['Yesterday']
            summary_date = 'Yesterday'
            print(f"Using Yesterday's summary for daily notification")
    
    # If no appropriate summary found, try other days
    if not summary_text:
        print(f"No specific {frequency} summary found, looking for any valid summary")
        for day, summary in day_summaries.items():
            if is_valid_summary(summary):
                summary_text = summary
                summary_date = day
                print(f"Using {day}'s summary instead")
                break
    
    if not summary_text:
        print(f"Error: No valid summary found for topic '{topic}'")
        return False
    
    # Create email
    msg = MIMEMultipart()
    msg['From'] = "Loop News <noreply@loop-news.com>"
    msg['To'] = recipient_email
    
    if frequency == 'hourly':
        msg['Subject'] = f"Hourly Update: {topic} - {date_str} EST"
    else:
        msg['Subject'] = f"Daily Summary: {topic} - {date_str} EST"
    
    # Determine which date to display in the email
    email_date_str = date_str
    if frequency == 'daily' and summary_date == 'Yesterday' and yesterday_date:
        email_date_str = yesterday_date
    
    # Create HTML for new articles section (for hourly emails)
    new_articles_html = ""
    if frequency == 'hourly' and new_articles:
        new_articles_html = f"""
        <div style="margin-bottom: 30px; padding: 15px; background-color: #f8f8f8; border-left: 4px solid #555;">
          <h2 style="font-size: 18px; color: #555; margin-bottom: 15px;">Latest Updates ({len(new_articles)} new article{'s' if len(new_articles) > 1 else ''})</h2>
          <ul style="padding-left: 20px; margin-bottom: 0;">
        """
        
        for article in new_articles:
            title = article.get('title', 'No title')
            description = article.get('description', 'No description available')
            url = article.get('url', '#')
            age = article.get('age', 'Unknown time')
            source = article.get('meta_url', {}).get('netloc', 'Unknown source')
            
            new_articles_html += f"""
            <li style="margin-bottom: 15px;">
              <a href="{url}" style="font-weight: bold; color: #333; text-decoration: none;">{title}</a>
              <div style="margin-top: 5px; color: #555; font-size: 13px;">
                {source} • {age}
              </div>
              <div style="margin-top: 5px; line-height: 1.4;">
                {description[:150]}{'...' if len(description) > 150 else ''}
              </div>
            </li>
            """
        
        new_articles_html += """
          </ul>
        </div>
        """
    
    # Email body - Use render_template
    html = render_template('notification_email.html',
                           topic=topic,
                           summary_date=summary_date,
                           email_date_str=email_date_str,
                           summary_text=summary_text,
                           new_articles_html=new_articles_html,
                           frequency=frequency)
    
    # Attach HTML content
    msg.attach(MIMEText(html, 'html'))
    
    try:
        # Print email to console for debugging
        print(f"\n--- NOTIFICATION EMAIL ---")
        print(f"To: {recipient_email}")
        print(f"Subject: {msg['Subject']}")
        print(f"Body preview: {summary_text[:100]}...")
        if new_articles:
            print(f"Including {len(new_articles)} new articles in the email")
        print(f"--- END EMAIL ---\n")
        
        # Attempt to send the email using SMTP
        try:
            # You would replace these with your actual SMTP server details
            # For Gmail, you need an app password if 2FA is enabled
            smtp_server = "smtp.gmail.com"
            port = 587  # For starttls
            sender_email = "shantanu.kum97@gmail.com"  # Replace with a real email for production
            password = "gqgajschaboevchb"  # You would need to use an app password for Gmail
            
            print(f"Connecting to SMTP server: {smtp_server}:{port}")
            
            # Create a secure SSL context
            context = ssl.create_default_context()
            
            # Try to log in to server and send email
            server = smtplib.SMTP(smtp_server, port)
            server.ehlo()  # Can be omitted
            print(f"Starting TLS connection")
            server.starttls(context=context)  # Secure the connection
            server.ehlo()  # Can be omitted
            
            # Send the email
            print(f"Logging in as {sender_email}")
            server.login(sender_email, password)
            print(f"Sending email to {recipient_email}")
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Email sent successfully, closing connection")
            server.quit()
            
            print(f"Email sent to {recipient_email} successfully")
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            traceback.print_exc()
            # Continue - we don't want to fail the notification process if email sending fails
    
    except Exception as e:
        print(f"Error preparing email: {e}")
        traceback.print_exc()
        return False
    
    return True

# Schedule notification checks
def schedule_notification_checks():
    """Start a background thread to periodically check for notifications"""
    def check_notifications_thread():
        check_count = 0
        while True:
            try:
                check_count += 1
                print(f"[{datetime.now().isoformat()}] Running notification check #{check_count}")
                
                # Scan the user_data directory
                user_dir = 'user_data'
                all_history_files = []
                
                # Add shared history file
                shared_file = 'user_data/shared/search_history.json'
                if os.path.exists(shared_file):
                    all_history_files.append(shared_file)
                
                # Find all user-specific history files
                for item in os.listdir(user_dir):
                    item_path = os.path.join(user_dir, item)
                    if not os.path.isdir(item_path) or item == 'shared' or item == 'anonymous':
                        continue
                    
                    history_file = os.path.join(item_path, 'search_history.json')
                    if os.path.exists(history_file):
                        all_history_files.append(history_file)
                
                print(f"Found {len(all_history_files)} history files to check for notifications")
                
                # Check each file for notification settings
                total_notification_topics = 0
                for file_path in all_history_files:
                    try:
                        with open(file_path, 'r') as f:
                            history = json.load(f)
                        
                        file_notification_topics = 0
                        for key, entry in history.items():
                            if 'notifications' in entry and 'recipients' in entry['notifications'] and entry['notifications']['recipients']:
                                file_notification_topics += 1
                        
                        if file_notification_topics > 0:
                            print(f"  - {file_path}: {file_notification_topics} topics with notification settings")
                            total_notification_topics += file_notification_topics
                        
                    except Exception as e:
                        print(f"  - Error reading {file_path}: {e}")
                
                print(f"Found {total_notification_topics} total topics with notification settings")
                
                # Run the actual notification check
                check_and_send_notifications()
                
                # Log completion
                print(f"[{datetime.now().isoformat()}] Completed notification check #{check_count}")
            except Exception as e:
                print(f"Error in notification check: {e}")
                import traceback
                traceback.print_exc()
            
            # Check every 15 minutes
            print(f"Next notification check scheduled in 15 minutes")
            time.sleep(900)
    
    thread = threading.Thread(target=check_notifications_thread)
    thread.daemon = True
    thread.start()
    print("Started notification check thread")

@app.route('/api/notifications/save', methods=['POST'])
def save_notification_api():
    """API endpoint to save notification settings"""
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'})
    
    topic = data.get('topic')
    frequency = data.get('frequency')
    
    if not topic:
        return jsonify({'success': False, 'error': 'Missing topic'})
    
    if not frequency:
        return jsonify({'success': False, 'error': 'Missing frequency'})
    
    # Validate frequency value
    if frequency not in ['hourly', 'daily']:
        return jsonify({'success': False, 'error': 'Invalid frequency. Must be "hourly" or "daily"'})
    
    # Get the email - this could come from the request or use a hardcoded value
    # This can be expanded to handle multiple recipients
    user_email = data.get('email')
    
    # If no email provided in the request, use the current user's email
    if not user_email and current_user and hasattr(current_user, 'email'):
        user_email = current_user.email
    
    # Fallback to hardcoded email if still no email
    if not user_email:
        user_email = "shantanu.kum97@gmail.com"
    
    # Try to save the notification settings
    success = save_notification_settings(user_email, topic, frequency)
    
    # Check if the save was successful
    if not success:
        return jsonify({
            'success': False, 
            'error': f'Failed to save notification settings for topic "{topic}"',
            'email': user_email,
            'frequency': frequency
        })
    
    return jsonify({
        'success': True,
        'message': f'Successfully saved {frequency} notifications for {topic}',
        'email': user_email,
        'topic': topic,
        'frequency': frequency
    })

@app.route('/api/test-notification/<topic>/<frequency>', methods=['GET'])
def test_notification(topic, frequency):
    """Test endpoint to manually trigger a notification email"""
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    # Find the topic in the history
    history = load_search_history()
    entry = None
    
    for key, history_entry in history.items():
        if history_entry.get('query') == topic:
            entry = history_entry
            break
    
    if not entry:
        return jsonify({'success': False, 'error': f'Topic not found: {topic}'})
    
    # Get email addresses to test with - can be comma-separated in the query param
    test_emails = request.args.get('emails', 'shantanu.kum97@gmail.com')
    email_list = [email.strip() for email in test_emails.split(',')]
    
    # Set up notifications object if it doesn't exist
    if 'notifications' not in entry:
        entry['notifications'] = {
            'recipients': [],
            'last_sent': {}
        }
    
    # Add test recipients if they don't exist
    if 'recipients' not in entry['notifications']:
        entry['notifications']['recipients'] = []
    
    success = True
    results = []
    
    # Send to each email
    for email in email_list:
        # Add or update recipient
        recipient_exists = False
        for recipient in entry['notifications']['recipients']:
            if recipient['email'] == email:
                recipient['frequency'] = frequency
                recipient_exists = True
                break
                
        if not recipient_exists:
            entry['notifications']['recipients'].append({
                'email': email,
                'frequency': frequency
            })
        
        # Send test email
        email_success = send_notification_email(topic, entry, frequency, email)
        results.append({'email': email, 'success': email_success})
        success = success and email_success
    
    # Save changes
    save_search_history(history)
    
    return jsonify({
        'success': success,
        'results': results,
        'message': f"Test notification results for {topic} ({frequency})"
    })

@app.route('/api/debug-notifications', methods=['GET'])
def debug_notifications():
    """Debug endpoint to view all notification settings across all history files"""
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    results = []
    
    # Scan the user_data directory
    user_dir = 'user_data'
    all_history_files = []
    
    # Add shared history file
    shared_file = 'user_data/shared/search_history.json'
    if os.path.exists(shared_file):
        all_history_files.append(shared_file)
    
    # Find all user-specific history files
    for item in os.listdir(user_dir):
        item_path = os.path.join(user_dir, item)
        if not os.path.isdir(item_path) or item == 'shared' or item == 'anonymous':
            continue
        
        history_file = os.path.join(item_path, 'search_history.json')
        if os.path.exists(history_file):
            all_history_files.append(history_file)
    
    # Check each file for notification settings
    for file_path in all_history_files:
        file_result = {
            'file': file_path,
            'topics': [],
            'allEntries': [],  # New field to show all entries for debugging
            'entryCount': 0
        }
        
        try:
            with open(file_path, 'r') as f:
                history = json.load(f)
            
            file_result['entryCount'] = len(history)
            
            # Add all entries for debugging
            for key, entry in history.items():
                entry_info = {
                    'key': key,
                    'query': entry.get('query', 'Unknown'),
                    'hasNotifications': 'notifications' in entry,
                }
                
                # Check if notifications structure is valid
                if 'notifications' in entry:
                    notifications = entry['notifications']
                    entry_info['notificationsStructure'] = {
                        'hasRecipients': 'recipients' in notifications,
                        'hasLastSent': 'last_sent' in notifications,
                        'recipientsIsArray': isinstance(notifications.get('recipients', None), list),
                        'lastSentIsDict': isinstance(notifications.get('last_sent', None), dict),
                    }
                
                file_result['allEntries'].append(entry_info)
                
                # Add detailed info for entries with notifications
                if 'notifications' in entry and 'recipients' in entry['notifications']:
                    # Safety checks for proper structure
                    recipients = entry['notifications'].get('recipients', [])
                    if recipients and isinstance(recipients, list):
                        last_sent = entry['notifications'].get('last_sent')
                        if last_sent is None:
                            last_sent = {}
                            
                        topic_info = {
                            'topic': entry.get('query', 'Unknown'),
                            'recipients': recipients,
                            'last_sent': last_sent,
                            'structured': True
                        }
                        file_result['topics'].append(topic_info)
            
            results.append(file_result)
            
        except Exception as e:
            file_result['error'] = str(e)
            file_result['traceback'] = traceback.format_exc()
            results.append(file_result)
    
    return jsonify({
        'success': True,
        'historyFiles': results
    })

if __name__ == '__main__':
    # Initialize models in background threads to avoid blocking app startup
    if OPENAI_AVAILABLE:
        init_thread = threading.Thread(target=init_models)
        init_thread.daemon = True
        init_thread.start()
    
    # Start notification scheduler
    schedule_notification_checks()
    
    # Create templates
    # REMOVED: with open('templates/index.html', 'w') as f: ... block
    
    # REMOVED: with open('templates/error.html', 'w') as f: ... block
    
    # Create user_data directory for user files
    os.makedirs('user_data', exist_ok=True)
    
    # Run the Flask app
    print("Starting Flask application on http://0.0.0.0:5000")
    print("NOTE: You need to install scikit-learn for the topic grouping to work:")
    print("pip install scikit-learn")
    print("NOTE: To enable OpenAI summaries, install:")
    print("pip install openai")
    print("NOTE: Set the following environment variables for Google authentication:")
    print("export GOOGLE_CLIENT_ID=your_google_client_id")
    print("export GOOGLE_CLIENT_SECRET=your_google_client_secret")
    app.run(host='127.0.0.1', port=5000, debug=True) 