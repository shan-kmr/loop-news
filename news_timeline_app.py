#!/usr/bin/env python3
"""
News Timeline Web Application

A Flask-based web app that provides a UI for searching news articles and
displaying them in a timeline view, sorted by recency.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash, render_template_string
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

# Add dependencies for LLM models
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from huggingface_hub import login
    LLAMA_AVAILABLE = True
except ImportError:
    print("Transformers or torch not available. To enable Llama summaries run:")
    print("pip install transformers torch huggingface_hub")
    LLAMA_AVAILABLE = False

# Add dependencies for OpenAI
try:
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

# Llama model configuration
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_model_cache")
llama_model = None
llama_tokenizer = None

# Create the Brave News API client
brave_api = None

@login_manager.user_loader
def load_user(user_id):
    """Load user from storage for Flask-Login"""
    return User.get(user_id)

def init_models():
    """Initialize the models based on provider configuration"""
    if MODEL_PROVIDER == "llama" and LLAMA_AVAILABLE:
        init_llama_model()
    elif MODEL_PROVIDER == "openai" and OPENAI_AVAILABLE:
        init_openai()
    else:
        print(f"Warning: Selected model provider '{MODEL_PROVIDER}' is not available or invalid.")

def init_llama_model():
    """Initialize the Llama model for summarization in a background thread"""
    global llama_model, llama_tokenizer
    
    if not LLAMA_AVAILABLE:
        return
    
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        
        # Log in to Hugging Face with the API token
        login(token=HF_API_TOKEN)
        
        print(f"Loading Llama model from/to cache: {MODEL_CACHE_DIR}")
        
        # Setup tokenizer with explicit cache directory
        llama_tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL_ID, 
            token=HF_API_TOKEN,
            cache_dir=MODEL_CACHE_DIR
        )
        
        # Initialize model with HF Transformers and explicit cache
        llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_ID,
            token=HF_API_TOKEN,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
            # low_cpu_mem_usage=True
            cache_dir=MODEL_CACHE_DIR
        )
        print("Llama model loaded successfully")
    except Exception as e:
        print(f"Error loading Llama model: {str(e)}")
        llama_model = None
        llama_tokenizer = None

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
    """Generate a summary of articles for a specific day using the configured model"""
    if MODEL_PROVIDER == "llama":
        return summarize_daily_news_llama(day_articles, query)
    elif MODEL_PROVIDER == "openai":
        return summarize_daily_news_openai(day_articles, query)
    else:
        return None

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

def summarize_daily_news_llama(day_articles, query):
    """Generate a summary of articles for a specific day using Llama model"""
    if not LLAMA_AVAILABLE or llama_model is None or llama_tokenizer is None:
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
        
        # Create prompt for the model
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise news summaries. Summarize the key events and topics from these news articles in about 2-3 sentences without adding any new information. Focus on identifying the main developments and common themes. Start directly with the summary content - do not use phrases like 'Here is the summary' or 'The main news is'."},
            {"role": "user", "content": f"Here are news articles about \"{query}\" from the same day:\n\n{all_articles_text}\n\nPlease provide a brief summary of the main news for this day."}
        ]
        
        # Format as chat completion
        prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate summary
        input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids.to(llama_model.device)
        
        with torch.no_grad():
            outputs = llama_model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id
            )
        
        summary = llama_tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return summary
    except Exception as e:
        print(f"Error generating summary with Llama: {str(e)}")
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
        
        # Get fresh results
        fresh_results = brave_api.search_news(
            query=query,
            count=max(count * 2, 50),  # Get more results to ensure gap coverage
            freshness=refresh_freshness
        )
        
        if not fresh_results or 'results' not in fresh_results or not old_results or 'results' not in old_results:
            return fresh_results or old_results, []
        
        # Get new and old articles
        new_articles = fresh_results['results']
        old_articles = old_results['results']
        
        # Find the age of the oldest "recent" article from the original search
        # We'll keep articles older than this and replace everything newer
        oldest_recent_age_seconds = float('inf')
        
        # Determine what timeframe we're replacing based on the freshness parameter
        cutoff_seconds = {
            'pd': 86400,        # 1 day
            'pw': 604800,       # 1 week
            'pm': 2592000,      # 1 month 
            'py': 31536000,     # 1 year
            'h': 3600           # 1 hour (fallback)
        }.get(refresh_freshness, 604800)  # Default to 1 week if unknown
        
        # Categorize old articles
        articles_to_keep = []
        existing_articles_in_window = []
        
        for article in old_articles:
            age_seconds = extract_age_in_seconds(article)
            
            # Keep very old articles (older than our refresh window)
            if age_seconds > cutoff_seconds:
                articles_to_keep.append(article)
            else:
                # Keep track of articles within the refresh window (we'll keep these too)
                existing_articles_in_window.append(article)
        
        # Log what we're doing
        print(f"Refreshing articles for '{query}' using freshness: {refresh_freshness}")
        print(f"Keeping {len(articles_to_keep)} older articles beyond the refresh window")
        print(f"Looking for new articles to add to {len(existing_articles_in_window)} existing articles within the window")
        
        # Combine older articles with fresh results
        # De-duplicate articles by URL to prevent duplicates
        seen_urls = set()
        combined_articles = []
        
        # Track which days have new articles added (for summary regeneration)
        days_with_new_articles = set()
        
        # First add all existing articles in window (preserve what we've already shown)
        for article in existing_articles_in_window:
            url = article.get('url', '')
            if url:
                seen_urls.add(url)
                combined_articles.append(article)
        
        # Then add new articles that aren't already present
        new_articles_added = 0
        for article in new_articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined_articles.append(article)
                new_articles_added += 1
                
                # Track which day this article belongs to for summary regeneration
                day_group = day_group_filter(article)
                days_with_new_articles.add(day_group)
        
        # Finally add the older articles
        for article in articles_to_keep:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined_articles.append(article)
        
        print(f"Added {new_articles_added} new unique articles to the results")
        if days_with_new_articles:
            print(f"Days with new content (will regenerate summaries): {', '.join(days_with_new_articles)}")
        
        # Create updated results
        updated_results = fresh_results.copy()
        updated_results['results'] = combined_articles
        
        return updated_results, list(days_with_new_articles)
    
    except Exception as e:
        print(f"Error updating results: {str(e)}")
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
    
    # Redirect to raison d'etre page if not logged in
    if not current_user.is_authenticated:
        return redirect(url_for('raison_detre'))
    
    # Initialize the API client if needed
    if brave_api is None:
        brave_api_key = app.config.get("BRAVE_API_KEY")
        if not brave_api_key:
            return render_template('error.html', 
                                  error="BRAVE_API_KEY environment variable or config not set. Please set it and restart the app.",
                                  user=current_user)
        brave_api = BraveNewsAPI(api_key=brave_api_key)
    
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
                            'day_summaries': day_summaries
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

@app.route('/history/<path:query>', methods=['GET'])
@login_required
def history_item(query):
    """View to display a specific history item."""
    # Load history
    history = load_search_history()
    
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
                results = entry['results']
                search_time = cached_time
                count = entry.get('count', 10)
                freshness = entry.get('freshness', 'pw')
                
                # Always refresh when a topic is clicked
                needs_day_refresh = True  # Force refresh every time
                
                # Handle refreshing and processing
                if needs_day_refresh:
                    print(f"Refreshing results for '{query}'")
                    updated_results, days_with_new_articles = update_current_day_results(query, count, freshness, results)
                    
                    if updated_results != results:
                        results = updated_results
                        search_time = datetime.now()
                        
                        # If we have days with new articles, clear their summaries to force regeneration
                        if days_with_new_articles and 'day_summaries' in entry:
                            print(f"Clearing summaries for days with new content: {', '.join(days_with_new_articles)}")
                            for day in days_with_new_articles:
                                if day in entry['day_summaries']:
                                    entry['day_summaries'][day] = None
                    else:
                        # Even if results are the same, update the timestamp
                        search_time = datetime.now()
                    
                    # Update the cache with refreshed results
                    entry['results'] = results
                    entry['timestamp'] = search_time.isoformat()
                    save_search_history(history)
                else:
                    print(f"Using cached results for '{query}' from {cached_time}, {int(time_diff.total_seconds() / 60)} minutes old")
                
                # Process day summaries
                day_summaries = entry.get('day_summaries', {})
                if day_summaries is None:
                    day_summaries = {}
                
                # Validate each summary to ensure we don't have invalid entries
                valid_summaries = 0
                for day, summary in list(day_summaries.items()):
                    if is_valid_summary(summary):
                        valid_summaries += 1
                    else:
                        # Clear invalid summaries so they'll be regenerated
                        day_summaries[day] = None
                
                print(f"Found {valid_summaries} valid day summaries out of {len(day_summaries)} total in cache")
                
                # Process articles
                if results and 'results' in results:
                    sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
                    # Pass existing day summaries to avoid regenerating summaries
                    topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, current_query, day_summaries)
                    
                    # Save the updated day_summaries back to the entry to ensure they're stored in the cache
                    # This is important for ensuring summaries persist between page visits
                    if 'day_summaries' not in entry or entry['day_summaries'] != day_summaries:
                        entry['day_summaries'] = day_summaries
                        save_search_history(history)
                        print(f"Saved {len(day_summaries)} day summaries to cache for query: '{query}'")
                    
                    # Check if we need to update summaries in history (redundant now but kept for safety)
                    summary_changed = False
                    for topic in topic_groups:
                        day = topic['day_group']
                        if day in day_summaries and day_summaries[day] != topic.get('day_summary') and is_valid_summary(topic.get('day_summary')):
                            summary_changed = True
                            day_summaries[day] = topic.get('day_summary')
                    
                    if summary_changed:
                        entry['day_summaries'] = day_summaries
                        save_search_history(history)
                        print(f"Updated cache with new summaries for query: '{query}'")
                
                break
    
    # If no exact match was found, perform a new search
    if not found:
        print(f"No exact match found for query: '{query}'. Performing new search.")
        try:
            # Initialize API if needed
            global brave_api
            if brave_api is None:
                brave_api_key = app.config.get("BRAVE_API_KEY")
                if brave_api_key:
                    brave_api = BraveNewsAPI(api_key=brave_api_key)
                else:
                    return render_template('error.html', error="BRAVE_API_KEY not set", user=current_user)
            
            # Default values
            count = 10
            freshness = 'pw'
            
            # Make a new API call
            print(f"Making new API call for '{query}'")
            results = brave_api.search_news(
                query=query,
                count=count,
                freshness=freshness
            )
            search_time = datetime.now()
            current_query = query
            
            # Process results
            if results and 'results' in results:
                sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
                day_summaries = {}
                topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, query, day_summaries)
                
                # Cache the results
                cache_key = f"{query}_{count}_{freshness}"
                history[cache_key] = {
                    'query': query,
                    'count': count,
                    'freshness': freshness,
                    'timestamp': search_time.isoformat(),
                    'results': results,
                    'day_summaries': day_summaries
                }
                save_search_history(history)
                print(f"Cached new results for query: '{query}' with {len(day_summaries)} summaries")
        except Exception as e:
            print(f"Error performing search: {str(e)}")
    
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

def group_articles_by_topic(articles, similarity_threshold=0.3, query="", day_summaries=None):
    """
    Group articles into topics based on content similarity.
    
    Args:
        articles: List of news articles
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
    
    # Extract title and description for content comparison
    article_contents = []
    for article in articles:
        title = article.get('title', '')
        desc = article.get('description', '')
        content = f"{title} {desc}"
        article_contents.append(clean_text(content))
    
    # Compute TF-IDF vectorization
    try:
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(article_contents)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Group articles based on similarity
        visited = [False] * len(articles)
        topic_groups = []
        
        for i in range(len(articles)):
            if visited[i]:
                continue
                
            visited[i] = True
            group = [articles[i]]
            
            # Find similar articles
            for j in range(i+1, len(articles)):
                if not visited[j] and similarity_matrix[i, j] >= similarity_threshold:
                    group.append(articles[j])
                    visited[j] = True
            
            # Only add groups with more than one article
            if len(group) > 1:
                # Sort group by age
                group = sorted(group, key=extract_age_in_seconds)
                
                # Generate a topic title from the newest article
                newest_article = group[0]
                topic_title = newest_article.get('title', 'Untitled Topic')
                
                topic_groups.append({
                    'title': topic_title,
                    'articles': group,
                    'count': len(group),
                    'newest_age': newest_article.get('age', 'Unknown'),
                    'day_group': day_group_filter(newest_article)
                })
            else:
                # Add as a single article topic
                topic_groups.append({
                    'title': group[0].get('title', 'Untitled Topic'),
                    'articles': group,
                    'count': 1,
                    'newest_age': group[0].get('age', 'Unknown'),
                    'day_group': day_group_filter(group[0])
                })
        
        # Sort the topic groups by the age of their newest article
        topic_groups = sorted(topic_groups, key=lambda x: extract_age_in_seconds(x['articles'][0]))
        
        # Track days that need summaries
        days_needing_summaries = set()
        
        # Add existing summaries to topic groups first 
        for topic in topic_groups:
            day = topic['day_group']
            # Check if we already have a valid summary for this day
            existing_summary = day_summaries.get(day)
            topic['day_summary'] = existing_summary
            
            if not is_valid_summary(existing_summary):
                if day not in days_needing_summaries:
                    days_needing_summaries.add(day)
            else:
                print(f"Using existing summary for day: {day}")
        
        # Only generate new summaries if needed and if models are available
        if days_needing_summaries and ((MODEL_PROVIDER == "llama" and LLAMA_AVAILABLE and llama_model is not None) or 
                                       (MODEL_PROVIDER == "openai" and OPENAI_AVAILABLE)):
            # Group articles by day, but only for days that need new summaries
            day_articles = {}
            for topic in topic_groups:
                day = topic['day_group']
                if day in days_needing_summaries:
                    if day not in day_articles:
                        day_articles[day] = []
                    day_articles[day].extend(topic['articles'])
            
            # Generate summaries only for days that need them
            for day, day_art in day_articles.items():
                # Skip days that have valid summaries (this is an additional safeguard)
                if day in day_summaries and is_valid_summary(day_summaries[day]):
                    print(f"Skipping summary generation for day: {day} (already valid)")
                    continue
                    
                print(f"Generating new summary for day: {day} using {MODEL_PROVIDER}")
                summary = summarize_daily_news(day_art, query)
                
                # Only update if we got a valid summary
                if is_valid_summary(summary):
                    print(f"Successfully generated summary for {day}: {summary[:50]}...")
                    day_summaries[day] = summary
                    
                    # Update topic groups with new summaries
                    for topic in topic_groups:
                        if topic['day_group'] == day:
                            topic['day_summary'] = summary
                    
                    # This code was moved out of the loop to avoid excessive summary generation
                    # We now directly save the summaries as they're generated
                    try:
                        # Try to find the history entry for this query and update it
                        history = load_search_history()
                        for key, entry in history.items():
                            if entry.get('query') == query:
                                entry['day_summaries'] = day_summaries
                                save_search_history(history)
                                print(f"Immediately saved new {day} summary to history for {query}")
                                break
                    except Exception as e:
                        print(f"Error saving summary to history: {e}")
                else:
                    print(f"Failed to generate valid summary for day: {day}")
        
        # Count how many topics have summaries 
        topics_with_summaries = 0
        for topic in topic_groups:
            if is_valid_summary(topic.get('day_summary')):
                topics_with_summaries += 1
                
        print(f"Grouped {len(topic_groups)} topics, {topics_with_summaries} have summaries. Day summaries dict has {len(day_summaries)} entries.")
        
        return topic_groups
    
    except Exception as e:
        print(f"Error grouping articles: {str(e)}")
        # Fallback: return each article as its own group if clustering fails
        return [{'title': a.get('title', 'Untitled'), 'articles': [a], 'count': 1, 
                 'newest_age': a.get('age', 'Unknown'), 'day_group': day_group_filter(a)} 
                for a in articles]

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
    
    # Email body
    html = f"""
    <html>
      <head></head>
      <body style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; color: #333;">
        <h1 style="font-size: 24px; color: #333; margin-bottom: 20px;">loop: {topic}</h1>
        
        <div style="margin-bottom: 30px;">
          <h2 style="font-size: 18px; color: #555; margin-bottom: 10px;">{summary_date}'s Summary ({email_date_str} EST)</h2>
          <p style="line-height: 1.6;">{summary_text}</p>
        </div>
        
        {new_articles_html}
        
        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 14px; color: #777;">
          <p>View full history at: <a href="http://loopnow.co/history/{topic}" style="color: #555;">View Details</a></p>
          <p>Notification frequency: {frequency}. To change your settings, click the bell icon next to the topic name.</p>
        </div>
      </body>
    </html>
    """
    
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
    if LLAMA_AVAILABLE or OPENAI_AVAILABLE:
        init_thread = threading.Thread(target=init_models)
        init_thread.daemon = True
        init_thread.start()
    
    # Start notification scheduler
    schedule_notification_checks()
    
    # Create templates
    with open('templates/index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>loop: track all moves in a single timeline</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        :root {
            --bg-color: #f8f8f8;
            --text-color: #333;
            --accent-color: #dad3c8;
            --secondary-color: #c5bcb1;
            --border-color: #e5e7eb;
            --card-bg: #ffffff;
        }
        
        [data-theme="dark"] {
            --bg-color: #121212;
            --text-color: #ffffff;
            --accent-color: #4a4a4a;
            --secondary-color: #333333;
            --border-color: #2c2c2c;
            --card-bg: #1e1e1e;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: "Proxima Nova", Tahoma, -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        h1, h2, h3, h4, h5, h6, p, a, span, div, button {
            text-transform: lowercase;
        }
        
        a {
            color: var(--accent-color);
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 15px;
        }
        
        .logo {
            font-size: 24px;
            font-weight: 700;
            color: #000000;
        }
        
        nav {
            display: flex;
            gap: 20px;
        }
        
        nav a {
            padding: 5px 10px;
            border-radius: 4px;
            color: var(--text-color);
            background-color: transparent;
            transition: color 0.2s ease;
            box-shadow: none;
            text-decoration: none;
        }
        
        nav a.active {
            background-color: transparent;
            color: var(--text-color);
            font-weight: 500;
            position: relative;
        }
        
        nav a.active:after {
            content: '';
            position: absolute;
            width: 100%;
            height: 2px;
            bottom: 0;
            left: 0;
            background-color: var(--text-color);
        }
        
        nav a:hover {
            background-color: transparent;
            text-decoration: underline;
        }
        
        .theme-toggle {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: transparent;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-color);
            font-size: 18px;
            margin-left: 10px;
        }
        
        .user-info {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
        }
        
        .user-avatar {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: var(--secondary-color);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #000000;
        }
        
        .main-container {
            display: flex;
            gap: 30px;
        }
        
        .brief-list {
            flex: 1;
            max-width: 800px;
        }
        
        .brief-card {
            background-color: var(--card-bg);
            padding: 24px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid var(--border-color);
            position: relative;
        }
        
        .home-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .brief-title {
            font-size: 24px;
            font-weight: 600;
            color: var(--text-color);
            flex: 1;
        }
        
        .card-right-elements {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .detail-link {
            display: flex;
            align-items: center;
            color: var(--text-color);
            opacity: 0.7;
            font-size: 14px;
            transition: opacity 0.2s ease;
        }
        
        .detail-link .arrow {
            margin-left: 5px;
            font-size: 16px;
        }
        
        .brief-card:hover .detail-link {
            opacity: 0.9;
        }
        
        [data-theme="dark"] .brief-card {
            box-shadow: none;
        }
        
        .brief-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        [data-theme="dark"] .brief-card:hover {
            box-shadow: none;
        }
        
        /* Separate styles for detail page */
        .detail-page .brief-meta {
            display: flex;
            justify-content: flex-end;
            font-size: 14px;
            color: #666;
            margin-bottom: 0;
            position: absolute;
            top: 24px;
            right: 24px;
        }
        
        .brief-meta {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: #666;
            margin-bottom: 16px;
        }
        
        .brief-summary-container {
            display: flex;
            flex-direction: column;
            margin-bottom: 16px;
        }
        
        .summary-date {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 5px;
        }
        
        .refresh-info {
            display: flex;
            align-items: center;
            background-color: transparent;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 14px;
            color: var(--text-color);
            opacity: 0.7;
            white-space: nowrap;
        }
        
        [data-theme="dark"] .refresh-info {
            background-color: transparent;
            color: #ffffff;
        }
        
        .reload-timer {
            font-weight: bold;
            color: var(--accent-color);
            margin-left: 4px;
            opacity: 0.9;
        }
        
        [data-theme="dark"] .reload-timer {
            color: #ffffff;
            opacity: 0.9;
        }
        
        .cta-button {
            background-color: var(--accent-color);
            color: var(--text-color);
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        [data-theme="dark"] .cta-button {
            box-shadow: none;
        }
        
        .cta-button:hover {
            background-color: var(--secondary-color);
        }
        
        .search-box {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 16px;
            margin-bottom: 20px;
        }
        
        .search-options {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .option-select {
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background-color: white;
            font-size: 14px;
        }
        
        .timeline-container {
            background-color: var(--card-bg);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-top: 20px;
            border: 1px solid var(--border-color);
        }
        
        .timeline-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
        }
        
        .timeline-day {
            font-size: 18px;
            font-weight: 600;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-color);
        }
        
        .day-summary {
            background-color: var(--accent-color);
            padding: 15px 20px;
            border-radius: 6px;
            margin-bottom: 20px;
            line-height: 1.7;
            font-style: italic;
            color: var(--text-color);
        }
        
        /* Flickering red dot */
        .live-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #ff3b30;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
            position: relative;
            top: -1px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% {
                opacity: 1;
            }
            50% {
                opacity: 0.4;
            }
            100% {
                opacity: 1;
            }
        }
        
        .latest-update {
            font-style: italic;
            color: #888;
            font-weight: 300;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        .timeline-item {
            border-left: 3px solid var(--accent-color);
            padding-left: 20px;
            margin-bottom: 25px;
            padding-bottom: 15px;
        }
        
        .topic-header {
            cursor: pointer;
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            position: relative;
        }
        
        .expand-icon {
            display: inline-block;
            width: 0;
            height: 0;
            border-style: solid;
            border-width: 5px 0 5px 8px;
            border-color: transparent transparent transparent var(--text-color);
            margin-right: 10px;
            transition: transform 0.3s ease, opacity 0.3s ease;
            opacity: 0.6;
            transform-origin: center;
        }
        
        .topic-header.expanded .expand-icon {
            transform: rotate(90deg);
            opacity: 0.9;
        }
        
        .topic-title {
            font-size: 17px;
            font-weight: 600;
            margin-right: 10px;
            color: var(--text-color);
            opacity: 0.85;
        }
        
        .topic-count {
            background-color: var(--secondary-color);
            color: #000000;
            border-radius: 12px;
            padding: 2px 10px;
            font-size: 13px;
            font-weight: 500;
        }
        
        .topic-time {
            color: #666;
            font-size: 14px;
            margin-left: auto;
        }
        
        .topic-content {
            display: none;
            margin-left: 20px;
        }
        
        .article-item {
            border-left: 2px solid var(--border-color);
            padding: 15px;
            margin-bottom: 20px;
            background-color: var(--card-bg);
            border-radius: 4px;
        }
        
        .timeline-source {
            color: var(--text-color);
            opacity: 0.7;
            font-size: 14px;
            margin-bottom: 10px;
        }
        
        .timeline-title {
            font-size: 17px;
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--text-color);
        }
        
        .timeline-desc {
            margin-bottom: 15px;
            color: var(--text-color);
            opacity: 0.9;
            line-height: 1.6;
        }
        
        .timeline-link {
            font-size: 14px;
            color: var(--accent-color);
            word-break: break-all;
        }
        
        .error {
            background-color: #fee2e2;
            color: #b91c1c;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }
        
        .fab-button {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: var(--accent-color);
            color: var(--text-color);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            cursor: pointer;
            z-index: 100;
            transition: all 0.3s ease;
        }
        
        [data-theme="dark"] .fab-button {
            box-shadow: none;
            color: #ffffff;
        }
        
        .fab-button:hover {
            background-color: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.25);
        }
        
        [data-theme="dark"] .fab-button:hover {
            box-shadow: none;
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0,0,0,0.5);
            z-index: 200;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background-color: var(--card-bg);
            padding: 30px;
            border-radius: 8px;
            width: 90%;
            max-width: 500px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        
        [data-theme="dark"] .modal-content {
            box-shadow: none;
        }
        
        .modal-title {
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 20px;
            color: var(--text-color);
        }
        
        .modal-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 20px;
        }
        
        .modal-button {
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            border: none;
            font-size: 16px;
        }
        
        .cancel-button {
            background-color: #e5e7eb;
            color: #4b5563;
        }
        
        .submit-button {
            background-color: var(--accent-color);
            color: #000000;
        }
        
        @media (max-width: 768px) {
            .main-container {
                flex-direction: column;
            }
            
            header {
                flex-direction: column;
                gap: 15px;
                align-items: flex-start;
            }
            
            .brief-card {
                padding: 20px;
            }
        }
        
        /* Dark mode overrides */
        [data-theme="dark"] .logo,
        [data-theme="dark"] h2,
        [data-theme="dark"] .brief-title,
        [data-theme="dark"] .topic-title,
        [data-theme="dark"] .summary-date,
        [data-theme="dark"] .timeline-day,
        [data-theme="dark"] .timeline-source,
        [data-theme="dark"] .timeline-title,
        [data-theme="dark"] .timeline-desc,
        [data-theme="dark"] .today,
        [data-theme="dark"] .yesterday,
        [data-theme="dark"] .latest-update,
        [data-theme="dark"] .day-summary,
        [data-theme="dark"] .topic-header,
        [data-theme="dark"] .brief-summary,
        [data-theme="dark"] .reload-timer-label,
        [data-theme="dark"] .reload-timer,
        [data-theme="dark"] .modal-title,
        [data-theme="dark"] .submit-button,
        [data-theme="dark"] .cancel-button {
            color: #ffffff;
        }
        /* More comprehensive dark mode shadow removal */
        [data-theme="dark"] .brief-card,
        [data-theme="dark"] .brief-card:hover,
        [data-theme="dark"] .fab-button,
        [data-theme="dark"] .fab-button:hover,
        [data-theme="dark"] .timeline-container,
        [data-theme="dark"] .modal-content,
        [data-theme="dark"] .modal-button,
        [data-theme="dark"] .cta-button,
        [data-theme="dark"] .cta-button:hover,
        [data-theme="dark"] nav a,
        [data-theme="dark"] nav a.active,
        [data-theme="dark"] .article-item {
            box-shadow: none !important;
            -webkit-box-shadow: none !important;
            -moz-box-shadow: none !important;
        }
        
        /* Global shadow removal for dark mode */
        [data-theme="dark"] * {
            box-shadow: none !important;
            -webkit-box-shadow: none !important;
            -moz-box-shadow: none !important;
        }
        
        .article-url-summary {
            font-size: 13px;
            color: var(--text-color);
            opacity: 0.7;
            font-style: italic;
            margin-top: 5px;
            margin-bottom: 10px;
        }
        .topic-count-display {
            font-size: 14px;
            color: var(--text-color);
            opacity: 0.7;
            margin-left: 10px;
            font-style: italic;
            padding-right: 5px;
            white-space: nowrap;
        }
        
        .loading-spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(0, 0, 0, 0.1);
            border-radius: 50%;
            border-top: 2px solid var(--accent-color);
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        
        [data-theme="dark"] .loading-spinner {
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-top: 2px solid var(--accent-color);
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .submit-button-container {
            display: flex;
            align-items: center;
        }
        
        .delete-button {
            position: absolute;
            bottom: 15px;
            right: 15px;
            color: #ff3b30;
            opacity: 0.6;
            font-size: 13px;
            background: transparent;
            border: none;
            padding: 5px 8px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .delete-button:hover {
            opacity: 0.9;
            background-color: rgba(255, 59, 48, 0.1);
        }
        
        [data-theme="dark"] .delete-button {
            color: #ff6b6b;
        }
        
        .flash-messages {
            margin-bottom: 20px;
            width: 100%;
        }
        
        .flash-message {
            padding: 12px 15px;
            margin-bottom: 10px;
            border-radius: 6px;
            font-size: 14px;
        }
        
        .flash-error {
            background-color: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }
        
        .flash-success {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        
        .flash-info {
            background-color: #dbeafe;
            color: #1e40af;
            border: 1px solid #bfdbfe;
        }
        
        .access-request-form {
            max-width: 400px;
            margin: 0 auto;
            padding: 30px;
            background-color: var(--card-bg);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: 1px solid var(--border-color);
        }
        
        .access-request-form h2 {
            margin-bottom: 20px;
            font-size: 24px;
        }
        
        .access-request-form p {
            margin-bottom: 20px;
            opacity: 0.8;
        }
        
        .philosophy-page {
            padding: 30px;
        }
        
        .philosophy-page .section-title {
            font-size: 32px;
            font-weight: 300;
            margin-bottom: 30px;
            color: var(--text-color);
        }
        
        .philosophy-section {
            margin-bottom: 40px;
        }
        
        .philosophy-section h2 {
            font-size: 24px;
            font-weight: 400;
            margin-bottom: 15px;
            color: var(--text-color);
        }
        
        .philosophy-section p {
            font-size: 16px;
            line-height: 1.7;
            margin-bottom: 15px;
            max-width: 700px;
            color: var(--text-color);
            opacity: 0.9;
        }
        
        [data-theme="dark"] .philosophy-section p {
            opacity: 0.85;
        }
        
        .notification-bell {
            margin-left: 8px;
            font-size: 14px;
            color: var(--text-color);
            opacity: 0.5;
            cursor: pointer;
            transition: opacity 0.2s ease;
        }
        
        .notification-bell:hover {
            opacity: 0.9;
        }
        
        [data-theme="dark"] .notification-bell {
            color: var(--text-color);
            opacity: 0.4;
        }
        
        [data-theme="dark"] .notification-bell:hover {
            opacity: 0.8;
        }
        
        .notification-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 10;
            justify-content: center;
            align-items: center;
        }
        
        .notification-modal-content {
            background-color: var(--card-bg);
            width: 100%;
            max-width: 400px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            position: relative;
        }
        
        .notification-modal-title {
            font-size: 20px;
            margin-bottom: 15px;
            color: var(--text-color);
        }
        
        .notification-topic {
            font-weight: bold;
            color: var(--text-color);
        }
        
        .notification-options {
            margin: 20px 0;
        }
        
        .notification-option {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .notification-option input {
            margin-right: 10px;
        }
        
        .notification-option label {
            color: var(--text-color);
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">loop: track all moves in a single timeline.</div>
        <nav>
            <a href="/" class="{% if active_tab == 'search' %}active{% endif %}">home</a>
            <a href="/history" class="{% if active_tab == 'history' %}active{% endif %}">briefs</a>
            <a href="/raison-detre" class="{% if active_tab == 'raison-detre' %}active{% endif %}">raison d'être</a>
        </nav>
        <div class="user-info">
            {% if user.is_authenticated %}
                <div class="user-avatar">
                    {% if user.profile_pic %}
                        <img src="{{ user.profile_pic }}" alt="{{ user.name }}" width="30" height="30">
                    {% else %}
                        <i class="fas fa-user"></i>
                    {% endif %}
                </div>
                <span>{{ user.name }}</span>
                <a href="{{ url_for('auth.logout') }}">logout</a>
            {% else %}
                <div class="user-avatar">
                    <i class="fas fa-user"></i>
                </div>
                <span>limited access</span>
                <a href="{{ url_for('auth.login') }}" style="margin-right: 8px;">login</a>
                <a href="#" onclick="openRequestAccessModal(); return false;">request access</a>
            {% endif %}
            <button class="theme-toggle" id="theme-toggle" aria-label="Toggle dark mode">
                <i class="fas fa-moon"></i>
            </button>
        </div>
    </header>

    <div class="main-container">
        <main class="brief-list">
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    <div class="flash-messages">
                        {% for category, message in messages %}
                            <div class="flash-message flash-{{ category }}">
                                {{ message }}
                            </div>
                        {% endfor %}
                    </div>
                {% endif %}
            {% endwith %}
            
            {% if error %}
                <div class="error">
                    {{ error }}
                </div>
            {% endif %}
            
            {% if active_tab == 'search' %}
                <!-- Search form removed from here -->
                
                {% if history_entries %}
                    <h2 style="margin: 20px 0 20px 0;">your briefs 
                        <span class="topic-count-display">{{ history_entries|length }} out of 3 briefs used</span>
                    </h2>
                    {% if history_entries|length > 0 %}
                        <p class="latest-update">last updated {{ history_entries[0].timestamp|format_datetime('%b %d, %Y') }}</p>
                    {% endif %}
                    {% for entry in history_entries %}
                        <div class="brief-card" onclick="window.location.href='/history/{{ entry.query }}'">
                            <div class="home-card-header">
                                <div class="brief-title">
                                    {{ entry.query }}
                                    <i class="fas fa-bell notification-bell" onclick="event.stopPropagation(); openNotificationModal('{{ entry.query }}')"></i>
                                </div>
                                <div class="card-right-elements">
                                    <div class="detail-link">
                                        detailed analysis <span class="arrow">→</span>
                                    </div>
                                    <div class="refresh-info">
                                        <span class="reload-timer-label">auto-refresh in:</span>
                                        <span class="reload-timer" data-query="{{ entry.query }}">--:--</span>
                                    </div>
                                </div>
                            </div>
                            
                            {% set has_summary = false %}
                            
                            {# First priority: Show Today's summary if available #}
                            {% for key, entry_data in history.items() %}
                                {% if entry_data.query == entry.query and entry_data.day_summaries and 'Today' in entry_data.day_summaries and is_valid_summary(entry_data.day_summaries['Today']) and not has_summary %}
                                    <div class="brief-summary-container">
                                        <div class="summary-date today">Today <span class="live-dot"></span></div>
                                        <p class="brief-summary">{{ entry_data.day_summaries['Today'] }}</p>
                                    </div>
                                    {% set has_summary = true %}
                                {% endif %}
                            {% endfor %}
                            
                            {# Second priority: Show Yesterday's summary if available #}
                            {% if not has_summary %}
                                {% for key, entry_data in history.items() %}
                                    {% if entry_data.query == entry.query and entry_data.day_summaries and 'Yesterday' in entry_data.day_summaries and is_valid_summary(entry_data.day_summaries['Yesterday']) and not has_summary %}
                                        <div class="brief-summary-container">
                                            <div class="summary-date yesterday">Yesterday</div>
                                            <p class="brief-summary">{{ entry_data.day_summaries['Yesterday'] }}</p>
                                        </div>
                                        {% set has_summary = true %}
                                    {% endif %}
                                {% endfor %}
                            {% endif %}
                            
                            {# Last resort: fallback text - NOW REMOVED #}
                            {% if not has_summary %}
                                {# Fallback text removed - now using the arrow in top-right instead #}
                            {% endif %}
                            
                            <button class="delete-button" onclick="event.stopPropagation(); deleteHistoryItem('{{ entry.query }}')">
                                delete
                            </button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% elif active_tab == 'history' %}
                {% if query %}
                    <div class="brief-card expanded detail-page">
                        <div class="brief-title">{{ query }}</div>
                        <div class="brief-meta">
                            <!-- Removed timestamp from here -->
                            <div class="refresh-info">
                                <span class="reload-timer-label">auto-refresh in:</span>
                                <span class="reload-timer" data-query="{{ query }}">--:--</span>
                            </div>
                        </div>
                        
                        {% if topic_groups and topic_groups|length > 0 and topic_groups[0].day_summary %}
                            <p class="brief-summary">{{ topic_groups[0].day_summary }}</p>
                        {% endif %}
                        
                        {% if results %}
                            <div class="timeline-container">
                                {% set ns = namespace(current_day=None, current_day_summary=None) %}
                                
                                {% if topic_groups | length == 0 %}
                                    <div style="text-align: center; padding: 30px; color: #666;">
                                        <p>no news articles found for this query.</p>
                                    </div>
                                {% endif %}
                                
                                {% for topic in topic_groups %}
                                    {% set day = topic.day_group %}
                                    
                                    {% if day != ns.current_day %}
                                        {% set ns.current_day = day %}
                                        {% set ns.current_day_summary = topic.day_summary %}
                                        <div class="timeline-day">
                                            {{ day }}
                                            {% if day == "Today" %}
                                                <span class="live-dot"></span>
                                            {% endif %}
                                        </div>
                                        
                                        {% if topic.day_summary %}
                                            <div class="day-summary">
                                                {{ topic.day_summary }}
                                            </div>
                                        {% endif %}
                                    {% endif %}
                                    
                                    <div class="timeline-item">
                                        <div class="topic-header" onclick="toggleTopic(this)">
                                            <span class="expand-icon"></span>
                                            <div class="topic-title">{{ topic.title }}</div>
                                            {% if topic.count > 1 %}
                                                <span class="topic-count">{{ topic.count }} articles</span>
                                            {% endif %}
                                            <div class="topic-time">{{ topic.newest_age }}</div>
                                        </div>
                                        
                                        <div class="topic-content">
                                            {% for article in topic.articles %}
                                                <div class="article-item">
                                                    <div class="timeline-source">{{ article.get('meta_url', {}).get('netloc', 'unknown source') }}</div>
                                                    {% if loop.index > 1 %}
                                                        <div class="timeline-title">{{ article.get('title', 'no title') }}</div>
                                                    {% endif %}
                                                    <div class="timeline-desc">{{ article.get('description', 'no description available') }}</div>
                                                    <div class="article-url-summary">{{ article.get('title', 'no title').split(' - ')[0] if ' - ' in article.get('title', 'no title') else article.get('title', 'no title').split('|')[0] if '|' in article.get('title', 'no title') else article.get('title', 'no title') }}</div>
                                                    <a href="{{ article.get('url', '#') }}" class="timeline-link" target="_blank">{{ article.get('url', 'no url available') }}</a>
                                                </div>
                                            {% endfor %}
                                        </div>
                                    </div>
                                {% endfor %}
                            </div>
                        {% endif %}
                    </div>
                    
                    <div style="margin-top: 20px; display: flex; gap: 10px;">
                        <button class="cta-button" onclick="window.location.href='/'" style="background-color: #6b7280;">
                            <i class="fas fa-arrow-left"></i> back to briefs
                        </button>
                        <button class="cta-button" onclick="forceRefresh('{{ query }}')">
                            <i class="fas fa-sync-alt"></i> refresh now
                        </button>
                        <button class="cta-button" onclick="deleteHistoryItem('{{ query }}')">
                            <i class="fas fa-trash"></i> delete this brief
                        </button>
                    </div>
                {% else %}
                    <div style="text-align: center; padding: 50px; color: #666;">
                        <h2>no briefs yet</h2>
                        <p style="margin: 20px 0;">add a search term to start tracking news topics</p>
                        <button class="cta-button" onclick="window.location.href='/'">
                            <i class="fas fa-plus"></i> add new brief
                        </button>
                    </div>
                {% endif %}
            {% elif active_tab == 'raison-detre' %}
                <div class="brief-card expanded philosophy-page">
                    
                    
                    <div class="philosophy-section">
                        <h2>why i built this</h2>
                        <p>
                            my vision for a sharper informational edge. in an era where information abundance creates 
                            its own form of scarcity—attention—the need for systems that value sequential, contextual, 
                            and verifiable information has never been more critical.
                        </p>
                    </div>
                    
                    <div class="philosophy-section">
                        <h2>beyond the feed</h2>
                        <p>
                            traditional news suffers from a fundamental reductionism—breaking complex narratives into
                            disconnected fragments that destroy context and coherence.
                        </p>
                        <p>
                            loop challenges this paradigm by restoring these connections, presenting not isolated data points
                            but complete trajectories—a temporal map of events and their relationships across multiple
                            dimensions of relevance. loop tracks trends, entities, and relationships across time–notifying you of immediate changes.
                        </p>
                        <p>
                            my coarse take on antireductionism as a 17-year old <a href="https://docs.google.com/document/d/1GXAdOSCwvc_XALn1WTwRVDOdDcFC1lfeAx1WLTpyg9w/" target="_blank">here</a>.
                        </p>
                    </div>
                    
                    <div class="philosophy-section">
                        <h2>intelligence as a service</h2>
                        <p>
                            the information landscape now operates with its own mens rea—a system designed to capture attention
                            without the ethical framework to ensure integrity. it's a subtle crime of modern media: presenting
                            partial truths as complete pictures. loop serves as a corrective measure, applying computational rigor to restore missing context.
                        </p>
                    </div>
                    
                    <!-- Removed "first principles approach" section -->
                    
                    <!-- Removed "personal mission" section -->
                </div>
            {% endif %}
        </main>
    </div>
    
    <!-- Floating action button for adding new briefs - only shown to logged in users with fewer than 3 topics -->
    {% if user.is_authenticated and history_entries|length < 3 %}
    <div class="fab-button" onclick="openNewBriefModal()">
        <i class="fas fa-plus"></i>
    </div>
    {% endif %}
    
    <!-- Modal for adding new briefs -->
    <div id="newBriefModal" class="modal">
        <div class="modal-content">
            <h2 class="modal-title">add new brief</h2>
            {% if user.is_authenticated %}
                {% if history_entries|length >= 3 %}
                    <p style="margin-bottom: 20px; color: #ff3b30;">You've reached the maximum of 3 briefs. Please remove one before adding another.</p>
                    <div class="modal-buttons">
                        <button type="button" class="modal-button cancel-button" onclick="closeNewBriefModal()">close</button>
                    </div>
                {% else %}
                    <form method="post" action="/" id="modalSearchForm">
                        <input type="text" name="query" placeholder="add a topic" class="search-box" required>
                        <div class="search-options" style="display: none;">
                            <select name="count" class="option-select">
                                <option value="10">10 results</option>
                                <option value="20">20 results</option>
                                <option value="30">30 results</option>
                                <option value="50" selected>50 results</option>
                            </select>
                            <select name="freshness" class="option-select">
                                <option value="pd">past day</option>
                                <option value="pw">past week</option>
                                <option value="pm">past month</option>
                                <option value="py" selected>past year</option>
                            </select>
                            <select name="similarity_threshold" class="option-select">
                                <option value="0.2">low similarity</option>
                                <option value="0.3">medium similarity</option>
                                <option value="0.4" selected>high similarity</option>
                            </select>
                            <input type="hidden" name="force_refresh" value="false">
                        </div>
                        <div class="modal-buttons">
                            <button type="button" class="modal-button cancel-button" onclick="closeNewBriefModal()">cancel</button>
                            <div class="submit-button-container">
                                <button type="submit" class="modal-button submit-button" id="submitBriefButton">add brief</button>
                                <div class="loading-spinner" id="briefLoadingSpinner"></div>
                            </div>
                        </div>
                    </form>
                {% endif %}
            {% else %}
                <p style="margin-bottom: 20px;">You need to <a href="{{ url_for('auth.login') }}" style="color: var(--accent-color);">login</a> to add briefs.</p>
                <div class="modal-buttons">
                    <button type="button" class="modal-button cancel-button" onclick="closeNewBriefModal()">close</button>
                    <a href="{{ url_for('auth.login') }}" class="modal-button submit-button" style="display: inline-block; text-align: center; text-decoration: none; margin-right: 10px;">login</a>
                    <a href="#" onclick="openRequestAccessModal(); closeNewBriefModal(); return false;" class="modal-button submit-button" style="display: inline-block; text-align: center; text-decoration: none;">request access</a>
                </div>
            {% endif %}
        </div>
    </div>
    
    <!-- Modal for requesting access -->
    <div id="requestAccessModal" class="modal">
        <div class="modal-content">
            <h2 class="modal-title">request access</h2>
            <form method="post" action="{{ url_for('request_access_redirect') }}" id="requestAccessForm">
                <input type="email" name="email" placeholder="your email address" class="search-box" required>
                <div class="modal-buttons">
                    <button type="button" class="modal-button cancel-button" onclick="closeRequestAccessModal()">cancel</button>
                    <div class="submit-button-container">
                        <button type="submit" class="modal-button submit-button" id="submitAccessButton">submit request</button>
                        <div class="loading-spinner" id="accessLoadingSpinner"></div>
                    </div>
                </div>
            </form>
        </div>
    </div>
    
    <!-- Modal for notification settings -->
    <div id="notificationModal" class="notification-modal">
        <div class="notification-modal-content">
            <h2 class="notification-modal-title">notification settings</h2>
            <p>set update frequency for <span id="notificationTopic" class="notification-topic"></span></p>
            
            <div class="notification-options">
                <div class="notification-option">
                    <input type="radio" id="notifyHourly" name="notifyFrequency" value="hourly">
                    <label for="notifyHourly">hourly updates</label>
                </div>
                <div class="notification-option">
                    <input type="radio" id="notifyDaily" name="notifyFrequency" value="daily" checked>
                    <label for="notifyDaily">daily summary</label>
                </div>
            </div>
            
            <div class="modal-buttons">
                <button type="button" class="modal-button cancel-button" onclick="closeNotificationModal()">cancel</button>
                <button type="button" class="modal-button submit-button" onclick="saveNotificationSettings()">save settings</button>
            </div>
        </div>
    </div>
    
    <script>
        function toggleTopic(element) {
            const content = element.nextElementSibling;
            const isExpanded = element.classList.contains('expanded');
            
            // Toggle expanded class
            if (isExpanded) {
                element.classList.remove('expanded');
                content.style.display = 'none';
            } else {
                element.classList.add('expanded');
                content.style.display = 'block';
            }
        }
        
        function deleteHistoryItem(query) {
            if (confirm('are you sure you want to delete this brief?')) {
                fetch('/api/delete_history/' + encodeURIComponent(query), {
                    method: 'POST',
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        window.location.href = '/history';
                    }
                });
            }
        }
        
        function clearHistory() {
            if (confirm('are you sure you want to clear all briefs?')) {
                fetch('/api/clear_history', {
                    method: 'POST',
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        window.location.href = '/';
                    }
                });
            }
        }
        
        function openNewBriefModal() {
            document.getElementById('newBriefModal').style.display = 'flex';
        }
        
        function closeNewBriefModal() {
            document.getElementById('newBriefModal').style.display = 'none';
        }
        
        function openRequestAccessModal() {
            document.getElementById('requestAccessModal').style.display = 'flex';
        }
        
        function closeRequestAccessModal() {
            document.getElementById('requestAccessModal').style.display = 'none';
        }
        
        function closeNotificationModal() {
            document.getElementById('notificationModal').style.display = 'none';
        }
        
        function saveNotificationSettings() {
            const topic = document.getElementById('notificationTopic').textContent;
            const frequency = document.querySelector('input[name="notifyFrequency"]:checked').value;
            
            // Save preference to localStorage
            localStorage.setItem(`notify_${topic}`, frequency);
            
            // Send to server
            fetch('/api/notifications/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    topic: topic,
                    frequency: frequency
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log(`Saved notification preference for "${topic}": ${frequency}`);
                } else {
                    console.error('Failed to save notification settings');
                }
            })
            .catch(error => {
                console.error('Error saving notification settings:', error);
            });
            
            // Close the modal
            closeNotificationModal();
            
            // Prevent navigation to the details page
            event.stopPropagation();
        }

        let reloadTimer;
        let currentSeconds = 3600; // 60 minutes default
        let currentQuery = ""; // Store current query for timer management
        let timerIntervals = {}; // Store intervals for each query
        
        // Initialize functions when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Get the current query from the page
            const queryElements = document.querySelectorAll('.brief-title');
            for (let element of queryElements) {
                if (window.location.pathname.includes('/history/')) {
                    currentQuery = element.textContent.trim();
                    break;
                }
            }
            
            // Dark mode toggle functionality
            const themeToggle = document.getElementById('theme-toggle');
            const themeIcon = themeToggle.querySelector('i');
            
            // Check for saved theme preference or use device preference
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                document.documentElement.setAttribute('data-theme', 'dark');
                themeIcon.className = 'fas fa-sun';
            }
            
            // Toggle theme when button is clicked
            themeToggle.addEventListener('click', function() {
                const currentTheme = document.documentElement.getAttribute('data-theme');
                if (currentTheme === 'dark') {
                    document.documentElement.removeAttribute('data-theme');
                    localStorage.setItem('theme', 'light');
                    themeIcon.className = 'fas fa-moon';
                } else {
                    document.documentElement.setAttribute('data-theme', 'dark');
                    localStorage.setItem('theme', 'dark');
                    themeIcon.className = 'fas fa-sun';
                }
            });
            
            // Initialize all timers on page
            const timerElements = document.getElementsByClassName('reload-timer');
            if (timerElements.length > 0) {
                // If we're on a specific history page, just handle that one
                if (currentQuery) {
                    initializeTimer(currentQuery);
                    timerIntervals[currentQuery] = setInterval(() => updateReloadTimer(currentQuery), 1000);
                } else {
                    // Otherwise initialize timers for all briefs on the home page
                    for (let element of timerElements) {
                        const query = element.getAttribute('data-query');
                        if (query) {
                            initializeTimer(query);
                            timerIntervals[query] = setInterval(() => updateReloadTimer(query), 1000);
                        }
                    }
                }
            }
            
            // Close modal if clicked outside
            window.onclick = function(event) {
                const briefModal = document.getElementById('newBriefModal');
                const accessModal = document.getElementById('requestAccessModal');
                const notificationModal = document.getElementById('notificationModal');
                
                if (event.target == briefModal) {
                    closeNewBriefModal();
                }
                if (event.target == accessModal) {
                    closeRequestAccessModal();
                }
                if (event.target == notificationModal) {
                    closeNotificationModal();
                }
            }
            
            // Show loading spinner when form is submitted
            const searchForm = document.getElementById('modalSearchForm');
            const submitButton = document.getElementById('submitBriefButton');
            const loadingSpinner = document.getElementById('briefLoadingSpinner');
            const cancelButton = document.querySelector('.cancel-button');
            
            if (searchForm) {
                searchForm.addEventListener('submit', function(e) {
                    // Show spinner
                    loadingSpinner.style.display = 'block';
                    
                    // Disable buttons
                    submitButton.disabled = true;
                    submitButton.style.opacity = '0.7';
                    submitButton.textContent = 'adding...';
                    
                    cancelButton.disabled = true;
                    cancelButton.style.opacity = '0.7';
                    
                    // Add a console log to verify this code is running
                    console.log('Form submitted, showing spinner and disabling buttons');
                    
                    // Let the form submission proceed
                    return true;
                });
            }
            
            // Show loading spinner when access request form is submitted
            const accessForm = document.getElementById('requestAccessForm');
            const accessButton = document.getElementById('submitAccessButton');
            const accessSpinner = document.getElementById('accessLoadingSpinner');
            
            if (accessForm) {
                accessForm.addEventListener('submit', function() {
                    // Show spinner
                    accessSpinner.style.display = 'block';
                    
                    // Disable buttons
                    accessButton.disabled = true;
                    accessButton.style.opacity = '0.7';
                    accessButton.textContent = 'submitting...';
                    
                    const accessCancelButton = accessForm.querySelector('.cancel-button');
                    accessCancelButton.disabled = true;
                    accessCancelButton.style.opacity = '0.7';
                });
            }
        });
        
        function initTimerForElement(element, query) {
            const timerStartKey = `timerStart_${query}`;
            const cycleStartKey = `cycleStart_${query}`;
            
            // Check if we have a stored cycle start time for this query
            let cycleStartTime = localStorage.getItem(cycleStartKey);
            let currentSecs = 3600;
            
            if (cycleStartTime) {
                // We have an existing cycle - use it
                const now = new Date().getTime();
                cycleStartTime = parseInt(cycleStartTime, 10);
                
                // Calculate elapsed seconds since the start of this cycle
                const elapsedSeconds = Math.floor((now - cycleStartTime) / 1000);
                
                // Calculate remaining seconds until next 60-minute mark (3600 seconds)
                currentSecs = Math.max(0, 3600 - elapsedSeconds);
            }
            
            // Update the element with the current time
            const minutes = Math.floor(currentSecs / 60);
            const seconds = currentSecs % 60;
            element.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
        
        function initializeTimer(query) {
            if (!query) return;
            
            // Use query-specific keys for localStorage
            const timerStartKey = `timerStart_${query}`;
            const cycleStartKey = `cycleStart_${query}`;
            const refreshDueKey = `refreshDue_${query}`;
            
            // Check if we have a stored cycle start time for this query
            let cycleStartTime = localStorage.getItem(cycleStartKey);
            let refreshDueTime = localStorage.getItem(refreshDueKey);
            
            const now = new Date().getTime();
            
            // If no refresh due time exists or it's invalid, create a new one
            if (!refreshDueTime || isNaN(parseInt(refreshDueTime, 10))) {
                // Set refresh due time to 60 minutes from now
                refreshDueTime = now + (3600 * 1000); // 60 minutes in milliseconds
                localStorage.setItem(refreshDueKey, refreshDueTime.toString());
                console.log(`Setting new refresh due time for ${query}: ${new Date(refreshDueTime).toLocaleTimeString()}`);
            } else {
                // Parse the existing refresh due time
                refreshDueTime = parseInt(refreshDueTime, 10);
                
                // If the refresh is already due, trigger it now
                if (now >= refreshDueTime) {
                    console.log(`Refresh already due for ${query}, triggering now`);
                    // Clear timer data before refresh
                    localStorage.removeItem(cycleStartKey);
                    localStorage.removeItem(refreshDueKey);
                    // Perform refresh
                    forceRefresh(query);
                    return;
                }
                console.log(`Refresh due for ${query} at: ${new Date(refreshDueTime).toLocaleTimeString()}`);
            }
            
            // If we don't have cycle start time, initialize it
            if (!cycleStartTime) {
                localStorage.setItem(cycleStartKey, now.toString());
            }
            
            // Also update the original search time if this is the first time
            if (!localStorage.getItem(timerStartKey)) {
                localStorage.setItem(timerStartKey, now.toString());
            }
            
            // Calculate and set the remaining seconds
            const remainingMilliseconds = refreshDueTime - now;
            const remainingSeconds = Math.max(0, Math.floor(remainingMilliseconds / 1000));
            
            if (!window.timerSeconds) {
                window.timerSeconds = {};
            }
            window.timerSeconds[query] = remainingSeconds;
            
            // Update the display immediately
            updateReloadTimer(query);
        }
        
        function startNewCycle(query) {
            // Start a fresh 60-minute cycle
            const now = new Date().getTime();
            const refreshDueTime = now + (3600 * 1000); // 60 minutes in milliseconds
            
            if (!window.timerSeconds) {
                window.timerSeconds = {};
            }
            window.timerSeconds[query] = 3600; // 60 minutes in seconds
            
            // Record when this cycle started and when it's due
            localStorage.setItem(`cycleStart_${query}`, now.toString());
            localStorage.setItem(`refreshDue_${query}`, refreshDueTime.toString());
            
            // Also update the original search time if this is the first time
            if (!localStorage.getItem(`timerStart_${query}`)) {
                localStorage.setItem(`timerStart_${query}`, now.toString());
            }
            
            console.log(`Starting new 60-minute refresh cycle for ${query}, due at: ${new Date(refreshDueTime).toLocaleTimeString()}`);
        }

        function updateReloadTimer(query) {
            if (!query) return;
            
            if (!window.timerSeconds) {
                window.timerSeconds = {};
            }
            
            // Initialize if not already set
            if (window.timerSeconds[query] === undefined) {
                initializeTimer(query);
                return;
            }
            
            // Instead of just decrementing, recalculate based on actual time
            const refreshDueKey = `refreshDue_${query}`;
            const refreshDueTime = parseInt(localStorage.getItem(refreshDueKey), 10);
            const now = new Date().getTime();
            
            // If we have a valid refresh due time
            if (refreshDueTime && !isNaN(refreshDueTime)) {
                // Calculate actual remaining seconds
                const remainingMilliseconds = refreshDueTime - now;
                const remainingSeconds = Math.max(0, Math.floor(remainingMilliseconds / 1000));
                window.timerSeconds[query] = remainingSeconds;
                
                // If time is up and we should refresh
                if (remainingSeconds <= 0) {
                    if (timerIntervals[query]) {
                        clearInterval(timerIntervals[query]);
                        delete timerIntervals[query];
                    }
                    
                    // Clear the cycle info before reloading
                    localStorage.removeItem(`cycleStart_${query}`);
                    localStorage.removeItem(`refreshDue_${query}`);
                    
                    // Use the same forceRefresh function for consistency
                    forceRefresh(query);
                    return;
                }
            } else {
                // If no due time exists, start a new cycle
                startNewCycle(query);
                return;
            }
            
            const minutes = Math.floor(window.timerSeconds[query] / 60);
            const seconds = window.timerSeconds[query] % 60;
            const timerString = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            const timerElements = document.querySelectorAll(`.reload-timer[data-query="${query}"]`);
            for (let element of timerElements) {
                element.textContent = timerString;
            }
        }

        function forceRefresh(query) {
            // Add force_refresh parameter to URL and reload
            const url = new URL(window.location.href);
            url.searchParams.set('force_refresh', 'true');
            
            // If a specific query is provided, force refresh that query
            if (query) {
                // If we're on home page, add query parameter
                if (window.location.pathname === '/' || window.location.pathname === '') {
                    url.searchParams.set('query', query);
                }
                
                // If we're on the detail page for this query, force refresh it
                if (window.location.pathname.includes(`/history/${query}`)) {
                    // Just reload the page with force_refresh=true
                    window.location.href = url.toString();
                    return;
                }
            }
            
            // Always do a full page reload
            window.location.href = url.toString();
        }
        
        function openNotificationModal(topic) {
            const modal = document.getElementById('notificationModal');
            const topicElement = document.getElementById('notificationTopic');
            
            // Set the topic in the modal
            topicElement.textContent = topic;
            
            // Check if we have saved preferences for this topic
            const savedPrefs = localStorage.getItem(`notify_${topic}`);
            if (savedPrefs) {
                const frequency = savedPrefs;
                document.querySelector(`input[name="notifyFrequency"][value="${frequency}"]`).checked = true;
            } else {
                // Default to daily
                document.querySelector('input[name="notifyFrequency"][value="daily"]').checked = true;
            }
            
            // Show the modal
            modal.style.display = 'flex';
        }
    </script>
</body>
</html>''')

    with open('templates/error.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>error - loop</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg-color: #f8f8f8;
            --text-color: #333;
            --accent-color: #6d28d9;
            --secondary-color: #a78bfa;
            --border-color: #e5e7eb;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: "Proxima Nova", Tahoma, -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            max-width: 600px;
            margin: 0 auto;
            padding: 40px 20px;
            text-align: center;
        }
        
        h1, h2, h3, p, a {
            text-transform: lowercase;
        }
        
        h1 {
            color: #b91c1c;
            margin-bottom: 20px;
            font-size: 24px;
        }
        
        .error-container {
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border: 1px solid #fee2e2;
        }
        
        .error-message {
            margin-bottom: 30px;
            font-size: 18px;
            color: #4b5563;
        }
        
        .button {
            display: inline-block;
            background-color: var(--accent-color);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .button:hover {
            background-color: #5b21b6;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="error-container">
        <h1>error</h1>
        <div class="error-message">
            {{ error }}
        </div>
        <a href="/" class="button">go back</a>
    </div>
</body>
</html>''')
    
    # Create user_data directory for user files
    os.makedirs('user_data', exist_ok=True)
    
    # Run the Flask app
    print("Starting Flask application on http://0.0.0.0:5000")
    print("NOTE: You need to install scikit-learn for the topic grouping to work:")
    print("pip install scikit-learn")
    print("NOTE: To enable Llama summaries, install:")
    print("pip install transformers torch huggingface_hub")
    print("NOTE: To enable OpenAI summaries, install:")
    print("pip install openai")
    print("NOTE: Set the following environment variables for Google authentication:")
    print("export GOOGLE_CLIENT_ID=your_google_client_id")
    print("export GOOGLE_CLIENT_SECRET=your_google_client_secret")
    app.run(host='0.0.0.0', port=5000, debug=True) 