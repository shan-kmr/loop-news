#!/usr/bin/env python3
"""
News Timeline Web Application

A Flask-based web app that provides a UI for searching news articles and
displaying them in a timeline view, sorted by recency.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
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
def load_search_history():
    history_file = get_history_file()
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
    return {}

# Save search history to JSON file
def save_search_history(history):
    history_file = get_history_file()
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
def get_history_file():
    if current_user.is_authenticated:
        history_file = current_user.get_history_file()
        if history_file:  # Check that we got a valid path
            return history_file
    
    # Default location for anonymous users or if user path is invalid
    default_dir = 'user_data/anonymous'
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
        Updated results dictionary with refreshed current day articles
    """
    global brave_api
    
    if brave_api is None:
        brave_api_key = app.config.get("BRAVE_API_KEY")
        if not brave_api_key:
            return old_results
        brave_api = BraveNewsAPI(api_key=brave_api_key)
    
    try:
        # For frequent refreshes (every 10 minutes), default to past day
        # For longer term refreshes, use the original freshness or at least past week
        refresh_freshness = 'pd'  # Default to past day for 10-minute refreshes
        
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
            return fresh_results or old_results
        
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
        articles_to_replace = []
        
        for article in old_articles:
            age_seconds = extract_age_in_seconds(article)
            
            # Keep very old articles (older than our refresh window)
            if age_seconds > cutoff_seconds:
                articles_to_keep.append(article)
            else:
                articles_to_replace.append(article)
        
        # Log what we're doing
        print(f"Refreshing articles for '{query}' using freshness: {refresh_freshness}")
        print(f"Keeping {len(articles_to_keep)} older articles beyond the refresh window")
        print(f"Replacing {len(articles_to_replace)} articles within the refresh window with {len(new_articles)} new articles")
        
        # Combine older articles with fresh results
        # De-duplicate articles by URL to prevent duplicates
        seen_urls = set()
        combined_articles = []
        
        # First add all the fresh articles (they take priority)
        for article in new_articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined_articles.append(article)
        
        # Then add the older articles we're keeping (if not already present by URL)
        for article in articles_to_keep:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined_articles.append(article)
        
        # Create updated results
        updated_results = fresh_results.copy()
        updated_results['results'] = combined_articles
        
        return updated_results
    
    except Exception as e:
        print(f"Error updating results: {str(e)}")
        return old_results

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route that handles both the search form and displaying results."""
    global brave_api
    
    # Initialize the API client if needed
    if brave_api is None:
        brave_api_key = app.config.get("BRAVE_API_KEY")
        if not brave_api_key:
            return render_template('error.html', 
                                  error="BRAVE_API_KEY environment variable or config not set. Please set it and restart the app.")
        brave_api = BraveNewsAPI(api_key=brave_api_key)
    
    # Default query and results
    query = request.args.get('query', 'breaking news')
    results = None
    search_time = None
    error = None
    sorted_articles = []
    topic_groups = []
    history = load_search_history()
    
    # Process form submission
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        count = int(request.form.get('count', 10))
        freshness = request.form.get('freshness', 'pw')
        similarity_threshold = float(request.form.get('similarity_threshold', 0.3))
        force_refresh = request.form.get('force_refresh') == 'true'
        
        if query:
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
                        
                        # Check if we need to refresh the current day's results (older than 10 minutes)
                        if time_diff.total_seconds() > 600:  # 10 minutes in seconds
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
                            updated_results = update_current_day_results(query, count, freshness, results)
                            
                            if updated_results != results:
                                results = updated_results
                                search_time = datetime.now()
                                
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
                        
                        # Cache the results with summaries
                        history[cache_key] = {
                            'query': query,
                            'count': count,
                            'freshness': freshness,
                            'timestamp': search_time.isoformat(),
                            'results': results,
                            'day_summaries': day_summaries
                        }
                        save_search_history(history)
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
                          sorted_articles=sorted_articles,
                          topic_groups=topic_groups,
                          search_time=search_time,
                          error=error,
                          history_entries=history_entries,
                          active_tab="search",
                          user=current_user)

@app.route('/history', methods=['GET'])
def history():
    """View to display search history with the latest search."""
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
                              sorted_articles=[],
                              topic_groups=[],
                              search_time=None,
                              error=None,
                              history_entries=[],
                              active_tab="history",
                              user=current_user)

@app.route('/history/<path:query>', methods=['GET'])
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
        print(f"History entry: '{stored_query}' with key: {key}")
    
    # Sort history by most recent first
    history_entries.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Find the requested history item
    results = None
    search_time = None
    topic_groups = []
    sorted_articles = []
    current_query = ''
    similarity_threshold = float(request.args.get('similarity_threshold', 0.3))
    
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
                
                # Check if we need to refresh
                needs_day_refresh = time_diff.total_seconds() > 600  # 10 minutes in seconds
                
                # Handle refreshing and processing
                if needs_day_refresh:
                    print(f"History item is {int(time_diff.total_seconds() / 60)} minutes old. Refreshing current day results for '{query}'")
                    updated_results = update_current_day_results(query, count, freshness, results)
                    
                    if updated_results != results:
                        results = updated_results
                        search_time = datetime.now()
                        
                        # Update the cache with refreshed results
                        entry['results'] = results
                        entry['timestamp'] = search_time.isoformat()
                        save_search_history(history)
                
                # Process day summaries
                day_summaries = entry.get('day_summaries', {})
                if day_summaries is None:
                    day_summaries = {}
                
                # Process articles
                if results and 'results' in results:
                    sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
                    topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, current_query, day_summaries)
                    
                    # Update summaries if needed
                    new_summaries_generated = False
                    for topic in topic_groups:
                        day = topic['day_group']
                        if day in day_summaries and day_summaries[day] is not None:
                            new_summaries_generated = True
                            break
                    
                    if new_summaries_generated and 'day_summaries' not in entry:
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
                    return render_template('error.html', error="BRAVE_API_KEY not set")
            
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
                print(f"Cached new results for query: '{query}'")
        except Exception as e:
            print(f"Error performing search: {str(e)}")
    
    # Ensure we always have a search_time for the ticker to work
    if search_time is None:
        search_time = datetime.now()
    
    return render_template('index.html', 
                          query=current_query, 
                          results=results, 
                          sorted_articles=sorted_articles,
                          topic_groups=topic_groups,
                          search_time=search_time,
                          error=None,
                          history_entries=history_entries,
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
            topic['day_summary'] = day_summaries.get(day)
            # If this day doesn't have a summary yet, add it to our list to process
            if topic['day_summary'] is None and day not in days_needing_summaries:
                days_needing_summaries.add(day)
        
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
                print(f"Generating new summary for day: {day} using {MODEL_PROVIDER}")
                summary = summarize_daily_news(day_art, query)
                day_summaries[day] = summary
                
                # Update topic groups with new summaries
                for topic in topic_groups:
                    if topic['day_group'] == day:
                        topic['day_summary'] = summary
        
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
    """Format a datetime object."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    return value.strftime(format)

if __name__ == '__main__':
    # Initialize models in background threads to avoid blocking app startup
    if LLAMA_AVAILABLE or OPENAI_AVAILABLE:
        init_thread = threading.Thread(target=init_models)
        init_thread.daemon = True
        init_thread.start()
    
    # Create templates
    with open('templates/index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>News Timeline</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f9f9f9;
            display: flex;
        }
        .sidebar {
            width: 280px;
            background-color: #2c3e50;
            color: white;
            height: 100vh;
            overflow-y: auto;
            position: fixed;
            left: 0;
            top: 0;
        }
        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid #395069;
        }
        .sidebar-title {
            margin: 0;
            font-size: 22px;
        }
        .sidebar-user {
            margin-top: 10px;
            display: flex;
            align-items: center;
            font-size: 14px;
        }
        .user-avatar {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            margin-right: 10px;
            background-color: #3498db;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .user-avatar img {
            width: 100%;
            height: auto;
        }
        .login-button, .logout-button {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 12px;
            margin-top: 5px;
        }
        .login-button {
            background-color: #3498db;
            color: white;
        }
        .logout-button {
            background-color: #7f8c8d;
            color: white;
            margin-left: 10px;
        }
        .tab-nav {
            display: flex;
            padding: 0 20px;
            border-bottom: 1px solid #395069;
        }
        .tab-button {
            padding: 15px 0;
            flex: 1;
            text-align: center;
            cursor: pointer;
            color: #ccc;
            font-weight: bold;
            border-bottom: 3px solid transparent;
        }
        .tab-button.active {
            color: white;
            border-bottom-color: #3498db;
        }
        .history-list {
            padding: 0;
            margin: 0;
            list-style: none;
        }
        .history-item {
            padding: 12px 20px;
            border-bottom: 1px solid #34495e;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .history-item:hover {
            background-color: #34495e;
        }
        .history-query {
            font-weight: bold;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .history-date {
            font-size: 12px;
            color: #bdc3c7;
        }
        .content {
            flex: 1;
            margin-left: 280px;
            padding: 20px;
            max-width: 1000px;
        }
        .content-inner {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .search-container {
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .search-form {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: flex-end;
        }
        .form-group {
            flex: 1;
            min-width: 200px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #2c3e50;
        }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        .button-secondary {
            background-color: #7f8c8d;
        }
        .button-secondary:hover {
            background-color: #6c7a7b;
        }
        .button-danger {
            background-color: #e74c3c;
        }
        .button-danger:hover {
            background-color: #c0392b;
        }
        .timeline-container {
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .timeline-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .timeline-day {
            margin-top: 30px;
            margin-bottom: 10px;
            font-weight: bold;
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .day-summary {
            background-color: #f0f7fb;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin-bottom: 20px;
            border-radius: 0 5px 5px 0;
            font-style: italic;
            color: #2c3e50;
        }
        .timeline-item {
            border-left: 3px solid #3498db;
            padding-left: 15px;
            margin-bottom: 25px;
            position: relative;
        }
        .timeline-item:before {
            content: '';
            position: absolute;
            width: 12px;
            height: 12px;
            background-color: #3498db;
            border-radius: 50%;
            left: -7.5px;
            top: 0;
        }
        .topic-header {
            cursor: pointer;
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .topic-title {
            font-size: 18px;
            font-weight: bold;
            margin-right: 10px;
        }
        .topic-count {
            background-color: #3498db;
            color: white;
            border-radius: 12px;
            padding: 2px 8px;
            font-size: 12px;
            margin-left: 8px;
        }
        .topic-time {
            color: #7f8c8d;
            font-size: 14px;
            margin-left: auto;
        }
        .topic-content {
            display: none;
            margin-left: 20px;
            margin-bottom: 15px;
        }
        .timeline-time {
            color: #3498db;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .timeline-source {
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .timeline-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .timeline-desc {
            margin-bottom: 10px;
            color: #555;
        }
        .timeline-link {
            display: inline-block;
            font-size: 14px;
            color: #3498db;
            text-decoration: none;
            word-break: break-all;
        }
        .timeline-link:hover {
            text-decoration: underline;
        }
        .article-item {
            border-left: 2px solid #bdc3c7;
            padding-left: 15px;
            margin-bottom: 15px;
        }
        .expand-icon {
            margin-right: 8px;
            transition: transform 0.3s;
        }
        .expanded .expand-icon {
            transform: rotate(90deg);
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .no-results {
            text-align: center;
            padding: 50px;
            color: #7f8c8d;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .controls-right {
            display: flex;
            gap: 10px;
        }
        .login-notice {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 10px 15px;
            margin-bottom: 20px;
            color: #2c3e50;
        }
        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                height: auto;
                position: relative;
            }
            .content {
                margin-left: 0;
            }
            body {
                flex-direction: column;
            }
            .search-form {
                flex-direction: column;
            }
            .form-group {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <h1 class="sidebar-title">News Timeline</h1>
            <div class="sidebar-user">
                {% if user.is_authenticated %}
                <div class="user-avatar">
                    {% if user.profile_pic %}
                    <img src="{{ user.profile_pic }}" alt="{{ user.name }}">
                    {% else %}
                    <i class="fas fa-user"></i>
                    {% endif %}
                </div>
                <div>
                    <div>{{ user.name }}</div>
                    <a href="{{ url_for('auth.logout') }}" class="logout-button">Logout</a>
                </div>
                {% else %}
                <div class="user-avatar">
                    <i class="fas fa-user"></i>
                </div>
                <div>
                    <div>Guest</div>
                    <a href="{{ url_for('auth.login') }}" class="login-button">Login with Google</a>
                </div>
                {% endif %}
            </div>
        </div>
        <div class="tab-nav">
            <div class="tab-button {% if active_tab == 'search' %}active{% endif %}" onclick="window.location.href='/'">Search</div>
            <div class="tab-button {% if active_tab == 'history' %}active{% endif %}" title="Click on a search entry in the sidebar to view history">History</div>
        </div>
        
        {% if history_entries %}
        <ul class="history-list">
            {% for entry in history_entries %}
            <li class="history-item" onclick="window.location.href='/history/{{ entry.query }}'">
                <div class="history-query">{{ entry.query }}</div>
                <div class="history-date">{{ entry.timestamp|format_datetime }}</div>
            </li>
            {% endfor %}
        </ul>
        {% else %}
        <div style="padding: 20px; text-align: center; color: #bdc3c7;">
            <p>No search history yet.</p>
        </div>
        {% endif %}
    </div>
    
    <div class="content">
        <div class="content-inner">
            {% if not user.is_authenticated %}
            <div class="login-notice">
                <p><i class="fas fa-info-circle"></i> Sign in with Google to save your search history across devices.</p>
            </div>
            {% endif %}
            
            {% if active_tab == 'search' %}
            <h1>Search News</h1>
            
            <div class="search-container">
                <form class="search-form" method="post" action="/">
                    <div class="form-group">
                        <label for="query">Search Query</label>
                        <input type="text" id="query" name="query" value="{{ query }}" required>
                    </div>
                    <div class="form-group">
                        <label for="count">Number of Results</label>
                        <select id="count" name="count">
                            <option value="10">10</option>
                            <option value="20">20</option>
                            <option value="30">30</option>
                            <option value="50">50</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="freshness">Time Period</label>
                        <select id="freshness" name="freshness">
                            <option value="pd">Past Day</option>
                            <option value="pw" selected>Past Week</option>
                            <option value="pm">Past Month</option>
                            <option value="py">Past Year</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="similarity_threshold">Topic Similarity</label>
                        <select id="similarity_threshold" name="similarity_threshold">
                            <option value="0.2">Low - More Topics</option>
                            <option value="0.3" selected>Medium</option>
                            <option value="0.4">High - Fewer Topics</option>
                        </select>
                    </div>
                    <input type="hidden" name="force_refresh" id="force_refresh" value="false">
                    <button type="submit">Search</button>
                </form>
            </div>
            
            {% if results %}
            <div class="controls">
                <div class="controls-left">
                    <button class="button-secondary" onclick="document.getElementById('force_refresh').value='true'; document.querySelector('.search-form').submit();">
                        Refresh Results
                    </button>
                </div>
            </div>
            {% endif %}
            
            {% elif active_tab == 'history' %}
            <h1>Search History</h1>
            
            <div class="controls">
                <div class="controls-left">
                    {% if query %}
                    <h2>Results for "{{ query }}"</h2>
                    {% else %}
                    <p>Select a search from the sidebar to view results.</p>
                    {% endif %}
                </div>
                <div class="controls-right">
                    {% if query %}
                    <button class="button-secondary" onclick="deleteHistoryItem('{{ query }}')">Delete This Search</button>
                    {% endif %}
                    <button class="button-danger" onclick="clearHistory()">Clear All History</button>
                </div>
            </div>
            
            {% if query and search_time %}
            <div class="timeline-header">
                <div>
                    <p>{{ search_time.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                    <p>Current time: <span class="live-ticker">--:--:--</span></p>
                    <p>Auto-refresh in: <span class="reload-timer">10:00</span></p>
                </div>
            </div>
            {% endif %}
            {% endif %}
            
            {% if error %}
            <div class="error">
                {{ error }}
            </div>
            {% endif %}
            
            {% if results %}
            <div class="timeline-container">
                <div class="timeline-header">
                    <h2>Results for "{{ query }}"</h2>
                    <div>
                        <p>{{ search_time.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                        <p>Current time: <span class="live-ticker">--:--:--</span></p>
                        <p>Auto-refresh in: <span class="reload-timer">10:00</span></p>
                    </div>
                </div>
                
                {% set ns = namespace(current_day=None, current_day_summary=None) %}
                
                {% if topic_groups | length == 0 %}
                <div class="no-results">
                    <p>No news articles found for your query.</p>
                </div>
                {% endif %}
                
                {% for topic in topic_groups %}
                    {% set day = topic.day_group %}
                    
                    {% if day != ns.current_day %}
                        {% set ns.current_day = day %}
                        {% set ns.current_day_summary = topic.day_summary %}
                        <div class="timeline-day">
                            {{ day.upper() }}
                        </div>
                        
                        {% if topic.day_summary %}
                        <div class="day-summary">
                            <strong>Daily Summary:</strong> {{ topic.day_summary }}
                        </div>
                        {% endif %}
                    {% endif %}
                    
                    <div class="timeline-item">
                        <div class="topic-header" onclick="toggleTopic(this)">
                            <span class="expand-icon">▶</span>
                            <div class="topic-title">{{ topic.title }}</div>
                            {% if topic.count > 1 %}
                            <span class="topic-count">{{ topic.count }} articles</span>
                            {% endif %}
                            <div class="topic-time">{{ topic.newest_age }}</div>
                        </div>
                        
                        <div class="topic-content">
                            {% for article in topic.articles %}
                                <div class="article-item">
                                    <div class="timeline-source">{{ article.get('meta_url', {}).get('netloc', 'Unknown source') }}</div>
                                    {% if loop.index > 1 %}
                                        <div class="timeline-title">{{ article.get('title', 'No title') }}</div>
                                    {% endif %}
                                    <div class="timeline-desc">{{ article.get('description', 'No description available') }}</div>
                                    <a href="{{ article.get('url', '#') }}" class="timeline-link" target="_blank">{{ article.get('url', 'No URL available') }}</a>
                                </div>
                            {% endfor %}
                        </div>
                    </div>
                {% endfor %}
            </div>
            {% endif %}
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
            if (confirm('Are you sure you want to delete this search from history?')) {
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
            if (confirm('Are you sure you want to clear all search history?')) {
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

        let reloadTimer;
        let currentSeconds = 600; // 10 minutes default
        let currentQuery = ""; // Store current query for timer management
        
        // Initialize functions when page loads
        document.addEventListener('DOMContentLoaded', function() {
            updateLiveTicker();
            
            // Get the current query from the page
            const queryElements = document.querySelectorAll('h2');
            for (let element of queryElements) {
                if (element.textContent.startsWith('Results for "')) {
                    currentQuery = element.textContent.replace('Results for "', '').replace('"', '');
                    break;
                }
            }
            
            // Only start the reload timer if we're on a results or history page with results
            if (document.querySelector('.timeline-container') || 
                (document.querySelector('[data-active-tab="history"]') && document.querySelector('.reload-timer'))) {
                startReloadTimer();
            }
        });
        
        // Function to update the live ticker
        function updateLiveTicker() {
            const tickerElements = document.getElementsByClassName('live-ticker');
            if (tickerElements.length > 0) {
                const now = new Date();
                const timeString = now.toTimeString().split(' ')[0];
                
                for (let element of tickerElements) {
                    element.textContent = timeString;
                }
                
                // Update every second
                setTimeout(updateLiveTicker, 1000);
            }
        }
        
        // Function to handle the auto-reload timer
        function startReloadTimer() {
            const timerElements = document.getElementsByClassName('reload-timer');
            if (timerElements.length > 0 && currentQuery) {
                initializeTimer(currentQuery);
                reloadTimer = setInterval(() => updateReloadTimer(currentQuery), 1000);
            }
        }

        function initializeTimer(query) {
            if (!query) return;
            
            // Use query-specific keys for localStorage
            const timerStartKey = `timerStart_${query}`;
            const timerDurationKey = `timerDuration_${query}`;
            
            // Check if we have a stored timer for this query
            const timerStart = localStorage.getItem(timerStartKey);
            const timerDuration = localStorage.getItem(timerDurationKey);
            
            if (timerStart && timerDuration) {
                const now = new Date().getTime();
                const startTime = parseInt(timerStart, 10);
                
                // Calculate elapsed seconds precisely from the start time
                const elapsedSeconds = Math.floor((now - startTime) / 1000);
                
                // Timer should refresh exactly every 10 minutes (600 seconds)
                const totalCycles = Math.floor(elapsedSeconds / 600);
                const nextCycleSeconds = (totalCycles + 1) * 600;
                
                // Calculate remaining seconds until next 10-minute mark
                currentSeconds = nextCycleSeconds - elapsedSeconds;
                
                // If the timer is about to expire or has already expired, refresh now
                if (currentSeconds <= 5) {
                    clearInterval(reloadTimer);
                    location.reload();
                    return;
                }
            } else {
                // First search for this query - start a new 10-minute timer
                currentSeconds = 600; // 10 minutes
                localStorage.setItem(timerStartKey, new Date().getTime().toString());
                localStorage.setItem(timerDurationKey, currentSeconds.toString());
            }
            
            // Update the display immediately
            updateReloadTimer(query);
        }
        
        function resetTimer(query) {
            if (!query) return;
            
            currentSeconds = 600; // 10 minutes
            
            // Use query-specific keys for localStorage
            const timerStartKey = `timerStart_${query}`;
            const timerDurationKey = `timerDuration_${query}`;
            
            localStorage.setItem(timerStartKey, new Date().getTime().toString());
            localStorage.setItem(timerDurationKey, currentSeconds.toString());
        }

        function updateReloadTimer(query) {
            if (!query) return;
            
            // Use query-specific keys for localStorage
            const timerStartKey = `timerStart_${query}`;
            const timerDurationKey = `timerDuration_${query}`;
            
            // Decrement only if we still have time left
            if (currentSeconds > 0) {
                currentSeconds--;
                // Update stored timer value for this specific query
                localStorage.setItem(timerDurationKey, currentSeconds.toString());
            }
            
            const minutes = Math.floor(currentSeconds / 60);
            const seconds = currentSeconds % 60;
            const timerString = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            const timerElements = document.getElementsByClassName('reload-timer');
            for (let element of timerElements) {
                element.textContent = timerString;
            }
            
            if (currentSeconds <= 0) {
                clearInterval(reloadTimer);
                // Clear the timer storage before reloading
                localStorage.removeItem(timerStartKey);
                localStorage.removeItem(timerDurationKey);
                location.reload();
            }
        }
    </script>
</body>
</html>''')

    with open('templates/error.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Error - News Timeline</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
            text-align: center;
        }
        h1 {
            color: #e74c3c;
        }
        .error-container {
            background-color: #fff;
            padding: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-top: 50px;
        }
        .error-message {
            margin-bottom: 30px;
            font-size: 18px;
        }
        .button {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
        }
        .button:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <div class="error-container">
        <h1>Error</h1>
        <div class="error-message">
            {{ error }}
        </div>
        <a href="/" class="button">Go Back</a>
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