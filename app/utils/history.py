import os
import json
from datetime import datetime
from flask import current_app
from flask_login import current_user

# Assuming API wrappers are in the services layer
from ..services.brave import get_brave_api # Need to define this in brave.py
from ..services.reddit import get_reddit_api # Need to define this in reddit.py

# Assuming utils are in the same directory or accessible
from .filters import day_group_filter
from .text import extract_age_in_seconds, is_valid_summary
from ..services.openai_service import summarize_daily_news # For repair function

def get_history_file(user=None):
    """
    Get the search history file path for a user.
    Args:
        user: Optional User object. 
        If not provided and current_user is authenticated, will use current_user. 
        For background tasks where current_user is not available, this allows passing 
        None to get the default location.
    """
    try:
        target_user = user
        # If user not provided, try to use current_user if available in context
        if target_user is None and 'current_user' in globals() and current_user and hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
            target_user = current_user
        
        if target_user and hasattr(target_user, 'is_authenticated') and target_user.is_authenticated:
            history_file = target_user.get_history_file() # Method on User model
            if history_file:  # Check that we got a valid path
                # Ensure directory exists
                dir_name = os.path.dirname(history_file)
                if dir_name: 
                    os.makedirs(dir_name, exist_ok=True)
                return history_file
    except Exception as e:
        print(f"Error getting user history file: {e}")
    
    # Default location for anonymous users, background threads, or if user path is invalid
    default_dir = os.path.join('user_data', 'shared')
    os.makedirs(default_dir, exist_ok=True)
    return os.path.join(default_dir, 'search_history.json')

def load_search_history(user=None):
    """Load search history from JSON file for a user."""
    history_file = get_history_file(user=user)
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                # Handle empty file case
                content = f.read()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from history file {history_file}: {e}. Returning empty history.")
            # Optionally: Backup the corrupted file
            # backup_file = f"{history_file}.corrupted.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            # try:
            #     os.rename(history_file, backup_file)
            #     print(f"Backed up corrupted file to {backup_file}")
            # except OSError as rename_error:
            #     print(f"Could not back up corrupted file: {rename_error}")
            return {}
        except Exception as e:
            print(f"Error loading history from {history_file}: {e}")
    return {}

def save_search_history(history, user=None):
    """Save search history to JSON file for a user."""
    history_file = get_history_file(user=user)
    try:
        # Ensure the directory exists
        dir_name = os.path.dirname(history_file)
        if dir_name: 
            os.makedirs(dir_name, exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2) # Add indent for readability
    except Exception as e:
        print(f"Error saving history to {history_file}: {e}")

def update_current_day_results(query, count, freshness, old_results):
    """
    Refresh all articles from the last search date to the current date,
    filling in any gap periods in between.
    
    Args:
        query: The search query
        count: Number of results to fetch
        freshness: Freshness parameter for the API
        old_results: Previous search results dictionary
        
    Returns:
        Tuple: (updated_results_dict, list_of_days_with_new_content)
    """
    brave_api = get_brave_api() # Get instance from service
    reddit_api = get_reddit_api() # Get instance from service
    
    if brave_api is None:
        print("Brave API service not available.")
        # Decide how to handle this - return old results or raise error?
        return old_results, [] 
        
    try:
        # Determine refresh freshness based on original freshness and cache validity
        cache_validity_hours = current_app.config.get('CACHE_VALIDITY_HOURS', 1)
        refresh_freshness = 'pd' # Default to past day for frequent refreshes
        if freshness in ['pw', 'pm', 'py'] and cache_validity_hours > 24:
            refresh_freshness = freshness
            
        print(f"Using freshness '{refresh_freshness}' for refreshing '{query}' (original: {freshness}) - Cache validity: {cache_validity_hours}h")
        
        # --- Fetch Fresh Data --- 
        fresh_results = None
        reddit_results_data = None
        
        try:
            fresh_results = brave_api.search_news(
                query=query,
                count=max(count * 2, 50),  # Get more results to cover gaps
                freshness=refresh_freshness
            )
        except Exception as e:
            print(f"Error fetching from Brave API: {e}")
            # Decide if we should proceed without Brave results

        if reddit_api is not None:
            try:
                time_filter_map = {'pd': 'day', 'pw': 'week', 'pm': 'month', 'py': 'year'}
                reddit_time_filter = time_filter_map.get(refresh_freshness, 'week')
                
                reddit_results_data = reddit_api.search(
                    query=query,
                    count=count,
                    time_filter=reddit_time_filter
                )
                print(f"Got {len(reddit_results_data.get('results', []))} Reddit results for '{query}'")
            except Exception as e:
                print(f"Error getting Reddit results: {str(e)}")
        
        # --- Process and Merge Results --- 
        
        # Ensure old_results is a dictionary with a 'results' list
        if not isinstance(old_results, dict):
            old_results = {'results': []}
        if 'results' not in old_results or not isinstance(old_results['results'], list):
             old_results['results'] = []
             
        # Ensure fresh_results is a dictionary with a 'results' list
        if not isinstance(fresh_results, dict):
             fresh_results = {'results': []}
        if 'results' not in fresh_results or not isinstance(fresh_results['results'], list):
            fresh_results['results'] = []

        # Ensure reddit_results_data is structured correctly
        reddit_articles = []
        if isinstance(reddit_results_data, dict) and 'results' in reddit_results_data and isinstance(reddit_results_data['results'], list):
            reddit_articles = reddit_results_data['results']

        # Combine new Brave results and new Reddit results
        new_combined_articles = fresh_results['results'] + reddit_articles
        
        # If we have no new articles at all, return old results
        if not new_combined_articles:
            print(f"No new results found (Brave or Reddit) for '{query}'. Returning old results.")
            return old_results, []
            
        old_articles = old_results['results']
        
        # Use URLs to identify unique articles
        old_urls = {article.get('url') for article in old_articles if article.get('url')}
        new_article_urls = {article.get('url') for article in new_combined_articles if article.get('url')}
        
        truly_new_urls = new_article_urls - old_urls
        
        days_with_new_articles = []
        
        if truly_new_urls:
            print(f"Found {len(truly_new_urls)} truly new articles for '{query}'")
            # Identify days with new articles
            for article in new_combined_articles:
                if article.get('url') in truly_new_urls:
                    day = day_group_filter(article)
                    if day not in days_with_new_articles:
                        days_with_new_articles.append(day)
            
            # Merge: Start with new articles, add old ones not present in the new set
            merged_articles_dict = {article.get('url'): article for article in new_combined_articles if article.get('url')}
            
            for article in old_articles:
                url = article.get('url')
                if url and url not in merged_articles_dict:
                    merged_articles_dict[url] = article
            
            merged_articles = list(merged_articles_dict.values())
            
            # Sort by most recent first (using age in seconds)
            merged_articles.sort(key=extract_age_in_seconds)
            
            # Create the final results structure (using Brave structure as base if available)
            merged_results_structure = fresh_results.copy() # Use structure of fresh Brave results
            merged_results_structure['results'] = merged_articles
            
            print(f"Merged into {len(merged_articles)} unique articles for '{query}'")
            print(f"Days with new content: {days_with_new_articles}")
            
            return merged_results_structure, days_with_new_articles
        else:
            print(f"No new content found for '{query}'. Returning old results.")
            return old_results, []
            
    except Exception as e:
        print(f"Error refreshing results for '{query}': {str(e)}")
        import traceback
        traceback.print_exc()
        return old_results, [] # Return old results on error

def collect_day_summaries(history):
    """Collect latest day summaries from history entries for display."""
    latest_summaries = {}
    
    # Priority order of days
    day_priority = {
        "Today": 1, "Yesterday": 2, "2 days ago": 3, "3 days ago": 4,
        "4 days ago": 5, "5 days ago": 6, "6 days ago": 7, "1 week ago": 8,
        "2 weeks ago": 9, "1 month ago": 10
    }
    
    for key, entry in history.items():
        query = entry.get('query', '')
        if not query:
            continue
            
        if 'day_summaries' in entry and isinstance(entry['day_summaries'], dict):
            best_day = None
            best_priority = float('inf')
            best_summary = None
            
            # Find the most recent day with a valid summary
            for day, summary in entry['day_summaries'].items():
                if is_valid_summary(summary):
                    priority = day_priority.get(day, 999) 
                    if priority < best_priority:
                        best_priority = priority
                        best_day = day
                        best_summary = summary
            
            if best_day and best_summary:
                # Use a unique key combining day and query to avoid collisions if 
                # multiple entries exist for the same query (though unlikely with current keying)
                summary_key = f"{best_day}_{query}"
                latest_summaries[summary_key] = {
                    'query': query,
                    'day': best_day,
                    'summary': best_summary,
                }
    
    return latest_summaries

def repair_history_file(user=None):
    """
    Repairs the history file by fixing inconsistencies.
    Focuses on ensuring summaries exist where articles do, especially for Yesterday.
    """
    history = load_search_history(user=user)
    changes_made = False
        
    for key, entry in history.items():
        query = entry.get('query')
        if not query:
             continue # Skip entries without a query

        # Ensure day_summaries exists and is a dict
        if 'day_summaries' not in entry or not isinstance(entry.get('day_summaries'), dict):
            entry['day_summaries'] = {}
            print(f"Fixed missing/invalid day_summaries dict for entry: {query}")
            changes_made = True
            
        # Ensure results structure is valid
        if 'results' in entry and isinstance(entry['results'], dict) and \
           'results' in entry['results'] and isinstance(entry['results']['results'], list):
            
            articles = entry['results']['results']
            articles_by_day = {}
            articles_fixed = 0

            # Group articles by day and check integrity
            for article in articles:
                if 'age' not in article or not isinstance(article['age'], str):
                    article['age'] = 'Unknown' # Fix missing/invalid age
                    articles_fixed += 1
                    
                day = day_group_filter(article)
                if day not in articles_by_day:
                    articles_by_day[day] = []
                articles_by_day[day].append(article)
            
            if articles_fixed > 0:
                print(f"Fixed {articles_fixed} articles with missing/invalid age data for '{query}'")
                changes_made = True # Mark changes if articles were fixed
                # Re-save the entry results with fixed articles
                entry['results']['results'] = articles

            # Check for days with articles but missing/invalid summaries
            for day, day_articles in articles_by_day.items():
                if not is_valid_summary(entry['day_summaries'].get(day)):
                    print(f"Missing/invalid summary for day '{day}' in query '{query}' ({len(day_articles)} articles). Attempting regeneration.")
                    try:
                        # Regenerate summary
                        # Check OpenAI availability before generating summaries
                        try:
                            import openai
                            OPENAI_AVAILABLE_CHECK = True
                        except ImportError:
                            OPENAI_AVAILABLE_CHECK = False
                            
                        if OPENAI_AVAILABLE_CHECK:
                            new_summary = summarize_daily_news(day_articles, query)
                            if is_valid_summary(new_summary):
                                entry['day_summaries'][day] = new_summary
                                print(f"Generated new {day} summary for '{query}': {new_summary[:50]}...")
                                changes_made = True
                            else:
                                print(f"Failed to generate valid summary for {day}, query '{query}'.")
                        else:
                             print(f"OpenAI not available, cannot generate summary for {day}, query '{query}'.")

                    except Exception as e:
                        print(f"Error generating summary for {day}, query '{query}': {str(e)}")

        else:
             # If results structure is missing or invalid, log it but don't necessarily delete summaries
             if 'results' not in entry or entry['results'] is None:
                 print(f"Warning: Entry for query '{query}' is missing 'results' data.")
             elif not isinstance(entry.get('results'), dict):
                 print(f"Warning: Entry for query '{query}' has invalid 'results' type (not a dict).")
             elif 'results' not in entry['results'] or not isinstance(entry['results'].get('results'), list):
                 print(f"Warning: Entry for query '{query}' has invalid 'results['results']' structure (not a list).")

    # Save changes if needed
    if changes_made:
        save_search_history(history, user=user)
        print(f"Saved repaired history file: {get_history_file(user=user)}")
        return True
    else:
        print(f"No issues found needing repair in history file: {get_history_file(user=user)}")
        return False 