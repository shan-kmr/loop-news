import os
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, flash, current_app
from flask_login import login_required, current_user
from datetime import datetime

# Import services and utils
from ..services.brave import get_brave_api
from ..services.reddit import get_reddit_api
from ..services.notifications import save_notification_settings, send_notification_email
from ..utils.history import (
    load_search_history, save_search_history, update_current_day_results, 
    collect_day_summaries, repair_history_file, get_history_file
)
from ..utils.grouping import group_articles_by_topic 
from ..utils.text import extract_age_in_seconds, is_valid_summary

# Create Blueprint
news_bp = Blueprint('news', __name__)

# --- Helper Functions (specific to this blueprint) ---

def _get_api_clients():
    """Initialize and return Brave and Reddit API clients."""
    # These functions now get the shared instance managed by the service module
    brave_api = get_brave_api()
    reddit_api = get_reddit_api()
    
    # Log availability
    if brave_api is None:
         print("Brave API client is unavailable in news routes.")
    if reddit_api is None:
         print("Reddit API client is unavailable in news routes.")
         
    return brave_api, reddit_api

# --- Routes --- 

@news_bp.route('/', methods=['GET', 'POST']) # Handles home page and new brief submissions
def index():
    """Main route: Displays briefs, handles new brief submission."""
    brave_api, reddit_api = _get_api_clients()

    if not current_user.is_authenticated:
        return redirect(url_for('news.raison_detre'))

    # Check if Brave API is configured (essential)
    if brave_api is None:
        # Check config directly for a more informative error before giving up
        if not current_app.config.get("BRAVE_API_KEY"):
             error_msg = "BRAVE_API_KEY not set. Please configure it and restart."
             print(error_msg)
             # Render the dedicated error page
             return render_template('error.html',
                                    error=error_msg,
                                    user=current_user)
        else:
            # Key exists but client failed to initialize - less common
             flash("Failed to initialize Brave News API client.", 'error')

    # Load user's history
    history = load_search_history(user=current_user)
    
    query = request.args.get('query', '') # Default empty if no query param
    results = None
    search_time = None
    error = None
    topic_groups = []
    day_summaries_for_template = {} # Summaries for the specific query being viewed/searched
    
    force_refresh = request.args.get('force_refresh') == 'true'

    # --- Handle POST request (New Brief Submission) --- 
    if request.method == 'POST' and current_user.is_authenticated:
        query = request.form.get('query', '').strip().lower() # Standardize query
        count = int(request.form.get('count', 50)) # Default 50
        freshness = request.form.get('freshness', 'pm') # Default past month
        similarity_threshold = float(request.form.get('similarity_threshold', 0.4)) # Default high
        # force_refresh handled by GET param or logic below
        
        # Check topic limit
        unique_queries = {entry.get('query') for entry in history.values() if entry.get('query')}
        if len(unique_queries) >= 3 and query not in unique_queries:
            error = "You've reached the maximum of 3 briefs. Please delete one before adding another." 
            flash(error, 'error')
            # Fall through to render GET part of the page with error
        
        elif query:
            print(f"New brief submission for query: '{query}'")
            cache_key = f"{query}_{count}_{freshness}" # Simple cache key
            cached_entry = history.get(cache_key)
            use_cache = False
            needs_day_refresh = False
            cache_validity_hours = current_app.config.get('CACHE_VALIDITY_HOURS', 1)

            # --- Cache Logic --- 
            if cached_entry and not force_refresh:
                try:
                    cached_time_str = cached_entry['timestamp']
                    cached_time = datetime.fromisoformat(cached_time_str)
                    # Ensure naive datetime for comparison if needed, or make aware
                    if cached_time.tzinfo is not None:
                         cached_time = cached_time.replace(tzinfo=None)
                    
                    now = datetime.now()
                    time_diff = now - cached_time
                    
                    if time_diff.total_seconds() < cache_validity_hours * 3600:
                         use_cache = True
                         if time_diff.total_seconds() > 3600: # Refresh today's content if > 60 mins old
                             needs_day_refresh = True
                         
                         results = cached_entry.get('results')
                         search_time = cached_time
                         day_summaries_for_template = cached_entry.get('day_summaries', {})
                         topic_groups = cached_entry.get('topic_groups', [])
                         print(f"Using cached results for '{query}' ({(time_diff.total_seconds()/60):.1f} mins old).")

                         if needs_day_refresh:
                             print(f"Refreshing Today's content for cached '{query}'.")
                             updated_results, days_with_new = update_current_day_results(query, count, freshness, results)
                             if updated_results != results:
                                 print(f"Updated Today's content found for '{query}'.")
                                 results = updated_results
                                 search_time = datetime.now() # Update timestamp
                                 # Clear affected summaries
                                 if days_with_new and isinstance(day_summaries_for_template, dict):
                                     for day in days_with_new:
                                         day_summaries_for_template[day] = None
                                 if 'Today' in day_summaries_for_template:
                                      day_summaries_for_template['Today'] = None
                                      
                                 # Update cache entry
                                 cached_entry['results'] = results
                                 cached_entry['timestamp'] = search_time.isoformat()
                                 cached_entry['day_summaries'] = day_summaries_for_template
                                 # Re-grouping might be needed here, or handled on display
                                 history[cache_key] = cached_entry
                                 save_search_history(history, user=current_user)
                                 # Re-run grouping after update
                                 if results and 'results' in results:
                                    sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
                                    # Pass the potentially cleared summaries dict
                                    topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, query, day_summaries_for_template)
                                    # Update cache with new groups and potentially regenerated summaries
                                    cached_entry['topic_groups'] = topic_groups
                                    cached_entry['day_summaries'] = day_summaries_for_template # Contains updated summaries
                                    save_search_history(history, user=current_user) 
                             else:
                                  print(f"No new content found during Today's refresh for '{query}'.")
                except Exception as e:
                    print(f"Error processing cache for '{query}': {e}")
                    use_cache = False # Force fetch if cache processing fails

            # --- Fetch New Data (if no cache or cache invalid/refresh needed) ---
            if not use_cache:
                print(f"Fetching new results for '{query}'. Force refresh: {force_refresh}")
                try:
                    # Fetch Brave results
                    brave_results_data = None
                    if brave_api:
                        brave_results_data = brave_api.search_news(query=query, count=count, freshness=freshness)
                    else:
                         print("Brave API unavailable, skipping Brave search.")
                         brave_results_data = {'results': []} # Empty structure
                         
                    search_time = datetime.now() # Record time after API calls
                    
                    # Fetch Reddit results
                    reddit_articles = []
                    if reddit_api:
                        try:
                            time_filter_map = {'pd': 'day', 'pw': 'week', 'pm': 'month', 'py': 'year'}
                            reddit_time_filter = time_filter_map.get(freshness, 'week')
                            reddit_results_data = reddit_api.search(query=query, count=count, time_filter=reddit_time_filter)
                            if isinstance(reddit_results_data, dict) and 'results' in reddit_results_data:
                                 reddit_articles = reddit_results_data['results']
                                 print(f"Got {len(reddit_articles)} Reddit results for '{query}'.")
                        except Exception as e:
                            print(f"Error fetching Reddit results for '{query}': {e}")
                            
                    # Combine results (start with Brave structure, add Reddit)
                    if not isinstance(brave_results_data, dict) or 'results' not in brave_results_data:
                         brave_results_data = {'results': []} # Ensure structure
                         
                    combined_articles = brave_results_data['results'] + reddit_articles
                    
                    # Remove duplicates based on URL (optional, update_current_day handles this better)
                    # unique_articles = {a['url']: a for a in combined_articles if a.get('url')}
                    # combined_articles = list(unique_articles.values()) 

                    # Final result structure
                    results = brave_results_data # Use Brave structure as base
                    results['results'] = combined_articles # Replace with combined list
                    
                    # Sort and Group
                    if results and 'results' in results:
                        sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
                        results['results'] = sorted_articles # Store sorted articles back
                        
                        day_summaries_for_template = {} # Reset for new fetch
                        topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, query, day_summaries_for_template)
                        
                        # Update cache with new results, groups, and summaries
                        history[cache_key] = {
                            'query': query,
                            'count': count,
                            'freshness': freshness,
                            'timestamp': search_time.isoformat(),
                            'results': results,
                            'day_summaries': day_summaries_for_template, # Now contains generated summaries
                            'topic_groups': topic_groups,
                            'search_time': search_time.timestamp() # Keep original timestamp for sorting history list
                        }
                        save_search_history(history, user=current_user)
                        print(f"Cached new results for '{query}'.")
                        # Redirect to the history view for this new brief
                        return redirect(url_for('news.history_item', query=query))
                        
                except Exception as e:
                    error_msg = f"Error fetching or processing news for '{query}': {str(e)}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    error = error_msg # Display error on the page
                    flash(error, 'error')
                    # Fall through to render GET part with error
            else:
                 # If using cache (and potentially refreshed Today), redirect to history item view
                 return redirect(url_for('news.history_item', query=query))

    # --- Handle GET request (Display Home Page) --- 
    
    # Auto-refresh check for stale entries on home page load
    if request.method == 'GET' and not request.args.get('query') and not force_refresh:
        refreshed_queries = []
        needs_saving = False
        cache_validity_minutes = current_app.config.get('CACHE_VALIDITY_HOURS', 1) * 60
        
        print(f"Checking {len(history)} history entries for staleness (threshold: {cache_validity_minutes} mins)...")
        
        for key, entry in history.items():
            try:
                cached_time_str = entry.get('timestamp')
                if not cached_time_str: continue
                
                cached_time = datetime.fromisoformat(cached_time_str)
                if cached_time.tzinfo is not None:
                    cached_time = cached_time.replace(tzinfo=None)
                
                time_diff_minutes = (datetime.now() - cached_time).total_seconds() / 60
                
                # Refresh if older than 60 minutes (adjust as needed)
                if time_diff_minutes > 60:
                    q = entry.get('query')
                    c = entry.get('count', 50)
                    f = entry.get('freshness', 'py')
                    r = entry.get('results')
                    
                    if q and r:
                        print(f"Home page: Auto-refreshing stale entry '{q}' ({time_diff_minutes:.1f} mins old)")
                        updated_r, days_new = update_current_day_results(q, c, f, r)
                        
                        if updated_r != r:
                            entry['results'] = updated_r
                            entry['timestamp'] = datetime.now().isoformat()
                            # Clear summaries that need regen
                            if days_new and isinstance(entry.get('day_summaries'), dict):
                                for day in days_new:
                                    entry['day_summaries'][day] = None
                            # Always clear Today's summary on refresh
                            if isinstance(entry.get('day_summaries'), dict):
                                 entry['day_summaries']['Today'] = None
                                 
                            refreshed_queries.append(q)
                            needs_saving = True
                            print(f"Auto-refreshed '{q}' with new content.")
                        else:
                             # Even if no new content, update timestamp to prevent constant checks
                             entry['timestamp'] = datetime.now().isoformat()
                             needs_saving = True
                             print(f"Auto-refreshed '{q}' - no new content, timestamp updated.")
            except Exception as e:
                print(f"Error checking timestamp for auto-refresh on key {key}: {e}")
        
        if needs_saving:
            save_search_history(history, user=current_user)
            if refreshed_queries:
                 print(f"Auto-refreshed {len(refreshed_queries)} entries: {', '.join(refreshed_queries)}")
            else:
                 print("Auto-refresh check complete, only timestamps updated.")
                 
    # Prepare data for template
    history_entries_for_template = []
    for key, entry in history.items():
        # Ensure basic structure for template rendering
        query_val = entry.get('query')
        timestamp_val = entry.get('timestamp')
        if query_val and timestamp_val:
             history_entries_for_template.append({
                'query': query_val,
                'timestamp': timestamp_val,
                'key': key
            })
    
    # Sort history by most recent search/update time
    history_entries_for_template.sort(key=lambda x: x.get('timestamp', '0'), reverse=True)
    
    # Collect summaries to display on home page cards (Today/Yesterday)
    home_page_summaries = collect_day_summaries(history)

    return render_template('index.html', 
                          query=query, # Current query in context (might be empty)
                          results=results, # Results if a search was just done (usually None on GET)
                          search_time=search_time,
                          history_entries=history_entries_for_template, # List for sidebar/cards
                          error=error,
                          topic_groups=topic_groups, # Groups if search done (usually empty on GET)
                          day_summaries=home_page_summaries, # All collected summaries for card display
                          history=history, # Pass the raw history dict if needed by template logic
                          active_tab="search",
                          user=current_user)

@news_bp.route('/history', methods=['GET']) # Redirects to the latest history item
@login_required
def history_list():
    """View to display search history, redirects to the latest item."""
    history = load_search_history(user=current_user)
    
    history_entries = []
    for key, entry in history.items():
        query_val = entry.get('query')
        timestamp_val = entry.get('timestamp')
        if query_val and timestamp_val:
             history_entries.append({
                'query': query_val,
                'timestamp': timestamp_val,
                'key': key
            })
    
    history_entries.sort(key=lambda x: x.get('timestamp', '0'), reverse=True)
    
    if history_entries:
        latest_query = history_entries[0]['query']
        print(f"Redirecting to latest history item: {latest_query}")
        return redirect(url_for('news.history_item', query=latest_query))
    else:
        # No history, show empty state on index/search page (or a dedicated history page)
        print("No history found, rendering empty index page.")
        flash("You haven't added any briefs yet.", 'info')
        # Render index template in its empty state
        return render_template('index.html',
                              query='',
                              results=None,
                              search_time=None,
                              error=None,
                              history_entries=[],
                              topic_groups=[],
                              day_summaries={},
                              history={},
                              active_tab="history", # Indicate history tab
                              user=current_user)

@news_bp.route('/history/<path:query>', methods=['GET']) # Displays a specific history item
@login_required
def history_item(query):
    """View to display a specific history item timeline."""
    # Optional: Repair history file on access
    repair_history_file(user=current_user) 
    
    history = load_search_history(user=current_user)
    
    # Find the requested history item by query
    target_entry = None
    target_key = None
    query = query.lower() # Standardize query for matching
    
    for key, entry in history.items():
        if entry.get('query', '').lower() == query:
            target_entry = entry
            target_key = key
            break
            
    if not target_entry:
        # If brief not found, check if we can even fetch it (API key needed)
        brave_api, _ = _get_api_clients() # Check API status
        if brave_api is None and not current_app.config.get("BRAVE_API_KEY"):
            error_msg = f"Brief '{query}' not found and BRAVE_API_KEY is not set. Cannot fetch new results."
            print(error_msg)
            return render_template('error.html', error=error_msg, user=current_user)
        else:
            # If API key *is* present, maybe it can be fetched as new? 
            # For now, keep original behavior: flash error and redirect
            # TODO: Optionally implement fetching as a new brief here
            flash(f"Brief '{query}' not found in your history.", 'error')
            return redirect(url_for('news.index'))

    print(f"Displaying history item: '{query}'")
    
    # Extract data from the found entry
    results = target_entry.get('results')
    search_time = datetime.fromisoformat(target_entry['timestamp']) if target_entry.get('timestamp') else datetime.now()
    day_summaries = target_entry.get('day_summaries', {})
    topic_groups = target_entry.get('topic_groups', []) # Use cached groups if available
    count = target_entry.get('count', 50)
    freshness = target_entry.get('freshness', 'py')
    similarity_threshold = target_entry.get('similarity_threshold', 0.4) # Get threshold used
    
    # Check for forced refresh or staleness (> 60 mins)
    force_refresh = request.args.get('force_refresh') == 'true'
    time_diff_minutes = (datetime.now() - search_time.replace(tzinfo=None)).total_seconds() / 60
    needs_refresh = force_refresh or (time_diff_minutes > 60)
    
    if needs_refresh:
        print(f"Refreshing history item '{query}' (Force: {force_refresh}, Age: {time_diff_minutes:.1f} mins)")
        updated_results, days_with_new = update_current_day_results(query, count, freshness, results)
        
        if updated_results != results:
            print(f"Content updated for '{query}' during refresh.")
            results = updated_results
            search_time = datetime.now() # Update timestamp
            # Clear summaries for affected days
            if days_with_new and isinstance(day_summaries, dict):
                for day in days_with_new:
                    day_summaries[day] = None
            if isinstance(day_summaries, dict):
                 day_summaries['Today'] = None # Always clear today
                 
            # Update history entry in memory
            target_entry['results'] = results
            target_entry['timestamp'] = search_time.isoformat()
            target_entry['day_summaries'] = day_summaries
            
            # CRITICAL: Re-run grouping after refresh
            if results and 'results' in results:
                sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
                # Pass the potentially cleared summaries dict
                topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, query, day_summaries)
                target_entry['topic_groups'] = topic_groups
                # The grouping function handles summary generation and saving within its logic
                # Ensure day_summaries reflects changes made by grouping function
                day_summaries = target_entry['day_summaries'] 
            
            # Save the entire updated entry back to history
            history[target_key] = target_entry 
            save_search_history(history, user=current_user)
            print(f"Saved refreshed data and regenerated groups/summaries for '{query}'.")
            
        else:
            print(f"No new content found during refresh for '{query}'. Updating timestamp only.")
            # Update timestamp even if no new content
            target_entry['timestamp'] = datetime.now().isoformat()
            history[target_key] = target_entry
            save_search_history(history, user=current_user)
    else:
        # If not refreshing, ensure groups exist (generate if missing from cache)
        if not topic_groups and results and 'results' in results:
             print(f"Topic groups missing for cached item '{query}'. Generating now.")
             sorted_articles = sorted(results['results'], key=extract_age_in_seconds)
             topic_groups = group_articles_by_topic(sorted_articles, similarity_threshold, query, day_summaries)
             target_entry['topic_groups'] = topic_groups
             # Save updated entry with groups
             history[target_key] = target_entry
             save_search_history(history, user=current_user)

    # Prepare history list for sidebar
    history_entries_for_template = []
    for key, entry in history.items():
        q_val = entry.get('query')
        t_val = entry.get('timestamp')
        if q_val and t_val:
            history_entries_for_template.append({
                'query': q_val,
                'timestamp': t_val,
                'key': key
            })
    history_entries_for_template.sort(key=lambda x: x.get('timestamp', '0'), reverse=True)
    
    # Sort final topic groups by day priority and then article age for display
    day_priority = {"Today": 1, "Yesterday": 2, "2 days ago": 3, "3 days ago": 4, "4 days ago": 5, "5 days ago": 6, "6 days ago": 7, "1 week ago": 8, "2 weeks ago": 9, "1 month ago": 10}
    topic_groups = sorted(topic_groups, key=lambda x: (
        day_priority.get(x.get('day_group', 'Unknown'), 999),
        extract_age_in_seconds(x['articles'][0]) if x.get('articles') else float('inf')
    ))

    return render_template('index.html', 
                          query=query, 
                          results=results, 
                          search_time=search_time,
                          error=None,
                          history_entries=history_entries_for_template,
                          topic_groups=topic_groups,
                          day_summaries=day_summaries, # Pass summaries associated with this query
                          history=history, # Raw history might be needed
                          active_tab="history",
                          user=current_user)

@news_bp.route('/api/delete_history/<path:query>', methods=['POST'])
@login_required
def delete_history_item_api(query):
    """API endpoint to delete a history item."""
    history = load_search_history(user=current_user)
    query = query.lower()
    items_to_remove = [key for key, entry in history.items() if entry.get('query', '').lower() == query]
    
    if not items_to_remove:
         return jsonify({'success': False, 'error': 'Query not found'}), 404
         
    for key in items_to_remove:
        history.pop(key, None)
        print(f"Deleted history item with key: {key} (Query: {query})")
    
    save_search_history(history, user=current_user)
    flash(f'Brief "{query}" deleted.', 'success') # Provide user feedback
    return jsonify({'success': True})

@news_bp.route('/api/clear_history', methods=['POST'])
@login_required
def clear_history_api():
    """API endpoint to clear all history for the user."""
    save_search_history({}, user=current_user) # Save an empty dict
    print(f"Cleared history for user {current_user.email}")
    flash('All briefs cleared.', 'success')
    return jsonify({'success': True})

@news_bp.route('/raison-detre', methods=['GET']) 
def raison_detre():
    """Display the raison d'être page."""
    history_entries = []
    history = {}
    # Load history only if user is logged in to show sidebar items
    if current_user.is_authenticated:
        history = load_search_history(user=current_user)
        for key, entry in history.items():
            q_val = entry.get('query')
            t_val = entry.get('timestamp')
            if q_val and t_val:
                 history_entries.append({
                    'query': q_val,
                    'timestamp': t_val,
                    'key': key
                 })
        history_entries.sort(key=lambda x: x.get('timestamp', '0'), reverse=True)
    
    # Define philosophy content here or load from a file/DB
    philosophy_content = {
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
            }
        ]
    }

    return render_template('index.html', # Reusing index.html structure
        active_tab='raison-detre',
        user=current_user,
        history_entries=history_entries,
        history=history,
        query=None,
        error=None,
        topic_groups=[],
        day_summaries={},
        philosophy_content=philosophy_content # Pass the content to the template
    )

# --- API Routes for Notifications --- 

@news_bp.route('/api/notifications/save', methods=['POST'])
@login_required
def save_notification_api():
    """API endpoint to save notification settings."""
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    topic = data.get('topic')
    frequency = data.get('frequency')
    
    if not topic or not frequency:
        return jsonify({'success': False, 'error': 'Missing topic or frequency'}), 400
    if frequency not in ['hourly', 'daily']:
        return jsonify({'success': False, 'error': 'Invalid frequency'}), 400
        
    user_email = current_user.email # Assumes user object has email
    if not user_email:
         return jsonify({'success': False, 'error': 'User email not found'}), 400
         
    success = save_notification_settings(user_email, topic, frequency)
    
    if success:
        return jsonify({'success': True, 'message': f'Settings saved for {topic}'}) 
    else:
        # Provide more context if save_notification_settings returns detailed errors
        return jsonify({'success': False, 'error': f'Failed to save settings for {topic}'}), 500

@news_bp.route('/api/test-notification/<path:topic>/<frequency>', methods=['GET'])
@login_required
def test_notification_api(topic, frequency):
    """Test endpoint to manually trigger a notification email for the current user."""
    if frequency not in ['hourly', 'daily']:
        return jsonify({'success': False, 'error': 'Invalid frequency'}), 400
        
    history = load_search_history(user=current_user)
    entry = None
    for key, history_entry in history.items():
        if history_entry.get('query', '').lower() == topic.lower():
            entry = history_entry
            break
    
    if not entry:
        return jsonify({'success': False, 'error': f"Topic '{topic}' not found"}), 404
    
    # Use current user's email for testing
    test_email = current_user.email
    if not test_email:
         return jsonify({'success': False, 'error': 'Current user email not found'}), 400
         
    print(f"Sending TEST notification for '{topic}' ({frequency}) to {test_email}")
    
    # Simulate finding some new articles for testing hourly
    new_articles_test = []
    if frequency == 'hourly' and entry.get('results') and entry['results'].get('results'):
        new_articles_test = entry['results']['results'][:2] # Use first 2 articles as "new"
        
    # Use the actual send function (requires app context)
    email_sent = False
    error_msg = ""
    try:
         # send_notification_email requires app context for config/templates
         with current_app.app_context():
             email_sent = send_notification_email(topic, entry, frequency, test_email, new_articles_test)
    except Exception as e:
         error_msg = f"Error during test send: {str(e)}"
         print(error_msg)
         traceback.print_exc()
         email_sent = False
         
    if email_sent:
        return jsonify({'success': True, 'message': f'Test email sent to {test_email}'}) 
    else:
        return jsonify({'success': False, 'error': f'Failed to send test email. {error_msg}'}), 500

@news_bp.route('/api/debug-notifications', methods=['GET'])
@login_required # Make this admin-only in a real app
def debug_notifications_api():
    """Debug endpoint to view all notification settings (requires careful access control)."""
    # WARNING: This exposes potentially sensitive data (emails, topics). Use with extreme caution.
    # In production, restrict this to admin users.
    # For now, restrict to a specific admin email defined in config? 
    # admin_email = current_app.config.get('ADMIN_EMAIL')
    # if not admin_email or current_user.email != admin_email:
    #     return jsonify({'error': 'Forbidden'}), 403
    
    results = []
    user_data_dir = 'user_data'
    all_history_files = []

    # Shared file
    shared_file = os.path.join(user_data_dir, 'shared', 'search_history.json')
    if os.path.exists(shared_file): all_history_files.append(shared_file)

    # User files
    try:
        for item in os.listdir(user_data_dir):
            item_path = os.path.join(user_data_dir, item)
            if os.path.isdir(item_path) and item not in ['shared', 'anonymous']:
                history_file = os.path.join(item_path, 'search_history.json')
                if os.path.exists(history_file): all_history_files.append(history_file)
    except Exception as e:
        print(f"Error scanning for debug notifications: {e}")

    for file_path in all_history_files:
        file_result = {'file': file_path, 'topics_with_notifications': []}
        try:
            with open(file_path, 'r') as f: history = json.load(f)
            for key, entry in history.items():
                if isinstance(entry.get('notifications'), dict) and entry['notifications'].get('recipients'):
                    topic_info = {
                        'topic': entry.get('query', 'Unknown'),
                        'recipients': entry['notifications']['recipients'],
                        'last_sent': entry['notifications'].get('last_sent', {})
                    }
                    file_result['topics_with_notifications'].append(topic_info)
            results.append(file_result)
        except Exception as e:
            results.append({'file': file_path, 'error': str(e)})

    return jsonify({'success': True, 'notification_settings_by_file': results})