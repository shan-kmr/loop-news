"""
Middleware functions for analytics.
"""

from functools import wraps
from flask import request, g
from flask_login import current_user
from .events import track_search_behavior, track_brief_interaction, track_article_interaction

def track_page_view(f):
    """
    Decorator to track page views.
    
    Usage:
        @app.route('/some/path')
        @track_page_view
        def some_view():
            ...
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Only track if user is authenticated
        if current_user.is_authenticated:
            # Implement page view tracking here if needed
            # This would be stored in a different table
            pass
        return f(*args, **kwargs)
    return decorated_function

def setup_request_tracking(app):
    """
    Set up request tracking for the Flask app.
    
    This function sets up before_request and after_request handlers
    to track request information.
    """
    @app.before_request
    def before_request():
        """Handle tracking before each request."""
        if not current_user.is_authenticated:
            return
            
        # Store request start time for duration calculation
        g.request_start_time = g.get('request_start_time', None)
        
        # Track search queries
        if request.endpoint == 'news.index' and request.method == 'POST':
            query = request.form.get('query', '').strip().lower()
            if query:
                filters = {
                    'count': request.form.get('count'),
                    'freshness': request.form.get('freshness'),
                    'similarity_threshold': request.form.get('similarity_threshold')
                }
                track_search_behavior(current_user.id, query, filters)
        
        # Track brief interactions
        if request.endpoint == 'news.index' and request.method == 'POST':
            query = request.form.get('query', '').strip().lower()
            if query:
                parameters = {
                    'count': request.form.get('count'),
                    'freshness': request.form.get('freshness'),
                    'similarity_threshold': request.form.get('similarity_threshold')
                }
                track_brief_interaction(current_user.id, query, 'create', parameters)
        elif request.endpoint == 'news.history_item':
            query = request.view_args.get('query', '')
            if query:
                track_brief_interaction(current_user.id, query, 'view')
        elif request.endpoint == 'news.delete_history_item_api':
            query = request.view_args.get('query', '')
            if query:
                track_brief_interaction(current_user.id, query, 'delete')
                
    @app.after_request
    def after_request(response):
        """Handle tracking after each request."""
        # Track additional information if needed
        return response 