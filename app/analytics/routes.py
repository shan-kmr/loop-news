"""
Analytics routes for tracking events that need their own endpoints.
"""

from flask import Blueprint, request, redirect, current_app, jsonify, send_from_directory, url_for
from flask_login import current_user, login_required
from .events import track_article_interaction
import os

# Create Blueprint
analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/analytics.js')
def analytics_js():
    """Serve the analytics.js file."""
    return send_from_directory(os.path.join(current_app.root_path, '..', 'templates', 'analytics'), 'analytics.js')

@analytics_bp.route('/track/article_click', methods=['GET'])
@login_required
def track_article_click():
    """
    Track an article click and redirect to the article URL.
    
    Query parameters:
    - url: The URL of the article to redirect to
    - brief: The brief query that led to this article
    """
    article_url = request.args.get('url', '')
    brief_query = request.args.get('brief', '')
    
    if article_url and brief_query and current_user.is_authenticated:
        # Track the article click
        track_article_interaction(
            user_id=current_user.id,
            brief_query=brief_query,
            article_url=article_url,
            action='click'
        )
    
    # Redirect to the article
    if article_url:
        return redirect(article_url)
    else:
        return redirect('/')

@analytics_bp.route('/api/track/article_view', methods=['POST'])
@login_required
def track_article_view():
    """
    API endpoint to track article views and time spent.
    
    This endpoint is called via AJAX when a user views an article
    in an embedded view or when they return from an external article.
    
    Request body:
    - url: The URL of the article
    - brief: The brief query
    - time_spent: Time spent viewing the article in seconds
    """
    data = request.get_json()
    
    article_url = data.get('url', '')
    brief_query = data.get('brief', '')
    time_spent = data.get('time_spent')
    
    if article_url and brief_query and current_user.is_authenticated:
        # Track the article view
        track_article_interaction(
            user_id=current_user.id,
            brief_query=brief_query,
            article_url=article_url,
            action='view',
            time_spent=time_spent
        )
        
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400 