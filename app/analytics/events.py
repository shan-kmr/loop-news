"""
Event tracking functions for analytics.
"""

import json
import uuid
import traceback
from datetime import datetime
from flask import request, session, g, current_app
from .database import get_db

def _get_session_id():
    """Get or create a session ID for tracking purposes."""
    try:
        if 'analytics_session_id' not in session:
            session_id = str(uuid.uuid4())
            session['analytics_session_id'] = session_id
            print(f"Created new analytics session ID: {session_id}")
        return session['analytics_session_id']
    except Exception as e:
        print(f"Error getting session ID: {str(e)}")
        # Fallback to a request-specific ID if session fails
        return str(uuid.uuid4())

def _get_client_info():
    """Extract client information from request."""
    try:
        return {
            'ip_address': request.remote_addr,
            'user_agent': request.user_agent.string if request.user_agent else None
        }
    except Exception as e:
        print(f"Error getting client info: {str(e)}")
        return {'ip_address': 'unknown', 'user_agent': 'unknown'}

def track_user_session(user_id, action):
    """
    Track user session events (login/logout).
    
    Args:
        user_id: The user's ID (typically email)
        action: Either 'login' or 'logout'
    """
    if not current_app.config.get('ANALYTICS_ENABLED', True):
        return

    try:
        db = get_db()
        session_id = _get_session_id()
        client_info = _get_client_info()
        now = datetime.utcnow().isoformat()
        
        print(f"Tracking user session: {action} for user {user_id} with session {session_id}")
        
        if action == 'login':
            db.execute(
                """
                INSERT INTO user_sessions 
                (user_id, session_id, login_time, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, session_id, now, client_info['ip_address'], client_info['user_agent'])
            )
        elif action == 'logout':
            # Update the most recent session for this user and session ID
            db.execute(
                """
                UPDATE user_sessions 
                SET logout_time = ?
                WHERE user_id = ? AND session_id = ? AND logout_time IS NULL
                """,
                (now, user_id, session_id)
            )
        
        db.commit()
        print(f"Successfully tracked user session: {action}")
    except Exception as e:
        print(f"Error tracking user session ({action}): {str(e)}")
        traceback.print_exc()

def track_brief_interaction(user_id, brief_query, action, parameters=None):
    """
    Track brief interaction events.
    
    Args:
        user_id: The user's ID
        brief_query: The brief query string
        action: One of 'create', 'view', 'delete', 'modify'
        parameters: Optional dict of parameters (will be stored as JSON)
    """
    if not current_app.config.get('ANALYTICS_ENABLED', True):
        return

    try:
        db = get_db()
        session_id = _get_session_id()
        parameters_json = json.dumps(parameters) if parameters else None
        
        print(f"Tracking brief interaction: {action} for query '{brief_query}' by user {user_id}")
        
        db.execute(
            """
            INSERT INTO brief_interactions
            (user_id, session_id, brief_query, action, parameters)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, session_id, brief_query, action, parameters_json)
        )
        
        db.commit()
        print(f"Successfully tracked brief interaction: {action}")
    except Exception as e:
        print(f"Error tracking brief interaction ({action}): {str(e)}")
        traceback.print_exc()

def track_article_interaction(user_id, brief_query, article_url, action, time_spent=None):
    """
    Track article interaction events.
    
    Args:
        user_id: The user's ID
        brief_query: The brief query string
        article_url: The URL of the article
        action: One of 'click', 'view'
        time_spent: Optional time spent in seconds
    """
    if not current_app.config.get('ANALYTICS_ENABLED', True):
        return

    try:
        db = get_db()
        session_id = _get_session_id()
        
        print(f"Tracking article interaction: {action} for article '{article_url}' by user {user_id}")
        
        db.execute(
            """
            INSERT INTO article_interactions
            (user_id, session_id, brief_query, article_url, action, time_spent)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, session_id, brief_query, article_url, action, time_spent)
        )
        
        db.commit()
        print(f"Successfully tracked article interaction: {action}")
    except Exception as e:
        print(f"Error tracking article interaction ({action}): {str(e)}")
        traceback.print_exc()

def track_search_behavior(user_id, query, filters=None, results_count=None):
    """
    Track search behavior events.
    
    Args:
        user_id: The user's ID
        query: The search query string
        filters: Optional dict of filters (will be stored as JSON)
        results_count: Optional count of results returned
    """
    if not current_app.config.get('ANALYTICS_ENABLED', True):
        return

    try:
        db = get_db()
        session_id = _get_session_id()
        filters_json = json.dumps(filters) if filters else None
        
        print(f"Tracking search behavior for query '{query}' by user {user_id}")
        
        db.execute(
            """
            INSERT INTO search_behaviors
            (user_id, session_id, query, filters, results_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, session_id, query, filters_json, results_count)
        )
        
        db.commit()
        print(f"Successfully tracked search behavior")
    except Exception as e:
        print(f"Error tracking search behavior: {str(e)}")
        traceback.print_exc() 