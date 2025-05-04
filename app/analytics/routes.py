"""
Analytics routes for tracking events that need their own endpoints.
"""

import os
import traceback
import json
import sqlite3
from flask import Blueprint, request, redirect, current_app, jsonify, send_from_directory, url_for
from flask_login import current_user, login_required
from .events import track_article_interaction
from datetime import datetime

# Create Blueprint
analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/analytics.js')
def analytics_js():
    """Serve the analytics.js file."""
    try:
        template_dir = os.path.join(current_app.root_path, '..', 'templates', 'analytics')
        print(f"Serving analytics.js from {template_dir}")
        return send_from_directory(template_dir, 'analytics.js')
    except Exception as e:
        print(f"Error serving analytics.js: {str(e)}")
        traceback.print_exc()
        return "console.error('Error loading analytics script');", 500, {'Content-Type': 'application/javascript'}

@analytics_bp.route('/track/article_click', methods=['GET'])
def track_article_click():
    """
    Track an article click and redirect to the article URL.
    
    Query parameters:
    - url: The URL of the article to redirect to
    - brief: The brief query that led to this article
    """
    try:
        article_url = request.args.get('url', '')
        brief_query = request.args.get('brief', '')
        
        print(f"Article click tracking request: URL={article_url}, Brief={brief_query}")
        
        if article_url and brief_query and current_user.is_authenticated:
            # Track the article click
            print(f"Tracking article click for user {current_user.id}")
            tracking_success = track_article_interaction(
                user_id=current_user.id,
                brief_query=brief_query,
                article_url=article_url,
                action='click'
            )
            if not tracking_success:
                print(f"Warning: Failed to track article click for URL: {article_url}")
        else:
            if not current_user.is_authenticated:
                print("Article click not tracked: User not authenticated")
            else:
                print(f"Article click not tracked: Missing parameters. URL={article_url}, Brief={brief_query}")
        
        # Redirect to the article
        if article_url:
            return redirect(article_url)
        else:
            return redirect('/')
    except Exception as e:
        print(f"Error tracking article click: {str(e)}")
        traceback.print_exc()
        
        # Still redirect to avoid breaking user experience
        if article_url:
            return redirect(article_url)
        return redirect('/')

@analytics_bp.route('/api/track/article_view', methods=['POST'])
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
    try:
        print("Article view tracking request received")
        
        # Get data from request
        data = request.get_json()
        if not data:
            print("No JSON data in request")
            print(f"Request data: {request.data}")
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
            
        print(f"Article view data: {json.dumps(data)}")
        
        article_url = data.get('url', '')
        brief_query = data.get('brief', '')
        time_spent = data.get('time_spent')
        
        if article_url and brief_query and current_user.is_authenticated:
            # Track the article view
            print(f"Tracking article view for user {current_user.id}, time spent: {time_spent}s")
            tracking_success = track_article_interaction(
                user_id=current_user.id,
                brief_query=brief_query,
                article_url=article_url,
                action='view',
                time_spent=time_spent
            )
            
            if tracking_success:
                return jsonify({'status': 'success'})
            else:
                print(f"Warning: Failed to track article view for URL: {article_url}")
                return jsonify({'status': 'warning', 'message': 'Analytics event recorded but not confirmed'}), 200
        else:
            if not current_user.is_authenticated:
                print("Article view not tracked: User not authenticated")
                return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
            else:
                print(f"Article view not tracked: Missing parameters. URL={article_url}, Brief={brief_query}")
                return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400
    except Exception as e:
        print(f"Error tracking article view: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

@analytics_bp.route('/debug')
@login_required
def debug_analytics():
    """Debugging endpoint to check if analytics is working."""
    try:
        # Get database stats
        from .database import get_db
        db = get_db()
        
        if not db:
            return jsonify({
                'status': 'error',
                'message': 'Could not connect to analytics database'
            }), 500
            
        # Get counts from each table
        stats = {}
        for table in ['user_sessions', 'brief_interactions', 'article_interactions', 'search_behaviors']:
            cursor = db.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = cursor.fetchone()['count']
        
        # Get database path
        db_path = current_app.config.get('ANALYTICS_DB_PATH', 'Not set')
        
        # Get session ID
        from .events import _get_session_id
        session_id = _get_session_id()
        
        return jsonify({
            'status': 'success',
            'analytics_enabled': current_app.config.get('ANALYTICS_ENABLED', True),
            'database_path': db_path,
            'session_id': session_id,
            'user_authenticated': current_user.is_authenticated,
            'user_id': current_user.id if current_user.is_authenticated else None,
            'table_stats': stats
        })
    except Exception as e:
        print(f"Error in debug endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

@analytics_bp.route('/test-db-write')
@login_required
def test_db_write():
    """Test endpoint to verify we can write to the database."""
    try:
        from .database import get_db
        db_path = current_app.config.get('ANALYTICS_DB_PATH', 'Not set')
        
        # Test 1: Check if we can connect directly with SQLite
        direct_conn = None
        direct_conn_error = None
        try:
            direct_conn = sqlite3.connect(db_path)
            # Try a simple query
            cursor = direct_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        except Exception as e:
            direct_conn_error = str(e)
        finally:
            if direct_conn:
                direct_conn.close()
        
        # Test 2: Check if we can get the connection through our utility
        db = get_db()
        if not db:
            return jsonify({
                'status': 'error',
                'message': 'Could not connect to analytics database through get_db()',
                'direct_connection': {
                    'success': direct_conn_error is None,
                    'error': direct_conn_error
                },
                'database_path': db_path,
                'path_exists': os.path.exists(db_path) if db_path != 'Not set' else False,
                'directory_exists': os.path.exists(os.path.dirname(db_path)) if db_path != 'Not set' else False
            }), 500
        
        # Test 3: Try a write operation
        test_table = """
        CREATE TABLE IF NOT EXISTS analytics_test (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        test_result = {
            'table_creation': False,
            'data_insertion': False,
            'data_retrieval': False,
            'errors': []
        }
        
        try:
            # Create test table
            db.execute(test_table)
            db.commit()
            test_result['table_creation'] = True
            
            # Insert test data
            test_data = f"Test data from {current_user.id} at {datetime.utcnow().isoformat()}"
            db.execute("INSERT INTO analytics_test (test_data) VALUES (?)", (test_data,))
            db.commit()
            test_result['data_insertion'] = True
            
            # Read test data
            cursor = db.execute("SELECT test_data FROM analytics_test ORDER BY timestamp DESC LIMIT 1")
            retrieved_data = cursor.fetchone()
            if retrieved_data and retrieved_data[0]:
                test_result['data_retrieval'] = True
        except Exception as e:
            test_result['errors'].append(str(e))
            traceback.print_exc()

        # Test 4: Check file permissions
        file_permissions = None
        if db_path != 'Not set' and os.path.exists(db_path):
            try:
                import stat
                st = os.stat(db_path)
                file_permissions = {
                    'mode': stat.filemode(st.st_mode),
                    'owner': st.st_uid,
                    'group': st.st_gid,
                    'size': st.st_size,
                }
            except Exception as e:
                file_permissions = {'error': str(e)}
        
        return jsonify({
            'status': 'success',
            'database_path': db_path,
            'path_exists': os.path.exists(db_path) if db_path != 'Not set' else False,
            'direct_connection': {
                'success': direct_conn_error is None,
                'error': direct_conn_error
            },
            'file_permissions': file_permissions,
            'test_results': test_result,
            'user_id': current_user.id if current_user.is_authenticated else None
        })
    except Exception as e:
        print(f"Error in test-db-write endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500 