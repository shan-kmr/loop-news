"""
Analytics module for tracking user interactions in the Loop News application.
"""

from flask import current_app
import os

# Initialize analytics when app is created
def init_analytics(app):
    """Initialize the analytics module with the Flask app instance."""
    from .database import init_db
    
    # Create database directory if it doesn't exist
    db_path = os.path.join(app.instance_path, 'analytics.db')
    app.config['ANALYTICS_DB_PATH'] = db_path
    
    # Initialize the database
    with app.app_context():
        init_db(app)
        
    print("Analytics module initialized.")
    
    return True 