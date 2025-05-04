"""
Analytics module for tracking user interactions in the Loop News application.
"""

import os
import traceback
from flask import current_app

def init_analytics(app):
    """Initialize the analytics module with the Flask app instance."""
    try:
        from .database import init_db
        
        # Check if analytics is enabled
        if not app.config.get('ANALYTICS_ENABLED', True):
            print("Analytics disabled via configuration")
            return False
        
        # Create database directory if it doesn't exist
        db_path = os.path.join(app.instance_path, 'analytics.db')
        app.config['ANALYTICS_DB_PATH'] = db_path
        
        print(f"Initializing analytics module with database at: {db_path}")
        
        # Create instance directory
        os.makedirs(app.instance_path, exist_ok=True)
        
        # Initialize the database
        with app.app_context():
            success = init_db(app)
            if success:
                print("Analytics module initialized successfully")
            else:
                print("WARNING: Analytics database initialization failed")
        
        return success
    except Exception as e:
        print(f"Error initializing analytics module: {str(e)}")
        traceback.print_exc()
        return False 