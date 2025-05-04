"""
Analytics module for tracking user interactions in the Loop News application.
"""

import os
import traceback
import sqlite3
from flask import current_app

def init_analytics(app):
    """Initialize the analytics module with the Flask app instance."""
    try:
        from .database import init_db
        
        # Check if analytics is enabled
        if not app.config.get('ANALYTICS_ENABLED', True):
            print("Analytics disabled via configuration")
            return False
        
        # Use absolute paths for database to avoid permission issues
        instance_path = os.path.abspath(app.instance_path)
        db_path = os.path.join(instance_path, 'analytics.db')
        app.config['ANALYTICS_DB_PATH'] = db_path
        
        print(f"Initializing analytics module with database at: {db_path}")
        
        # Create instance directory with proper permissions
        os.makedirs(instance_path, exist_ok=True)
        
        # Test database file access before proceeding
        try:
            # Check if we can access the directory
            if not os.access(os.path.dirname(db_path), os.W_OK):
                print(f"WARNING: No write access to directory: {os.path.dirname(db_path)}")
                return False
                
            # Try to create a direct connection to validate SQLite works
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE IF NOT EXISTS _test_table (id INTEGER PRIMARY KEY)")
            conn.execute("SELECT 1 FROM _test_table LIMIT 1")
            conn.close()
            print("Direct SQLite connection test successful")
        except Exception as e:
            print(f"ERROR: Failed SQLite connection test: {str(e)}")
            traceback.print_exc()
            return False
        
        # Initialize the database
        with app.app_context():
            success = init_db(app)
            if success:
                print("Analytics module initialized successfully")
            else:
                print("WARNING: Analytics database initialization failed")
                return False
            
            # Do a quick test insert to validate everything works
            try:
                from .database import get_db
                db = get_db()
                test_data = f"Initialization test at {os.path.basename(__file__)}"
                
                # Create test table if not exists
                db.execute("""
                CREATE TABLE IF NOT EXISTS analytics_init_test (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Insert test data
                db.execute("INSERT INTO analytics_init_test (test_data) VALUES (?)", (test_data,))
                db.commit()
                
                # Verify data was inserted
                cursor = db.execute("SELECT test_data FROM analytics_init_test ORDER BY id DESC LIMIT 1")
                result = cursor.fetchone()
                if result and result[0] == test_data:
                    print("Analytics database write test successful")
                else:
                    print("WARNING: Analytics database write test failed - data mismatch")
                    return False
            except Exception as e:
                print(f"ERROR: Analytics database write test failed: {str(e)}")
                traceback.print_exc()
                return False
        
        return True
    except Exception as e:
        print(f"Error initializing analytics module: {str(e)}")
        traceback.print_exc()
        return False 