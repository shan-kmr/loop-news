"""
SQLite database setup and connection management for analytics.
"""

import sqlite3
import os
import traceback
from flask import g, current_app
from datetime import datetime

# Database schema definition
SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS user_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        login_time TIMESTAMP NOT NULL,
        logout_time TIMESTAMP,
        ip_address TEXT,
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brief_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        brief_query TEXT NOT NULL,
        action TEXT NOT NULL,  -- 'create', 'view', 'delete', 'modify'
        parameters TEXT,  -- JSON string of parameters (count, freshness, etc.)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS article_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        brief_query TEXT NOT NULL,
        article_url TEXT NOT NULL,
        action TEXT NOT NULL,  -- 'click', 'view'
        time_spent INTEGER,  -- seconds, if tracked
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS search_behaviors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        query TEXT NOT NULL,
        filters TEXT,  -- JSON string of filters applied
        results_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
]

def get_db():
    """Get database connection from Flask g object or create a new one."""
    try:
        if 'analytics_db' not in g:
            db_path = current_app.config.get('ANALYTICS_DB_PATH')
            if not db_path:
                print("ERROR: ANALYTICS_DB_PATH not set in app config")
                return None
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Create the database connection with isolation level
            g.analytics_db = sqlite3.connect(db_path, isolation_level=None)  # autocommit mode
            g.analytics_db.row_factory = sqlite3.Row
            
            print(f"Database connection established at {db_path}")
        
        return g.analytics_db
    except Exception as e:
        print(f"Error getting database connection: {str(e)}")
        traceback.print_exc()
        return None

def close_db(e=None):
    """Close database connection."""
    try:
        db = g.pop('analytics_db', None)
        if db is not None:
            db.close()
            print("Database connection closed")
    except Exception as e:
        print(f"Error closing database: {str(e)}")
        traceback.print_exc()

def init_db(app):
    """Initialize the database with the schema."""
    try:
        print(f"Initializing analytics database at {app.config.get('ANALYTICS_DB_PATH')}")
        
        # Ensure the instance directory exists
        os.makedirs(app.instance_path, exist_ok=True)
        
        with app.app_context():
            db = get_db()
            if not db:
                print("ERROR: Could not get database connection for initialization")
                return False
                
            # Create tables
            for table_sql in SCHEMA:
                db.execute(table_sql)
            
            db.commit()
            print("Analytics database initialized successfully.")
            return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        traceback.print_exc()
        return False

def register_db_teardown(app):
    """Register database teardown with the Flask app."""
    app.teardown_appcontext(close_db) 