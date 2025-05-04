"""
SQLite database setup and connection management for analytics.
"""

import sqlite3
import os
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
    if 'analytics_db' not in g:
        db_path = current_app.config['ANALYTICS_DB_PATH']
        g.analytics_db = sqlite3.connect(db_path)
        g.analytics_db.row_factory = sqlite3.Row
    
    return g.analytics_db

def close_db(e=None):
    """Close database connection."""
    db = g.pop('analytics_db', None)
    if db is not None:
        db.close()

def init_db(app):
    """Initialize the database with the schema."""
    with app.app_context():
        db = get_db()
        
        # Create tables
        for table_sql in SCHEMA:
            db.execute(table_sql)
        
        db.commit()
        print("Analytics database initialized.")

def register_db_teardown(app):
    """Register database teardown with the Flask app."""
    app.teardown_appcontext(close_db) 