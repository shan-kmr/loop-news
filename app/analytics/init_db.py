"""
Script to initialize the analytics database.
"""

import sqlite3
import os
from flask import Flask
from app.analytics.database import SCHEMA

def init_db(db_path):
    """Initialize the analytics database with the schema."""
    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Create tables
    for table_sql in SCHEMA:
        conn.execute(table_sql)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Analytics database initialized at {db_path}")

if __name__ == "__main__":
    app = Flask(__name__)
    db_path = os.path.join(app.instance_path, 'analytics.db')
    
    # Create instance directory if it doesn't exist
    os.makedirs(app.instance_path, exist_ok=True)
    
    init_db(db_path)
    print("Database initialization complete.") 