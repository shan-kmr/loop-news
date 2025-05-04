import os
import threading
from flask import Flask
from flask_login import LoginManager

# Import configurations and extensions
# from config import Config # Assuming config.py is at root
from .auth.oauth import init_oauth, oauth
from .auth.routes import login_manager
from .models import User # Import User model for user_loader

# Import utility functions
from .utils.filters import day_group_filter, format_datetime_filter
from .utils.text import is_valid_summary

# Import services (needed for background tasks or direct use in factory?)
from .services.openai_service import init_openai
from .services.notifications import schedule_notification_checks

def create_app(config_object='config.Config'): # Use a default config path
    """Application factory pattern."""
    # Tell Flask where to find templates relative to the app directory
    app = Flask(__name__, template_folder='../templates') 
    
    # Load configuration
    # Check if config_object is a string path or an object
    if isinstance(config_object, str):
        app.config.from_object(config_object)
    else:
        app.config.from_mapping(config_object) # Allows passing dict for testing

    # Ensure instance folder exists (if needed for sqlite, sessions etc.)
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass # Ignore if it already exists
        
    print("Flask app created. Initializing extensions...")

    # Initialize Flask extensions
    login_manager.init_app(app)
    print("LoginManager initialized.")
    
    init_oauth(app) # Initialize OAuth provider
    print("OAuth initialized.")

    # Initialize other services that need app context or config
    # Check if OpenAI key exists before trying to initialize
    openai_api_key = app.config.get('OPENAI_API_KEY')
    if openai_api_key:
         if not init_openai(openai_api_key): # Pass the key directly
              print("Warning: OpenAI initialization failed. Summaries/Grouping may be unavailable.")
    else:
         print("OpenAI API Key not found in config. OpenAI features disabled.")
         # Explicitly mark as unavailable in app context?
         # app.config['OPENAI_AVAILABLE'] = False 

    # Register Blueprints
    from .auth.routes import auth_bp
    from .news.routes import news_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(news_bp, url_prefix='/') # Register news at root
    print("Blueprints registered.")

    # Register Jinja Filters and Globals
    app.jinja_env.filters['day_group'] = day_group_filter
    app.jinja_env.filters['format_datetime'] = format_datetime_filter
    app.jinja_env.globals['is_valid_summary'] = is_valid_summary 
    # Make get_flashed_messages available implicitly by Flask
    print("Jinja filters and globals registered.")
    
    # Start background tasks (like notification scheduler)
    # Ensure it only runs once, not per worker in production setups
    # Using a simple flag on the app object for dev server
    if not app.config.get('TESTING', False): # Don't run scheduler during tests
         # Check if running with Werkzeug reloader to avoid duplicate threads
         if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or os.environ.get('WERKZEUG_RUN_MAIN') is None:
             # Start scheduler in a separate thread, passing the app instance
             schedule_notification_checks(app)
         else:
              print("Skipping notification scheduler start in Werkzeug reloader process.")

    print("Application factory setup complete.")
    return app 