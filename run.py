import os
from app import create_app

# Load configuration from config.py by default
# You can switch configs by setting the environment variable, e.g., 
# export FLASK_CONFIG=development or production
# config_name = os.getenv('FLASK_CONFIG', 'default') 
# For simplicity now, we load directly using the default path in create_app

app = create_app('config') # Pass the config module name/path

if __name__ == '__main__':
    # Get host and port from environment variables or use defaults
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_RUN_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Print startup messages (similar to the old ones)
    print("Starting Loop News application...")
    print(f" * Environment: {'development' if debug else 'production'}")
    print(f" * Debug mode: {debug}")
    print(f" * Running on http://{host}:{port}")
    
    # Print library availability notes
    try:
         import sklearn
         print(" * scikit-learn found.")
    except ImportError:
         print("NOTE: scikit-learn not installed. Run 'pip install scikit-learn' for TF-IDF topic grouping fallback.")
         
    try:
         import openai
         print(" * OpenAI library found.")
         if not app.config.get('OPENAI_API_KEY'):
              print("Warning: OpenAI library found, but OPENAI_API_KEY is not set in config.")
    except ImportError:
         print("NOTE: OpenAI library not installed. Run 'pip install openai' to enable AI summaries and grouping.")
         
    # Check for Google OAuth credentials
    if not app.config.get('GOOGLE_CLIENT_ID') or not app.config.get('GOOGLE_CLIENT_SECRET'):
         print("Warning: GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET not set in config. Google Login will fail.")
    else:
         print(" * Google OAuth credentials found.")
         
    if not app.config.get('BRAVE_API_KEY'):
         print("Warning: BRAVE_API_KEY not set in config. News fetching will fail.")
    else:
         print(" * Brave API Key found.")
         
    # Check for Mail settings (important for notifications)
    if not app.config.get('MAIL_SERVER') or not app.config.get('MAIL_USERNAME') or not app.config.get('MAIL_PASSWORD'):
        print("Warning: Mail settings (MAIL_SERVER, MAIL_USERNAME, MAIL_PASSWORD) not fully configured. Email notifications will fail.")
    else:
        print(" * Mail settings found.")

    # Start the Flask development server
    # The notification scheduler is started within create_app based on environment
    app.run(host=host, port=port, debug=debug) 