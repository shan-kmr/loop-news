from flask import current_app
from ..sources.brave_news_api import BraveNewsAPI

_brave_api_client = None

def get_brave_api():
    """Gets a shared instance of the BraveNewsAPI client."""
    global _brave_api_client
    if _brave_api_client is None:
        brave_api_key = current_app.config.get("BRAVE_API_KEY")
        if not brave_api_key:
            print("Error: BRAVE_API_KEY not configured in Flask app.")
            return None # Or raise an exception
        try:
            _brave_api_client = BraveNewsAPI(api_key=brave_api_key)
            print("Brave API client initialized.")
        except Exception as e:
            print(f"Error initializing Brave API client: {e}")
            # Decide how to handle: return None, raise, etc.
            return None
    return _brave_api_client

