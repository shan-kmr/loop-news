from ..sources.reddit_api import RedditAPI

_reddit_api_client = None

def get_reddit_api():
    """Gets a shared instance of the RedditAPI client."""
    global _reddit_api_client
    if _reddit_api_client is None:
        try:
            _reddit_api_client = RedditAPI()
            _reddit_api_client.init() # Assuming init needs to be called
            print("Reddit API client initialized.")
        except Exception as e:
            print(f"Warning: Could not initialize Reddit API: {str(e)}")
            print("Reddit content will not be available.")
            _reddit_api_client = None # Ensure it remains None on failure
            return None
    return _reddit_api_client

