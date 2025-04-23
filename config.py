import os

# Flask app configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Brave News API configuration
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")

# OpenAI configuration
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4.1-nano"

# Model provider: "openai" or "llama"
MODEL_PROVIDER = "openai"

# Cache settings
CACHE_VALIDITY_HOURS = 1 