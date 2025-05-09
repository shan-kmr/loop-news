import os

# Flask app configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "451540283630-a8o5frjc0na6jsgbffu7ke675t9g78p2.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "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")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Brave News API configuration
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "BSARtNDwLSo6fXyitbrM3-vdUY1H6rV")

# OpenAI configuration
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4.1-nano"

# Model provider: "openai" or "llama"
MODEL_PROVIDER = "openai"

# Mail server settings for notifications
MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "True").lower() == "true"
MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "shantanu.kum97@gmail.com")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "gqgajschaboevchb")
MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "shantanu.kum97@gmail.com")

# Cache settings
CACHE_VALIDITY_HOURS = 1 
