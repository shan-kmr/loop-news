from flask import Blueprint, request, redirect, url_for, session, current_app
from flask_login import login_user, logout_user, current_user
from authlib.integrations.flask_client import OAuth
import json
import requests
from models import User

# Blueprint
auth_bp = Blueprint('auth', __name__)

# OAuth setup
oauth = OAuth()

"""
When your application starts, you call init_oauth(app) to wire Authlib into Flask and register Google 
as an OAuth provider. Under the hood, this tells Authlib where to find Google’s discovery document 
and which client ID/secret to use. From then on, Authlib can generate the correct endpoints and handle 
token exchanges without you writing low‑level HTTP code.
"""

def init_oauth(app):
    """Initialize OAuth with the app"""
    oauth.init_app(app)
    
    # Google OAuth configuration
    oauth.register(
        name='google',
        client_id=app.config.get('GOOGLE_CLIENT_ID'),
        client_secret=app.config.get('GOOGLE_CLIENT_SECRET'),
        server_metadata_url=app.config.get('GOOGLE_DISCOVERY_URL'),
        client_kwargs={
            'scope': 'openid email profile'
        }
    )

"""
Hitting the /login route kicks off the user‑facing side of that flow. First, it asks Flask‑Login whether 
someone’s already authenticated—if they are, it skips any extra steps and sends them back home. 
If they’re not, it hands off to Authlib to build a redirect to Google’s consent screen, so the user can grant 
your app access to their basic profile and email.
"""

@auth_bp.route('/login')
def login():
    """Redirect to Google for authentication"""
    # Check if already logged in
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    # Print the redirect URI for debugging
    redirect_uri = url_for('auth.callback', _external=True)
    print(f"Redirect URI being used: {redirect_uri}")
        
    # Generate redirect URL to Google's OAuth page
    return oauth.google.authorize_redirect(redirect_uri)

"""
The callback route is where the magic happens. When the user returns from Google, the URL contains a code 
and state parameter. The callback route uses that code to request an access token and user information.
"""

@auth_bp.route('/callback')
def callback():
    """Handle the callback from Google"""
    # Get token and user information from Google
    token = oauth.google.authorize_access_token()
    user_info = token.get('userinfo')
    
    if user_info:
        # Create or update user
        user = User(
            id=user_info['sub'],
            name=user_info['name'],
            email=user_info['email'],
            profile_pic=user_info.get('picture', '')
        )
        user.save()
        
        # Log in the user
        login_user(user)
        
        # Redirect to main page or next URL (if specified)
        next_url = session.pop('next', '/')
        return redirect(next_url)
    
    return 'Authentication failed', 401


"""
The logout route is straightforward. It uses Flask‑Login’s logout_user() function to log out the current user.
"""
@auth_bp.route('/logout')
def logout():
    """Log out the user"""
    logout_user()
    return redirect(url_for('index')) 