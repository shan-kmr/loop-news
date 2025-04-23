from flask import Blueprint, request, redirect, url_for, session, current_app, flash, render_template
from flask_login import login_user, logout_user, current_user
from authlib.integrations.flask_client import OAuth
import json
import requests
import os
from models import User

# Blueprint
auth_bp = Blueprint('auth', __name__)

# OAuth setup
oauth = OAuth()

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

def is_email_allowed(email):
    """Check if the email is in the whitelist file"""
    whitelist_file = os.path.join("user_data", "allowed_emails.txt")
    
    # Create whitelist file if it doesn't exist
    if not os.path.exists(whitelist_file):
        os.makedirs("user_data", exist_ok=True)
        with open(whitelist_file, "w") as f:
            # Add a default admin email - replace with your own
            f.write("admin@example.com\n")
    
    # Read the whitelist file and check if the email is present
    try:
        with open(whitelist_file, "r") as f:
            allowed_emails = [line.strip().lower() for line in f.readlines()]
            return email.lower() in allowed_emails
    except Exception as e:
        print(f"Error checking whitelist: {e}")
        return False

def add_email_to_waitlist(email):
    """Add an email to the waitlist file"""
    waitlist_file = os.path.join("user_data", "access_requests.txt")
    os.makedirs("user_data", exist_ok=True)
    
    try:
        # Check if email already exists in the waitlist
        if os.path.exists(waitlist_file):
            with open(waitlist_file, "r") as f:
                existing_emails = [line.strip().lower() for line in f.readlines()]
                if email.lower() in existing_emails:
                    return False  # Email already in waitlist
        
        # Append the email to the waitlist
        with open(waitlist_file, "a") as f:
            f.write(f"{email}\n")
        return True
    except Exception as e:
        print(f"Error adding to waitlist: {e}")
        return False

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

@auth_bp.route('/callback')
def callback():
    """Handle the callback from Google"""
    try:
        # Get token and user information from Google
        token = oauth.google.authorize_access_token()
        user_info = token.get('userinfo')
        
        if user_info and 'email' in user_info:
            # Check if the email is allowed
            if not is_email_allowed(user_info['email']):
                # Email not in whitelist - store in session for the request form
                session['pending_email'] = user_info['email']
                flash('Your email is not on the allowed list. Please request access.', 'error')
                return redirect(url_for('index'))
            
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
        
        flash('Authentication failed. Please try again.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        flash('Authentication error. Please try again later.', 'error')
        return redirect(url_for('index'))

@auth_bp.route('/logout')
def logout():
    """Log out the user"""
    logout_user()
    return redirect(url_for('index')) 