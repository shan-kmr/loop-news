import os # Added import
from flask import Blueprint, url_for, redirect, session, flash, current_app # Added current_app
from flask_login import LoginManager, login_user, logout_user, login_required, current_user # Added current_user

from .oauth import oauth # Import the initialized OAuth object
from ..models import User
from .whitelist import is_email_allowed 


# Setup LoginManager
login_manager = LoginManager()
login_manager.login_view = 'auth.login' # Route name for the login page

@login_manager.user_loader
def load_user(user_id):
    """Load user from storage for Flask-Login."""
    # user_id here is expected to be the email address based on current setup
    return User.get(user_id)

# Create Blueprint
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login')
def login():
    """Initiate the Google OAuth login flow."""
    # Specify the redirect URI for the callback
    redirect_uri = url_for('auth.callback', _external=True) # Changed to auth.callback
    print(f"Redirecting to Google OAuth with callback: {redirect_uri}")
    # Use the registered 'google' provider
    return oauth.google.authorize_redirect(redirect_uri)

@auth_bp.route('/callback') 
def callback(): 
    """Handle the callback from Google after successful OAuth."""
    try:
        # Fetch the token from Google
        token = oauth.google.authorize_access_token()
        # Fetch user info using the token (old method)
        user_info = token.get('userinfo')

        if user_info and 'email' in user_info:
            user_email = user_info['email']
            print(f"OAuth successful for: {user_email}")

            # --- Whitelist Check (using imported function) ---
            if not is_email_allowed(user_email):
                print(f"Access denied for {user_email}: Not in allowed list.")
                flash('Your email is not on the allowed list. Please request access.', 'error')
                return redirect(url_for('news.index'))

            print(f"Access granted for allowed user: {user_email}")

            # Get or create the user object using email as ID
            user = User.get(user_email)
            if not user:
                print(f"User {user_email} not found locally, creating new user record.")
                user = User.create(
                    id=user_email, # Use email as ID
                    name=user_info.get('name'),
                    email=user_email,
                    profile_pic=user_info.get('picture')
                )
                # No user.save() needed as User.create handles persistence
            else:
                 print(f"Found existing user: {user_email}")

            # Log the user in using Flask-Login
            login_user(user)
            print(f"User {user.email} logged in successfully.")

            # Redirect to the main application page after login (news.index)
            next_url = session.pop('next', url_for('news.index'))
            return redirect(next_url)

        # Fallback if user_info or email is missing
        flash('Authentication failed: Could not retrieve user information.', 'error')
        return redirect(url_for('news.index'))

    except Exception as e:
        print(f"Error during OAuth callback: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Authentication failed. Please try again.', 'error')
        return redirect(url_for('news.index')) # Redirect to index on error

@auth_bp.route('/logout')
@login_required # Ensure user is logged in before they can log out
def logout():
    """Log the current user out."""
    if current_user and hasattr(current_user, 'email'):
         print(f"Logging out user: {current_user.email}")
    else:
         print("Logging out user (no email info available).")
    logout_user()
    # Optionally clear session data if needed
    # session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('news.index')) # Redirect to index page after logout 