import os
import secrets
from flask import Blueprint, url_for, redirect, session, flash, current_app, render_template, request
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from .oauth import oauth
from ..models import User
from .whitelist import is_email_allowed
from .forms import RegistrationForm, LoginForm, RequestResetForm, ResetPasswordForm


# Setup LoginManager
login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # Route name for the login page

@login_manager.user_loader
def load_user(user_id):
    """Load user from storage for Flask-Login."""
    # user_id here is expected to be the email address based on current setup
    return User.get(user_id)

# Create Blueprint
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle both Google OAuth login and email/password login."""
    # If the user is already authenticated, redirect them
    if current_user.is_authenticated:
        return redirect(url_for('news.index'))
    
    # Initialize login form
    form = LoginForm()
    
    # If form is submitted and valid
    if form.validate_on_submit():
        user = User.get_by_email(form.email.data)
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page if next_page else url_for('news.index'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'error')
    
    # Return the login page with form
    return render_template('auth/login.html', form=form)

@auth_bp.route('/google_login')
def google_login():
    """Initiate the Google OAuth login flow."""
    # Specify the redirect URI for the callback
    redirect_uri = url_for('auth.callback', _external=True)
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
                    profile_pic=user_info.get('picture'),
                    auth_provider='google'
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

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration with email and password."""
    if current_user.is_authenticated:
        return redirect(url_for('news.index'))
    
    form = RegistrationForm()
    
    if form.validate_on_submit():
        email = form.email.data
        
        # Check whitelist
        if not is_email_allowed(email):
            flash('Your email is not on the allowed list. Please request access.', 'error')
            return redirect(url_for('auth.register'))
        
        # Create new user
        user = User.create(
            id=email,
            name=form.name.data,
            email=email,
            password=form.password.data,
            auth_provider='email'
        )
        
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html', form=form)

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

@auth_bp.route('/reset_password', methods=['GET', 'POST'])
def reset_request():
    """Handle password reset requests."""
    if current_user.is_authenticated:
        return redirect(url_for('news.index'))
    
    form = RequestResetForm()
    
    if form.validate_on_submit():
        user = User.get_by_email(form.email.data)
        if user:
            # In a real app, you would send an email with a reset token
            # For this example, we'll just use a session variable
            token = secrets.token_hex(16)
            session['reset_token'] = token
            session['reset_email'] = user.email
            
            # Simulate email sending
            flash(f'A password reset link would be sent to {user.email}. For this demo, you can use this link:', 'info')
            reset_url = url_for('auth.reset_token', token=token, _external=True)
            flash(f'<a href="{reset_url}">Reset Password</a>', 'info')
        else:
            # Don't reveal if user exists for security
            flash('If that email exists in our system, a password reset link has been sent.', 'info')
        
        return redirect(url_for('auth.login'))
    
    return render_template('auth/reset_request.html', form=form)

@auth_bp.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    """Handle password reset with token."""
    if current_user.is_authenticated:
        return redirect(url_for('news.index'))
    
    # Check if the token is valid
    if 'reset_token' not in session or 'reset_email' not in session or session['reset_token'] != token:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('auth.reset_request'))
    
    form = ResetPasswordForm()
    
    if form.validate_on_submit():
        user = User.get_by_email(session['reset_email'])
        if user:
            user.set_password(form.password.data)
            user.save()
            
            # Clear session variables
            session.pop('reset_token', None)
            session.pop('reset_email', None)
            
            flash('Your password has been updated! You can now log in.', 'success')
            return redirect(url_for('auth.login'))
    
    return render_template('auth/reset_token.html', form=form) 