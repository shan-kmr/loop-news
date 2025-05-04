import os
import json
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin):
    """User model for Flask-Login"""
    def __init__(self, id, name, email, profile_pic, password_hash=None, auth_provider='google'):
        self.id = id
        self.name = name
        self.email = email
        self.profile_pic = profile_pic
        self.password_hash = password_hash
        self.auth_provider = auth_provider  # 'google' or 'email'

    def set_password(self, password):
        """Set the password hash for the user."""
        self.password_hash = generate_password_hash(password)
        return self

    def check_password(self, password):
        """Check if the provided password matches the hash."""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def get(user_id):
        # In a real app, you'd fetch from a database
        # For this example, we'll just check against a hardcoded user or allow if whitelisted
        # NOTE: This is NOT secure for production. User data should be stored securely.
        
        # Attempt to load from user-specific data if exists
        safe_email = user_id.replace("@", "_at_").replace(".", "_dot_")
        user_data_file = os.path.join('user_data', safe_email, 'user_profile.json')

        if os.path.exists(user_data_file):
            try:
                with open(user_data_file, 'r') as f:
                    user_data = json.load(f)
                    # Ensure the loaded ID matches the requested user_id (email)
                    if user_data.get('id') == user_id:
                        return User(
                            id=user_data.get('id'),
                            name=user_data.get('name'),
                            email=user_data.get('email'),
                            profile_pic=user_data.get('profile_pic'),
                            password_hash=user_data.get('password_hash'),
                            auth_provider=user_data.get('auth_provider', 'google')
                        )
            except Exception as e:
                print(f"Error loading user profile {user_data_file}: {e}")
        
        # Fallback for initial login or if user file doesn't exist yet
        # A more robust system would query a user database here
        print(f"User profile not found for {user_id}, potentially creating a new user object instance.")
        # We don't have name/profile pic without the stored data, so we might return None 
        # or a partially filled object depending on requirements.
        # For now, return None if file doesn't exist, as the user shouldn't be 'gettable' yet.
        return None

    @staticmethod
    def get_by_email(email):
        """Get a user by email address."""
        return User.get(email)

    @staticmethod
    def create(id, name, email, profile_pic=None, password=None, auth_provider='google'):
        # In a real app, you'd save to a database
        # For this example, save to a user-specific JSON file
        safe_email = email.replace("@", "_at_").replace(".", "_dot_")
        user_dir = os.path.join('user_data', safe_email)
        os.makedirs(user_dir, exist_ok=True)
        user_data_file = os.path.join(user_dir, 'user_profile.json')
        
        # Create user instance
        user = User(id, name, email, profile_pic, auth_provider=auth_provider)
        
        # Set password if provided
        if password:
            user.set_password(password)
        
        user_data = {
            'id': id,
            'name': name,
            'email': email,
            'profile_pic': profile_pic,
            'password_hash': user.password_hash,
            'auth_provider': auth_provider
        }
        
        try:
            with open(user_data_file, 'w') as f:
                json.dump(user_data, f)
            print(f"User profile saved for {email} in {user_data_file}")
        except Exception as e:
            print(f"Error saving user profile {user_data_file}: {e}")
            
        return user

    def save(self):
        """Save the current user state to storage."""
        safe_email = self.email.replace("@", "_at_").replace(".", "_dot_")
        user_dir = os.path.join('user_data', safe_email)
        os.makedirs(user_dir, exist_ok=True)
        user_data_file = os.path.join(user_dir, 'user_profile.json')
        
        user_data = {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'profile_pic': self.profile_pic,
            'password_hash': self.password_hash,
            'auth_provider': self.auth_provider
        }
        
        try:
            with open(user_data_file, 'w') as f:
                json.dump(user_data, f)
            print(f"User profile updated for {self.email}")
            return True
        except Exception as e:
            print(f"Error updating user profile: {e}")
            return False

    def get_history_file(self):
        """Get the path to the user's specific history file."""
        if not self.email:
            return None # Cannot get history file without an email
            
        safe_email = self.email.replace("@", "_at_").replace(".", "_dot_")
        user_dir = os.path.join('user_data', safe_email)
        os.makedirs(user_dir, exist_ok=True) # Ensure directory exists
        return os.path.join(user_dir, 'search_history.json') 