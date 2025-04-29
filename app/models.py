import os
import json
from flask_login import UserMixin

class User(UserMixin):
    """User model for Flask-Login"""
    def __init__(self, id, name, email, profile_pic):
        self.id = id
        self.name = name
        self.email = email
        self.profile_pic = profile_pic

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
                            profile_pic=user_data.get('profile_pic')
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
    def create(id, name, email, profile_pic):
        # In a real app, you'd save to a database
        # For this example, save to a user-specific JSON file
        safe_email = email.replace("@", "_at_").replace(".", "_dot_")
        user_dir = os.path.join('user_data', safe_email)
        os.makedirs(user_dir, exist_ok=True)
        user_data_file = os.path.join(user_dir, 'user_profile.json')
        
        user_data = {
            'id': id,
            'name': name,
            'email': email,
            'profile_pic': profile_pic
        }
        
        try:
            with open(user_data_file, 'w') as f:
                json.dump(user_data, f)
            print(f"User profile saved for {email} in {user_data_file}")
        except Exception as e:
            print(f"Error saving user profile {user_data_file}: {e}")
            
        return User(id, name, email, profile_pic)

    def get_history_file(self):
        """Get the path to the user's specific history file."""
        if not self.email:
            return None # Cannot get history file without an email
            
        safe_email = self.email.replace("@", "_at_").replace(".", "_dot_")
        user_dir = os.path.join('user_data', safe_email)
        os.makedirs(user_dir, exist_ok=True) # Ensure directory exists
        return os.path.join(user_dir, 'search_history.json') 