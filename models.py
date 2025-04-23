from flask_login import UserMixin
import json
import os
import time

class User(UserMixin):
    """User class for Flask-Login"""
    
    def __init__(self, id, name, email, profile_pic):
        self.id = id
        self.name = name
        self.email = email
        self.profile_pic = profile_pic

    @staticmethod
    def get(user_id):
        """Retrieve user by ID from our user store"""
        users_dir = "user_data"
        user_file = os.path.join(users_dir, f"{user_id}.json")
        
        if not os.path.exists(user_file):
            return None
            
        try:
            with open(user_file, "r") as f:
                user_data = json.load(f)
                return User(
                    id=user_data["id"],
                    name=user_data["name"],
                    email=user_data["email"],
                    profile_pic=user_data["profile_pic"]
                )
        except Exception as e:
            print(f"Error loading user {user_id}: {e}")
            return None

    def save(self):
        """Save user details to our user store"""
        users_dir = "user_data"
        os.makedirs(users_dir, exist_ok=True)
        
        user_data = {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "profile_pic": self.profile_pic,
            "updated_at": time.time()
        }
        
        try:
            with open(os.path.join(users_dir, f"{self.id}.json"), "w") as f:
                json.dump(user_data, f)
            return True
        except Exception as e:
            print(f"Error saving user {self.id}: {e}")
            return False

    def get_history_file(self):
        """Get the path to this user's search history file"""
        users_dir = "user_data"
        
        # Use email as the identifier for search history
        # Clean the email to make it safe for filesystem use
        safe_email = self.email.replace("@", "_at_").replace(".", "_dot_")
        user_dir = os.path.join(users_dir, safe_email)
        os.makedirs(user_dir, exist_ok=True)
        
        # Path to the new email-based history file
        email_history_file = os.path.join(user_dir, "search_history.json")
        
        # Check if we need to migrate from the old ID-based location
        self.migrate_history_to_email()
        
        return email_history_file
        
    def migrate_history_to_email(self):
        """Migrate history from user ID-based storage to email-based storage"""
        users_dir = "user_data"
        
        # Old ID-based paths
        old_user_dir = os.path.join(users_dir, self.id)
        old_history_file = os.path.join(old_user_dir, "search_history.json")
        
        # New email-based paths
        safe_email = self.email.replace("@", "_at_").replace(".", "_dot_")
        new_user_dir = os.path.join(users_dir, safe_email)
        new_history_file = os.path.join(new_user_dir, "search_history.json")
        
        # If old history exists but new doesn't, migrate the data
        if os.path.exists(old_history_file) and not os.path.exists(new_history_file):
            try:
                os.makedirs(new_user_dir, exist_ok=True)
                
                # Copy the history file
                import shutil
                shutil.copy2(old_history_file, new_history_file)
                
                print(f"Migrated search history for {self.email} from ID-based to email-based storage")
            except Exception as e:
                print(f"Error migrating history for {self.email}: {e}") 