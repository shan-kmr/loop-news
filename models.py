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
        user_dir = os.path.join(users_dir, self.id)
        os.makedirs(user_dir, exist_ok=True)
        return os.path.join(user_dir, "search_history.json") 