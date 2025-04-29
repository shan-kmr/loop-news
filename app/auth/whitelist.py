import os
import json

WAITLIST_FILE = 'user_data/waitlist.json'
ALLOWED_USERS_FILE = 'user_data/allowed_users.json' # Or integrate with DB

"""Load emails from a JSON file, can be used for both waitlist and allowed users."""
def _load_emails(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if not content:
                 return []
            return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading emails from {filepath}: {e}")
        return []

"""Save emails to a JSON file, can be used for both waitlist and allowed users."""
def _save_emails(filepath, emails):
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name:
             os.makedirs(dir_name, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(emails, f, indent=2)
    except IOError as e:
        print(f"Error saving emails to {filepath}: {e}")

def is_email_allowed(email):
    """Check if an email is in the allowed list."""
    allowed_emails = _load_emails(ALLOWED_USERS_FILE)
    return email in allowed_emails

def add_email_to_waitlist(email):
    """Add an email to the waitlist if not already allowed or waitlisted."""
    if is_email_allowed(email):
        print(f"Email {email} is already allowed.")
        return False # Or indicate already allowed
        
    waitlist_emails = _load_emails(WAITLIST_FILE)
    if email in waitlist_emails:
        print(f"Email {email} is already on the waitlist.")
        return False # Indicate already on waitlist
        
    waitlist_emails.append(email)
    _save_emails(WAITLIST_FILE, waitlist_emails)
    print(f"Email {email} added to the waitlist.")
    # Here you might trigger a notification to the admin
    return True # Indicate successfully added

"""Take an email from the waitlist and add it to the allowed list."""
def approve_email(email):
    waitlist_emails = _load_emails(WAITLIST_FILE)
    allowed_emails = _load_emails(ALLOWED_USERS_FILE)
    
    if email not in waitlist_emails:
        print(f"Email {email} not found on the waitlist.")
        return False
        
    if email in allowed_emails:
        print(f"Email {email} is already allowed. Removing from waitlist.")
    else:
        allowed_emails.append(email)
        _save_emails(ALLOWED_USERS_FILE, allowed_emails)
        print(f"Email {email} approved and added to allowed list.")
        
    # Remove from waitlist
    waitlist_emails.remove(email)
    _save_emails(WAITLIST_FILE, waitlist_emails)
    return True

"""List emails currently on the waitlist."""
def list_waitlist():
    return _load_emails(WAITLIST_FILE)

"""List emails currently allowed."""
def list_allowed():
    return _load_emails(ALLOWED_USERS_FILE) 