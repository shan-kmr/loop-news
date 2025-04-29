import re
import string

def clean_text(text):
    """Clean and preprocess text for similarity comparison."""
    if not text:
        return ""
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_age_in_seconds(article):
    """Convert age strings to seconds for sorting"""
    age = article.get("age", "Unknown")
    
    # Try to extract time information from the age string
    if isinstance(age, str):
        try:
            # Handle common formats like "2 hours ago", "1 day ago", etc.
            if "minute" in age:
                minutes = int(''.join(filter(str.isdigit, age)))
                return minutes * 60
            elif "hour" in age:
                hours = int(''.join(filter(str.isdigit, age)))
                return hours * 3600
            elif "day" in age:
                days = int(''.join(filter(str.isdigit, age)))
                return days * 86400
            elif "week" in age:
                weeks = int(''.join(filter(str.isdigit, age)))
                return weeks * 604800
            elif "month" in age:
                months = int(''.join(filter(str.isdigit, age)))
                return months * 2592000
        except ValueError:
             # If digits cannot be extracted, fall through to default
            pass
    
    # Default to a large number for unknown ages or failed parsing
    return float('inf')

def is_valid_summary(summary):
    """Check if a summary exists and is valid."""
    if summary is None:
        return False
    if not isinstance(summary, str):
        return False
    if len(summary.strip()) == 0:
        return False
    # Optional: Add more checks like minimum length, etc.
    return True 