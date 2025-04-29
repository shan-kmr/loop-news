import pytz
from datetime import datetime

def day_group_filter(article):
    """Get the day group for an article (Today, Yesterday, etc.)"""
    age = article.get("age", "Unknown")
    
    # Extract the day for timeline grouping
    day_str = "Today"
    if isinstance(age, str):
        if "day" in age:
            try:
                days = int(''.join(filter(str.isdigit, age)))
                if days == 1:
                    day_str = "Yesterday"
                else:
                    day_str = f"{days} days ago"
            except ValueError:
                pass # Keep default if digits can't be extracted
        elif "week" in age or "month" in age:
            day_str = age
        # Consider adding "hour" and "minute" cases to map to "Today"
        elif "hour" in age or "minute" in age:
            day_str = "Today"
            
    return day_str

def format_datetime_filter(value, format='%Y-%m-%d %H:%M'):
    """Format a datetime object using Eastern Time."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            try:
                # Attempt to parse from timestamp if ISO format fails
                value = datetime.fromtimestamp(float(value))
            except (ValueError, TypeError):
                 return value # Return original string if all parsing fails
    
    # Check if value is a datetime object
    if not isinstance(value, datetime):
        return value

    # Convert to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    
    # If datetime is naive (no timezone), assume it's in UTC
    if value.tzinfo is None:
        value = pytz.utc.localize(value)
    
    # Convert to Eastern Time
    value_eastern = value.astimezone(eastern)
    
    # Format with EST suffix if format includes date elements
    if any(x in format for x in ['%Y', '%m', '%d', '%b']):
        return f"{value_eastern.strftime(format)} EST"
    else:
        return value_eastern.strftime(format) 