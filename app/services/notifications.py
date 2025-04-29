import os
import json
import smtplib
import ssl
import time
import threading
import pytz
import traceback
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import current_app, render_template

"""
About this code:
    This code is responsible for sending email notifications to users when new content is available for topics they've subscribed to.
    It checks if it's time to send an update based on the user's requested frequency (hourly or daily).
    It also checks if there is genuinely new content to send, else doesn't send anything.
    If there is, it prepares an email with the summary and new articles and sends it to the user.
    It also remembers the time it sent the update so it knows when to check again for the user next time.
    It also saves the refreshed news and summaries it gathered.
"""


# Assuming utils are accessible
from ..utils.history import update_current_day_results
from ..utils.text import is_valid_summary, extract_age_in_seconds
from ..utils.filters import day_group_filter
from .openai_service import summarize_daily_news # For refreshing summaries

def save_notification_settings(user_email, topic, frequency):
    """
     Save notification settings to the appropriate history file.
     Finds the file where the app keeps track of news for that topic. Looks inside for that topic and adds 
     the user_email and their choice of frequency (hourly or daily) to a list. If the user was already getting 
     updates, it just changes the frequency if needed. Then, it saves the file with the updated list.
    """
    print(f"Attempting to save notification settings: {user_email}, {topic}, {frequency}")
    
    
    user_specific_history_file = None
    if user_email:
        safe_email = user_email.replace("@", "_at_").replace(".", "_dot_")
        user_specific_dir = os.path.join('user_data', safe_email)
        user_specific_history_file = os.path.join(user_specific_dir, 'search_history.json')

    shared_history_file = os.path.join('user_data', 'shared', 'search_history.json')

    files_to_check = []
    if user_specific_history_file and os.path.exists(user_specific_history_file):
        files_to_check.append(user_specific_history_file)
    if os.path.exists(shared_history_file):
         # Avoid checking shared if user-specific exists and contains the topic ideally
         # But for simplicity now, check both if they exist.
        files_to_check.append(shared_history_file)
        
    if not files_to_check:
        print(f"No history files found to save notification settings for {topic} / {user_email}")
        return False

    found_and_saved = False
    for history_file in files_to_check:
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                
            entry_updated = False
            # Look for the topic in this history file
            for entry in history.items():
                if entry.get('query') == topic:
                    # Ensure notifications structure exists
                    if 'notifications' not in entry or not isinstance(entry['notifications'], dict):
                        entry['notifications'] = {'recipients': [], 'last_sent': {}}
                    if 'recipients' not in entry['notifications'] or not isinstance(entry['notifications']['recipients'], list):
                        entry['notifications']['recipients'] = []
                    if 'last_sent' not in entry['notifications'] or not isinstance(entry['notifications']['last_sent'], dict):
                        entry['notifications']['last_sent'] = {}
                    
                    # Check if this email is already in recipients
                    recipient_exists = False
                    for recipient in entry['notifications']['recipients']:
                        if recipient.get('email') == user_email:
                            recipient['frequency'] = frequency # Update existing
                            recipient_exists = True
                            break
                    
                    # Add new recipient if not found
                    if not recipient_exists:
                        entry['notifications']['recipients'].append({
                            'email': user_email,
                            'frequency': frequency
                        })
                    
                    entry_updated = True
                    found_and_saved = True # Mark that we found and potentially updated the topic
                    print(f"Updated notification settings for '{topic}' to {user_email}: {frequency} in {history_file}")
                    break # Stop searching within this file once topic is found
            
            # Save the updated history file if changes were made
            if entry_updated:
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
                # If saved in user-specific, potentially stop checking shared?
                # For now, let it potentially save in both if topic exists in both
                # break # Uncomment if you only want to save in the first file found

        except FileNotFoundError:
            print(f"History file not found: {history_file}")
        except json.JSONDecodeError:
             print(f"Error decoding JSON from {history_file}. Cannot save notification settings.")
        except Exception as e:
            print(f"Error processing history file {history_file} for notifications: {e}")
            traceback.print_exc()

    if not found_and_saved:
        print(f"Failed to save notification settings: topic '{topic}' not found in checked history files ({files_to_check})")
        return False
        
    return True # Return True if saved in at least one file

def send_notification_email(topic, entry, frequency, recipient_email, new_articles=None):
    """Send notification email for a specific topic."""
    if not recipient_email:
        print(f"Error: No recipient email provided for topic '{topic}'.")
        return False
    
    print(f"Preparing to send {frequency} notification email for topic '{topic}' to {recipient_email}")
    
    # Get necessary data from the entry
    day_summaries = entry.get('day_summaries', {})
    if not isinstance(day_summaries, dict):
         day_summaries = {} # Ensure it's a dict
         
    content_timestamp = entry.get('timestamp')
    if content_timestamp:
        try:
            content_time = datetime.fromisoformat(content_timestamp)
            # Make timezone-aware if naive (assume UTC)
            if content_time.tzinfo is None:
                content_time = pytz.utc.localize(content_time)
            age_minutes = (datetime.now(pytz.utc) - content_time).total_seconds() / 60
            print(f"Content timestamp: {content_timestamp} (age: {int(age_minutes)} minutes)")
        except Exception as e:
            print(f"Error parsing content timestamp: {e}")
    else:
        print("Warning: No content timestamp found in history entry.")

    # Determine which summary and date to use
    summary_text = None
    summary_date_label = None # e.g., "Today", "Yesterday"
    email_date_str = None # Formatted date string for display
    
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    today_date_str = now_eastern.strftime('%b %d, %Y')
    yesterday_date_str = (now_eastern - timedelta(days=1)).strftime('%b %d, %Y')

    # Prioritize based on frequency
    if frequency == 'hourly':
        if is_valid_summary(day_summaries.get('Today')):
            summary_text = day_summaries['Today']
            summary_date_label = 'Today'
            email_date_str = today_date_str
            print(f"Using Today's summary for hourly notification.")
    elif frequency == 'daily':
        if is_valid_summary(day_summaries.get('Yesterday')):
            summary_text = day_summaries['Yesterday']
            summary_date_label = 'Yesterday'
            email_date_str = yesterday_date_str # Use yesterday's date
            print(f"Using Yesterday's summary for daily notification.")

    # Fallback if priority summary wasn't found
    if not summary_text:
        print(f"Priority summary ('{frequency}') not found or invalid for '{topic}'. Looking for fallback.")
        # Look for Today's summary first as fallback
        if is_valid_summary(day_summaries.get('Today')):
            summary_text = day_summaries['Today']
            summary_date_label = 'Today'
            email_date_str = today_date_str
            print(f"Using fallback: Today's summary.")
        # Then Yesterday's
        elif is_valid_summary(day_summaries.get('Yesterday')):
             summary_text = day_summaries['Yesterday']
             summary_date_label = 'Yesterday'
             email_date_str = yesterday_date_str
             print(f"Using fallback: Yesterday's summary.")
        # Then any other valid summary (less ideal)
        else:
            day_priority = {"2 days ago": 3, "3 days ago": 4, "4 days ago": 5, "5 days ago": 6, "6 days ago": 7, "1 week ago": 8, "2 weeks ago": 9, "1 month ago": 10}
            sorted_fallback_days = sorted(day_priority.keys(), key=lambda d: day_priority[d])
            for day in sorted_fallback_days:
                if is_valid_summary(day_summaries.get(day)):
                    summary_text = day_summaries[day]
                    summary_date_label = day
                    # Need to calculate the date string for older days - skip for now for simplicity
                    email_date_str = f"({day})" # Placeholder date
                    print(f"Using fallback: {day}'s summary.")
                    break

    if not summary_text:
        print(f"Error: No valid summary found for topic '{topic}' to send notification.")
        return False
    
    # Prepare new articles HTML (only for hourly for now)
    new_articles_html_content = ""
    if frequency == 'hourly' and new_articles:
        # Ensure new_articles is a list
        if not isinstance(new_articles, list):
             new_articles = []
             
        # Render the snippet for new articles
        # We need an app context to use render_template here
        try:
            with current_app.app_context():
                 # This assumes a separate template snippet exists, e.g., 'new_articles_snippet.html'
                 # Or modify the main notification template to handle this conditionally.
                 # For now, let's build it directly for simplicity
                 new_articles_html_content = f'''
                 <div style="margin-bottom: 30px; padding: 15px; background-color: #f8f8f8; border-left: 4px solid #555;">
                   <h2 style="font-size: 18px; color: #555; margin-bottom: 15px;">Latest Updates ({len(new_articles)} new article{'s' if len(new_articles) > 1 else ''})</h2>
                   <ul style="padding-left: 20px; margin-bottom: 0;">
                 '''
                 for article in new_articles:
                     title = article.get('title', 'No title')
                     description = article.get('description', 'No description available')
                     url = article.get('url', '#')
                     age = article.get('age', 'Unknown time')
                     source_netloc = 'Unknown source'
                     if isinstance(article.get('meta_url'), dict):
                         source_netloc = article['meta_url'].get('netloc', 'Unknown source')
                     
                     new_articles_html_content += f'''
                     <li style="margin-bottom: 15px;">
                       <a href="{url}" style="font-weight: bold; color: #333; text-decoration: none;">{title}</a>
                       <div style="margin-top: 5px; color: #555; font-size: 13px;">
                         {source_netloc} • {age}
                       </div>
                       <div style="margin-top: 5px; line-height: 1.4;">
                         {description[:150]}{ '...' if len(description) > 150 else ''}
                       </div>
                     </li>
                     '''
                 new_articles_html_content += '''
                   </ul>
                 </div>
                 '''
        except RuntimeError as e:
             print(f"Error rendering new articles HTML (likely needs app context): {e}")
             # Fallback: simple text list?
             if new_articles:
                 new_articles_html_content = "<p><b>Latest Updates:</b><br>"
                 for article in new_articles:
                     new_articles_html_content += f"- {article.get('title', 'No title')} ({article.get('age', '')})<br>"
                 new_articles_html_content += "</p>"

    # Render the main email body using the template
    html_body = None
    try:
        # Use app context to render template
        with current_app.app_context():
            html_body = render_template('notification_email.html',
                                       topic=topic,
                                       summary_date=summary_date_label,
                                       email_date_str=email_date_str if email_date_str else "Current",
                                       summary_text=summary_text,
                                       new_articles_html=new_articles_html_content,
                                       frequency=frequency)
    except RuntimeError as e:
        print(f"Error rendering notification template (likely needs app context): {e}")
        # Fallback to plain text or basic HTML
        html_body = f"<h1>Loop: {topic}</h1><p><b>{summary_date_label}'s Summary ({email_date_str} EST)</b></p><p>{summary_text}</p>{new_articles_html_content}"
    except Exception as e:
         print(f"Error rendering notification template: {e}")
         traceback.print_exc()
         return False # Cannot proceed without rendered body
         
    if not html_body:
         print("Failed to render HTML body for notification email.")
         return False

    # Create email message
    msg = MIMEMultipart()
    # Use a generic sender for now, should be configured
    sender_display_name = current_app.config.get('MAIL_SENDER_NAME', "Loop News")
    sender_email_addr = current_app.config.get('MAIL_USERNAME') # Expects sender email from config
    if not sender_email_addr:
         print("Error: MAIL_USERNAME not configured. Cannot send email.")
         return False
         
    msg['From'] = f"{sender_display_name} <{sender_email_addr}>"
    msg['To'] = recipient_email
    msg['Subject'] = f"{frequency.capitalize()} Update: {topic} - {today_date_str} EST"
    
    msg.attach(MIMEText(html_body, 'html'))

    # Send email using configured SMTP settings
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = current_app.config.get('MAIL_PORT', 587) # Default to 587 for TLS
        mail_use_tls = current_app.config.get('MAIL_USE_TLS', True)
        mail_username = "shantanu.kum97@gmail.com"
        mail_password = "gqgajschaboevchb"

        if not all([smtp_server, mail_username, mail_password]):
            print("Error: Mail server settings (MAIL_SERVER, MAIL_USERNAME, MAIL_PASSWORD) incomplete.")
            # Optionally log the email content here instead of sending
            print(f"--- Email Content (Not Sent) ---")
            print(f"To: {recipient_email}")
            print(f"Subject: {msg['Subject']}")
            print(f"Body Preview: {summary_text[:100]}...")
            print(f"----------------------------- ")
            return False # Indicate failure to send

        print(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        context = ssl.create_default_context()
        server = smtplib.SMTP(smtp_server, smtp_port)
        
        if mail_use_tls:
            print("Starting TLS connection")
            server.starttls(context=context)
        
        print(f"Logging in as {mail_username}")
        server.login(mail_username, mail_password)
        
        print(f"Sending email to {recipient_email}")
        server.sendmail(sender_email_addr, recipient_email, msg.as_string())
        print(f"Email sent successfully to {recipient_email}, closing connection")
        server.quit()
        return True

    except smtplib.SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: {e}. Check MAIL_USERNAME and MAIL_PASSWORD.")
        traceback.print_exc()
        return False
    except smtplib.SMTPServerDisconnected as e:
         print(f"SMTP Server Disconnected: {e}. Check server address and port.")
         traceback.print_exc()
         return False
    except ConnectionRefusedError as e:
         print(f"SMTP Connection Refused: {e}. Check server address, port, and firewall.")
         traceback.print_exc()
         return False
    except Exception as e:
        print(f"General Error sending email: {e}")
        traceback.print_exc()
        return False


def process_history_file(file_path, now_eastern, updated_files_dict):
    """Process a single history file for notifications."""
    if not os.path.exists(file_path):
        print(f"History file path does not exist: {file_path}")
        return
        
    try:
        with open(file_path, 'r') as f:
            history = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from history file: {file_path}. Skipping.")
        # Consider backing up corrupted file here
        return
    except Exception as e:
        print(f"Error loading history file {file_path}: {e}")
        return
    
    file_updated = False
    
    # Process each entry in this history file
    for key, entry in history.items():
        # Basic validation of entry structure
        if not isinstance(entry, dict) or 'notifications' not in entry or 'query' not in entry:
            continue
            
        notifications = entry.get('notifications')
        if not isinstance(notifications, dict) or 'recipients' not in notifications or not isinstance(notifications['recipients'], list) or not notifications['recipients']:
            continue # Skip if no valid recipients configured
        
        topic = entry.get('query')
        count = entry.get('count', 10) # Default count
        freshness = entry.get('freshness', 'pw') # Default freshness
        
        # Process each recipient configured for this topic
        for recipient in notifications['recipients']:
            if not isinstance(recipient, dict): continue # Skip invalid recipient entries
                
            email = recipient.get('email')
            frequency = recipient.get('frequency')
            
            if not email or not frequency or frequency not in ['hourly', 'daily']:
                print(f"Skipping invalid recipient config for topic '{topic}': {recipient}")
                continue # Skip invalid recipient config
            
            # Get last sent time, ensuring the structure is correct
            last_sent_dict = notifications.get('last_sent')
            if not isinstance(last_sent_dict, dict):
                last_sent_dict = {}
                notifications['last_sent'] = last_sent_dict # Fix structure
                file_updated = True # Mark file as updated due to structure fix
                
            last_sent_str = last_sent_dict.get(email)
            last_sent_dt = None
            should_send = False
            
            # Determine if notification is due
            if not last_sent_str:
                should_send = True # Never sent before
                print(f"Notification due for '{topic}' to {email} (never sent before). Frequency: {frequency}")
            else:
                try:
                    last_sent_dt = datetime.fromisoformat(last_sent_str)
                    # Make timezone-aware if naive (assume UTC)
                    if last_sent_dt.tzinfo is None:
                        last_sent_dt = pytz.utc.localize(last_sent_dt)
                    
                    last_sent_eastern = last_sent_dt.astimezone(now_eastern.tzinfo)
                    time_diff = now_eastern - last_sent_eastern
                    
                    if frequency == 'hourly' and time_diff.total_seconds() >= 3600: # 1 hour
                        should_send = True
                    elif frequency == 'daily' and time_diff.total_seconds() >= 86400: # 24 hours
                         # Optional: Add logic for specific time of day for daily emails
                        should_send = True
                        
                    if should_send:
                         print(f"Notification due for '{topic}' to {email}. Last sent: {last_sent_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}. Frequency: {frequency}")
                         
                except Exception as e:
                    print(f"Error parsing last sent time '{last_sent_str}' for {email}, topic '{topic}': {e}. Assuming send is due.")
                    should_send = True
            
            # If notification is due, proceed with refresh and send logic
            if should_send:
                print(f"Processing due notification for '{topic}' ({frequency}) to {email}")
                
                has_new_content = False
                new_articles_list = []
                previous_today_summary = entry.get('day_summaries', {}).get('Today')
                
                # Refresh content - requires app context for API keys etc.
                print(f"Refreshing content for '{topic}' before sending notification...")
                results = entry.get('results')
                
                updated_results_data = None
                days_with_new = []
                
                # Ensure results is a dict before passing
                if not isinstance(results, dict):
                     results = {"results": []} # Default structure
                     
                try:
                    # This needs Flask app context
                    with current_app.app_context():
                         updated_results_data, days_with_new = update_current_day_results(topic, count, freshness, results)
                except RuntimeError as e:
                    print(f"RuntimeError (likely missing app context) during content refresh for '{topic}': {e}")
                    # Decide how to proceed: skip send, send old content?
                    print("Skipping notification send due to refresh error.")
                    continue # Skip to next recipient/topic
                except Exception as e:
                     print(f"Unexpected error during content refresh for '{topic}': {e}")
                     traceback.print_exc()
                     print("Skipping notification send due to refresh error.")
                     continue # Skip to next recipient/topic
                
                # Check if content actually updated
                if updated_results_data != results:
                    print(f"Content updated for '{topic}'.")
                    # Update entry in memory
                    entry['results'] = updated_results_data
                    entry['timestamp'] = datetime.now().isoformat()
                    file_updated = True # Mark file for saving later
                    
                    # Find truly new articles (comparing URLs)
                    old_urls = {a.get('url') for a in results.get('results', []) if a.get('url')}
                    new_urls = {a.get('url') for a in updated_results_data.get('results', []) if a.get('url')}
                    truly_new_urls = new_urls - old_urls
                    
                    if truly_new_urls:
                         has_new_content = True
                         # Get the full article dicts for the new URLs, sorted by age
                         new_articles_full = [a for a in updated_results_data.get('results', []) if a.get('url') in truly_new_urls]
                         new_articles_full.sort(key=extract_age_in_seconds)
                         new_articles_list = new_articles_full[:3] # Limit to 3 newest
                         print(f"Found {len(new_articles_list)} new articles to include in notification for '{topic}'.")

                    # Regenerate summaries if needed
                    if 'day_summaries' not in entry or not isinstance(entry['day_summaries'], dict):
                        entry['day_summaries'] = {}
                    
                    current_articles = updated_results_data.get('results', [])
                    articles_by_day = {}
                    for article in current_articles:
                         day = day_group_filter(article)
                         if day not in articles_by_day:
                             articles_by_day[day] = []
                         articles_by_day[day].append(article)
                         
                    # Generate summaries (needs app context)
                    try:
                        with current_app.app_context():
                             # Focus on Today's summary, and days with new content
                             days_to_summarize = set(days_with_new) | {'Today'}
                             for day in days_to_summarize:
                                 if day in articles_by_day:
                                     # Check if summary needs update (missing, invalid, or Today)
                                     if day == 'Today' or not is_valid_summary(entry['day_summaries'].get(day)):
                                         print(f"Generating/Refreshing summary for {day} for topic '{topic}'")
                                         day_summary = summarize_daily_news(articles_by_day[day], topic)
                                         if is_valid_summary(day_summary):
                                             entry['day_summaries'][day] = day_summary
                                             file_updated = True
                                             print(f"Generated/Updated {day} summary: {day_summary[:50]}...")
                                             # Check if Today's summary specifically changed
                                             if day == 'Today' and previous_today_summary != day_summary:
                                                 print("Today's summary has changed.")
                                                 has_new_content = True
                    except RuntimeError as e:
                         print(f"RuntimeError (likely missing app context) during summary generation for '{topic}': {e}")
                    except Exception as e:
                         print(f"Unexpected error during summary generation for '{topic}': {e}")
                         traceback.print_exc()
                         
                else:
                    print(f"Content for '{topic}' did not change after refresh.")
                
                # Decide whether to actually send based on frequency and new content
                should_actually_send = True
                if frequency == 'hourly' and not has_new_content:
                    print(f"Hourly frequency, but no new content detected for '{topic}'. Skipping send to {email}.")
                    should_actually_send = False
                    # Crucially, DO NOT update last_sent time if we skip due to no new content
                    # This allows the next check to correctly evaluate the time diff again.
                
                # Send the email if required
                if should_actually_send:
                    # Send email (needs app context for render_template and mail config)
                    email_sent = False
                    try:
                         with current_app.app_context():
                             email_sent = send_notification_email(topic, entry, frequency, email, new_articles_list)
                    except RuntimeError as e:
                         print(f"RuntimeError (likely missing app context) during email send for '{topic}': {e}")
                    except Exception as e:
                         print(f"Unexpected error during email send for '{topic}': {e}")
                         traceback.print_exc()

                    # Update last sent timestamp ONLY if email was sent successfully
                    if email_sent:
                        current_iso_time = datetime.now(pytz.utc).isoformat() # Store in UTC ISO format
                        print(f"Successfully sent notification for '{topic}' to {email}, updating last_sent to {current_iso_time}")
                        notifications['last_sent'][email] = current_iso_time
                        file_updated = True # Mark file for saving
                    else:
                         print(f"Failed to send notification email for '{topic}' to {email}. Last sent time NOT updated.")

    # After processing all entries, if file was updated, add it to the dict for saving
    if file_updated:
        updated_files_dict[file_path] = history

def check_and_send_notifications():
    """Check for due notifications and send emails by scanning all history files."""
    updated_history_files_to_save = {}
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    print(f"[{now.isoformat()}] Starting notification check...")
    
    # Scan the user_data directory for all history files
    user_data_dir = 'user_data'
    history_files_to_process = []
    
    # 1. Add shared history file if it exists
    shared_history_path = os.path.join(user_data_dir, 'shared', 'search_history.json')
    if os.path.exists(shared_history_path):
        history_files_to_process.append(shared_history_path)
        
    # 2. Add user-specific history files
    try:
        for item_name in os.listdir(user_data_dir):
            item_path = os.path.join(user_data_dir, item_name)
            # Look for directories that aren't 'shared' or 'anonymous' (potential user dirs)
            if os.path.isdir(item_path) and item_name not in ['shared', 'anonymous']:
                user_history_path = os.path.join(item_path, 'search_history.json')
                if os.path.exists(user_history_path):
                    history_files_to_process.append(user_history_path)
    except FileNotFoundError:
        print(f"User data directory '{user_data_dir}' not found.")
    except Exception as e:
         print(f"Error scanning user data directory: {e}")

    print(f"Found {len(history_files_to_process)} history files to check.")

    # Process each found history file
    for file_path in history_files_to_process:
        print(f"Processing file: {file_path}")
        # The actual processing and sending happens within process_history_file
        # It needs the app context, which the background thread should provide.
        process_history_file(file_path, now, updated_history_files_to_save)
    
    # Save any history files that were modified
    if updated_history_files_to_save:
        print(f"Saving updates to {len(updated_history_files_to_save)} history file(s)...")
        for file_path, history_data in updated_history_files_to_save.items():
            try:
                # Ensure directory exists before saving
                dir_name = os.path.dirname(file_path)
                if dir_name:
                     os.makedirs(dir_name, exist_ok=True)
                     
                with open(file_path, 'w') as f:
                    json.dump(history_data, f, indent=2)
                print(f"Successfully saved updates to {file_path}")
            except Exception as e:
                print(f"Error saving updated history file {file_path}: {e}")
                traceback.print_exc()
    else:
         print("No history files required saving after notification check.")
         
    print(f"[{datetime.now(eastern).isoformat()}] Notification check finished.")

def schedule_notification_checks(app):
    """Start a background thread to periodically check for notifications."""
    
    # Check if scheduler is already running (e.g., during reload)
    if hasattr(app, 'notification_scheduler_running') and app.notification_scheduler_running:
        print("Notification scheduler thread already seems to be running.")
        return
        
    def check_notifications_thread():
        check_count = 0
        # Prevent running immediately on start, wait a bit
        print("Notification thread started, initial wait...")
        time.sleep(60) # Wait 1 minute before first check
        
        while True:
            try:
                check_count += 1
                print(f"[{datetime.now().isoformat()}] Running notification check #{check_count}")
                
                # Run the actual check within the application context
                # The app object is passed to the scheduler function
                with app.app_context():
                    check_and_send_notifications()
                
                print(f"[{datetime.now().isoformat()}] Completed notification check #{check_count}")
                
            except Exception as e:
                print(f"Error in notification check thread loop: {e}")
                traceback.print_exc()
            
            # Check every 15 minutes (900 seconds)
            wait_time = 900 
            print(f"Next notification check scheduled in {wait_time / 60} minutes.")
            time.sleep(wait_time)
    
    # Create and start the thread
    thread = threading.Thread(target=check_notifications_thread, name="NotificationScheduler")
    thread.daemon = True # Allows app to exit even if thread is running
    thread.start()
    app.notification_scheduler_running = True # Flag that scheduler is running
    print("Started notification check background thread.") 
