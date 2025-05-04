# Loop News Analytics Module

## Overview

The analytics module tracks user interactions with the Loop News application, including:

- User sessions (login/logout)
- Brief interactions (creation, viewing, deletion, modification)
- Article interactions (clicks, view time)
- Search behaviors (queries, filters, result counts)

Data is stored in a SQLite database for simplicity and portability.

## Database Structure

The analytics database consists of four main tables:

1. **user_sessions**: Tracks user login/logout events with timing information.
2. **brief_interactions**: Records all interactions with briefs (saved search queries).
3. **article_interactions**: Captures when users click on or view articles.
4. **search_behaviors**: Records search queries and filters used.

## Integration Points

The analytics module is integrated with the application at these key points:

1. **Flask App Initialization**: The module is initialized in the app factory.
2. **Auth Routes**: Login/logout events are tracked in auth/routes.py.
3. **News Routes**: Brief interaction and search behaviors are tracked in news/routes.py.
4. **Client-side Tracking**: Article clicks and viewing time are tracked via JavaScript.

## JavaScript Analytics

The client-side analytics script (`analytics.js`) provides:

- Article click tracking by redirecting through a tracking endpoint
- Article view time tracking using page visibility events
- Integration with article links via data attributes

## Configuration

Analytics can be enabled/disabled via the `ANALYTICS_ENABLED` setting in config.py.

## Usage

No additional setup is required beyond the standard application initialization. The analytics module will automatically initialize when the Flask app starts, creating the SQLite database if it doesn't exist.

To manually initialize the database, run:

```
python -m app.analytics.init_db
``` 