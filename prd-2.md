# Loop News - Application Documentation

## Project Overview

Loop News is a Flask-based web application designed to provide users with personalized news aggregation and summarization. It fetches news articles from external sources based on user-defined topics (briefs), uses AI to generate concise daily summaries and group related articles, and allows users to save their briefs and receive email notifications for updates.

## Project Goal

Help users stay informed about specific topics efficiently by providing curated, summarized news updates in a timeline format. It aims to reduce information overload by grouping similar stories with a timeline view and offering periodic email digests, allowing users to quickly grasp the latest developments on subjects they care about.

## Core Functionalities

*   **News Fetching:** Retrieves news articles from external APIs (Brave Search, Reddit) based on user queries, freshness, and count parameters.
*   **AI Summarization:** Utilizes OpenAI models (e.g., GPT-4 Turbo) to generate concise daily summaries for each saved topic.
*   **AI Topic Grouping:** Employs OpenAI models to analyze articles within a day and group them into distinct, meaningful topic clusters with titles and summaries.
*   **User Authentication:** Handles user login via Google OAuth, ensuring secure access.
*   **Access Control:** Manages user access through a whitelist system (`allowed_users.json`) and a waitlist (`waitlist.json`) for users requesting access.
*   **Brief Management:** Allows authenticated users to create, view, and delete saved search queries (briefs).
*   **History Viewing:** Displays past search results, including fetched articles, generated summaries, and topic groups for saved briefs.
*   **Email Notifications:** Sends periodic (hourly or daily) email updates to users for their subscribed briefs, summarizing new developments.
*   **Caching:** Implements basic caching of search results to reduce redundant API calls.

## File Structure & Overview

```
app/
|-- __init__.py             # Initializes the Flask application (App Factory), configures extensions (LoginManager, OAuth), registers blueprints, Jinja filters, and starts background tasks.
|-- models.py               # Defines data models, primarily the User model for Flask-Login, including methods for loading/saving user data from/to JSON files.
|
|-- auth/                   # Handles authentication and authorization.
|   |-- __init__.py         # (Empty - Standard Python package marker)
|   |-- oauth.py            # Initializes the Authlib OAuth client, specifically for Google.
|   |-- routes.py           # Defines Flask routes for login (/login), OAuth callback (/callback), and logout (/logout). Handles user session management.
|   |-- whitelist.py        # Manages the email whitelist and waitlist using JSON files (allowed_users.json, waitlist.json).
|
|-- news/                   # Handles core news browsing and brief management features.
|   |-- __init__.py         # (Empty - Standard Python package marker)
|   |-- routes.py           # Defines main application routes: home page (/), new brief submission, history viewing (/history/<query>), API endpoints for deleting/clearing history, managing notifications.
|
|-- services/               # Contains modules for interacting with external APIs and services.
|   |-- __init__.py         # (Empty - Standard Python package marker)
|   |-- brave.py            # Service providing access to the BraveNewsAPI client (defined in sources/).
|   |-- notifications.py    # Manages sending email notifications (hourly/daily) based on user subscriptions and schedules checks.
|   |-- openai_service.py   # Handles interactions with the OpenAI API for summarization and topic grouping.
|   |-- reddit.py           # Service providing access to the RedditAPI client (defined in sources/).
|
|-- sources/                # Contains the API client implementations for external data sources.
|   |-- __init__.py         # (Empty - Standard Python package marker)
|   |-- brave_news_api.py   # Implements the client for the Brave Search News API.
|   |-- reddit_api.py       # Implements the client for the Reddit API using PRAW.
|   |-- twitter_api.py      # Implements the client for the Twitter API (currently unused by the app).
|
|-- utils/                  # Contains utility functions used across the application.
|   |-- __init__.py         # (Empty - Standard Python package marker)
|   |-- filters.py          # Defines custom Jinja2 filters for template rendering (e.g., date formatting, grouping labels).
|   |-- grouping.py         # Implements logic for grouping articles into topics, likely using NLP techniques and/or OpenAI.
|   |-- history.py          # Manages loading, saving, and updating user search history (briefs, results, summaries) stored in JSON files.
|   |-- text.py             # Contains text processing utilities (e.g., extracting timestamps, validating summaries).

(Note: `__pycache__` directories are omitted as they contain compiled Python bytecode.)
```

`config.py`
* Centralised constants: API keys, OpenAI model name, **`MODEL_PROVIDER` switch**, cache TTL, file paths.
* Defaults to `MODEL_PROVIDER = "openai"`; Llama path is inert unless explicitly flipped.

`run.py`
* Sets up all the environment variables, ensures they're there and then runs the flask app.


### Static HTML Template Files
* All HTML is rendered using external Jinja templates located in the `templates/` directory (`index.html`, `error.html`, `notification_email.html`).
* Templates handle the main UI, timeline view, error pages, and email notifications.
* Flask's `render_template` function is used exclusively for rendering.
* CSS/JS assets kept minimal; Bootstrap CDN only.


## Storage System

The application primarily relies on the local filesystem using **JSON files** for data persistence, located within the `user_data/` directory (relative to the project root).

*   **User Profiles:** Basic user information (ID/email, name, profile picture) obtained from Google OAuth is stored in `user_data/<user_email_safe>/user_profile.json`.
*   **Search History (Briefs):** Saved briefs, including the query parameters, fetched articles, generated daily summaries, and topic groups, are stored in `user_data/<user_email_safe>/search_history.json` for logged-in users or `user_data/shared/search_history.json` for potentially shared/anonymous data or background tasks.
*   **Access Control Lists:**
    *   `user_data/allowed_users.json`: A list of email addresses permitted to use the application.
    *   `user_data/waitlist.json`: A list of email addresses requesting access.
*   **Notification Settings:** User preferences for email notifications (frequency) are embedded within the corresponding topic entry in their `search_history.json` file.

This file-based storage is suitable for single-instance deployments or development environments but may pose challenges for scalability and concurrent access in larger production scenarios where a database would be more appropriate.
