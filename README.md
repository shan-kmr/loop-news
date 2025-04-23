# News Timeline

A web application that provides a timeline view of news articles, with smart topic grouping and AI-powered summaries.

![News Timeline Screenshot](https://via.placeholder.com/800x450.png?text=News+Timeline+Screenshot)

## Overview

News Timeline is a Flask-based web application that allows you to search for news articles and view them in a chronological timeline, organized by day and topic. The application features:

- Timeline view of news articles sorted by recency
- Topic-based grouping of related articles
- AI-powered daily summaries using OpenAI GPT-4.1 Nano (with Llama 3.2 as an optional alternative)
- History tab to view past searches
- Intelligent caching to reduce API calls and load times
- Google authentication for personalized user experience
- Automatic background refresh of outdated search results

## Installation

### Prerequisites

- Python 3.7+
- [Brave News API](https://brave.com/search/api/) key
- [OpenAI API key](https://platform.openai.com/) (default) or [Hugging Face](https://huggingface.co/) account with access to Llama models (optional)
- [Google Cloud Console](https://console.cloud.google.com/) project with OAuth credentials (for authentication)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/news-timeline.git
cd news-timeline
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
# Required
export BRAVE_API_KEY="your-brave-api-key"

# For Google authentication
export GOOGLE_CLIENT_ID="your-google-client-id"
export GOOGLE_CLIENT_SECRET="your-google-client-secret"
export GOOGLE_DISCOVERY_URL="https://accounts.google.com/.well-known/openid-configuration"

# For OpenAI summaries (default)
export OPENAI_API_KEY="your-openai-api-key"

# For Llama model (optional alternative)
export HF_API_TOKEN="your-huggingface-token"
```

4. Configure Google OAuth:
   - Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
   - Set up OAuth consent screen
   - Create OAuth credentials (Web application type)
   - Add authorized redirect URI: `http://localhost:5000/auth/callback`
   - Download credentials and set environment variables

## Usage

Run the application:

```bash
python news_timeline_app.py
```

Access the web interface at http://localhost:5000.

## Features in Detail

### Search Options

| Option | Description | Technical Implementation |
|--------|-------------|--------------------------|
| **Search Query** | Keywords to search for in news articles | Directly passed to Brave News API's `query` parameter |
| **Number of Results** | Maximum number of articles to fetch (10, 20, 30, 50) | Controls the `count` parameter in API calls |
| **Time Period** | Time range for articles (Past Day, Week, Month, Year) | Maps to Brave API's `freshness` parameter: pd (day), pw (week), pm (month), py (year) |
| **Topic Similarity** | Controls how articles are grouped | Adjusts the cosine similarity threshold: Low (0.2), Medium (0.3), High (0.4) |

### Topic Grouping System

Articles are grouped based on content similarity using TF-IDF vectorization and cosine similarity:

1. Each article's title and description are cleaned and vectorized
2. A cosine similarity matrix is calculated between all articles
3. Articles with similarity above the threshold are grouped together
4. Each group forms a "topic" with the newest article's title as the topic title
5. Topics are displayed with expandable/collapsible sections

The similarity threshold controls how aggressively articles are grouped:
- **Low (0.2)**: More groups with fewer articles per group
- **Medium (0.3)**: Balanced grouping
- **High (0.4)**: Fewer groups with more articles per group

### AI-Powered Summaries

Each day's articles are summarized using an AI model. By default, the application uses OpenAI's GPT-4.1 Nano, but it can also use Meta's Llama 3.2 model.

#### OpenAI GPT-4.1 Nano (Default)

- **Model**: gpt-4.1-nano
- **Prompt Template**:
  ```
  System: You are a helpful assistant that creates concise news summaries. 
  Summarize the key events and topics from these news articles in about 
  2-3 sentences without adding any new information. Focus on identifying 
  the main developments and common themes. Start directly with the summary 
  content - do not use phrases like 'Here is the summary' or 'The main news is'.
  
  User: Here are news articles about "{query}" from the same day:
  {article_texts}
  Please provide a brief summary of the main news for this day.
  ```
- **Generation Parameters**: temperature=0.3, top_p=0.9, max_tokens=150

#### Llama 3.2 (Optional Alternative)

- **Model**: meta-llama/Llama-3.2-3B-Instruct
- **Same prompt template as OpenAI**
- **Generation Parameters**: temperature=0.3, top_p=0.9, max_new_tokens=150

To switch between models, modify the `MODEL_PROVIDER` variable in the code:
```python
# Set to "llama" or "openai" to choose the AI model provider
MODEL_PROVIDER = "openai"  # Change to "llama" to use Llama model
```

These summaries appear at the top of each day's section, providing a quick overview.

### User Authentication with Google

The application features Google OAuth integration for user authentication:

1. Users can sign in with their Google account by clicking the "Login with Google" button
2. Authentication provides:
   - Personalized experience with user name and profile picture display
   - User-specific search history that persists across devices
   - Ability to delete individual search items or clear all history

When signed in:
- Search history is stored in user-specific files
- History management options become available
- Profile information from Google is displayed in the sidebar

### Automatic Background Refresh

The application includes an intelligent background refresh mechanism:

1. **Staleness Detection**:
   - When viewing search results older than 10 minutes, a refresh is automatically triggered
   - This occurs both in the main search page and when viewing history items

2. **Adaptive Time Window**:
   - The system determines an appropriate refresh window based on the original search parameters
   - Short time periods (past day) are upgraded to "past week" for better coverage
   - Longer periods (past month, past year) are maintained

3. **Gap Filling**:
   - If a search is more than a day old, the system intelligently fills in the gap with new articles
   - Articles from the refresh window are updated with fresh results
   - Articles older than the refresh window are preserved
   - Duplicates are eliminated by URL comparison

4. **Seamless Updates**:
   - Users don't need to manually refresh old searches
   - The timeline automatically includes the latest articles
   - The cache is updated with refreshed content
   - Original timestamps are preserved for articles outside the refresh window

This ensures you always see the most current information without missing developments that occurred between searches.

### Caching Mechanism

The application uses multi-level caching to improve performance:

1. **Search Results Caching**:
   - Search results are cached with a key of `{query}_{count}_{freshness}`
   - Default cache validity is 1 hour
   - Stored in a local JSON file (`search_history.json` for anonymous users or user-specific files for logged-in users)

2. **Model Caching**:
   - Llama model files are cached in a local directory (`llama_model_cache`)
   - This prevents redownloading the model on each run

3. **Summary Caching**:
   - Daily summaries are cached alongside search results
   - If a summary exists in cache, it's reused instead of regenerating

Users can force a refresh of search results with the "Refresh Results" button.

## Code Flow

Here's how the application works behind the scenes:

### Initial Page Load

1. Flask loads the main route (`/`)
2. Model initialization starts in a background thread (OpenAI client or Llama model)
3. If Brave API key is missing, an error is shown
4. Default query is set to "breaking news"
5. User authentication status is checked
6. Search history is loaded from the appropriate file based on login status
7. Templates are rendered with empty results

### Authentication Flow

1. User clicks "Login with Google"
2. Browser redirects to Google's OAuth consent screen
3. User grants permissions
4. Google redirects back to `/auth/callback` with authorization code
5. Application exchanges code for access token
6. User information is retrieved and stored in the session
7. User is redirected to the main page with personalized experience

### Search Execution

1. User submits a search form (POST request)
2. Application checks for cached results with the same parameters
3. If valid cache exists:
   - Age of the cached results is checked
   - If older than 10 minutes, a background refresh is triggered
   - Cached results and summaries are loaded
   - Articles are sorted by age
   - Topic grouping is applied using cached summaries
4. If no valid cache or "Refresh" is clicked:
   - Brave News API is called with the search parameters
   - Results are sorted by age
   - Topic grouping is applied
   - Summaries are generated for each day using the configured AI model
   - Results and summaries are cached

### Background Refresh Process

1. Age of cached results is checked
2. If older than 10 minutes:
   - Appropriate freshness parameter is determined based on original search
   - New API call is made with adjusted parameters
   - Results are categorized as "within refresh window" or "beyond refresh window"
   - Articles beyond the refresh window are preserved
   - Articles within the window are replaced with fresh content
   - Duplicates are eliminated by URL comparison
   - Updated results replace the old cache entry

### Topic Grouping Process

1. Article contents (title + description) are cleaned and vectorized using TF-IDF
2. Cosine similarity is calculated between all article pairs
3. Articles with similarity above threshold are grouped
4. Groups are sorted by the age of their newest article
5. Each day's articles are collected
6. For each day without a cached summary:
   - Articles for that day are compiled
   - The configured AI model generates a summary
   - Summary is stored in the cache
7. Topic groups are returned with their associated day summaries

### History Tab

1. User clicks "History" tab
2. App lists all past searches in the sidebar
3. When a history item is selected:
   - Cached search results are retrieved
   - If results are older than 10 minutes, a background refresh is triggered
   - Articles are sorted and grouped
   - If summaries don't exist in cache, they're generated and saved back
   - Results are displayed with the same UI as search results

## Directory Structure

```
news-timeline/
├── news_timeline_app.py    # Main application file
├── brave_news_api.py       # API client for Brave News
├── auth.py                 # Authentication handling
├── models.py               # User model and data storage
├── config.py               # Application configuration
├── requirements.txt        # Python dependencies
├── search_history.json     # Cache file for anonymous users
├── user_data/              # Directory for user-specific files
├── llama_model_cache/      # Directory for cached model files
└── templates/              # Auto-generated HTML templates
```

## Requirements

See `requirements.txt` for the full list of dependencies. Key requirements:

- flask
- flask-login
- authlib
- scikit-learn
- transformers (for Llama model)
- torch (for Llama model)
- huggingface_hub (for Llama model)
- openai (for GPT-4.1 Nano)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 