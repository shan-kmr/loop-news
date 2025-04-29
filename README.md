# Loop: Track All Moves in a Single Timeline

A web application that provides a timeline view of news articles, with smart topic grouping and AI-powered summaries.

Loom for how-to-use [here]([url](https://www.loom.com/share/61e00683f56240a0b7c39a933c2cb900?sid=74d80a52-bb0e-4be7-86da-917f5f97aecf)).

## Overview

Loop is a Flask-based web application that allows you to track topics through a chronological timeline, organized by day and topic. The application features:

- Timeline view of news articles sorted by recency
- Topic-based grouping of related articles with expandable/collapsible sections
- AI-powered hourly/daily summaries using OpenAI GPT-4.1 Nano (with Llama 3.2 as an optional alternative)
- User account system with 3-brief limit per user
- Light and dark mode with automatic theme detection
- Intelligent caching with 60-minute auto-refresh
- Google authentication for personalized user experience
- Clean, minimalist UI optimized for quick information consumption

## Installation

### Prerequisites

- Python 3.7+
- [Brave News API](https://brave.com/search/api/) key
- [OpenAI API key](https://platform.openai.com/) (default) or [Hugging Face](https://huggingface.co/) account with access to Llama models (optional)
- [Google Cloud Console](https://console.cloud.google.com/) project with OAuth credentials (for authentication)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/loop-timeline.git
cd loop-timeline
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

### User Authentication & Brief Limits

Loop requires user authentication for adding briefs:

1. **Login Requirement**: 
   - Users must login with Google to add briefs
   - Guest mode allows viewing but not adding briefs

2. **Brief Limit System**:
   - Each user is limited to 3 briefs maximum
   - A counter displays current usage (e.g., "2 out of 3 briefs used")
   - Users must delete an existing brief before adding a new one when at the limit
   - Delete buttons are available on each brief card for easy management

3. **Personalized Experience**:
   - User-specific search history that persists across devices
   - User name and profile picture display
   - Ability to delete individual briefs or clear all history

### Topic Tracking

Add topics to track by:

1. Clicking the "+" floating action button (when under the 3-brief limit)
2. Entering a topic in the "add a topic" modal
3. Viewing your brief on the home page with automatic summaries

Each brief shows:
- The topic title
- Auto-refresh countdown timer (60-minute cycles)
- "Today" summary with live indicator dot
- "Detailed analysis" link to the full timeline view

### Timeline View

When viewing a specific topic, you'll see:

1. **Chronological Organization**:
   - Articles grouped by day (Today, Yesterday, 2 days ago, etc.)
   - Daily AI-generated summaries at the top of each day section

2. **Topic Grouping**:
   - Related articles clustered by content similarity
   - Expandable/collapsible sections with rotation animation
   - Topic count showing number of articles in each group
   - Brief 1-liner summary of each article
   - Source information and full article links

3. **Visual Indicators**:
   - Live red dot next to "Today" to indicate fresh content
   - Auto-refresh countdown showing when new data will be loaded
   - Subtle animations for user interactions

### UI Features

Loop features a modern, clean UI with:

1. **Dual Theme Support**:
   - Light mode (default)
   - Dark mode with properly contrasted text and no shadows
   - Moon/Sun toggle in top navbar
   - Automatic theme detection based on system preferences
   - Theme persistence between sessions

2. **Responsive Design**:
   - Works on desktop and mobile devices
   - Adapts to different screen sizes
   - Optimized typography for readability

3. **Interactive Elements**:
   - Loading indicators during brief creation
   - Animated arrows for expandable content
   - Subtle hover effects
   - Delete buttons for brief management
   - Auto-refresh timer for each brief

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

### Automatic Refresh System

Loop includes a 60-minute auto-refresh cycle:

1. **Refresh Timer**:
   - Each brief displays a countdown timer showing minutes:seconds until refresh
   - Default refresh cycle is 60 minutes (up from previous 10 minutes)
   - Timer persists between page visits using localStorage

2. **Staleness Detection**:
   - When viewing search results older than 60 minutes, a refresh is automatically triggered
   - Background refresh occurs both on the home page and detailed views

3. **Adaptive Time Window**:
   - The system determines an appropriate refresh window based on original search parameters
   - Short time periods (past day) are used for 60-minute refreshes
   - Longer periods (past week, month, year) are maintained for less frequent refreshes

4. **Seamless Updates**:
   - Timeline automatically includes latest articles without manual intervention
   - Cache updates with refreshed content
   - Original timestamps are preserved for articles outside the refresh window

### Caching Mechanism

The application uses multi-level caching to improve performance:

1. **Search Results Caching**:
   - Search results are cached with a key of `{query}_{count}_{freshness}`
   - Default cache validity is 1 hour
   - Stored in a local JSON file (user-specific for logged-in users)

2. **Model Caching**:
   - Llama model files are cached in a local directory (`llama_model_cache`)
   - This prevents redownloading the model on each run

3. **Summary Caching**:
   - Daily summaries are cached alongside search results
   - If a summary exists in cache, it's reused instead of regenerating

Users can force a refresh of search results with the "Refresh" button in the detailed view.

## Technical Details

### Code Structure

The application is built around a single Flask app with the following components:

- **Authentication**: Google OAuth integration via `flask-login`
- **API Integration**: Brave News API client for search functionality
- **NLP Processing**: scikit-learn for TF-IDF vectorization and topic grouping
- **AI Integration**: OpenAI API and Hugging Face transformers for summaries
- **Frontend**: Pure HTML/CSS/JS without external frameworks
- **Data Storage**: JSON files for user data and history

### Deployment Considerations

For production deployment:

1. Set `debug=False` in app.run() call
2. Set up proper WSGI server (e.g., Gunicorn)
3. Configure proper authentication with HTTPS
4. Implement rate limiting for API calls
5. Use a database instead of JSON files for user data
6. Set appropriate cache durations based on expected usage
7. Consider using a CDN for static assets
8. Implement proper logging and monitoring
9. Add error tracking

## Acknowledgments

- [Brave Search API](https://brave.com/search/api/) for news search functionality
- [OpenAI API](https://platform.openai.com/) for AI-powered summaries
- [Meta AI](https://ai.meta.com/) for Llama model alternative
- [Google OAuth](https://developers.google.com/identity/protocols/oauth2) for authentication 
