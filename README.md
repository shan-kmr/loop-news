# News Timeline

A web application that provides a timeline view of news articles, with smart topic grouping and AI-powered summaries.

![News Timeline Screenshot](https://via.placeholder.com/800x450.png?text=News+Timeline+Screenshot)

## Overview

News Timeline is a Flask-based web application that allows you to search for news articles and view them in a chronological timeline, organized by day and topic. The application features:

- Timeline view of news articles sorted by recency
- Topic-based grouping of related articles
- AI-powered daily summaries using Meta's Llama 3.2 model
- History tab to view past searches
- Intelligent caching to reduce API calls and load times

## Installation

### Prerequisites

- Python 3.7+
- [Brave News API](https://brave.com/search/api/) key
- [Hugging Face](https://huggingface.co/) account with access to Llama models

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
export BRAVE_API_KEY="your-brave-api-key"
```

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

### AI Summaries with Llama 3.2

Each day's articles are summarized using Meta's Llama 3.2 model:

- **Model**: meta-llama/Llama-3.2-3B-Instruct
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
- **Generation Parameters**: temperature=0.3, top_p=0.9, max_new_tokens=150

These summaries appear at the top of each day's section, providing a quick overview.

### Caching Mechanism

The application uses multi-level caching to improve performance:

1. **Search Results Caching**:
   - Search results are cached with a key of `{query}_{count}_{freshness}`
   - Default cache validity is 1 hour
   - Stored in a local JSON file (`search_history.json`)

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
2. Llama model initialization starts in a background thread
3. If Brave API key is missing, an error is shown
4. Default query is set to "breaking news"
5. Search history is loaded from the local JSON file
6. Templates are rendered with empty results

### Search Execution

1. User submits a search form (POST request)
2. Application checks for cached results with the same parameters
3. If valid cache exists:
   - Cached results and summaries are loaded
   - Articles are sorted by age
   - Topic grouping is applied using cached summaries
4. If no valid cache or "Refresh" is clicked:
   - Brave News API is called with the search parameters
   - Results are sorted by age
   - Topic grouping is applied
   - Summaries are generated for each day
   - Results and summaries are cached

### Topic Grouping Process

1. Article contents (title + description) are cleaned and vectorized using TF-IDF
2. Cosine similarity is calculated between all article pairs
3. Articles with similarity above threshold are grouped
4. Groups are sorted by the age of their newest article
5. Each day's articles are collected
6. For each day without a cached summary:
   - Articles for that day are compiled
   - Llama model generates a summary
   - Summary is stored in the cache
7. Topic groups are returned with their associated day summaries

### History Tab

1. User clicks "History" tab
2. App lists all past searches in the sidebar
3. When a history item is selected:
   - Cached search results are retrieved
   - Articles are sorted and grouped
   - If summaries don't exist in cache, they're generated and saved back
   - Results are displayed with the same UI as search results

## Directory Structure

```
news-timeline/
├── news_timeline_app.py    # Main application file
├── requirements.txt        # Python dependencies
├── search_history.json     # Cache file for search results
├── llama_model_cache/      # Directory for cached model files
└── templates/              # Auto-generated HTML templates
```

## Requirements

See `requirements.txt` for the full list of dependencies. Key requirements:

- flask
- scikit-learn
- transformers
- torch
- huggingface_hub

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 