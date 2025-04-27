# Loop News – Project Synopsis

## Project Overview  
Loop News is a Flask-based web application that lets users **“stay in the loop”** on any topic by:
* pulling fresh news from multiple public sources (Brave News Search, Reddit, Twitter, Perplexity, etc.),
* grouping related articles into a chronological timeline,
* generating concise AI summaries for each day or topic, and
* persisting personal search history so users can revisit or share timelines later.

The system was packaged into a single repository and delivered to you as a large Markdown file for analysis.

## Project Goal  
Provide an opinionated, DRY timeline of events so users can absorb the evolution of a story quickly without wading through duplicate headlines.  
The end-product must respect copyright by always linking back to original sources and keeping summaries brief.

---

## Core Functionalities (files & key routines)

### `loop-news/news_timeline_app.py`
* **`load_user()`** – Flask-Login helper to retrieve a user from storage.  
* **`init_models()`** – Chooses between OpenAI or local Llama summariser depending on `MODEL_PROVIDER`.  
* **`init_llama_model()` / `init_openai()`** – Lazy-load the appropriate LLM (with Hugging Face caching for Llama).  
* **`summarize_daily_news_openai()` / `summarize_daily_news_llama()`** – Produce one-paragraph daily digests.  
* **`group_articles_by_topic_openai()` / fallback `group_articles_by_topic()`** – Cluster raw articles into coherent events.  
* **`update_current_day_results()`** – End-to-end pipeline: fetch, group, summarise, cache to history.  
* **Search-history endpoints** – `index()`, `history()`, `history_item()`, `delete_history_item()`, `clear_history()` expose CRUD routes for past queries.  
* **Utility helpers** – `clean_text()`, `is_valid_summary()`, time/formatting filters, and background notification scheduler.

### `loop-news/brave_news_api.py`
* **`BraveNewsAPI` (in `main()` factory)** – Thin wrapper around Brave Search News endpoint; handles paging, deduplication, and returns normalized JSON used by the timeline grouper.

### `loop-news/reddit_api.py`
* **`RedditAPI` class**  
  * uses **PRAW** internally, supports OAuth credentials, and exposes  
    * `search_submissions()` – keyword-based pull,  
    * `fetch_hot_posts()` – topical discovery,  
    * utility to strip tracking params and convert Reddit timestamps to UTC strings.

### `loop-news/twitter_api.py`
* **`TwitterAPI` class** (if keys are provided) – fetches recent tweets, rate-limits on bearer-token windows, normalises author & media URLs.

### `loop-news/models/*`
* Embedding helpers for semantic similarity (Torch + Sentence-Transformers).
* Local cache directory management for downloaded HF weights.

### `loop-news/auth/*.py`
* Blueprint implementing Google OAuth2 login with **Authlib**.
* Session handling via **Flask-Login**.

### `loop-news/config.py`
* Centralised constants: API keys, OpenAI model name, **`MODEL_PROVIDER` switch**, cache TTL, file paths.
* Defaults to `MODEL_PROVIDER = "openai"`; Llama path is inert unless explicitly flipped.

### Static & templates
* Jinja templates for timeline view, search bar, and history list.
* CSS/JS assets kept minimal; Bootstrap CDN only.

### Storage System  
All user data lives in plain JSON files under a single `user_data/` folder—no database required. When you search a topic, the app reads and writes a file called:  
```
user_data/<safe_email>/search_history.json
```  
That one JSON holds every search query, its raw articles, per-day summaries, topic clusters, timestamps, and any notification settings.

#### Google Authentication  
Instead of building a username/password system, the app uses **“Log in with Google”** via OAuth2. You click the Google button, grant basic profile access, and Google returns a unique user ID (the “sub”), your name, email, and avatar URL. The app then:  
1. Creates or updates `user_data/<sub>.json` with your profile.  
2. Stores your `<sub>` in a secure session cookie so every request knows exactly who you are.

#### JSON Data Files  
1. **`<sub>.json`** (e.g. `1234567890.json`)  
   - Keys: `id`, `name`, `email`, `profile_pic`, `updated_at`.  
   - Used exclusively by Flask-Login to map your session to your profile.  
2. **`search_history.json`** inside `user_data/<safe_email>/`  
   - Keys are your queries (with parameters).  
   - Values include:  
     - `results` (raw API articles grouped by date)  
     - `day_summaries` (AI-generated text per day)  
     - `topic_groups` (event clusters)  
     - `timestamp` & `search_time` (for sorting)  
     - Optional `notifications` settings  