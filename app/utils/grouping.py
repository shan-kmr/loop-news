from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import clean_text, extract_age_in_seconds, is_valid_summary
from .filters import day_group_filter

from ..services.openai_service import group_articles_by_topic_openai, summarize_daily_news


# Assume history functions are available via direct import
from .history import load_search_history, save_search_history


def group_articles_by_topic(articles, similarity_threshold=0.3, query="", day_summaries=None):
    """
    Group articles into topics based on content similarity.
    CRITICAL: Enforce day boundaries - never mix articles from different days in the same topic.
    
    Args:
        articles: List of news articles.
        similarity_threshold: Threshold for considering articles as similar (0-1).
        query: The search query used (for summarization).
        day_summaries: Dictionary of existing summaries by day.
        
    Returns:
        List of topic groups, each containing related articles.
    """
    if not articles:
        return []
    
    # Initialize day_summaries if not provided
    if day_summaries is None:
        day_summaries = {}
    
    # CRITICAL CHANGE: First, organize articles by day to enforce day boundaries
    day_grouped_articles = {}
    for article in articles:
        day = day_group_filter(article)
        if day not in day_grouped_articles:
            day_grouped_articles[day] = []
        day_grouped_articles[day].append(article)
    
    # Sort days in order of recency (Today, Yesterday, etc.)
    day_priority = {
        "Today": 1,
        "Yesterday": 2,
        "2 days ago": 3,
        "3 days ago": 4,
        "4 days ago": 5,
        "5 days ago": 6,
        "6 days ago": 7,
        "1 week ago": 8,
        "2 weeks ago": 9,
        "1 month ago": 10
    }
    sorted_days = sorted(day_grouped_articles.keys(), key=lambda d: day_priority.get(d, 999))
    
    # Print day distribution for debugging
    print(f"Articles by day before grouping:")
    for day in sorted_days:
        print(f"  - {day}: {len(day_grouped_articles[day])} articles")
    
    # Initialize final topic groups list - will contain topics from all days
    final_topic_groups = []
    
    # Try to use OpenAI for more intelligent grouping if available
    # But still enforce day boundaries by processing each day separately
    try:
        print("Attempting OpenAI for topic grouping, enforcing day boundaries...")
        
        # Process each day separately with OpenAI to maintain day boundaries
        for day in sorted_days:
            day_articles = day_grouped_articles[day]
            if not day_articles:
                continue
                
            print(f"Processing {len(day_articles)} articles for day: {day} with OpenAI")
            
            # Use OpenAI to group just this day's articles
            day_openai_groups = group_articles_by_topic_openai(day_articles, query)
            
            if day_openai_groups:
                # Set the day group explicitly for all topics from this day
                for topic in day_openai_groups:
                    topic['day_group'] = day
                    
                    # Add the day's summary to all topics from this day
                    existing_summary = day_summaries.get(day)
                    topic['day_summary'] = existing_summary
                
                # Add this day's topics to our final list
                final_topic_groups.extend(day_openai_groups)
                print(f"Added {len(day_openai_groups)} OpenAI-grouped topics for {day}")
            else:
                # Fallback to simple grouping if OpenAI failed for this day
                print(f"OpenAI grouping failed for {day}, falling back to simple grouping")
                
                # Create a single topic for each article as a fallback
                for article in day_articles:
                    final_topic_groups.append({
                        'title': article.get('title', 'Untitled Topic'),
                        'articles': [article],
                        'count': 1,
                        'newest_age': article.get('age', 'Unknown'),
                        'day_group': day,
                        'day_summary': day_summaries.get(day)
                    })
        
        # If we successfully created any topics with OpenAI, process summaries and return
        if final_topic_groups:
            # Process day summaries - generate any that are missing
            days_needing_summaries = set()
            for topic in final_topic_groups:
                day = topic['day_group']
                if not is_valid_summary(topic.get('day_summary')):
                    if day not in days_needing_summaries:
                        days_needing_summaries.add(day)
            
            if days_needing_summaries:
                 # Check OpenAI availability again before generating summaries
                 # This check should ideally happen once globally
                 try:
                     import openai
                     OPENAI_AVAILABLE_CHECK = True
                 except ImportError:
                     OPENAI_AVAILABLE_CHECK = False
                     
                 if OPENAI_AVAILABLE_CHECK:
                     for day in days_needing_summaries:
                         if day in day_summaries and is_valid_summary(day_summaries[day]):
                             print(f"Skipping summary generation for day: {day} (already valid)")
                             continue
                             
                         print(f"Generating new summary for day: {day} using OpenAI")
                         day_articles = day_grouped_articles.get(day, [])
                         if not day_articles:
                             continue
                             
                         summary = summarize_daily_news(day_articles, query)
                         
                         if is_valid_summary(summary):
                             print(f"Successfully generated summary for {day}: {summary[:50]}...")
                             day_summaries[day] = summary
                             
                             # Update all topics from this day with the new summary
                             for topic in final_topic_groups:
                                 if topic['day_group'] == day:
                                     topic['day_summary'] = summary
                             
                             try:
                                 # Save summaries to history
                                 history = load_search_history() # Need user context or path
                                 for key, entry in history.items():
                                     if entry.get('query') == query:
                                         if 'day_summaries' not in entry or entry['day_summaries'] is None:
                                             entry['day_summaries'] = {}
                                         entry['day_summaries'][day] = summary
                                         save_search_history(history) # Need user context or path
                                         print(f"Saved new {day} summary to history for {query}")
                                         break
                             except Exception as e:
                                 print(f"Error saving summary to history: {e}")
            
            topics_with_summaries = sum(1 for topic in final_topic_groups if is_valid_summary(topic.get('day_summary')))
            print(f"Created {len(final_topic_groups)} total topics across all days (OpenAI attempt), {topics_with_summaries} have summaries.")
            
            # Return the final list with all days' topics
            return final_topic_groups

    except ImportError:
        print("OpenAI not available, falling back to TF-IDF grouping.")
    except Exception as e:
        print(f"Error during OpenAI grouping attempt: {e}. Falling back to TF-IDF.")

    # --- Fallback to TF-IDF based approach --- 
    print("Using TF-IDF for topic grouping, enforcing day boundaries...")
    final_topic_groups = [] # Reset if OpenAI failed
    
    # Process each day separately with TF-IDF to maintain day boundaries
    for day in sorted_days:
        day_articles = day_grouped_articles[day]
        if not day_articles:
            continue
            
        print(f"Processing {len(day_articles)} articles for day: {day} with TF-IDF")
        
        # Extract title and description for content comparison - only for this day's articles
        article_contents = []
        for article in day_articles:
            title = article.get('title', '')
            desc = article.get('description', '')
            content = f"{title} {desc}"
            article_contents.append(clean_text(content))
        
        # Compute TF-IDF vectorization for this day's articles
        try:
            day_topic_groups = []
            
            if len(day_articles) > 1:
                vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(article_contents)
                
                # Compute cosine similarity
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Group articles based on similarity - only within this day
                visited = [False] * len(day_articles)
                
                for i in range(len(day_articles)):
                    if visited[i]:
                        continue
                        
                    visited[i] = True
                    group = [day_articles[i]]
                    
                    # Find similar articles - only looking at this day's articles
                    for j in range(i+1, len(day_articles)):
                        if not visited[j] and similarity_matrix[i, j] >= similarity_threshold:
                            group.append(day_articles[j])
                            visited[j] = True
                    
                    # Create a topic for this group - explicitly setting the day
                    group = sorted(group, key=extract_age_in_seconds)
                    newest_article = group[0]
                    topic_title = newest_article.get('title', 'Untitled Topic')
                    
                    day_topic_groups.append({
                        'title': topic_title,
                        'articles': group,
                        'count': len(group),
                        'newest_age': newest_article.get('age', 'Unknown'),
                        'day_group': day  # Explicitly set the day to maintain boundaries
                    })
            else:
                # For a single article, just add it as its own topic
                if day_articles:
                   article = day_articles[0]
                   day_topic_groups.append({
                        'title': article.get('title', 'Untitled Topic'),
                        'articles': [article],
                        'count': 1,
                        'newest_age': article.get('age', 'Unknown'),
                        'day_group': day  # Explicitly set the day
                    })
            
            # Add existing summary for this day to all topics from this day
            existing_summary = day_summaries.get(day)
            for topic in day_topic_groups:
                topic['day_summary'] = existing_summary
            
            # Add this day's topics to our final list
            final_topic_groups.extend(day_topic_groups)
            print(f"Added {len(day_topic_groups)} TF-IDF grouped topics for {day}")
            
        except Exception as e:
            print(f"Error during TF-IDF grouping for day {day}: {str(e)}")
            # Fallback to simple grouping without TF-IDF if TF-IDF fails
            for article in day_articles:
                final_topic_groups.append({
                    'title': article.get('title', 'Untitled Topic'),
                    'articles': [article],
                    'count': 1,
                    'newest_age': article.get('age', 'Unknown'),
                    'day_group': day,
                    'day_summary': day_summaries.get(day)
                })
    
    # Process day summaries - generate any that are missing (TF-IDF path)
    days_needing_summaries = set()
    for topic in final_topic_groups:
        day = topic['day_group']
        if not is_valid_summary(topic.get('day_summary')):
            if day not in days_needing_summaries:
                days_needing_summaries.add(day)
    
    if days_needing_summaries:
         # Check OpenAI availability again before generating summaries
         try:
             import openai
             OPENAI_AVAILABLE_CHECK = True
         except ImportError:
             OPENAI_AVAILABLE_CHECK = False
             
         if OPENAI_AVAILABLE_CHECK:
             for day in days_needing_summaries:
                 if day in day_summaries and is_valid_summary(day_summaries[day]):
                     print(f"Skipping summary generation for day: {day} (already valid)")
                     continue
                     
                 print(f"Generating new summary for day: {day} using OpenAI (TF-IDF path)")
                 day_articles = day_grouped_articles.get(day, [])
                 if not day_articles:
                     continue
                     
                 summary = summarize_daily_news(day_articles, query)
                 
                 if is_valid_summary(summary):
                     print(f"Successfully generated summary for {day}: {summary[:50]}...")
                     day_summaries[day] = summary
                     
                     # Update all topics from this day with the new summary
                     for topic in final_topic_groups:
                         if topic['day_group'] == day:
                             topic['day_summary'] = summary
                     
                     try:
                         # Save summaries to history
                         history = load_search_history() # Need user context or path
                         for key, entry in history.items():
                             if entry.get('query') == query:
                                 if 'day_summaries' not in entry or entry['day_summaries'] is None:
                                     entry['day_summaries'] = {}
                                 entry['day_summaries'][day] = summary
                                 save_search_history(history) # Need user context or path
                                 print(f"Saved new {day} summary to history for {query} (TF-IDF path)")
                                 break
                     except Exception as e:
                         print(f"Error saving summary to history: {e}")

    topics_with_summaries = sum(1 for topic in final_topic_groups if is_valid_summary(topic.get('day_summary')))
    print(f"Created {len(final_topic_groups)} total topics across all days (TF-IDF path), {topics_with_summaries} have summaries.")
    
    return final_topic_groups 