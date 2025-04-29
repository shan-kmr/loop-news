import json
from flask import current_app

# Add dependencies for LLM models
try:
    # OpenAI is the only supported model currently
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI package not available. To enable OpenAI summaries run: pip install openai")
    OPENAI_AVAILABLE = False

def init_openai(api_key):
    """Initialize the OpenAI client using the provided API key."""
    if not OPENAI_AVAILABLE:
        print("OpenAI package not installed, cannot initialize.")
        return False
    
    try:
        if not api_key:
            print("Error: No OPENAI_API_KEY provided to init_openai. OpenAI client not initialized.")
            return False
            
        openai.api_key = api_key
        print("OpenAI client initialized successfully")
        return True # Indicate success
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")
        return False


def summarize_daily_news(day_articles, query):
    """Generate a summary of articles for a specific day using OpenAI's API."""
    if not OPENAI_AVAILABLE:
        print("OpenAI not available, cannot generate summary.")
        return None
    
    if not day_articles:
        print("No articles provided for summary generation.")
        return None
        
    try:
        # content from articles being concatenated
        article_texts = []
        for article in day_articles:
            title = article.get('title', '')
            desc = article.get('description', '')
            source_netloc = 'Unknown source'
            if isinstance(article.get('meta_url'), dict):
                 source_netloc = article['meta_url'].get('netloc', 'Unknown source')
            
            article_texts.append(f"- {title}: {desc} (Source: {source_netloc})")
        
        # Limit input size to prevent token overflow
        # Consider smarter truncation or selection if needed
        all_articles_text = "\n".join(article_texts[:15]) 
        
        # Create messages for the OpenAI API
        system_prompt = ("""
        You are a helpful assistant that creates concise news summaries.
        Summarize the key events and topics from these news articles in about 2–3 sentences without adding any new information.
        Focus on identifying the main developments and common themes.
        Start directly with the summary content – do not use phrases like 'Here is the summary' or 'The main news is'.
        """)

        user_prompt = f"""\
        Here are news articles about "{query}" from the same day:
        {all_articles_text}
        Please provide a brief summary of the main news for this day.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        
        # Get model from config
        model = current_app.config.get("OPENAI_MODEL", "gpt-4.1-nano") # Provide a default
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=150,
            top_p=0.9
        )
        
        # Extract the summary from the response
        if response.choices and response.choices[0].message:
            summary = response.choices[0].message.content.strip()
            return summary
        else:
             print("OpenAI response format unexpected or empty.")
             return None

    except Exception as e:
        print(f"Error generating summary with OpenAI: {str(e)}")
        import traceback
        traceback.print_exc() # More detailed error for debugging
        return None

def group_articles_by_topic_openai(articles, query=""):
    """
    Group articles into topics based on content similarity using OpenAI.
    NOTE: This function should only be called with articles from a single day.
    
    Args:
        articles: List of news articles (all from the SAME DAY).
        query: The search query used (for contextual grouping).
        
    Returns:
        List of topic group dictionaries or None on failure.
    """
    if not OPENAI_AVAILABLE:
        print("OpenAI not available, cannot group topics.")
        return None
        
    if not articles:
        print("No articles provided for OpenAI topic grouping.")
        return [] # Return empty list for no articles
    
    # Import necessary utils locally within the function or ensure they are passed
    from ..utils.filters import day_group_filter
    from ..utils.text import extract_age_in_seconds
    
    try:
        # Prepare article data for OpenAI
        article_data = []
        common_day = "Unknown" # Determine common day
        if articles:
             common_day = day_group_filter(articles[0])
             
        for idx, article in enumerate(articles):
             # Verify all articles are from the same day (optional strict check)
             article_day = day_group_filter(article)
             if article_day != common_day:
                 print(f"WARNING: group_articles_by_topic_openai received articles from different days ({article_day} vs {common_day}). Grouping might be inaccurate.")
                 # Decide how to handle: skip article, raise error, or proceed cautiously.
                 # For now, proceed but log warning.

             title = article.get('title', '')
             desc = article.get('description', '')
             source_netloc = 'Unknown source'
             if isinstance(article.get('meta_url'), dict):
                 source_netloc = article['meta_url'].get('netloc', 'Unknown source')
             age = article.get('age', 'Unknown')
             
             article_data.append({
                 "id": idx,
                 "title": title, 
                 "description": desc,
                 "source": source_netloc,
                 "age": age
             })
        
        # Limit input size
        if len(json.dumps(article_data)) > 30000: # Estimate token count roughly by length
             print("Warning: Article data potentially too large for OpenAI grouping. Truncating.")
             # Simple truncation, smarter methods might be needed
             article_data = article_data[:len(articles)//2] 
             
        # Create the prompt for OpenAI
        messages = [
            {
                "role": "system", 
                "content": (
                    """You are an expert at organizing news articles into coherent topic groups. 
                    Your task is to:
                    1. Group similar articles into distinct topics.
                    2. Give each topic a concise, descriptive title (max 10 words).
                    3. Write a brief 1-sentence summary for each topic group.
                    4. Place each article in exactly one group.
                    5. Create more restrictive themes rather than broad categories."""
                )
            },
            {
                "role": "user", 
                "content": 
                    f"""Here are news articles about "{query}".
                    Please group them into topics, with a concise title and brief summary for each group.
                    Articles: {json.dumps(article_data, indent=2)}
                    Return JSON in this exact format:
                    {{
                    "topic_groups": [
                        {{
                        "title": "Concise topic title",
                        "summary": "Brief one-sentence summary of what this topic group is about",
                        "article_ids": [0, 3, 5]
                        }},
                        ...
                    ]
                    }}
                    """
            }
        ]
        
        # Use a capable model for JSON output and complex instruction following
        model = "gpt-4o-mini" # Or another suitable model like gpt-4-turbo
        if "nano" in current_app.config.get("OPENAI_MODEL", ""): 
            print("Warning: nano model might struggle with complex JSON grouping. Consider gpt-4o-mini.")
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=model, 
            messages=messages,
            temperature=0.2,
            max_tokens=1500, # Increase max_tokens for potentially large responses
            response_format={"type": "json_object"} # Request JSON output
        )
        
        # Extract the response content
        if response.choices and response.choices[0].message and response.choices[0].message.content:
             result_str = response.choices[0].message.content
             result = json.loads(result_str)
        else:
             print("OpenAI grouping response unexpected or empty.")
             return None
        
        # Process the groups
        processed_topic_groups = []
        all_grouped_indices = set()
        
        for group in result.get("topic_groups", []):
            article_ids = group.get("article_ids", [])
            if not article_ids or not isinstance(article_ids, list):
                continue
                
            group_articles = []
            valid_ids = []
            for idx in article_ids:
                 if isinstance(idx, int) and 0 <= idx < len(articles):
                     # Check if article already grouped (handle potential overlaps from LLM)
                     if idx not in all_grouped_indices:
                         group_articles.append(articles[idx])
                         valid_ids.append(idx)
                         all_grouped_indices.add(idx)
                     else:
                         print(f"Warning: Article index {idx} already assigned to another group by OpenAI. Skipping.")
                 else:
                     print(f"Warning: Invalid article index {idx} received from OpenAI.")
                         
            if not group_articles:
                continue
                
            # Sort the articles within the group by age
            group_articles = sorted(group_articles, key=extract_age_in_seconds)
            
            # Create the topic group entry
            newest_article = group_articles[0]
            processed_topic_groups.append({
                'title': group.get("title", "Untitled Topic"),
                'summary': group.get("summary", ""), # Topic-specific summary from LLM
                'articles': group_articles,
                'count': len(group_articles),
                'newest_age': newest_article.get('age', 'Unknown'),
                'day_group': common_day  # Ensure day boundary is set
            })
        
        # Handle ungrouped articles (optional: create individual topics for them)
        ungrouped_articles = []
        for idx, article in enumerate(articles):
             if idx not in all_grouped_indices:
                 ungrouped_articles.append(article)
                 print(f"Article index {idx} was not grouped by OpenAI.")
                 # Add as individual topic
                 processed_topic_groups.append({
                    'title': article.get('title', 'Untitled Topic'),
                    'summary': '', # No LLM summary for this
                    'articles': [article],
                    'count': 1,
                    'newest_age': article.get('age', 'Unknown'),
                    'day_group': common_day
                })
                 
        # Sort final topic groups by the age of their newest article
        processed_topic_groups = sorted(processed_topic_groups, key=lambda x: extract_age_in_seconds(x['articles'][0]))
        
        return processed_topic_groups
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from OpenAI grouping: {e}")
        print(f"Received: {result_str}") # Log the raw response
        return None # Indicate failure
    except Exception as e:
        print(f"Error grouping articles with OpenAI: {str(e)}")
        import traceback
        traceback.print_exc()
        return None # Indicate failure 