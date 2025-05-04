/**
 * Client-side analytics tracking for Loop News
 */

// Track article clicks 
function trackArticleClick(articleUrl, briefQuery) {
    // Instead of directly navigating to the article URL, 
    // redirect through the tracking endpoint
    if (articleUrl && briefQuery) {
        return `/analytics/track/article_click?url=${encodeURIComponent(articleUrl)}&brief=${encodeURIComponent(briefQuery)}`;
    }
    return articleUrl;
}

// Track article view time
function trackArticleViewTime() {
    // Map to store start times for articles
    const articleViewTimes = new Map();
    
    // Start tracking view time for an article
    function startTracking(articleUrl, briefQuery) {
        if (!articleUrl || !briefQuery) return;
        
        const key = `${articleUrl}|${briefQuery}`;
        articleViewTimes.set(key, {
            url: articleUrl,
            brief: briefQuery,
            startTime: Date.now()
        });
    }
    
    // End tracking and send data
    function endTracking(articleUrl, briefQuery) {
        if (!articleUrl || !briefQuery) return;
        
        const key = `${articleUrl}|${briefQuery}`;
        const viewData = articleViewTimes.get(key);
        
        if (viewData) {
            const timeSpent = Math.floor((Date.now() - viewData.startTime) / 1000); // Convert to seconds
            
            // Only track if time spent is meaningful (> 2 seconds)
            if (timeSpent > 2) {
                // Send tracking data to the server
                fetch('/analytics/api/track/article_view', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        url: viewData.url,
                        brief: viewData.brief,
                        time_spent: timeSpent
                    })
                }).catch(error => {
                    console.error('Error tracking article view:', error);
                });
            }
            
            // Remove from tracking map
            articleViewTimes.delete(key);
        }
    }
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        // When page becomes hidden, end tracking for all articles
        if (document.visibilityState === 'hidden') {
            articleViewTimes.forEach((viewData, key) => {
                const [url, brief] = key.split('|');
                endTracking(url, brief);
            });
        }
    });
    
    // Handle before unload
    window.addEventListener('beforeunload', () => {
        articleViewTimes.forEach((viewData, key) => {
            const [url, brief] = key.split('|');
            endTracking(url, brief);
        });
    });
    
    return {
        startTracking,
        endTracking
    };
}

// Initialize tracking when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize article view time tracking
    const viewTracker = trackArticleViewTime();
    
    // Update all article links to go through the tracking endpoint
    document.querySelectorAll('a.article-link').forEach(link => {
        const articleUrl = link.getAttribute('href');
        const briefQuery = link.getAttribute('data-brief');
        
        if (articleUrl && briefQuery) {
            link.setAttribute('href', trackArticleClick(articleUrl, briefQuery));
            
            // Track when an article is viewed
            link.addEventListener('click', () => {
                viewTracker.startTracking(articleUrl, briefQuery);
            });
        }
    });
    
    // Add data-brief attribute to all news items when they're loaded
    const currentQuery = document.getElementById('current-query')?.value;
    if (currentQuery) {
        document.querySelectorAll('.news-item').forEach(item => {
            item.setAttribute('data-brief', currentQuery);
        });
    }
}); 