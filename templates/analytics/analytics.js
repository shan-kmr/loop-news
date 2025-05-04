/**
 * Client-side analytics tracking for Loop News
 */

// Enable debugging
const DEBUG = true;

// Helper function for logging
function debugLog(...args) {
    if (DEBUG) {
        console.log('[Analytics]', ...args);
    }
}

// Track article clicks 
function trackArticleClick(articleUrl, briefQuery) {
    // Instead of directly navigating to the article URL, 
    // redirect through the tracking endpoint
    if (articleUrl && briefQuery) {
        const trackingUrl = `/analytics/track/article_click?url=${encodeURIComponent(articleUrl)}&brief=${encodeURIComponent(briefQuery)}`;
        debugLog('Tracking click:', { article: articleUrl, brief: briefQuery, trackingUrl });
        return trackingUrl;
    }
    debugLog('Not tracking click - missing data:', { article: articleUrl, brief: briefQuery });
    return articleUrl;
}

// Track article view time
function trackArticleViewTime() {
    // Map to store start times for articles
    const articleViewTimes = new Map();
    
    // Start tracking view time for an article
    function startTracking(articleUrl, briefQuery) {
        if (!articleUrl || !briefQuery) {
            debugLog('Cannot start tracking - missing data:', { article: articleUrl, brief: briefQuery });
            return;
        }
        
        const key = `${articleUrl}|${briefQuery}`;
        articleViewTimes.set(key, {
            url: articleUrl,
            brief: briefQuery,
            startTime: Date.now()
        });
        debugLog('Started tracking view time for:', { article: articleUrl, brief: briefQuery });
    }
    
    // End tracking and send data
    function endTracking(articleUrl, briefQuery) {
        if (!articleUrl || !briefQuery) {
            debugLog('Cannot end tracking - missing data:', { article: articleUrl, brief: briefQuery });
            return;
        }
        
        const key = `${articleUrl}|${briefQuery}`;
        const viewData = articleViewTimes.get(key);
        
        if (viewData) {
            const timeSpent = Math.floor((Date.now() - viewData.startTime) / 1000); // Convert to seconds
            
            // Only track if time spent is meaningful (> 2 seconds)
            if (timeSpent > 2) {
                debugLog('Sending tracking data:', { article: articleUrl, brief: briefQuery, timeSpent });
                
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
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    debugLog('Tracking response:', data);
                })
                .catch(error => {
                    console.error('Error tracking article view:', error);
                });
            } else {
                debugLog('View time too short, not tracking:', { article: articleUrl, timeSpent });
            }
            
            // Remove from tracking map
            articleViewTimes.delete(key);
        } else {
            debugLog('No view data found for:', { article: articleUrl, brief: briefQuery });
        }
    }
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        // When page becomes hidden, end tracking for all articles
        if (document.visibilityState === 'hidden') {
            debugLog('Page hidden, ending all tracking');
            articleViewTimes.forEach((viewData, key) => {
                const [url, brief] = key.split('|');
                endTracking(url, brief);
            });
        }
    });
    
    // Handle before unload
    window.addEventListener('beforeunload', () => {
        debugLog('Page unloading, ending all tracking');
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

// Check analytics health
function checkAnalyticsHealth() {
    // Only check if in debug mode
    if (!DEBUG) return;
    
    fetch('/analytics/debug')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('[Analytics] Health check:', data);
        })
        .catch(error => {
            console.error('[Analytics] Health check failed:', error);
        });
}

// Initialize tracking when document is ready
document.addEventListener('DOMContentLoaded', () => {
    debugLog('Initializing analytics tracking');
    
    // Initialize article view time tracking
    const viewTracker = trackArticleViewTime();
    
    // Update all article links to go through the tracking endpoint
    const articleLinks = document.querySelectorAll('a.article-link');
    debugLog(`Found ${articleLinks.length} article links`);
    
    articleLinks.forEach(link => {
        const articleUrl = link.getAttribute('href');
        const briefQuery = link.getAttribute('data-brief');
        
        if (articleUrl && briefQuery) {
            // Replace the href with the tracking URL
            const trackingUrl = trackArticleClick(articleUrl, briefQuery);
            link.setAttribute('href', trackingUrl);
            
            // Add data attributes for debugging
            link.setAttribute('data-original-url', articleUrl);
            link.setAttribute('data-tracked', 'true');
            
            // Track when an article is viewed
            link.addEventListener('click', () => {
                debugLog('Article link clicked:', { article: articleUrl, brief: briefQuery });
                viewTracker.startTracking(articleUrl, briefQuery);
            });
        } else {
            debugLog('Article link missing data:', { element: link, href: articleUrl, brief: briefQuery });
        }
    });
    
    // Add data-brief attribute to all news items when they're loaded
    const currentQuery = document.getElementById('current-query')?.value;
    if (currentQuery) {
        debugLog('Current query:', currentQuery);
        const newsItems = document.querySelectorAll('.news-item');
        debugLog(`Found ${newsItems.length} news items`);
        
        newsItems.forEach(item => {
            item.setAttribute('data-brief', currentQuery);
        });
    } else {
        debugLog('No current query found');
    }
    
    // Check analytics health after a short delay
    setTimeout(checkAnalyticsHealth, 1000);
}); 