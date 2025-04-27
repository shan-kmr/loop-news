#!/usr/bin/env python3
"""
Brief Utilities Module for Loop

This module provides utility functions for rendering briefs,
used by both the history and shared brief views.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import render_template

def prepare_brief_for_display(
    query: str,
    entry: Dict[str, Any],
    history_entries: List[Dict[str, Any]],
    history: Dict[str, Any],
    is_shared_view: bool = False,
    shared_by: str = None,
    user=None
) -> Dict[str, Any]:
    """
    Prepare a brief for display, used by both history and shared views.
    
    Args:
        query: The query string for the brief
        entry: The brief entry from history
        history_entries: List of history entries to display in sidebar
        history: The full history dictionary
        is_shared_view: Whether this is a shared view
        shared_by: Username of user who shared the brief (for shared views)
        user: Current user object
        
    Returns:
        Dictionary with template parameters
    """
    # Extract the data needed for display
    current_query = entry['query']
    results = entry.get('results', None)
    search_time = datetime.fromisoformat(entry['timestamp']) if 'timestamp' in entry else datetime.now()
    topic_groups = entry.get('topic_groups', [])
    day_summaries = entry.get('day_summaries', {})
    
    # Determine active tab
    active_tab = "shared" if is_shared_view else "history"
    
    # Return template parameters
    return {
        'query': current_query,
        'results': results,
        'search_time': search_time,
        'error': None,
        'history_entries': history_entries,
        'topic_groups': topic_groups,
        'day_summaries': day_summaries,
        'history': history,
        'is_shared_view': is_shared_view,
        'shared_by': shared_by,
        'active_tab': active_tab,
        'user': user
    }

def render_brief_template(template_params: Dict[str, Any]) -> str:
    """
    Render the brief template with the given parameters.
    
    Args:
        template_params: Dictionary with template parameters
        
    Returns:
        Rendered HTML
    """
    return render_template('index.html', **template_params) 