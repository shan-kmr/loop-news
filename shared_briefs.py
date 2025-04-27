#!/usr/bin/env python3
"""
Shared Briefs Module for Loop

This module provides functionality for creating and accessing 
publicly shared briefs that don't require authentication.
"""

import os
import json
import uuid
import time
from datetime import datetime
import hashlib
from typing import Dict, Optional, Any, List

# File to store the mapping between shared IDs and original briefs
SHARED_BRIEFS_FILE = 'shared_briefs.json'

def generate_shared_id(user_id: str, query: str) -> str:
    """
    Generate a unique identifier for a shared brief.
    
    Args:
        user_id: The ID of the user sharing the brief
        query: The query string of the brief
        
    Returns:
        A unique string identifier for the shared brief
    """
    timestamp = int(time.time())
    input_string = f"{user_id}:{query}:{timestamp}"
    hash_object = hashlib.sha256(input_string.encode())
    # Take first 12 characters of the hex digest
    return hash_object.hexdigest()[:12]

def load_shared_briefs() -> Dict[str, Dict[str, Any]]:
    """
    Load the shared briefs mapping from the JSON file.
    
    Returns:
        Dictionary mapping shared IDs to brief information
    """
    if not os.path.exists(SHARED_BRIEFS_FILE):
        return {}
    
    try:
        with open(SHARED_BRIEFS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is empty or invalid, return empty dict
        return {}

def save_shared_briefs(shared_briefs: Dict[str, Dict[str, Any]]) -> None:
    """
    Save the shared briefs mapping to the JSON file.
    
    Args:
        shared_briefs: Dictionary mapping shared IDs to brief information
    """
    with open(SHARED_BRIEFS_FILE, 'w') as f:
        json.dump(shared_briefs, f, indent=2)

def share_brief(user_id: str, query: str, count: int, freshness: str) -> str:
    """
    Share a brief by creating a unique identifier and storing the mapping.
    
    Args:
        user_id: The ID of the user sharing the brief
        query: The query string of the brief
        count: The result count parameter
        freshness: The freshness parameter
        
    Returns:
        The generated shared ID
    """
    # Generate a unique ID for this brief
    shared_id = generate_shared_id(user_id, query)
    
    # Load existing shared briefs
    shared_briefs = load_shared_briefs()
    
    # Add this brief to the shared briefs mapping
    shared_briefs[shared_id] = {
        'user_id': user_id,
        'query': query,
        'count': count,
        'freshness': freshness,
        'shared_at': datetime.now().isoformat(),
        'view_count': 0
    }
    
    # Save the updated mapping
    save_shared_briefs(shared_briefs)
    
    return shared_id

def get_shared_brief(shared_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the information for a shared brief by its ID.
    
    Args:
        shared_id: The unique identifier for the shared brief
        
    Returns:
        Dictionary with brief information or None if not found
    """
    shared_briefs = load_shared_briefs()
    
    # Check if this shared ID exists
    if shared_id not in shared_briefs:
        return None
    
    # Update view count
    shared_briefs[shared_id]['view_count'] += 1
    save_shared_briefs(shared_briefs)
    
    return shared_briefs[shared_id]

def delete_shared_brief(shared_id: str) -> bool:
    """
    Delete a shared brief by its ID.
    
    Args:
        shared_id: The unique identifier for the shared brief
        
    Returns:
        True if deleted successfully, False if not found
    """
    shared_briefs = load_shared_briefs()
    
    # Check if this shared ID exists
    if shared_id not in shared_briefs:
        return False
    
    # Remove this shared brief
    del shared_briefs[shared_id]
    save_shared_briefs(shared_briefs)
    
    return True

def get_user_shared_briefs(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all shared briefs for a specific user.
    
    Args:
        user_id: The ID of the user
        
    Returns:
        List of dictionaries with shared brief information
    """
    shared_briefs = load_shared_briefs()
    
    # Filter shared briefs by user ID
    user_briefs = []
    for shared_id, brief in shared_briefs.items():
        if brief['user_id'] == user_id:
            user_brief = brief.copy()
            user_brief['shared_id'] = shared_id
            user_briefs.append(user_brief)
    
    return user_briefs 