"""
Utility functions for source connectors.
"""

from typing import Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def parse_timestamp(timestamp_str: Union[str, datetime]) -> datetime:
    """
    Parse timestamp string to datetime object.
    
    Args:
        timestamp_str: Timestamp as string or datetime object
        
    Returns:
        Parsed datetime object
    """
    if isinstance(timestamp_str, datetime):
        return timestamp_str
    
    # Handle different timestamp formats
    try:
        if 'T' in timestamp_str:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}. Using current time.")
        return datetime.now()


def calculate_end_time(start_time: Union[str, datetime], duration_ms: float) -> datetime:
    """
    Calculate end time from start time and duration.
    
    Args:
        start_time: Start time as string or datetime
        duration_ms: Duration in milliseconds
        
    Returns:
        Calculated end time
    """
    start_dt = parse_timestamp(start_time)
    return start_dt + timedelta(milliseconds=duration_ms)
