"""
Interfaces for source connectors and other components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from datetime import timedelta

if TYPE_CHECKING:
    from .models.events import GenAIEventBase
    from .models import Span


class SourceConnector(ABC):
    """Abstract interface for source connectors that provide trace data."""
    
    @abstractmethod
    def query_span_by_id(self, span_id: str, time_range: timedelta = timedelta(hours=1)) -> Optional["Span"]:
        """
        Query a specific span by its ID and return a Span object.
        
        Args:
            span_id: The span ID to query for
            time_range: Time range to search within
            
        Returns:
            Span object or None if not found
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def query_child_spans(self, parent_span_id: str, time_range: timedelta = timedelta(hours=1), 
                         gen_ai_operation_name: str = None) -> List["Span"]:
        """
        Query all child spans of a given parent span.
        
        Args:
            parent_span_id: The parent span ID
            time_range: Time range to search within
            gen_ai_operation_name: Optional filter for gen_ai.operation.name in customDimensions
            
        Returns:
            List of hydrated Span objects
        """
        pass
