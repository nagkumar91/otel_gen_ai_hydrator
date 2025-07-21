"""
Core span hydration and trace analysis functionality for processing distributed traces.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field

from .sources.interfaces import SourceConnector
from .models import Span, Trace
from .models.events import (
    GenAIEventBase,
    GenAIUserMessageEvent,
    GenAIAssistantMessageEvent,
    GenAISystemMessageEvent,
    GenAIToolMessageEvent,
    GenAIChoiceEvent
)

logger = logging.getLogger(__name__)


class SpanHydrator:
    """
    Hydrates span data with additional information from a source connector.
    """
    
    def __init__(self, source_connector: SourceConnector):
        """
        Initialize the SpanHydrator.
        
        Args:
            source_connector: Source connector implementing SourceConnector interface
        """
        self.source_connector = source_connector
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_span_by_id(self, span_id: str, time_range: timedelta = timedelta(days=30)) -> Optional[Span]:
        """
        Get a span by its ID. The source connector now returns a fully hydrated Span object.
        
        Args:
            span_id: The span ID to retrieve
            time_range: Time range to search within
            
        Returns:
            Hydrated Span object or None if not found
        """
        try:
            # The source connector now returns a complete Span object
            span = self.source_connector.query_span_by_id(span_id, time_range)
            if not span:
                self.logger.warning(f"Span with ID '{span_id}' not found")
                return None
            
            self.logger.debug(f"Successfully retrieved span '{span_id}' with {len(span.events)} events")
            return span
            
        except Exception as e:
            self.logger.error(f"Failed to get span '{span_id}': {e}")
            return None
    
    def get_child_spans(self, parent_span_id: str, time_range: timedelta = timedelta(days=30), gen_ai_operation_name: str = None) -> List[Span]:
        """
        Get all child spans of a given parent span.
        
        Args:
            parent_span_id: The parent span ID
            time_range: Time range to search within
            gen_ai_operation_name: Optional filter for gen_ai.operation.name
            
        Returns:
            List of hydrated Span objects
        """
        try:
            child_spans = self.source_connector.query_child_spans(parent_span_id, time_range, gen_ai_operation_name)
            self.logger.debug(f"Retrieved {len(child_spans)} child spans for parent span '{parent_span_id}'")
            return child_spans
        except Exception as e:
            self.logger.error(f"Failed to get child spans for parent span '{parent_span_id}': {e}")
            return []
