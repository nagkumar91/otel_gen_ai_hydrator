"""
Span model for representing individual spans in distributed traces.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

from .events import GenAIEventBase


class Span(BaseModel):
    """Represents a single span in a distributed trace."""
    span_id: str = Field(..., description="Unique identifier for the span")
    trace_id: str = Field(..., description="Identifier for the trace this span belongs to")
    operation_id: str = Field(..., description="Operation identifier")
    parent_span_id: Optional[str] = Field(None, description="Identifier of the parent span")
    name: str = Field(..., description="Name of the span")
    start_time: datetime = Field(..., description="Start time of the span")
    end_time: datetime = Field(..., description="End time of the span")
    duration_ms: float = Field(..., description="Duration of the span in milliseconds")
    status: str = Field(..., description="Status of the span")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Span attributes")
    events: List[Union[GenAIEventBase, Dict[str, Any]]] = Field(
        default_factory=list, 
        description="List of events associated with the span"
    )
    span_type: str = Field(..., description="Type of span (e.g., 'request', 'dependency')")

    class Config:
        """Pydantic configuration."""
        extra = "allow"
