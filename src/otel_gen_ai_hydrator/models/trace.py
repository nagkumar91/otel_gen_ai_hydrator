"""
Trace model for representing complete traces as trees of spans.
"""

from typing import List
from pydantic import BaseModel, Field

from .span import Span


class Trace(BaseModel):
    """Represents a complete trace as a tree of spans."""
    trace_id: str = Field(..., description="Unique identifier for the trace")
    operation_id: str = Field(..., description="Operation identifier")
    spans: List[Span] = Field(default_factory=list, description="List of spans in the trace")
    total_duration_ms: float = Field(..., description="Total duration of the trace in milliseconds")
    span_count: int = Field(..., description="Number of spans in the trace")
    error_count: int = Field(..., description="Number of spans with errors")

    class Config:
        """Pydantic configuration."""
        extra = "allow"
