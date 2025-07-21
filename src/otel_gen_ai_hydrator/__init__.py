"""
OpenTelemetry GenAI Hydrator - A toolkit for hydrating OpenTelemetry traces with GenAI event data.

This package provides tools and utilities for:
- Collecting and processing distributed traces from OpenTelemetry sources
- Hydrating spans with GenAI-specific event data
- Parsing and validating GenAI events using Pydantic models
- Integrating with various tracing systems (Application Insights, Jaeger, etc.)
- Analyzing GenAI application performance and reliability patterns
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import (
    Span, 
    Trace,
    GenAIChoiceEvent, 
    GenAISystemMessageEvent, 
    GenAIUserMessageEvent, 
    GenAIAssistantMessageEvent, 
    GenAIToolMessageEvent,
    GenAIEventBase,
    GenAISystem,
    FinishReason,
    ToolCallType,
    ToolCall,
    FunctionToolCall,
    FunctionCall,
    ChoiceMessage,
    GenAIUserMessageBody,
    GenAIAssistantMessageBody,
    GenAISystemMessageBody,
    GenAIToolMessageBody,
    GenAIChoiceBody
)
from .span_hydrator import SpanHydrator
from .sources.interfaces import SourceConnector
from .sources.application_insights import ApplicationInsightsConnector

__all__ = [
    "Span",
    "Trace",
    "SpanHydrator",
    "SourceConnector",
    "ApplicationInsightsConnector",
    # Events
    "GenAIChoiceEvent",
    "GenAISystemMessageEvent",
    "GenAIUserMessageEvent", 
    "GenAIAssistantMessageEvent",
    "GenAIToolMessageEvent",
    "GenAIEventBase",
    "GenAISystem",
    "FinishReason",
    "ToolCallType",
    "ToolCall",
    "FunctionToolCall",
    "FunctionCall",
    "ChoiceMessage",
    # Event Bodies
    "GenAIUserMessageBody",
    "GenAIAssistantMessageBody",
    "GenAISystemMessageBody",
    "GenAIToolMessageBody",
    "GenAIChoiceBody",
]
