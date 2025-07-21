"""
Core data models for distributed trace analysis.
"""

from .span import Span
from .trace import Trace
from .events import (
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

__all__ = [
    "Span",
    "Trace",
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
