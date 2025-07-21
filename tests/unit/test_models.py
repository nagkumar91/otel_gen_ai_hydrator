"""
Unit tests for GenAI event models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from otel_gen_ai_hydrator.models.events import (
    GenAIUserMessageEvent,
    GenAIAssistantMessageEvent,
    GenAISystemMessageEvent,
    GenAIChoiceEvent,
    GenAIUserMessageBody,
    GenAIAssistantMessageBody,
    GenAISystemMessageBody
)


class TestGenAIEventModels:
    """Test cases for GenAI event models."""
    
    def test_user_message_event_valid(self):
        """Test creating a valid GenAI user message event."""
        event_data = {
            'event_name': 'gen_ai.user.message',
            'gen_ai_system': 'openai',
            'body': {
                'role': 'user',
                'content': 'Hello, how are you?'
            }
        }
        
        event = GenAIUserMessageEvent.model_validate(event_data)
        
        assert event.event_name == 'gen_ai.user.message'
        assert event.gen_ai_system == 'openai'
        assert event.body.role == 'user'
        assert event.body.content == 'Hello, how are you?'
    
    def test_assistant_message_event_valid(self):
        """Test creating a valid GenAI assistant message event."""
        event_data = {
            'event_name': 'gen_ai.assistant.message',
            'gen_ai_system': 'openai',
            'body': {
                'role': 'assistant',
                'content': 'I am doing well, thank you!'
            }
        }
        
        event = GenAIAssistantMessageEvent.model_validate(event_data)
        
        assert event.event_name == 'gen_ai.assistant.message'
        assert event.gen_ai_system == 'openai'
        assert event.body.role == 'assistant'
        assert event.body.content == 'I am doing well, thank you!'
    
    def test_system_message_event_valid(self):
        """Test creating a valid GenAI system message event."""
        event_data = {
            'event_name': 'gen_ai.system.message',
            'gen_ai_system': 'openai',
            'body': {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            }
        }
        
        event = GenAISystemMessageEvent.model_validate(event_data)
        
        assert event.event_name == 'gen_ai.system.message'
        assert event.gen_ai_system == 'openai'
        assert event.body.role == 'system'
        assert event.body.content == 'You are a helpful assistant.'
    
    def test_choice_event_valid(self):
        """Test creating a valid GenAI choice event."""
        event_data = {
            'event_name': 'gen_ai.choice',
            'gen_ai_system': 'openai',
            'body': {
                'finish_reason': 'stop',
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'I am doing well, thank you!'
                }
            }
        }
        
        event = GenAIChoiceEvent.model_validate(event_data)
        
        assert event.event_name == 'gen_ai.choice'
        assert event.gen_ai_system == 'openai'
        assert event.body.finish_reason == 'stop'
        assert event.body.index == 0
        assert event.body.message.role == 'assistant'
        assert event.body.message.content == 'I am doing well, thank you!'
    
    def test_user_message_invalid_role(self):
        """Test validation error when user message has wrong role."""
        event_data = {
            'event_name': 'gen_ai.user.message',
            'gen_ai_system': 'openai',
            'body': {
                'role': 'assistant',  # Wrong role for user message
                'content': 'Hello, how are you?'
            }
        }
        
        with pytest.raises(ValidationError):
            GenAIUserMessageEvent.model_validate(event_data)
    
    def test_assistant_message_invalid_role(self):
        """Test validation error when assistant message has wrong role."""
        event_data = {
            'event_name': 'gen_ai.assistant.message',
            'gen_ai_system': 'openai',
            'body': {
                'role': 'user',  # Wrong role for assistant message
                'content': 'I am doing well, thank you!'
            }
        }
        
        with pytest.raises(ValidationError):
            GenAIAssistantMessageEvent.model_validate(event_data)
    
    def test_system_message_invalid_role(self):
        """Test validation error when system message has wrong role."""
        event_data = {
            'event_name': 'gen_ai.system.message',
            'gen_ai_system': 'openai',
            'body': {
                'role': 'user',  # Wrong role for system message
                'content': 'You are a helpful assistant.'
            }
        }
        
        with pytest.raises(ValidationError):
            GenAISystemMessageEvent.model_validate(event_data)
    
    def test_missing_required_fields(self):
        """Test validation error when required fields are missing."""
        event_data = {
            'event_name': 'gen_ai.user.message',
            # Missing gen_ai_system and body
        }
        
        with pytest.raises(ValidationError):
            GenAIUserMessageEvent.model_validate(event_data)
    
    def test_message_body_serialization(self):
        """Test that message bodies can be serialized and deserialized."""
        original_body = GenAIUserMessageBody(
            role='user',
            content='Hello, world!'
        )
        
        # Serialize to dict
        body_dict = original_body.model_dump()
        
        # Deserialize back
        recreated_body = GenAIUserMessageBody.model_validate(body_dict)
        
        assert recreated_body.role == original_body.role
        assert recreated_body.content == original_body.content
