"""Unit tests for settings management."""
import pytest
from rag_backend.config.dynamic_settings import (
    LLMSettings,
    DocumentProcessingSettings,
    ResponseSettings,
    CacheSettings,
    DynamicSettings
)

def test_llm_settings_validation():
    """Test LLM settings validation."""
    # Valid settings
    settings = LLMSettings(temperature=0.7, max_tokens=1000, model="gpt-3.5-turbo")
    assert settings.validate() is True

    # Invalid temperature
    settings = LLMSettings(temperature=2.5, max_tokens=1000, model="gpt-3.5-turbo")
    assert settings.validate() is False

    # Invalid max_tokens
    settings = LLMSettings(temperature=0.7, max_tokens=0, model="gpt-3.5-turbo")
    assert settings.validate() is False

def test_document_processing_settings_validation():
    """Test document processing settings validation."""
    # Valid settings
    settings = DocumentProcessingSettings(chunk_size=500, chunk_overlap=50)
    assert settings.validate() is True

    # Invalid chunk_size
    settings = DocumentProcessingSettings(chunk_size=50, chunk_overlap=50)
    assert settings.validate() is False

    # Invalid chunk_overlap
    settings = DocumentProcessingSettings(chunk_size=500, chunk_overlap=600)
    assert settings.validate() is False

def test_response_settings_validation():
    """Test response settings validation."""
    # Valid settings
    settings = ResponseSettings(
        system_prompt="Test prompt",
        source_citation_prompt="Test citation prompt"
    )
    assert settings.validate() is True

    # Empty system prompt
    settings = ResponseSettings(
        system_prompt="",
        source_citation_prompt="Test citation prompt"
    )
    assert settings.validate() is False

    # Empty citation prompt
    settings = ResponseSettings(
        system_prompt="Test prompt",
        source_citation_prompt=""
    )
    assert settings.validate() is False

def test_cache_settings_validation():
    """Test cache settings validation."""
    # Valid settings
    settings = CacheSettings(enabled=True, size=1000)
    assert settings.validate() is True

    # Invalid size
    settings = CacheSettings(enabled=True, size=0)
    assert settings.validate() is False

def test_dynamic_settings_update():
    """Test dynamic settings update functionality."""
    settings = DynamicSettings()
    
    # Test observer notification
    notifications = []
    def test_observer(setting_name, new_value):
        notifications.append((setting_name, new_value))
    
    settings.add_observer(test_observer)
    
    # Update LLM settings
    new_settings = {
        'llm': {
            'temperature': 0.5,
            'max_tokens': 2000,
            'model': 'gpt-4'
        }
    }
    
    success = settings.update_settings(new_settings)
    assert success is True
    assert len(notifications) == 1
    assert notifications[0][0] == 'llm'
    assert notifications[0][1]['temperature'] == 0.5
    assert notifications[0][1]['max_tokens'] == 2000
    assert notifications[0][1]['model'] == 'gpt-4'

def test_dynamic_settings_invalid_update():
    """Test dynamic settings update with invalid values."""
    settings = DynamicSettings()
    
    # Invalid temperature
    new_settings = {
        'llm': {
            'temperature': 3.0,  # Invalid: > 2.0
            'max_tokens': 1000,
            'model': 'gpt-3.5-turbo'
        }
    }
    
    success = settings.update_settings(new_settings)
    assert success is False
    assert settings.llm.temperature != 3.0  # Original value should be preserved

def test_dynamic_settings_partial_update():
    """Test partial update of settings."""
    settings = DynamicSettings()
    original_max_tokens = settings.llm.max_tokens
    
    # Update only temperature
    new_settings = {
        'llm': {
            'temperature': 0.8
        }
    }
    
    success = settings.update_settings(new_settings)
    assert success is True
    assert settings.llm.temperature == 0.8
    assert settings.llm.max_tokens == original_max_tokens  # Should remain unchanged

def test_dynamic_settings_get_all():
    """Test getting all settings."""
    settings = DynamicSettings()
    all_settings = settings.get_all_settings()
    
    assert 'llm' in all_settings
    assert 'document_processing' in all_settings
    assert 'response' in all_settings
    assert 'cache' in all_settings
    
    assert 'temperature' in all_settings['llm']
    assert 'max_tokens' in all_settings['llm']
    assert 'model' in all_settings['llm']
