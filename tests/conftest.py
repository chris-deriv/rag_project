"""Test configuration and fixtures."""
import pytest
from unittest.mock import MagicMock

# Mock settings with default values
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    mock_llm_settings = {
        'temperature': 0.3,
        'max_tokens': 1000,
        'model': 'gpt-3.5-turbo'
    }
    
    mock_doc_settings = {
        'chunk_size': 500,
        'chunk_overlap': 50
    }
    
    mock_cache_settings = {
        'enabled': True,
        'size': 1000
    }
    
    # Mock the settings imports
    monkeypatch.setattr(
        'config.dynamic_settings.LLM_SETTINGS',
        mock_llm_settings
    )
    monkeypatch.setattr(
        'config.dynamic_settings.DOCUMENT_PROCESSING_SETTINGS',
        mock_doc_settings
    )
    monkeypatch.setattr(
        'config.dynamic_settings.CACHE_SETTINGS',
        mock_cache_settings
    )
    monkeypatch.setattr(
        'config.dynamic_settings.BASIC_SYSTEM_PROMPT',
        'Test system prompt'
    )
    monkeypatch.setattr(
        'config.dynamic_settings.SOURCE_CITATION_PROMPT',
        'Test citation prompt'
    )

@pytest.fixture
def mock_openai_key(monkeypatch):
    """Mock OpenAI API key."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

@pytest.fixture
def mock_openai_client(monkeypatch):
    """Mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='Test response'))]
    mock_client.chat.completions.create.return_value = mock_response
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: mock_client)
    return mock_client
