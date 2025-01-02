"""Test configuration and fixtures."""
import os
import pytest
from unittest.mock import MagicMock

def pytest_configure(config):
    """Configure test environment before any imports."""
    # Mock all required environment variables
    env_vars = {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MODEL': 'gpt-3.5-turbo',
        'DEFAULT_TEMPERATURE': '0.3',
        'DEFAULT_MAX_TOKENS': '1000',
        'DEFAULT_CHUNK_SIZE': '500',
        'DEFAULT_CHUNK_OVERLAP': '50',
        'RESPONSE_CACHE_SIZE': '1000',
        'RESPONSE_CACHE_ENABLED': 'true',
        'CHROMA_COLLECTION_NAME': 'test_collection',
        'CHROMA_PERSIST_DIR': './test_db',
        'SYSTEM_PROMPT': 'Test system prompt',
        'SOURCE_CITATION_PROMPT': 'Test citation prompt'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value

@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client and collection."""
    # Create mock client
    mock_client = MagicMock()
    
    # Create mock collection
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.get.return_value = {
        'ids': [],
        'embeddings': [],
        'documents': [],
        'metadatas': []
    }
    
    # Setup client methods
    mock_client.get_collection.return_value = mock_collection
    mock_client.create_collection.return_value = mock_collection
    mock_client.get_or_create_collection.return_value = mock_collection
    
    # Return both mocks as expected by the tests
    return mock_client, mock_collection

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
def mock_openai_client(monkeypatch):
    """Mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='Test response'))]
    mock_client.chat.completions.create.return_value = mock_response
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: mock_client)
    return mock_client
