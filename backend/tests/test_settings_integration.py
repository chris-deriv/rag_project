"""Integration tests for settings management."""
import pytest
from unittest.mock import Mock, patch
from src.config.dynamic_settings import settings_manager

# Mock OpenAI before importing Chatbot
with patch('openai.OpenAI'):
    from src.chatbot import Chatbot
    from src.api import app
    from src.search import SearchEngine

@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture(autouse=True)
def clean_observers():
    """Clean up observers before and after each test."""
    settings_manager._observers = []
    yield
    settings_manager._observers = []

@pytest.fixture
def mock_chatbot():
    """Mock Chatbot to prevent observer registration."""
    with patch('src.search.Chatbot') as mock:
        mock_instance = Mock()
        mock_instance._response_cache = {}
        mock.return_value = mock_instance
        yield mock_instance

def test_chatbot_settings_update():
    """Test that Chatbot properly handles settings updates."""
    chatbot = Chatbot()
    
    # Verify observer was registered
    assert len(settings_manager._observers) == 1
    
    # Get original prompt
    original_prompt = chatbot.settings['response']['source_citation_prompt']
    
    # Update source citation prompt
    new_settings = {
        'response': {
            'source_citation_prompt': 'New test prompt for citations',
            'system_prompt': chatbot.settings['response']['system_prompt']
        }
    }
    result = settings_manager.update_settings(new_settings)
    assert result is True  # Verify update was successful
    
    # Verify chatbot's settings were updated
    assert chatbot.settings['response']['source_citation_prompt'] == 'New test prompt for citations'
    assert chatbot.settings['response']['source_citation_prompt'] != original_prompt
    
    # Verify cache was cleared
    assert len(chatbot._response_cache) == 0

def test_chatbot_system_prompt_update():
    """Test that Chatbot properly handles system prompt updates."""
    chatbot = Chatbot()
    
    # Verify observer was registered
    assert len(settings_manager._observers) == 1
    
    # Get original system prompt
    original_prompt = chatbot.settings['response']['system_prompt']
    
    # Update system prompt
    new_settings = {
        'response': {
            'system_prompt': 'New system prompt for testing',
            'source_citation_prompt': chatbot.settings['response']['source_citation_prompt']
        }
    }
    result = settings_manager.update_settings(new_settings)
    assert result is True  # Verify update was successful
    
    # Verify chatbot's settings were updated
    assert chatbot.settings['response']['system_prompt'] == 'New system prompt for testing'
    assert chatbot.settings['response']['system_prompt'] != original_prompt
    
    # Verify cache was cleared
    assert len(chatbot._response_cache) == 0

def test_nested_settings_update():
    """Test that nested settings updates are properly handled."""
    chatbot = Chatbot()
    
    # Verify observer was registered
    assert len(settings_manager._observers) == 1
    
    # Update multiple nested settings
    new_settings = {
        'response': {
            'source_citation_prompt': 'New citation format',
            'system_prompt': 'New system prompt'
        },
        'llm': {
            'temperature': 0.7,
            'max_tokens': chatbot.settings['llm']['max_tokens'],
            'model': chatbot.settings['llm']['model']
        }
    }
    result = settings_manager.update_settings(new_settings)
    assert result is True  # Verify update was successful
    
    # Verify all nested settings were updated
    assert chatbot.settings['response']['source_citation_prompt'] == 'New citation format'
    assert chatbot.settings['response']['system_prompt'] == 'New system prompt'
    assert chatbot.settings['llm']['temperature'] == 0.7

def test_search_engine_settings_update(mock_chatbot):
    """Test that SearchEngine properly handles settings updates."""
    search_engine = SearchEngine()
    
    # Only SearchEngine should be registered as observer (Chatbot is mocked)
    assert len(settings_manager._observers) == 1
    
    # Mock the search method to return deterministic results
    original_results = [
        {'id': '1', 'combined_score': 0.8},
        {'id': '2', 'combined_score': 0.6}
    ]
    new_results = [
        {'id': '1', 'combined_score': 0.9},
        {'id': '2', 'combined_score': 0.7}
    ]
    
    search_engine.search = Mock(side_effect=[original_results, new_results])
    
    # Create a test query and get initial results
    query = "test query"
    initial_results = search_engine.search(query, n_results=5)
    
    # Update LLM settings
    new_settings = {
        'llm': {
            'temperature': 0.3,
            'max_tokens': 1500,
            'model': search_engine.settings['llm']['model']
        }
    }
    result = settings_manager.update_settings(new_settings)
    assert result is True  # Verify update was successful
    
    # Verify relevance cache was cleared
    assert len(search_engine._relevance_cache) == 0
    
    # Get new results
    new_results = search_engine.search(query, n_results=5)
    
    # Verify search was called twice
    assert search_engine.search.call_count == 2
    
    # Verify results are different
    assert initial_results != new_results

def test_observer_cleanup(mock_chatbot):
    """Test that observers are properly cleaned up."""
    # Create chatbot and verify observer is registered
    chatbot = Chatbot()
    assert len(settings_manager._observers) == 1
    
    # Create search engine and verify observers
    # SearchEngine adds one observer (Chatbot is mocked)
    search_engine = SearchEngine()
    assert len(settings_manager._observers) == 2
    
    # Update settings and verify both components are notified
    new_settings = {
        'llm': {
            'temperature': 0.5
        }
    }
    result = settings_manager.update_settings(new_settings)
    assert result is True
    
    # Verify both caches were cleared
    assert len(chatbot._response_cache) == 0
    assert len(search_engine._relevance_cache) == 0
