"""Tests for settings API endpoints."""
import pytest
import json
from unittest.mock import patch
from src.config.dynamic_settings import settings_manager
from src.api import app

@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        # Mock OpenAI to prevent actual API calls
        with patch('openai.OpenAI'):
            yield client

@pytest.fixture(autouse=True)
def clean_observers():
    """Clean up observers before and after each test."""
    settings_manager._observers = []
    yield
    settings_manager._observers = []

def test_get_settings(client):
    """Test getting settings via API."""
    response = client.get('/settings')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Verify all required settings categories are present
    assert 'llm' in data
    assert 'document_processing' in data
    assert 'response' in data
    assert 'cache' in data
    
    # Verify LLM settings structure
    assert 'temperature' in data['llm']
    assert 'max_tokens' in data['llm']
    assert 'model' in data['llm']
    
    # Verify document processing settings structure
    assert 'chunk_size' in data['document_processing']
    assert 'chunk_overlap' in data['document_processing']
    
    # Verify response settings structure
    assert 'system_prompt' in data['response']
    assert 'source_citation_prompt' in data['response']
    
    # Verify cache settings structure
    assert 'enabled' in data['cache']
    assert 'size' in data['cache']

def test_update_settings(client):
    """Test updating settings via API."""
    # Get current settings
    original_settings = settings_manager.get_all_settings()
    
    # Prepare new settings
    new_settings = {
        'llm': {
            'temperature': 0.7,
            'max_tokens': 1500,
            'model': 'gpt-3.5-turbo'
        }
    }
    
    # Update settings
    response = client.post('/settings',
                         json=new_settings,
                         content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Verify success message
    assert 'message' in data
    assert data['message'] == 'Settings updated successfully'
    
    # Verify settings were updated
    updated_settings = data['settings']
    assert updated_settings['llm']['temperature'] == 0.7
    assert updated_settings['llm']['max_tokens'] == 1500
    assert updated_settings['llm']['model'] == 'gpt-3.5-turbo'
    
    # Verify other settings remained unchanged
    assert updated_settings['document_processing'] == original_settings['document_processing']
    assert updated_settings['response'] == original_settings['response']
    assert updated_settings['cache'] == original_settings['cache']

def test_update_settings_validation(client):
    """Test settings validation during update."""
    invalid_settings = {
        'llm': {
            'temperature': 3.0,  # Invalid: > 2.0
            'max_tokens': 1500,
            'model': 'gpt-3.5-turbo'
        }
    }
    
    response = client.post('/settings',
                         json=invalid_settings,
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_update_settings_invalid_content_type(client):
    """Test updating settings with invalid content type."""
    response = client.post('/settings',
                         data='not json',
                         content_type='text/plain')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'Invalid Content-Type, expected application/json'

def test_update_settings_empty_payload(client):
    """Test updating settings with empty payload."""
    response = client.post('/settings',
                         json={},
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'No settings provided'

def test_update_nested_settings(client):
    """Test updating nested settings via API."""
    new_settings = {
        'response': {
            'system_prompt': 'New test prompt',
            'source_citation_prompt': 'New citation prompt'
        },
        'document_processing': {
            'chunk_size': 800,
            'chunk_overlap': 100
        }
    }
    
    response = client.post('/settings',
                         json=new_settings,
                         content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Verify settings were updated
    updated_settings = data['settings']
    assert updated_settings['response']['system_prompt'] == 'New test prompt'
    assert updated_settings['response']['source_citation_prompt'] == 'New citation prompt'
    assert updated_settings['document_processing']['chunk_size'] == 800
    assert updated_settings['document_processing']['chunk_overlap'] == 100
