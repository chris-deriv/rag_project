"""Integration tests for settings management."""
import pytest
import json
from src.web import app
from src.config.dynamic_settings import settings_manager

@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_get_settings(client):
    """Test GET /settings endpoint."""
    response = client.get('/settings')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'llm' in data
    assert 'document_processing' in data
    assert 'response' in data
    assert 'cache' in data

def test_update_settings_valid(client):
    """Test POST /settings endpoint with valid settings."""
    new_settings = {
        'llm': {
            'temperature': 0.5,
            'max_tokens': 2000,
            'model': 'gpt-4'
        },
        'document_processing': {
            'chunk_size': 800,
            'chunk_overlap': 100
        }
    }
    
    response = client.post(
        '/settings',
        data=json.dumps(new_settings),
        content_type='application/json'
    )
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['message'] == 'Settings updated successfully'
    assert data['settings']['llm']['temperature'] == 0.5
    assert data['settings']['llm']['max_tokens'] == 2000
    assert data['settings']['document_processing']['chunk_size'] == 800

def test_update_settings_invalid(client):
    """Test POST /settings endpoint with invalid settings."""
    new_settings = {
        'llm': {
            'temperature': 3.0,  # Invalid: > 2.0
            'max_tokens': 2000
        }
    }
    
    response = client.post(
        '/settings',
        data=json.dumps(new_settings),
        content_type='application/json'
    )
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data

def test_update_settings_invalid_content_type(client):
    """Test POST /settings endpoint with invalid content type."""
    response = client.post(
        '/settings',
        data='not json'
    )
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data

def test_update_settings_empty(client):
    """Test POST /settings endpoint with empty settings."""
    response = client.post(
        '/settings',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data

def test_settings_persistence(client):
    """Test that settings changes persist."""
    # Update settings
    new_settings = {
        'llm': {
            'temperature': 0.6,
            'max_tokens': 1500
        }
    }
    
    response = client.post(
        '/settings',
        data=json.dumps(new_settings),
        content_type='application/json'
    )
    assert response.status_code == 200
    
    # Get settings and verify changes persisted
    response = client.get('/settings')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['llm']['temperature'] == 0.6
    assert data['llm']['max_tokens'] == 1500

def test_settings_validation_chunk_size(client):
    """Test validation of chunk size settings."""
    new_settings = {
        'document_processing': {
            'chunk_size': 50  # Invalid: < 100
        }
    }
    
    response = client.post(
        '/settings',
        data=json.dumps(new_settings),
        content_type='application/json'
    )
    assert response.status_code == 400

def test_settings_validation_prompts(client):
    """Test validation of prompt settings."""
    new_settings = {
        'response': {
            'system_prompt': '',  # Invalid: empty
            'source_citation_prompt': 'Valid prompt'
        }
    }
    
    response = client.post(
        '/settings',
        data=json.dumps(new_settings),
        content_type='application/json'
    )
    assert response.status_code == 400

def test_settings_partial_update(client):
    """Test partial update of settings."""
    # Get original settings
    response = client.get('/settings')
    original_settings = json.loads(response.data)
    
    # Update only temperature
    new_settings = {
        'llm': {
            'temperature': 0.8
        }
    }
    
    response = client.post(
        '/settings',
        data=json.dumps(new_settings),
        content_type='application/json'
    )
    assert response.status_code == 200
    
    # Verify only temperature changed
    response = client.get('/settings')
    current_settings = json.loads(response.data)
    
    assert current_settings['llm']['temperature'] == 0.8
    assert current_settings['llm']['max_tokens'] == original_settings['llm']['max_tokens']
    assert current_settings['llm']['model'] == original_settings['llm']['model']
