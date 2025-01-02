"""Integration tests for web settings endpoints."""
import pytest
import json
from src.api import app
from src.config.dynamic_settings import settings_manager

@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_settings_endpoint_integration(client):
    """Test complete settings update flow with web endpoints."""
    # Get initial settings
    response = client.get('/settings')
    assert response.status_code == 200
    initial_settings = json.loads(response.data)
    
    # Update settings
    new_settings = {
        'llm': {
            'temperature': 0.7,
            'max_tokens': 1500,
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
    
    # Verify settings were updated
    response = client.get('/settings')
    assert response.status_code == 200
    updated_settings = json.loads(response.data)
    
    assert updated_settings['llm']['temperature'] == 0.7
    assert updated_settings['llm']['max_tokens'] == 1500
    assert updated_settings['llm']['model'] == 'gpt-4'
    assert updated_settings['document_processing']['chunk_size'] == 800
    assert updated_settings['document_processing']['chunk_overlap'] == 100

def test_settings_validation_in_web(client):
    """Test settings validation through web endpoints."""
    # Test invalid temperature
    invalid_settings = {
        'llm': {
            'temperature': 3.0  # Invalid: > 2.0
        }
    }
    
    response = client.post(
        '/settings',
        data=json.dumps(invalid_settings),
        content_type='application/json'
    )
    assert response.status_code == 400
    
    # Test invalid chunk size
    invalid_settings = {
        'document_processing': {
            'chunk_size': 50  # Invalid: < 100
        }
    }
    
    response = client.post(
        '/settings',
        data=json.dumps(invalid_settings),
        content_type='application/json'
    )
    assert response.status_code == 400

def test_settings_persistence_in_web(client):
    """Test settings persistence through web endpoints."""
    # Update settings
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
    
    # Verify settings persist
    response = client.get('/settings')
    assert response.status_code == 200
    settings = json.loads(response.data)
    assert settings['llm']['temperature'] == 0.8

def test_invalid_content_type(client):
    """Test handling of invalid content type."""
    response = client.post(
        '/settings',
        data='not json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_empty_settings(client):
    """Test handling of empty settings."""
    response = client.post(
        '/settings',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_partial_settings_update(client):
    """Test partial update of settings."""
    # Get initial settings
    response = client.get('/settings')
    assert response.status_code == 200
    initial_settings = json.loads(response.data)
    
    # Update only temperature
    new_settings = {
        'llm': {
            'temperature': 0.9
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
    assert response.status_code == 200
    updated_settings = json.loads(response.data)
    
    assert updated_settings['llm']['temperature'] == 0.9
    assert updated_settings['llm']['max_tokens'] == initial_settings['llm']['max_tokens']
    assert updated_settings['llm']['model'] == initial_settings['llm']['model']

def test_settings_observer_notification(client):
    """Test that settings changes notify observers."""
    # Create a test observer
    notifications = []
    def test_observer(setting_name, new_value):
        notifications.append((setting_name, new_value))
    
    settings_manager.add_observer(test_observer)
    
    try:
        # Update settings
        new_settings = {
            'llm': {
                'temperature': 0.6,
                'max_tokens': 1200
            }
        }
        
        response = client.post(
            '/settings',
            data=json.dumps(new_settings),
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify observer was notified
        assert len(notifications) == 1
        assert notifications[0][0] == 'llm'
        assert notifications[0][1]['temperature'] == 0.6
        assert notifications[0][1]['max_tokens'] == 1200
        
    finally:
        # Clean up
        settings_manager.remove_observer(test_observer)
