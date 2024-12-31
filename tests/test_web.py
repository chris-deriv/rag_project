import pytest
from src.web import app
from unittest.mock import patch, Mock, ANY
import json
import io
import os

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_vector_db():
    with patch('src.web.vector_db') as mock_db:
        yield mock_db

@pytest.fixture
def mock_rag_app():
    with patch('src.web.rag_app') as mock_app:
        yield mock_app

@pytest.fixture
def mock_add_document():
    with patch('src.web.add_document') as mock_add:
        yield mock_add

@pytest.fixture
def mock_get_documents():
    with patch('src.web.get_documents') as mock_get:
        yield mock_get

@pytest.fixture
def mock_process_document():
    with patch('src.web.process_document') as mock_process:
        mock_process.return_value = [{
            'id': 1,
            'source_name': 'test.pdf',
            'title': 'Test Document',
            'text': 'Test content',
            'chunk_index': 0,
            'total_chunks': 1
        }]
        yield mock_process

def test_upload_file_success(client, mock_process_document, mock_add_document, mock_get_documents):
    """Test successful file upload and processing."""
    # Create a mock file
    file_content = b'Test PDF content'
    file = (io.BytesIO(file_content), 'test.pdf')
    
    # Make request
    response = client.post(
        '/upload',
        data={'file': file},
        content_type='multipart/form-data'
    )
    
    # Verify initial response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['message'] == 'File upload started'
    assert data['filename'] == 'test.pdf'

def test_upload_status_success(client, mock_process_document, mock_add_document):
    """Test upload status endpoint for successful processing."""
    # Create a mock file and upload it
    file = (io.BytesIO(b'Test PDF content'), 'test.pdf')
    response = client.post(
        '/upload',
        data={'file': file},
        content_type='multipart/form-data'
    )
    
    # Wait a moment for async processing
    import time
    time.sleep(0.1)
    
    # Check status - should show completed
    response = client.get('/upload-status/test.pdf')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'completed'
    assert data.get('error') is None

def test_upload_status_error(client, mock_process_document):
    """Test upload status endpoint for failed processing."""
    # Mock processing error
    mock_process_document.side_effect = Exception('Processing error')
    
    # Create a mock file and upload it
    file = (io.BytesIO(b'Test PDF content'), 'test.pdf')
    response = client.post(
        '/upload',
        data={'file': file},
        content_type='multipart/form-data'
    )
    
    # Wait a moment for async processing
    import time
    time.sleep(0.1)
    
    # Check status - should show error
    response = client.get('/upload-status/test.pdf')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'error' in data
    assert 'Processing error' in data['error']

def test_upload_status_unknown_file(client):
    """Test upload status endpoint for unknown file."""
    response = client.get('/upload-status/nonexistent.pdf')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['status'] == 'unknown'

def test_upload_file_no_file(client):
    """Test file upload with no file."""
    response = client.post('/upload')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'No file part'

def test_upload_file_empty_filename(client):
    """Test file upload with empty filename."""
    file = (io.BytesIO(b''), '')
    response = client.post(
        '/upload',
        data={'file': file},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'No selected file'

def test_upload_file_invalid_type(client):
    """Test file upload with invalid file type."""
    file = (io.BytesIO(b'test'), 'test.txt')
    response = client.post(
        '/upload',
        data={'file': file},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'Invalid file type'

def test_document_names_success(client, mock_vector_db):
    """Test successful document names retrieval."""
    # Mock document list with complete metadata
    mock_vector_db.list_document_names.return_value = [
        {
            'source_name': 'test1.pdf',
            'title': 'Test Document 1',
            'chunk_count': 5,
            'total_chunks': 5,
            'status': 'completed'
        },
        {
            'source_name': 'test2.docx',
            'title': 'Test Document 2',
            'chunk_count': 3,
            'total_chunks': 3,
            'status': 'completed'
        }
    ]
    
    response = client.get('/document-names')
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 2
    
    # Verify required fields
    required_fields = ['source_name', 'title', 'chunk_count', 'total_chunks', 'status']
    for doc in data:
        for field in required_fields:
            assert field in doc
    
    # Verify specific document data
    assert data[0]['source_name'] == 'test1.pdf'
    assert data[1]['source_name'] == 'test2.docx'
    
    # Verify list_document_names was called
    mock_vector_db.list_document_names.assert_called_once()

def test_document_names_error(client, mock_vector_db):
    """Test error handling in document names retrieval."""
    # Mock database error
    mock_vector_db.list_document_names.side_effect = Exception('Database error')
    
    response = client.get('/document-names')
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Database error' in data['error']

def test_search_titles_success(client, mock_vector_db):
    """Test successful title search."""
    # Mock search results with complete metadata
    mock_vector_db.search_titles.return_value = [
        {
            'title': 'Python Programming Guide',
            'source_name': 'python_guide.pdf',
            'file_type': 'pdf',
            'section_type': 'content'
        },
        {
            'title': 'Learning Python',
            'source_name': 'learning.pdf',
            'file_type': 'pdf',
            'section_type': 'content'
        }
    ]
    
    # Make request
    response = client.get('/search-titles?q=python')
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 2
    
    # Verify required fields
    required_fields = ['source_name', 'title', 'file_type', 'section_type']
    for doc in data:
        for field in required_fields:
            assert field in doc
    
    # Verify specific document data
    assert data[0]['title'] == 'Python Programming Guide'
    assert data[0]['source_name'] == 'python_guide.pdf'
    
    # Verify search was called with correct query
    mock_vector_db.search_titles.assert_called_once_with('python')

def test_search_titles_no_query(client):
    """Test title search with no query parameter."""
    response = client.get('/search-titles')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No search query provided'

def test_search_titles_empty_query(client):
    """Test title search with empty query string."""
    response = client.get('/search-titles?q=')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No search query provided'

def test_search_titles_error_handling(client, mock_vector_db):
    """Test error handling in title search."""
    # Mock search error
    mock_vector_db.search_titles.side_effect = Exception('Database error')
    
    response = client.get('/search-titles?q=python')
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Database error' in data['error']

def test_chat_with_title_filter(client, mock_rag_app):
    """Test chat endpoint with title filter."""
    # Mock response
    mock_rag_app.query_documents.return_value = "Response about Python"
    
    # Make request with title filter
    response = client.post('/chat', json={
        'query': 'What is this about?',
        'title': 'python'
    })
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['response'] == 'Response about Python'
    
    # Verify query was called with title filter
    mock_rag_app.query_documents.assert_called_once_with(
        'What is this about?',
        source_names=[],  # Empty list by default
        title='python'
    )

def test_chat_with_source_names_filter(client, mock_rag_app):
    """Test chat endpoint with source names filter."""
    # Mock response
    mock_rag_app.query_documents.return_value = "Response from test.pdf"
    
    # Make request with source names filter
    source_names = ['test1.pdf', 'test2.pdf']
    response = client.post('/chat', json={
        'query': 'What is this about?',
        'source_names': source_names
    })
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['response'] == 'Response from test.pdf'
    
    # Verify query was called with source names filter
    mock_rag_app.query_documents.assert_called_once_with(
        'What is this about?',
        source_names=source_names,
        title=None
    )

def test_chat_with_both_filters(client, mock_rag_app):
    """Test chat endpoint with both title and source names filters."""
    # Mock response
    mock_rag_app.query_documents.return_value = "Filtered response"
    
    # Make request with both filters
    source_names = ['test1.pdf', 'test2.pdf']
    response = client.post('/chat', json={
        'query': 'What is this about?',
        'source_names': source_names,
        'title': 'python'
    })
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['response'] == 'Filtered response'
    
    # Verify query was called with both filters
    mock_rag_app.query_documents.assert_called_once_with(
        'What is this about?',
        source_names=source_names,
        title='python'
    )

def test_chat_no_query(client):
    """Test chat endpoint with missing query."""
    response = client.post('/chat', json={
        'title': 'python'
    })
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No query provided'

def test_chat_error_handling(client, mock_rag_app):
    """Test error handling in chat endpoint."""
    # Mock query error
    mock_rag_app.query_documents.side_effect = Exception('Query error')
    
    response = client.post('/chat', json={
        'query': 'What is this about?',
        'title': 'python'
    })
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Query error' in data['error']
