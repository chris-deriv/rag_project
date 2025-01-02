import pytest
from rag_backend.web import app
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
    with patch('rag_backend.web.vector_db') as mock_db:
        yield mock_db

@pytest.fixture
def mock_rag_app():
    with patch('rag_backend.web.rag_app') as mock_app:
        yield mock_app

@pytest.fixture
def mock_add_document():
    with patch('rag_backend.web.add_document') as mock_add:
        yield mock_add

@pytest.fixture
def mock_get_documents():
    with patch('rag_backend.web.get_documents') as mock_get:
        yield mock_get

@pytest.fixture
def mock_document_store():
    with patch('rag_backend.web.document_store') as mock_store:
        # Create a mock ProcessingState
        mock_state = Mock()
        mock_state.status = 'completed'
        mock_state.error = None
        mock_state.source_name = 'test.pdf'
        mock_state.chunk_count = 5
        mock_state.total_chunks = 5
        
        # Configure mock store
        mock_store.process_and_store_document.return_value = mock_state
        mock_store.get_processing_state.return_value = mock_state
        mock_store.get_document_info.return_value = {
            'source_name': 'test.pdf',
            'title': 'Test Document',
            'chunk_count': 5,
            'total_chunks': 5
        }
        yield mock_store

def test_upload_file_success(client, mock_document_store, mock_rag_app):
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

def test_upload_status_success(client, mock_document_store):
    """Test upload status endpoint for successful processing."""
    # Configure mock state
    mock_state = Mock()
    mock_state.status = 'completed'
    mock_state.error = None
    mock_state.source_name = 'test.pdf'
    mock_state.chunk_count = 5
    mock_state.total_chunks = 5
    mock_document_store.get_processing_state.return_value = mock_state
    
    # Check status
    response = client.get('/upload-status/test.pdf')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'completed'
    assert data.get('error') is None
    assert data['source_name'] == 'test.pdf'
    assert data['chunk_count'] == 5
    assert data['total_chunks'] == 5

def test_upload_status_error(client, mock_document_store):
    """Test upload status endpoint for failed processing."""
    # Configure mock state with error
    mock_state = Mock()
    mock_state.status = 'error'
    mock_state.error = 'Processing error'
    mock_state.source_name = 'test.pdf'
    mock_state.chunk_count = 0
    mock_state.total_chunks = 0
    mock_document_store.get_processing_state.return_value = mock_state
    
    # Check status
    response = client.get('/upload-status/test.pdf')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert data['error'] == 'Processing error'
    assert data['chunk_count'] == 0

def test_upload_status_processing(client, mock_document_store):
    """Test upload status endpoint for document still processing."""
    # Configure mock state for processing
    mock_state = Mock()
    mock_state.status = 'processing'
    mock_state.error = None
    mock_state.source_name = 'test.pdf'
    mock_state.chunk_count = 2
    mock_state.total_chunks = 5
    mock_document_store.get_processing_state.return_value = mock_state
    
    # Check status
    response = client.get('/upload-status/test.pdf')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'processing'
    assert data['chunk_count'] == 2
    assert data['total_chunks'] == 5

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
