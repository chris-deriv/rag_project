import pytest
from src.web import app
from unittest.mock import patch, Mock
import json
import io

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

def test_upload_file_success(client, mock_add_document, mock_get_documents):
    """Test successful file upload."""
    # Create a mock file
    file_content = b'Test PDF content'
    file = (io.BytesIO(file_content), 'test.pdf')
    
    # Make request
    response = client.post(
        '/upload',
        data={'file': file},
        content_type='multipart/form-data'
    )
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['message'] == 'File processed successfully'
    
    # Verify document was added and indexed
    mock_add_document.assert_called_once()
    mock_get_documents.assert_called_once()

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

def test_upload_file_processing_error(client, mock_add_document):
    """Test error handling during file processing."""
    # Mock processing error
    mock_add_document.side_effect = Exception('Processing error')
    
    file = (io.BytesIO(b'Test PDF content'), 'test.pdf')
    response = client.post(
        '/upload',
        data={'file': file},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Processing error' in data['error']

def test_document_names_success(client, mock_vector_db):
    """Test successful document names retrieval."""
    # Mock document list
    mock_vector_db.list_document_names.return_value = [
        {
            'source_name': 'test1.pdf',
            'title': 'Test Document 1',
            'chunk_count': 5,
            'total_chunks': 5
        },
        {
            'source_name': 'test2.pdf',
            'title': 'Test Document 2',
            'chunk_count': 0,  # Still processing
            'total_chunks': 3
        }
    ]
    
    response = client.get('/document-names')
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 2
    assert data[0]['title'] == 'Test Document 1'
    assert data[0]['chunk_count'] == 5
    assert data[1]['title'] == 'Test Document 2'
    assert data[1]['chunk_count'] == 0  # Still processing
    
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
    # Mock search results
    mock_vector_db.search_titles.return_value = [
        {
            'title': 'Python Programming Guide',
            'source_name': 'python_guide.pdf'
        },
        {
            'title': 'Learning Python',
            'source_name': 'learning.pdf'
        }
    ]
    
    # Make request
    response = client.get('/search-titles?q=python')
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 2
    assert data[0]['title'] == 'Python Programming Guide'
    assert data[1]['title'] == 'Learning Python'
    
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
        source_name=None,
        title='python'
    )

def test_chat_with_source_filter(client, mock_rag_app):
    """Test chat endpoint with source name filter."""
    # Mock response
    mock_rag_app.query_documents.return_value = "Response from test.pdf"
    
    # Make request with source filter
    response = client.post('/chat', json={
        'query': 'What is this about?',
        'source_name': 'test.pdf'
    })
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['response'] == 'Response from test.pdf'
    
    # Verify query was called with source filter
    mock_rag_app.query_documents.assert_called_once_with(
        'What is this about?',
        source_name='test.pdf',
        title=None
    )

def test_chat_with_both_filters(client, mock_rag_app):
    """Test chat endpoint with both title and source filters."""
    # Mock response
    mock_rag_app.query_documents.return_value = "Filtered response"
    
    # Make request with both filters
    response = client.post('/chat', json={
        'query': 'What is this about?',
        'source_name': 'test.pdf',
        'title': 'python'
    })
    
    # Verify response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['response'] == 'Filtered response'
    
    # Verify query was called with both filters
    mock_rag_app.query_documents.assert_called_once_with(
        'What is this about?',
        source_name='test.pdf',
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
