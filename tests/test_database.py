import pytest
from src.database import VectorDatabase
import chromadb
from unittest.mock import Mock, patch, MagicMock
import numpy as np

@pytest.fixture
def mock_chroma_client():
    with patch('chromadb.Client') as mock_client:
        # Create mock collection
        mock_collection = Mock()
        
        # Configure mock client to return mock collection
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        yield mock_client.return_value, mock_collection

@pytest.fixture
def db(mock_chroma_client):
    _, mock_collection = mock_chroma_client
    db = VectorDatabase()
    return db

def test_init(mock_chroma_client):
    """Test database initialization."""
    mock_client, mock_collection = mock_chroma_client
    
    db = VectorDatabase()
    
    mock_client.get_or_create_collection.assert_called_once()
    assert db.collection == mock_collection

def test_add_documents(db, mock_chroma_client):
    """Test adding documents to the database."""
    _, mock_collection = mock_chroma_client
    
    documents = [
        {
            'id': 1,
            'text': 'Test document 1',
            'embedding': np.array([0.1, 0.2, 0.3]),
            'source_name': 'test1.pdf',
            'title': 'Test Document 1'
        },
        {
            'id': 2,
            'text': 'Test document 2',
            'embedding': np.array([0.4, 0.5, 0.6]),
            'source_name': 'test2.pdf',
            'title': 'Test Document 2'
        }
    ]
    
    db.add_documents(documents)
    
    mock_collection.add.assert_called_once()
    call_kwargs = mock_collection.add.call_args[1]
    
    assert len(call_kwargs['embeddings']) == 2
    assert len(call_kwargs['documents']) == 2
    assert len(call_kwargs['metadatas']) == 2
    assert len(call_kwargs['ids']) == 2

def test_query_with_title_filter(db, mock_chroma_client):
    """Test querying documents with title filter."""
    _, mock_collection = mock_chroma_client
    
    # Configure mock response
    mock_collection.query.return_value = {
        'ids': [['1']],
        'distances': [[0.1]],
        'metadatas': [[{
            'text': 'Test document',
            'source_name': 'test.pdf',
            'title': 'Python Guide'
        }]]
    }
    
    # Query with title filter
    results = db.query(
        query_embedding=np.array([0.1, 0.2, 0.3]),
        title='python'
    )
    
    # Verify the query was called with correct where clause
    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args[1]
    assert 'where' in call_kwargs
    assert call_kwargs['where']['title'] == {'$contains': 'python'}

def test_query_with_source_name_filter(db, mock_chroma_client):
    """Test querying documents with source name filter."""
    _, mock_collection = mock_chroma_client
    
    # Configure mock response
    mock_collection.query.return_value = {
        'ids': [['1']],
        'distances': [[0.1]],
        'metadatas': [[{
            'text': 'Test document',
            'source_name': 'test.pdf',
            'title': 'Test Document'
        }]]
    }
    
    # Query with source name filter
    results = db.query(
        query_embedding=np.array([0.1, 0.2, 0.3]),
        source_name='test.pdf'
    )
    
    # Verify the query was called with correct where clause
    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args[1]
    assert 'where' in call_kwargs
    assert call_kwargs['where']['source_name'] == 'test.pdf'

def test_search_titles(db, mock_chroma_client):
    """Test searching for documents by title."""
    _, mock_collection = mock_chroma_client
    
    # Configure mock response
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Python Programming Guide',
                'source_name': 'python_guide.pdf'
            },
            {
                'title': 'Learning Python Basics',
                'source_name': 'learning.pdf'
            },
            {
                'title': 'JavaScript Tutorial',
                'source_name': 'js.pdf'
            }
        ]
    }
    
    # Search for Python-related documents
    results = db.search_titles('python')
    
    # Verify results
    assert len(results) == 2
    assert any(r['title'] == 'Python Programming Guide' for r in results)
    assert any(r['title'] == 'Learning Python Basics' for r in results)
    assert not any(r['title'] == 'JavaScript Tutorial' for r in results)

def test_search_titles_case_insensitive(db, mock_chroma_client):
    """Test case-insensitive title search."""
    _, mock_collection = mock_chroma_client
    
    # Configure mock response
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Python Programming Guide',
                'source_name': 'python_guide.pdf'
            },
            {
                'title': 'PYTHON BASICS',
                'source_name': 'basics.pdf'
            }
        ]
    }
    
    # Search with different cases
    results_lower = db.search_titles('python')
    results_upper = db.search_titles('PYTHON')
    results_mixed = db.search_titles('PyThOn')
    
    # Verify all searches return the same results
    assert len(results_lower) == 2
    assert len(results_upper) == 2
    assert len(results_mixed) == 2

def test_search_titles_partial_match(db, mock_chroma_client):
    """Test partial matching in title search."""
    _, mock_collection = mock_chroma_client
    
    # Configure mock response
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Introduction to Programming',
                'source_name': 'intro.pdf'
            },
            {
                'title': 'Programming Basics',
                'source_name': 'basics.pdf'
            },
            {
                'title': 'Advanced Topics',
                'source_name': 'advanced.pdf'
            }
        ]
    }
    
    # Search with partial term
    results = db.search_titles('program')
    
    # Verify partial matches are found
    assert len(results) == 2
    assert any(r['title'] == 'Introduction to Programming' for r in results)
    assert any(r['title'] == 'Programming Basics' for r in results)

def test_search_titles_empty_query(db, mock_chroma_client):
    """Test title search with empty query."""
    _, mock_collection = mock_chroma_client
    
    # Configure mock response
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Test Document',
                'source_name': 'test.pdf'
            }
        ]
    }
    
    # Search with empty string
    results = db.search_titles('')
    
    # Verify no results are returned for empty query
    assert len(results) == 0

def test_search_titles_no_duplicates(db, mock_chroma_client):
    """Test that title search doesn't return duplicate documents."""
    _, mock_collection = mock_chroma_client
    
    # Configure mock response with duplicate titles (from different chunks)
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Python Guide',
                'source_name': 'guide.pdf'
            },
            {
                'title': 'Python Guide',  # Same document, different chunk
                'source_name': 'guide.pdf'
            }
        ]
    }
    
    # Search for Python documents
    results = db.search_titles('python')
    
    # Verify duplicates are removed
    assert len(results) == 1
    assert results[0]['title'] == 'Python Guide'
    assert results[0]['source_name'] == 'guide.pdf'
