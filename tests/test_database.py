import pytest
from src.database import VectorDatabase
import numpy as np

def test_init(mock_chroma_client):
    """Test database initialization."""
    mock_client, mock_collection = mock_chroma_client
    
    db = VectorDatabase()
    
    mock_client.get_or_create_collection.assert_called_once()
    assert db.collection == mock_collection

def test_add_documents_with_cleanup(mock_chroma_client):
    """Test adding documents to the database."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()  # Use actual implementation
    
    documents = [
        {
            'id': 1,
            'text': 'Test document 1',
            'embedding': np.array([0.1, 0.2, 0.3]),
            'source_name': 'test1.pdf',
            'title': 'Test Document 1',
            'file_type': 'pdf',
            'section_type': 'content'
        },
        {
            'id': 2,
            'text': 'Test document 2',
            'embedding': np.array([0.4, 0.5, 0.6]),
            'source_name': 'test2.docx',
            'title': 'Test Document 2',
            'file_type': 'docx',
            'section_type': 'content'
        }
    ]
    
    # Configure mock to return existing documents
    mock_collection.get.return_value = {
        'ids': ['old1', 'old2'],
        'metadatas': [
            {'source_name': 'test1.pdf'},
            {'source_name': 'test1.pdf'}
        ],
        'documents': ['Old doc 1', 'Old doc 2']
    }
    
    db.add_documents(documents)
    
    # Verify delete was called for existing documents
    mock_collection.delete.assert_called_once_with(ids=['old1', 'old2'])
    
    # Verify add was called for new documents
    assert mock_collection.add.call_count == 2
    
    # Verify first call
    first_call = mock_collection.add.call_args_list[0][1]
    assert first_call['embeddings'] == [[0.1, 0.2, 0.3]]
    assert first_call['documents'] == ['Test document 1']
    assert first_call['ids'] == ['1']
    assert first_call['metadatas'][0]['source_name'] == 'test1.pdf'
    
    # Verify second call
    second_call = mock_collection.add.call_args_list[1][1]
    assert second_call['embeddings'] == [[0.4, 0.5, 0.6]]
    assert second_call['documents'] == ['Test document 2']
    assert second_call['ids'] == ['2']
    assert second_call['metadatas'][0]['source_name'] == 'test2.docx'

def test_add_documents_with_inconsistent_chunks(mock_chroma_client):
    """Test adding documents with inconsistent chunk counts."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()
    
    # Create documents with inconsistent total_chunks
    documents = [
        {
            'id': 1,
            'text': 'Test document 1',
            'embedding': np.array([0.1, 0.2, 0.3]),
            'source_name': 'test.pdf',
            'title': 'Test Document',
            'chunk_index': 0,
            'total_chunks': 3  # Wrong total
        },
        {
            'id': 2,
            'text': 'Test document 2',
            'embedding': np.array([0.4, 0.5, 0.6]),
            'source_name': 'test.pdf',
            'title': 'Test Document',
            'chunk_index': 1,
            'total_chunks': 2  # Inconsistent with first chunk
        }
    ]
    
    # Configure mock to return the added documents
    mock_collection.get.return_value = {
        'ids': ['1', '2'],
        'metadatas': [
            {
                'source_name': 'test.pdf',
                'chunk_index': 0,
                'total_chunks': 3
            },
            {
                'source_name': 'test.pdf',
                'chunk_index': 1,
                'total_chunks': 2
            }
        ],
        'documents': ['Test document 1', 'Test document 2']
    }
    
    # Adding documents with inconsistent chunks should raise an error
    with pytest.raises(ValueError) as exc_info:
        db.add_documents(documents)
    assert "total_chunks mismatch" in str(exc_info.value)

def test_query_with_title_filter(mock_chroma_client):
    """Test querying documents with title filter."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()  # Use actual implementation
    
    # Configure mock response with complete metadata
    mock_collection.query.return_value = {
        'ids': [['1']],
        'distances': [[0.1]],
        'metadatas': [[{
            'text': 'Test document',
            'source_name': 'test.pdf',
            'title': 'Python Guide',
            'file_type': 'pdf',
            'section_type': 'content'
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

def test_query_with_source_names_filter(mock_chroma_client):
    """Test querying documents with source names filter."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()  # Use actual implementation
    
    # Configure mock response with complete metadata
    mock_collection.query.return_value = {
        'ids': [['1', '2']],
        'distances': [[0.1, 0.2]],
        'metadatas': [[
            {
                'text': 'Test document 1',
                'source_name': 'test1.pdf',
                'title': 'Test Document 1',
                'file_type': 'pdf',
                'section_type': 'content'
            },
            {
                'text': 'Test document 2',
                'source_name': 'test2.pdf',
                'title': 'Test Document 2',
                'file_type': 'pdf',
                'section_type': 'content'
            }
        ]]
    }
    
    # Query with multiple source names
    source_names = ['test1.pdf', 'test2.pdf']
    results = db.query(
        query_embedding=np.array([0.1, 0.2, 0.3]),
        source_names=source_names
    )
    
    # Verify the query was called with correct where clause using $in operator
    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args[1]
    assert 'where' in call_kwargs
    assert call_kwargs['where']['source_name'] == {'$in': source_names}

def test_query_with_source_names_and_title(mock_chroma_client):
    """Test querying documents with both source names and title filters."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()  # Use actual implementation
    
    # Configure mock response with complete metadata
    mock_collection.query.return_value = {
        'ids': [['1']],
        'distances': [[0.1]],
        'metadatas': [[{
            'text': 'Test document',
            'source_name': 'test1.pdf',
            'title': 'Python Guide',
            'file_type': 'pdf',
            'section_type': 'content'
        }]]
    }
    
    # Query with both source names and title filters
    source_names = ['test1.pdf', 'test2.pdf']
    results = db.query(
        query_embedding=np.array([0.1, 0.2, 0.3]),
        source_names=source_names,
        title='python'
    )
    
    # Verify the query was called with correct where clause combining both filters
    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args[1]
    assert 'where' in call_kwargs
    assert call_kwargs['where']['source_name'] == {'$in': source_names}
    assert call_kwargs['where']['title'] == {'$contains': 'python'}

def test_search_titles(mock_chroma_client):
    """Test searching for documents by title."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()  # Use actual implementation
    
    # Configure mock response with complete metadata
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Python Programming Guide',
                'source_name': 'python_guide.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
            },
            {
                'title': 'Learning Python Basics',
                'source_name': 'learning.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
            },
            {
                'title': 'JavaScript Tutorial',
                'source_name': 'js.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
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
    
    # Verify metadata fields are preserved
    required_fields = ['source_name', 'title', 'file_type', 'section_type']
    for result in results:
        for field in required_fields:
            assert field in result

def test_search_titles_case_insensitive(mock_chroma_client):
    """Test case-insensitive title search."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()  # Use actual implementation
    
    # Configure mock response with complete metadata
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Python Programming Guide',
                'source_name': 'python_guide.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
            },
            {
                'title': 'PYTHON BASICS',
                'source_name': 'basics.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
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
    
    # Verify metadata fields
    required_fields = ['source_name', 'title', 'file_type', 'section_type']
    for result in results_lower:
        for field in required_fields:
            assert field in result

def test_search_titles_partial_match(mock_chroma_client):
    """Test partial matching in title search."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()  # Use actual implementation
    
    # Configure mock response with complete metadata
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Introduction to Programming',
                'source_name': 'intro.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
            },
            {
                'title': 'Programming Basics',
                'source_name': 'basics.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
            },
            {
                'title': 'Advanced Topics',
                'source_name': 'advanced.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
            }
        ]
    }
    
    # Search with partial term
    results = db.search_titles('program')
    
    # Verify partial matches are found
    assert len(results) == 2
    assert any(r['title'] == 'Introduction to Programming' for r in results)
    assert any(r['title'] == 'Programming Basics' for r in results)
    
    # Verify metadata fields
    required_fields = ['source_name', 'title', 'file_type', 'section_type']
    for result in results:
        for field in required_fields:
            assert field in result

def test_search_titles_empty_query(mock_chroma_client):
    """Test title search with empty query."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()  # Use actual implementation
    
    # Configure mock response with complete metadata
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Test Document',
                'source_name': 'test.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
            }
        ]
    }
    
    # Search with empty string
    results = db.search_titles('')
    
    # Verify no results are returned for empty query
    assert len(results) == 0

def test_search_titles_no_duplicates(mock_chroma_client):
    """Test that title search doesn't return duplicate documents."""
    _, mock_collection = mock_chroma_client
    
    db = VectorDatabase()  # Use actual implementation
    
    # Configure mock response with duplicate titles (from different chunks)
    mock_collection.get.return_value = {
        'metadatas': [
            {
                'title': 'Python Guide',
                'source_name': 'guide.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
            },
            {
                'title': 'Python Guide',  # Same document, different chunk
                'source_name': 'guide.pdf',
                'file_type': 'pdf',
                'section_type': 'content'
            }
        ]
    }
    
    # Search for Python documents
    results = db.search_titles('python')
    
    # Verify duplicates are removed
    assert len(results) == 1
    assert results[0]['title'] == 'Python Guide'
    assert results[0]['source_name'] == 'guide.pdf'
    
    # Verify metadata fields
    required_fields = ['source_name', 'title', 'file_type', 'section_type']
    for result in results:
        for field in required_fields:
            assert field in result
