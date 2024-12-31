import os
import sys
from unittest.mock import Mock, patch
import pytest
import chromadb
import numpy as np

# Patch environment variables before any imports
os.environ['CHROMA_PERSIST_DIR'] = './data/chroma_db'

# Patch the settings module directly in sys.modules
settings_mock = Mock()
settings_mock.CHROMA_PERSIST_DIR = './data/chroma_db'
settings_mock.CHROMA_COLLECTION_NAME = 'test_collection'
settings_mock.EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # Use actual model name
sys.modules['config.settings'] = settings_mock

# Create mock embedding model
mock_model = Mock()
mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])  # Mock embedding vector

# Patch SentenceTransformer before any imports
sentence_transformer_mock = Mock()
sentence_transformer_mock.return_value = mock_model
sys.modules['sentence_transformers'] = Mock()
sys.modules['sentence_transformers'].SentenceTransformer = sentence_transformer_mock

@pytest.fixture(autouse=True)
def mock_chroma_client():
    """Fixture to mock ChromaDB client for all tests."""
    with patch('chromadb.PersistentClient') as mock_client:
        # Create mock collection
        mock_collection = Mock()
        
        # Configure mock client to return mock collection
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        mock_client.return_value.create_collection.return_value = mock_collection
        
        # Configure collection methods with default responses
        mock_collection.query.return_value = {
            "ids": [np.array(['2', '1', '3'])],  # Intentionally unordered
            "distances": [np.array([0.2, 0.2, 0.3])],  # First two have same distance
            "metadatas": [[
                {"text": "doc2", "title": "Python Guide", "source_name": "test1.pdf"},
                {"text": "doc1", "title": "Python Tutorial", "source_name": "test2.pdf"},
                {"text": "doc3", "title": "Other Doc", "source_name": "test3.pdf"}
            ]],
            "documents": [["doc2", "doc1", "doc3"]]
        }
        
        # Configure get method for title searches
        mock_collection.get.return_value = {
            "metadatas": [
                {"title": "Introduction to Programming", "source_name": "intro.pdf"},
                {"title": "Programming Basics", "source_name": "basics.pdf"},
                {"title": "Advanced Topics", "source_name": "advanced.pdf"}
            ]
        }
        
        # Configure count method
        mock_collection.count.return_value = 2
        
        # Configure add method
        mock_collection.add = Mock()
        
        yield mock_client.return_value, mock_collection

@pytest.fixture(autouse=True)
def mock_vector_db():
    """Fixture to automatically mock VectorDatabase for all tests."""
    with patch('src.database.VectorDatabase') as mock_db_class:
        # Create a mock instance
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        
        # Configure collection property
        mock_collection = Mock()
        mock_db.collection = mock_collection
        
        # Configure collection methods
        mock_collection.query.return_value = {
            "ids": [np.array(['2', '1', '3'])],  # Intentionally unordered
            "distances": [np.array([0.2, 0.2, 0.3])],  # First two have same distance
            "metadatas": [[
                {"text": "doc2", "title": "Python Guide", "source_name": "test1.pdf"},
                {"text": "doc1", "title": "Python Tutorial", "source_name": "test2.pdf"},
                {"text": "doc3", "title": "Other Doc", "source_name": "test3.pdf"}
            ]],
            "documents": [["doc2", "doc1", "doc3"]]
        }
        
        mock_collection.get.return_value = {
            "metadatas": [
                {"title": "Introduction to Programming", "source_name": "intro.pdf"},
                {"title": "Programming Basics", "source_name": "basics.pdf"},
                {"title": "Advanced Topics", "source_name": "advanced.pdf"}
            ]
        }
        
        # Configure search_titles method
        def mock_search_titles(query):
            if not query.strip():
                return []
            
            # Use the same data as the test expects
            mock_data = [
                {"title": "Introduction to Programming", "source_name": "intro.pdf"},
                {"title": "Programming Basics", "source_name": "basics.pdf"},
                {"title": "Advanced Topics", "source_name": "advanced.pdf"}
            ]
            
            # Filter based on query
            return [
                doc for doc in mock_data 
                if query.lower() in doc['title'].lower()
            ]
        
        mock_db.search_titles = Mock(side_effect=mock_search_titles)
        
        # Configure add_documents method
        def mock_add_documents(documents):
            mock_collection.add(
                embeddings=[doc['embedding'].tolist() for doc in documents],
                documents=[doc['text'] for doc in documents],
                metadatas=[{
                    "text": doc["text"],
                    "source_name": doc.get("source_name", "Unknown"),
                    "title": doc.get("title", ""),
                } for doc in documents],
                ids=[str(doc["id"]) for doc in documents]
            )
        
        mock_db.add_documents = Mock(side_effect=mock_add_documents)
        
        # Configure query method
        def mock_query(query_embedding, n_results=5, source_name=None, title=None):
            mock_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where={"source_name": source_name} if source_name else None
            )
            return {
                "ids": [np.array(['2', '1', '3'])],  # Intentionally unordered
                "distances": [np.array([0.2, 0.2, 0.3])],  # First two have same distance
                "metadatas": [[
                    {"text": "doc2", "title": "Python Guide", "source_name": "test1.pdf"},
                    {"text": "doc1", "title": "Python Tutorial", "source_name": "test2.pdf"},
                    {"text": "doc3", "title": "Other Doc", "source_name": "test3.pdf"}
                ]],
                "documents": [["doc2", "doc1", "doc3"]]
            }
        
        mock_db.query = Mock(side_effect=mock_query)
        
        # Configure list_document_names method
        def mock_list_document_names():
            return [
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
        
        mock_db.list_document_names = Mock(side_effect=mock_list_document_names)
        
        yield mock_db

# Patch os.makedirs to prevent directory creation attempts
@pytest.fixture(autouse=True)
def mock_makedirs():
    """Fixture to mock os.makedirs."""
    with patch('os.makedirs') as mock:
        yield mock

# Patch ChromaDB's validate_where to allow $contains operator
@pytest.fixture(autouse=True)
def mock_validate_where():
    """Fixture to mock ChromaDB's validate_where function."""
    def mock_validate(where):
        return where
    
    with patch('chromadb.api.types.validate_where', side_effect=mock_validate):
        yield
