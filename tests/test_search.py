import unittest
import numpy as np
from unittest.mock import Mock, patch
from src.search import SearchEngine

@patch('src.database.VectorDatabase')
class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.search_engine = SearchEngine()

    def test_parse_query(self, _):
        """Test query parsing functionality."""
        # Test basic query
        result = self.search_engine.parse_query("test query")
        self.assertEqual(result["original_query"], "test query")
        self.assertEqual(result["processed_query"], "test query")

        # Test query with whitespace
        result = self.search_engine.parse_query("  test  query  ")
        self.assertEqual(result["processed_query"], "test  query")

        # Test query with case
        result = self.search_engine.parse_query("TEST QUERY")
        self.assertEqual(result["processed_query"], "test query")

        # Test invalid query
        with self.assertRaises(Exception):
            self.search_engine.parse_query(None)

    def test_generate_query_embedding(self, _):
        """Test query embedding generation."""
        # Test successful embedding generation
        mock_embeddings = np.array([0.1, 0.2, 0.3])
        with patch('src.embedding.EmbeddingGenerator.generate_embeddings', return_value=np.array([mock_embeddings])):
            result = self.search_engine.generate_query_embedding("test query")
            np.testing.assert_array_equal(result, mock_embeddings)

        # Test embedding error
        with patch('src.embedding.EmbeddingGenerator.generate_embeddings', side_effect=Exception("Embedding error")):
            with self.assertRaises(Exception) as context:
                self.search_engine.generate_query_embedding("test query")
            self.assertIn("Embedding error", str(context.exception))

    def test_perform_similarity_search(self, mock_db_class):
        """Test similarity search functionality."""
        # Test successful search with deterministic ordering
        mock_results = {
            'ids': [['2', '1', '3']],  # Intentionally unordered
            'distances': [[0.2, 0.2, 0.3]],  # Note: first two have same distance
            'metadatas': [[
                {
                    'text': 'doc2',
                    'source_name': 'doc2.pdf',
                    'title': 'Document 2',
                    'file_type': 'pdf',
                    'section_type': 'content'
                },
                {
                    'text': 'doc1',
                    'source_name': 'doc1.pdf',
                    'title': 'Document 1',
                    'file_type': 'pdf',
                    'section_type': 'content'
                },
                {
                    'text': 'doc3',
                    'source_name': 'doc3.pdf',
                    'title': 'Document 3',
                    'file_type': 'pdf',
                    'section_type': 'content'
                }
            ]]
        }
        # Create a new search engine to use the mocked database
        self.search_engine = SearchEngine()

        # Patch the query method after creating the search engine
        with patch.object(self.search_engine.vector_db, 'query', return_value=mock_results):
            query_embedding = np.array([0.1, 0.2, 0.3])
            result = self.search_engine.perform_similarity_search(query_embedding, n_results=3)

        # Verify deterministic ordering (by distance first, then ID)
        self.assertEqual(result['ids'][0], ['1', '2', '3'])
        self.assertEqual(result['distances'][0], [0.2, 0.2, 0.3])

        # Verify metadata fields are preserved
        required_fields = ['source_name', 'title', 'file_type', 'section_type']
        for metadata in result['metadatas'][0]:
            for field in required_fields:
                self.assertIn(field, metadata)

    def test_search_error_handling(self, mock_db_class):
        """Test error handling in search pipeline."""
        # Test invalid n_results
        with self.assertRaises(Exception) as context:
            self.search_engine.perform_similarity_search(np.array([0.1, 0.2, 0.3]), n_results=0)
        self.assertIn("n_results must be a positive integer", str(context.exception))

        # Test database error
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        # Create a new search engine to use the mocked database
        self.search_engine = SearchEngine()

        # Patch the query method after creating the search engine
        with patch.object(self.search_engine.vector_db, 'query', side_effect=Exception("Database error")):
            with self.assertRaises(Exception) as context:
                self.search_engine.perform_similarity_search(np.array([0.1, 0.2, 0.3]))
            self.assertIn("Database error", str(context.exception))

if __name__ == '__main__':
    unittest.main()
