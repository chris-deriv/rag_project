import unittest
import numpy as np
from unittest.mock import Mock, patch
from src.search import SearchEngine

class TestSearchEngine(unittest.TestCase):
    @patch('sentence_transformers.SentenceTransformer')
    def setUp(self, mock_transformer):
        """Set up test fixtures before each test method."""
        # Mock the SentenceTransformer model
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        self.search_engine = SearchEngine()

    def test_parse_query(self):
        """Test query parsing functionality."""
        # Test basic query parsing
        query = "  What is Machine Learning?  "
        result = self.search_engine.parse_query(query)
        self.assertEqual(result['original_query'], query)
        self.assertEqual(result['processed_query'], "what is machine learning?")

        # Test empty query
        empty_query = "   "
        result = self.search_engine.parse_query(empty_query)
        self.assertEqual(result['processed_query'], "")

        # Test query with special characters
        special_query = "What's the meaning of life? (42)"
        result = self.search_engine.parse_query(special_query)
        self.assertEqual(result['processed_query'], "what's the meaning of life? (42)")

    @patch('src.embedding.EmbeddingGenerator.generate_embeddings')
    def test_generate_query_embedding(self, mock_generate_embeddings):
        """Test query embedding generation."""
        # Test successful embedding generation
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_generate_embeddings.return_value = np.array([mock_embedding])
        
        query = "test query"
        result = self.search_engine.generate_query_embedding(query)
        
        mock_generate_embeddings.assert_called_once_with([query])
        np.testing.assert_array_equal(result, mock_embedding)

        # Test embedding generation with empty query
        mock_generate_embeddings.reset_mock()
        mock_generate_embeddings.return_value = np.array([[0.0, 0.0, 0.0]])
        
        empty_query = ""
        result = self.search_engine.generate_query_embedding(empty_query)
        mock_generate_embeddings.assert_called_once_with([empty_query])
        self.assertEqual(result.shape, (3,))

        # Test error handling
        mock_generate_embeddings.reset_mock()
        mock_generate_embeddings.side_effect = Exception("Embedding generation failed")
        
        with self.assertRaises(Exception) as context:
            self.search_engine.generate_query_embedding("test query")
        self.assertTrue("Embedding generation failed" in str(context.exception))

    @patch('src.database.VectorDatabase.query')
    def test_perform_similarity_search(self, mock_query):
        """Test similarity search functionality."""
        # Test successful search with deterministic ordering
        mock_results = {
            'ids': [['2', '1', '3']],  # Intentionally unordered
            'distances': [[0.2, 0.2, 0.3]],  # Note: first two have same distance
            'metadatas': [[{'text': 'doc2'}, {'text': 'doc1'}, {'text': 'doc3'}]]
        }
        mock_query.return_value = mock_results
        
        query_embedding = np.array([0.1, 0.2, 0.3])
        result = self.search_engine.perform_similarity_search(query_embedding, n_results=3)
        
        # Verify deterministic ordering (by distance first, then ID)
        self.assertEqual(result['ids'][0], ['1', '2', '3'])
        self.assertEqual(result['distances'][0], [0.2, 0.2, 0.3])
        self.assertEqual([m['text'] for m in result['metadatas'][0]], ['doc1', 'doc2', 'doc3'])

        # Test search with no results
        mock_query.reset_mock()
        mock_query.return_value = {'ids': [[]], 'distances': [[]], 'metadatas': [[]]}
        
        result = self.search_engine.perform_similarity_search(query_embedding, n_results=2)
        self.assertEqual(len(result['ids'][0]), 0)

        # Test error handling
        mock_query.reset_mock()
        mock_query.side_effect = Exception("Database query failed")
        
        with self.assertRaises(Exception) as context:
            self.search_engine.perform_similarity_search(query_embedding)
        self.assertTrue("Database query failed" in str(context.exception))

    def test_get_cache_key(self):
        """Test cache key generation."""
        query = "  test query  "
        text = "  sample text  "
        key = self.search_engine._get_cache_key(query, text)
        expected_key = "test query|||sample text"
        self.assertEqual(key, expected_key)

        # Test with different whitespace
        key2 = self.search_engine._get_cache_key("test query", "sample text")
        self.assertEqual(key, key2)

    @patch('src.chatbot.Chatbot.generate_response')
    def test_rerank_results(self, mock_generate_response):
        """Test results reranking functionality with caching and deterministic ordering."""
        # Test successful reranking
        search_results = {
            'ids': [['2', '1']],  # Intentionally unordered
            'distances': [[0.2, 0.2]],  # Same distances
            'metadatas': [[
                {'text': 'second document'},
                {'text': 'first document'}
            ]]
        }
        
        mock_generate_response.return_value = "8\n8"  # Same relevance scores
        
        query = "test query"
        results = self.search_engine.rerank_results(query, search_results)
        
        # Verify deterministic ordering by ID when scores are equal
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], '1')
        self.assertEqual(results[1]['id'], '2')

        # Test caching
        # Second call with same query and texts should not call LLM
        mock_generate_response.reset_mock()
        results2 = self.search_engine.rerank_results(query, search_results)
        mock_generate_response.assert_not_called()
        self.assertEqual(results, results2)

        # Test with new texts (should call LLM only for new texts)
        new_results = {
            'ids': [['3', '1']],  # '1' is cached, '3' is new
            'distances': [[0.2, 0.2]],
            'metadatas': [[
                {'text': 'third document'},
                {'text': 'first document'}
            ]]
        }
        mock_generate_response.return_value = "7"  # Only one score needed for the new text
        results3 = self.search_engine.rerank_results(query, new_results)
        self.assertEqual(len(mock_generate_response.mock_calls), 1)

        # Test single result case
        single_result = {
            'ids': [['1']],
            'distances': [[0.1]],
            'metadatas': [[{'text': 'single document'}]]
        }
        results = self.search_engine.rerank_results(query, single_result)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], '1')
        self.assertEqual(results[0]['text'], 'single document')

        # Test empty results handling
        empty_results = {
            'ids': [[]],
            'distances': [[]],
            'metadatas': [[]]
        }
        results = self.search_engine.rerank_results(query, empty_results)
        self.assertEqual(len(results), 0)

    @patch.object(SearchEngine, 'parse_query')
    @patch.object(SearchEngine, 'generate_query_embedding')
    @patch.object(SearchEngine, 'perform_similarity_search')
    @patch.object(SearchEngine, 'rerank_results')
    def test_search(self, mock_rerank, mock_search, mock_embed, mock_parse):
        """Test complete search pipeline."""
        # Test successful search
        mock_parse.return_value = {'processed_query': 'test query'}
        mock_embed.return_value = np.array([0.1, 0.2, 0.3])
        mock_search.return_value = {
            'ids': [['1', '2']],
            'distances': [[0.1, 0.2]],
            'metadatas': [[{'text': 'doc1'}, {'text': 'doc2'}]]
        }
        mock_rerank.return_value = [
            {'id': '1', 'text': 'doc1', 'combined_score': 0.9},
            {'id': '2', 'text': 'doc2', 'combined_score': 0.8}
        ]
        
        query = "test query"
        results = self.search_engine.search(query, n_results=2)
        
        mock_parse.assert_called_once_with(query)
        mock_embed.assert_called_once()
        mock_search.assert_called_once()
        mock_rerank.assert_called_once()
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], '1')
        self.assertEqual(results[0]['text'], 'doc1')

        # Test error handling in pipeline
        mock_parse.reset_mock()
        mock_embed.reset_mock()
        mock_search.reset_mock()
        mock_rerank.reset_mock()
        
        mock_embed.side_effect = Exception("Embedding failed")
        
        with self.assertRaises(Exception) as context:
            self.search_engine.search("test query")
        self.assertTrue("Error performing search" in str(context.exception))

        # Test with invalid n_results
        with self.assertRaises(Exception) as context:
            self.search_engine.search("test query", n_results=0)
        self.assertTrue("Error performing search" in str(context.exception))

    def test_edge_cases(self):
        """Test various edge cases and input validation."""
        # Test with None query
        with self.assertRaises(Exception):
            self.search_engine.search(None)

        # Test with very long query
        long_query = "a" * 1000
        result = self.search_engine.parse_query(long_query)
        self.assertEqual(result['processed_query'], long_query.lower())

        # Test with special characters
        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/~`"
        result = self.search_engine.parse_query(special_chars)
        self.assertEqual(result['processed_query'], special_chars.lower())

        # Test with multiple spaces and newlines
        messy_query = "  Hello  \n  World  \t  ! "
        result = self.search_engine.parse_query(messy_query)
        self.assertEqual(result['processed_query'], "hello  \n  world  \t  !")

if __name__ == '__main__':
    unittest.main()
