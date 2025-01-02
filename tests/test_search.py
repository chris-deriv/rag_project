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

    def test_rerank_results_multi_source(self, _):
        """Test reranking with multiple sources."""
        mock_results = {
            'ids': [['1', '2', '3', '4']],
            'distances': [[0.2, 0.3, 0.2, 0.3]],
            'metadatas': [[
                {
                    'text': 'content from doc1',
                    'source_name': 'doc1.pdf'
                },
                {
                    'text': 'content from doc1 again',
                    'source_name': 'doc1.pdf'
                },
                {
                    'text': 'content from doc2',
                    'source_name': 'doc2.pdf'
                },
                {
                    'text': 'more content from doc2',
                    'source_name': 'doc2.pdf'
                }
            ]]
        }

        # Mock LLM response for relevance scores
        with patch.object(self.search_engine.chatbot, 'generate_response', return_value="8\n7\n9\n6"):
            reranked = self.search_engine.rerank_results("test query", mock_results)

            # Verify we got results from both sources
            source_names = set(r['metadata']['source_name'] for r in reranked)
            self.assertEqual(len(source_names), 2)
            self.assertIn('doc1.pdf', source_names)
            self.assertIn('doc2.pdf', source_names)

            # Verify scoring combines similarity and relevance
            first_result = reranked[0]
            self.assertIn('similarity_score', first_result)
            self.assertIn('relevance_score', first_result)
            self.assertIn('combined_score', first_result)

            # Verify cache is used for subsequent calls
            with patch.object(self.search_engine.chatbot, 'generate_response') as mock_generate:
                self.search_engine.rerank_results("test query", mock_results)
                mock_generate.assert_not_called()

    def test_rerank_results_cache_consistency(self, _):
        """Test reranking cache is consistent across different orderings."""
        mock_results1 = {
            'ids': [['1', '2']],
            'distances': [[0.2, 0.3]],
            'metadatas': [[
                {'text': 'first text', 'source_name': 'doc1.pdf'},
                {'text': 'second text', 'source_name': 'doc2.pdf'}
            ]]
        }

        mock_results2 = {
            'ids': [['2', '1']],
            'distances': [[0.3, 0.2]],
            'metadatas': [[
                {'text': 'second text', 'source_name': 'doc2.pdf'},
                {'text': 'first text', 'source_name': 'doc1.pdf'}
            ]]
        }

        with patch.object(self.search_engine.chatbot, 'generate_response', return_value="8\n7"):
            # First reranking should use LLM
            reranked1 = self.search_engine.rerank_results("test query", mock_results1)
            
            # Second reranking with different order should use cache
            with patch.object(self.search_engine.chatbot, 'generate_response') as mock_generate:
                reranked2 = self.search_engine.rerank_results("test query", mock_results2)
                mock_generate.assert_not_called()

            # Results should be consistent
            self.assertEqual(
                [r['text'] for r in reranked1],
                [r['text'] for r in reranked2]
            )

    def test_search_with_source_filtering(self, mock_db_class):
        """Test search with source name filtering."""
        mock_results = {
            'ids': [['1', '2']],
            'distances': [[0.2, 0.3]],
            'metadatas': [[
                {'text': 'doc1 content', 'source_name': 'doc1.pdf'},
                {'text': 'doc2 content', 'source_name': 'doc2.pdf'}
            ]]
        }

        # Create a new search engine to use the mocked database
        self.search_engine = SearchEngine()

        # Test source filtering is passed correctly
        with patch.object(self.search_engine.vector_db, 'query', return_value=mock_results) as mock_query:
            source_names = ['doc1.pdf', 'doc2.pdf']
            self.search_engine.perform_similarity_search(
                np.array([0.1, 0.2, 0.3]),
                source_names=source_names
            )
            
            # Verify source names were passed to query
            call_kwargs = mock_query.call_args[1]
            self.assertEqual(call_kwargs['source_names'], source_names)

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

    def test_rerank_results_llm_failure(self, _):
        """Test reranking fallback when LLM fails."""
        mock_results = {
            'ids': [['1', '2']],
            'distances': [[0.2, 0.3]],
            'metadatas': [[
                {'text': 'doc1 content', 'source_name': 'doc1.pdf'},
                {'text': 'doc2 content', 'source_name': 'doc2.pdf'}
            ]]
        }

        # Test LLM failure fallback
        with patch.object(self.search_engine.chatbot, 'generate_response', side_effect=Exception("LLM error")):
            reranked = self.search_engine.rerank_results("test query", mock_results)
            
            # Verify fallback to similarity scores
            self.assertEqual(reranked[0]['similarity_score'], 0.8)  # 1 - distance
            self.assertEqual(reranked[1]['similarity_score'], 0.7)
            
            # Verify results are still ordered
            self.assertGreater(
                reranked[0]['combined_score'],
                reranked[1]['combined_score']
            )

if __name__ == '__main__':
    unittest.main()
