"""Test application functionality."""
import unittest
import numpy as np
from unittest.mock import Mock, patch, PropertyMock
from src.app import RAGApplication

class TestRAGApplication(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = RAGApplication()

    def test_sort_contexts(self):
        """Test context sorting is deterministic."""
        contexts = [
            {
                "text": "text3",
                "source": "doc2",
                "title": "title2",
                "chunk_index": 1
            },
            {
                "text": "text1",
                "source": "doc1",
                "title": "title1",
                "chunk_index": 0
            },
            {
                "text": "text2",
                "source": "doc1",
                "title": "title1",
                "chunk_index": 1
            }
        ]

        # Test sorting is consistent
        sorted1 = self.app._sort_contexts(contexts)
        sorted2 = self.app._sort_contexts(contexts[::-1])  # Reverse order
        self.assertEqual(sorted1, sorted2)

        # Verify sort order
        self.assertEqual(sorted1[0]["text"], "text1")  # doc1, title1, index 0
        self.assertEqual(sorted1[1]["text"], "text2")  # doc1, title1, index 1
        self.assertEqual(sorted1[2]["text"], "text3")  # doc2, title2, index 1

    def test_balance_results_single_source(self):
        """Test result balancing with a single source."""
        results = [
            {
                'text': 'text1',
                'metadata': {'source_name': 'doc1.pdf'},
                'combined_score': 0.9
            },
            {
                'text': 'text2',
                'metadata': {'source_name': 'doc1.pdf'},
                'combined_score': 0.8
            }
        ]
        source_names = ['doc1.pdf']
        
        balanced = self.app._balance_results(results, source_names)
        self.assertEqual(len(balanced), 2)
        self.assertEqual(balanced, results)  # Should remain unchanged

    def test_balance_results_multiple_sources(self):
        """Test result balancing across multiple sources."""
        results = [
            {
                'text': 'text1',
                'metadata': {'source_name': 'doc1.pdf'},
                'combined_score': 0.9
            },
            {
                'text': 'text2',
                'metadata': {'source_name': 'doc1.pdf'},
                'combined_score': 0.8
            },
            {
                'text': 'text3',
                'metadata': {'source_name': 'doc2.pdf'},
                'combined_score': 0.7
            },
            {
                'text': 'text4',
                'metadata': {'source_name': 'doc2.pdf'},
                'combined_score': 0.6
            }
        ]
        source_names = ['doc1.pdf', 'doc2.pdf']
        
        balanced = self.app._balance_results(results, source_names)
        
        # Verify minimum representation from each source
        doc1_results = [r for r in balanced if r['metadata']['source_name'] == 'doc1.pdf']
        doc2_results = [r for r in balanced if r['metadata']['source_name'] == 'doc2.pdf']
        
        self.assertGreaterEqual(len(doc1_results), 2)
        self.assertGreaterEqual(len(doc2_results), 2)
        
        # Verify overall ordering by score within each source
        self.assertGreater(doc1_results[0]['combined_score'], doc1_results[1]['combined_score'])
        self.assertGreater(doc2_results[0]['combined_score'], doc2_results[1]['combined_score'])

    def test_balance_results_uneven_sources(self):
        """Test result balancing with uneven source distribution."""
        results = [
            {
                'text': 'text1',
                'metadata': {'source_name': 'doc1.pdf'},
                'combined_score': 0.9
            },
            {
                'text': 'text2',
                'metadata': {'source_name': 'doc1.pdf'},
                'combined_score': 0.8
            },
            {
                'text': 'text3',
                'metadata': {'source_name': 'doc1.pdf'},
                'combined_score': 0.7
            },
            {
                'text': 'text4',
                'metadata': {'source_name': 'doc2.pdf'},
                'combined_score': 0.6
            }
        ]
        source_names = ['doc1.pdf', 'doc2.pdf']
        
        balanced = self.app._balance_results(results, source_names)
        
        # Verify minimum representation from each source
        doc1_results = [r for r in balanced if r['metadata']['source_name'] == 'doc1.pdf']
        doc2_results = [r for r in balanced if r['metadata']['source_name'] == 'doc2.pdf']
        
        self.assertGreaterEqual(len(doc1_results), 2)
        self.assertGreaterEqual(len(doc2_results), 1)

    def test_query_documents_result_count_scaling(self):
        """Test that n_results scales with number of sources."""
        with patch.object(self.app.search_engine, 'search') as mock_search:
            with patch.object(self.app.chatbot, 'generate_response_with_sources', return_value="test response"):
                # Test with single source
                self.app.query_documents("test", source_names=["doc1.pdf"])
                self.assertEqual(mock_search.call_args[1]['n_results'], 5)  # Default
                
                # Test with multiple sources
                self.app.query_documents("test", source_names=["doc1.pdf", "doc2.pdf"])
                self.assertEqual(mock_search.call_args[1]['n_results'], 6)  # 2 sources * 3

    def test_index_documents_verification(self):
        """Test document indexing verifies documents in vector database."""
        documents = [
            {
                "source_name": "doc1.pdf",
                "title": "Document 1"
            },
            {
                "source_name": "doc2.pdf",
                "title": "Document 2"
            }
        ]

        # Mock document_store responses
        mock_doc_info = {
            'source_name': 'doc1.pdf',
            'title': 'Document 1',
            'chunk_count': 5,
            'total_chunks': 5
        }

        with patch('src.documents.document_store.get_document_info', return_value=mock_doc_info) as mock_get_info:
            self.app.index_documents(documents)
            
            # Verify document info was checked for each document
            self.assertEqual(mock_get_info.call_count, 2)
            mock_get_info.assert_any_call('doc1.pdf')
            mock_get_info.assert_any_call('doc2.pdf')

    def test_index_documents_missing_source_name(self):
        """Test indexing handles documents without source_name."""
        documents = [
            {
                "title": "Document 1"
            }
        ]

        with patch('src.documents.document_store.get_document_info') as mock_get_info:
            self.app.index_documents(documents)
            
            # Verify no document info check was attempted
            mock_get_info.assert_not_called()

    def test_index_documents_not_found(self):
        """Test indexing handles documents not found in vector database."""
        documents = [
            {
                "source_name": "doc1.pdf",
                "title": "Document 1"
            }
        ]

        # Mock document_store to return None (document not found)
        with patch('src.documents.document_store.get_document_info', return_value=None) as mock_get_info:
            self.app.index_documents(documents)
            
            # Verify document info was checked
            mock_get_info.assert_called_once_with('doc1.pdf')

    @patch('src.embedding.EmbeddingGenerator.generate_embeddings')
    def test_query_documents_deterministic(self, mock_generate_embeddings):
        """Test query processing is deterministic."""
        # Setup mock embeddings
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_generate_embeddings.return_value = np.array([mock_embedding])

        # Setup mock database results
        mock_results = {
            "metadatas": [[
                {
                    "text": "text2",
                    "source_name": "doc2",
                    "title": "title2",
                    "chunk_index": 0
                },
                {
                    "text": "text1",
                    "source_name": "doc1",
                    "title": "title1",
                    "chunk_index": 0
                }
            ]]
        }

        with patch.object(self.app.search_engine, 'search', return_value=[
            {
                'text': 'text1',
                'metadata': {
                    'text': 'text1',
                    'source_name': 'doc1',
                    'title': 'title1',
                    'chunk_index': 0
                }
            },
            {
                'text': 'text2',
                'metadata': {
                    'text': 'text2',
                    'source_name': 'doc2',
                    'title': 'title2',
                    'chunk_index': 0
                }
            }
        ]):
            with patch.object(self.app.chatbot, 'generate_response_with_sources', return_value="test response") as mock_generate:
                response = self.app.query_documents("test query")

                # Verify contexts were sorted before generating response
                contexts = mock_generate.call_args[0][0]
                self.assertEqual(contexts[0]["source"], "doc1")
                self.assertEqual(contexts[1]["source"], "doc2")

    @patch('src.embedding.EmbeddingGenerator.generate_embeddings')
    def test_query_documents_with_source_names(self, mock_generate_embeddings):
        """Test query processing with source names filter."""
        # Setup mock embeddings
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_generate_embeddings.return_value = np.array([mock_embedding])

        # Setup mock database results
        mock_results = {
            "metadatas": [[
                {
                    "text": "text1",
                    "source_name": "doc1.pdf",
                    "title": "title1",
                    "chunk_index": 0
                }
            ]]
        }

        with patch.object(self.app.search_engine, 'search', return_value=[{
            'text': 'text1',
            'metadata': {
                'text': 'text1',
                'source_name': 'doc1.pdf',
                'title': 'title1',
                'chunk_index': 0
            }
        }]) as mock_search:
            with patch.object(self.app.chatbot, 'generate_response_with_sources', return_value="test response"):
                source_names = ["doc1.pdf", "doc2.pdf"]
                response = self.app.query_documents("test query", source_names=source_names)

                # Verify source_names was passed to search
                mock_search.assert_called_once()
                call_kwargs = mock_search.call_args[1]
                self.assertEqual(call_kwargs['source_names'], source_names)

    @patch('src.embedding.EmbeddingGenerator.generate_embeddings')
    def test_query_documents_with_source_names_and_title(self, mock_generate_embeddings):
        """Test query processing with both source names and title filters."""
        # Setup mock embeddings
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_generate_embeddings.return_value = np.array([mock_embedding])

        # Setup mock database results
        mock_results = {
            "metadatas": [[
                {
                    "text": "text1",
                    "source_name": "doc1.pdf",
                    "title": "Python Guide",
                    "chunk_index": 0
                }
            ]]
        }

        with patch.object(self.app.search_engine, 'search', return_value=[{
            'text': 'text1',
            'metadata': {
                'text': 'text1',
                'source_name': 'doc1.pdf',
                'title': 'Python Guide',
                'chunk_index': 0
            }
        }]) as mock_search:
            with patch.object(self.app.chatbot, 'generate_response_with_sources', return_value="test response"):
                source_names = ["doc1.pdf", "doc2.pdf"]
                title = "python"
                response = self.app.query_documents(
                    "test query",
                    source_names=source_names,
                    title=title
                )

                # Verify both filters were passed to search
                mock_search.assert_called_once()
                call_kwargs = mock_search.call_args[1]
                self.assertEqual(call_kwargs['source_names'], source_names)
                self.assertEqual(call_kwargs['title'], title)

    def test_error_handling(self):
        """Test error handling in main operations."""
        # Test indexing error with invalid document
        self.app.index_documents([{"id": "1", "text": "test"}])  # Should log warning and skip

        # Test query error
        with patch('src.search.SearchEngine.search', side_effect=Exception("Query error")):
            with self.assertRaises(Exception) as context:
                self.app.query_documents("test query")
            self.assertIn("Query error", str(context.exception))

if __name__ == '__main__':
    unittest.main()
