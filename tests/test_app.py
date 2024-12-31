import unittest
import numpy as np
from unittest.mock import Mock, patch
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

    @patch('src.embedding.EmbeddingGenerator.generate_embeddings')
    def test_index_documents_ordering(self, mock_generate_embeddings):
        """Test document indexing maintains deterministic order."""
        # Setup mock embeddings
        mock_generate_embeddings.return_value = np.array([[0.1, 0.2, 0.3]])

        documents = [
            {
                "id": "2",
                "text": "text2",
                "source_file": "doc2"
            },
            {
                "id": "1",
                "text": "text1",
                "source_file": "doc1"
            }
        ]

        with patch.object(self.app.vector_db, 'add_documents') as mock_add:
            self.app.index_documents(documents)
            
            # Verify documents were sorted by ID before adding to database
            added_docs = mock_add.call_args[0][0]
            self.assertEqual(added_docs[0]['id'], "1")
            self.assertEqual(added_docs[1]['id'], "2")

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
        # Test indexing error
        with patch('src.embedding.EmbeddingGenerator.generate_embeddings', side_effect=Exception("Embedding error")):
            with self.assertRaises(Exception) as context:
                self.app.index_documents([{"id": "1", "text": "test"}])
            self.assertIn("Embedding error", str(context.exception))

        # Test query error
        with patch('src.embedding.EmbeddingGenerator.generate_embeddings', side_effect=Exception("Query error")):
            with self.assertRaises(Exception) as context:
                self.app.query_documents("test query")
            self.assertIn("Query error", str(context.exception))

if __name__ == '__main__':
    unittest.main()
