import unittest
from unittest.mock import Mock, patch, call
from rag_backend.chatbot import Chatbot

class TestChatbot(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        with patch('openai.OpenAI'):  # Prevent actual OpenAI client creation
            self.chatbot = Chatbot()

    def test_get_cache_key(self):
        """Test cache key generation is consistent."""
        # Test basic key generation
        key1 = self.chatbot._get_cache_key("Test Context", "Test Query")
        key2 = self.chatbot._get_cache_key("Test Context", "Test Query")
        self.assertEqual(key1, key2)

        # Test whitespace normalization
        key3 = self.chatbot._get_cache_key("  Test   Context  ", "  Test   Query  ")
        self.assertEqual(key1, key3)

        # Test case normalization
        key4 = self.chatbot._get_cache_key("TEST CONTEXT", "TEST QUERY")
        self.assertEqual(key1, key4)

    def test_format_contexts_for_cache_with_source_grouping(self):
        """Test context formatting with source grouping and metadata."""
        contexts = [
            {
                "text": "Second chunk",
                "source": "doc1.pdf",
                "title": "Document 1",
                "chunk_index": 1,
                "total_chunks": 2
            },
            {
                "text": "First chunk",
                "source": "doc1.pdf",
                "title": "Document 1",
                "chunk_index": 0,
                "total_chunks": 2
            },
            {
                "text": "Content from doc2",
                "source": "doc2.pdf",
                "title": "Document 2",
                "chunk_index": 0,
                "total_chunks": 1
            }
        ]
        
        formatted = self.chatbot._format_contexts_for_cache(contexts)
        
        # Verify source grouping
        self.assertIn("Source: doc1.pdf", formatted)
        self.assertIn("Source: doc2.pdf", formatted)
        
        # Verify title inclusion
        self.assertIn("Title: Document 1", formatted)
        self.assertIn("Title: Document 2", formatted)
        
        # Verify chunk ordering within sources
        doc1_index = formatted.index("doc1.pdf")
        first_chunk_index = formatted.index("[Chunk 1/2]")
        second_chunk_index = formatted.index("[Chunk 2/2]")
        self.assertLess(first_chunk_index, second_chunk_index)
        
        # Verify content inclusion
        self.assertIn("First chunk", formatted)
        self.assertIn("Second chunk", formatted)
        self.assertIn("Content from doc2", formatted)

    def test_generate_response_with_sources_multi_document(self):
        """Test source-cited response generation with multiple documents."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response with sources"))]
        
        contexts = [
            {
                "text": "Content from first doc",
                "source": "doc1.pdf",
                "title": "Document 1",
                "chunk_index": 0,
                "total_chunks": 1
            },
            {
                "text": "Content from second doc",
                "source": "doc2.pdf",
                "title": "Document 2",
                "chunk_index": 0,
                "total_chunks": 1
            }
        ]
        
        with patch.object(self.chatbot.client.chat.completions, 'create', return_value=mock_response) as mock_create:
            response = self.chatbot.generate_response_with_sources(contexts, "test query")
            
            # Verify source overview was included
            call_args = mock_create.call_args[1]
            messages = call_args['messages']
            prompt = messages[1]['content']
            
            # Check source overview format
            self.assertIn("* [Source 1: doc1.pdf]", prompt)
            self.assertIn("* [Source 2: doc2.pdf]", prompt)
            
            # Check synthesis instructions
            self.assertIn("synthesizes information across all sources", prompt)
            self.assertIn("Compare and contrast information", prompt)
            
            # Verify source details format
            self.assertIn("Source: doc1.pdf", prompt)
            self.assertIn("Title: Document 1", prompt)
            self.assertIn("Content from first doc", prompt)
            self.assertIn("Source: doc2.pdf", prompt)
            self.assertIn("Title: Document 2", prompt)
            self.assertIn("Content from second doc", prompt)

    def test_format_contexts_empty_metadata(self):
        """Test context formatting handles missing metadata gracefully."""
        contexts = [
            {
                "text": "Content without metadata"
            }
        ]
        
        formatted = self.chatbot._format_contexts_for_cache(contexts)
        
        # Verify default values are used
        self.assertIn("Source: Unknown", formatted)
        self.assertIn("Title: Untitled", formatted)
        self.assertIn("[Chunk 1/1]", formatted)
        self.assertIn("Content without metadata", formatted)

    def test_generate_response_caching(self):
        """Test response caching behavior."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        
        with patch.object(self.chatbot.client.chat.completions, 'create', return_value=mock_response) as mock_create:
            # First call should use API
            response1 = self.chatbot.generate_response("test context", "test query")
            self.assertEqual(response1, "Test response")
            mock_create.assert_called_once()

            # Second call should use cache
            mock_create.reset_mock()
            response2 = self.chatbot.generate_response("test context", "test query")
            self.assertEqual(response2, "Test response")
            mock_create.assert_not_called()

            # Different query should use API again
            response3 = self.chatbot.generate_response("test context", "different query")
            self.assertEqual(response3, "Test response")
            mock_create.assert_called_once()

    def test_generate_response_with_sources_caching(self):
        """Test source-cited response caching behavior."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response with sources"))]
        
        with patch.object(self.chatbot.client.chat.completions, 'create', return_value=mock_response) as mock_create:
            contexts = [
                {
                    "text": "Context 1",
                    "source": "doc1.pdf",
                    "title": "Document 1",
                    "chunk_index": 0,
                    "total_chunks": 1
                },
                {
                    "text": "Context 2",
                    "source": "doc2.pdf",
                    "title": "Document 2",
                    "chunk_index": 0,
                    "total_chunks": 1
                }
            ]

            # First call should use API
            response1 = self.chatbot.generate_response_with_sources(contexts, "test query")
            self.assertEqual(response1, "Test response with sources")
            mock_create.assert_called_once()

            # Second call should use cache
            mock_create.reset_mock()
            response2 = self.chatbot.generate_response_with_sources(contexts, "test query")
            self.assertEqual(response2, "Test response with sources")
            mock_create.assert_not_called()

            # Same contexts in different order should use cache
            mock_create.reset_mock()
            response3 = self.chatbot.generate_response_with_sources(contexts[::-1], "test query")
            self.assertEqual(response3, "Test response with sources")
            mock_create.assert_not_called()

    def test_api_parameters(self):
        """Test API is called with correct parameters for comprehensive output."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        
        with patch.object(self.chatbot.client.chat.completions, 'create', return_value=mock_response) as mock_create:
            self.chatbot.generate_response("test context", "test query")
            
            # Verify parameters for comprehensive responses
            call_kwargs = mock_create.call_args[1]
            self.assertEqual(call_kwargs['temperature'], 0.3)
            self.assertEqual(call_kwargs['seed'], 42)
            self.assertEqual(call_kwargs['max_tokens'], 1000)

    def test_error_handling(self):
        """Test error handling in response generation."""
        with patch.object(self.chatbot.client.chat.completions, 'create', side_effect=Exception("API error")):
            with self.assertRaises(Exception) as context:
                self.chatbot.generate_response("test context", "test query")
            self.assertIn("Error generating response", str(context.exception))

            with self.assertRaises(Exception) as context:
                self.chatbot.generate_response_with_sources([{"text": "test"}], "test query")
            self.assertIn("Error generating response with sources", str(context.exception))

if __name__ == '__main__':
    unittest.main()
