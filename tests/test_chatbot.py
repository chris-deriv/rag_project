import unittest
from unittest.mock import Mock, patch, call
from src.chatbot import Chatbot

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

    def test_format_contexts_for_cache(self):
        """Test context formatting is deterministic."""
        contexts = [
            {"text": "Second text"},
            {"text": "First text"},
            {"text": "Third text"}
        ]
        
        # Test sorting is consistent
        formatted1 = self.chatbot._format_contexts_for_cache(contexts)
        formatted2 = self.chatbot._format_contexts_for_cache(contexts[::-1])  # Reverse order
        self.assertEqual(formatted1, formatted2)
        
        # Verify format
        expected = "Source 1:\nFirst text\n\nSource 2:\nSecond text\n\nSource 3:\nThird text"
        self.assertEqual(formatted1, expected)

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
                {"text": "Context 1"},
                {"text": "Context 2"}
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
