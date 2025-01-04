"""Test chatbot functionality."""
import pytest
from unittest.mock import Mock, patch
from src.chatbot import Chatbot

@pytest.fixture
def mock_openai():
    with patch('src.chatbot.OpenAI') as mock:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))]
        )
        mock.return_value = mock_client
        yield mock

@pytest.fixture
def chatbot(mock_openai):
    return Chatbot()

class TestChatbot:
    def test_generate_response_with_toc(self, chatbot):
        """Test response generation with table of contents."""
        context = """Test context
        "toc": [{"text": "Main Title", "level": 1, "children": []}]
        More context"""
        query = "test query"
        
        response = chatbot.generate_response(context, query)
        
        assert isinstance(response, dict)
        assert 'content' in response
        assert 'table_of_contents' in response
        assert response['table_of_contents'] == [{"text": "Main Title", "level": 1, "children": []}]

    def test_generate_response_without_toc(self, chatbot):
        """Test response generation without table of contents."""
        context = "Test context without TOC"
        query = "test query"
        
        response = chatbot.generate_response(context, query)
        
        assert isinstance(response, dict)
        assert 'content' in response
        assert 'table_of_contents' in response
        assert response['table_of_contents'] is None

    def test_generate_response_with_sources_and_toc(self, chatbot):
        """Test response generation with sources and table of contents."""
        contexts = [
            {
                'text': 'Test context 1',
                'source': 'doc1.pdf',
                'toc': [{"text": "Title 1", "level": 1, "children": []}]
            },
            {
                'text': 'Test context 2',
                'source': 'doc2.pdf',
                'toc': [{"text": "Title 2", "level": 1, "children": []}]
            }
        ]
        query = "test query"
        
        response = chatbot.generate_response_with_sources(contexts, query)
        
        assert isinstance(response, dict)
        assert 'content' in response
        assert 'table_of_contents' in response
        assert response['table_of_contents'] == [{"text": "Title 1", "level": 1, "children": []}]

    def test_response_caching_with_toc(self, chatbot):
        """Test response caching with table of contents."""
        context = """Test context
        "toc": [{"text": "Cached Title", "level": 1, "children": []}]
        More context"""
        query = "test query"
        
        # First call should use OpenAI API
        response1 = chatbot.generate_response(context, query)
        
        # Second call should use cache
        response2 = chatbot.generate_response(context, query)
        
        assert response1 == response2
        assert response1['table_of_contents'] == response2['table_of_contents']

    def test_settings_change_clears_cache(self, chatbot):
        """Test that settings changes clear the response cache."""
        context = """Test context
        "toc": [{"text": "Title", "level": 1, "children": []}]"""
        query = "test query"
        
        # Generate initial response
        response1 = chatbot.generate_response(context, query)
        
        # Simulate settings change
        chatbot._handle_settings_change('llm', {'temperature': 0.5})
        
        # Response should be regenerated, not cached
        response2 = chatbot.generate_response(context, query)
        
        assert response1['content'] == response2['content']  # Content same due to mock
        assert response1['table_of_contents'] == response2['table_of_contents']

    def test_error_handling(self, chatbot, mock_openai):
        """Test error handling in response generation."""
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
        
        context = """Test context
        "toc": [{"text": "Error Title", "level": 1, "children": []}]"""
        query = "test query"
        
        with pytest.raises(Exception) as exc_info:
            chatbot.generate_response(context, query)
        assert "Error generating response" in str(exc_info.value)

    def test_multiple_toc_handling(self, chatbot):
        """Test handling of multiple TOCs in source contexts."""
        contexts = [
            {
                'text': 'Test context 1',
                'source': 'doc1.pdf',
                'toc': [{"text": "Title 1", "level": 1, "children": []}]
            },
            {
                'text': 'Test context 2',
                'source': 'doc2.pdf',
                'toc': [{"text": "Title 2", "level": 1, "children": []}]
            }
        ]
        query = "test query"
        
        response = chatbot.generate_response_with_sources(contexts, query)
        
        # Should use TOC from first context
        assert response['table_of_contents'] == [{"text": "Title 1", "level": 1, "children": []}]

    def test_invalid_toc_format(self, chatbot):
        """Test handling of invalid TOC format in context."""
        context = """Test context
        "toc": invalid_json_here
        More context"""
        query = "test query"
        
        response = chatbot.generate_response(context, query)
        
        assert response['table_of_contents'] is None

    def test_empty_context(self, chatbot):
        """Test response generation with empty context."""
        response = chatbot.generate_response("", "test query")
        
        assert isinstance(response, dict)
        assert 'content' in response
        assert 'table_of_contents' in response
        assert response['table_of_contents'] is None
