import unittest
from unittest.mock import Mock, patch
import numpy as np
from src.documents import DocumentProcessor, DocumentStore
import tempfile
import os

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor()
        
    def test_pdf_section_detection(self):
        # Create a temporary PDF file with test content
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            # We'll use the file path but not write to it since we can't easily create PDFs
            self.test_pdf = f.name
            
        # Mock the PDF content in _extract_pdf_text
        test_pdf_path = self.test_pdf  # Capture in closure
        def mock_extract_text(self, file_or_path):
            return [
                {
                    'text': '1.0 Introduction\nThis is the introduction section.',
                    'metadata': {
                        'title': 'Test Document',
                        'file_type': 'pdf',
                        'source_file': os.path.basename(test_pdf_path),
                        'section_type': 'heading',
                        'section_title': '1.0 Introduction'
                    }
                },
                {
                    'text': '2.0 Methods\nThis is the methods section.',
                    'metadata': {
                        'title': 'Test Document',
                        'file_type': 'pdf',
                        'source_file': os.path.basename(test_pdf_path),
                        'section_type': 'heading',
                        'section_title': '2.0 Methods'
                    }
                }
            ]
            
        # Store original method and replace with mock
        original_extract = DocumentProcessor._extract_pdf_text
        DocumentProcessor._extract_pdf_text = mock_extract_text
        
        try:
            # Process the document
            chunks = self.processor.process_document(self.test_pdf)
            
            # Verify sections were detected
            self.assertTrue(any(chunk.metadata.get('section_title') == '1.0 Introduction' for chunk in chunks))
            self.assertTrue(any(chunk.metadata.get('section_title') == '2.0 Methods' for chunk in chunks))
            
        finally:
            # Restore original method and clean up
            DocumentProcessor._extract_pdf_text = original_extract
            os.unlink(self.test_pdf)
    
    def test_section_header_detection(self):
        # Test various section header patterns
        test_headers = [
            "1.0 Introduction",
            "Section 1: Overview",
            "Chapter 2: Background",
            "Methods",
            "Results and Discussion",
            "Abstract",
            "Conclusion"
        ]
        
        non_headers = [
            "The quick brown fox",
            "This is a regular sentence.",
            "A section about something",
            "Some random text that is quite long and definitely not a header because it contains too many words and ends with a period."
        ]
        
        # Create a document with these headers, ensuring each is on its own line
        content = "\n".join(test_headers) + "\n\n" + "\n".join(non_headers)
        
        # Mock PDF extraction to return our test content
        def mock_extract_text(self, file_or_path):
            # Return each header as a separate section
            return [
                {
                    'text': header,
                    'metadata': {
                        'title': 'Test Document',
                        'file_type': 'pdf',
                        'source_file': 'test.pdf',
                        'section_type': 'heading',
                        'section_title': header
                    }
                }
                for header in test_headers
            ]
        
        # Store original method and replace with mock
        original_extract = DocumentProcessor._extract_pdf_text
        DocumentProcessor._extract_pdf_text = mock_extract_text
        
        try:
            # Process the document
            chunks = self.processor.process_document('test.pdf')
            
            # Get all detected section titles
            section_titles = {chunk.metadata.get('section_title', '') for chunk in chunks}
            
            # Verify each header was preserved in the chunks
            for header in test_headers:
                self.assertTrue(
                    any(chunk.metadata.get('section_title') == header for chunk in chunks),
                    f"Failed to detect header: {header}"
                )
            
            # Verify section types are correct
            for chunk in chunks:
                self.assertEqual(chunk.metadata.get('section_type'), 'heading')
                
        finally:
            # Restore original method
            DocumentProcessor._extract_pdf_text = original_extract

class TestDocumentStore(unittest.TestCase):
    @patch('src.documents.DocumentProcessor._extract_pdf_text')
    def test_add_document(self, mock_extract_text):
        # Create mock embeddings
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b', delete=False) as f:
            test_pdf = f.name
            
        # Configure mock PDF extraction
        mock_extract_text.return_value = [
            {
                'text': 'Test section 1',
                'metadata': {
                    'title': 'Test Document',
                    'file_type': 'pdf',
                    'source_file': os.path.basename(test_pdf),
                    'section_type': 'content'
                }
            },
            {
                'text': 'Test section 2',
                'metadata': {
                    'title': 'Test Document',
                    'file_type': 'pdf',
                    'source_file': os.path.basename(test_pdf),
                    'section_type': 'content'
                }
            }
        ]
        
        # Create mock EmbeddingGenerator
        mock_embedding_generator = Mock()
        mock_embedding_generator.generate_embeddings.return_value = mock_embeddings
        
        # Create mock VectorDatabase
        mock_vector_db = Mock()
        
        try:
            # Initialize store with mocked dependencies
            with patch('src.documents.EmbeddingGenerator', return_value=mock_embedding_generator):
                with patch('src.documents.VectorDatabase', return_value=mock_vector_db):
                    store = DocumentStore()
                    store.add_document(test_pdf)
            
            # Verify embeddings were generated
            mock_embedding_generator.generate_embeddings.assert_called_once()
            texts_arg = mock_embedding_generator.generate_embeddings.call_args[0][0]
            self.assertEqual(len(texts_arg), 2)
            self.assertTrue('Test section 1' in texts_arg)
            self.assertTrue('Test section 2' in texts_arg)
            
            # Verify documents were added to database with embeddings
            mock_vector_db.add_documents.assert_called_once()
            documents_arg = mock_vector_db.add_documents.call_args[0][0]
            self.assertEqual(len(documents_arg), 2)
            
            # Check first document
            self.assertEqual(documents_arg[0]['text'], 'Test section 1')
            np.testing.assert_array_equal(documents_arg[0]['embedding'], mock_embeddings[0])
            
            # Check second document
            self.assertEqual(documents_arg[1]['text'], 'Test section 2')
            np.testing.assert_array_equal(documents_arg[1]['embedding'], mock_embeddings[1])
            
        finally:
            # Clean up
            os.unlink(test_pdf)

if __name__ == '__main__':
    unittest.main()
