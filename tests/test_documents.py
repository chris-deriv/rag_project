import unittest
from src.documents import DocumentProcessor
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

if __name__ == '__main__':
    unittest.main()
