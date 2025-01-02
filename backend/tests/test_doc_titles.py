"""Test document title functionality directly."""
import os
import io
from unittest.mock import Mock, patch
from rag_backend.documents import DocumentProcessor

def test_pdf_title_from_metadata():
    """Test PDF title extraction from metadata."""
    # Create mock PDF with metadata title
    mock_metadata = {'/Title': 'Test Document Title'}
    mock_pages = [Mock()]
    mock_pages[0].extract_text.return_value = "Page content"
    mock_pages[0].mediabox.upper_right = (612, 792)
    mock_pages[0].rotation = 0
    
    class MockPdfReader:
        def __init__(self, *args, **kwargs):
            self.metadata = mock_metadata
            self.pages = mock_pages
    
    # Monkeypatch PdfReader and os.path.exists
    import rag_backend.documents
    original_pdfreader = rag_backend.documents.PdfReader
    rag_backend.documents.PdfReader = MockPdfReader
    
    try:
        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            processor = DocumentProcessor()
            result = processor.process_document("test.pdf")
            
            assert len(result) > 0
            assert result[0].metadata['title'] == 'Test Document Title'
            print("✓ PDF title from metadata test passed")
    finally:
        # Restore original
        rag_backend.documents.PdfReader = original_pdfreader

def test_pdf_title_from_content():
    """Test PDF title extraction from first page content."""
    # Create mock PDF without metadata title but with title in content
    mock_metadata = {}
    mock_pages = [Mock()]
    mock_pages[0].extract_text.return_value = "Document Title\nThis is the content\nMore content"
    mock_pages[0].mediabox.upper_right = (612, 792)
    mock_pages[0].rotation = 0
    
    class MockPdfReader:
        def __init__(self, *args, **kwargs):
            self.metadata = mock_metadata
            self.pages = mock_pages
    
    # Monkeypatch PdfReader and os.path.exists
    import rag_backend.documents
    original_pdfreader = rag_backend.documents.PdfReader
    rag_backend.documents.PdfReader = MockPdfReader
    
    try:
        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            processor = DocumentProcessor()
            result = processor.process_document("test.pdf")
            
            assert len(result) > 0
            assert result[0].metadata['title'] == 'Document Title'
            print("✓ PDF title from content test passed")
    finally:
        # Restore original
        rag_backend.documents.PdfReader = original_pdfreader

def test_pdf_title_fallback():
    """Test PDF title fallback to filename."""
    # Create mock PDF without metadata title or content title
    mock_metadata = {}
    mock_pages = [Mock()]
    mock_pages[0].extract_text.return_value = "just some content"
    mock_pages[0].mediabox.upper_right = (612, 792)
    mock_pages[0].rotation = 0
    
    class MockPdfReader:
        def __init__(self, *args, **kwargs):
            self.metadata = mock_metadata
            self.pages = mock_pages
    
    # Monkeypatch PdfReader and os.path.exists
    import rag_backend.documents
    original_pdfreader = rag_backend.documents.PdfReader
    rag_backend.documents.PdfReader = MockPdfReader
    
    try:
        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            processor = DocumentProcessor()
            result = processor.process_document("test_document.pdf")
            
            assert len(result) > 0
            assert result[0].metadata['title'] == 'test_document'
            print("✓ PDF title fallback test passed")
    finally:
        # Restore original
        rag_backend.documents.PdfReader = original_pdfreader

def test_docx_title_from_properties():
    """Test DOCX title extraction from core properties."""
    # Create mock DOCX with core properties title
    mock_core_properties = Mock()
    mock_core_properties.title = "Test DOCX Title"
    mock_doc = Mock()
    mock_doc.core_properties = mock_core_properties
    mock_doc.paragraphs = [Mock()]
    mock_doc.paragraphs[0].text = "Test content"
    
    class MockDocument:
        def __init__(self, *args, **kwargs):
            self.core_properties = mock_core_properties
            self.paragraphs = mock_doc.paragraphs
    
    # Monkeypatch Document and os.path.exists
    import rag_backend.documents
    original_document = rag_backend.documents.Document
    rag_backend.documents.Document = MockDocument
    
    try:
        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            processor = DocumentProcessor()
            result = processor.process_document("test.docx")
            
            assert len(result) > 0
            assert result[0].metadata['title'] == 'Test DOCX Title'
            print("✓ DOCX title from properties test passed")
    finally:
        # Restore original
        rag_backend.documents.Document = original_document

def test_docx_title_fallback():
    """Test DOCX title fallback to filename."""
    # Create mock DOCX without core properties title
    mock_core_properties = Mock()
    mock_core_properties.title = ""
    mock_doc = Mock()
    mock_doc.core_properties = mock_core_properties
    mock_doc.paragraphs = [Mock()]
    mock_doc.paragraphs[0].text = "Test content"
    
    class MockDocument:
        def __init__(self, *args, **kwargs):
            self.core_properties = mock_core_properties
            self.paragraphs = mock_doc.paragraphs
    
    # Monkeypatch Document and os.path.exists
    import rag_backend.documents
    original_document = rag_backend.documents.Document
    rag_backend.documents.Document = MockDocument
    
    try:
        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            processor = DocumentProcessor()
            result = processor.process_document("test_document.docx")
            
            assert len(result) > 0
            assert result[0].metadata['title'] == 'test_document'
            print("✓ DOCX title fallback test passed")
    finally:
        # Restore original
        rag_backend.documents.Document = original_document

if __name__ == '__main__':
    print("\nRunning document title tests...\n")
    test_pdf_title_from_metadata()
    test_pdf_title_from_content()
    test_pdf_title_fallback()
    test_docx_title_from_properties()
    test_docx_title_fallback()
    print("\nAll tests passed! ✨\n")
