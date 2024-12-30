"""Unit tests for document processing functionality."""
import os
import io
import zipfile
import pytest
from unittest.mock import Mock, patch
from src.documents import (
    process_document,
    add_document,
    get_documents,
    document_store,
    DocumentProcessor,
    DocumentChunk
)

def test_invalid_file_extension():
    """Test that invalid file extensions raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        process_document("test.txt")
    assert "Unsupported file type" in str(exc_info.value)

def test_document_processor_initialization():
    """Test DocumentProcessor initialization with custom settings."""
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=100,
        length_function="token"
    )
    assert processor.chunk_size == 1000
    assert processor.chunk_overlap == 100
    assert processor.length_function == "token"

def test_text_cleaning():
    """Test text cleaning functionality."""
    processor = DocumentProcessor()
    dirty_text = "This   has\nextra   spaces\n\nand-\nbroken words"
    clean_text = processor._clean_text(dirty_text)
    assert clean_text == "This has extra spaces and broken words"

def test_pdf_title_from_metadata(monkeypatch):
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
    
    monkeypatch.setattr("src.documents.PdfReader", MockPdfReader)
    
    processor = DocumentProcessor()
    result = processor.process_document("test.pdf")
    
    assert len(result) > 0
    assert result[0].metadata['title'] == 'Test Document Title'

def test_pdf_title_from_content(monkeypatch):
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
    
    monkeypatch.setattr("src.documents.PdfReader", MockPdfReader)
    
    processor = DocumentProcessor()
    result = processor.process_document("test.pdf")
    
    assert len(result) > 0
    assert result[0].metadata['title'] == 'Document Title'

def test_pdf_title_fallback_to_filename(monkeypatch):
    """Test PDF title fallback to filename."""
    # Create mock PDF without metadata title or content title
    mock_metadata = {}
    mock_pages = [Mock()]
    # Use lowercase content to avoid it being detected as a title
    mock_pages[0].extract_text.return_value = "just some content"
    mock_pages[0].mediabox.upper_right = (612, 792)
    mock_pages[0].rotation = 0
    
    class MockPdfReader:
        def __init__(self, *args, **kwargs):
            self.metadata = mock_metadata
            self.pages = mock_pages
    
    monkeypatch.setattr("src.documents.PdfReader", MockPdfReader)
    
    processor = DocumentProcessor()
    result = processor.process_document("test_document.pdf")
    
    assert len(result) > 0
    assert result[0].metadata['title'] == 'test_document'

def test_docx_title_from_properties(monkeypatch):
    """Test DOCX title extraction from core properties."""
    # Create mock DOCX with core properties title
    mock_core_properties = Mock()
    mock_core_properties.title = "Test DOCX Title"
    mock_doc = Mock()
    mock_doc.core_properties = mock_core_properties
    mock_doc.paragraphs = []
    
    def mock_Document(*args, **kwargs):
        return mock_doc
    
    monkeypatch.setattr("src.documents.Document", mock_Document)
    
    processor = DocumentProcessor()
    result = processor.process_document("test.docx")
    
    assert len(result) > 0
    assert result[0].metadata['title'] == 'Test DOCX Title'

def test_docx_title_fallback_to_filename(monkeypatch):
    """Test DOCX title fallback to filename."""
    # Create mock DOCX without core properties title
    mock_core_properties = Mock()
    mock_core_properties.title = ""
    mock_doc = Mock()
    mock_doc.core_properties = mock_core_properties
    mock_doc.paragraphs = []
    
    def mock_Document(*args, **kwargs):
        return mock_doc
    
    monkeypatch.setattr("src.documents.Document", mock_Document)
    
    processor = DocumentProcessor()
    result = processor.process_document("test_document.docx")
    
    assert len(result) > 0
    assert result[0].metadata['title'] == 'test_document'

def test_chunk_size_and_overlap():
    """Test chunk size and overlap settings."""
    processor = DocumentProcessor(chunk_size=20, chunk_overlap=5, length_function="char")
    text = "This is a test sentence. Another test sentence for overlap."
    chunks = processor.text_splitter.split_text(text)
    
    # Verify we have multiple chunks
    assert len(chunks) > 1
    
    # Verify chunks are roughly the expected size (allowing some flexibility)
    assert all(len(chunk) <= 25 for chunk in chunks)  # chunk_size + some buffer
    
    # Verify content is preserved
    combined = " ".join(chunks)
    assert "test sentence" in combined
    assert "overlap" in combined

def test_document_store_continuous_ids(monkeypatch):
    """Test that DocumentStore maintains continuous IDs across multiple documents."""
    document_store.documents.clear()
    
    def mock_process(self, path):
        # Create new chunks for each call to avoid sharing references
        return [
            DocumentChunk(
                id=1,
                text="test chunk",
                metadata={"source_file": path}
            )
        ]
    
    monkeypatch.setattr(DocumentProcessor, "process_document", mock_process)
    
    # Add documents
    add_document("doc1.pdf")
    add_document("doc2.pdf")
    
    docs = get_documents()
    assert len(docs) == 2
    assert docs[0]["id"] == 1
    assert docs[1]["id"] == 2

def test_metadata_in_get_documents(monkeypatch):
    """Test that get_documents includes metadata in the output."""
    document_store.documents.clear()
    
    mock_chunks = [
        DocumentChunk(
            id=1,
            text="test chunk",
            metadata={
                "source_file": "test.pdf",
                "title": "Test Document",
                "page_number": 1,
                "token_count": 2
            }
        )
    ]
    
    def mock_process(self, path):
        return mock_chunks
    
    monkeypatch.setattr(DocumentProcessor, "process_document", mock_process)
    add_document("test.pdf")
    
    docs = get_documents()
    assert len(docs) == 1
    assert "source_file" in docs[0]
    assert "title" in docs[0]
    assert "page_number" in docs[0]
    assert "token_count" in docs[0]
    assert docs[0]["title"] == "Test Document"
