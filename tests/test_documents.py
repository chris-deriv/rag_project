"""Unit tests for document processing functionality."""
import os
import io
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

def test_pdf_processing_with_metadata(monkeypatch):
    """Test PDF processing with metadata extraction."""
    # Create mock extraction result
    mock_result = [{
        'text': "Test content for page one. This is a complete sentence.",
        'metadata': {
            'page_number': 1,
            'page_size': (612, 792),
            'rotation': 0,
            'title': 'Test Document',
            'file_type': 'pdf'
        }
    }]
    
    # Mock the _extract_pdf_text method
    def mock_extract_pdf_text(self, file_or_path):
        return mock_result
    
    monkeypatch.setattr(DocumentProcessor, "_extract_pdf_text", mock_extract_pdf_text)
    
    # Process document
    processor = DocumentProcessor()
    result = processor.process_document("test.pdf")
    
    # Verify results
    assert len(result) > 0
    assert result[0].metadata['page_number'] == 1
    assert result[0].metadata['title'] == 'Test Document'
    assert result[0].metadata['file_type'] == 'pdf'
    assert "Test content for page one" in result[0].text

def test_docx_processing_with_structure(monkeypatch):
    """Test DOCX processing with structure preservation."""
    class MockStyle:
        def __init__(self, name):
            self.name = name
    
    class MockParagraph:
        def __init__(self, text, style_name):
            self.text = text
            self.style = MockStyle(style_name)
    
    class MockDoc:
        def __init__(self, *args, **kwargs):
            self.paragraphs = [
                MockParagraph("Document Title", "Title"),
                MockParagraph("Section 1", "Heading 1"),
                MockParagraph("Regular paragraph 1", "Normal"),
                MockParagraph("Section 2", "Heading 1"),
                MockParagraph("Regular paragraph 2", "Normal")
            ]
            self.part = Mock()  # Mock the part property
    
    # Mock Document to return our mock directly
    monkeypatch.setattr("docx.Document", lambda *args, **kwargs: MockDoc())
    
    result = process_document("test.docx")
    
    # Verify structure preservation
    sections = [chunk for chunk in result if chunk['metadata'].get('section_type', '').startswith(('Title', 'Heading'))]
    assert len(sections) > 0
    assert any(s['metadata']['section_type'] == 'Title' for s in sections)
    assert any(s['metadata']['section_type'] == 'Heading 1' for s in sections)

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
    assert "page_number" in docs[0]
    assert "token_count" in docs[0]
