"""Unit tests for document processing functionality."""
import pytest
from src.documents import (
    process_document,
    add_document,
    get_documents,
    documents
)

def test_invalid_file_extension():
    """Test that invalid file extensions raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        process_document("test.txt")
    assert "Unsupported file type" in str(exc_info.value)

def test_document_id_generation(monkeypatch):
    """Test that document chunks get correct sequential IDs."""
    # Mock process_pdf to return known chunks
    def mock_process_pdf(file_path):
        return ["chunk1", "chunk2"]
    
    monkeypatch.setattr("src.documents.process_pdf", mock_process_pdf)
    
    # Clear existing documents
    documents.clear()
    
    # Process first document
    result = process_document("test.pdf")
    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2
    assert result[0]["text"] == "chunk1"
    assert result[1]["text"] == "chunk2"

def test_add_document_continuous_ids(monkeypatch):
    """Test that adding multiple documents maintains continuous IDs."""
    def mock_process_pdf(file_path):
        return ["chunk"]
    
    monkeypatch.setattr("src.documents.process_pdf", mock_process_pdf)
    
    # Clear existing documents
    documents.clear()
    
    # Add first document
    add_document("doc1.pdf")
    assert len(documents) == 1
    assert documents[0]["id"] == 1
    
    # Add second document
    add_document("doc2.pdf")
    assert len(documents) == 2
    assert documents[1]["id"] == 2

def test_get_documents(monkeypatch):
    """Test retrieving all documents."""
    def mock_process_pdf(file_path):
        return ["test chunk"]
    
    monkeypatch.setattr("src.documents.process_pdf", mock_process_pdf)
    
    # Clear existing documents
    documents.clear()
    
    # Add a document
    add_document("test.pdf")
    
    # Get all documents
    result = get_documents()
    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["text"] == "test chunk"

def test_docx_extension_accepted(monkeypatch):
    """Test that .docx files are accepted."""
    def mock_process_docx(file_path):
        return ["docx chunk"]
    
    monkeypatch.setattr("src.documents.process_docx", mock_process_docx)
    
    result = process_document("test.docx")
    assert len(result) == 1
    assert result[0]["text"] == "docx chunk"

def test_doc_extension_accepted(monkeypatch):
    """Test that .doc files are accepted."""
    def mock_process_docx(file_path):
        return ["doc chunk"]
    
    monkeypatch.setattr("src.documents.process_docx", mock_process_docx)
    
    result = process_document("test.doc")
    assert len(result) == 1
    assert result[0]["text"] == "doc chunk"
