"""Document processing for the RAG application."""
import os
from typing import List, Dict
from PyPDF2 import PdfReader
from docx import Document

def process_pdf(file_path: str) -> List[str]:
    """Extract text from PDF and split into chunks."""
    reader = PdfReader(file_path)
    chunks = []
    
    for page in reader.pages:
        text = page.extract_text()
        # Split into paragraphs and filter out empty lines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
    
    return chunks

def process_docx(file_path: str) -> List[str]:
    """Extract text from DOCX and split into chunks."""
    doc = Document(file_path)
    chunks = []
    
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            chunks.append(paragraph.text.strip())
    
    return chunks

def process_document(file_path: str) -> List[Dict[str, str]]:
    """Process a document and return chunks with IDs."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        chunks = process_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        chunks = process_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    # Convert chunks to the expected format with IDs
    return [{"id": i + 1, "text": chunk} for i, chunk in enumerate(chunks)]

# Initialize with empty documents list instead of sample data
documents = []

def add_document(file_path: str) -> None:
    """Process a document and add its chunks to the documents list."""
    global documents
    new_chunks = process_document(file_path)
    
    # Update IDs to be continuous with existing documents
    if documents:
        max_id = max(doc["id"] for doc in documents)
        for chunk in new_chunks:
            chunk["id"] += max_id
    
    documents.extend(new_chunks)

def get_documents() -> List[Dict[str, str]]:
    """Return all processed document chunks."""
    return documents
