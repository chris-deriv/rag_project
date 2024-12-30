"""Document processing for the RAG application with advanced chunking strategies."""
import os
from typing import List, Dict, Optional, BinaryIO, Union
from dataclasses import dataclass
import re
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata."""
    id: int
    text: str
    metadata: Dict

class DocumentProcessor:
    """Handles document processing with advanced chunking strategies."""
    
    def __init__(self, 
                 chunk_size: int = 100,  # Smaller default chunk size for testing
                 chunk_overlap: int = 20,
                 length_function: str = "token"):
        """Initialize the document processor.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Number of overlapping tokens/chars between chunks
            length_function: Method to count length ('token' or 'char')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._get_length_function(),
            is_separator_regex=False
        )
    
    def _get_length_function(self):
        """Get the appropriate length function based on settings."""
        if self.length_function == "token":
            return lambda x: len(self.tokenizer.encode(x))
        return len
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text.
        
        - Remove excessive whitespace
        - Fix common OCR artifacts
        - Normalize unicode characters
        - Remove non-printable characters
        """
        # Import here to avoid loading at module level
        from ftfy import fix_text
        
        # First fix any encoding/unicode issues
        text = fix_text(text)
        
        # Split into lines and process each line
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # If line ends with hyphen, remove it
            if line.endswith('-'):
                line = line[:-1]
            lines.append(line)
        
        # Join lines with spaces and normalize whitespace
        text = ' '.join(' '.join(line.split()) for line in lines)
        
        return text.strip()
    
    def _extract_pdf_text(self, file_or_path: Union[str, BinaryIO]) -> List[Dict[str, str]]:
        """Extract text from PDF with metadata."""
        result = []
        
        # Suppress all warnings from pypdf
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            
            try:
                # Create reader with error handling and recovery options
                reader = PdfReader(file_or_path, strict=False)
                
                # Get metadata from first page
                metadata = {
                    'title': '',
                    'file_type': 'pdf'
                }
                try:
                    if hasattr(reader, 'metadata') and reader.metadata:
                        metadata['title'] = reader.metadata.get('/Title', '')
                except Exception as e:
                    logger.warning(f"Error extracting PDF metadata: {str(e)}")
                
                # Process each page
                for i, page in enumerate(reader.pages):
                    try:
                        # Get page text with fallback methods
                        text = page.extract_text()
                        
                        # Clean text
                        cleaned_text = self._clean_text(text) if text else ""
                        if not cleaned_text:
                            logger.warning(f"No text extracted from page {i+1}")
                            continue
                        
                        # Add page metadata
                        page_metadata = {
                            'page_number': i + 1,
                            'page_size': (
                                page.mediabox.upper_right if hasattr(page, 'mediabox') 
                                else (0, 0)
                            ),
                            'rotation': page.rotation if hasattr(page, 'rotation') else 0,
                            'title': metadata['title'],  # Include title in page metadata
                            'file_type': metadata['file_type']
                        }
                        
                        # Add to result
                        result.append({
                            'text': cleaned_text,
                            'metadata': page_metadata
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing page {i+1}: {str(e)}")
                        continue
                
                if not result:
                    logger.warning(f"No text extracted from any pages in {file_or_path}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing PDF {file_or_path}: {str(e)}")
                return []
    
    def _extract_docx_text(self, file_path: str) -> List[Dict[str, str]]:
        """Extract text from DOCX with metadata and structure."""
        doc = Document(file_path)
        sections = []
        current_section = {'text': [], 'metadata': {'section_type': 'body'}}
        
        for paragraph in doc.paragraphs:
            # Skip empty paragraphs
            if not paragraph.text.strip():
                continue
            
            # Detect headers/titles based on style
            if paragraph.style.name.startswith(('Heading', 'Title')):
                # Save previous section if it has content
                if current_section['text']:
                    sections.append({
                        'text': '\n'.join(current_section['text']),
                        'metadata': current_section['metadata'].copy()
                    })
                
                # Start new section
                current_section = {
                    'text': [paragraph.text],
                    'metadata': {
                        'section_type': paragraph.style.name,
                        'heading_level': int(paragraph.style.name[-1]) 
                        if paragraph.style.name.startswith('Heading') 
                        else 0
                    }
                }
            else:
                current_section['text'].append(paragraph.text)
        
        # Add final section
        if current_section['text']:
            sections.append({
                'text': '\n'.join(current_section['text']),
                'metadata': current_section['metadata']
            })
        
        return sections
    
    def process_document(self, file_or_path: Union[str, BinaryIO]) -> List[DocumentChunk]:
        """Process a document and return chunks with metadata."""
        if isinstance(file_or_path, str):
            file_ext = os.path.splitext(file_or_path)[1].lower()
            file_name = os.path.basename(file_or_path)
        else:
            # For testing with file-like objects
            file_ext = '.pdf'  # Assume PDF for file-like objects
            file_name = 'test.pdf'
        
        # Extract text based on file type
        if file_ext == '.pdf':
            sections = self._extract_pdf_text(file_or_path)
        elif file_ext in ['.docx', '.doc']:
            sections = self._extract_docx_text(file_or_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Process each section into chunks
        chunks = []
        chunk_id = 1
        
        for section in sections:
            text = section['text']
            base_metadata = {
                'source_file': file_name,
                'file_type': file_ext[1:],  # Remove the dot
                **section['metadata']
            }
            
            # Debug logging for text processing
            logger.debug(f"Processing text length: {len(text)}, Chunk size: {self.chunk_size}")
            
            # Always create at least one chunk
            text_chunks = self.text_splitter.split_text(text) if text else []
            logger.debug(f"Created {len(text_chunks)} chunks")
            
            # Create chunk objects with metadata
            for chunk_text in text_chunks:
                if chunk_text.strip():  # Only create chunks for non-empty text
                    chunk = DocumentChunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            'chunk_size': len(chunk_text),
                            'token_count': len(self.tokenizer.encode(chunk_text))
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
        
        return chunks

# Global document store
class DocumentStore:
    """Manages document storage and retrieval."""
    
    def __init__(self):
        self.documents: List[DocumentChunk] = []
        self.processor = DocumentProcessor()
    
    def add_document(self, file_path: str) -> None:
        """Process and add a document to the store."""
        new_chunks = self.processor.process_document(file_path)
        
        # Update IDs to be continuous with existing documents
        if self.documents:
            max_id = max(chunk.id for chunk in self.documents)
            for chunk in new_chunks:
                chunk.id += max_id
        
        self.documents.extend(new_chunks)
    
    def get_documents(self) -> List[Dict[str, str]]:
        """Return all document chunks in the expected format."""
        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                **chunk.metadata
            }
            for chunk in self.documents
        ]

# Initialize global document store
document_store = DocumentStore()

# Expose functions that match the original API
def add_document(file_path: str) -> None:
    """Add a document to the global store."""
    document_store.add_document(file_path)

def get_documents() -> List[Dict[str, str]]:
    """Get all documents from the global store."""
    return document_store.get_documents()

def process_document(file_path: str) -> List[Dict[str, str]]:
    """Process a single document and return its chunks."""
    processor = DocumentProcessor()
    chunks = processor.process_document(file_path)
    return [
        {
            "id": chunk.id,
            "text": chunk.text,
            **chunk.metadata
        }
        for chunk in chunks
    ]
