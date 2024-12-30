"""Document processing for the RAG application with advanced chunking strategies."""
import os
import json
from typing import List, Dict, Optional, BinaryIO, Union
from dataclasses import dataclass, asdict
import re
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import logging
import warnings
import subprocess
import tempfile
from config.settings import CHROMA_PERSIST_DIR

# Configure logging with immediate output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata."""
    id: int
    text: str
    metadata: Dict

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data["metadata"]
        )

class DocumentProcessor:
    """Handles document processing with advanced chunking strategies."""
    
    def __init__(self, 
                 chunk_size: int = 150,  # Default to smaller chunks
                 chunk_overlap: int = 50,  # Larger overlap to avoid splitting sentences
                 length_function: str = "char"):  # Use char count by default
        """Initialize the document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize text splitter with better separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ".",     # Sentence breaks
                "!",     # Exclamation marks
                "?",     # Question marks
                ";",     # Semicolons
                ":",     # Colons
                " ",     # Words
                ""       # Characters
            ],
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
        """Clean and preprocess text."""
        from ftfy import fix_text
        text = fix_text(text)
        
        # Split into lines and process each line
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.endswith('-'):
                line = line[:-1]
            lines.append(line)
        
        # Join lines and normalize whitespace
        text = ' '.join(lines)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _convert_doc_to_docx(self, file_path: str) -> str:
        """Convert a .doc file to .docx format using LibreOffice."""
        try:
            # Create temp directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, 'output.docx')
                
                # Use LibreOffice to convert
                cmd = [
                    'soffice',
                    '--headless',
                    '--convert-to',
                    'docx',
                    '--outdir',
                    temp_dir,
                    file_path
                ]
                
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if process.returncode != 0:
                    raise ValueError(f"LibreOffice conversion failed: {process.stderr}")
                
                # Copy the converted file to a new location
                with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                    with open(output_path, 'rb') as f:
                        temp_file.write(f.read())
                    return temp_file.name
            
        except Exception as e:
            logger.error(f"Error converting .doc to .docx: {str(e)}")
            raise
    
    def _extract_docx_text(self, file_path: str) -> List[Dict[str, str]]:
        """Extract text from DOCX/DOC with metadata and structure."""
        sections = []
        
        try:
            # If it's a .doc file, convert to .docx first
            if file_path.lower().endswith('.doc'):
                file_path = self._convert_doc_to_docx(file_path)
                logger.info(f"Converted .doc to .docx: {file_path}")
            
            doc = Document(file_path)
            
            # Extract document title from core properties
            title = None
            if hasattr(doc, 'core_properties') and doc.core_properties:
                title = doc.core_properties.title
                if title:
                    title = title.strip()
            
            # If no title found, use filename without extension
            if not title:
                title = os.path.splitext(os.path.basename(file_path))[0]
            
            # Create at least one section even if there's no content
            # This ensures we preserve the document title
            current_section = {'text': [], 'metadata': {
                'section_type': 'body',
                'title': title,
                'source_file': os.path.basename(file_path)
            }}
            
            for paragraph in doc.paragraphs:
                # Skip empty paragraphs
                if not paragraph.text.strip():
                    continue
                
                # Detect headers/titles based on style
                if paragraph.style.name.startswith(('Heading', 'Title')):
                    # Save previous section if it has content
                    if current_section['text']:
                        sections.append({
                            'text': ' '.join(current_section['text']),
                            'metadata': current_section['metadata'].copy()
                        })
                    
                    # Start new section
                    current_section = {
                        'text': [paragraph.text],
                        'metadata': {
                            'section_type': paragraph.style.name,
                            'section_title': paragraph.text,
                            'heading_level': int(paragraph.style.name[-1]) 
                            if paragraph.style.name.startswith('Heading') 
                            else 0,
                            'title': title,
                            'source_file': os.path.basename(file_path)
                        }
                    }
                else:
                    # Clean and add paragraph text
                    cleaned_text = self._clean_text(paragraph.text)
                    if cleaned_text:
                        current_section['text'].append(cleaned_text)
            
            # Add final section or empty section if no content
            if current_section['text']:
                sections.append({
                    'text': ' '.join(current_section['text']),
                    'metadata': current_section['metadata']
                })
            else:
                # Add empty section to preserve document metadata
                sections.append({
                    'text': '',
                    'metadata': current_section['metadata']
                })
            
            return sections
            
        except Exception as e:
            logger.error(f"Error processing Word document {file_path}: {str(e)}")
            raise ValueError(f"Error processing Word document: {str(e)}")
        finally:
            # Clean up temporary file if it was created
            if file_path.lower().endswith('.doc'):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    def _extract_pdf_text(self, file_or_path: Union[str, BinaryIO]) -> List[Dict[str, str]]:
        """Extract text from PDF with metadata."""
        result = []
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            try:
                reader = PdfReader(file_or_path, strict=False)
                
                # Extract document title from metadata
                title = None
                if hasattr(reader, 'metadata') and reader.metadata and '/Title' in reader.metadata:
                    title = reader.metadata['/Title']
                    if isinstance(title, bytes):
                        title = title.decode('utf-8', errors='ignore')
                    title = title.strip()
                
                # If no title in metadata, try to get it from the first page text
                if not title and len(reader.pages) > 0:
                    first_page_text = reader.pages[0].extract_text()
                    first_lines = [line.strip() for line in first_page_text.split('\n') if line.strip()][:3]
                    for line in first_lines:
                        # Look for a line that appears to be a title:
                        # - Starts with a capital letter
                        # - No sentence endings
                        # - Not too long
                        # - Not all caps (likely a header)
                        if (line and len(line) < 100 and 
                            re.match(r'^[A-Z]', line) and 
                            not re.search(r'[.!?]$', line) and
                            not line.isupper()):
                            title = line
                            break
                
                # If still no title, use filename without extension
                if not title:
                    if isinstance(file_or_path, str):
                        title = os.path.splitext(os.path.basename(file_or_path))[0]
                    else:
                        title = "Untitled Document"
                
                metadata = {
                    'title': title,
                    'file_type': 'pdf',
                    'source_file': os.path.basename(file_or_path) if isinstance(file_or_path, str) else "uploaded.pdf"
                }
                
                # Process each page
                for i, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        cleaned_text = self._clean_text(text) if text else ""
                        if not cleaned_text:
                            continue
                        
                        # Split text into lines and process each line
                        lines = cleaned_text.split('\n')
                        sections = []
                        current_section = {"text": [], "title": "", "type": "content"}
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Check if this is a section header
                            is_header = (
                                re.match(r'^(?:(?:\d+\.)*\d+\s+|\bSection\s+\d+:|\bChapter\s+\d+:?|\b(?:Introduction|Abstract|Conclusion|Summary|Background|Methods|Results|Discussion)\b)', line, re.IGNORECASE) or
                                (len(line) < 100 and  # Not too long
                                 re.match(r'^[A-Z]', line) and  # Starts with capital letter
                                 not re.search(r'[.!?]$', line) and  # No sentence endings
                                 not re.match(r'^(?:(?:The|A|An|This|That|These|Those)\s+)', line))  # Not starting with articles
                            )
                            
                            if is_header:
                                # Save previous section if it has content
                                if current_section["text"]:
                                    section_metadata = metadata.copy()
                                    section_metadata.update({
                                        'section_type': current_section["type"],
                                        'section_title': current_section["title"],
                                        'page_number': i + 1,
                                        'page_size': page.mediabox.upper_right if hasattr(page, 'mediabox') else (0, 0),
                                        'rotation': page.rotation if hasattr(page, 'rotation') else 0
                                    })
                                    sections.append({
                                        'text': ' '.join(current_section["text"]),
                                        'metadata': section_metadata
                                    })
                                
                                # Start new section
                                current_section = {
                                    "text": [],
                                    "title": line,
                                    "type": "heading"
                                }
                            else:
                                current_section["text"].append(line)
                        
                        # Add final section
                        if current_section["text"]:
                            section_metadata = metadata.copy()
                            section_metadata.update({
                                'section_type': current_section["type"],
                                'section_title': current_section["title"],
                                'page_number': i + 1,
                                'page_size': page.mediabox.upper_right if hasattr(page, 'mediabox') else (0, 0),
                                'rotation': page.rotation if hasattr(page, 'rotation') else 0
                            })
                            sections.append({
                                'text': ' '.join(current_section["text"]),
                                'metadata': section_metadata
                            })
                        
                        # Add all sections from this page
                        result.extend(sections)
                        
                    except Exception as e:
                        logger.warning(f"Error processing page {i+1}: {str(e)}")
                        continue
                
                # If no sections were created (empty PDF), create one with empty text
                if not result:
                    result.append({
                        'text': '',
                        'metadata': {
                            **metadata,
                            'section_type': 'content',
                            'section_title': '',
                            'page_number': 1,
                            'page_size': (0, 0),
                            'rotation': 0
                        }
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing PDF {file_or_path}: {str(e)}")
                return []
    
    def process_document(self, file_or_path: Union[str, BinaryIO]) -> List[DocumentChunk]:
        """Process a document and return chunks with metadata."""
        if isinstance(file_or_path, str):
            file_ext = os.path.splitext(file_or_path)[1].lower()
            file_name = os.path.basename(file_or_path)
        else:
            file_ext = '.pdf'
            file_name = 'uploaded.pdf'
        
        logger.info(f"\n{'='*80}\nProcessing document: {file_name}\n{'='*80}")
        
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
        
        logger.info(f"\nFound {len(sections)} sections to process")
        
        # Handle empty documents
        if not sections:
            # Create a single empty chunk with basic metadata
            chunks.append(DocumentChunk(
                id=1,
                text="",
                metadata={
                    'source_file': file_name,
                    'file_type': file_ext[1:],
                    'title': os.path.splitext(file_name)[0],
                    'section_type': 'content',
                    'section_title': '',
                    'chunk_size': 0,
                    'token_count': 0,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            ))
            return chunks
        
        # Calculate total chunks for non-empty documents
        total_chunks = sum(1 if not section['text'] else len(self.text_splitter.split_text(section['text'])) 
                         for section in sections)
        current_chunk = 0
        
        for section_idx, section in enumerate(sections, 1):
            text = section['text']
            base_metadata = {
                'source_file': file_name,
                'file_type': file_ext[1:],
                **section['metadata']
            }
            
            logger.info(f"\n{'-'*80}\nProcessing Section {section_idx}/{len(sections)}")
            logger.info(f"Section Title: {base_metadata.get('section_title', 'Untitled')}")
            logger.info(f"Section Type: {base_metadata.get('section_type', 'Unknown')}")
            
            if not text:
                # Create a single empty chunk for empty sections
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    text="",
                    metadata={
                        **base_metadata,
                        'chunk_size': 0,
                        'token_count': 0,
                        'chunk_index': current_chunk,
                        'total_chunks': total_chunks
                    }
                ))
                chunk_id += 1
                current_chunk += 1
                continue
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            logger.info(f"Generated {len(text_chunks)} chunks from section")
            
            # Create chunk objects
            for chunk_idx, chunk_text in enumerate(text_chunks, 1):
                chunk_text = chunk_text.strip()
                if chunk_text:
                    # Remove leading punctuation
                    chunk_text = re.sub(r'^[.,!?;:]\s*', '', chunk_text)
                    
                    chunk = DocumentChunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            'chunk_size': len(chunk_text),
                            'token_count': len(self.tokenizer.encode(chunk_text)),
                            'chunk_index': current_chunk,
                            'total_chunks': total_chunks
                        }
                    )
                    chunks.append(chunk)
                    
                    # Enhanced chunk logging
                    logger.info(f"\nChunk {chunk_id} (Section {section_idx}, Chunk {chunk_idx}):")
                    logger.info("-" * 40)
                    logger.info("Content:")
                    logger.info(f"{chunk_text}")
                    logger.info("-" * 40)
                    logger.info(f"Character Count: {len(chunk_text)}")
                    logger.info(f"Token Count: {len(self.tokenizer.encode(chunk_text))}")
                    logger.info("Metadata:")
                    logger.info(json.dumps(chunk.metadata, indent=2))
                    logger.info("=" * 40)
                    
                    chunk_id += 1
                    current_chunk += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Document processing complete")
        logger.info(f"Total sections processed: {len(sections)}")
        logger.info(f"Total chunks created: {len(chunks)}")
        
        if chunks:
            logger.info(f"Average chunk size: {sum(len(c.text) for c in chunks)/len(chunks):.2f} characters")
            logger.info(f"Average tokens per chunk: {sum(len(self.tokenizer.encode(c.text)) for c in chunks)/len(chunks):.2f}")
        else:
            logger.info("No chunks created (empty document)")
            
        logger.info(f"{'='*80}\n")
        
        return chunks

# Global document store
class DocumentStore:
    """Manages document storage and retrieval."""
    
    def __init__(self):
        self.documents: List[DocumentChunk] = []
        self.processor = DocumentProcessor()
        self.storage_path = os.path.join(CHROMA_PERSIST_DIR, "documents.json")
        self._load_documents()
    
    def _load_documents(self) -> None:
        """Load documents from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.documents = [DocumentChunk.from_dict(doc) for doc in data]
                logger.info(f"Loaded {len(self.documents)} documents from storage")
            except Exception as e:
                logger.error(f"Error loading documents: {str(e)}")
                self.documents = []

    def _save_documents(self) -> None:
        """Save documents to disk."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump([doc.to_dict() for doc in self.documents], f)
            logger.info(f"Saved {len(self.documents)} documents to storage")
        except Exception as e:
            logger.error(f"Error saving documents: {str(e)}")
    
    def add_document(self, file_path: str) -> None:
        """Process and add a document to the store."""
        new_chunks = self.processor.process_document(file_path)
        
        # Update IDs to be continuous with existing documents
        if self.documents:
            max_id = max(chunk.id for chunk in self.documents)
            for chunk in new_chunks:
                chunk.id += max_id
        
        self.documents.extend(new_chunks)
        self._save_documents()
    
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
