"""Document processing for the RAG application with advanced chunking strategies."""
import os
import uuid
import json
from typing import List, Dict, Optional, BinaryIO, Union, Any, Tuple
from dataclasses import dataclass, asdict
import re
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import logging
import warnings
import subprocess
import tempfile
from .database import VectorDatabase
from .embedding import EmbeddingGenerator
from .document_analyzer import document_analyzer, DocumentAnalyzer, DocumentSection
from config.dynamic_settings import settings_manager
from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Configure logging with immediate output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingState:
    """Tracks the state of document processing."""
    status: str  # 'processing', 'completed', 'error'
    error: Optional[str] = None
    source_name: Optional[str] = None
    chunk_count: int = 0
    total_chunks: int = 0
    classification: Optional[str] = None
    toc: Optional[List[Dict]] = None

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata."""
    id: str
    text: str
    metadata: Dict

class DocumentProcessor:
    """Handles document processing with advanced chunking strategies."""
    
    def __init__(self, length_function: str = "char", analyzer: Optional[DocumentAnalyzer] = None):
        """Initialize the document processor."""
        # Get initial settings
        self.settings = settings_manager.get_all_settings()
        self.length_function = length_function
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.analyzer = analyzer or document_analyzer
        
        # Store chunk size locally
        self.chunk_size = self.settings['document_processing'].get('chunk_size', DEFAULT_CHUNK_SIZE)
        self.chunk_overlap = self.settings['document_processing'].get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)
        
        # Initialize text splitter with settings
        self._init_text_splitter()
        
        # Register as observer for settings changes
        settings_manager.add_observer(self._handle_settings_change)
        
        logger.info(f"Initialized DocumentProcessor with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}")

    def _handle_settings_change(self, setting_name: str, new_value: dict) -> None:
        """Handle settings changes from the settings manager."""
        if setting_name == 'document_processing':
            logger.info(f"Updating document processing settings: {new_value}")
            self.settings['document_processing'] = new_value
            self.chunk_size = new_value.get('chunk_size', DEFAULT_CHUNK_SIZE)
            self.chunk_overlap = new_value.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)
            self._init_text_splitter()

    def _init_text_splitter(self) -> None:
        """Initialize or reinitialize the text splitter with current settings."""
        logger.info(f"Initializing text splitter with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        
        # Use more aggressive splitting for better chunk size control
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", ".", "!", "?", ";", ":", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._get_length_function(),
            is_separator_regex=False
        )

    def _get_length_function(self) -> callable:
        """Get the appropriate length function based on settings."""
        if self.length_function == "token":
            return lambda x: len(self.tokenizer.encode(x))
        return len

    def _split_section(self, section: DocumentSection, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split a section into chunks while preserving metadata."""
        if not section.text.strip():
            return []
            
        # Split text into chunks
        raw_chunks = self.text_splitter.split_text(section.text)
        
        if len(raw_chunks) > 1:
            logger.info(f"Split section into {len(raw_chunks)} chunks")
        
        # Create document chunks with metadata
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(raw_chunks)
            }
            
            # Add section metadata if available
            if section.heading:
                chunk_metadata.update({
                    'section_title': section.heading.text,
                    'section_level': section.heading.level
                })
            
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
            
        return chunks

    def _get_title_from_content(self, text: str, filename: str) -> Optional[str]:
        """Extract title from content or use filename as fallback."""
        if not text or not text.strip():
            return os.path.splitext(filename)[0]
            
        first_line = text.strip().split('\n')[0].strip()
        if not first_line:  # Skip empty lines
            return os.path.splitext(filename)[0]
            
        # Only consider it a title if it's short, doesn't end with punctuation,
        # and contains words that might indicate it's a title
        if (len(first_line) <= 100 and 
            len(first_line) > 0 and  # Ensure line has content
            not first_line[-1] in '.!?' and 
            first_line[0].isupper() and
            not first_line.lower().startswith(('the ', 'this ', 'just ', 'test '))):
            return first_line
            
        return os.path.splitext(filename)[0]

    def _extract_pdf_text(self, file_path: str) -> Tuple[str, str]:
        """Extract text and metadata from PDF file."""
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            # Get title from metadata if available
            title = reader.metadata.get('/Title', '')
            
            # If no metadata title, try to get from first page content
            if not title and total_pages > 0:
                first_page_text = reader.pages[0].extract_text()
                title = self._get_title_from_content(first_page_text, os.path.basename(file_path))
            elif not title:
                title = os.path.splitext(os.path.basename(file_path))[0]

            # Extract full text
            full_text = "\n".join(page.extract_text() for page in reader.pages)
            
            logger.info(f"Extracted text from PDF: {len(full_text)} characters")
            return title, full_text
            
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

    def _extract_docx_text(self, file_path: str) -> Tuple[str, str]:
        """Extract text and metadata from DOCX file."""
        try:
            # Convert DOC to DOCX if needed
            if file_path.endswith('.doc'):
                logger.info(f"Converting DOC to DOCX: {file_path}")
                docx_path = self._convert_doc_to_docx(file_path)
            else:
                docx_path = file_path
            
            # Process DOCX file
            try:
                doc = Document(docx_path)
            except Exception as e:
                logger.error(f"Failed to open DOCX file: {str(e)}")
                raise ValueError(f"Failed to open DOCX file: {str(e)}")
            
            if not hasattr(doc, 'paragraphs'):
                raise ValueError("Invalid DOCX file: document has no paragraphs")
                
            # Get title from document properties if available
            try:
                title = doc.core_properties.title if hasattr(doc, 'core_properties') and doc.core_properties and doc.core_properties.title else ''
            except Exception as e:
                logger.warning(f"Error accessing document properties: {str(e)}")
                title = ''
            
            # Extract text first so we can use it for title extraction if needed
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = "\n".join(paragraphs)
            
            # If no title in properties, try to get from content
            if not title:
                title = self._get_title_from_content(full_text, os.path.basename(file_path))
            
            logger.info(f"Extracted text from DOCX: {len(full_text)} characters")
            
            # Clean up temporary file if it was converted
            if file_path.endswith('.doc') and os.path.exists(docx_path):
                try:
                    os.remove(docx_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary DOCX file: {str(e)}")
            
            return title, full_text
            
        except Exception as e:
            raise ValueError(f"Error processing Word document: {str(e)}")

    def _convert_doc_to_docx(self, doc_path: str) -> str:
        """Convert DOC file to DOCX format."""
        try:
            # Create temporary directory for conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate output filename
                docx_name = os.path.splitext(os.path.basename(doc_path))[0] + '.docx'
                docx_path = os.path.join(temp_dir, docx_name)
                
                # Run LibreOffice conversion
                result = subprocess.run(
                    ['soffice', '--headless', '--convert-to', 'docx', '--outdir', temp_dir, doc_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Verify conversion
                if not os.path.exists(docx_path):
                    raise ValueError(
                        f"LibreOffice conversion failed. Expected file not found at {docx_path}. "
                        f"Command output: {result.stdout}. "
                        f"Error output: {result.stderr}"
                    )
                
                return docx_path
                
        except subprocess.CalledProcessError as e:
            raise ValueError(f"DOC to DOCX conversion failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error during DOC to DOCX conversion: {str(e)}")

    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document file into chunks with metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get original filename and extension for metadata
        original_filename = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Extract text based on file type
        if file_ext == '.pdf':
            title, full_text = self._extract_pdf_text(file_path)
        elif file_ext in ['.doc', '.docx']:
            title, full_text = self._extract_docx_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
            
        # Analyze document structure and content
        logger.info(f"Analyzing document: {title}")
        analysis = self.analyzer.analyze_document(full_text, title)
        logger.info(f"Document analysis complete")
        
        # Base metadata for all chunks
        base_metadata = {
            'source_name': original_filename,
            'title': title,
            'file_type': file_ext.lstrip('.'),
            'classification': analysis['classification'],
            'toc': json.dumps(analysis['toc'])  # Convert TOC to JSON string
        }
        
        # Process each section into chunks
        chunks = []
        for section in analysis['sections']:
            section_chunks = self._split_section(section, base_metadata)
            chunks.extend(section_chunks)
            
        logger.info(f"Processed document into {len(chunks)} chunks")
        return chunks

    def __del__(self):
        """Clean up by removing observer when object is destroyed."""
        try:
            settings_manager.remove_observer(self._handle_settings_change)
        except:
            pass  # Ignore errors during cleanup

class DocumentStore:
    """Manages document storage and retrieval with atomic operations."""
    
    def __init__(self, processor: Optional[DocumentProcessor] = None):
        self.processor = processor or DocumentProcessor()
        self.db = VectorDatabase()
        self.embedding_generator = EmbeddingGenerator()
        self._processing_states = {}  # Track processing states
    
    def get_processing_state(self, filename: str) -> Optional[ProcessingState]:
        """Get the current processing state for a document."""
        return self._processing_states.get(filename)
    
    def _update_processing_state(self, filename: str, state: ProcessingState) -> None:
        """Update the processing state for a document."""
        self._processing_states[filename] = state
        logger.info(f"Updated processing state for {filename}: {state}")
    
    def process_and_store_document(self, file_path: str) -> ProcessingState:
        """
        Process and store a document with atomic operations.
        This is the single point of entry for document processing.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ProcessingState: Final state of document processing
        """
        filename = os.path.basename(file_path)
        state = ProcessingState(status='processing')
        self._update_processing_state(filename, state)
        
        try:
            # 1. Process document into chunks
            logger.info(f"Processing document: {filename}")
            chunks = self.processor.process_document(file_path)
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            # Update state with chunk information
            state.chunk_count = len(chunks)
            state.total_chunks = len(chunks)
            state.source_name = filename
            # Add classification and TOC information from first chunk's metadata
            if chunks:
                state.classification = chunks[0].metadata['classification']
                state.toc = json.loads(chunks[0].metadata['toc'])  # Parse JSON string back to list
            self._update_processing_state(filename, state)
            
            # 2. Generate embeddings (single point of embedding generation)
            logger.info("Generating embeddings...")
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            # 3. Prepare documents for database
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "embedding": embeddings[i],
                    **chunk.metadata
                }
                documents.append(doc)
            
            # 4. Delete any existing document with same source name
            logger.info(f"Checking for existing document: {state.source_name}")
            existing_chunks = self.db.get_document_chunks(state.source_name)
            if existing_chunks:
                logger.info(f"Found existing document with {len(existing_chunks)} chunks. Removing...")
                existing_ids = [chunk['id'] for chunk in existing_chunks]
                self.db.collection.delete(ids=existing_ids)
                logger.info(f"Deleted {len(existing_ids)} existing chunks")
            
            # 5. Atomic database operation
            logger.info("Adding documents to database...")
            self.db.add_documents(documents)
            
            # 6. Verify storage and chunk consistency
            stored_chunks = self.db.get_document_chunks(state.source_name)
            if not stored_chunks:
                raise ValueError(f"Storage verification failed - no chunks found for {filename}")
            
            if len(stored_chunks) != len(chunks):
                raise ValueError(
                    f"Storage verification failed - chunk count mismatch for {filename}. "
                    f"Expected {len(chunks)}, found {len(stored_chunks)}"
                )
            
            # Verify chunk indices and total_chunks are consistent
            chunk_indices = sorted(int(chunk.get('chunk_index', -1)) for chunk in stored_chunks)
            expected_indices = list(range(len(chunks)))
            if chunk_indices != expected_indices:
                raise ValueError(
                    f"Storage verification failed - inconsistent chunk indices for {filename}. "
                    f"Expected sequential indices 0-{len(chunks)-1}, got {chunk_indices}"
                )
            
            # Verify total_chunks matches actual count
            for chunk in stored_chunks:
                if int(chunk.get('total_chunks', 0)) != len(chunks):
                    raise ValueError(
                        f"Storage verification failed - total_chunks mismatch for {filename}. "
                        f"Expected {len(chunks)}, got {chunk.get('total_chunks')}"
                    )
            
            # Update final state
            state.status = 'completed'
            self._update_processing_state(filename, state)
            logger.info(f"Successfully processed and stored {filename}")
            
            return state
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing document {filename}: {error_msg}")
            state.status = 'error'
            state.error = error_msg
            self._update_processing_state(filename, state)
            raise
    
    def get_documents(self) -> List[Dict[str, str]]:
        """Return all document chunks."""
        return self.db.get_all_documents()
    
    def get_document_info(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        chunks = self.db.get_document_chunks(source_name)
        if not chunks:
            return None
            
        # Parse TOC JSON string back to list
        toc = chunks[0].get('toc')
        if isinstance(toc, str):
            toc = json.loads(toc)
            
        return {
            'source_name': source_name,
            'title': chunks[0].get('title', ''),
            'chunk_count': len(chunks),
            'total_chunks': chunks[0].get('total_chunks', len(chunks)),
            'classification': chunks[0].get('classification'),
            'toc': toc
        }

# Initialize global document store
document_store = DocumentStore()

# Expose simplified API
def process_document(file_path: str) -> List[Dict[str, str]]:
    """Process a document and return its chunks."""
    state = document_store.process_and_store_document(file_path)
    if state.status == 'error':
        raise ValueError(f"Document processing failed: {state.error}")
    return document_store.db.get_document_chunks(state.source_name)

def get_documents() -> List[Dict[str, str]]:
    """Get all documents from the store."""
    return document_store.get_documents()

def get_processing_state(filename: str) -> Optional[ProcessingState]:
    """Get processing state for a document."""
    return document_store.get_processing_state(filename)
