"""Document processing for the RAG application with advanced chunking strategies."""
import os
import uuid
from typing import List, Dict, Optional, BinaryIO, Union
from dataclasses import dataclass
import re
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import logging
import warnings
import subprocess
import tempfile
from .database import VectorDatabase
from .embedding import EmbeddingGenerator

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
    id: str  # Changed from int to str for UUID-based IDs
    text: str
    metadata: Dict

class DocumentProcessor:
    """Handles document processing with advanced chunking strategies."""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 length_function: str = "char"):
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

    def _clean_title(self, title: str) -> str:
        """Clean and normalize document title."""
        if not title:
            return ""
        
        # Convert to string if not already
        title = str(title)
        
        # Fix encoding issues
        from ftfy import fix_text
        title = fix_text(title)
        
        # Remove extra spaces, including within words
        title = re.sub(r'\s+', ' ', title)  # Normalize all whitespace to single space
        title = re.sub(r'(?<=\w)\s+(?=\w)', '', title)  # Remove spaces between parts of words
        title = title.strip()
        
        return title
    
    def _get_base_metadata(self, file_path: str, title: Optional[str] = None) -> Dict[str, str]:
        """Get base metadata for a document."""
        # Get filename and extension
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()[1:]  # Remove the dot
        
        # If no title provided, use filename without extension
        if not title:
            title = os.path.splitext(filename)[0]
        
        return {
            'source_name': filename,
            'file_type': file_ext,
            'title': title,
            'section_type': 'content'  # Default section type
        }
    
    def _generate_chunk_id(self, source_name: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk that includes the source name."""
        # Create a deterministic but unique ID based on source name and chunk index
        unique_id = f"{source_name}_{chunk_index}"
        # Use UUID5 with a namespace UUID to generate a consistent hash
        namespace_uuid = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID namespace for URLs
        return str(uuid.uuid5(namespace_uuid, unique_id))
    
    def _extract_pdf_text(self, file_path: str) -> List[Dict[str, str]]:
        """Extract text from PDF with metadata."""
        result = []
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            try:
                reader = PdfReader(file_path, strict=False)
                
                # Extract document title from metadata
                title = None
                if hasattr(reader, 'metadata') and reader.metadata and '/Title' in reader.metadata:
                    title = reader.metadata['/Title']
                    if isinstance(title, bytes):
                        title = title.decode('utf-8', errors='ignore')
                    title = self._clean_title(title)
                
                # Get base metadata with title if found
                base_metadata = self._get_base_metadata(file_path, title)
                logger.info(f"PDF metadata: {base_metadata}")
                
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
                                    section_metadata = base_metadata.copy()
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
                            section_metadata = base_metadata.copy()
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
                            **base_metadata,
                            'section_type': 'content',
                            'section_title': '',
                            'page_number': 1,
                            'page_size': (0, 0),
                            'rotation': 0
                        }
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing PDF {file_path}: {str(e)}")
                return []
    
    def _extract_docx_text(self, file_path: str) -> List[Dict[str, str]]:
        """Extract text from DOCX/DOC with metadata and structure."""
        sections = []
        converted_file = None
        
        try:
            # If it's a .doc file, convert to .docx first
            if file_path.lower().endswith('.doc'):
                converted_file = self._convert_doc_to_docx(file_path)
                file_path = converted_file
                logger.info(f"Converted .doc to .docx: {file_path}")
            
            doc = Document(file_path)
            
            # Extract document title from core properties
            title = None
            if hasattr(doc, 'core_properties') and doc.core_properties:
                title = doc.core_properties.title
                if title:
                    title = self._clean_title(title)
            
            # Get base metadata
            base_metadata = self._get_base_metadata(file_path, title)
            
            # Create at least one section even if there's no content
            current_section = {'text': [], 'metadata': base_metadata}
            
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
                    section_metadata = base_metadata.copy()
                    section_metadata.update({
                        'section_type': paragraph.style.name,
                        'section_title': paragraph.text,
                        'heading_level': int(paragraph.style.name[-1]) 
                        if paragraph.style.name.startswith('Heading') 
                        else 0
                    })
                    
                    current_section = {
                        'text': [paragraph.text],
                        'metadata': section_metadata
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
            if converted_file:
                try:
                    os.remove(converted_file)
                except:
                    pass
    
    def process_document(self, file_or_path: Union[str, BinaryIO]) -> List[DocumentChunk]:
        """Process a document and return chunks with metadata."""
        if isinstance(file_or_path, str):
            file_ext = os.path.splitext(file_or_path)[1].lower()
            file_name = os.path.basename(file_or_path)
        else:
            raise ValueError("File objects not supported - save to disk first")
        
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
        current_chunk = 0
        
        logger.info(f"\nFound {len(sections)} sections to process")
        
        # Handle empty documents
        if not sections:
            logger.warning("No sections found in document")
            base_metadata = self._get_base_metadata(file_name)
            chunks.append(DocumentChunk(
                id=self._generate_chunk_id(file_name, 0),
                text="",
                metadata={
                    **base_metadata,
                    'chunk_size': 0,
                    'token_count': 0,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            ))
            return chunks
        
        # Calculate total chunks for non-empty documents
        total_chunks = 0
        for section in sections:
            text = section['text']
            if text:
                section_chunks = self.text_splitter.split_text(text)
                total_chunks += len(section_chunks)
                logger.info(f"Section will generate {len(section_chunks)} chunks")
            else:
                total_chunks += 1
        
        logger.info(f"Total chunks to be created: {total_chunks}")
        
        for section_idx, section in enumerate(sections, 1):
            text = section['text']
            base_metadata = section['metadata']
            
            logger.info(f"\n{'-'*80}\nProcessing Section {section_idx}/{len(sections)}")
            logger.info(f"Section Title: {base_metadata.get('section_title', 'Untitled')}")
            logger.info(f"Section Type: {base_metadata.get('section_type', 'Unknown')}")
            logger.info(f"Section Text Length: {len(text)}")
            
            if not text:
                logger.info("Empty section, creating empty chunk")
                chunks.append(DocumentChunk(
                    id=self._generate_chunk_id(file_name, current_chunk),
                    text="",
                    metadata={
                        **base_metadata,
                        'chunk_size': 0,
                        'token_count': 0,
                        'chunk_index': current_chunk,
                        'total_chunks': total_chunks
                    }
                ))
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
                        id=self._generate_chunk_id(file_name, current_chunk),
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
                    logger.info(f"\nChunk {chunk_idx} (Section {section_idx}, Chunk {chunk_idx}):")
                    logger.info("-" * 40)
                    logger.info("Content:")
                    logger.info(f"{chunk_text[:200]}..." if len(chunk_text) > 200 else chunk_text)
                    logger.info("-" * 40)
                    logger.info(f"Character Count: {len(chunk_text)}")
                    logger.info(f"Token Count: {len(self.tokenizer.encode(chunk_text))}")
                    logger.info("Metadata:")
                    logger.info(str(chunk.metadata))
                    logger.info("=" * 40)
                    
                    current_chunk += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Document processing complete")
        logger.info(f"Total sections processed: {len(sections)}")
        logger.info(f"Total chunks created: {len(chunks)}")
        
        if chunks:
            logger.info(f"Average chunk size: {sum(len(c.text) for c in chunks)/len(chunks):.2f} characters")
            logger.info(f"Average tokens per chunk: {sum(len(self.tokenizer.encode(c.text)) for c in chunks)/len(chunks):.2f}")
        else:
            logger.warning("No chunks created (empty document)")
            
        logger.info(f"{'='*80}\n")
        
        return chunks

# Global document store
class DocumentStore:
    """Manages document storage and retrieval."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.db = VectorDatabase()
        self.embedding_generator = EmbeddingGenerator()
    
    def add_document(self, file_path: str) -> None:
        """Process and add a document to the store."""
        try:
            logger.info(f"\nAdding document to store: {file_path}")
            
            # Process the document into chunks
            chunks = self.processor.process_document(file_path)
            logger.info(f"Generated {len(chunks)} chunks")
            
            if chunks:
                logger.info(f"First chunk metadata: {chunks[0].metadata}")
                logger.info(f"Using source_name: {chunks[0].metadata.get('source_name', 'Unknown')}")
            
            # Extract texts for embedding generation
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings for all texts
            embeddings = self.embedding_generator.generate_embeddings(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Convert chunks to the format expected by VectorDatabase
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "id": chunk.id,  # Now using UUID-based ID
                    "text": chunk.text,
                    "embedding": embeddings[i],
                    **chunk.metadata
                }
                documents.append(doc)
                logger.info(f"Chunk {i+1}/{len(chunks)}: ID={doc['id']}, source_name={doc.get('source_name', 'Unknown')}")
            
            # Add documents with embeddings to ChromaDB
            logger.info("Adding documents to ChromaDB...")
            self.db.add_documents(documents)
            logger.info("Documents added successfully")
            
            # Verify document was added
            source_name = chunks[0].metadata.get('source_name', 'Unknown')
            doc_chunks = self.db.get_document_chunks(source_name)
            if not doc_chunks:
                raise ValueError(f"Could not find {source_name} in vector database after adding")
            logger.info(f"Added {source_name} to vector database")
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def get_documents(self) -> List[Dict[str, str]]:
        """Return all document chunks."""
        # Get documents directly from ChromaDB
        return self.db.get_all_documents()

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
