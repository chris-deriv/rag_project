# RAG System Architecture

This document details the architectural design and implementation of the RAG (Retrieval-Augmented Generation) system.

## Document Processing Architecture

### Text Extraction

The system employs different strategies for extracting text based on document type:

1. **PDF Text Extraction**
```python
def _extract_pdf_text(self, file_or_path: Union[str, BinaryIO]) -> List[Dict[str, str]]:
```
- Uses PyPDF (PdfReader) for extraction
- Processes documents page by page
- Extracts metadata including:
  - Title
  - Page numbers
  - Page size
  - Rotation
- Identifies section headers using regex patterns
- Groups content into logical sections with metadata

2. **Word Document Extraction**
```python
def _extract_docx_text(self, file_path: str) -> List[Dict[str, str]]:
```
- Uses python-docx for DOCX files
- Automatically converts DOC to DOCX using LibreOffice
- Preserves document structure:
  - Headers/titles based on paragraph styles
  - Section grouping
  - Document formatting
- Maintains metadata about section types and hierarchy

3. **DOC to DOCX Conversion**
```python
def _convert_doc_to_docx(self, file_path: str) -> str:
```
- Uses LibreOffice in headless mode
- Command-line conversion process
- Creates temporary files for processing
- Handles cleanup after conversion

### Text Chunking Strategy

The system uses a sophisticated approach to break documents into meaningful chunks while preserving context:

1. **Text Splitter Configuration**
```python
RecursiveCharacterTextSplitter(
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
    chunk_size=150,     # Default chunk size
    chunk_overlap=50,   # Overlap between chunks
)
```

2. **Chunking Process Flow**
- Text cleaning and normalization
- Hierarchical splitting:
  1. Try to split at paragraph boundaries
  2. Fall back to sentence boundaries
  3. Use punctuation marks as delimiters
  4. Split by words if necessary
- Overlap maintenance between chunks
- Metadata preservation

3. **Text Cleaning Process**
```python
def _clean_text(self, text: str) -> str:
```
- Fixes text encoding issues (using ftfy)
- Handles hyphenation at line breaks
- Normalizes whitespace
- Fixes punctuation spacing
- Removes redundant whitespace

4. **Chunk Metadata**
Each chunk includes:
- Source file information
- Section type and title
- Position in document
- Size metrics:
  - Character count
  - Token count
- Original document metadata

### Quality Control Measures

1. **Chunk Quality**
- Empty chunks are filtered out
- Chunks start with complete words/sentences
- Leading punctuation is removed
- Context is preserved through overlap

2. **Logging and Monitoring**
- Detailed logging of processing steps
- Chunk statistics:
  - Total chunks created
  - Average chunk size
  - Token distribution
- Error handling and reporting

3. **Storage and Persistence**
- Chunks are saved with continuous IDs
- Document metadata is preserved
- Efficient retrieval system
- Disk-based persistence for reliability

## Vector Database Integration

The system uses ChromaDB for vector storage and retrieval:

1. **Document Storage**
```python
def add_documents(self, documents: List[Dict[str, Any]]) -> None:
```
- Stores document embeddings
- Maintains metadata
- Handles batch operations

2. **Metadata Management**
```python
def get_metadata(self) -> Dict[str, Any]:
```
- Collection information
- Document counts
- Collection metadata

3. **Document Retrieval**
```python
def get_all_documents(self) -> Dict[str, Any]:
```
- Returns all documents
- Includes embeddings
- Preserves metadata

## API Endpoints

The system exposes several REST endpoints:

1. **Document Management**
- `POST /upload` - Upload and process new documents
- `GET /documents` - Retrieve processed documents
- `GET /metadata` - Get collection metadata
- `GET /vector-documents` - Get documents with embeddings

2. **Query Interface**
- `POST /chat` - Query documents using natural language
- Returns AI-generated responses with source citations

## Dependencies

Key libraries and tools:
- PyPDF (pypdf) - PDF processing
- python-docx - Word document processing
- LibreOffice - DOC to DOCX conversion
- langchain - Text splitting
- tiktoken - Token counting
- ChromaDB - Vector storage
- Flask - Web API
