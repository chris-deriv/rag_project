# RAG System Architecture

This document details the architectural design and implementation of the RAG (Retrieval-Augmented Generation) system.

## Document Processing Architecture

### Document Title Extraction

The system implements a robust title extraction strategy for different document types:

1. **PDF Documents**
```python
def _extract_pdf_text(self, file_or_path: Union[str, BinaryIO]) -> List[Dict[str, str]]:
```
- Primary: Extracts from PDF metadata (/Title)
- Secondary: Intelligent first page content analysis
  - Examines first 3 lines
  - Identifies title-like text based on:
    - Capitalization
    - Length constraints
    - Sentence ending patterns
    - Case formatting
- Fallback: Uses filename without extension

2. **DOCX Documents**
```python
def _extract_docx_text(self, file_path: str) -> List[Dict[str, str]]:
```
- Primary: Extracts from core properties
- Handles empty/missing properties
- Fallback: Uses filename without extension

3. **Metadata Structure**
```python
{
    "title": str,          # Extracted document title
    "source_file": str,    # Original filename
    "file_type": str,     # Document type (pdf/docx)
    "section_type": str,   # Content organization
    "section_title": str,  # Section heading if any
    ...
}
```

## Vector Database Integration

The system uses ChromaDB for vector storage and retrieval:

1. **Document Storage**
```python
def add_documents(self, documents: List[Dict[str, Any]]) -> None:
```
- Stores document embeddings
- Maintains metadata including:
  - Source document name
  - Document title
  - Chunk index and total chunks
  - Document text
- Handles batch operations

2. **Document Listing**
```python
def list_document_names(self) -> List[Dict[str, Any]]:
```
- Groups chunks by source document
- Calculates chunk statistics:
  - Number of chunks per document
  - Total chunks in document
- Handles missing metadata gracefully
- Returns document-level overview

3. **Chunk Retrieval**
```python
def get_document_chunks(self, source_name: str) -> List[Dict[str, Any]]:
```
- Retrieves all chunks for a document
- Orders chunks by index
- Includes chunk metadata:
  - Chunk text
  - Document title
  - Position in document
  - Total chunks
- Handles missing documents

4. **Metadata Management**
```python
def get_metadata(self) -> Dict[str, Any]:
```
- Collection information
- Document counts
- Collection metadata

5. **Document Retrieval**
```python
def get_all_documents(self) -> Dict[str, Any]:
```
- Returns all documents
- Includes embeddings and metadata
- Supports selective field inclusion

### Document Organization

1. **Chunk Metadata Structure**
```python
{
    "text": str,           # Chunk content
    "title": str,          # Document title
    "source_name": str,    # Original document name
    "chunk_index": int,    # Position in document
    "total_chunks": int,   # Total chunks in document
    "section_type": str,   # Content organization
    "section_title": str   # Section heading if any
}
```

2. **Document Statistics**
```python
{
    "source_name": str,    # Document identifier
    "title": str,         # Document title
    "chunk_count": int,    # Actual chunks stored
    "total_chunks": int    # Expected total chunks
}
```

3. **Error Handling**
- Missing metadata defaults:
  - Unknown source name
  - Empty title uses filename
  - Zero chunk index
  - Single chunk total
- Document not found returns empty list
- Invalid metadata handled gracefully

## Search and Retrieval Architecture

The system implements a sophisticated search pipeline with deterministic behavior:

1. **Query Processing**
```python
def parse_query(self, query: str) -> Dict[str, Any]:
```
- Standardizes query format
- Preserves special characters
- Handles whitespace consistently

2. **Embedding Generation**
```python
def generate_query_embedding(self, query: str) -> np.ndarray:
```
- Uses sentence-transformers model
- Generates consistent embeddings
- Handles errors gracefully

3. **Similarity Search**
```python
def perform_similarity_search(self, query_embedding: np.ndarray, n_results: int) -> Dict[str, Any]:
```
- Deterministic ordering:
  - Primary sort by distance
  - Secondary sort by document ID for ties
- Consistent result structure
- Configurable result count

4. **Result Reranking**
```python
def rerank_results(self, query: str, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
```
- LLM-based relevance scoring with caching:
  - Caches scores per query-text pair
  - Ensures consistent scoring
  - Only queries LLM for new texts
- Deterministic ordering:
  - Primary sort by combined score
  - Secondary sort by document ID
- Score combination:
  - Vector similarity (40%)
  - LLM relevance (60%)
- Fallback mechanisms for LLM failures

5. **Result Structure**
```python
{
    'id': str,                # Document chunk ID
    'text': str,              # Chunk content
    'metadata': Dict,         # Full chunk metadata
    'similarity_score': float, # Vector similarity
    'relevance_score': float, # LLM relevance
    'combined_score': float   # Final ranking score
}
```

## Query Interface

The system implements a sophisticated response generation pipeline:

1. **Query Processing**
```python
def parse_query(self, query: str) -> Dict[str, Any]:
```
- Standardizes query format
- Preserves special characters
- Handles whitespace consistently

2. **Response Generation**
```python
def generate_response(self, context: str, query: str) -> str:
```
- Comprehensive response generation:
  - Detailed explanations with examples and analogies
  - Well-structured information organization
  - Thorough coverage of relevant aspects
  - Enhanced context synthesis
- Configuration for quality:
  - Extended response length (1000 tokens)
  - Balanced temperature (0.3) for natural yet consistent responses
  - Fixed seed (42) for reproducibility
- Response caching:
  - Deterministic cache keys
  - Normalized query and context matching
  - Efficient cache retrieval

3. **Source-Cited Responses**
```python
def generate_response_with_sources(self, contexts: List[dict], query: str) -> str:
```
- Enhanced source integration:
  - Clear source citations using [Source X] notation
  - Synthesis of information across multiple sources
  - Detailed explanations of source relevance
  - Identification of missing information
- Structured output:
  - Organized sections for clarity
  - Proper formatting for readability
  - Clear delineation of source material
- Cache management:
  - Deterministic context formatting
  - Consistent source ordering
  - Efficient cache lookup

4. **Result Structure**
```python
{
    'response': str,           # Generated response text
    'sources': List[Dict],     # Source citations and metadata
    'cached': bool,            # Cache hit indicator
    'generation_time': float   # Response generation duration
}
```

## API Endpoints

The system exposes several REST endpoints:

1. **Document Management**
- `POST /upload` - Upload and process new documents
- `GET /documents` - Retrieve processed documents
- `GET /document-names` - List unique documents with statistics
  ```python
  # Response format:
  [
      {
          "source_name": str,     # Document name
          "title": str,           # Document title
          "chunk_count": int,     # Number of chunks
          "total_chunks": int     # Total expected chunks
      }
  ]
  ```
- `GET /document-chunks/<source_name>` - Get ordered chunks for a document
  ```python
  # Response format:
  [
      {
          "id": str,             # Chunk identifier
          "text": str,           # Chunk content
          "title": str,          # Document title
          "chunk_index": int,    # Position in document
          "total_chunks": int    # Total chunks in document
      }
  ]
  ```
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
- sentence-transformers - Text embedding generation
- numpy - Numerical operations
- openai - GPT model integration
- chromadb - Vector storage
- Flask - Web API
