# RAG System Architecture

## System Overview

The system implements a sophisticated RAG (Retrieval Augmented Generation) pipeline with the following key components:

1. **Document Processing Pipeline**
- Input: PDF and Word documents (.pdf, .doc, .docx)
- Processing:
  * Intelligent chunking with RecursiveCharacterTextSplitter
  * Configurable chunk size (500) and overlap (50)
  * Document structure and metadata preservation
  * Title and section extraction
- Output: Structured document chunks with metadata

2. **Embedding Generation**
- Model: sentence-transformers (all-MiniLM-L6-v2)
- Features:
  * Centralized embedding generation
  * Batched processing
  * Document and query embedding consistency
  * Cosine similarity metric

3. **Vector Storage (ChromaDB)**
- Storage:
  * HNSW index for efficient similarity search
  * Rich metadata storage
  * Atomic operations
- Features:
  * Source-based filtering
  * Efficient chunk retrieval
  * Comprehensive metadata querying

4. **Search Architecture**
- Two-stage process:
  * Vector similarity search
  * LLM-based reranking
- Weighted scoring:
  * 40% vector similarity
  * 60% LLM relevance
- Performance optimizations:
  * Result caching
  * Batch processing
  * Fallback mechanisms

5. **Data Flow**
```
Document → Chunks → Embeddings → Vector DB
                                    ↓
Query → Embedding → Vector Search → Rerank → Response
```

6. **Key Optimizations**
- Caching:
  * LLM relevance scores
  * Response caching
  * Embedding caching
- Performance:
  * Batched operations
  * Efficient chunk management
  * Deterministic ordering
- Reliability:
  * Atomic operations
  * Error handling
  * Fallback mechanisms

## Document Processing Architecture

The system implements a centralized document processing pipeline with robust state tracking:

1. **Document Store**
```python
class DocumentStore:
    """Manages document storage and retrieval with atomic operations."""
    def process_and_store_document(self, file_path: str) -> ProcessingState:
```
- Single point of responsibility for document processing
- Atomic operations for document updates
- Consistent state tracking
- Centralized embedding generation

2. **Processing State Management**
```python
@dataclass
class ProcessingState:
    """Tracks the state of document processing."""
    status: str          # 'processing', 'completed', 'error'
    error: Optional[str] # Error message if failed
    source_name: str     # Document identifier
    chunk_count: int     # Current chunks processed
    total_chunks: int    # Expected total chunks
```
- Real-time processing status
- Error tracking
- Progress monitoring
- Atomic state updates

3. **Document Processing Pipeline**
```
Upload -> Process -> Generate Embeddings -> Store -> Verify -> Index
```
- Chunking with metadata preservation
- Single embedding generation point
- Atomic database operations
- Storage verification
- Document indexing

4. **Processing Flow**
```python
try:
    # 1. Process document into chunks
    chunks = processor.process_document(file_path)
    
    # 2. Generate embeddings (single point)
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_generator.generate_embeddings(texts)
    
    # 3. Prepare documents with embeddings
    documents = [
        {
            "id": chunk.id,
            "text": chunk.text,
            "embedding": embedding,
            **chunk.metadata
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    
    # 4. Atomic database operation
    vector_db.add_documents(documents)
    
    # 5. Verify storage
    stored_chunks = vector_db.get_document_chunks(source_name)
    if not stored_chunks:
        raise ValueError("Storage verification failed")
```

5. **Error Handling**
- Consistent error states
- Atomic rollbacks
- Clean temporary file handling
- Detailed error logging


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

3. **Title-Based Search**
```python
def search_titles(self, title_query: str) -> List[Dict[str, Any]]:
```
- Case-insensitive partial matching:
  - Converts query and titles to lowercase
  - Matches substrings within titles
- Deduplication:
  - Tracks unique title/source pairs
  - Prevents duplicate chunks from same document
- Empty query handling:
  - Returns empty list for empty/whitespace queries
- Result structure:
  ```python
  {
      'title': str,        # Original document title
      'source_name': str   # Document filename
  }
  ```

4. **Combined Search**
```python
def query(self, 
         query_embedding: np.ndarray, 
         n_results: int = 5,
         source_name: Optional[str] = None,
         title: Optional[str] = None) -> Dict[str, Any]:
```
- Metadata filtering:
  - Optional source filename (exact match)
  - Optional document title (partial match)
- ChromaDB integration:
  - Uses `where` clause for filtering
  - Combines with vector similarity search
- Filter operators:
  - `$contains` for partial title matching
  - Direct equality for source name
- Result ordering:
  - Primary sort by vector similarity
  - Secondary sort by document metadata

5. **Two-Stage Similarity Search**

The system implements a sophisticated two-stage search process combining vector similarity with LLM-based reranking:

Stage 1: Vector Similarity
```python
def perform_similarity_search(self, query_embedding: np.ndarray, n_results: int) -> Dict[str, Any]:
```
- Implementation:
  - Uses sentence-transformers to generate embeddings
  - ChromaDB performs cosine similarity search
  - Configured with "hnsw:space": "cosine" for accurate similarity calculations
- Deterministic ordering:
  - Primary sort by distance
  - Secondary sort by document ID for ties
- Performance optimizations:
  - Efficient vector indexing through ChromaDB
  - Cached embeddings for documents
  - Optimized similarity calculations

Stage 2: LLM Reranking
```python
def rerank_results(self, query: str, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
```
- LLM-based relevance scoring:
  - Prompts LLM to rate each chunk's relevance (0-10)
  - Considers direct answers (high relevance)
  - Evaluates related information (medium relevance)
  - Identifies tangential content (low relevance)
- Score combination:
  - Vector similarity weight: 40%
  - LLM relevance weight: 60%
  - Normalized to 0-1 scale
- Performance optimizations:
  - Score caching per query-text pair
  - Only queries LLM for uncached texts
  - Batch processing of uncached texts
- Fallback mechanisms:
  - Uses similarity scores if LLM fails
  - Handles parsing errors gracefully
  - Maintains deterministic ordering

7. **Result Structure**
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
- Extracts optional filters:
  - Document title (partial match)
  - Source filename (exact match)

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
  ```python
  # Response format:
  {
      "message": str,            # Upload status message
      "filename": str            # Original filename
  }
  ```

- `GET /upload-status/<filename>` - Get document processing status
  ```python
  # Response format:
  {
      "status": str,            # 'processing', 'completed', 'error'
      "error": Optional[str],   # Error message if failed
      "source_name": str,       # Document identifier
      "chunk_count": int,       # Current chunks processed
      "total_chunks": int       # Expected total chunks
  }
  ```

- `GET /documents` - Retrieve processed documents
- `GET /document-names` - List unique documents with statistics
  ```python
  # Response format:
  [
      {
          "source_name": str,     # Document name
          "title": str,           # Document title
          "chunk_count": int,     # Number of chunks
          "total_chunks": int,    # Total expected chunks
          "status": str,          # Processing status
          "error": Optional[str]  # Error message if failed
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
          "total_chunks": int,   # Total chunks in document
          "section_title": str,  # Section heading
          "section_type": str    # Content type
      }
  ]
  ```
- `GET /metadata` - Get collection metadata
- `GET /vector-documents` - Get documents with embeddings
- `GET /search-titles` - Search documents by title
  ```python
  # Response format:
  [
      {
          "title": str,          # Document title
          "source_name": str     # Document filename
      }
  ]
  ```

2. **Query Interface**
- `POST /chat` - Query documents using natural language
  ```python
  # Request format:
  {
      "query": str,              # User's question
      "title": str,              # Optional title filter
      "source_name": str         # Optional filename filter
  }
  ```
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
