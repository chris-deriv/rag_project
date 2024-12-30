# RAG-based Document Query System

[Previous sections remain the same until Features...]

## Features

- Document Processing:
  - Support for PDF (.pdf) and Word (.doc, .docx) file uploads
  - Intelligent document title extraction:
    - PDF: Metadata, content analysis, filename fallback
    - DOCX: Core properties, filename fallback
  - Intelligent document chunking by paragraphs
  - Section metadata extraction:
    - Section titles from headings and document structure
    - Section types (heading, body, content)
    - Preserves document hierarchy and organization
  - Automatic text extraction and preprocessing
  - Document embedding generation using Sentence Transformers
  
- Storage and Retrieval:
  - Vector storage and retrieval using ChromaDB
  - Rich metadata storage including:
    - Document titles and source information
    - Section titles and types
    - Chunk position and organization
  - Semantic search for finding relevant document sections
  - Continuous document ID management for multiple uploads
  - Efficient chunk storage and retrieval
  - Document listing with titles and chunk statistics
  - Ordered chunk retrieval by document
  
- Advanced Search and Retrieval:
  - Natural language query parsing and preprocessing
  - Context-aware search using section metadata:
    - Section-level relevance matching
    - Structural context preservation
    - Hierarchical document understanding
  - Query embedding generation for semantic matching
  - Vector similarity search with configurable results
  - LLM-based re-ranking for improved relevance
  - Weighted scoring combining vector similarity and LLM relevance
  - Robust error handling and fallback mechanisms
  - Title-based search with fuzzy matching:
    - Case-insensitive partial matching
    - Document filtering by title or filename
    - Combined semantic and title-based search
  
- Query Processing:
  - Natural language querying of document content
  - Comprehensive response generation using OpenAI's GPT models:
    - Detailed explanations with examples and analogies
    - Well-structured and organized responses
    - Thorough coverage of all relevant aspects
    - Enhanced context synthesis from multiple sources
  - Source citations for response traceability
  - Configurable response length (up to 1000 tokens)
  - Balanced temperature settings for natural yet consistent responses
  - Document-specific querying with title/filename filters
  
- System Features:
  - RESTful API for easy integration
  - Document exploration and verification capabilities
  - Automatic file cleanup after processing
  - Comprehensive error handling and logging
  - Type hints for better code maintainability

## API Endpoints

The system exposes several REST endpoints for document management and querying:

### Document Management
- `POST /upload`
  - Upload and process new documents
  - Returns document processing status and metadata

- `GET /documents`
  - Retrieve all processed documents with metadata
  - Includes section information and chunk organization

- `GET /document-names`
  - List unique documents with statistics
  - Returns:
    ```json
    [
      {
        "source_name": "document.pdf",
        "title": "Document Title",
        "chunk_count": 5,
        "total_chunks": 5
      }
    ]
    ```

- `GET /document-chunks/<source_name>`
  - Get ordered chunks for a specific document
  - Returns:
    ```json
    [
      {
        "id": "chunk_id",
        "text": "Chunk content",
        "title": "Document Title",
        "chunk_index": 0,
        "total_chunks": 5,
        "section_title": "Introduction",
        "section_type": "heading"
      }
    ]
    ```

- `GET /metadata`
  - Get collection metadata including document counts
  - Returns collection-level statistics

- `GET /search-titles?q=<query>`
  - Search for documents by title (fuzzy matching)
  - Returns:
    ```json
    [
      {
        "title": "Python Programming Guide",
        "source_name": "guide.pdf"
      }
    ]
    ```

### Query Interface
- `POST /chat`
  - Query documents using natural language
  - Optional document filtering by title or filename
  - Request format:
    ```json
    {
      "query": "What is this about?",
      "title": "python",        // Optional: filter by title (partial match)
      "source_name": "guide.pdf" // Optional: filter by filename (exact match)
    }
    ```
  - Returns AI-generated responses with source citations
  - Preserves document structure and section context in responses

## Environment Setup

The system requires several environment variables to be set:

```bash
OPENAI_API_KEY=your_openai_api_key
CHROMA_PERSIST_DIR=/path/to/storage
CHROMA_COLLECTION_NAME=your_collection_name
```

## Installation

### Local Development

1. Clone the repository
2. Set up data directories:
   ```bash
   ./setup_data_dirs.sh
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in .env file
5. Run the application:
   ```bash
   python src/app.py
   ```

### Docker Deployment

1. Clone the repository
2. Set up environment variables in .env file
3. Start the application:
   ```bash
   docker compose up -d
   ```
4. To stop and clean up:
   ```bash
   docker compose down -v
   ```

## Data Storage

### Local Development
- Data is stored in local directories under ./data/
- Cache directories are created by setup_data_dirs.sh
- Manual cleanup required if needed

### Docker Environment
- Data is stored in Docker named volumes
- Volumes are automatically managed by Docker
- Use `docker compose down -v` to clean up all data
- Uploads directory is mounted from local ./uploads/

## Usage

1. Start the server
2. Upload documents via POST /upload
3. Query documents via POST /chat
   - Use title/filename filters for targeted queries
   - Combine with semantic search for best results
4. Explore documents using:
   - GET /search-titles for finding documents
   - GET /document-chunks for viewing content
   - GET /document-names for listing all documents

## Development

- Run tests:
  ```bash
  python -m pytest tests/
  ```
- Format code:
  ```bash
  black .
