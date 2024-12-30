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

### Query Interface
- `POST /chat`
  - Query documents using natural language
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

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables
4. Run the application:
   ```bash
   python src/app.py
   ```

## Usage

1. Start the server
2. Upload documents via POST /upload
3. Query documents via POST /chat
4. Explore documents using the document management endpoints

## Development

- Run tests:
  ```bash
  python -m pytest tests/
  ```
- Format code:
  ```bash
  black .
  ```
