# RAG-based Document Query System

[Previous sections remain the same until Features...]

## Features

- Document Processing:
  - Support for PDF (.pdf) and Word (.doc, .docx) file uploads
  - Intelligent document title extraction:
    - PDF: Metadata, content analysis, filename fallback
    - DOCX: Core properties, filename fallback
  - Intelligent document chunking by paragraphs
  - Automatic text extraction and preprocessing
  - Document embedding generation using Sentence Transformers
  
- Storage and Retrieval:
  - Vector storage and retrieval using ChromaDB
  - Semantic search for finding relevant document sections
  - Continuous document ID management for multiple uploads
  - Efficient chunk storage and retrieval
  - Document listing with titles and chunk statistics
  - Ordered chunk retrieval by document
  
- Advanced Search and Retrieval:
  - Natural language query parsing and preprocessing
  - Query embedding generation for semantic matching
  - Vector similarity search with configurable results
  - LLM-based re-ranking for improved relevance
  - Weighted scoring combining vector similarity and LLM relevance
  - Robust error handling and fallback mechanisms
  
- Query Processing:
  - Natural language querying of document content
  - Context-aware responses using OpenAI's GPT models
  - Source citations for response traceability
  - Configurable number of relevant chunks for context
  
- System Features:
  - RESTful API for easy integration
  - Document exploration and verification capabilities
  - Automatic file cleanup after processing
  - Comprehensive error handling and logging
  - Type hints for better code maintainability

[Project Structure section remains the same...]

## API Endpoints

The system provides a comprehensive REST API for document management and querying:

### Document Management

1. **Upload Document** - `POST /upload`
   - Accepts PDF (.pdf) and Word (.doc, .docx) files
   - Automatically extracts document title
   - Processes and chunks the document
   - Generates embeddings and stores in vector database
   ```bash
   curl -X POST -F "file=@path/to/document.pdf" http://localhost:3000/upload
   ```

2. **List Documents** - `GET /documents`
   - Retrieve all processed document chunks
   - Shows chunk IDs, titles, and content
   ```bash
   curl http://localhost:3000/documents
   ```

3. **List Document Names** - `GET /document-names`
   - Get list of unique document names with titles and statistics
   - Shows chunk counts and total chunks per document
   ```bash
   curl http://localhost:3000/document-names
   # Returns:
   # [
   #   {
   #     "source_name": "example.pdf",
   #     "title": "Example Document Title",
   #     "chunk_count": 10,
   #     "total_chunks": 10
   #   }
   # ]
   ```

4. **Get Document Chunks** - `GET /document-chunks/<source_name>`
   - Retrieve all chunks for a specific document
   - Returns chunks in correct order with metadata
   ```bash
   curl http://localhost:3000/document-chunks/example.pdf
   # Returns:
   # [
   #   {
   #     "id": "chunk1",
   #     "text": "Content of chunk 1",
   #     "title": "Example Document Title",
   #     "chunk_index": 0,
   #     "total_chunks": 10
   #   },
   #   ...
   # ]
   ```

### System Information

1. **Get Collection Metadata** - `GET /metadata`
   - Retrieve metadata about the ChromaDB collection
   - Shows collection name, document count, and metadata
   ```bash
   curl http://localhost:3000/metadata
   ```

2. **Get Vector Documents** - `GET /vector-documents`
   - Retrieve all documents with embeddings and metadata
   - Shows document IDs, titles, embeddings, and metadata
   ```bash
   curl http://localhost:3000/vector-documents
   ```

### Query Interface

1. **Query Documents** - `POST /chat`
   - Submit natural language queries about document content
   - Returns AI-generated responses with source citations including document titles
   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"query":"What are the main points of the document?"}' \
        http://localhost:3000/chat
   ```

## Components

### DocumentProcessor
- Handles document uploads and processing
- Supports PDF and Word documents
- Intelligent title extraction:
  - PDF: Metadata, content analysis, filename fallback
  - DOCX: Core properties, filename fallback
- Text extraction and cleaning
- Document chunking and metadata generation
- Empty document handling
- Comprehensive error handling

### VectorDatabase
- ChromaDB integration for efficient storage
- Semantic similarity search capabilities
- Document listing with titles and chunk retrieval
- Maintains document metadata and statistics
- Handles batch operations
- Provides collection metadata and document retrieval
- Robust error handling and fallback mechanisms

[Rest of the file remains the same...]
