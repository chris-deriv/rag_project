# RAG-based Document Query System

A production-ready implementation of a Retrieval-Augmented Generation (RAG) system for document indexing and querying. This system allows users to upload documents and query them using natural language, combining the power of vector databases for semantic search with large language models for generating contextually relevant responses.

## Features

- Document Processing:
  - Support for PDF and DOCX file uploads
  - Intelligent document chunking by paragraphs
  - Automatic text extraction and preprocessing
  - Document embedding generation using Sentence Transformers
  
- Storage and Retrieval:
  - Vector storage and retrieval using ChromaDB
  - Semantic search for finding relevant document sections
  - Continuous document ID management for multiple uploads
  - Efficient chunk storage and retrieval
  
- Query Processing:
  - Natural language querying of document content
  - Context-aware responses using OpenAI's GPT models
  - Source citations for response traceability
  - Configurable number of relevant chunks for context
  
- System Features:
  - RESTful API for easy integration
  - Automatic file cleanup after processing
  - Comprehensive error handling and logging
  - Type hints for better code maintainability

## Project Structure

```
│
├── config/
│   └── settings.py      # Configuration settings
│
├── src/
│   ├── documents.py     # Document processing and chunk management
│   ├── embedding.py     # Embedding generation using Sentence Transformers
│   ├── database.py      # Vector database operations with ChromaDB
│   ├── chatbot.py      # OpenAI integration for response generation
│   ├── app.py          # Core RAG application logic
│   └── web.py          # Flask API endpoints
│
├── uploads/            # Temporary storage for uploaded files
├── requirements.txt    # Project dependencies
└── README.md          # Documentation
```

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd src
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Testing

The project uses pytest for unit testing. The test suite covers core functionality including document processing, ID management, and file type validation.

### Running Tests

Run the full test suite:
```bash
python -m pytest tests/ -v
```

### Test Structure

```
tests/
├── __init__.py
└── test_documents.py    # Tests for document processing functionality
```

The test suite includes:
- File extension validation
- Document ID generation and continuity
- Document management (adding and retrieving)
- Support for different file types (PDF, DOCX)

Tests use pytest's monkeypatch fixture to mock file processing functions, allowing for thorough testing without requiring actual document files.

## Usage

The system provides a RESTful API for document management and querying:

### API Endpoints

1. **Upload Document** - `POST /upload`
   - Accepts PDF or DOCX files
   - Automatically processes and chunks the document
   - Generates embeddings and stores in vector database
   - Handles duplicate uploads and maintains continuous chunk IDs
   ```bash
   curl -X POST -F "file=@document.pdf" http://127.0.0.1:5000/upload
   ```

2. **Query Documents** - `POST /chat`
   - Submit natural language queries about document content
   - Returns AI-generated responses with relevant context
   - Includes source citations for transparency
   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"query":"What are the main points of the document?"}' \
        http://127.0.0.1:5000/chat
   ```

3. **List Documents** - `GET /documents`
   - Retrieve all processed document chunks
   - Shows chunk IDs and content
   - Useful for debugging and verification
   ```bash
   curl http://127.0.0.1:5000/documents
   ```

### Document Processing Flow

1. **Upload Phase**:
   - Document is uploaded and temporarily stored
   - Text is extracted based on file type (PDF/DOCX)
   - Content is split into logical chunks (paragraphs)
   - Each chunk receives a unique ID
   - Original file is removed after processing

2. **Indexing Phase**:
   - Embeddings are generated for each chunk
   - Chunks and embeddings are stored in ChromaDB
   - Vector database is updated for semantic search

3. **Query Phase**:
   - User query is converted to embedding
   - Similar chunks are retrieved from vector database
   - Context is provided to GPT model
   - Response is generated with source citations

## Components

### Document Processor
- Handles PDF and DOCX file formats
- Extracts text while preserving structure
- Implements intelligent paragraph-based chunking
- Manages continuous chunk IDs across multiple documents

### EmbeddingGenerator
- Uses Sentence Transformers for embedding generation
- Implements "all-MiniLM-L6-v2" model
- Provides consistent vector representations

### VectorDatabase
- ChromaDB integration for efficient storage
- Semantic similarity search capabilities
- Maintains document metadata
- Handles batch operations

### Chatbot
- OpenAI GPT integration
- Context-aware response generation
- Source citation inclusion
- Error handling and retry logic

### RAGApplication
- Coordinates component interactions
- Manages document indexing workflow
- Processes queries and generates responses
- Implements comprehensive logging

## Error Handling

The system includes robust error handling:
- File validation and type checking
- Processing error management
- API error responses
- Automatic cleanup of temporary files
- Detailed error logging

## Production Considerations

1. **Security**:
   - File type validation
   - Secure filename handling
   - Temporary file cleanup
   - Environment variable management

2. **Scalability**:
   - Efficient chunk management
   - Batch processing capabilities
   - Vector database optimization

3. **Maintenance**:
   - Comprehensive logging
   - Type hints throughout
   - Modular component design
   - Clear error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)
