# RAG-based Document Query System

A production-ready implementation of a Retrieval-Augmented Generation (RAG) system for document indexing and querying. This system combines the power of vector databases for semantic search with large language models for generating contextually relevant responses.

## Features

- Document embedding generation using Sentence Transformers
- Vector storage and retrieval using ChromaDB
- Natural language responses using OpenAI's GPT models
- Source citation in responses
- Robust error handling and logging
- Type hints for better code maintainability

## Project Structure

```
rag_project/
│
├── config/
│   └── settings.py      # Configuration settings
│
├── src/
│   ├── embedding.py     # Embedding generation
│   ├── database.py      # Vector database operations
│   ├── chatbot.py       # OpenAI integration
│   └── app.py          # Main application logic
│
├── requirements.txt     # Project dependencies
└── README.md           # Documentation
```

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_project
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

## Usage

The system can be used either as a library or through the command-line interface:

### As a Library

```python
from src.app import RAGApplication

# Initialize the application
app = RAGApplication()

# Index documents
documents = [
    {
        "id": 1,
        "text": "Your document text here"
    }
]
app.index_documents(documents)

# Query the system
response = app.query_documents("Your question here")
print(response)
```

### Command Line

Run the example script:
```bash
python -m src.app
```

## Components

### EmbeddingGenerator

Generates document embeddings using the Sentence Transformers library. Uses the "all-MiniLM-L6-v2" model by default.

### VectorDatabase

Manages document storage and retrieval using ChromaDB. Supports:
- Document addition with embeddings
- Similarity search
- Collection management

### Chatbot

Handles interaction with OpenAI's API to generate responses. Features:
- Context-aware responses
- Source citation
- Error handling

### RAGApplication

Main application class that coordinates all components. Provides:
- Document indexing
- Query processing
- Logging

## Error Handling

The system includes comprehensive error handling:
- Input validation
- API error handling
- Database operation error handling
- Logging of all operations and errors

## Production Considerations

1. **Environment Variables**: Sensitive data is managed through environment variables
2. **Logging**: Comprehensive logging for monitoring and debugging
3. **Type Hints**: Improved code maintainability and IDE support
4. **Error Handling**: Robust error handling throughout the application
5. **Documentation**: Detailed docstrings and README

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)
