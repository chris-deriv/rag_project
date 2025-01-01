from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
from src.app import RAGApplication
from src.documents import add_document, get_documents, process_document, document_store
from src.database import VectorDatabase
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Configure JSON formatting
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.json.sort_keys = False  # Preserve key order in JSON responses

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize RAG application and use the same vector database instance as DocumentStore
rag_app = RAGApplication()
vector_db = document_store.db  # Use the same instance from DocumentStore

# Track processing documents
processing_documents = {}

def process_document_async(filepath, filename):
    """Process document asynchronously and update processing status."""
    try:
        logger.info(f"Starting async processing of {filename}")
        processing_documents[filename] = {'status': 'processing', 'error': None}
        
        # Process the document to get chunks
        chunks = process_document(filepath)
        if not chunks:
            raise Exception("No chunks generated from document")
            
        # Get the actual source_name from the first chunk
        actual_source_name = chunks[0].get('source_name', filename)
        logger.info(f"Generated {len(chunks)} chunks from {filename}")
        logger.info(f"Document source_name from chunks: {actual_source_name}")
        
        # Generate embeddings and add to vector database
        texts = [chunk['text'] for chunk in chunks]
        embeddings = document_store.embedding_generator.generate_embeddings(texts)
        
        # Convert chunks to the format expected by VectorDatabase
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "id": chunk['id'],
                "text": chunk['text'],
                "embedding": embeddings[i],
                **{k: v for k, v in chunk.items() if k not in ['id', 'text']}
            }
            documents.append(doc)
        
        # Add documents with embeddings to ChromaDB
        logger.info("Adding documents to ChromaDB...")
        vector_db.add_documents(documents)
        logger.info("Documents added successfully")
        
        # Get the document directly using the known source name
        doc_chunks = vector_db.get_document_chunks(actual_source_name)
        if not doc_chunks:
            raise Exception(f"Could not verify document {actual_source_name} in vector database after adding")
        
        # Create document info for indexing
        doc_to_index = {
            'source_name': actual_source_name,
            'title': doc_chunks[0].get('title', ''),
            'chunk_count': len(doc_chunks),
            'total_chunks': doc_chunks[0].get('total_chunks', len(doc_chunks))
        }
        
        # Index the document
        rag_app.index_documents([doc_to_index])
        logger.info(f"Successfully indexed {doc_to_index['source_name']}")
        
        # Update processing status with success
        processing_documents[filename] = {
            'status': 'completed',
            'error': None,
            'actual_filename': actual_source_name
        }
        
        logger.info(f"Completed processing {filename}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing {filename}: {error_msg}")
        processing_documents[filename] = {'status': 'error', 'error': error_msg}
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

# Index existing documents on startup
try:
    logger.info("Checking for existing documents to index...")
    existing_documents = get_documents()
    if existing_documents:
        if not isinstance(existing_documents, list):
            logger.error(f"Invalid documents format: {type(existing_documents)}")
        else:
            logger.info(f"Found {len(existing_documents)} documents to index")
            rag_app.index_documents(existing_documents)
    else:
        logger.info("No existing documents found to index")
except Exception as e:
    logger.error(f"Error indexing existing documents on startup: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Start async processing
        thread = threading.Thread(
            target=process_document_async,
            args=(filepath, filename)
        )
        thread.start()
        
        return jsonify({
            'message': 'File upload started',
            'filename': filename
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/upload-status/<filename>', methods=['GET'])
def get_upload_status(filename):
    """Get the processing status of an uploaded document."""
    if filename not in processing_documents:
        return jsonify({'status': 'unknown'}), 404
    
    return jsonify(processing_documents[filename])

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Extract optional document filters
        source_names = data.get('source_names', [])  # Now expecting an array
        title = data.get('title')
        
        # Log the filters being applied
        logger.info(f"Query filters - source_names: {source_names}, title: {title}")
        
        response = rag_app.query_documents(
            data['query'],
            source_names=source_names,  # Pass the array directly
            title=title
        )
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def get_all_documents():
    try:
        docs = get_documents()
        if not isinstance(docs, list):
            logger.error(f"Invalid documents format: {type(docs)}")
            return jsonify([])
        return jsonify(docs)
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        return jsonify([])

@app.route('/metadata', methods=['GET'])
def get_collection_metadata():
    """Get metadata about the ChromaDB collection."""
    try:
        metadata = vector_db.get_metadata()
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/vector-documents', methods=['GET'])
def get_vector_documents():
    """Get all documents with their embeddings and metadata from ChromaDB."""
    try:
        documents = vector_db.get_all_documents()
        return jsonify(documents)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/document-names', methods=['GET'])
def list_document_names():
    """
    Get a list of all unique document names/titles with their chunk counts.
    Also includes processing status for documents being uploaded.
    
    Returns:
        JSON array of documents with their names and chunk statistics:
        [
            {
                "source_name": "example.pdf",
                "title": "Example Document",
                "chunk_count": 10,
                "total_chunks": 10,
                "status": "completed"
            },
            ...
        ]
    """
    try:
        # Get documents from vector database
        documents = vector_db.list_document_names()
        
        # Add processing status to documents
        for doc in documents:
            filename = doc['source_name']
            # Check both original filename and .doc version
            doc_status = None
            if filename in processing_documents:
                doc_status = processing_documents[filename]
            elif filename.replace('.docx', '.doc') in processing_documents:
                doc_status = processing_documents[filename.replace('.docx', '.doc')]
            
            if doc_status:
                doc['status'] = doc_status['status']
                if doc_status.get('error'):
                    doc['error'] = doc_status['error']
            else:
                doc['status'] = 'completed'
        
        # Add any documents that are still processing but not yet in vector database
        for filename, status in processing_documents.items():
            if status['status'] == 'processing' and not any(
                d['source_name'] in [filename, filename.replace('.doc', '.docx')] 
                for d in documents
            ):
                documents.append({
                    'source_name': filename,
                    'title': filename,
                    'chunk_count': 0,
                    'total_chunks': 0,
                    'status': 'processing'
                })
        
        return jsonify(documents)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search-titles', methods=['GET'])
def search_titles():
    """
    Search for documents with similar titles.
    
    Query Parameters:
        q: The title search query
        
    Returns:
        JSON array of matching documents:
        [
            {
                "title": "Example Document",
                "source_name": "example.pdf"
            },
            ...
        ]
    """
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    
    try:
        matches = vector_db.search_titles(query)
        return jsonify(matches)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['DELETE'])
def reset_database():
    """Reset the vector database by deleting all documents."""
    try:
        vector_db.delete_collection()
        # Clear processing documents tracking
        processing_documents.clear()
        return jsonify({'message': 'Database reset successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/document-chunks/<source_name>', methods=['GET'])
def get_document_chunks(source_name):
    """
    Get all chunks for a specific document, ordered by chunk index.
    
    Args:
        source_name: Name/title of the document to retrieve chunks for
        
    Returns:
        JSON array of chunks with their text and metadata:
        [
            {
                "id": "chunk1",
                "text": "Content of chunk 1",
                "title": "Example Document",
                "chunk_index": 0,
                "total_chunks": 10,
                "section_title": "Introduction",
                "section_type": "heading"
            },
            ...
        ]
    """
    try:
        chunks = vector_db.get_document_chunks(source_name)
        return jsonify(chunks)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
