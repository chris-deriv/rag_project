from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from src.app import RAGApplication
from src.documents import add_document, get_documents
from src.database import VectorDatabase

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize RAG application and vector database
rag_app = RAGApplication()
vector_db = VectorDatabase()

# Index existing documents on startup
try:
    existing_documents = get_documents()
    if existing_documents:
        rag_app.index_documents(existing_documents)
except Exception as e:
    app.logger.error(f"Error indexing existing documents on startup: {str(e)}")

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
        
        try:
            add_document(filepath)
            # Index the new documents in the RAG system
            rag_app.index_documents(get_documents())
            # Clean up the uploaded file after processing
            os.remove(filepath)
            return jsonify({'message': 'File processed successfully'}), 200
        except Exception as e:
            # Clean up the file if processing fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        response = rag_app.query_documents(data['query'])
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def get_all_documents():
    return jsonify(get_documents())

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

if __name__ == '__main__':
    app.run(debug=True)
