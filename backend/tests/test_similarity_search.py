from src.search import SearchEngine
from src.database import VectorDatabase
from src.documents import DocumentProcessor
import numpy as np

def test_similarity_search():
    # Initialize components
    search_engine = SearchEngine()
    vector_db = VectorDatabase()
    doc_processor = DocumentProcessor()
    
    # Create test documents with known content
    test_docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning is a type of machine learning.",
        "Neural networks are used in deep learning.",
        "Artificial intelligence includes machine learning and other approaches."
    ]
    
    print("Processing test documents...")
    # Process and store test documents
    for i, text in enumerate(test_docs):
        doc_id = f"test_doc_{i}"
        # Generate embeddings
        embedding = search_engine.embedding_generator.generate_embeddings([text])[0]
        # Store in vector database
        vector_db.add_documents([{
            'id': doc_id,
            'text': text,
            'embedding': embedding,
            'source_name': f'test_doc_{i}.txt',
            'title': f'Test Document {i}',
            'file_type': 'txt',
            'section_type': 'content'
        }])
        print(f"Added document {doc_id}: {text}")
    
    print("\nPerforming search test...")
    # Test query that should match multiple documents
    query = "What is machine learning?"
    print(f"\nQuery: {query}")
    
    # Get search results
    results = search_engine.search(query, n_results=4)
    
    print("\nSearch Results:")
    print("-" * 80)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"ID: {result['id']}")
        print(f"Text: {result['text']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Relevance Score: {result['relevance_score']:.4f}")
        print(f"Combined Score: {result['combined_score']:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    test_similarity_search()
