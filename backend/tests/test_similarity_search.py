"""Test similarity search functionality."""
from src.search import SearchEngine
from src.database import VectorDatabase
from src.documents import DocumentProcessor
import json

def test_similarity_search():
    """Test similarity search with known content."""
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
    for i, text in enumerate(test_docs):  # Fixed: Added comma to properly unpack tuple
        doc_id = f"test_doc_{i}"
        # Generate embeddings
        embedding = search_engine.embedding_generator.generate_embeddings([text])[0]
        # Store in vector database
        vector_db.add_documents([{
            'id': doc_id,
            'text': text,
            'embedding': embedding,
            'source_name': f'test_doc_{i}.txt',
            'title': f'test document {i}',  # Lowercase title to match query
            'file_type': 'txt',
            'section_type': 'content',
            'chunk_index': 0,
            'total_chunks': 1,
            'section_title': '',  # Empty string instead of None
            'toc': json.dumps([])  # Convert list to JSON string
        }])

    # Test similarity search
    query = "What is machine learning?"
    query_embedding = search_engine.generate_query_embedding(query)
    results = vector_db.query(query_embedding, n_results=2)

    # Verify results contain relevant documents
    assert len(results['ids'][0]) == 2, "Expected 2 results"
    assert any("machine learning" in doc.lower() for doc in results['documents'][0]), \
        "Expected results to contain machine learning content"

    # Test source filtering
    filtered_results = vector_db.query(
        query_embedding,
        n_results=2,
        source_names=['test_doc_0.txt']
    )
    assert len(filtered_results['ids'][0]) > 0, "Expected filtered results"
    assert all(meta['source_name'] == 'test_doc_0.txt' 
              for meta in filtered_results['metadatas'][0]), \
        "Expected only test_doc_0.txt in filtered results"

    # Test title filtering
    title_results = vector_db.query(
        query_embedding,
        n_results=2,
        title='test document 0'  # Lowercase title to match stored title
    )
    assert len(title_results['ids'][0]) > 0, "Expected title filtered results"
    assert all(meta['title'] == 'test document 0' 
              for meta in title_results['metadatas'][0]), \
        "Expected only Test Document 0 in title filtered results"

    # Test combined filtering
    combined_results = vector_db.query(
        query_embedding,
        n_results=2,
        source_names=['test_doc_0.txt'],
        title='test document 0'  # Lowercase title to match stored title
    )
    assert len(combined_results['ids'][0]) > 0, "Expected combined filtered results"
    assert all(meta['source_name'] == 'test_doc_0.txt' and meta['title'] == 'test document 0'
              for meta in combined_results['metadatas'][0]), \
        "Expected only test_doc_0.txt with Test Document 0 in combined filtered results"
