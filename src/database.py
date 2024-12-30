import chromadb
from chromadb.config import Settings
from config.settings import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from typing import List, Dict, Any

class VectorDatabase:
    def __init__(self):
        """Initialize the vector database with persistence."""
        self.client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of dictionaries with 'id', 'text', 'embedding', and metadata keys.
                Each document should have:
                - id: Unique identifier for the document
                - text: The document text
                - embedding: The pre-computed embedding vector
                - source_name: Name/title of the source document
                - title: Document title
                - chunk_index: Index of this chunk in the document
                - total_chunks: Total number of chunks in the document
                - metadata: Additional metadata
                - section_title: Title of the document section
                - section_type: Type of section (e.g., 'body', 'heading')
        """
        try:
            # Print debug information
            print("\nAdding documents to ChromaDB:")
            for doc in documents:
                print(f"\nDocument ID: {doc['id']}")
                print(f"Source Name: {doc.get('source_name', 'Unknown')}")
                print(f"Title: {doc.get('title', '')}")
                print(f"Chunk Index: {doc.get('chunk_index', 0)}")
                print(f"Total Chunks: {doc.get('total_chunks', 1)}")
                print(f"Section Title: {doc.get('section_title', '')}")
                print(f"Section Type: {doc.get('section_type', 'content')}")
            
            self.collection.add(
                embeddings=[doc['embedding'].tolist() for doc in documents],
                documents=[doc['text'] for doc in documents],
                metadatas=[{
                    "text": doc["text"],
                    "source_name": doc.get("source_name", "Unknown"),
                    "title": doc.get("title", ""),
                    "chunk_index": doc.get("chunk_index", 0),
                    "total_chunks": doc.get("total_chunks", 1),
                    "section_title": doc.get("section_title", ""),
                    "section_type": doc.get("section_type", "content")
                } for doc in documents],
                ids=[str(doc["id"]) for doc in documents]
            )
        except Exception as e:
            raise Exception(f"Error adding documents to vector database: {str(e)}")

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector database for similar documents.
        
        Args:
            query_embedding: The embedding vector of the query
            n_results: Number of results to return (default: 5)
            
        Returns:
            Dict containing query results with ids, distances, and metadatas
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['metadatas', 'distances', 'documents']
            )
            return results
        except Exception as e:
            raise Exception(f"Error querying vector database: {str(e)}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the collection.
        
        Returns:
            Dict containing collection metadata including:
            - name: Collection name
            - count: Number of documents in collection
            - metadata: Collection metadata
        """
        try:
            count = self.collection.count()
            metadata = {
                "name": self.collection.name,
                "count": count,
                "metadata": self.collection.metadata
            }
            return metadata
        except Exception as e:
            raise Exception(f"Error getting collection metadata: {str(e)}")

    def get_all_documents(self) -> Dict[str, Any]:
        """
        Get all documents and their metadata from the collection.
        
        Returns:
            Dict containing all documents with their ids, embeddings, and metadata
        """
        try:
            return self.collection.get(include=['embeddings', 'metadatas', 'documents'])
        except Exception as e:
            raise Exception(f"Error getting all documents: {str(e)}")

    def list_document_names(self) -> List[Dict[str, Any]]:
        """
        Get a list of unique document names/titles with their chunk counts.
        
        Returns:
            List of dictionaries containing:
            - source_name: Name/title of the document
            - title: Document title
            - chunk_count: Number of chunks for this document
            - total_chunks: Total expected chunks
        """
        try:
            # Print debug information
            print("\nListing documents from ChromaDB:")
            all_docs = self.collection.get(include=['metadatas'])
            print(f"Total documents found: {len(all_docs['metadatas'])}")
            for metadata in all_docs['metadatas']:
                print(f"\nMetadata: {metadata}")
            
            doc_stats = {}
            
            # Group chunks by source document
            for metadata in all_docs['metadatas']:
                source_name = metadata.get('source_name', 'Unknown')
                if source_name not in doc_stats:
                    doc_stats[source_name] = {
                        'source_name': source_name,
                        'title': metadata.get('title', ''),
                        'chunk_count': 1,
                        'total_chunks': metadata.get('total_chunks', 1)
                    }
                else:
                    doc_stats[source_name]['chunk_count'] += 1
            
            return list(doc_stats.values())
        except Exception as e:
            raise Exception(f"Error listing document names: {str(e)}")

    def get_document_chunks(self, source_name: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document, ordered by chunk index.
        
        Args:
            source_name: Name/title of the document to retrieve chunks for
            
        Returns:
            List of chunks with their metadata, ordered by chunk_index:
            - id: Chunk identifier
            - text: Chunk content
            - title: Document title
            - chunk_index: Position in document
            - total_chunks: Total chunks in document
            - section_title: Title of the document section
            - section_type: Type of section (e.g., 'body', 'heading')
        """
        try:
            # Get all chunks for the document
            all_docs = self.collection.get(include=['metadatas', 'documents'])
            doc_chunks = []
            
            # Filter and collect chunks for the specified document
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('source_name') == source_name:
                    doc_chunks.append({
                        'id': all_docs['ids'][i],
                        'text': metadata['text'],
                        'title': metadata.get('title', ''),
                        'chunk_index': metadata.get('chunk_index', 0),
                        'total_chunks': metadata.get('total_chunks', 1),
                        'section_title': metadata.get('section_title', ''),
                        'section_type': metadata.get('section_type', 'content')
                    })
            
            # Sort chunks by index
            doc_chunks.sort(key=lambda x: x['chunk_index'])
            return doc_chunks
        except Exception as e:
            raise Exception(f"Error getting document chunks: {str(e)}")

    def delete_collection(self) -> None:
        """Delete the current collection from the database."""
        try:
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        except Exception as e:
            raise Exception(f"Error deleting collection: {str(e)}")
