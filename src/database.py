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
            documents: List of dictionaries with 'id', 'text', and 'embedding' keys.
                Each document should have:
                - id: Unique identifier for the document
                - text: The document text
                - embedding: The pre-computed embedding vector
        """
        try:
            self.collection.add(
                embeddings=[doc['embedding'].tolist() for doc in documents],
                metadatas=[{"text": doc["text"]} for doc in documents],
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
                n_results=n_results
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
            return self.collection.get()
        except Exception as e:
            raise Exception(f"Error getting all documents: {str(e)}")

    def delete_collection(self) -> None:
        """Delete the current collection from the database."""
        try:
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        except Exception as e:
            raise Exception(f"Error deleting collection: {str(e)}")
