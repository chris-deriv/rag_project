from typing import List, Dict, Any
import logging
from .embedding import EmbeddingGenerator
from .database import VectorDatabase
from .chatbot import Chatbot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGApplication:
    def __init__(self):
        """Initialize the RAG application components."""
        logger.info("Initializing RAG application...")
        self.embedder = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        self.chatbot = Chatbot()

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index a list of documents into the vector database.
        
        Args:
            documents: List of dictionaries, each containing 'id' and 'text' keys
        """
        try:
            logger.info(f"Indexing {len(documents)} documents...")
            
            # Generate embeddings for all documents
            for doc in documents:
                doc['embedding'] = self.embedder.generate_embeddings([doc['text']])[0]
            
            # Add documents to vector database
            self.vector_db.add_documents(documents)
            logger.info("Documents indexed successfully")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    def query_documents(self, query: str, n_results: int = 5) -> str:
        """
        Process a query and return a response based on relevant documents.
        
        Args:
            query: The user's question
            n_results: Number of relevant documents to retrieve
            
        Returns:
            str: Generated response based on relevant documents
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Generate embedding for the query
            query_embedding = self.embedder.generate_embeddings([query])[0]
            
            # Retrieve relevant documents
            results = self.vector_db.query(query_embedding, n_results=n_results)
            
            # Extract contexts from results
            contexts = [
                {"text": metadata["text"]} 
                for metadata in results["metadatas"][0]
            ]
            
            # Generate response with source citations
            response = self.chatbot.generate_response_with_sources(contexts, query)
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

def main():
    """Example usage of the RAG application."""
    try:
        # Initialize the application
        app = RAGApplication()
        
        # Sample documents
        documents = [
            {
                "id": 1,
                "text": "Python is a high-level programming language known for its simplicity and readability."
            },
            {
                "id": 2,
                "text": "Virtual environments in Python are isolated spaces where you can install specific package versions."
            },
            {
                "id": 3,
                "text": "The pip package manager is used to install Python packages from PyPI (Python Package Index)."
            }
        ]
        
        # Index the documents
        app.index_documents(documents)
        
        # Example query
        query = "How do I manage Python packages?"
        response = app.query_documents(query)
        
        print("\nQuery:", query)
        print("\nResponse:", response)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
