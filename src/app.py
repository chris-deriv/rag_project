from typing import List, Dict, Any, Optional
import logging
from .embedding import EmbeddingGenerator
from .database import VectorDatabase
from .chatbot import Chatbot
from .documents import get_documents, document_store
from .search import SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class RAGApplication:
    def __init__(self):
        """Initialize the RAG application components."""
        logger.info("Initializing RAG application...")
        self.embedder = EmbeddingGenerator()
        self.vector_db = document_store.db  # Use the same instance from DocumentStore
        self.chatbot = Chatbot()
        self.search_engine = SearchEngine()

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index a list of documents into the vector database.
        
        Args:
            documents: List of dictionaries containing document information
        """
        try:
            if not documents:
                logger.info("No documents to index")
                return
                
            logger.info(f"Indexing {len(documents)} documents...")
            
            # Generate embeddings for all documents
            processed_docs = []
            for doc in documents:
                # Skip if document is empty or invalid
                if not isinstance(doc, dict):
                    logger.warning(f"Skipping invalid document: {doc}")
                    continue
                    
                if 'text' not in doc:
                    logger.warning(f"Skipping document without text: {doc}")
                    continue
                
                # Extract metadata
                metadata = {k: v for k, v in doc.items() if k not in ['id', 'text', 'embedding']}
                source_name = metadata.get('source_name', metadata.get('source_file', 'Unknown'))
                title = metadata.get('title', '')
                chunk_index = metadata.get('chunk_index', 0)
                total_chunks = metadata.get('total_chunks', 1)
                
                logger.info(f"Processing document: {source_name}")
                logger.info(f"Title: {title}")
                logger.info(f"Chunk: {chunk_index + 1}/{total_chunks}")
                
                # Create processed document with metadata
                processed_doc = {
                    'id': doc.get('id', len(processed_docs) + 1),
                    'text': doc['text'],
                    'embedding': self.embedder.generate_embeddings([doc['text']])[0],
                    'source_name': source_name,
                    'title': title,
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks
                }
                processed_docs.append(processed_doc)
            
            # Sort processed documents by ID for deterministic ordering
            processed_docs.sort(key=lambda x: x['id'])
            
            # Add documents to vector database
            if processed_docs:
                logger.info(f"Adding {len(processed_docs)} documents to vector database")
                self.vector_db.add_documents(processed_docs)
                logger.info("Documents indexed successfully")
            else:
                logger.warning("No valid documents to index")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    def _sort_contexts(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort contexts deterministically by source, title, and chunk index.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            List of sorted context dictionaries
        """
        return sorted(
            contexts,
            key=lambda x: (
                x.get('source', ''),
                x.get('title', ''),
                x.get('chunk_index', 0)
            )
        )

    def query_documents(self, 
                       query: str, 
                       n_results: int = 5,
                       source_names: Optional[List[str]] = None,
                       title: Optional[str] = None) -> str:
        """
        Process a query and return a response based on relevant documents.
        
        Args:
            query: The user's question
            n_results: Number of relevant documents to retrieve
            source_names: Optional list of source filenames to filter by
            title: Optional filter by document title
            
        Returns:
            str: Generated response based on relevant documents
        """
        try:
            logger.info(f"Processing query: {query}")
            if source_names:
                logger.info(f"Filtering by source names: {source_names}")
            if title:
                logger.info(f"Filtering by title: {title}")
            
            # Use SearchEngine to get relevant documents
            results = self.search_engine.search(
                query=query,
                n_results=n_results,
                source_names=source_names,
                title=title
            )
            
            # Extract contexts from results with source information
            contexts = []
            for result in results:
                context = {
                    "text": result['text'],
                    "source": result['metadata'].get('source_name', 'Unknown'),
                    "title": result['metadata'].get('title', ''),
                    "chunk_index": result['metadata'].get('chunk_index', 0),
                    "total_chunks": result['metadata'].get('total_chunks', 1)
                }
                contexts.append(context)
            
            # Sort contexts deterministically
            sorted_contexts = self._sort_contexts(contexts)
            
            # Generate response with source citations
            response = self.chatbot.generate_response_with_sources(sorted_contexts, query)
            
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
        
        # Index any existing documents
        documents = get_documents()
        if documents:
            logger.info(f"Found {len(documents)} documents to index")
            app.index_documents(documents)
        else:
            logger.warning("No documents found to index")
        
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
