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
        This method now handles only metadata indexing, as document processing
        is managed by the DocumentStore.
        
        Args:
            documents: List of dictionaries containing document metadata
        """
        try:
            if not documents:
                logger.info("No documents to index")
                return
                
            logger.info(f"Indexing {len(documents)} documents...")
            
            # Verify each document exists in the vector database
            for doc in documents:
                if not isinstance(doc, dict):
                    logger.warning(f"Skipping invalid document metadata: {doc}")
                    continue
                
                source_name = doc.get('source_name')
                if not source_name:
                    logger.warning("Skipping document without source_name")
                    continue
                
                # Verify document exists in vector database
                doc_info = document_store.get_document_info(source_name)
                if not doc_info:
                    logger.warning(f"Document {source_name} not found in vector database")
                    continue
                
                logger.info(f"Verified document {source_name} in vector database")
                logger.info(f"Chunks: {doc_info['chunk_count']}/{doc_info['total_chunks']}")
            
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

    def _balance_results(self, results: List[Dict[str, Any]], source_names: List[str]) -> List[Dict[str, Any]]:
        """
        Balance results across multiple documents to ensure representation from each source.
        
        Args:
            results: List of search results
            source_names: List of source names to balance across
            
        Returns:
            List of balanced results
        """
        if not source_names or len(source_names) <= 1:
            return results
            
        # Group results by source
        results_by_source = {}
        for result in results:
            source = result['metadata'].get('source_name', 'Unknown')
            if source not in results_by_source:
                results_by_source[source] = []
            results_by_source[source].append(result)
        
        # Calculate minimum results per source
        min_per_source = max(2, len(results) // len(source_names))
        
        # Build balanced results list
        balanced_results = []
        remaining_results = []
        
        # First, take minimum number from each source
        for source in source_names:
            source_results = results_by_source.get(source, [])
            balanced_results.extend(source_results[:min_per_source])
            remaining_results.extend(source_results[min_per_source:])
        
        # Add remaining results sorted by score
        remaining_results.sort(key=lambda x: x['combined_score'], reverse=True)
        balanced_results.extend(remaining_results)
        
        return balanced_results

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
                # Increase n_results when querying multiple documents
                n_results = max(n_results, len(source_names) * 3)
            if title:
                logger.info(f"Filtering by title: {title}")
            
            # Use SearchEngine to get relevant documents
            results = self.search_engine.search(
                query=query,
                n_results=n_results,
                source_names=source_names,
                title=title
            )
            
            # Balance results across multiple documents if needed
            if source_names and len(source_names) > 1:
                results = self._balance_results(results, source_names)
            
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
