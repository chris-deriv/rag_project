import chromadb
from chromadb.config import Settings
from config.settings import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from typing import List, Dict, Any, Optional
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        """Initialize the vector database with persistence."""
        try:
            logger.info(f"Initializing ChromaDB with persist_directory: {CHROMA_PERSIST_DIR}")
            
            # Ensure persist directory exists
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            
            settings = Settings(
                persist_directory=CHROMA_PERSIST_DIR,
                anonymized_telemetry=False,
                is_persistent=True,
                allow_reset=True
            )
            
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=settings)
            
            # Get or create collection with specific distance function
            self.collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info("ChromaDB initialized successfully")
            logger.info(f"Collection count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of dictionaries with 'id', 'text', 'embedding', and metadata keys.
                Each document should have:
                - id: Unique identifier for the document
                - text: The document text
                - embedding: The pre-computed embedding vector (numpy array)
                - source_name: Name/title of the source document
                - title: Document title
                - chunk_index: Index of this chunk in the document
                - total_chunks: Total number of chunks in the document
                - metadata: Additional metadata
                - section_title: Title of the document section
                - section_type: Type of section (e.g., 'body', 'heading')
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return
                
            # Print debug information
            logger.info("\nAdding documents to ChromaDB:")
            for doc in documents:
                logger.info(f"\nDocument ID: {doc['id']}")
                logger.info(f"Source Name: {doc.get('source_name', 'Unknown')}")
                logger.info(f"Title: {doc.get('title', '')}")
                logger.info(f"Chunk Index: {doc.get('chunk_index', 0)}")
                logger.info(f"Total Chunks: {doc.get('total_chunks', 1)}")
            
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
            logger.info(f"Successfully added {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {str(e)}")
            raise

    def query(self, 
             query_embedding: np.ndarray, 
             n_results: int = 5,
             source_name: Optional[str] = None,
             title: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the vector database for similar documents with optional filtering.
        
        Args:
            query_embedding: The embedding vector of the query (numpy array)
            n_results: Number of results to return (default: 5)
            source_name: Optional filter by source filename (exact match)
            title: Optional filter by document title (partial match)
            
        Returns:
            Dict containing query results with ids, distances, and metadatas
        """
        try:
            # Build where clause for filtering
            where = {}
            where_document = {}
            
            if source_name:
                where["source_name"] = source_name
            
            if title:
                # Use $contains operator for partial title matching
                where["title"] = {"$contains": title.lower()}
            
            # Execute query with filters if any are specified
            if where:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results,
                    where=where,
                    include=['metadatas', 'distances', 'documents']
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results,
                    include=['metadatas', 'distances', 'documents']
                )
            return results
        except Exception as e:
            logger.error(f"Error querying vector database: {str(e)}")
            raise

    def search_titles(self, title_query: str) -> List[Dict[str, Any]]:
        """
        Search for documents with similar titles.
        
        Args:
            title_query: Partial title to search for
            
        Returns:
            List of documents with matching titles
        """
        try:
            # Return empty list for empty queries
            if not title_query.strip():
                return []
                
            # Get all documents
            all_docs = self.collection.get(include=['metadatas'])
            
            # Create a set to track unique titles we've found
            found_titles = set()
            matches = []
            
            # Convert query to lowercase for case-insensitive matching
            title_query = title_query.lower()
            
            # Check each document's title
            for metadata in all_docs['metadatas']:
                title = metadata.get('title', '').lower()
                source_name = metadata.get('source_name', '')
                
                # Skip if we've already found this title
                if (title, source_name) in found_titles:
                    continue
                
                # Check if query is contained in the title
                if title_query in title:
                    found_titles.add((title, source_name))
                    matches.append({
                        'title': metadata.get('title', ''),
                        'source_name': source_name
                    })
            
            return matches
        except Exception as e:
            logger.error(f"Error searching titles: {str(e)}")
            raise

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
            logger.error(f"Error getting collection metadata: {str(e)}")
            raise

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents and their metadata from the collection.
        
        Returns:
            List of documents with their ids, embeddings, and metadata
        """
        try:
            # Check if collection is empty
            count = self.collection.count()
            if count == 0:
                logger.info("ChromaDB collection is empty")
                return []
                
            logger.info(f"Getting {count} documents from ChromaDB")
            result = self.collection.get(include=['embeddings', 'metadatas', 'documents'])
            
            # Convert ChromaDB result into a list of documents
            documents = []
            for i in range(len(result['ids'])):
                doc = {
                    'id': result['ids'][i],
                    'text': result['documents'][i],
                    'embedding': result['embeddings'][i],
                    **result['metadatas'][i]  # Include all metadata fields
                }
                documents.append(doc)
            
            logger.info(f"Successfully retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            raise

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
            # Check if collection is empty
            count = self.collection.count()
            if count == 0:
                logger.info("ChromaDB collection is empty")
                return []
                
            # Print debug information
            logger.info("\nListing documents from ChromaDB:")
            all_docs = self.collection.get(include=['metadatas'])
            logger.info(f"Total documents found: {len(all_docs['metadatas'])}")
            
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
            logger.error(f"Error listing document names: {str(e)}")
            raise

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
            # Check if collection is empty
            if self.collection.count() == 0:
                logger.info("ChromaDB collection is empty")
                return []
                
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
            logger.error(f"Error getting document chunks: {str(e)}")
            raise

    def delete_collection(self) -> None:
        """Delete the current collection from the database."""
        try:
            logger.info(f"Deleting collection: {CHROMA_COLLECTION_NAME}")
            # Delete the collection
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
            
            # Reset the client to ensure clean state
            settings = Settings(
                persist_directory=CHROMA_PERSIST_DIR,
                anonymized_telemetry=False,
                is_persistent=True,
                allow_reset=True
            )
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=settings)
            
            # Create a new collection
            self.collection = self.client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection deleted and recreated successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
