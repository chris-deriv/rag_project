"""Vector database management with ChromaDB."""
import chromadb
from chromadb.config import Settings
from rag_backend.config.settings import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from typing import List, Dict, Any, Optional, Union
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

    def _get_existing_doc_ids(self, source_name: str) -> List[str]:
        """Get existing document IDs for a given source name."""
        try:
            # Get all documents with matching source name
            logger.info(f"\nLooking for existing documents with source_name: {source_name}")
            
            # Debug: List all documents first
            all_docs = self.collection.get(include=['metadatas'])
            logger.info("All document source names in collection:")
            for metadata in all_docs['metadatas']:
                logger.info(f"- {metadata.get('source_name')}")
            
            # Get documents matching source name
            result = self.collection.get(
                where={"source_name": {"$eq": source_name}}
            )
            ids = result.get("ids", [])
            logger.info(f"Found {len(ids)} existing documents for {source_name}")
            if ids:
                logger.info(f"Document IDs to delete: {ids}")
            return ids
        except Exception as e:
            logger.error(f"Error getting existing document IDs: {str(e)}")
            return []

    def _validate_chunk_consistency(self, documents: List[Dict[str, Any]]) -> None:
        """Validate that all chunks for a document have consistent total_chunks."""
        docs_by_source = {}
        for doc in documents:
            source_name = doc.get('source_name', 'Unknown')
            if source_name not in docs_by_source:
                docs_by_source[source_name] = []
            docs_by_source[source_name].append(doc)
        
        for source_name, source_docs in docs_by_source.items():
            total_chunks = source_docs[0].get('total_chunks')
            if not all(doc.get('total_chunks') == total_chunks for doc in source_docs):
                raise ValueError(f"Inconsistent total_chunks values for document {source_name}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector database."""
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
            
            # Validate chunk consistency before proceeding
            self._validate_chunk_consistency(documents)
            
            # Group documents by source name
            docs_by_source = {}
            for doc in documents:
                source_name = doc.get('source_name', 'Unknown')
                if source_name not in docs_by_source:
                    docs_by_source[source_name] = []
                docs_by_source[source_name].append(doc)
            
            # Process each source's documents
            for source_name, source_docs in docs_by_source.items():
                # Get existing document IDs for this source
                existing_ids = self._get_existing_doc_ids(source_name)
                
                if existing_ids:
                    logger.info(f"Found existing documents for {source_name}, updating...")
                    # Delete existing documents
                    logger.info(f"Attempting to delete documents with IDs: {existing_ids}")
                    try:
                        self.collection.delete(ids=existing_ids)
                        # Verify deletion
                        remaining = self.collection.get(
                            where={"source_name": {"$eq": source_name}}
                        )
                        if remaining.get("ids"):
                            logger.warning(f"Deletion may have failed. Found {len(remaining['ids'])} remaining documents")
                            logger.warning(f"Remaining IDs: {remaining['ids']}")
                        else:
                            logger.info("Deletion verified - no remaining documents found")
                    except Exception as e:
                        logger.error(f"Error during deletion: {str(e)}")
                        raise
                    logger.info(f"Deleted {len(existing_ids)} existing documents")
                
                # Add new documents
                self.collection.add(
                    embeddings=[doc['embedding'].tolist() for doc in source_docs],
                    documents=[doc['text'] for doc in source_docs],
                    metadatas=[{
                        "source_name": doc.get("source_name", "Unknown"),
                        "title": doc.get("title", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "total_chunks": doc.get("total_chunks", 1),
                        "section_title": doc.get("section_title", ""),
                        "section_type": doc.get("section_type", "content"),
                        "file_type": doc.get("file_type", ""),
                        "text": doc["text"]  # Include text in metadata for easier retrieval
                    } for doc in source_docs],
                    ids=[str(doc["id"]) for doc in source_docs]
                )
                logger.info(f"Added {len(source_docs)} documents for {source_name}")
            
            logger.info(f"Successfully processed all documents")
            
            # Debug: List all documents after adding
            all_docs = self.collection.get(include=['metadatas'])
            logger.info("\nAll documents in collection after adding:")
            for metadata in all_docs['metadatas']:
                logger.info(f"Source Name: {metadata.get('source_name')}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {str(e)}")
            raise

    def query(self, 
             query_embedding: np.ndarray, 
             n_results: int = 5,
             source_names: Optional[List[str]] = None,
             title: Optional[str] = None) -> Dict[str, Any]:
        """Query the vector database for similar documents with optional filtering."""
        try:
            # Build where clause for filtering
            where = None
            
            if source_names and title:
                # Use $and operator to combine source_names and title filters
                where = {
                    "$and": [
                        {"source_name": {"$in": source_names}},
                        {"title": {"$eq": title.lower()}}
                    ]
                }
            elif source_names:
                where = {"source_name": {"$in": source_names}}
            elif title:
                where = {"title": {"$eq": title.lower()}}
            
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
        """Search for documents with similar titles."""
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
                        'source_name': source_name,
                        'file_type': metadata.get('file_type', ''),
                        'section_type': metadata.get('section_type', 'content')
                    })
            
            return matches
        except Exception as e:
            logger.error(f"Error searching titles: {str(e)}")
            raise

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the collection."""
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
        """Get all documents and their metadata from the collection."""
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
        """Get a list of unique document names/titles with their chunk counts."""
        try:
            # Check if collection is empty
            count = self.collection.count()
            if count == 0:
                logger.info("ChromaDB collection is empty")
                return []
                
            # Print debug information
            logger.info("\nListing documents from ChromaDB:")
            
            # Get all documents with their metadata
            all_docs = self.collection.get(
                include=['metadatas', 'documents'],
                limit=count  # Ensure we get all documents
            )
            logger.info(f"Total documents found: {len(all_docs['metadatas'])}")
            
            # Debug log the first few documents
            for i in range(min(3, len(all_docs['metadatas']))):
                logger.info(f"\nDocument {i+1} metadata:")
                logger.info(f"Source Name: {all_docs['metadatas'][i].get('source_name', 'Unknown')}")
                logger.info(f"Title: {all_docs['metadatas'][i].get('title', '')}")
                logger.info(f"Chunk Index: {all_docs['metadatas'][i].get('chunk_index', 0)}")
                logger.info(f"Total Chunks: {all_docs['metadatas'][i].get('total_chunks', 1)}")
            
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
            
            # Debug log the results
            logger.info("\nDocument statistics:")
            for doc in doc_stats.values():
                logger.info(f"\nSource: {doc['source_name']}")
                logger.info(f"Title: {doc['title']}")
                logger.info(f"Chunks: {doc['chunk_count']}/{doc['total_chunks']}")
            
            return list(doc_stats.values())
        except Exception as e:
            logger.error(f"Error listing document names: {str(e)}")
            raise

    def get_document_chunks(self, source_name: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document, ordered by chunk index."""
        try:
            logger.info(f"\nQuerying for document chunks with source_name: {source_name}")
            
            # Debug: List all documents and their source names
            all_docs = self.collection.get(include=['metadatas'])
            logger.info("\nAll documents in collection:")
            source_names = set()
            for metadata in all_docs['metadatas']:
                source_name_in_db = metadata.get('source_name')
                if source_name_in_db not in source_names:
                    source_names.add(source_name_in_db)
                    logger.info(f"Found source name in DB: {source_name_in_db}")
            
            # Get all chunks for the document
            logger.info(f"\nSearching for chunks with source_name: {source_name}")
            result = self.collection.get(
                where={"source_name": {"$eq": source_name}},
                include=['metadatas', 'documents']
            )
            
            logger.info(f"Found {len(result.get('ids', []))} chunks")
            
            if not result['ids']:
                logger.info(f"No chunks found for document: {source_name}")
                return []
            
            # Convert to list of dictionaries and sort by chunk index
            chunks = []
            for i in range(len(result['ids'])):
                chunk = {
                    'id': result['ids'][i],
                    'text': result['documents'][i],
                    **result['metadatas'][i]
                }
                chunks.append(chunk)
            
            # Sort chunks by index
            chunks.sort(key=lambda x: x.get('chunk_index', 0))
            return chunks
            
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
