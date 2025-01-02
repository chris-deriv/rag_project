from typing import List, Dict, Any, Optional
import numpy as np
import logging
from .embedding import EmbeddingGenerator
from .database import VectorDatabase
from .chatbot import Chatbot

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self):
        """Initialize the search engine with required components."""
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        self.chatbot = Chatbot()
        # Cache for LLM relevance scores to ensure consistency
        self._relevance_cache = {}

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse the natural language query to extract key information.
        
        Args:
            query (str): The user's natural language query
            
        Returns:
            Dict containing parsed query information including:
            - original_query: The original query text
            - processed_query: Any preprocessing applied to the query
            
        Raises:
            Exception: If query is None
        """
        if query is None:
            raise Exception("Query cannot be None")
            
        # Basic preprocessing while preserving special characters
        # and handling multiple spaces/newlines
        processed_query = query.strip().lower()
        return {
            "original_query": query,
            "processed_query": processed_query
        }

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for the query text.
        
        Args:
            query (str): The processed query text
            
        Returns:
            numpy.ndarray: The query embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            return self.embedding_generator.generate_embeddings([query])[0]
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")

    def perform_similarity_search(self, 
                                query_embedding: np.ndarray, 
                                n_results: int = 10,
                                source_names: Optional[List[str]] = None,
                                title: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform similarity search in the vector database.
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to retrieve (default: 10)
            source_names: Optional list of source filenames to filter by
            title: Optional filter by document title
            
        Returns:
            Dict containing search results with distances and metadata
            
        Raises:
            Exception: If database query fails or if n_results is invalid
        """
        if not isinstance(n_results, int) or n_results < 1:
            raise Exception("n_results must be a positive integer")
            
        try:
            results = self.vector_db.query(
                query_embedding=query_embedding, 
                n_results=n_results,
                source_names=source_names,
                title=title
            )
            
            # Ensure results have expected structure
            if not all(key in results for key in ['ids', 'distances', 'metadatas']):
                raise Exception("Invalid results structure from database")
                
            # Handle empty results case first
            ids = results['ids'][0]
            if isinstance(ids, np.ndarray):
                if ids.size == 0:  # Use size for numpy arrays
                    return {
                        'ids': [[]],
                        'distances': [[]],
                        'metadatas': [[]]
                    }
            elif len(ids) == 0:  # Use len for lists
                return {
                    'ids': [[]],
                    'distances': [[]],
                    'metadatas': [[]]
                }
                
            # Convert numpy arrays to lists if necessary
            ids = ids.tolist() if isinstance(ids, np.ndarray) else ids
            distances = results['distances'][0].tolist() if isinstance(results['distances'][0], np.ndarray) else results['distances'][0]
            metadatas = results['metadatas'][0]
                
            # Log initial search results
            logger.info("\nInitial similarity search results:")
            for d, i, m in zip(distances, ids, metadatas):
                logger.info(f"Distance: {d:.4f}, ID: {i}, Text: {m['text'][:50]}...")

            # Sort results by distance first, then by ID for consistent ordering
            # Create a list of tuples with all the data
            sorted_data = sorted(zip(distances, ids, metadatas), key=lambda x: (x[0], x[1]))
            
            # Unzip the sorted data back into separate lists
            sorted_distances, sorted_ids, sorted_metadatas = zip(*sorted_data)

            logger.info("\nSorted similarity search results:")
            for d, i, m in zip(sorted_distances, sorted_ids, sorted_metadatas):
                logger.info(f"Distance: {d:.4f}, ID: {i}, Text: {m['text'][:50]}...")
            
            # Return results in the expected format
            return {
                'ids': [list(sorted_ids)],
                'distances': [list(sorted_distances)],
                'metadatas': [list(sorted_metadatas)]
            }
        except Exception as e:
            raise Exception(f"Database query failed: {str(e)}")

    def _get_cache_key(self, query: str, text: str) -> str:
        """Generate a deterministic cache key for LLM relevance scores."""
        return f"{query.strip().lower()}|||{text.strip()}"

    def rerank_results(self, query: str, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rerank search results using LLM to ensure most relevant results appear first.
        
        Args:
            query: Original query string
            search_results: Initial search results from vector database
            
        Returns:
            List of reranked results with relevance scores
        """
        # Handle empty results
        if not search_results['metadatas'][0]:
            return []
            
        # Extract texts and metadata from search results
        texts = search_results['metadatas'][0]
        distances = search_results['distances'][0]
        ids = search_results['ids'][0]
        
        # Handle single result case
        if len(texts) == 1:
            return [{
                'id': ids[0],
                'text': texts[0]['text'],
                'metadata': texts[0],
                'similarity_score': 1 - distances[0],
                'relevance_score': 1.0,
                'combined_score': 1.0
            }]
        
        # Check cache first for all texts
        uncached_texts = []
        uncached_indices = []
        relevance_scores = [0.0] * len(texts)
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(query, text['text'])
            cached_score = self._relevance_cache.get(cache_key)
            if cached_score is not None:
                relevance_scores[i] = cached_score
            else:
                uncached_texts.append(text['text'])
                uncached_indices.append(i)
        
        # Only query LLM for uncached texts
        if uncached_texts:
            prompt = f"""
            Query: {query}
            
            For each text chunk below, assign a relevance score from 0-10 based on how well it answers the query.
            Consider:
            - Direct answer to the query (high relevance)
            - Related information (medium relevance)
            - Tangential information (low relevance)
            
            Return only the numerical scores in order, one per line.
            """
            
            try:
                # Get relevance scores from LLM for uncached texts
                scores_text = self.chatbot.generate_response(
                    context="\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(uncached_texts)]),
                    query=prompt
                )
                
                # Parse scores
                try:
                    new_scores = [float(score) for score in scores_text.strip().split('\n')]
                    
                    # Verify we got the expected number of scores
                    if len(new_scores) != len(uncached_texts):
                        raise ValueError("Number of scores doesn't match number of texts")
                    
                    # Update cache and scores array
                    for i, score in enumerate(new_scores):
                        original_idx = uncached_indices[i]
                        relevance_scores[original_idx] = score
                        cache_key = self._get_cache_key(query, texts[original_idx]['text'])
                        self._relevance_cache[cache_key] = score
                        
                except (ValueError, IndexError):
                    # Use similarity scores as fallback for parsing failure
                    for idx in uncached_indices:
                        relevance_scores[idx] = 1 - distances[idx]
                        
            except Exception:
                # Use similarity scores as fallback for LLM failure
                for idx in uncached_indices:
                    relevance_scores[idx] = 1 - distances[idx]
        
        # Combine scores and create results
        max_distance = max(distances) if distances else 1.0
        results = []
        
        for i in range(len(texts)):
            # Normalize distance to 0-1 (lower is better)
            norm_distance = 1 - (distances[i] / max_distance)
            # Normalize relevance to 0-1 (higher is better)
            norm_relevance = relevance_scores[i] / 10
            # Weighted combination (adjustable weights)
            combined_score = (0.4 * norm_distance) + (0.6 * norm_relevance)
            
            results.append({
                'id': ids[i],
                'text': texts[i]['text'],
                'metadata': texts[i],
                'similarity_score': 1 - distances[i],
                'relevance_score': relevance_scores[i],
                'combined_score': combined_score
            })
        
        logger.info(f"\nReranking results for query: {query}")
        logger.info("Pre-rerank ordering:")
        for r in results:
            logger.info(f"ID: {r['id']}, Text: {r['text'][:50]}...")
            logger.info(f"  Similarity: {r['similarity_score']:.4f}, Relevance: {r['relevance_score']:.4f}, Combined: {r['combined_score']:.4f}")

        # Sort by combined score and ID for consistent ordering
        results.sort(key=lambda x: (-x['combined_score'], x['id']))

        logger.info("\nPost-rerank ordering:")
        for r in results:
            logger.info(f"ID: {r['id']}, Text: {r['text'][:50]}...")
            logger.info(f"  Similarity: {r['similarity_score']:.4f}, Relevance: {r['relevance_score']:.4f}, Combined: {r['combined_score']:.4f}")
        
        return results

    def search(self, query: str, n_results: int = 5, source_names: Optional[List[str]] = None, title: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform the complete search process from query to ranked results.
        
        Args:
            query (str): The user's natural language query
            n_results (int): Number of results to return (default: 5)
            source_names (Optional[List[str]]): Optional list of source filenames to filter by
            title (Optional[str]): Optional filter by document title
            
        Returns:
            List of relevant chunks with metadata, ordered by relevance
            
        Raises:
            Exception: If any step in the search pipeline fails
        """
        if not isinstance(n_results, int) or n_results < 1:
            raise Exception("Error performing search: n_results must be a positive integer")
            
        try:
            # Log search parameters
            logger.info(f"Query filters - source_names: {source_names}, title: {title}")
            logger.info(f"Processing query: {query}")
            if source_names:
                logger.info(f"Filtering by source names: {source_names}")
            
            # 1. Parse query
            parsed_query = self.parse_query(query)
            
            # 2. Generate query embedding
            query_embedding = self.generate_query_embedding(parsed_query['processed_query'])
            
            # 3. Perform similarity search
            # Get more results than needed for reranking
            search_results = self.perform_similarity_search(
                query_embedding=query_embedding,
                n_results=n_results * 2,
                source_names=source_names,
                title=title
            )
            
            # 4. Rerank results
            reranked_results = self.rerank_results(query, search_results)
            
            # 5. Return top N results after reranking
            return reranked_results[:n_results]
            
        except Exception as e:
            raise Exception(f"Error performing search: {str(e)}")
