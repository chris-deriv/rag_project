from typing import List, Dict, Any
import numpy as np
from .embedding import EmbeddingGenerator
from .database import VectorDatabase
from .chatbot import Chatbot

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

    def perform_similarity_search(self, query_embedding: np.ndarray, n_results: int = 10) -> Dict[str, Any]:
        """
        Perform similarity search in the vector database.
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to retrieve (default: 10)
            
        Returns:
            Dict containing search results with distances and metadata
            
        Raises:
            Exception: If database query fails or if n_results is invalid
        """
        if not isinstance(n_results, int) or n_results < 1:
            raise Exception("n_results must be a positive integer")
            
        try:
            results = self.vector_db.query(query_embedding, n_results=n_results)
            
            # Ensure results have expected structure
            if not all(key in results for key in ['ids', 'distances', 'metadatas']):
                raise Exception("Invalid results structure from database")
                
            # Sort results by distance and ID for consistent ordering
            sorted_indices = np.lexsort((results['ids'][0], results['distances'][0]))
            
            # Reorder all result arrays using the sorted indices
            results['ids'][0] = [results['ids'][0][i] for i in sorted_indices]
            results['distances'][0] = [results['distances'][0][i] for i in sorted_indices]
            results['metadatas'][0] = [results['metadatas'][0][i] for i in sorted_indices]
            
            return results
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
        
        # Sort by combined score and ID for consistent ordering
        results.sort(key=lambda x: (-x['combined_score'], x['id']))
        
        return results

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform the complete search process from query to ranked results.
        
        Args:
            query (str): The user's natural language query
            n_results (int): Number of results to return (default: 5)
            
        Returns:
            List of relevant chunks with metadata, ordered by relevance
            
        Raises:
            Exception: If any step in the search pipeline fails
        """
        if not isinstance(n_results, int) or n_results < 1:
            raise Exception("Error performing search: n_results must be a positive integer")
            
        try:
            # 1. Parse query
            parsed_query = self.parse_query(query)
            
            # 2. Generate query embedding
            query_embedding = self.generate_query_embedding(parsed_query['processed_query'])
            
            # 3. Perform similarity search
            # Get more results than needed for reranking
            search_results = self.perform_similarity_search(query_embedding, n_results=n_results * 2)
            
            # 4. Rerank results
            reranked_results = self.rerank_results(query, search_results)
            
            # 5. Return top N results after reranking
            return reranked_results[:n_results]
            
        except Exception as e:
            raise Exception(f"Error performing search: {str(e)}")
