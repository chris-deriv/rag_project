from typing import List, Dict, Any, Optional
import numpy as np
import logging
import copy
from src.embedding import EmbeddingGenerator
from src.database import VectorDatabase
from src.chatbot import Chatbot
from src.config.dynamic_settings import settings_manager

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self):
        """Initialize the search engine with required components."""
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        self.chatbot = Chatbot()
        # Cache for LLM relevance scores to ensure consistency
        self._relevance_cache = {}
        
        # Get initial settings
        self.settings = settings_manager.get_all_settings()
        
        # Register as observer for settings changes
        settings_manager.add_observer(self._handle_settings_change)

    def _handle_settings_change(self, setting_name: str, new_value: dict) -> None:
        """Handle settings changes from the settings manager."""
        if setting_name in ['llm', 'response']:
            # Deep copy the new settings to ensure nested dicts are properly updated
            self.settings[setting_name] = copy.deepcopy(new_value)
            # Clear cache when relevant settings change
            self._relevance_cache.clear()
            logger.info(f"Cleared relevance cache due to {setting_name} settings change")

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse the natural language query to extract key information."""
        if query is None:
            raise Exception("Query cannot be None")
        return {
            "original_query": query,
            "processed_query": query.strip().lower()
        }

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query text."""
        try:
            return self.embedding_generator.generate_embeddings([query])[0]
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")

    def perform_similarity_search(self, 
                                query_embedding: np.ndarray, 
                                n_results: int = 10,
                                source_names: Optional[List[str]] = None,
                                title: Optional[str] = None) -> Dict[str, Any]:
        """Perform similarity search in the vector database."""
        if not isinstance(n_results, int) or n_results < 1:
            raise Exception("n_results must be a positive integer")
            
        try:
            results = self.vector_db.query(
                query_embedding=query_embedding, 
                n_results=n_results,
                source_names=source_names,
                title=title
            )
            
            # Handle empty results
            if not results['ids'][0]:
                return {
                    'ids': [[]],
                    'distances': [[]],
                    'metadatas': [[]]
                }
            
            # Convert numpy arrays to lists and sort results
            ids = results['ids'][0].tolist() if isinstance(results['ids'][0], np.ndarray) else results['ids'][0]
            distances = results['distances'][0].tolist() if isinstance(results['distances'][0], np.ndarray) else results['distances'][0]
            metadatas = results['metadatas'][0]
            
            # Sort by distance and ID for consistency
            sorted_data = sorted(zip(distances, ids, metadatas), key=lambda x: (x[0], x[1]))
            sorted_distances, sorted_ids, sorted_metadatas = zip(*sorted_data)
            
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
        """Rerank search results using LLM relevance scoring."""
        if not search_results['metadatas'][0]:
            return []
            
        texts = search_results['metadatas'][0]
        distances = search_results['distances'][0]
        ids = search_results['ids'][0]
        
        # Single result case
        if len(texts) == 1:
            return [{
                'id': ids[0],
                'text': texts[0]['text'],
                'metadata': texts[0],
                'similarity_score': 1 - distances[0],
                'relevance_score': 1.0,
                'combined_score': 1.0
            }]
        
        # Check cache and collect uncached texts
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
        
        # Get scores for uncached texts
        if uncached_texts:
            prompt = f"""
            Query: {query}
            Rate each text chunk's relevance (0-10) based on how well it answers the query.
            Consider:
            - Direct answers (high relevance)
            - Related information (medium)
            - Tangential information (low)
            Return only numerical scores, one per line.
            """
            
            try:
                scores_text = self.chatbot.generate_response(
                    context="\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(uncached_texts)]),
                    query=prompt
                )
                
                # Parse and validate scores
                new_scores = [float(score) for score in scores_text.strip().split('\n')]
                if len(new_scores) == len(uncached_texts):
                    for i, score in enumerate(new_scores):
                        original_idx = uncached_indices[i]
                        relevance_scores[original_idx] = score
                        cache_key = self._get_cache_key(query, texts[original_idx]['text'])
                        self._relevance_cache[cache_key] = score
                else:
                    # Fallback to similarity scores
                    for idx in uncached_indices:
                        relevance_scores[idx] = 1 - distances[idx]
            except Exception:
                # Fallback to similarity scores
                for idx in uncached_indices:
                    relevance_scores[idx] = 1 - distances[idx]
        
        # Combine scores using current settings
        results = []
        max_distance = max(distances) if distances else 1.0
        
        for i in range(len(texts)):
            norm_distance = 1 - (distances[i] / max_distance)
            norm_relevance = relevance_scores[i] / 10
            # Use temperature from settings to adjust weighting
            temp = self.settings['llm']['temperature']
            # Higher temperature -> more weight on relevance scores
            relevance_weight = 0.5 + (temp * 0.2)  # 0.5-0.9 based on temperature
            distance_weight = 1 - relevance_weight
            combined_score = (distance_weight * norm_distance) + (relevance_weight * norm_relevance)
            
            results.append({
                'id': ids[i],
                'text': texts[i]['text'],
                'metadata': texts[i],
                'similarity_score': 1 - distances[i],
                'relevance_score': relevance_scores[i],
                'combined_score': combined_score
            })
        
        # Sort by combined score and ID for consistency
        results.sort(key=lambda x: (-x['combined_score'], x['id']))
        return results

    def search(self, query: str, n_results: int = 5, source_names: Optional[List[str]] = None, title: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform complete search process from query to ranked results."""
        if not isinstance(n_results, int) or n_results < 1:
            raise Exception("n_results must be a positive integer")
            
        try:
            # Process query and get embeddings
            parsed_query = self.parse_query(query)
            query_embedding = self.generate_query_embedding(parsed_query['processed_query'])
            
            # Get initial results
            search_results = self.perform_similarity_search(
                query_embedding=query_embedding,
                n_results=n_results * 2,  # Get extra for reranking
                source_names=source_names,
                title=title
            )
            
            # Rerank and return top results
            reranked_results = self.rerank_results(query, search_results)
            return reranked_results[:n_results]
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")

    def __del__(self):
        """Clean up by removing observer when object is destroyed."""
        try:
            settings_manager.remove_observer(self._handle_settings_change)
        except:
            pass  # Ignore errors during cleanup
