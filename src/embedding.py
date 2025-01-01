import os
import time
import logging
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Set cache directory for sentence-transformers in the project directory
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_cache')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

class EmbeddingGenerator:
    def __init__(self):
        """Initialize the embedding generator with the specified model."""
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to generate embeddings for
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        if not isinstance(texts, list):
            texts = [texts]
            
        start_time = time.time()
        logger.info(f"Starting embedding generation for {len(texts)} texts...")
        
        # Optimize batch size for CPU
        batch_size = min(32, len(texts))  # Use smaller batch size for CPU
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False,  # Don't need PyTorch tensors
            convert_to_numpy=True,    # Faster conversion
            normalize_embeddings=False # Don't need normalized embeddings
        )
        
        end_time = time.time()
        duration = end_time - start_time
        per_text = duration / len(texts) if texts else 0
        logger.info(f"Embedding generation completed in {duration:.2f}s ({per_text:.2f}s per text)")
        
        return embeddings
