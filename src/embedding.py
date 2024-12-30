from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL_NAME

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
            
        return self.model.encode(texts, show_progress_bar=True)
