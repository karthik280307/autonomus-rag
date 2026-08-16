from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    
    def __init__(self, model_name:str= "all-MiniLM-L6-v2"):
        self.model_name=model_name
        self.model=None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model=SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded: {self.model_name} with dimension {self.model.get_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Exception loading embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: list[str]) ->np.ndarray:

        if self.model is None:
            logger.error(f"Model not loaded: {self.model_name}")
            raise ValueError('model not loaded')
        
        if len(texts) == 0:
            logger.error("Cannot generate embeddings: texts are empty")
            raise ValueError('texts are empty')
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings=self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        logger.info(f"Embedding dimension: {embeddings.shape}")

        return embeddings


