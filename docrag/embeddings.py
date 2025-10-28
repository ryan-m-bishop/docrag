from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a sentence-transformers model"""
        self.model_name = model_name
        self.model = None

    def load(self):
        """Lazy load the model"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        self.load()
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.embed([text])[0]
