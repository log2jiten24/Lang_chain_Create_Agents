"""
Embeddings module for creating document embeddings using HuggingFace models.
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingManager:
    """
    Manager class for handling document embeddings.
    Uses HuggingFace sentence-transformers models.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager with a specific model.

        Args:
            model_name: HuggingFace model name for embeddings.
                       Default is 'all-MiniLM-L6-v2' (fast, lightweight, 384 dimensions)

        Popular models:
            - sentence-transformers/all-MiniLM-L6-v2: Fast, good quality (384 dim)
            - sentence-transformers/all-mpnet-base-v2: Better quality, slower (768 dim)
            - BAAI/bge-small-en-v1.5: Good balance (384 dim)
            - BAAI/bge-base-en-v1.5: Higher quality (768 dim)
        """
        self.model_name = model_name
        self.embeddings = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize the embeddings model."""
        try:
            print(f"Initializing embedding model: {self.model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Use 'cuda' for GPU
                encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
            )
            print("Embedding model initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {e}")

    def get_embeddings(self):
        """
        Get the initialized embeddings model.

        Returns:
            HuggingFaceEmbeddings instance
        """
        if self.embeddings is None:
            self._initialize_embeddings()
        return self.embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        if self.embeddings is None:
            self._initialize_embeddings()
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Create embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        if self.embeddings is None:
            self._initialize_embeddings()
        return self.embeddings.embed_query(text)


def get_default_embeddings():
    """
    Get default embeddings model (all-MiniLM-L6-v2).

    Returns:
        HuggingFaceEmbeddings instance
    """
    manager = EmbeddingManager()
    return manager.get_embeddings()


def get_high_quality_embeddings():
    """
    Get high-quality embeddings model (all-mpnet-base-v2).
    Note: Slower but more accurate than default model.

    Returns:
        HuggingFaceEmbeddings instance
    """
    manager = EmbeddingManager(model_name="sentence-transformers/all-mpnet-base-v2")
    return manager.get_embeddings()
