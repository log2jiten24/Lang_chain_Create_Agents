"""
Vector store module for managing FAISS vector database.
Handles document storage, retrieval, and persistence.
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    """
    Manager class for FAISS vector store operations.
    Handles document chunking, embedding, storage, and retrieval.
    """

    def __init__(self, embeddings, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the vector store manager.

        Args:
            embeddings: Embeddings model instance (e.g., HuggingFaceEmbeddings)
            chunk_size: Size of text chunks (default: 1000 characters)
            chunk_overlap: Overlap between chunks (default: 200 characters)
        """
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a new vector store from documents.

        Args:
            documents: List of Document objects to index

        Returns:
            FAISS vector store instance
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")

        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        print("Creating vector store with embeddings...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("Vector store created successfully")

        return self.vector_store

    def add_documents(self, documents: List[Document]):
        """
        Add new documents to existing vector store.

        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Use create_vector_store first.")

        chunks = self.text_splitter.split_documents(documents)
        print(f"Adding {len(chunks)} new chunks to vector store...")
        self.vector_store.add_documents(chunks)
        print("Documents added successfully")

    def save(self, folder_path: str):
        """
        Save the vector store to disk.

        Args:
            folder_path: Directory path where vector store will be saved
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")

        os.makedirs(folder_path, exist_ok=True)
        self.vector_store.save_local(folder_path)
        print(f"Vector store saved to: {folder_path}")

    def load(self, folder_path: str) -> FAISS:
        """
        Load a vector store from disk.

        Args:
            folder_path: Directory path where vector store is saved

        Returns:
            Loaded FAISS vector store instance
        """
        if not os.path.exists(folder_path):
            raise ValueError(f"Vector store path does not exist: {folder_path}")

        print(f"Loading vector store from: {folder_path}")
        self.vector_store = FAISS.load_local(
            folder_path,
            self.embeddings,
            allow_dangerous_deserialization=True  # Required for FAISS
        )
        print("Vector store loaded successfully")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query text
            k: Number of results to return (default: 4)

        Returns:
            List of most similar Document objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        results = self.vector_store.similarity_search(query, k=k)
        return results

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Search for documents with similarity scores.

        Args:
            query: Search query text
            k: Number of results to return (default: 4)

        Returns:
            List of (Document, score) tuples, where lower score = more similar
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever for use in chains.

        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 4})

        Returns:
            VectorStoreRetriever instance
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        if search_kwargs is None:
            search_kwargs = {"k": 4}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)


def create_vector_store_from_documents(
    documents: List[Document],
    embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> VectorStoreManager:
    """
    Convenience function to create a vector store from documents.

    Args:
        documents: List of Document objects
        embeddings: Embeddings model instance
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Initialized VectorStoreManager with created vector store
    """
    manager = VectorStoreManager(embeddings, chunk_size, chunk_overlap)
    manager.create_vector_store(documents)
    return manager


def load_vector_store(
    folder_path: str,
    embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> VectorStoreManager:
    """
    Convenience function to load a vector store from disk.

    Args:
        folder_path: Path to saved vector store
        embeddings: Embeddings model instance
        chunk_size: Size of text chunks (for adding new docs)
        chunk_overlap: Overlap between chunks (for adding new docs)

    Returns:
        VectorStoreManager with loaded vector store
    """
    manager = VectorStoreManager(embeddings, chunk_size, chunk_overlap)
    manager.load(folder_path)
    return manager
