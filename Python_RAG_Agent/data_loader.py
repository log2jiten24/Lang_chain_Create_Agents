"""
Data loader module for loading documents from various sources.
Supports PDF and text file formats.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.documents import Document


def load_all_documents(data_path: str) -> List[Document]:
    """
    Load all documents (PDF and text files) from the specified directory.

    Args:
        data_path: Path to the directory containing documents

    Returns:
        List of Document objects containing the loaded content
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Path does not exist: {data_path}")

    documents = []

    # Check if it's a single file or directory
    if os.path.isfile(data_path):
        # Load single file
        if data_path.endswith('.pdf'):
            loader = PyPDFLoader(data_path)
            documents.extend(loader.load())
        elif data_path.endswith('.txt'):
            loader = TextLoader(data_path)
            documents.extend(loader.load())
        else:
            print(f"Unsupported file type: {data_path}")
    else:
        # Load all files from directory
        # Load PDF files
        try:
            pdf_loader = DirectoryLoader(
                data_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            print(f"Error loading PDFs: {e}")

        # Load text files
        try:
            text_loader = DirectoryLoader(
                data_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            text_docs = text_loader.load()
            documents.extend(text_docs)
            print(f"Loaded {len(text_docs)} text documents")
        except Exception as e:
            print(f"Error loading text files: {e}")

    return documents


def load_pdf_documents(data_path: str) -> List[Document]:
    """
    Load only PDF documents from the specified directory.

    Args:
        data_path: Path to the directory containing PDF files

    Returns:
        List of Document objects containing the loaded PDF content
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Path does not exist: {data_path}")

    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()
    return documents


def load_text_documents(data_path: str) -> List[Document]:
    """
    Load only text documents from the specified directory.

    Args:
        data_path: Path to the directory containing text files

    Returns:
        List of Document objects containing the loaded text content
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Path does not exist: {data_path}")

    loader = DirectoryLoader(
        data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )

    documents = loader.load()
    return documents
