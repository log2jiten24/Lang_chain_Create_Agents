from data_loader import load_all_documents
from Embeddings import get_default_embeddings
from vector_store import VectorStoreManager, load_vector_store
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def create_and_save_vector_store():
    """Create a new vector store from documents and save it."""
    print("=" * 60)
    print("Creating new vector store from documents...")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../sample_data/pdf_files")
    vector_store_path = os.path.join(script_dir, "../data_storage/vector_store")

    # Step 1: Load documents
    print("\n[1/4] Loading documents...")
    docs = load_all_documents(data_path)
    print(f"Successfully loaded {len(docs)} documents")

    # Step 2: Initialize embeddings
    print("\n[2/4] Initializing embeddings model...")
    embeddings = get_default_embeddings()

    # Step 3: Create vector store
    print("\n[3/4] Creating vector store...")
    vector_manager = VectorStoreManager(embeddings)
    vector_manager.create_vector_store(docs)

    # Step 4: Save vector store
    print("\n[4/4] Saving vector store to disk...")
    vector_manager.save(vector_store_path)

    print("\n" + "=" * 60)
    print("Vector store created and saved successfully!")
    print(f"Location: {vector_store_path}")
    print("=" * 60)

    return vector_manager, vector_store_path


def load_and_query_vector_store(vector_store_path: str):
    """Load an existing vector store and perform a test query."""
    print("\n" + "=" * 60)
    print("Loading existing vector store...")
    print("=" * 60)

    # Step 1: Initialize embeddings
    print("\n[1/2] Initializing embeddings model...")
    embeddings = get_default_embeddings()

    # Step 2: Load vector store
    print("\n[2/2] Loading vector store from disk...")
    vector_manager = load_vector_store(vector_store_path, embeddings)

    print("\n" + "=" * 60)
    print("Vector store loaded successfully!")
    print("=" * 60)

    # Perform a test query
    print("\n" + "=" * 60)
    print("Testing similarity search...")
    print("=" * 60)

    query = "What is the company's revenue?"
    print(f"\nQuery: '{query}'")
    print("\nSearching for relevant documents...\n")

    results = vector_manager.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (similarity score: {score:.4f}):")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Page: {doc.metadata.get('page', 'N/A')}")
        print(f"  Content preview: {doc.page_content[:200]}...")
        print()

    return vector_manager


# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vector_store_path = os.path.join(script_dir, "../data_storage/vector_store")

    # Check if vector store already exists
    if os.path.exists(vector_store_path):
        print("\nVector store already exists!")
        print(f"Location: {vector_store_path}")
        print("\nOptions:")
        print("  1. Load existing vector store (default)")
        print("  2. Create new vector store (will overwrite existing)")

        choice = input("\nEnter your choice (1 or 2): ").strip()

        if choice == "2":
            vector_manager = create_and_save_vector_store()[0]
        else:
            vector_manager = load_and_query_vector_store(vector_store_path)
    else:
        # Vector store doesn't exist, create it
        vector_manager = create_and_save_vector_store()[0]

    print("\n" + "=" * 60)
    print("Ready to use! You can now:")
    print("  - Query documents using vector_manager.similarity_search()")
    print("  - Get retriever for chains using vector_manager.get_retriever()")
    print("=" * 60)
