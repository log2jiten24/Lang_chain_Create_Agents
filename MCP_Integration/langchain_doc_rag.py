"""
LangChain Documentation RAG System
Provides intelligent Q&A over LangChain documentation
"""

import json
import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChainDocRAG:
    """RAG system for LangChain documentation"""

    def __init__(self, docs_file: str = "langchain_docs.json"):
        self.docs_file = docs_file
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for documentation
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self) -> List[Document]:
        """Load scraped documentation and convert to LangChain Documents"""
        if not os.path.exists(self.docs_file):
            raise FileNotFoundError(
                f"Documentation file not found: {self.docs_file}\n"
                f"Run langchain_doc_scraper.py first to scrape documentation."
            )

        with open(self.docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)

        documents = []
        for doc in docs_data:
            # Main content
            content = doc['content']

            # Add code examples if present
            if doc.get('code_examples'):
                content += "\n\nCode Examples:\n" + "\n\n".join(doc['code_examples'])

            documents.append(Document(
                page_content=content,
                metadata={
                    'source': doc['url'],
                    'title': doc['title'],
                    'scraped_at': doc.get('scraped_at', 'unknown')
                }
            ))

        print(f"✓ Loaded {len(documents)} documentation pages")
        return documents

    def create_vector_store(self, documents: List[Document]):
        """Create FAISS vector store with embeddings"""
        print("\n[1/3] Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embeddings model ready")

        print("\n[2/3] Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")

        print("\n[3/3] Creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("✓ Vector store created successfully")

    def save_vector_store(self, path: str = "langchain_docs_vectorstore"):
        """Persist vector store to disk"""
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")

        self.vector_store.save_local(path)
        print(f"\n✓ Vector store saved to: {path}")

        # Save metadata
        metadata = {
            'source_file': self.docs_file,
            'chunk_size': self.text_splitter._chunk_size,
            'chunk_overlap': self.text_splitter._chunk_overlap,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_vector_store(self, path: str = "langchain_docs_vectorstore"):
        """Load existing vector store from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Vector store not found at: {path}\n"
                f"Run create_vector_store() first."
            )

        print(f"Loading vector store from: {path}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("✓ Vector store loaded successfully")

        # Load metadata if available
        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"  Embedding model: {metadata['embedding_model']}")
                print(f"  Chunk size: {metadata['chunk_size']}")

    def setup_qa_chain(self, temperature: float = 0):
        """Setup Q&A chain with Claude"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load or create one first.")

        print("\nSetting up Q&A chain...")

        # Initialize Claude
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=temperature,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        print("✓ Claude LLM initialized")

        # Create prompt template
        prompt_template = """You are an expert on LangChain, a framework for building LLM applications.

Use the following documentation excerpts to answer the question.
If you find relevant code examples, include them in your answer.
If the answer is not in the documentation, say you don't know.

LangChain Documentation Context:
{context}

Question: {question}

Answer: Provide a detailed, accurate answer based on the documentation.
Include:
1. Clear explanation
2. Code examples when relevant
3. Best practices
4. Links to relevant documentation sections"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        print("✓ Prompt template created")

        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}  # Return top 4 most relevant chunks
        )
        print("✓ Retriever configured (k=4)")

        # Build Q&A chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        print("✓ Q&A chain ready\n")

    def query(self, question: str, verbose: bool = False) -> Dict:
        """Query the documentation"""
        if self.qa_chain is None:
            self.setup_qa_chain()

        if verbose:
            print(f"Querying: {question}")
            print("Searching documentation...\n")

        response = self.qa_chain.invoke({"query": question})

        result = {
            'question': question,
            'answer': response['result'],
            'sources': []
        }

        # Extract unique sources
        seen_urls = set()
        for doc in response['source_documents']:
            url = doc.metadata['source']
            if url not in seen_urls:
                result['sources'].append({
                    'url': url,
                    'title': doc.metadata['title']
                })
                seen_urls.add(url)

        return result

    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documentation chunks"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        docs = self.vector_store.similarity_search_with_score(query, k=k)

        results = []
        for doc, score in docs:
            results.append({
                'content': doc.page_content[:300] + "...",
                'score': float(score),
                'source': doc.metadata['source'],
                'title': doc.metadata['title']
            })

        return results


def main():
    """Main execution for setup"""
    from dotenv import load_dotenv
    load_dotenv()

    print("="*70)
    print("LangChain Documentation RAG System - Setup")
    print("="*70)

    rag = LangChainDocRAG()

    # Check if vector store already exists
    vector_store_path = "langchain_docs_vectorstore"
    if os.path.exists(vector_store_path):
        print(f"\n✓ Vector store already exists at: {vector_store_path}")
        print("To rebuild, delete the directory and run again.\n")
        return

    # Load documents
    print("\nStep 1: Loading documentation...")
    documents = rag.load_documents()

    # Create vector store
    print("\nStep 2: Creating vector store...")
    rag.create_vector_store(documents)

    # Save vector store
    print("\nStep 3: Saving vector store...")
    rag.save_vector_store(vector_store_path)

    print("\n" + "="*70)
    print("✓ Setup Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python langchain_doc_assistant.py")
    print("  2. Start asking questions about LangChain!")
    print("="*70)


if __name__ == "__main__":
    main()
