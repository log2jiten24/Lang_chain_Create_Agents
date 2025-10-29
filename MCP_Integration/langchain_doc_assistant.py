"""
LangChain Documentation Assistant
Interactive CLI for querying LangChain documentation
"""

from langchain_doc_rag import LangChainDocRAG
from dotenv import load_dotenv
import sys
import os


def print_header():
    """Print application header"""
    print("\n" + "="*70)
    print("  LangChain Documentation Assistant")
    print("  Powered by RAG + Claude")
    print("="*70)


def print_result(result: dict):
    """Pretty print query result"""
    print("\n" + "─"*70)
    print("📖 Answer:")
    print("─"*70)
    print(result['answer'])

    if result['sources']:
        print("\n" + "─"*70)
        print("📚 Documentation Sources:")
        print("─"*70)
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['title']}")
            print(f"     {source['url']}")

    print("="*70)


def main():
    """Main interactive loop"""
    load_dotenv()

    print_header()

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("Please set it in your .env file or environment variables")
        sys.exit(1)

    # Initialize RAG system
    print("\nInitializing RAG system...")
    print("  [1/2] Loading vector store...")

    try:
        rag = LangChainDocRAG()
        rag.load_vector_store()
        print("  ✓ Vector store loaded")

        print("  [2/2] Setting up Q&A chain...")
        rag.setup_qa_chain()
        print("  ✓ Q&A chain ready")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run these commands first:")
        print("  1. python langchain_doc_scraper.py")
        print("  2. python langchain_doc_rag.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error initializing: {e}")
        sys.exit(1)

    # Interactive loop
    print("\n" + "="*70)
    print("✓ Ready! Ask me anything about LangChain.")
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'search: <query>' for similarity search")
    print("  - Type 'examples' for example questions")
    print("  - Type 'quit' or 'exit' to quit")
    print("="*70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Happy coding with LangChain!")
                break

            elif user_input.lower() == 'examples':
                print("\n💡 Example Questions:")
                print("  1. How do I create a ChatAnthropic model with memory?")
                print("  2. What is the difference between chains and agents?")
                print("  3. How do I use LCEL (LangChain Expression Language)?")
                print("  4. Show me how to build a RAG system with LangChain")
                print("  5. How do I implement streaming with LangChain?")
                print("  6. What are the best practices for prompt engineering?")
                continue

            elif user_input.lower().startswith('search:'):
                query = user_input[7:].strip()
                print(f"\n🔍 Searching for: {query}")
                results = rag.search_similar(query, k=5)

                print("\nTop 5 Similar Documentation Sections:")
                print("─"*70)
                for i, res in enumerate(results, 1):
                    print(f"\n{i}. {res['title']} (similarity: {res['score']:.4f})")
                    print(f"   {res['source']}")
                    print(f"   {res['content'][:200]}...")
                print("="*70)
                continue

            # Regular question
            print("\n🤔 Thinking...")
            result = rag.query(user_input)
            print_result(result)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
