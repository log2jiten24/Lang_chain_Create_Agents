"""
Example usage of the Langchain Agent with Anthropic API

This script demonstrates how to use the agent for various tasks.
"""

from agent import run_agent


def main():
    """Run example queries through the agent."""
    
    print("=" * 60)
    print("Langchain Agent with Anthropic API - Examples")
    print("=" * 60)
    
    # List of example queries
    examples = [
        "What is the current time?",
        "Calculate 123 * 456",
        "What is the square root of 144?",
        "What is the current time and what is 100 divided by 5?"
    ]
    
    for i, query in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: {query}")
        print('='*60)
        
        try:
            result = run_agent(query)
            
            # Extract the final answer from the messages
            if 'messages' in result:
                final_message = result['messages'][-1]
                print(f"\nAnswer: {final_message.content}")
            else:
                print(f"\nAnswer: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")
            if "ANTHROPIC_API_KEY" in str(e):
                print("\nPlease ensure you have:")
                print("1. Created a .env file based on .env.example")
                print("2. Added your Anthropic API key to the .env file")
                break
        
        print()


if __name__ == "__main__":
    main()
