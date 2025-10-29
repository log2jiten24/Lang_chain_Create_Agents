#!/usr/bin/env python3
"""
Copilot CLI - Command-line interface for LangChain Agent

This module provides an interactive and one-shot command-line interface
for interacting with the LangChain agent powered by Anthropic's Claude.
"""

import argparse
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import create_agent, run_agent


class CopilotCLI:
    """Command-line interface for the LangChain agent."""
    
    def __init__(self, model="claude-3-5-sonnet-20241022", temperature=0.7, verbose=False):
        """
        Initialize the CLI.
        
        Args:
            model (str): Claude model to use
            temperature (float): Temperature for response generation (0.0-1.0)
            verbose (bool): Enable verbose output
        """
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.agent = None
        self.session_history = []
        
    def initialize_agent(self):
        """Initialize the agent with configured parameters."""
        try:
            if self.verbose:
                print(f"Initializing agent with model: {self.model}, temperature: {self.temperature}")
            self.agent = create_agent()
            return True
        except Exception as e:
            print(f"Error initializing agent: {e}", file=sys.stderr)
            return False
    
    def query(self, message):
        """
        Send a query to the agent and get a response.
        
        Args:
            message (str): The user's query
            
        Returns:
            str: The agent's response
        """
        try:
            if self.verbose:
                print(f"\nProcessing query: {message}")
                
            result = run_agent(message)
            
            # Extract the final answer from the messages
            if 'messages' in result:
                response = result['messages'][-1].content
            else:
                response = str(result)
            
            # Store in session history
            self.session_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': message,
                'response': response
            })
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            if self.verbose:
                import traceback
                error_msg += f"\n{traceback.format_exc()}"
            return error_msg
    
    def interactive_mode(self):
        """Run the CLI in interactive mode with continuous conversation."""
        print("=" * 70)
        print("LangChain Agent - Interactive Mode")
        print("=" * 70)
        print("\nWelcome! I'm your AI assistant powered by Claude.")
        print("I can help you with:")
        print("  • Getting current date and time")
        print("  • Performing mathematical calculations")
        print("  • Answering questions and general assistance")
        print("\nType 'exit', 'quit', or 'q' to end the session.")
        print("Type 'help' for available commands.")
        print("Type 'history' to see your conversation history.")
        print("=" * 70)
        
        if not self.initialize_agent():
            print("\nFailed to initialize agent. Exiting.", file=sys.stderr)
            return 1
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye! Thanks for using the LangChain Agent.")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    continue
                
                # Process the query
                response = self.query(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit or continue chatting.")
                continue
            except EOFError:
                print("\n\nGoodbye!")
                break
        
        return 0
    
    def one_shot_mode(self, query):
        """
        Run a single query and return the response.
        
        Args:
            query (str): The user's query
            
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        if not self.initialize_agent():
            return 1
        
        response = self.query(query)
        print(response)
        
        return 0
    
    def show_help(self):
        """Display help information."""
        print("\n" + "=" * 70)
        print("Available Commands:")
        print("=" * 70)
        print("  help     - Show this help message")
        print("  history  - Show conversation history")
        print("  clear    - Clear the screen")
        print("  exit     - Exit the interactive session (also: quit, q)")
        print("\nAvailable Tools:")
        print("  • get_current_time - Get the current date and time")
        print("  • calculate        - Perform mathematical calculations")
        print("\nExample Queries:")
        print("  • What is the current time?")
        print("  • Calculate 123 * 456")
        print("  • What is 2 to the power of 10?")
        print("  • What is the current time and what is 100 / 5?")
        print("=" * 70)
    
    def show_history(self):
        """Display conversation history."""
        if not self.session_history:
            print("\nNo conversation history yet.")
            return
        
        print("\n" + "=" * 70)
        print("Conversation History:")
        print("=" * 70)
        
        for i, entry in enumerate(self.session_history, 1):
            print(f"\n[{i}] {entry['timestamp']}")
            print(f"You: {entry['query']}")
            print(f"Assistant: {entry['response'][:200]}..." if len(entry['response']) > 200 else f"Assistant: {entry['response']}")
        
        print("=" * 70)


def main():
    """Main entry point for the CLI."""
    # Load environment variables
    load_dotenv()
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="LangChain Agent CLI - Interact with Claude-powered AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python cli.py
  
  # One-shot query
  python cli.py "What is the current time?"
  
  # One-shot with options
  python cli.py --verbose "Calculate 123 * 456"
  
  # Interactive mode with custom model
  python cli.py --interactive --model claude-3-opus-20240229
        """
    )
    
    # Positional argument for one-shot query
    parser.add_argument(
        'query',
        nargs='?',
        help='Query to send to the agent (omit for interactive mode)'
    )
    
    # Optional arguments
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Force interactive mode even if query is provided'
    )
    
    parser.add_argument(
        '-m', '--model',
        default='claude-3-5-sonnet-20241022',
        help='Claude model to use (default: claude-3-5-sonnet-20241022)'
    )
    
    parser.add_argument(
        '-t', '--temperature',
        type=float,
        default=0.7,
        help='Temperature for response generation, 0.0-1.0 (default: 0.7)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='LangChain Agent CLI v1.0.0'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate temperature
    if not 0.0 <= args.temperature <= 1.0:
        print("Error: Temperature must be between 0.0 and 1.0", file=sys.stderr)
        return 1
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not found in environment variables.", file=sys.stderr)
        print("\nPlease ensure you have:", file=sys.stderr)
        print("1. Created a .env file based on .env.example", file=sys.stderr)
        print("2. Added your Anthropic API key to the .env file", file=sys.stderr)
        return 1
    
    # Create CLI instance
    cli = CopilotCLI(
        model=args.model,
        temperature=args.temperature,
        verbose=args.verbose
    )
    
    # Determine mode: interactive or one-shot
    if args.interactive or args.query is None:
        # Interactive mode
        return cli.interactive_mode()
    else:
        # One-shot mode
        return cli.one_shot_mode(args.query)


if __name__ == "__main__":
    sys.exit(main())
