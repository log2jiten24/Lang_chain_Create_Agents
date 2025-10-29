"""
Langchain Agent using Anthropic API

This module provides a simple agent implementation using Langchain and Anthropic's Claude model.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
# Note: This import is deprecated but will continue to work
# The suggested replacement doesn't have compatible API yet
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Perform basic mathematical calculations. Input should be a valid Python expression with numbers and operators (+, -, *, /, **, %)."""
    try:
        # Use a safer approach with ast.literal_eval for numeric expressions
        # Only allow basic mathematical operations
        import ast
        import operator
        
        # Define allowed operators
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
        }
        
        def eval_expr(node):
            if isinstance(node, ast.Constant):  # Python 3.8+
                return node.value
            elif isinstance(node, ast.Num):  # number (for older Python versions)
                return node.n
            elif isinstance(node, ast.BinOp):  # binary operation
                return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):  # unary operation
                return operators[type(node.op)](eval_expr(node.operand))
            else:
                raise ValueError(f"Unsupported operation: {type(node).__name__}")
        
        # Parse and evaluate the expression
        node = ast.parse(expression, mode='eval')
        result = eval_expr(node.body)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def create_agent(model="claude-3-5-sonnet-20241022", temperature=0.7):
    """
    Create and return a Langchain agent with Anthropic's Claude model.
    
    Args:
        model (str): Claude model to use (default: "claude-3-5-sonnet-20241022")
        temperature (float): Temperature for response generation (default: 0.7)
    
    Returns:
        Compiled graph agent ready to process requests.
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")
    
    # Initialize the Claude model
    llm = ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        temperature=temperature
    )
    
    # Define tools for the agent
    tools = [get_current_time, calculate]

    # Create the agent using langgraph's create_react_agent
    agent_executor = create_react_agent(llm, tools)
    
    return agent_executor


def run_agent(query: str):
    """
    Run the agent with a given query.
    
    Args:
        query (str): The question or task for the agent to process.
        
    Returns:
        dict: The agent's response with messages.
    """
    agent_executor = create_agent()
    response = agent_executor.invoke({"messages": [("user", query)]})
    return response


if __name__ == "__main__":
    # Example usage
    print("Langchain Agent with Anthropic API\n")
    print("=" * 50)
    
    try:
        # Test query
        query = "What is the current time and what is 25 * 4?"
        print(f"\nQuery: {query}\n")
        
        result = run_agent(query)
        
        # Extract the final answer from the messages
        if 'messages' in result:
            final_message = result['messages'][-1]
            print(f"\nFinal Answer: {final_message.content}")
        else:
            print(f"\nResponse: {result}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have:")
        print("1. Created a .env file based on .env.example")
        print("2. Added your Anthropic API key to the .env file")
