"""
Langchain Agent using Anthropic API

This module provides a simple agent implementation using Langchain and Anthropic's Claude model.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

# Load environment variables
load_dotenv()


def get_current_time():
    """Tool to get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Tool to perform basic calculations."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def create_agent():
    """
    Create and return a Langchain agent with Anthropic's Claude model.
    
    Returns:
        AgentExecutor: Configured agent executor ready to process requests.
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")
    
    # Initialize the Claude model
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        anthropic_api_key=api_key,
        temperature=0.7
    )
    
    # Define tools for the agent
    tools = [
        Tool(
            name="GetCurrentTime",
            func=get_current_time,
            description="Useful for getting the current date and time. No input needed."
        ),
        Tool(
            name="Calculator",
            func=calculate,
            description="Useful for performing mathematical calculations. Input should be a valid Python expression."
        )
    ]
    
    # Get the ReAct prompt from Langchain hub
    prompt = hub.pull("hwchase17/react")
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


def run_agent(query: str):
    """
    Run the agent with a given query.
    
    Args:
        query (str): The question or task for the agent to process.
        
    Returns:
        dict: The agent's response.
    """
    agent_executor = create_agent()
    response = agent_executor.invoke({"input": query})
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
        print(f"\nFinal Answer: {result['output']}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have:")
        print("1. Created a .env file based on .env.example")
        print("2. Added your Anthropic API key to the .env file")
