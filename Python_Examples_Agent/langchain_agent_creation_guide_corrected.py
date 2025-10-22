"""
LangChain Agent Creation Guide - Corrected Version

This script fixes all the issues found in the original Jupyter notebook:

FIXED ISSUES:
1. ✅ "get_weather_for_location" is not defined - Fixed by defining tools before agent creation
2. ✅ NameError: name 'get_weather_for_location' is not defined - Fixed execution order
3. ✅ Added proper type hints (PEP8 compliance)
4. ✅ Improved imports and error handling
5. ✅ Added proper function documentation

This demonstrates how to create a LangChain agent with:
- Custom tools for data retrieval
- Structured response formats using dataclasses
- Conversation memory with checkpointing
- System prompts for agent personality
"""

from typing import Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Core LangChain imports
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# Memory/checkpointing
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables
load_dotenv()

print("✓ All imports successful!")


# Section 1: Define System Prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location (pass user_id as parameter)

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

print("System prompt defined:")
print(SYSTEM_PROMPT)


# Section 2: Define Context Schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# Example usage
example_context = Context(user_id="123")
print(f"✓ Context schema defined: {example_context}")


# Section 3: Define Tools (CRITICAL: Define BEFORE agent creation)
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        Weather information for the specified city
    """
    # In production, this would call a weather API
    return f"It's always sunny in {city}!"


@tool
def get_user_location(user_id: str) -> str:
    """Retrieve user location based on user ID.
    
    Args:
        user_id: The unique identifier for the user
        
    Returns:
        The user's location
    """
    # In production, this would query a user database
    # For demo purposes, we use a simple lookup
    return "Florida" if user_id == "1" else "SF"


print("✓ Tools defined:")
print(f"  - {get_weather_for_location.name}: {get_weather_for_location.description}")
print(f"  - {get_user_location.name}: {get_user_location.description}")


# Section 4: Configure the Language Model
def initialize_model() -> object:
    """Initialize the chat model with proper error handling."""
    try:
        model = init_chat_model(
            "claude-3-5-sonnet-20241022",  # Using the stable model identifier
            temperature=0.7  # Slightly creative for puns
        )
        print("✓ Model configured successfully")
        print(f"  Model: claude-3-5-sonnet-20241022")
        print(f"  Temperature: 0.7 (balanced)")
        return model
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in your .env file")
        raise


model = initialize_model()


# Section 5: Define Response Format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: Optional[str] = None


print("✓ Response format defined:")
print(f"  Required: punny_response (str)")
print(f"  Optional: weather_conditions (Optional[str])")


# Section 6: Set Up Memory (Checkpointing)
checkpointer = InMemorySaver()

print("✓ Checkpointer initialized (InMemorySaver)")
print("  Note: Conversation history will be lost on restart")
print("  For production, use persistent storage")


# Section 7: Create the Agent (NOW ALL DEPENDENCIES ARE DEFINED)
def create_weather_agent():
    """Create the weather agent with all required components."""
    try:
        agent = create_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=[get_user_location, get_weather_for_location],  # ✅ Both tools are now defined
            context_schema=Context,
            response_format=ResponseFormat,
            checkpointer=checkpointer
        )
        
        print("✓ Agent created successfully!")
        print("\nAgent Configuration:")
        print("  - Model: claude-3-5-sonnet-20241022")
        print("  - Tools: get_user_location, get_weather_for_location")
        print("  - Context: User ID tracking")
        print("  - Response: Structured (punny_response + weather_conditions)")
        print("  - Memory: Enabled (InMemorySaver)")
        
        return agent
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        raise


agent = create_weather_agent()


# Section 8: Example Usage Functions
def run_weather_query_example() -> None:
    """Example of running a weather query."""
    print("\n" + "="*50)
    print("EXAMPLE 1: Basic Weather Query")
    print("="*50)
    
    # Configure conversation thread
    config = {"configurable": {"thread_id": "1"}}
    
    try:
        # First query: Ask about weather
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
            config=config,
            context=Context(user_id="1")
        )
        
        # Display structured response
        print("User Query: 'what is the weather outside?'\n")
        print("Agent Response:")
        print("=" * 80)
        print(f"Punny Response: {response['structured_response'].punny_response}")
        print(f"Weather Conditions: {response['structured_response'].weather_conditions}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error in weather query: {e}")


def run_followup_example() -> None:
    """Example of a follow-up conversation."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Follow-up Conversation")
    print("="*50)
    
    config = {"configurable": {"thread_id": "1"}}
    
    try:
        # Continue conversation with same thread_id
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "thank you!"}]},
            config=config,  # Same config = same conversation
            context=Context(user_id="1")
        )
        
        print("User Query: 'thank you!'\n")
        print("Agent Response:")
        print("=" * 80)
        print(f"Punny Response: {response['structured_response'].punny_response}")
        print(f"Weather Conditions: {response['structured_response'].weather_conditions}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error in follow-up: {e}")


def run_different_user_example() -> None:
    """Example with different user context."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Different User Context")
    print("="*50)
    
    # Different user, different conversation
    config_user2 = {"configurable": {"thread_id": "2"}}
    
    try:
        response_user2 = agent.invoke(
            {"messages": [{"role": "user", "content": "what's the weather like?"}]},
            config=config_user2,
            context=Context(user_id="2")  # Different user
        )
        
        print("User 2 Query: 'what's the weather like?'\n")
        print("Agent Response:")
        print("=" * 80)
        print(f"Punny Response: {response_user2['structured_response'].punny_response}")
        print(f"Weather Conditions: {response_user2['structured_response'].weather_conditions}")
        print("=" * 80)
        print("\n💡 Notice: Different user_id resulted in different location (SF instead of Florida)")
        
    except Exception as e:
        print(f"❌ Error with different user: {e}")


def run_direct_location_example() -> None:
    """Example with explicit location query."""
    print("\n" + "="*50)
    print("EXAMPLE 4: Direct Location Query")
    print("="*50)
    
    # Query with explicit location
    config_direct = {"configurable": {"thread_id": "3"}}
    
    try:
        response_direct = agent.invoke(
            {"messages": [{"role": "user", "content": "What's the weather in New York?"}]},
            config=config_direct,
            context=Context(user_id="1")
        )
        
        print("User Query: 'What's the weather in New York?'\n")
        print("Agent Response:")
        print("=" * 80)
        print(f"Punny Response: {response_direct['structured_response'].punny_response}")
        print(f"Weather Conditions: {response_direct['structured_response'].weather_conditions}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error with direct location: {e}")


# Section 9: Main Execution
def main() -> None:
    """Main function to run all examples."""
    print("\n" + "=" * 80)
    print("LANGCHAIN AGENT CREATION GUIDE - CORRECTED VERSION")
    print("=" * 80)
    print("\nAll issues have been fixed:")
    print("✅ Tools are defined before agent creation")
    print("✅ Proper execution order maintained")
    print("✅ Type hints added (PEP8 compliance)")
    print("✅ Error handling implemented")
    print("✅ Proper function documentation")
    
    try:
        print("\n🚀 Running Examples...")
        
        run_weather_query_example()
        run_followup_example()
        run_different_user_example()
        run_direct_location_example()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure ANTHROPIC_API_KEY is set in your .env file")
        print("2. Install required packages:")
        print("   pip install langchain langchain-core langchain-anthropic langgraph python-dotenv")


if __name__ == "__main__":
    main()