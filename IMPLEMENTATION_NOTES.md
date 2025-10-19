# Implementation Notes

## Overview
This repository contains a complete implementation of a Langchain agent using Anthropic's Claude API. The agent uses the ReAct (Reasoning and Acting) framework for decision-making.

## Key Components

### 1. Agent Implementation (`agent.py`)
- Uses Langchain 1.0 with `langgraph.prebuilt.create_react_agent`
- Integrates Anthropic's Claude-3 Sonnet model
- Includes two built-in tools:
  - **get_current_time**: Returns current date and time
  - **calculate**: Performs safe mathematical calculations using AST parsing (not eval)

### 2. Security Features
- **Safe Expression Evaluation**: The calculator uses `ast` module to safely parse and evaluate mathematical expressions without allowing arbitrary code execution
- **API Key Protection**: API keys are stored in `.env` file which is excluded from git
- **Error Handling**: Comprehensive error handling for missing API keys and invalid inputs

### 3. Testing (`test_agent.py`)
- Unit tests for all major components
- Tests for security (API key validation)
- Portable test paths (no hardcoded absolute paths)
- All tests passing

### 4. Dependencies
- `langchain>=0.1.0`: Core Langchain library
- `langchain-anthropic>=0.1.0`: Anthropic integration
- `python-dotenv>=1.0.0`: Environment variable management

## Architecture Decisions

### Why Langgraph's create_react_agent?
In Langchain 1.0, the agent architecture changed significantly. The old `AgentExecutor` and `create_react_agent` from `langchain.agents` were moved/replaced. We use `langgraph.prebuilt.create_react_agent` which is the recommended approach in Langchain 1.0.

### Why @tool Decorator?
The `@tool` decorator from `langchain_core.tools` is the modern way to define tools in Langchain 1.0. It provides:
- Automatic schema generation
- Better integration with the agent
- Type safety

### Why AST Instead of eval()?
The calculator originally used `eval()` which is a serious security vulnerability. We replaced it with AST-based parsing that:
- Only allows basic mathematical operations (+, -, *, /, **, %)
- Blocks function calls and imports
- Prevents arbitrary code execution

## Usage Examples

### Basic Usage
```python
from agent import run_agent

result = run_agent("What is 2 + 2?")
print(result['messages'][-1].content)
```

### Adding Custom Tools
```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Your implementation
    return results

# Add to create_agent() function
tools = [get_current_time, calculate, search_web]
```

## Future Enhancements
- Add more sophisticated tools (web search, database queries, etc.)
- Implement conversation history/memory
- Add streaming responses
- Support for multiple LLM backends
- Add more comprehensive error handling and logging
