# Lang_chain_Create_Agents

This repository provides an implementation of an AI agent using Langchain and Anthropic's Claude API.

## Overview

The agent is built using the ReAct (Reasoning and Acting) framework and comes with built-in tools for:
- Getting the current date and time
- Performing mathematical calculations

## Prerequisites

- Python 3.8 or higher
- An Anthropic API key ([Get one here](https://console.anthropic.com/))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/log2jiten24/Lang_chain_Create_Agents.git
cd Lang_chain_Create_Agents
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Copy `.env.example` to `.env`
   - Add your Anthropic API key to the `.env` file

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### Basic Usage

Run the agent with the default example:

```bash
python agent.py
```

### Using the Agent in Your Code

```python
from agent import run_agent

# Ask a question
result = run_agent("What is the current time?")
print(result['output'])

# Perform calculations
result = run_agent("Calculate 25 * 4")
print(result['output'])
```

### Running Examples

Run the provided examples:

```bash
python example.py
```

## Project Structure

```
.
├── agent.py          # Main agent implementation
├── example.py        # Example usage scripts
├── requirements.txt  # Python dependencies
├── .env.example     # Environment variable template
├── .gitignore       # Git ignore rules
└── README.md        # This file
```

## Features

- **ReAct Agent**: Uses the ReAct (Reasoning and Acting) framework for decision-making
- **Claude Integration**: Powered by Anthropic's Claude-3 Sonnet model
- **Extensible Tools**: Easy to add custom tools for your specific needs
- **Error Handling**: Robust error handling and parsing

## Adding Custom Tools

You can easily extend the agent with custom tools using the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def my_custom_function(input_str: str) -> str:
    """Description of what your tool does."""
    # Your custom logic here
    return result

# Add your tool to the tools list in create_agent()
# tools = [get_current_time, calculate, my_custom_function]
```

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
