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
2. **Create and activate a virtual environment**:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:

Create a `.env` file in the project root and add your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
```

## Usage

### Starting the Jupyter Notebook

```bash
jupyter notebook
```

Navigate to the `Jupyter_Lang_Chain_Notebook/` folder and open `langchain_chat_application.ipynb`, then run the cells sequentially.

### Running the Python Agent

```bash
# Run the main agent with a test query
python Python_Examples_Agent/agent.py

# Run all examples
python Python_Examples_Agent/example.py

# Run the CLI in interactive mode
python Python_Examples_Agent/cli.py

# Run a one-shot query with the CLI
python Python_Examples_Agent/cli.py "What is the current time?"

# Run the test suite
python Python_Examples_Agent/test_agent.py
```

### Notebook Sections

The notebook is organized into 12 comprehensive sections:

1. **Setup and Configuration** - Import libraries and load API keys
2. **Initialize Chat Model** - Configure Claude with custom parameters
3. **Basic Chat Interaction** - Simple question-answering
4. **Advanced Prompt Engineering** - Domain-specific templates (Q&A, creative writing, code)
5. **Conversational Chain with Memory** - Multi-turn dialogues with context
6. **Streaming Responses** - Real-time token streaming
7. **Research Assistant** - Context-aware research with follow-up capabilities
8. **Error Handling** - Robust error handling with retry logic
9. **Practical Use Cases** - Documentation generation, code review, Socratic tutoring
10. **Conversation Analysis** - Track and analyze conversation metrics
11. **Interactive Chat Interface** - Build interactive chat sessions
12. **Summary and Utilities** - Export conversations and session management

## Project Structure

```
Lang_chain_Create_Agents/
├── Jupyter_Lang_Chain_Notebook/           # Jupyter notebooks for tutorials
│   ├── langchain_chat_application.ipynb              # Main tutorial (12 sections)
│   └── langchain_chat_models_prompt_templates.ipynb  # Learning-focused notebook
├── Python_Examples_Agent/                 # Python agent implementations
│   ├── agent.py                                      # ReAct agent with tools
│   ├── cli.py                                        # Command-line interface
│   ├── example.py                                    # Usage examples
│   ├── test_agent.py                                 # Agent test suite
│   └── test_cli.py                                   # CLI test suite
├── .env                                   # Your API keys (do not commit!)
├── .env.example                           # Environment variable template
├── .gitignore                             # Git ignore rules
├── CLAUDE.md                              # Developer documentation
├── IMPLEMENTATION_NOTES.md                # Agent implementation notes
├── README.md                              # This file
└── requirements.txt                       # Python dependencies
```

## Key Components

### 1. Jupyter Notebooks (Tutorial & Learning)

#### ChatAnthropic Model
The primary interface to Claude models. Configuration options:
- `model`: Choose from Claude models (default: `claude-3-5-sonnet-20241022`)
- `temperature`: Control randomness (0.0-1.0)
- `max_tokens`: Maximum response length

#### Prompt Templates
Three main template patterns:
1. **Simple templates** for direct interactions
2. **Memory-enabled templates** with `MessagesPlaceholder`
3. **Domain-specific templates** with customized system messages

#### Memory Management
- **InMemoryChatMessageHistory**: Stores conversation history per session
- **RunnableWithMessageHistory**: Maintains context across turns
- **Session IDs**: Isolate different conversation threads

#### Chain Composition
LangChain's pipe (`|`) operator for building pipelines:
```python
chain = prompt | model | output_parser
```

### 2. Python Agent (ReAct Framework)

#### Agent Architecture
- **Framework**: ReAct (Reasoning and Acting)
- **Model**: Claude 3.5 Sonnet
- **Tools**:
  - `get_current_time()` - Returns current date/time
  - `calculate()` - Performs mathematical calculations

#### Agent Features
- Autonomous tool selection based on query
- Multi-step reasoning with tool chaining
- Error handling and input validation
- Extensible tool system

#### Example Agent Usage
```python
from Python_Examples_Agent.agent import run_agent

result = run_agent("What is the current time and what is 25 * 4?")
print(result['messages'][-1].content)
# Output: The current time is 2025-10-20 00:26:32, and 25 * 4 = 100.
```

### 3. Copilot CLI (Command-Line Interface)

The Copilot CLI provides a user-friendly command-line interface for interacting with the LangChain agent.

#### CLI Features
- **Interactive Mode**: Continuous conversation with the agent
- **One-Shot Mode**: Execute single queries and exit
- **Conversation History**: Track and review past interactions
- **Custom Configuration**: Adjust model, temperature, and verbosity
- **Built-in Commands**: Help, history, clear, and exit commands

#### CLI Usage

**Interactive Mode** (default):
```bash
# Start interactive session
python Python_Examples_Agent/cli.py

# Interactive mode with custom model
python Python_Examples_Agent/cli.py --interactive --model claude-3-opus-20240229
```

**One-Shot Mode**:
```bash
# Single query
python Python_Examples_Agent/cli.py "What is the current time?"

# With verbose output
python Python_Examples_Agent/cli.py --verbose "Calculate 123 * 456"

# With custom temperature
python Python_Examples_Agent/cli.py --temperature 0.3 "Explain quantum computing"
```

#### CLI Options
- `-i, --interactive`: Force interactive mode
- `-m, --model MODEL`: Specify Claude model (default: claude-3-5-sonnet-20241022)
- `-t, --temperature`: Set temperature 0.0-1.0 (default: 0.7)
- `-v, --verbose`: Enable verbose output
- `--version`: Show version information
- `-h, --help`: Display help message

#### Interactive Commands
- `help`: Show available commands and examples
- `history`: Display conversation history
- `clear`: Clear the screen
- `exit`, `quit`, `q`: Exit the session

## Example Use Cases

### 1. Question Answering
```python
qa_chain = qa_prompt | chat_model | StrOutputParser()
response = qa_chain.invoke({
    "domain": "machine learning",
    "question": "Explain supervised learning"
})
```

### 2. Creative Writing
```python
creative_chain = creative_prompt | chat_model | StrOutputParser()
story = creative_chain.invoke({
    "genre": "science fiction",
    "request": "Write an opening paragraph"
})
```

### 3. Code Generation
```python
code_chain = code_prompt | chat_model | StrOutputParser()
code = code_chain.invoke({
    "language": "Python",
    "task": "Create a binary search function"
})
```

### 4. Research Assistant
```python
assistant = ResearchAssistant("quantum_computing")
response = assistant.ask("What is quantum entanglement?")
```

### 5. Socratic Tutor
```python
tutor = SocraticTutor("Python programming")
guidance = tutor.teach("I want to learn about decorators")
```

## Configuration Options

### Model Selection
- `claude-3-5-sonnet-20241022` (recommended, balanced)
- `claude-3-opus-20240229` (most capable)
- `claude-3-haiku-20240307` (fastest)

### Temperature Settings
- **0.0-0.3**: Factual, consistent responses (research, code review)
- **0.4-0.7**: Balanced creativity and accuracy (default)
- **0.8-1.0**: Highly creative responses (creative writing, brainstorming)

### Memory Configuration
- In-memory storage (default, suitable for development)
- For production: Implement persistent storage (database, Redis)

## Best Practices

1. **API Key Security**
   - Never commit `.env` file to version control
   - Use environment variables for all secrets
   - Rotate API keys regularly

2. **Error Handling**
   - Always wrap API calls in try-except blocks
   - Implement retry logic for rate limits
   - Validate inputs before sending to API

3. **Performance**
   - Use streaming for long responses
   - Adjust `max_tokens` based on needs
   - Consider caching for repeated queries

4. **Conversation Management**
   - Use meaningful session IDs
   - Clear old sessions periodically
   - Export important conversations

5. **Prompt Engineering**
   - Be specific in system messages
   - Provide examples when possible
   - Iterate and test different phrasings

## Troubleshooting

### API Key Issues
```
ValueError: ANTHROPIC_API_KEY not found
```
**Solution**: Ensure `.env` file exists and contains valid API key

### Import Errors
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution**: Activate virtual environment and run `pip install -r requirements.txt`

### Rate Limiting
```
Error: Rate limit exceeded
```
**Solution**: The `RobustChatInterface` includes retry logic with exponential backoff

### Memory Issues
```
Session not found
```
**Solution**: Ensure you're using the same session_id across calls

## Contributing

Feel free to extend this project with:
- Additional use cases and examples
- Persistent storage implementations
- Web interface (Streamlit, Flask, FastAPI)
- RAG (Retrieval-Augmented Generation) integration
- Custom tools and function calling
- Multi-agent systems

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [LangSmith Tracing](https://docs.smith.langchain.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## License

This project is provided as-is for educational and practice purposes.

## Acknowledgments

Built with:
- LangChain (1.0.0)
- LangChain Core (1.0.0)
- LangChain Anthropic (1.0.0)
- Anthropic SDK (0.71.0)
- Jupyter Notebook

---

**Note**: This is a practice project designed to demonstrate LangChain capabilities with Claude. For production use, consider implementing persistent storage, proper authentication, rate limiting, and monitoring.
