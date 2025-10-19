# LangChain Chat Application with Claude

A comprehensive Python project demonstrating sophisticated LLM chat applications using LangChain and Anthropic's Claude model. This project showcases advanced techniques including prompt engineering, conversational memory, streaming responses, and practical implementations for various use cases.

## Features

- 🤖 **Advanced Prompt Engineering**: System and human message templates for different domains
- 💬 **Conversational Memory**: Multi-turn dialogues with context awareness
- ⚡ **Streaming Responses**: Real-time token-by-token output
- 🎯 **Multiple Use Cases**: Q&A, creative writing, code generation, documentation, and code review
- 🧠 **Specialized Assistants**: Research assistant and Socratic tutor implementations
- 🛡️ **Error Handling**: Comprehensive retry logic and input validation
- 📊 **Conversation Analysis**: Track and analyze dialogue patterns
- 🔐 **Secure Configuration**: Environment-based API key management

## Prerequisites

- Python 3.11 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))
- (Optional) LangSmith API key for tracking ([Sign up here](https://smith.langchain.com/))

## Installation

1. **Clone or download this repository**

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

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# Windows: notepad .env
# macOS/Linux: nano .env
```

Add your API keys to the `.env` file:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
```

## Usage

### Starting the Jupyter Notebook

```bash
jupyter notebook
```

Open `langchain_chat_application.ipynb` and run the cells sequentially.

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
Lang_Chain_practice_Project/
├── langchain_chat_application.ipynb  # Main Jupyter notebook
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
├── .env                              # Your API keys (do not commit!)
├── .gitignore                        # Git ignore rules
├── CLAUDE.md                         # Developer documentation
└── README.md                         # This file
```

## Key Components

### ChatAnthropic Model

The primary interface to Claude models. Configuration options:
- `model`: Choose from Claude models (default: `claude-3-5-sonnet-20241022`)
- `temperature`: Control randomness (0.0-1.0)
- `max_tokens`: Maximum response length

### Prompt Templates

Three main template patterns:
1. **Simple templates** for direct interactions
2. **Memory-enabled templates** with `MessagesPlaceholder`
3. **Domain-specific templates** with customized system messages

### Memory Management

- **InMemoryChatMessageHistory**: Stores conversation history per session
- **RunnableWithMessageHistory**: Maintains context across turns
- **Session IDs**: Isolate different conversation threads

### Chain Composition

LangChain's pipe (`|`) operator for building pipelines:
```python
chain = prompt | model | output_parser
```

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
- LangChain (0.3.13)
- LangChain Core (0.3.28)
- LangChain Anthropic (0.3.3)
- Anthropic SDK (0.39.0)
- Jupyter Notebook

---

**Note**: This is a practice project designed to demonstrate LangChain capabilities with Claude. For production use, consider implementing persistent storage, proper authentication, rate limiting, and monitoring.
