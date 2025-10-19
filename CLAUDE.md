# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A comprehensive LangChain practice project demonstrating sophisticated LLM chat applications using Anthropic's Claude model. The project showcases advanced prompt engineering, conversational chains with memory, streaming responses, and multiple practical use cases including Q&A, creative writing, code generation, and Socratic tutoring.

## Setup and Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create .env file with: ANTHROPIC_API_KEY=your_key_here
# Optionally add: LANGSMITH_API_KEY=your_key_here
```

## Development Commands

```bash
# Start Jupyter notebook server
jupyter notebook

# Install kernel for virtual environment (one-time setup)
python -m ipykernel install --user --name=langchain-env

# List active Jupyter kernels
jupyter kernelspec list

# Update dependencies after requirements.txt changes
pip install -r requirements.txt --upgrade
```

## Notebook Structure

### langchain_chat_application.ipynb
The main tutorial notebook with 12 sections covering end-to-end LangChain implementation:
- **Sections 1-2**: Environment setup, API configuration, ChatAnthropic initialization
- **Sections 3-4**: Basic invoke() usage, prompt template patterns (Q&A, creative, code generation)
- **Sections 5-6**: Memory management with `RunnableWithMessageHistory`, streaming with `model.stream()`
- **Sections 7-8**: Context-aware `ResearchAssistant` class, `RobustChatInterface` with retry logic
- **Section 9**: Production use cases: documentation generator, code reviewer, `SocraticTutor`
- **Sections 10-12**: Conversation analytics, interactive REPL, utility functions

### langchain_chat_models_prompt_templates.ipynb
Focused learning notebook with detailed instructional markdown before each cell explaining:
- Core concepts and technical details
- Expected outputs and use cases
- Best practices and when to apply patterns
- Step-by-step task breakdowns

## Architecture

### Chain Composition Pattern
All chains follow the pipe operator pattern:
```python
chain = prompt | chat_model | output_parser
response = chain.invoke({"variable": "value"})
```

The `|` operator creates a data pipeline where output of one component becomes input to the next.

### Memory Architecture
```
store = {}  # Global dict holding all sessions
  └─ session_id_1 → InMemoryChatMessageHistory()
  └─ session_id_2 → InMemoryChatMessageHistory()

RunnableWithMessageHistory(
    chain,
    get_session_history,  # Retrieves history by session_id
    input_messages_key="input",
    history_messages_key="history"  # Where MessagesPlaceholder injects messages
)
```

**Key insight**: `MessagesPlaceholder(variable_name="history")` in the prompt must match `history_messages_key="history"` in the wrapper.

### Temperature Guidelines by Use Case
- **0.0-0.3**: Code generation, factual Q&A, research (ResearchAssistant uses 0.3)
- **0.4-0.7**: General chat, balanced responses (default 0.7)
- **0.8-1.0**: Creative writing, brainstorming, varied outputs

### Key Classes

**ResearchAssistant** (Section 7):
- Maintains context across multiple research queries
- Lower temperature (0.3) for factual accuracy
- Uses session-based history with `RunnableWithMessageHistory`
- Has `ask()`, `get_history()`, and `clear_history()` methods

**SocraticTutor** (Section 9):
- Guides learning through questions rather than direct answers
- Subject-specific system prompt in initialization
- Each subject gets its own session_id

**RobustChatInterface** (Section 8):
- Wraps API calls with try-except and retry logic
- Exponential backoff: `2 ** retry_count` seconds
- Returns dict with `{success, response, error, retry_count}`
- Handles rate limits, validation errors, and API errors separately

## Model Configuration

**Default Model**: `claude-3-5-sonnet-20241022`
- Balanced performance and cost
- Suitable for most use cases

**Available Models**:
- `claude-3-5-sonnet-20241022`: Recommended for general use
- `claude-3-opus-20240229`: Highest capability for complex tasks
- `claude-3-haiku-20240307`: Fastest and most cost-effective

**API Keys**:
- `ANTHROPIC_API_KEY`: Required - get from https://console.anthropic.com/
- `LANGSMITH_API_KEY`: Optional - enables tracing at https://smith.langchain.com/

**Memory**: In-memory only (data lost on restart). For production, implement persistent storage (PostgreSQL, MongoDB, Redis).

## Common Patterns

### Creating a Conversational Chain with Memory
```python
# 1. Define prompt with MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant..."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 2. Create base chain
chain = prompt | model

# 3. Wrap with message history
chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 4. Invoke with session_id
response = chat_with_memory.invoke(
    {"input": "Hello"},
    config={"configurable": {"session_id": "user_123"}}
)
```

### Streaming Long Responses
```python
# Instead of invoke() which waits for complete response:
for chunk in chat_model.stream([HumanMessage(content=message)]):
    print(chunk.content, end="", flush=True)
```

### Error Handling Pattern
```python
try:
    # Validate input
    if not message or len(message) > 100000:
        raise ValueError("Invalid message")

    response = chat_model.invoke([HumanMessage(content=message)])
    return response.content

except Exception as e:
    # Check for rate limits and retry
    if "rate limit" in str(e).lower() and retry_count < max_retries:
        time.sleep(2 ** retry_count)  # Exponential backoff
        return retry_request()
    else:
        return f"Error: {str(e)}"
```
