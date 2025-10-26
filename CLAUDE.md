# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A comprehensive LangChain practice project demonstrating sophisticated LLM chat applications using Anthropic's Claude model. The project showcases advanced prompt engineering, conversational chains with memory, streaming responses, and multiple practical use cases including Q&A, creative writing, code generation, and Socratic tutoring.

## Setup and Installation

```bash
# Create and activate virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create .env file with: ANTHROPIC_API_KEY=your_key_here
# Optionally add: LANGSMITH_API_KEY=your_key_here
```

## Development Commands

```bash
# Jupyter Notebook
jupyter notebook                                     # Start notebook server
python -m ipykernel install --user --name=langchain-env  # One-time kernel setup
jupyter kernelspec list                              # List available kernels

# Python Agent (ReAct Framework)
python Python_Examples_Agent/agent.py                # Run standalone agent example
python Python_Examples_Agent/example.py              # Run all usage examples
python Python_Examples_Agent/test_agent.py           # Run test suite

# Python RAG Agent (Document Q&A)
python Python_RAG_Agent/Example_Usage.py             # Create/load vector store and test queries
# On first run: Creates vector store from PDFs in sample_data/
# Subsequent runs: Offers to load existing or recreate

# Dependencies
pip install -r requirements.txt --upgrade            # Update dependencies
```

## Project Structure

```
Lang_chain_Create_Agents/
├── Jupyter_Lang_Chain_Notebook/
│   ├── langchain_chat_application.ipynb          # Main tutorial (12 sections)
│   ├── langchain_chat_models_prompt_templates.ipynb
│   ├── RAG_Agent_PDf_Documents.ipynb             # RAG implementation notebooks
│   └── RAG_Data_Ingestion_Vector_DB_Pipeline.ipynb
├── Python_Examples_Agent/                        # ReAct agent with tools
│   ├── agent.py                                  # Main agent implementation
│   ├── example.py                                # Usage examples
│   └── test_agent.py                             # Test suite
├── Python_RAG_Agent/                             # RAG system for document Q&A
│   ├── data_loader.py                            # PDF/text document loaders
│   ├── Embeddings.py                             # HuggingFace embeddings manager
│   ├── vector_store.py                           # FAISS vector store operations
│   ├── Example_Usage.py                          # Create/load vector store demo
│   └── search.py                                 # (placeholder)
├── sample_data/                                  # Sample documents
│   ├── pdf_files/                                # PDF documents (e.g., Nike 10-K)
│   └── text_files/                               # Text documents
├── data_storage/                                 # Persistent vector stores
│   └── vector_store/                             # FAISS index and metadata
├── .env (not committed)
├── .env.example
├── .gitignore
├── CLAUDE.md
├── IMPLEMENTATION_NOTES.md
├── README.md
└── requirements.txt
```

## Jupyter Notebooks (`Jupyter_Lang_Chain_Notebook/`)

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

## Python Agent Examples (`Python_Examples_Agent/`)

### agent.py
Main agent implementation using LangChain's ReAct framework:
- **Tools**: `get_current_time()` and `calculate()` for math operations
- **Model**: Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`)
- **Pattern**: Uses `create_react_agent` from `langgraph.prebuilt`
- **Security**: Calculator uses AST parsing (not eval) to safely evaluate math expressions
- **Usage**: Can be run standalone or imported as module

### example.py
Demonstrates practical agent usage with 4 examples:
1. Current time query
2. Mathematical calculation (123 * 456)
3. Square root calculation
4. Combined time and division query

### test_agent.py
Test suite for validating agent functionality:
- Tool availability tests
- Agent creation tests
- Query execution tests
- Error handling validation
- Run with: `python Python_Examples_Agent/test_agent.py`

## Python RAG Agent (`Python_RAG_Agent/`)

RAG (Retrieval-Augmented Generation) system for question-answering over documents. Loads PDFs/text files, creates embeddings, stores in FAISS vector database, and enables semantic search.

### data_loader.py
Document loading utilities supporting multiple formats:
- **`load_all_documents(path)`**: Load both PDF and text files from directory
- **`load_pdf_documents(path)`**: Load only PDF files
- **`load_text_documents(path)`**: Load only text files
- Uses `PyPDFLoader` for PDFs and `TextLoader` for text files
- Returns list of `Document` objects with content and metadata (source, page)

### Embeddings.py
Manages HuggingFace sentence-transformer embeddings:
- **`EmbeddingManager`** class: Initialize and manage embedding models
- **Default model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, fast)
- **Alternative models**:
  - `all-mpnet-base-v2`: Higher quality, 768 dimensions (slower)
  - `BAAI/bge-small-en-v1.5`: Good balance, 384 dimensions
  - `BAAI/bge-base-en-v1.5`: Higher quality, 768 dimensions
- Methods: `embed_documents()`, `embed_query()`, `get_embeddings()`

### vector_store.py
FAISS vector store operations with persistence:
- **`VectorStoreManager`** class: Complete vector store lifecycle management
- **Text Chunking**: `RecursiveCharacterTextSplitter` with configurable size/overlap
  - Default: 1000 chars per chunk, 200 char overlap
  - Splits on: paragraph → sentence → word boundaries
- **Key Methods**:
  - `create_vector_store(docs)`: Create new FAISS index from documents
  - `save(folder_path)`: Persist vector store to disk (index.faiss + index.pkl)
  - `load(folder_path)`: Load existing vector store from disk
  - `similarity_search(query, k)`: Find top-k similar documents
  - `similarity_search_with_score(query, k)`: Return (document, score) tuples
  - `get_retriever(search_kwargs)`: Get retriever for LangChain chains
- **Storage**: Saves to `data_storage/vector_store/` by default

### Example_Usage.py
Complete RAG pipeline demonstration:
- **First run**: Creates vector store from `sample_data/pdf_files/`
  1. Load documents (e.g., 107 pages from Nike 10-K PDF)
  2. Initialize embeddings model
  3. Split into chunks (e.g., 516 chunks from 107 pages)
  4. Create FAISS index and save to disk
- **Subsequent runs**: Offers to load existing or recreate
- **Test Query**: Performs similarity search on sample question
- **Interactive**: Prompts user for create/load decision

### RAG Pipeline Architecture
```
Documents (PDF/Text)
  ↓ [data_loader.py]
Document Objects (content + metadata)
  ↓ [vector_store.py - text splitter]
Text Chunks (1000 chars, 200 overlap)
  ↓ [Embeddings.py - sentence transformers]
Embedding Vectors (384/768 dimensions)
  ↓ [vector_store.py - FAISS]
Vector Store (saved to disk)
  ↓ [similarity search]
Query → Top-K Relevant Documents
```

### RAG Usage Pattern
```python
from data_loader import load_all_documents
from Embeddings import get_default_embeddings
from vector_store import VectorStoreManager

# Create vector store (first time)
docs = load_all_documents("sample_data/pdf_files")
embeddings = get_default_embeddings()
manager = VectorStoreManager(embeddings)
manager.create_vector_store(docs)
manager.save("data_storage/vector_store")

# Load existing vector store
from vector_store import load_vector_store
manager = load_vector_store("data_storage/vector_store", embeddings)

# Query documents
results = manager.similarity_search("What is the revenue?", k=3)
# Or with scores (lower = more similar)
results = manager.similarity_search_with_score("What is the revenue?", k=3)

# Use with LangChain chains
retriever = manager.get_retriever(search_kwargs={"k": 4})
```

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

## Dependencies

**Core LangChain packages** (from requirements.txt):
- `langchain>=0.3.13` - Core LangChain library
- `langchain-core>=0.3.30` - LangChain core abstractions
- `langchain-anthropic>=0.3.3` - Anthropic integration
- `langchain-community>=0.1.0` - Community integrations (loaders, vector stores)
- `langchain-huggingface>=0.1.0` - HuggingFace embeddings integration
- `anthropic>=0.41.0` - Anthropic SDK
- `langgraph` - Required for `create_react_agent` (implicitly installed)

**RAG & Vector Store packages**:
- `faiss-cpu>=1.8.0` - Facebook's similarity search library for vector stores
- `sentence-transformers>=2.5.1` - HuggingFace sentence embeddings
- `pypdf` - PDF document loading
- `pymupdf` - Alternative PDF parsing library
- `tiktoken>=0.7.0` - Token counting for OpenAI models
- `chromadb>=0.5.0` - Alternative vector database (optional)

**Development packages**:
- `python-dotenv==1.0.1` - Environment variable management
- `jupyter==1.1.1` - Notebook interface
- `ipykernel==6.29.5` - Jupyter kernel
- `requests>=2.32.5` - HTTP library

**Notes**:
- `langgraph.prebuilt.create_react_agent` is the recommended agent pattern in LangChain 1.0+. The old `AgentExecutor` pattern has been replaced.
- FAISS requires `allow_dangerous_deserialization=True` when loading from disk (security consideration for trusted sources only)
- First-time embedding model downloads are ~80MB (all-MiniLM-L6-v2) or ~420MB (all-mpnet-base-v2)

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

### Adding Custom Tools to Agent
```python
from langchain_core.tools import tool

@tool
def custom_tool(input_param: str) -> str:
    """Clear description of what the tool does. The agent uses this description."""
    # Implementation
    return result

# In create_agent() function in agent.py:
tools = [get_current_time, calculate, custom_tool]
agent_executor = create_react_agent(llm, tools)
```

**Important**: Tool docstrings are critical - the agent reads them to decide when to use each tool.

### Building a RAG Q&A System
```python
from data_loader import load_all_documents
from Embeddings import EmbeddingManager
from vector_store import VectorStoreManager
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load and index documents (one-time setup)
docs = load_all_documents("sample_data/pdf_files")
embedding_manager = EmbeddingManager()
vector_manager = VectorStoreManager(embedding_manager.get_embeddings())
vector_manager.create_vector_store(docs)
vector_manager.save("data_storage/vector_store")

# 2. Load existing vector store (subsequent runs)
from vector_store import load_vector_store
embeddings = EmbeddingManager().get_embeddings()
vector_manager = load_vector_store("data_storage/vector_store", embeddings)

# 3. Create retriever
retriever = vector_manager.get_retriever(search_kwargs={"k": 3})

# 4. Build Q&A chain with custom prompt
prompt_template = """Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" puts all docs in context
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 5. Query the system
response = qa_chain.invoke({"query": "What is the company's revenue?"})
print(response["result"])
print(f"Sources: {[doc.metadata for doc in response['source_documents']]}")
```

### RAG Import Patterns
When working in the `Python_RAG_Agent/` directory:
```python
# Use relative imports when running scripts from within the directory
from data_loader import load_all_documents
from Embeddings import get_default_embeddings
from vector_store import VectorStoreManager

# Use absolute imports when importing from other directories
from Python_RAG_Agent.data_loader import load_all_documents
from Python_RAG_Agent.Embeddings import get_default_embeddings
from Python_RAG_Agent.vector_store import VectorStoreManager
```
