# LangChain Documentation Assistant

Query LangChain documentation using natural language powered by RAG and Claude.

## 🎯 What This Does

This system allows you to:
- ✅ Ask questions about LangChain in natural language
- ✅ Get accurate answers grounded in official documentation
- ✅ See code examples and best practices
- ✅ Get source citations with direct links
- ✅ Work offline once documentation is indexed

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install langchain langchain-anthropic langchain-community langchain-huggingface
pip install faiss-cpu sentence-transformers beautifulsoup4 requests
pip install python-dotenv
```

### 2. Set API Key

Create `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
```

### 3. Scrape Documentation

```bash
python langchain_doc_scraper.py
```

This downloads LangChain documentation pages and saves to `langchain_docs.json`.

### 4. Build Vector Store

```bash
python langchain_doc_rag.py
```

This creates a searchable FAISS vector database from the documentation.

### 5. Start Querying!

```bash
python langchain_doc_assistant.py
```

## 📖 Usage Examples

```
You: How do I create a ChatAnthropic model with memory?

📖 Answer:
To create a ChatAnthropic model with memory, you'll need to use RunnableWithMessageHistory...
[detailed answer with code examples]

📚 Documentation Sources:
  1. Chat Models - LangChain
     https://python.langchain.com/docs/modules/model_io/models/chat/
```

## 🔧 Advanced Usage

### Similarity Search

```
You: search: vector databases in langchain
```

Returns top 5 most similar documentation sections without generating an answer.

### Custom Queries

Modify `langchain_doc_rag.py` to adjust:
- `k`: Number of retrieved chunks (default: 4)
- `chunk_size`: Text chunk size (default: 1500)
- `temperature`: LLM creativity (default: 0)

## 📁 Project Structure

```
MCP_Integration/
├── langchain_doc_scraper.py      # Scrapes documentation
├── langchain_doc_rag.py           # RAG system implementation
├── langchain_doc_assistant.py     # Interactive CLI
├── langchain_docs.json            # Scraped documentation (generated)
├── langchain_docs_vectorstore/    # FAISS index (generated)
└── README.md                      # This file
```

## 🔄 Updating Documentation

To refresh documentation:

```bash
# Re-scrape
python langchain_doc_scraper.py

# Rebuild vector store
rm -rf langchain_docs_vectorstore
python langchain_doc_rag.py
```

## 🐛 Troubleshooting

### Error: "ANTHROPIC_API_KEY not found"
**Solution**: Create `.env` file with your API key

### Error: "Vector store not found"
**Solution**: Run `python langchain_doc_rag.py` to create it

### Error: "Documentation file not found"
**Solution**: Run `python langchain_doc_scraper.py` first

### Scraping fails
**Solution**: Check internet connection and LangChain docs availability

## 🎓 How It Works

1. **Scraping**: Downloads LangChain documentation pages
2. **Chunking**: Splits documents into 1500-character chunks with 300-char overlap
3. **Embedding**: Converts chunks to 384-dimensional vectors using sentence-transformers
4. **Indexing**: Stores vectors in FAISS for fast similarity search
5. **Retrieval**: Finds top-K most relevant chunks for each query
6. **Generation**: Claude generates answer based on retrieved context

## 📊 Performance

- **Setup Time**: 2-5 minutes (one-time)
- **Query Speed**: 1-3 seconds per question
- **Accuracy**: High (grounded in official docs)
- **Storage**: ~5MB vector store for 100 pages

## 🔗 Related Projects

- **Main Repository**: [Lang_chain_Create_Agents](https://github.com/log2jiten24/Lang_chain_Create_Agents)
- **MCP Documentation**: https://modelcontextprotocol.io/
- **LangChain Docs**: https://python.langchain.com/docs

## 📝 License

Educational use. Respect LangChain's documentation terms of service.

## 🤝 Contributing

Found a bug or want to improve? Open an issue or PR on the main repository!

---

**Made with ❤️ using LangChain, Claude, and FAISS**
