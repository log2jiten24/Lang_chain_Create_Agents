# MCP Integration Guide for LangChain Documentation

## 🎯 Overview

This guide explains how to integrate LangChain documentation access into your development workflow using multiple approaches:

1. **Local RAG System** - Index and query LangChain docs locally
2. **MCP Server Setup** - Configure Claude Desktop to access your documentation
3. **Web Scraping Integration** - Keep documentation up-to-date

---

## ⚠️ Important Clarifications About MCP

### What MCP Actually Is:

**MCP (Model Context Protocol)** is Anthropic's protocol for connecting Claude Desktop and Claude API with external data sources. It's important to understand:

- ✅ MCP servers run **on your local machine** or **your infrastructure**
- ✅ You configure Claude Desktop to **connect to your MCP server**
- ✅ The MCP server **fetches/provides data** to Claude
- ❌ LangChain.com does **NOT** host an MCP server at `/mcp`
- ❌ Claude cannot **directly access** external websites through MCP
- ❌ MCP is **not a web API** - it's a local protocol

### What This Guide Provides:

Instead of a non-existent remote MCP endpoint, this guide shows you how to:

1. **Create your own MCP server** that serves LangChain documentation
2. **Build a RAG system** that indexes LangChain docs
3. **Configure Claude Desktop** to access your local documentation server
4. **Keep documentation synchronized** with upstream sources

---

## 🏗️ Architecture Options

### Option 1: RAG System (Recommended)
```
LangChain Docs (Web Scraping)
    ↓
Local Document Store
    ↓
Vector Database (FAISS)
    ↓
Query Interface
    ↓
Claude via LangChain
```

### Option 2: MCP Server
```
LangChain Docs (Cached)
    ↓
MCP Server (Python)
    ↓
Claude Desktop (MCP Client)
    ↓
Your Conversations
```

### Option 3: Hybrid Approach
```
LangChain Docs → RAG System ←→ MCP Server ←→ Claude Desktop
```

---

## 📦 Solution 1: LangChain Documentation RAG System

This is the most practical and powerful approach.

### Step 1: Install Dependencies

```bash
pip install langchain langchain-community langchain-anthropic
pip install beautifulsoup4 requests html2text
pip install faiss-cpu sentence-transformers
pip install selenium  # For JavaScript-heavy pages (optional)
```

### Step 2: Create Documentation Scraper

Create `langchain_doc_scraper.py`:

```python
import requests
from bs4 import BeautifulSoup
from typing import List
import time
import os

class LangChainDocScraper:
    """Scrape LangChain documentation for local indexing"""

    def __init__(self, base_url="https://python.langchain.com/docs"):
        self.base_url = base_url
        self.visited_urls = set()

    def scrape_page(self, url: str) -> dict:
        """Scrape a single documentation page"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract main content
            main_content = soup.find('main') or soup.find('article')
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            # Extract title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else url

            return {
                'url': url,
                'title': title_text,
                'content': text,
                'source': 'LangChain Documentation'
            }

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def scrape_sitemap(self, sitemap_url: str = None) -> List[dict]:
        """Scrape multiple pages from sitemap"""
        if sitemap_url is None:
            sitemap_url = f"{self.base_url}/sitemap.xml"

        documents = []

        try:
            response = requests.get(sitemap_url, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')

            urls = [loc.text for loc in soup.find_all('loc')]

            print(f"Found {len(urls)} URLs in sitemap")

            for i, url in enumerate(urls[:50]):  # Limit to 50 pages for demo
                if url not in self.visited_urls:
                    print(f"Scraping {i+1}/{len(urls[:50])}: {url}")
                    doc = self.scrape_page(url)
                    if doc:
                        documents.append(doc)
                        self.visited_urls.add(url)
                    time.sleep(0.5)  # Be respectful

            return documents

        except Exception as e:
            print(f"Error scraping sitemap: {e}")
            return documents

    def save_documents(self, documents: List[dict], output_file: str):
        """Save scraped documents to file"""
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(documents)} documents to {output_file}")

# Usage example
if __name__ == "__main__":
    scraper = LangChainDocScraper()
    docs = scraper.scrape_sitemap()
    scraper.save_documents(docs, "langchain_docs.json")
```

### Step 3: Create RAG System for Documentation

Create `langchain_doc_rag.py`:

```python
import json
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

class LangChainDocRAG:
    """RAG system for LangChain documentation"""

    def __init__(self, docs_file: str = "langchain_docs.json"):
        self.docs_file = docs_file
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None

    def load_documents(self) -> List[Document]:
        """Load scraped documents"""
        with open(self.docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)

        documents = []
        for doc in docs_data:
            documents.append(Document(
                page_content=doc['content'],
                metadata={
                    'source': doc['url'],
                    'title': doc['title']
                }
            ))

        print(f"Loaded {len(documents)} documents")
        return documents

    def create_vector_store(self, documents: List[Document]):
        """Create FAISS vector store"""
        print("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Creating vector store...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print("Vector store created")

    def save_vector_store(self, path: str = "langchain_docs_vectorstore"):
        """Save vector store to disk"""
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")

    def load_vector_store(self, path: str = "langchain_docs_vectorstore"):
        """Load existing vector store"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded")

    def setup_qa_chain(self):
        """Setup Q&A chain with Claude"""
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        prompt_template = """You are a LangChain documentation expert.
Use the following documentation context to answer the question.
If the answer is not in the context, say you don't know.

Context from LangChain documentation:
{context}

Question: {question}

Answer: Provide a detailed answer with code examples when relevant.
Include the source URL for reference."""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        print("Q&A chain ready")

    def query(self, question: str) -> dict:
        """Query the documentation"""
        if self.qa_chain is None:
            self.setup_qa_chain()

        response = self.qa_chain.invoke({"query": question})

        return {
            'answer': response['result'],
            'sources': [
                {
                    'url': doc.metadata['source'],
                    'title': doc.metadata['title']
                }
                for doc in response['source_documents']
            ]
        }

# Usage example
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # One-time setup
    rag = LangChainDocRAG()
    documents = rag.load_documents()
    rag.create_vector_store(documents)
    rag.save_vector_store()

    # Query the documentation
    result = rag.query("How do I create a chat model with memory?")
    print(f"Answer: {result['answer']}")
    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - {source['title']}: {source['url']}")
```

### Step 4: Interactive CLI

Create `langchain_doc_assistant.py`:

```python
from langchain_doc_rag import LangChainDocRAG
from dotenv import load_dotenv

def main():
    load_dotenv()

    print("="*70)
    print("LangChain Documentation Assistant")
    print("="*70)
    print("\nInitializing...")

    rag = LangChainDocRAG()
    rag.load_vector_store()
    rag.setup_qa_chain()

    print("\n✓ Ready! Ask questions about LangChain.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nSearching documentation...")
        result = rag.query(question)

        print(f"\nAssistant: {result['answer']}")
        print(f"\n📚 Sources:")
        for source in result['sources']:
            print(f"  • {source['title']}")
            print(f"    {source['url']}")
        print()

if __name__ == "__main__":
    main()
```

---

## 📦 Solution 2: MCP Server Configuration

If you want to use Claude Desktop with MCP, here's how to set it up.

### Prerequisites

1. Install Claude Desktop: https://claude.ai/download
2. Install Python 3.10+
3. Install MCP SDK:

```bash
npm install -g @modelcontextprotocol/server-stdio
```

### Create MCP Server

Create `mcp_langchain_server.py`:

```python
#!/usr/bin/env python3
"""
MCP Server for LangChain Documentation
Provides LangChain documentation access through MCP protocol
"""

import json
import sys
from typing import Any
from langchain_doc_rag import LangChainDocRAG

class MCPLangChainServer:
    def __init__(self):
        self.rag = LangChainDocRAG()
        try:
            self.rag.load_vector_store()
            self.rag.setup_qa_chain()
        except:
            print("Error: Vector store not found. Run langchain_doc_scraper.py first",
                  file=sys.stderr)
            sys.exit(1)

    def handle_request(self, request: dict) -> dict:
        """Handle MCP requests"""
        method = request.get('method')
        params = request.get('params', {})

        if method == 'tools/list':
            return {
                'tools': [
                    {
                        'name': 'search_langchain_docs',
                        'description': 'Search LangChain documentation',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'query': {
                                    'type': 'string',
                                    'description': 'Search query for LangChain docs'
                                }
                            },
                            'required': ['query']
                        }
                    }
                ]
            }

        elif method == 'tools/call':
            tool_name = params.get('name')
            arguments = params.get('arguments', {})

            if tool_name == 'search_langchain_docs':
                query = arguments.get('query', '')
                result = self.rag.query(query)

                return {
                    'content': [
                        {
                            'type': 'text',
                            'text': result['answer']
                        },
                        {
                            'type': 'resource',
                            'resource': {
                                'uri': result['sources'][0]['url'],
                                'mimeType': 'text/html',
                                'text': f"Source: {result['sources'][0]['title']}"
                            }
                        }
                    ]
                }

        return {'error': 'Unknown method'}

    def run(self):
        """Run MCP server stdio loop"""
        for line in sys.stdin:
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except Exception as e:
                error_response = {'error': str(e)}
                print(json.dumps(error_response), flush=True)

if __name__ == '__main__':
    server = MCPLangChainServer()
    server.run()
```

### Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "langchain-docs": {
      "command": "python",
      "args": ["/path/to/your/mcp_langchain_server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Usage in Claude Desktop

Once configured, you can ask Claude Desktop:

```
Can you search the LangChain docs for how to create a chat model with memory?
```

Claude will use your MCP server to query the documentation.

---

## 🔄 Keeping Documentation Updated

Create `update_docs.sh`:

```bash
#!/bin/bash

echo "Updating LangChain documentation..."

# Scrape latest docs
python langchain_doc_scraper.py

# Rebuild vector store
python -c "
from langchain_doc_rag import LangChainDocRAG
rag = LangChainDocRAG()
docs = rag.load_documents()
rag.create_vector_store(docs)
rag.save_vector_store()
print('✓ Documentation updated')
"

echo "✓ Update complete"
```

Make it executable:
```bash
chmod +x update_docs.sh
```

Schedule with cron (Linux/Mac):
```bash
# Update weekly on Sunday at 2 AM
0 2 * * 0 /path/to/update_docs.sh
```

---

## 📊 Comparison Matrix

| Feature | RAG System | MCP Server | Direct Web |
|---------|-----------|------------|------------|
| **Setup Complexity** | Medium | High | Low |
| **Response Speed** | Fast | Fast | Slow |
| **Offline Access** | ✅ Yes | ✅ Yes | ❌ No |
| **Always Up-to-date** | Manual | Manual | ✅ Auto |
| **Claude Desktop Integration** | ❌ No | ✅ Yes | ❌ No |
| **API Integration** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Cost** | Free | Free | Free |
| **Context Size** | Limited | Limited | N/A |

---

## 🚀 Quick Start Guide

### Option A: RAG System (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Scrape documentation
python langchain_doc_scraper.py

# 3. Build vector store
python langchain_doc_rag.py

# 4. Start querying
python langchain_doc_assistant.py
```

### Option B: MCP Server

```bash
# 1. Complete Option A setup first

# 2. Make MCP server executable
chmod +x mcp_langchain_server.py

# 3. Configure Claude Desktop
# Edit config file as shown above

# 4. Restart Claude Desktop

# 5. Use in conversations
# Ask: "Search LangChain docs for..."
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors
**Solution:**
```bash
pip install --upgrade langchain langchain-community langchain-anthropic
```

### Issue: Vector store loading fails
**Solution:**
```bash
# Rebuild vector store
python langchain_doc_rag.py
```

### Issue: MCP server not connecting
**Solution:**
1. Check Claude Desktop config path
2. Verify Python path in config
3. Check server logs: `~/Library/Logs/Claude/mcp.log`

### Issue: Documentation scraping blocked
**Solution:**
- Add User-Agent header
- Increase delay between requests
- Use robots.txt compliant scraper

---

## 📚 Additional Resources

- **MCP Documentation**: https://modelcontextprotocol.io/
- **LangChain Docs**: https://python.langchain.com/docs
- **Claude Desktop**: https://claude.ai/download
- **Project Repository**: https://github.com/log2jiten24/Lang_chain_Create_Agents

---

## 🎯 Conclusion

While there's no direct MCP endpoint at `https://docs.langchain.com/mcp`, you can achieve the same goal by:

1. **Building a local RAG system** that indexes LangChain documentation
2. **Creating an MCP server** that Claude Desktop can connect to
3. **Keeping documentation synchronized** through automated scraping

The RAG system approach is recommended as it provides the most flexibility and can be integrated into any application.

---

## 📝 License & Attribution

This integration guide is provided as-is for educational purposes. Always respect LangChain's terms of service and robots.txt when scraping documentation.
