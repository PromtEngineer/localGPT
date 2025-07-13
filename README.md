# LocalGPT - Private Document Intelligence Platform

<div align="center">

![LocalGPT Logo](https://img.shields.io/badge/LocalGPT-Private%20AI-blue?style=for-the-badge)

**Transform your documents into intelligent, searchable knowledge with complete privacy**

[![GitHub Stars](https://img.shields.io/github/stars/PromtEngineer/localGPT?style=flat-square)](https://github.com/PromtEngineer/localGPT/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/PromtEngineer/localGPT?style=flat-square)](https://github.com/PromtEngineer/localGPT/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/PromtEngineer/localGPT?style=flat-square)](https://github.com/PromtEngineer/localGPT/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/PromtEngineer/localGPT?style=flat-square)](https://github.com/PromtEngineer/localGPT/pulls)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg?style=flat-square)](https://www.docker.com/)

[Quick Start](#quick-start) • [Features](#features) • [Installation](#detailed-installation) • [Getting&nbsp;Started](#getting-started) • [API&nbsp;Reference](#api-reference)

</div>

## 🚀 What is LocalGPT?

LocalGPT is a **fully private, on-premise Document Intelligence platform**. Ask questions, summarise, and uncover insights from your files with state-of-the-art AI—no data ever leaves your machine.

More than a traditional RAG (Retrieval-Augmented Generation) tool, LocalGPT features a **hybrid search engine** that blends semantic similarity, keyword matching, and [Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) for long-context precision. A **smart router** automatically selects between RAG and direct LLM answering for every query, while **contextual enrichment** and sentence-level [Context Pruning](https://huggingface.co/naver/provence-reranker-debertav3-v1) surface only the most relevant content. An independent **verification** pass adds an extra layer of accuracy.

The architecture is **modular and lightweight**—enable only the components you need. With a pure-Python core and minimal dependencies, LocalGPT is simple to deploy, run, and maintain on any infrastructure.The system has minimal dependencies on frameworks and libraries, making it easy to deploy and maintain. The RAG system is pure python and does not require any additional dependencies.

### 🖼️ UI Preview

| Home | Create Index | Chat |
|------|--------------|------|
| ![](Documentation/images/Home.png) | ![](Documentation/images/Index%20Creation.png) | ![](Documentation/images/Retrieval%20Process.png) |

## ✨ Features

- **Utmost Privacy**: Your data remains on your computer, ensuring 100% security.
- **Versatile Model Support**: Seamlessly integrate a variety of open-source models via Ollama.
- **Diverse Embeddings**: Choose from a range of open-source embeddings.
- **Reuse Your LLM**: Once downloaded, reuse your LLM without the need for repeated downloads.
- **Chat History**: Remembers your previous conversations (in a session).
- **API**: LocalGPT has an API that you can use for building RAG Applications.
- **GPU, CPU, HPU & MPS Support**: Supports multiple platforms out of the box, Chat with your data using `CUDA`, `CPU`, `HPU (Intel® Gaudi®)` or `MPS` and more!

### 📖 Document Processing
- **Multi-format Support**: PDF, DOCX, TXT, Markdown, and more (Currently only PDF is supported)
- **Contextual Enrichment**: Enhanced document understanding with AI-generated context, inspired by [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- **Batch Processing**: Handle multiple documents simultaneously

### 🤖 AI-Powered Chat
- **Natural Language Queries**: Ask questions in plain English
- **Source Attribution**: Every answer includes document references
- **Smart Routing**: Automatically chooses the best approach for each query
- **Multiple AI Models**: Support for Ollama, (support for   OpenAI and Hugging Face models in the future)


### 🛠️ Developer-Friendly
- **RESTful APIs**: Complete API access for integration
- **Real-time Progress**: Live updates during document processing
- **Flexible Configuration**: Customize models, chunk sizes, and search parameters
- **Extensible Architecture**: Plugin system for custom components

### 🎨 Modern Interface
- **Intuitive Web UI**: Clean, responsive design
- **Session Management**: Organize conversations by topic
- **Index Management**: Easy document collection management
- **Real-time Chat**: Streaming responses for immediate feedback

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher (tested with Python 3.11.5)
- Node.js 16+ and npm (tested with Node.js 23.10.0, npm 10.9.2)
- Docker (optional, for containerized deployment)
- 8GB+ RAM (16GB+ recommended)
- Ollama (required for both deployment approaches)

### Option 1: Docker Deployment (Recommended for Production)

```bash
# Clone the repository
git clone https://github.com/yourusername/localgpt.git
cd localgpt

# Install Ollama locally (required even for Docker)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:0.6b
ollama pull qwen3:8b

# Start Ollama
ollama serve

# Start with Docker (in a new terminal)
./start-docker.sh

# Access the application
open http://localhost:3000
```

**Docker Management Commands:**
```bash
# Check container status
docker compose ps

# View logs
docker compose logs -f

# Stop containers
./start-docker.sh stop
```

### Option 2: Direct Development (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:0.6b
ollama pull qwen3:8b
ollama serve

# Start the system (in a new terminal)
python run_system.py

# Access the application
open http://localhost:3000
```

**Direct Development Management:**
```bash
# Check system health (comprehensive diagnostics)
python system_health_check.py

# Check service status
python run_system.py --health

# Stop all services
python run_system.py --stop
# Or press Ctrl+C in the terminal running python run_system.py
```

### Option 3: Manual Component Startup

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start RAG API
python -m rag_system.api_server

# Terminal 3: Start Backend
cd backend && python server.py

# Terminal 4: Start Frontend
npm run dev

# Access at http://localhost:3000
```

---

### Detailed Installation

#### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.8 python3-pip nodejs npm docker.io docker-compose
```

**macOS:**
```bash
brew install python@3.8 node npm docker docker-compose
```

**Windows:**
```bash
# Install Python 3.8+, Node.js, and Docker Desktop
# Then use PowerShell or WSL2
```

#### 2. Install AI Models

**Install Ollama (Recommended):**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull qwen3:0.6b          # Fast generation model
ollama pull qwen3:8b            # High-quality generation model
```

#### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Key Configuration Options:**
```env
# AI Models
OLLAMA_HOST=http://localhost:11434
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_GENERATION_MODEL=qwen3:0.6b

# Database
DATABASE_PATH=./backend/chat_data.db
VECTOR_DB_PATH=./lancedb

# Server Settings
BACKEND_PORT=8000
FRONTEND_PORT=3000
```

#### 4. Initialize the System

```bash
# Run system health check
python system_health_check.py

# Initialize databases
python -c "from backend.database import ChatDatabase; ChatDatabase().init_database()"

# Test installation
python -c "from rag_system.main import get_agent; print('✅ Installation successful!')"

# Validate complete setup
python run_system.py --health
```

---

## 🎯 Getting Started

### 1. Create Your First Index

An **index** is a collection of processed documents that you can chat with.

#### Using the Web Interface:
1. Open http://localhost:3000
2. Click "Create New Index"
3. Upload your documents (PDF, DOCX, TXT)
4. Configure processing options
5. Click "Build Index"

#### Using Scripts:
```bash
# Simple script approach
./simple_create_index.sh "My Documents" "path/to/document.pdf"

# Interactive script
python create_index_script.py
```

#### Using API:
```bash
# Create index
curl -X POST http://localhost:8000/indexes \
  -H "Content-Type: application/json" \
  -d '{"name": "My Index", "description": "My documents"}'

# Upload documents
curl -X POST http://localhost:8000/indexes/INDEX_ID/upload \
  -F "files=@document.pdf"

# Build index
curl -X POST http://localhost:8000/indexes/INDEX_ID/build
```

### 2. Start Chatting

Once your index is built:

1. **Create a Chat Session**: Click "New Chat" or use an existing session
2. **Select Your Index**: Choose which document collection to query
3. **Ask Questions**: Type natural language questions about your documents
4. **Get Answers**: Receive AI-generated responses with source citations

### 3. Advanced Features

#### Custom Model Configuration
```bash
# Use different models for different tasks
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "title": "High Quality Session",
    "model": "qwen3:8b",
    "embedding_model": "Qwen/Qwen3-Embedding-4B"
  }'
```

#### Batch Document Processing
```bash
# Process multiple documents at once
python demo_batch_indexing.py --config batch_indexing_config.json
```

#### API Integration
```python
import requests

# Chat with your documents via API
response = requests.post('http://localhost:8000/chat', json={
    'query': 'What are the key findings in the research papers?',
    'session_id': 'your-session-id',
    'search_type': 'hybrid',
    'retrieval_k': 20
})

print(response.json()['response'])
```

---

## 🔧 Configuration

### Model Configuration

LocalGPT supports multiple AI model providers:

#### Ollama Models (Local)
```python
OLLAMA_CONFIG = {
    'host': 'http://localhost:11434',
    'generation_model': 'qwen3:0.6b',
    'embedding_model': 'nomic-embed-text'
}
```

#### Hugging Face Models
```python
EXTERNAL_MODELS = {
    'embedding': {
        'Qwen/Qwen3-Embedding-0.6B': {'dimensions': 1024},
        'Qwen/Qwen3-Embedding-4B': {'dimensions': 2048},
        'Qwen/Qwen3-Embedding-8B': {'dimensions': 4096}
    }
}
```

### Processing Configuration

```python
PIPELINE_CONFIGS = {
    'default': {
        'chunk_size': 512,
        'chunk_overlap': 64,
        'retrieval_mode': 'hybrid',
        'window_size': 5,
        'enable_enrich': True,
        'latechunk': True,
        'docling_chunk': True
    },
    'fast': {
        'chunk_size': 256,
        'chunk_overlap': 32,
        'retrieval_mode': 'vector',
        'enable_enrich': False
    }
}
```

### Search Configuration

```python
SEARCH_CONFIG = {
    'hybrid': {
        'dense_weight': 0.7,
        'sparse_weight': 0.3,
        'retrieval_k': 20,
        'reranker_top_k': 10
    }
}
```
---

## 🛠️ Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep -E "(torch|transformers|lancedb)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Model Loading Issues
```bash
# Check Ollama status
ollama list
curl http://localhost:11434/api/tags

# Pull missing models
ollama pull qwen3:0.6b
```

#### Database Issues
```bash
# Check database connectivity
python -c "from backend.database import ChatDatabase; db = ChatDatabase(); print('✅ Database OK')"

# Reset database (WARNING: This deletes all data)
rm backend/chat_data.db
python -c "from backend.database import ChatDatabase; ChatDatabase().init_database()"
```

#### Performance Issues
```bash
# Check system resources
python system_health_check.py

# Monitor memory usage
htop  # or Task Manager on Windows

# Optimize for low-memory systems
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Getting Help

1. **Check Logs**: Look at `logs/system.log` for detailed error messages
2. **System Health**: Run `python system_health_check.py`
3. **Documentation**: Check the [Technical Documentation](TECHNICAL_DOCS.md)
4. **GitHub Issues**: Report bugs and request features
5. **Community**: Join our Discord/Slack community

---

## 🔗 API Reference

### Core Endpoints

#### Chat API
```http
POST /chat
Content-Type: application/json

{
  "query": "What are the main topics discussed?",
  "session_id": "uuid",
  "search_type": "hybrid",
  "retrieval_k": 20
}
```

#### Index Management
```http
# Create index
POST /indexes
{"name": "My Index", "description": "Description"}

# Upload documents
POST /indexes/{id}/upload
Content-Type: multipart/form-data

# Build index
POST /indexes/{id}/build

# Get index status
GET /indexes/{id}
```

#### Session Management
```http
# Create session
POST /sessions
{"title": "My Session", "model": "qwen3:0.6b"}

# Get sessions
GET /sessions

# Link index to session
POST /sessions/{session_id}/indexes/{index_id}
```

### Advanced Features

#### Streaming Chat
```http
POST /chat/stream
Content-Type: application/json

{
  "query": "Explain the methodology",
  "session_id": "uuid",
  "stream": true
}
```

#### Batch Processing
```http
POST /batch/index
Content-Type: application/json

{
  "file_paths": ["doc1.pdf", "doc2.pdf"],
  "config": {
    "chunk_size": 512,
    "enable_enrich": true
  }
}
```

For complete API documentation, see [API_REFERENCE.md](API_REFERENCE.md).

---

## 🏗️ Architecture

LocalGPT is built with a modular, scalable architecture:

```mermaid
graph TB
    UI[Web Interface] --> API[Backend API]
    API --> Agent[RAG Agent]
    Agent --> Retrieval[Retrieval Pipeline]
    Agent --> Generation[Generation Pipeline]
    
    Retrieval --> Vector[Vector Search]
    Retrieval --> BM25[BM25 Search]
    Retrieval --> Rerank[Reranking]
    
    Vector --> LanceDB[(LanceDB)]
    BM25 --> BM25DB[(BM25 Index)]
    
    Generation --> Ollama[Ollama Models]
    Generation --> HF[Hugging Face Models]
    
    API --> SQLite[(SQLite DB)]
```

Overview of the Retrieval Agent

```mermaid
graph TD
    classDef llmcall fill:#e6f3ff,stroke:#007bff;
    classDef pipeline fill:#e6ffe6,stroke:#28a745;
    classDef cache fill:#fff3e0,stroke:#fd7e14;
    classDef logic fill:#f8f9fa,stroke:#6c757d;
    classDef thread stroke-dasharray: 5 5;

    A(Start: Agent.run) --> B_asyncio.run(_run_async);
    B --> C{_run_async};

    C --> C1[Get Chat History];
    C1 --> T1[Build Triage Prompt <br/> Query + Doc Overviews ];
    T1 --> T2["(asyncio.to_thread)<br/>LLM Triage: RAG or LLM_DIRECT?"]; class T2 llmcall,thread;
    T2 --> T3{Decision?};

    T3 -- RAG --> RAG_Path;
    T3 -- LLM_DIRECT --> LLM_Path;

    subgraph RAG Path
        RAG_Path --> R1[Format Query + History];
        R1 --> R2["(asyncio.to_thread)<br/>Generate Query Embedding"]; class R2 pipeline,thread;
        R2 --> R3{{Check Semantic Cache}}; class R3 cache;
        R3 -- Hit --> R_Cache_Hit(Return Cached Result);
        R_Cache_Hit --> R_Hist_Update;
        R3 -- Miss --> R4{Decomposition <br/> Enabled?};

        R4 -- Yes --> R5["(asyncio.to_thread)<br/>Decompose Raw Query"]; class R5 llmcall,thread;
        R5 --> R6{{Run Sub-Queries <br/> Parallel RAG Pipeline}}; class R6 pipeline,thread;
        R6 --> R7[Collect Results & Docs];
        R7 --> R8["(asyncio.to_thread)<br/>Compose Final Answer"]; class R8 llmcall,thread;
        R8 --> V1(RAG Answer);

        R4 -- No --> R9["(asyncio.to_thread)<br/>Run Single Query <br/>(RAG Pipeline)"]; class R9 pipeline,thread;
        R9 --> V1;

        V1 --> V2{{Verification <br/> await verify_async}}; class V2 llmcall;
        V2 --> V3(Final RAG Result);
        V3 --> R_Cache_Store{{Store in Semantic Cache}}; class R_Cache_Store cache;
        R_Cache_Store --> FinalResult;
    end

    subgraph Direct LLM Path
        LLM_Path --> L1[Format Query + History];
        L1 --> L2["(asyncio.to_thread)<br/>Generate Direct LLM Answer <br/> (No RAG)"]; class L2 llmcall,thread;
        L2 --> FinalResult(Final Direct Result);
    end

    FinalResult --> R_Hist_Update(Update Chat History);
    R_Hist_Update --> ZZZ(End: Return Result);
```

### Key Components

- **Frontend**: React/Next.js web interface
- **Backend**: Python FastAPI server
- **RAG Agent**: Intelligent query routing and processing
- **Vector Database**: LanceDB for semantic search
- **Search Engine**: BM25 for keyword search
- **AI Models**: Ollama and Hugging Face integration

---

## 🤝 Contributing

We welcome contributions from developers of all skill levels! LocalGPT is an open-source project that benefits from community involvement.

### 🚀 Quick Start for Contributors

```bash
# Fork and clone the repository
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Set up development environment
pip install -r requirements.txt
npm install

# Install Ollama and models
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:0.6b qwen3:8b

# Verify setup
python system_health_check.py
python run_system.py --mode dev
```

### 📋 How to Contribute

1. **🐛 Report Bugs**: Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
2. **💡 Request Features**: Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
3. **🔧 Submit Code**: Follow our [development workflow](CONTRIBUTING.md#development-workflow)
4. **📚 Improve Docs**: Help make our documentation better

### 📖 Detailed Guidelines

For comprehensive contributing guidelines, including:
- Development setup and workflow
- Coding standards and best practices
- Testing requirements
- Documentation standards
- Release process

**👉 See our [CONTRIBUTING.md](CONTRIBUTING.md) guide**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. For models, please check their respective licenses.

---

## 🙏 Acknowledgments

- **Ollama**: For providing excellent local AI model serving
- **LanceDB**: For high-performance vector database
- **Hugging Face**: For state-of-the-art AI models
- **React/Next.js**: For the modern web interface
- **FastAPI**: For the robust backend framework

---

## 📞 Support

- **Documentation**: [Technical Docs](TECHNICAL_DOCS.md)
- **Issues**: [GitHub Issues](https://github.com/PromtEngineer/localGPT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PromtEngineer/localGPT/discussions)
- **Email**: support@localgpt.com

---

<div align="center">

**Made with ❤️ for private, intelligent document processing**

[⭐ Star us on GitHub](https://github.com/PromtEngineer/localgpt) • [🐛 Report Bug](https://github.com/PromtEngineer/localgpt/issues) • [💡 Request Feature](https://github.com/PromtEngineer/localgpt/issues)

</div>
