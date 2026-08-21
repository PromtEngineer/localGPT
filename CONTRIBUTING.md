# Contributing to LocalGPT

Thank you for your interest in contributing to LocalGPT! This guide will help you get
started with contributing to our private document intelligence platform.

## 🚀 Quick Start for Contributors

### Prerequisites
- Python 3.10+ (3.11 recommended)
- Node.js 20+
- Git
- Ollama (for local AI models)

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/localGPT.git
   cd localGPT

   # Add upstream remote
   git remote add upstream https://github.com/PromtEngineer/localGPT.git
   ```

2. **Set Up Development Environment**
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt

   # Install Node.js dependencies
   npm install

   # Install Ollama and the two default models
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull qwen3.5:9b   # generation
   ollama pull qwen3.5:4b   # routing / enrichment / verification
   ```

   Defaults work without any configuration. To override model names, service URLs or
   `DB_PATH`, put them in a `.env` at the repository root — `rag_system/main.py` calls
   `load_dotenv()` on import, and `backend/server.py` imports it. `.env.example` lists
   every variable the code actually reads.

3. **Verify Setup**
   ```bash
   # Config, agent construction, embedding model and LanceDB access
   python system_health_check.py

   # Start Ollama + RAG API + backend + frontend
   python run_system.py --mode dev

   # In another shell: are all four services actually healthy?
   python run_system.py --health
   ```

## 📋 Development Workflow

### Branch Strategy

We use a feature branch workflow off `main`, which is the only long-lived branch:

- `main` - Production-ready code
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Making Changes

1. **Create a Feature Branch**
   ```bash
   # Update your main branch
   git checkout main
   git pull upstream main

   # Create feature branch
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow our [coding standards](#-coding-standards)
   - Update documentation as needed — docs are expected to describe what the code does
     today, not what it might do later

3. **Check Your Changes** — see [Verifying changes](#-verifying-changes)

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

## 🎯 Types of Contributions

### 🐛 Bug Fixes
- Check existing issues first
- Include reproduction steps
- Describe how you verified the fix

### ✨ New Features
- Discuss in issues before implementing
- Follow existing architecture patterns
- Update documentation

### 📚 Documentation
- Fix typos and improve clarity
- Add examples and use cases
- **Verify every command, port, endpoint, config key, model name and default against the
  code before writing it.** A doc that describes a feature the code does not have is worse
  than no doc.

### 🧪 Testing
- There is no automated test suite yet; adding one is welcome
- Until then, describe the manual verification you ran in the PR

## 📝 Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Use type hints
def convert_to_markdown(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Convert a document to Markdown, preserving layout and tables.

    Args:
        file_path: Path to the document file

    Returns:
        A list of (markdown, metadata) tuples, optionally with the
        DoclingDocument as a third element.
    """
    ...

# Use descriptive variable names
embedding_model_name = "microsoft/harrier-oss-v1-0.6b"
retrieved_docs = retriever.retrieve(text_query=query, table_name=table, k=20)
```

Conventions specific to this codebase:

- **Configuration is plain dicts**, defined once in `rag_system/main.py` and handed out as
  deep copies by `rag_system/factory.py::get_pipeline_config()`. Do not introduce a second
  place where a model name or a default lives.
- **Every config key must be read by code.** If you add a key, wire it; if you find a key
  nothing reads, delete it rather than documenting it.
- **No hardcoded model names or embedding dimensions.** Resolve models from
  `OLLAMA_CONFIG` / `EXTERNAL_MODELS` (which are environment-overridable), and derive
  vector width from the embeddings the loaded model produced.
- **Fail loudly on misconfiguration, degrade quietly on optional components.** A missing
  `embedding_model_name` raises; a reranker that cannot be loaded logs a warning and the
  query proceeds without reranking.
- Keep comments purposeful. No decorative banners for trivial code.

### TypeScript/React Code Style

```typescript
// Use TypeScript interfaces
interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: string;
}

// Use functional components with hooks
const ChatInterface: React.FC<ChatProps> = ({ sessionId }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const handleSendMessage = useCallback(async (content: string) => {
    // Implementation
  }, [sessionId]);

  return (
    <div className="chat-interface">
      {/* Component JSX */}
    </div>
  );
};
```

- All API calls go through `src/lib/api.ts`. Base URLs come from `NEXT_PUBLIC_API_URL` and
  `NEXT_PUBLIC_RAG_API_URL`; never hardcode a host.
- Response types in `src/lib/api.ts` must match what the server actually returns.

### File Organization

```
rag_system/
├── main.py          # Master configuration + CLI
├── factory.py       # The single factory (get_agent / get_indexing_pipeline)
├── api_server.py    # HTTP API on :8001
├── agent/           # Triage, decomposition, orchestration and verification loop
├── ingestion/       # Document conversion (Docling) and chunking
├── indexing/        # Embedding, LanceDB writing, enrichment, overviews
├── retrieval/       # Retrievers and query transformation
├── pipelines/       # End-to-end indexing and retrieval pipelines
├── rerankers/       # Reranking and sentence pruning
└── utils/           # LLM clients and shared helpers

backend/             # Gateway on :8000 (sessions, uploads, SQLite)

src/
├── components/      # React components
├── lib/             # API client and shared types
├── utils/           # Small helpers
└── app/             # Next.js app router pages
```

## 🧪 Verifying changes

There is no `tests/` directory and no pytest suite. Use these checks:

### Python
```bash
# Syntax check everything
find rag_system backend -name '*.py' -exec python -m py_compile {} +

# Config, agent construction, embedding model, LanceDB access, sample query
python system_health_check.py

# Are the running services healthy? (exits non-zero if not)
python run_system.py --health

# Exercise a pipeline directly
python -m rag_system.main index ./path/to/docs --mode fast
python -m rag_system.main chat "test question" --mode fast
```

### Frontend
```bash
npx tsc --noEmit     # type check
npm run lint         # Next.js ESLint
npm run build        # production build
```

`next.config.ts` sets `eslint.ignoreDuringBuilds` and `typescript.ignoreBuildErrors`, so a
successful `npm run build` does **not** imply the code type-checks. Run `npx tsc --noEmit`
separately.

### Docker
```bash
./test_docker_build.sh
```

### End-to-end smoke test
```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl -X POST http://localhost:8001/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "what is this document about?"}'
```

## 📖 Documentation Standards

### Code Documentation
```python
def run(self, file_paths: List[str] | None = None, *, documents: List[str] | None = None):
    """Process and index documents according to the pipeline configuration.

    Steps: Docling conversion -> chunking -> optional contextual enrichment ->
    embedding into LanceDB (plus the native FTS index) -> optional late chunking.

    Args:
        file_paths: Absolute paths of the documents to index
        documents: Legacy alias for file_paths

    Raises:
        TypeError: If neither argument is supplied
        ValueError: If the embeddings do not match the target table's vector width
    """
```

### HTTP handler documentation

Both servers are built on the standard library's `http.server`; there is no FastAPI, no
Pydantic models and no generated OpenAPI schema. Document a route with a handler docstring
and keep the field list in the relevant README in sync:

```python
def handle_chat(self):
    """POST /chat — answer a query with the agentic RAG pipeline.

    Body (camelCase accepted, normalised to snake_case):
      query (required), session_id, table_name, model,
      retrieval_mode (hybrid|vector_only|fts_only), force_rag,
      query_decompose, ai_rerank, context_expand, verify,
      retrieval_k, context_window_size, reranker_top_k.

    Returns {"answer": str, "source_documents": list}.
    """
```

Route tables live in [`backend/README.md`](backend/README.md) (port 8000) and
[`rag_system/DOCUMENTATION.md`](rag_system/DOCUMENTATION.md) (port 8001).

## 🔧 Development Tools

### Recommended VS Code Extensions
```json
{
  "recommendations": [
    "ms-python.python",
    "bradlc.vscode-tailwindcss",
    "ms-vscode.vscode-typescript-next"
  ]
}
```

### Formatters and linters

The repository ships an ESLint flat config (`eslint.config.mjs`, used by `npm run lint`)
and nothing else — there is no Black, pylint, mypy, Prettier or pre-commit configuration
checked in, and `npm run format` does not exist. If you run a Python formatter locally,
keep the diff limited to the lines you actually changed.

Available npm scripts (`package.json`): `dev`, `build`, `start`, `lint`.

## 🐛 Issue Reporting

### Bug Reports
When reporting bugs, please include:

1. **Environment Information**
   ```
   - OS: macOS 15.5
   - Python: 3.11.5
   - Node.js: 20.11.0
   - Ollama: 0.9.5
   - LLM_BACKEND: ollama
   ```

2. **Steps to Reproduce**
   ```
   1. Start system with `python run_system.py`
   2. Upload document via web interface
   3. Ask question "What is this document about?"
   4. Error occurs during response generation
   ```

3. **Expected vs Actual Behavior**
4. **Error Messages and Logs** — the RAG API prints most of its pipeline trace to stdout;
   `RAG_LOG_LEVEL=DEBUG` adds more
5. **Screenshots (if applicable)**

### Feature Requests
Include:
- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: What other approaches were considered?
- **Additional Context**: Any relevant examples or references

## 📦 Release Process

### Version Numbering
We use semantic versioning (semver):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Checklist
- [ ] `system_health_check.py` and `run_system.py --health` pass
- [ ] `npx tsc --noEmit` and `npm run build` pass
- [ ] Documentation updated and re-verified against the code
- [ ] Version bumped in relevant files
- [ ] Changelog updated
- [ ] Docker images built and tested (`./test_docker_build.sh`)
- [ ] Release notes prepared

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

### Getting Help
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check existing docs first
- **Code Review**: Provide thoughtful, actionable feedback

## 🎯 Project Priorities

### Current Focus Areas
1. **Performance Optimization**: Improving indexing and retrieval speed
2. **Model Support**: Adding more embedding and generation models
3. **User Experience**: Enhancing the web interface
4. **Documentation**: Keeping setup and usage guides truthful
5. **Testing**: Establishing an automated test suite

### Architecture Goals
- **Modularity**: Components should be loosely coupled
- **Extensibility**: Easy to add new models and features
- **Performance**: Optimize for speed and memory usage
- **Reliability**: Robust error handling and recovery
- **Privacy**: Keep user data secure and local

## 📚 Additional Resources

### Project documentation
- [RAG package overview](rag_system/README.md)
- [RAG reference: pipelines, config keys, HTTP API](rag_system/DOCUMENTATION.md)
- [Backend gateway](backend/README.md)
- [Watson X backend](WATSONX_README.md)
- [Architecture Overview](Documentation/architecture_overview.md)
- [API Reference](Documentation/api_reference.md)
- [Deployment Guide](Documentation/deployment_guide.md)
- [Docker Troubleshooting](DOCKER_TROUBLESHOOTING.md)

### External References
- [Ollama](https://github.com/ollama/ollama)
- [Docling](https://github.com/docling-project/docling)
- [LanceDB](https://github.com/lancedb/lancedb)
- [rerankers](https://github.com/AnswerDotAI/rerankers)
- [Next.js Documentation](https://nextjs.org/docs)

---

## 🙏 Thank You!

Thank you for contributing to LocalGPT! Your contributions help make private document
intelligence accessible to everyone.

For questions about contributing, please:
1. Check existing documentation
2. Search existing issues
3. Create a new issue with the `question` label
4. Join our community discussions

Happy coding! 🚀
