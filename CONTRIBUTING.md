# Contributing to LocalGPT

Thank you for your interest in contributing to LocalGPT! This guide will help you get started with contributing to our private document intelligence platform.

## 🚀 Quick Start for Contributors

### Prerequisites
- Python 3.8+ (we test with 3.11.5)
- Node.js 16+ (we test with 23.10.0)
- Git
- Ollama (for local AI models)

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/multimodal_rag.git
   cd multimodal_rag
   
   # Add upstream remote
   git remote add upstream https://github.com/PromtEngineer/multimodal_rag.git
   ```

2. **Set Up Development Environment**
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Install Node.js dependencies
   npm install
   
   # Install Ollama and models
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull qwen3:0.6b
   ollama pull qwen3:8b
   ```

3. **Verify Setup**
   ```bash
   # Run health check
   python system_health_check.py
   
   # Start development system
   python run_system.py --mode dev
   ```

## 📋 Development Workflow

### Branch Strategy

We use a feature branch workflow:

- `main` - Production-ready code
- `docker` - Docker deployment features and documentation
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
   - Follow our [coding standards](#coding-standards)
   - Write tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run health checks
   python system_health_check.py
   
   # Test specific components
   python -m pytest tests/ -v
   
   # Test system integration
   python run_system.py --health
   ```

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
- Add tests to prevent regression

### ✨ New Features
- Discuss in issues before implementing
- Follow existing architecture patterns
- Include comprehensive tests
- Update documentation

### 📚 Documentation
- Fix typos and improve clarity
- Add examples and use cases
- Update API documentation
- Improve setup guides

### 🧪 Testing
- Add unit tests
- Improve integration tests
- Add performance benchmarks
- Test edge cases

## 📝 Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Use type hints
def process_document(file_path: str, config: Dict[str, Any]) -> ProcessingResult:
    """Process a document with the given configuration.
    
    Args:
        file_path: Path to the document file
        config: Processing configuration dictionary
        
    Returns:
        ProcessingResult object with metadata and chunks
    """
    pass

# Use descriptive variable names
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
retrieval_results = retriever.search(query, top_k=20)

# Use dataclasses for structured data
@dataclass
class IndexingConfig:
    embedding_batch_size: int = 50
    enable_late_chunking: bool = True
    chunk_size: int = 512
```

### TypeScript/React Code Style

```typescript
// Use TypeScript interfaces
interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: DocumentSource[];
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

### File Organization

```
rag_system/
├── agent/           # ReAct agent implementation
├── indexing/        # Document processing and indexing
├── retrieval/       # Search and retrieval components
├── pipelines/       # End-to-end processing pipelines
├── rerankers/       # Result reranking implementations
└── utils/           # Shared utilities

src/
├── components/      # React components
├── lib/            # Utility functions and API clients
└── app/            # Next.js app router pages
```

## 🧪 Testing Guidelines

### Unit Tests
```python
# Test file: tests/test_embeddings.py
import pytest
from rag_system.indexing.embedders import HuggingFaceEmbedder

def test_embedding_generation():
    embedder = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.create_embeddings(["test text"])
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 384  # Model dimension
    assert embeddings.dtype == np.float32
```

### Integration Tests
```python
# Test file: tests/test_integration.py
def test_end_to_end_indexing():
    """Test complete document indexing pipeline."""
    agent = get_agent("test")
    result = agent.index_documents(["test_document.pdf"])
    
    assert result.success
    assert len(result.indexed_chunks) > 0
```

### Frontend Tests
```typescript
// Test file: src/components/__tests__/ChatInterface.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatInterface } from '../ChatInterface';

test('sends message when form is submitted', async () => {
  render(<ChatInterface sessionId="test-session" />);
  
  const input = screen.getByPlaceholderText('Type your message...');
  const button = screen.getByRole('button', { name: /send/i });
  
  fireEvent.change(input, { target: { value: 'test message' } });
  fireEvent.click(button);
  
  expect(screen.getByText('test message')).toBeInTheDocument();
});
```

## 📖 Documentation Standards

### Code Documentation
```python
def create_index(
    documents: List[str],
    config: IndexingConfig,
    progress_callback: Optional[Callable[[float], None]] = None
) -> IndexingResult:
    """Create a searchable index from documents.
    
    This function processes documents through the complete indexing pipeline:
    1. Text extraction and chunking
    2. Embedding generation
    3. Vector database storage
    4. BM25 index creation
    
    Args:
        documents: List of document file paths to index
        config: Indexing configuration with model settings and parameters
        progress_callback: Optional callback function for progress updates
        
    Returns:
        IndexingResult containing success status, metrics, and any errors
        
    Raises:
        IndexingError: If document processing fails
        ModelLoadError: If embedding model cannot be loaded
        
    Example:
        >>> config = IndexingConfig(embedding_batch_size=32)
        >>> result = create_index(["doc1.pdf", "doc2.pdf"], config)
        >>> print(f"Indexed {result.chunk_count} chunks")
    """
```

### API Documentation
```python
# Use OpenAPI/FastAPI documentation
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Chat with indexed documents.
    
    Send a natural language query and receive an AI-generated response
    based on the indexed document collection.
    
    - **query**: The user's question or prompt
    - **session_id**: Chat session identifier
    - **search_type**: Type of search (vector, hybrid, bm25)
    - **retrieval_k**: Number of documents to retrieve
    
    Returns a response with the AI-generated answer and source documents.
    """
```

## 🔧 Development Tools

### Recommended VS Code Extensions
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.pylint",
    "ms-python.black-formatter",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next"
  ]
}
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Development Scripts
```bash
# Lint Python code
python -m pylint rag_system/

# Format Python code
python -m black rag_system/

# Type check
python -m mypy rag_system/

# Lint TypeScript
npm run lint

# Format TypeScript
npm run format
```

## 🐛 Issue Reporting

### Bug Reports
When reporting bugs, please include:

1. **Environment Information**
   ```
   - OS: macOS 13.4
   - Python: 3.11.5
   - Node.js: 23.10.0
   - Ollama: 0.9.5
   ```

2. **Steps to Reproduce**
   ```
   1. Start system with `python run_system.py`
   2. Upload document via web interface
   3. Ask question "What is this document about?"
   4. Error occurs during response generation
   ```

3. **Expected vs Actual Behavior**
4. **Error Messages and Logs**
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
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped in relevant files
- [ ] Changelog updated
- [ ] Docker images built and tested
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
4. **Documentation**: Improving setup and usage guides
5. **Testing**: Expanding test coverage

### Architecture Goals
- **Modularity**: Components should be loosely coupled
- **Extensibility**: Easy to add new models and features
- **Performance**: Optimize for speed and memory usage
- **Reliability**: Robust error handling and recovery
- **Privacy**: Keep user data secure and local

## 📚 Additional Resources

### Learning Resources
- [RAG System Architecture Overview](Documentation/architecture_overview.md)
- [API Reference](Documentation/api_reference.md)
- [Deployment Guide](Documentation/deployment_guide.md)
- [Troubleshooting Guide](DOCKER_TROUBLESHOOTING.md)

### External References
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 🙏 Thank You!

Thank you for contributing to LocalGPT! Your contributions help make private document intelligence accessible to everyone.

For questions about contributing, please:
1. Check existing documentation
2. Search existing issues
3. Create a new issue with the `question` label
4. Join our community discussions

Happy coding! 🚀 