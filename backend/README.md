# localGPT Backend

Simple Python backend that connects your frontend to Ollama for local LLM chat.

## Prerequisites

1. **Install Ollama** (if not already installed):
   ```bash
   # Visit https://ollama.ai or run:
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Start Ollama**:
   ```bash
   ollama serve
   ```

3. **Pull a model** (optional, server will suggest if needed):
   ```bash
   ollama pull llama3.2
   ```

## Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Ollama connection**:
   ```bash
   python ollama_client.py
   ```

3. **Start the backend server**:
   ```bash
   python server.py
   ```

Server will run on `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```
Returns server status and available models.

### Chat
```bash
POST /chat
Content-Type: application/json

{
  "message": "Hello!",
  "model": "llama3.2:latest",
  "conversation_history": []
}
```

Returns:
```json
{
  "response": "Hello! How can I help you?",
  "model": "llama3.2:latest",
  "message_count": 1
}
```

## Testing

Test the chat endpoint:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "model": "llama3.2:latest"}'
```

## Frontend Integration

Your React frontend should connect to:
- **Backend**: `http://localhost:8000`
- **Chat endpoint**: `http://localhost:8000/chat`

## What's Next

This simple backend is ready for:
- ✅ **Real-time chat** with local LLMs
- 🔜 **Document upload** for RAG
- 🔜 **Vector database** integration
- 🔜 **Streaming responses**
- 🔜 **Chat history** persistence 