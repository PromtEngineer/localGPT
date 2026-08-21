// Both must be baked in at build time (Next.js inlines NEXT_PUBLIC_* into the client bundle).
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const RAG_API_BASE_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8001';

// Defaults must match rag_system/main.py OLLAMA_CONFIG.
export const DEFAULT_GENERATION_MODEL = 'qwen3.5:9b';
export const DEFAULT_ENRICHMENT_MODEL = 'qwen3.5:4b';
export const DEFAULT_EMBEDDING_MODEL = 'microsoft/harrier-oss-v1-0.6b';

// 🆕 Simple UUID generator for client-side message IDs
export const generateUUID = () => {
  if (typeof window !== 'undefined' && window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID();
  }
  // Fallback for older browsers or non-secure contexts
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

export interface Step {
  key: string;
  label: string;
  status: 'pending' | 'active' | 'done' | 'error';
  details: any;
}

export interface ChatMessage {
  id: string;
  content: string | Array<Record<string, any>> | { steps: Step[] };
  sender: 'user' | 'assistant';
  timestamp: string;
  isLoading?: boolean;
  metadata?: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  model_used: string;
  message_count: number;
}

export interface ChatRequest {
  message: string;
  model?: string;
  conversation_history?: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

export interface ChatResponse {
  response: string;
  model: string;
  message_count: number;
}

export interface HealthResponse {
  status: string;
  ollama_running: boolean;
  available_models: string[];
  database_stats?: {
    total_sessions: number;
    total_messages: number;
    most_used_model: string | null;
  };
}

export interface ModelsResponse {
  generation_models: string[];
  embedding_models: string[];
}

export interface SessionResponse {
  sessions: ChatSession[];
  total: number;
}

export interface SessionChatResponse {
  response: string;
  session: ChatSession;
  source_documents: any[];
  used_rag: boolean;
}

class ChatAPI {
  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: request.message,
          model: request.model || DEFAULT_GENERATION_MODEL,
          conversation_history: request.conversation_history || [],
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Chat API error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Chat API failed:', error);
      throw error;
    }
  }

  // Convert ChatMessage array to conversation history format
  messagesToHistory(messages: ChatMessage[]): Array<{ role: 'user' | 'assistant'; content: string }> {
    return messages
      .filter(msg => typeof msg.content === 'string' && msg.content.trim())
      .map(msg => ({
        role: msg.sender,
        content: msg.content as string,
      }));
  }

  // Session Management
  async getSessions(): Promise<SessionResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`);
      if (!response.ok) {
        throw new Error(`Failed to get sessions: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get sessions failed:', error);
      throw error;
    }
  }

  async createSession(title: string = 'New Chat', model: string = DEFAULT_GENERATION_MODEL): Promise<ChatSession> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title, model }),
      });

      if (!response.ok) {
        throw new Error(`Failed to create session: ${response.status}`);
      }

      const data = await response.json();
      return data.session;
    } catch (error) {
      console.error('Create session failed:', error);
      throw error;
    }
  }

  async getSession(sessionId: string): Promise<{ session: ChatSession; messages: ChatMessage[] }> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`);
      if (!response.ok) {
        throw new Error(`Failed to get session: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get session failed:', error);
      throw error;
    }
  }

  async sendSessionMessage(
    sessionId: string,
    message: string,
    opts: {
      model?: string;
      composeSubAnswers?: boolean;
      decompose?: boolean;
      aiRerank?: boolean;
      contextExpand?: boolean;
      verify?: boolean;
      retrievalK?: number;
      contextWindowSize?: number;
      rerankerTopK?: number;
      searchType?: string;
      forceRag?: boolean;
      provencePrune?: boolean;
    } = {}
  ): Promise<SessionChatResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          ...(opts.model && { model: opts.model }),
          ...(typeof opts.composeSubAnswers === 'boolean' && { compose_sub_answers: opts.composeSubAnswers }),
          ...(typeof opts.decompose === 'boolean' && { query_decompose: opts.decompose }),
          ...(typeof opts.aiRerank === 'boolean' && { ai_rerank: opts.aiRerank }),
          ...(typeof opts.contextExpand === 'boolean' && { context_expand: opts.contextExpand }),
          ...(typeof opts.verify === 'boolean' && { verify: opts.verify }),
          ...(typeof opts.retrievalK === 'number' && { retrieval_k: opts.retrievalK }),
          ...(typeof opts.contextWindowSize === 'number' && { context_window_size: opts.contextWindowSize }),
          ...(typeof opts.rerankerTopK === 'number' && { reranker_top_k: opts.rerankerTopK }),
          ...(typeof opts.searchType === 'string' && { search_type: opts.searchType }),
          ...(typeof opts.forceRag === 'boolean' && { force_rag: opts.forceRag }),
          ...(typeof opts.provencePrune === 'boolean' && { provence_prune: opts.provencePrune }),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Session chat error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Session chat failed:', error);
      throw error;
    }
  }

  async deleteSession(sessionId: string): Promise<{ deleted: boolean }> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Delete session error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Delete session failed:', error);
      throw error;
    }
  }

  async renameSession(sessionId: string, newTitle: string): Promise<{ message: string; session: ChatSession }> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/rename`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title: newTitle }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Rename session error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Rename session failed:', error);
      throw error;
    }
  }

  async uploadFiles(sessionId: string, files: File[]): Promise<{ 
    message: string; 
    uploaded_files: {filename: string, stored_path: string}[]; 
  }> {
    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file, file.name);
      });

      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Upload failed' }));
        throw new Error(`Upload error: ${errorData.error || response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('File upload failed:', error);
      throw error;
    }
  }

  async indexDocuments(sessionId: string): Promise<{ message: string }> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Indexing failed' }));
        throw new Error(`Indexing error: ${errorData.error || response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Indexing failed:', error);
      throw error;
    }
  }

  // Convert database message format to ChatMessage format
  convertDbMessage(dbMessage: Record<string, unknown>): ChatMessage {
    const metadata = dbMessage.metadata as Record<string, unknown> | undefined;
    // Streamed turns persist their pipeline cascade in metadata.steps —
    // reconstruct the structured content so reloaded sessions show the
    // retrieval/rerank/verify trail and per-step citations, not a flat bubble.
    const steps = metadata?.steps;
    const content = Array.isArray(steps) && steps.length > 0
      ? { steps: steps as Step[] }
      : dbMessage.content as string;
    return {
      id: dbMessage.id as string,
      content,
      sender: dbMessage.sender as 'user' | 'assistant',
      timestamp: dbMessage.timestamp as string,
      metadata,
    };
  }

  // Create a new ChatMessage with UUID (for loading states)
  createMessage(
    content: string, 
    sender: 'user' | 'assistant', 
    isLoading = false
  ): ChatMessage {
    return {
      id: generateUUID(),
      content,
      sender,
      timestamp: new Date().toISOString(),
      isLoading,
    };
  }

  // ---------------- Models ----------------
  async getModels(): Promise<ModelsResponse> {
    const resp = await fetch(`${API_BASE_URL}/models`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch models list: ${resp.status}`);
    }
    return resp.json();
  }

  // ---------- Index endpoints ----------

  async createIndex(name: string, description?: string, metadata: Record<string, unknown> = {}): Promise<{ index_id: string }> {
    const resp = await fetch(`${API_BASE_URL}/indexes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description, metadata }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(`Create index error: ${err.error || resp.statusText}`);
    }
    return resp.json();
  }

  async uploadFilesToIndex(indexId: string, files: File[]): Promise<{ message: string; uploaded_files: any[] }> {
    const fd = new FormData();
    files.forEach((f) => fd.append('files', f, f.name));
    const resp = await fetch(`${API_BASE_URL}/indexes/${indexId}/upload`, { method: 'POST', body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(`Upload to index error: ${err.error || resp.statusText}`);
    }
    return resp.json();
  }

  async buildIndex(indexId: string, opts: {
    latechunk?: boolean;
    doclingChunk?: boolean;
    chunkSize?: number;
    retrievalMode?: string;
    windowSize?: number;
    enableEnrich?: boolean;
    embeddingModel?: string;
    enrichModel?: string;
    overviewModel?: string;
    batchSizeEmbed?: number;
    batchSizeEnrich?: number;
  // The gateway returns { response, ...meta } on a fresh build and
  // { message, note } only on the idempotent already-built path.
  } = {}): Promise<{ response: unknown; [key: string]: unknown } | { message: string; note?: string }> {
    try {
      const response = await fetch(`${API_BASE_URL}/indexes/${indexId}/build`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          latechunk: opts.latechunk ?? false,
          doclingChunk: opts.doclingChunk ?? true,
          chunkSize: opts.chunkSize ?? 512,
          retrievalMode: opts.retrievalMode ?? 'hybrid',
          windowSize: opts.windowSize ?? 2,
          enableEnrich: opts.enableEnrich ?? true,
          embeddingModel: opts.embeddingModel,
          enrichModel: opts.enrichModel,
          overviewModel: opts.overviewModel,
          batchSizeEmbed: opts.batchSizeEmbed ?? 50,
          batchSizeEnrich: opts.batchSizeEnrich ?? 25,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Build index error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Build index failed:', error);
      throw error;
    }
  }

  async linkIndexToSession(sessionId: string, indexId: string): Promise<{ message: string }> {
    const resp = await fetch(`${API_BASE_URL}/sessions/${sessionId}/indexes/${indexId}`, { method: 'POST' });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(`Link index error: ${err.error || resp.statusText}`);
    }
    return resp.json();
  }

  async listIndexes(): Promise<{ indexes: any[]; total: number }> {
    const resp = await fetch(`${API_BASE_URL}/indexes`);
    if (!resp.ok) {
      throw new Error(`Failed to list indexes: ${resp.status}`);
    }
    return resp.json();
  }

  async getSessionIndexes(sessionId: string): Promise<{ indexes: any[]; total: number }> {
    const resp = await fetch(`${API_BASE_URL}/sessions/${sessionId}/indexes`);
    if (!resp.ok) throw new Error(`Failed to get session indexes: ${resp.status}`);
    return resp.json();
  }

  async deleteIndex(indexId: string): Promise<{ message: string }> {
    const resp = await fetch(`${API_BASE_URL}/indexes/${indexId}`, {
      method: 'DELETE',
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({ error: 'Unknown error'}));
      throw new Error(data.error || `Failed to delete index: ${resp.status}`);
    }
    return resp.json();
  }

  // Persist a completed streamed turn. The stream itself goes straight to the
  // RAG API, so the backend gateway never sees those messages otherwise.
  async saveStreamedTurn(
    sessionId: string,
    userMessage: string,
    assistantMessage: string,
    sourceDocuments?: any[],
    steps?: Step[],
  ): Promise<{ session: ChatSession }> {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/messages/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_message: userMessage,
        assistant_message: assistantMessage,
        source_documents: sourceDocuments ?? [],
        steps: steps ?? null,
      }),
    });
    if (!response.ok) {
      throw new Error(`Failed to save streamed turn: ${response.status}`);
    }
    return response.json();
  }

  // -------------------- Streaming (SSE-over-fetch) --------------------
  // SSE variant of sendMessage: raw LLM chat through the gateway, token by
  // token. Same event framing as streamSessionMessage (token/complete/error
  // objects on data: lines), so both parsers behave identically.
  async streamChatMessage(
    params: { message: string; conversation_history?: any[]; model?: string },
    onEvent: (event: { type: string; data: any }) => void,
  ): Promise<void> {
    const resp = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: params.message,
        conversation_history: params.conversation_history ?? [],
        ...(params.model && { model: params.model }),
      }),
    });

    if (!resp.ok || !resp.body) {
      let detail = `Stream request failed: ${resp.status}`;
      try {
        const errData = await resp.json();
        if (errData && typeof errData.error === 'string') detail = errData.error;
      } catch { /* non-JSON error body */ }
      throw new Error(detail);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let streamClosed = false;
    while (!streamClosed) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        try {
          const evt = JSON.parse(line.replace(/^data:\s*/, ''));
          onEvent(evt);
          if (evt.type === 'complete' || evt.type === 'error') {
            try { await reader.cancel(); } catch {}
            streamClosed = true;
            break;
          }
        } catch { /* noop */ }
      }
    }
  }

  async streamSessionMessage(
    params: {
      query: string;
      model?: string;
      session_id?: string;
      table_name?: string;
      composeSubAnswers?: boolean;
      decompose?: boolean;
      aiRerank?: boolean;
      contextExpand?: boolean;
      verify?: boolean;
      retrievalK?: number;
      contextWindowSize?: number;
      rerankerTopK?: number;
      searchType?: string;
      forceRag?: boolean;
      provencePrune?: boolean;
    },
    onEvent: (event: { type: string; data: any }) => void,
  ): Promise<void> {
    const { query, model, session_id, table_name, composeSubAnswers, decompose, aiRerank, contextExpand, verify, retrievalK, contextWindowSize, rerankerTopK, searchType, forceRag, provencePrune } = params;

    const payload: Record<string, unknown> = { query };
    if (model) payload.model = model;
    if (session_id) payload.session_id = session_id;
    if (table_name) payload.table_name = table_name;
    if (typeof composeSubAnswers === 'boolean') payload.compose_sub_answers = composeSubAnswers;
    if (typeof decompose === 'boolean') payload.query_decompose = decompose;
    if (typeof aiRerank === 'boolean') payload.ai_rerank = aiRerank;
    if (typeof contextExpand === 'boolean') payload.context_expand = contextExpand;
    if (typeof verify === 'boolean') payload.verify = verify;
    if (typeof retrievalK === 'number') payload.retrieval_k = retrievalK;
    if (typeof contextWindowSize === 'number') payload.context_window_size = contextWindowSize;
    if (typeof rerankerTopK === 'number') payload.reranker_top_k = rerankerTopK;
    if (typeof searchType === 'string') payload.search_type = searchType;
    if (typeof forceRag === 'boolean') payload.force_rag = forceRag;
    if (typeof provencePrune === 'boolean') payload.provence_prune = provencePrune;

    const resp = await fetch(`${RAG_API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!resp.ok || !resp.body) {
      // The gateway answers chat failures with JSON {"error": ...} and a proper
      // status (502 unreachable / 504 timeout) — surface the server's message.
      let detail = `Stream request failed: ${resp.status}`;
      try {
        const errData = await resp.json();
        if (errData && typeof errData.error === 'string') detail = errData.error;
      } catch { /* non-JSON error body */ }
      throw new Error(detail);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    let streamClosed = false;
    while (!streamClosed) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.replace(/^data:\s*/, '');
        try {
          const evt = JSON.parse(jsonStr);
          onEvent(evt);
          if (evt.type === 'complete' || evt.type === 'error') {
            // Gracefully close the stream so the caller unblocks
            try { await reader.cancel(); } catch {}
            streamClosed = true;
            break;
          }
        } catch {
          /* noop */
        }
      }
    }
  }
}

export const chatAPI = new ChatAPI(); 