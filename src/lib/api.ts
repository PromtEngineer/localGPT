const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// Preferred chat (synthesis) models, best first. The local gpt-oss:20b
// answers measurably better on document questions (67% vs 53% judged
// correct on the eval set) at ~2-3x the latency; qwen3:8b is the fast
// fallback. Indexing/internal models are configured server-side.
export const PREFERRED_CHAT_MODELS = ['gpt-oss:20b', 'qwen3:8b'];

export function pickDefaultChatModel(available: string[]): string {
  for (const m of PREFERRED_CHAT_MODELS) {
    if (available.includes(m)) return m;
  }
  return available[0] ?? 'qwen3:8b';
}

// Resolved against the models actually installed, cached after first lookup.
// Falls back to the conservative qwen3:8b if the models endpoint is down —
// never to a model that may not exist on this machine.
let _resolvedDefaultModel: string | null = null;
export async function resolveDefaultChatModel(): Promise<string> {
  if (_resolvedDefaultModel) return _resolvedDefaultModel;
  try {
    const resp = await fetch(`${API_BASE_URL}/models`);
    const data = await resp.json();
    _resolvedDefaultModel = pickDefaultChatModel(data.generation_models ?? []);
  } catch {
    _resolvedDefaultModel = PREFERRED_CHAT_MODELS[PREFERRED_CHAT_MODELS.length - 1];
  }
  return _resolvedDefaultModel;
}

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
  details: unknown;
}

export type ApiRecord = Record<string, unknown>;

export type SourceDocument = ApiRecord & {
  chunk_id?: string;
  text?: string;
  rerank_score?: number;
  score?: number;
  _distance?: number;
};

export type UploadedFile = {
  filename: string;
  stored_path: string;
};

export type IndexDocument = {
  filename: string;
};

export type IndexSummary = ApiRecord & {
  id?: string;
  index_id?: string;
  name?: string;
  title?: string;
  documents?: IndexDocument[];
  metadata?: ApiRecord;
  session?: ChatSession;
  model_used?: string;
};

export type IndexingResult = {
  total_files_considered?: number;
  files_processed?: number;
  chunks_generated?: number;
  incremental?: boolean;
  unchanged_files?: number;
  chunk_cache_hits?: number;
  force_reindex?: boolean;
};

export type BuildIndexResponse = {
  message: string;
  indexing_result?: IndexingResult | null;
};

export type IndexBuildPreflight = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  document_count: number;
  total_bytes: number;
  total_size: string;
  missing_files: Array<{ filename: string; stored_path: string }>;
  unreadable_files: Array<{ filename: string; stored_path: string }>;
  rag_api_available?: boolean | null;
};

export type IndexDiagnostics = {
  index_id: string;
  name?: string;
  health: 'healthy' | 'warning' | 'unhealthy';
  ok: boolean;
  errors: string[];
  warnings: string[];
  recommendations: string[];
  recommended_action: 'none' | 'rebuild' | 'force_rebuild' | 'fix_sources';
  can_repair: boolean;
  document_count: number;
  total_bytes: number;
  total_size: string;
  missing_files: Array<{ filename: string; stored_path: string }>;
  unreadable_files: Array<{ filename: string; stored_path: string }>;
  metadata_status?: string;
  vector_table?: {
    expected_table?: string;
    exists: boolean;
    path?: string | null;
    row_count?: number | null;
    latechunk_exists?: boolean;
    error?: string | null;
  };
  overview?: {
    exists: boolean;
    path?: string | null;
    line_count?: number | null;
  };
  latest_job?: IndexJob | null;
  file_status_counts?: Record<string, number>;
};

export type IndexDiagnosticsSummary = {
  index_id: string;
  name?: string;
  health: 'healthy' | 'warning' | 'unhealthy';
  ok: boolean;
  recommended_action: 'none' | 'rebuild' | 'force_rebuild' | 'fix_sources';
  can_repair: boolean;
  error_count: number;
  warning_count: number;
  document_count: number;
  total_size: string;
  vector_exists: boolean;
  vector_rows?: number | null;
  metadata_status?: string;
  error?: string;
};

export type MaintenanceIndexHealthEntry = {
  index_id: string;
  name?: string;
  health: 'healthy' | 'warning' | 'unhealthy';
  status?: string | null;
  documents: number;
  latest_job: {
    status?: string | null;
    error?: string | null;
    created_at?: string | null;
  };
  metadata: {
    created_at?: string | null;
    chunk_size?: number | null;
    embedding_model?: string | null;
    enable_enrich?: boolean | null;
    vector_table?: string | null;
  };
};

export type MaintenanceHealthReport = {
  timestamp: string;
  indexes: MaintenanceIndexHealthEntry[];
  summary: {
    total: number;
    healthy: number;
    warning: number;
    unhealthy: number;
  };
  error?: string;
};

export type EnrichProvider = 'ollama' | 'anthropic' | 'openai' | 'groq';

export type IndexBuildOptions = {
  latechunk?: boolean;
  doclingChunk?: boolean;
  chunkSize?: number;
  chunkOverlap?: number;
  retrievalMode?: string;
  windowSize?: number;
  enableEnrich?: boolean;
  embeddingModel?: string;
  enrichModel?: string;
  enrichProvider?: EnrichProvider;
  enrichApiKey?: string;
  overviewModel?: string;
  batchSizeEmbed?: number;
  batchSizeEnrich?: number;
  forceReindex?: boolean;
};

export type IndexJob = {
  id: string;
  index_id: string;
  status: 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  stage: string;
  progress: number;
  message?: string;
  error?: string;
  cancel_requested?: boolean;
  result?: BuildIndexResponse;
  files?: Array<{
    id?: number;
    filename?: string;
    stored_path?: string;
    status: 'pending' | 'processing' | 'done' | 'failed' | 'skipped' | 'cancelled';
    stage?: string;
    chunks_generated?: number;
    error?: string;
    started_at?: string;
    finished_at?: string;
    updated_at?: string;
  }>;
  created_at?: string;
  updated_at?: string;
  finished_at?: string;
};

export interface ChatMessage {
  id: string;
  content: string | Array<ApiRecord> | { steps: Step[] };
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
  rag_system_available?: boolean;
  python_executable?: string;
  python_version?: string;
  virtual_env?: string | null;
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
  user_message_id: string;
  ai_message_id: string;
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
          model: request.model || 'llama3.2:latest',
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

  async createSession(title: string = 'New Chat', model?: string): Promise<ChatSession> {
    try {
      // Don't persist a model this machine doesn't have: resolve the
      // preference against the installed list (cached after first call)
      if (!model) {
        model = await resolveDefaultChatModel();
      }
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
      // ✨ NEW RETRIEVAL PARAMETERS
      retrievalK?: number;
      contextWindowSize?: number;
      rerankerTopK?: number;
      searchType?: string;
      denseWeight?: number;
      forceRag?: boolean;
      forceDirect?: boolean;
      provencePrune?: boolean;
      filters?: Record<string, unknown>;
      agentic?: boolean;
      reflect?: boolean;
      rewriteQuery?: boolean;
      reflectionModel?: string;
      reflectionMaxLoops?: number;
    } = {}
  ): Promise<SessionChatResponse & { source_documents: SourceDocument[] }> {
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
          // ✨ ADD NEW RETRIEVAL PARAMETERS
          ...(typeof opts.retrievalK === 'number' && { retrieval_k: opts.retrievalK }),
          ...(typeof opts.contextWindowSize === 'number' && { context_window_size: opts.contextWindowSize }),
          ...(typeof opts.rerankerTopK === 'number' && { reranker_top_k: opts.rerankerTopK }),
          ...(typeof opts.searchType === 'string' && { search_type: opts.searchType }),
          ...(typeof opts.denseWeight === 'number' && { dense_weight: opts.denseWeight }),
          ...(typeof opts.forceRag === 'boolean' && { force_rag: opts.forceRag }),
          ...(typeof opts.forceDirect === 'boolean' && { force_direct: opts.forceDirect }),
          ...(typeof opts.provencePrune === 'boolean' && { provence_prune: opts.provencePrune }),
          ...(opts.filters && { filters: opts.filters }),
          ...(typeof opts.agentic === 'boolean' && { agentic: opts.agentic }),
          ...(typeof opts.reflect === 'boolean' && { reflect: opts.reflect }),
          ...(typeof opts.rewriteQuery === 'boolean' && { rewrite_query: opts.rewriteQuery }),
          ...(opts.reflectionModel && { reflection_model: opts.reflectionModel }),
          ...(typeof opts.reflectionMaxLoops === 'number' && { reflection_max_loops: opts.reflectionMaxLoops }),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        const detail = typeof errorData.detail === 'string' ? errorData.detail : errorData.detail?.message;
        throw new Error(`Session chat error: ${errorData.error || detail || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Session chat failed:', error);
      throw error;
    }
  }

  async deleteSession(sessionId: string): Promise<{ message: string; deleted_session_id: string }> {
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

  async cleanupEmptySessions(): Promise<{ message: string; cleanup_count: number }> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/cleanup`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Cleanup sessions error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Cleanup sessions failed:', error);
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

  // Legacy upload function - can be removed if no longer needed
  async uploadPDFs(sessionId: string, files: File[]): Promise<{ 
    message: string; 
    uploaded_files: UploadedFile[]; 
    processing_results: ApiRecord[];
    session_documents: ApiRecord[];
    total_session_documents: number;
  }> {
    try {
      // Test if files have content and show size info
      let totalSize = 0;
      for (const file of files) {
        if (file.size === 0) {
          throw new Error(`File ${file.name} is empty (0 bytes)`);
        }
        totalSize += file.size;
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        console.log(`📄 File ${file.name}: ${sizeMB}MB (${file.size} bytes), type: ${file.type}`);
      }
      
      const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
      console.log(`📄 Total upload size: ${totalSizeMB}MB`);
      
      if (totalSize > 50 * 1024 * 1024) { // 50MB limit
        throw new Error(`Total file size ${totalSizeMB}MB exceeds 50MB limit`);
      }
      
      const formData = new FormData();
      
      // Use a generic field name 'file' that the backend expects
      let i = 0;
      for (const file of files) {
        formData.append(`file_${i}`, file, file.name);
        i++;
      }
      
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Upload error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('PDF upload failed:', error);
      throw error;
    }
  }

  // Convert database message format to ChatMessage format
  convertDbMessage(dbMessage: Record<string, unknown>): ChatMessage {
    return {
      id: dbMessage.id as string,
      content: dbMessage.content as string,
      sender: dbMessage.sender as 'user' | 'assistant',
      timestamp: dbMessage.timestamp as string,
      metadata: dbMessage.metadata as Record<string, unknown> | undefined,
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

  async getSessionDocuments(sessionId: string): Promise<{ files: string[]; file_count: number; session: ChatSession }> {
    const resp = await fetch(`${API_BASE_URL}/sessions/${sessionId}/documents`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch session documents: ${resp.status}`);
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

  async setIndexMetadataSchema(indexId: string, schema: Array<Record<string, unknown>>): Promise<void> {
    const resp = await fetch(`${API_BASE_URL}/indexes/${indexId}/metadata-schema`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ schema }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(`Metadata schema error: ${err.detail || err.error || resp.statusText}`);
    }
  }

  async uploadFilesToIndex(
    indexId: string,
    files: File[],
    metadata?: Record<string, unknown> | Record<string, Record<string, unknown>>,
  ): Promise<{ message: string; uploaded_files: UploadedFile[] }> {
    const fd = new FormData();
    files.forEach((f) => fd.append('files', f, f.name));
    if (metadata) fd.append('metadata', JSON.stringify(metadata));
    const resp = await fetch(`${API_BASE_URL}/indexes/${indexId}/upload`, { method: 'POST', body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(`Upload to index error: ${err.detail || err.error || resp.statusText}`);
    }
    return resp.json();
  }

  private indexBuildPayload(opts: IndexBuildOptions & { background?: boolean } = {}) {
    return {
      latechunk: opts.latechunk ?? false,
      doclingChunk: opts.doclingChunk ?? false,
      chunkSize: opts.chunkSize ?? 512,
      chunkOverlap: opts.chunkOverlap ?? 64,
      retrievalMode: opts.retrievalMode ?? 'hybrid',
      windowSize: opts.windowSize ?? 2,
      enableEnrich: opts.enableEnrich ?? true,
      embeddingModel: opts.embeddingModel,
      enrichModel: opts.enrichModel,
      enrichProvider: opts.enrichProvider ?? 'ollama',
      enrichApiKey: opts.enrichApiKey,
      overviewModel: opts.overviewModel,
      batchSizeEmbed: opts.batchSizeEmbed ?? 50,
      batchSizeEnrich: opts.batchSizeEnrich ?? 25,
      forceReindex: opts.forceReindex ?? false,
      background: opts.background ?? false,
    };
  }

  async buildIndex(indexId: string, opts: IndexBuildOptions = {}): Promise<BuildIndexResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/indexes/${indexId}/build`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(this.indexBuildPayload(opts)),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Build index error: ${errorData.error || errorData.detail || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Build index failed:', error);
      throw error;
    }
  }

  async preflightIndexBuild(indexId: string, opts: IndexBuildOptions = {}): Promise<IndexBuildPreflight> {
    const response = await fetch(`${API_BASE_URL}/indexes/${indexId}/build/preflight`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(this.indexBuildPayload(opts)),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Preflight error: ${errorData.error || errorData.detail || response.statusText}`);
    }
    return response.json();
  }

  async getIndexDiagnostics(indexId: string): Promise<IndexDiagnostics> {
    const response = await fetch(`${API_BASE_URL}/indexes/${indexId}/diagnostics`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Diagnostics error: ${errorData.error || errorData.detail || response.statusText}`);
    }
    return response.json();
  }

  async getIndexesDiagnostics(): Promise<{ diagnostics: IndexDiagnosticsSummary[]; total: number }> {
    const response = await fetch(`${API_BASE_URL}/indexes/diagnostics`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Diagnostics error: ${errorData.error || errorData.detail || response.statusText}`);
    }
    return response.json();
  }

  async getMaintenanceHealthReport(indexId?: string): Promise<MaintenanceHealthReport> {
    const url = indexId
      ? `${API_BASE_URL}/maintenance/index-health?index_id=${encodeURIComponent(indexId)}`
      : `${API_BASE_URL}/maintenance/index-health`;
    const response = await fetch(url);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Maintenance health report error: ${errorData.error || errorData.detail || response.statusText}`);
    }
    return response.json();
  }

  async startIndexBuild(indexId: string, opts: IndexBuildOptions = {}): Promise<{ message: string; job_id: string; status: string }> {
    const response = await fetch(`${API_BASE_URL}/indexes/${indexId}/build`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(this.indexBuildPayload({ ...opts, background: true })),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Start build error: ${errorData.error || errorData.detail || response.statusText}`);
    }
    return response.json();
  }

  async getIndexJob(jobId: string): Promise<IndexJob> {
    const response = await fetch(`${API_BASE_URL}/index-jobs/${jobId}`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Index job error: ${errorData.error || errorData.detail || response.statusText}`);
    }
    return response.json();
  }

  async cancelIndexJob(jobId: string): Promise<IndexJob> {
    const response = await fetch(`${API_BASE_URL}/index-jobs/${jobId}/cancel`, { method: 'POST' });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Cancel job error: ${errorData.error || errorData.detail || response.statusText}`);
    }
    return response.json();
  }

  async resumeIndexJob(jobId: string): Promise<{ job_id: string; status: string; message: string }> {
    const response = await fetch(`${API_BASE_URL}/index-jobs/${jobId}/resume`, { method: 'POST' });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Resume job error: ${errorData.error || errorData.detail || response.statusText}`);
    }
    return response.json();
  }

  async linkIndexToSession(sessionId: string, indexId: string): Promise<{ message: string }> {
    const resp = await fetch(`${API_BASE_URL}/sessions/${sessionId}/indexes/${indexId}`, { method: 'POST' });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      const detail = typeof err.detail === 'string' ? err.detail : err.detail?.message;
      throw new Error(`Link index error: ${err.error || detail || resp.statusText}`);
    }
    return resp.json();
  }

  async listIndexes(): Promise<{ indexes: IndexSummary[]; total: number }> {
    const resp = await fetch(`${API_BASE_URL}/indexes`);
    if (!resp.ok) {
      throw new Error(`Failed to list indexes: ${resp.status}`);
    }
    return resp.json();
  }

  async getSessionIndexes(sessionId: string): Promise<{ indexes: IndexSummary[]; total: number }> {
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

  // -------------------- Streaming (SSE-over-fetch) --------------------
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
      // ✨ NEW RETRIEVAL PARAMETERS
      retrievalK?: number;
      contextWindowSize?: number;
      rerankerTopK?: number;
      searchType?: string;
      denseWeight?: number;
      forceRag?: boolean;
      provencePrune?: boolean;
      filters?: Record<string, unknown>;
      agentic?: boolean;
      reflect?: boolean;
      rewriteQuery?: boolean;
      reflectionModel?: string;
      reflectionMaxLoops?: number;
    },
    onEvent: (event: { type: string; data: ApiRecord }) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const { query, model, session_id, table_name, composeSubAnswers, decompose, aiRerank, contextExpand, verify, retrievalK, contextWindowSize, rerankerTopK, searchType, denseWeight, forceRag, provencePrune, reflect, rewriteQuery, reflectionModel, reflectionMaxLoops } = params;

    const payload: Record<string, unknown> = { query };
    if (model) payload.model = model;
    if (session_id) payload.session_id = session_id;
    if (table_name) payload.table_name = table_name;
    if (typeof composeSubAnswers === 'boolean') payload.compose_sub_answers = composeSubAnswers;
    if (typeof decompose === 'boolean') payload.query_decompose = decompose;
    if (typeof aiRerank === 'boolean') payload.ai_rerank = aiRerank;
    if (typeof contextExpand === 'boolean') payload.context_expand = contextExpand;
    if (typeof verify === 'boolean') payload.verify = verify;
    // ✨ ADD NEW RETRIEVAL PARAMETERS TO PAYLOAD
    if (typeof retrievalK === 'number') payload.retrieval_k = retrievalK;
    if (typeof contextWindowSize === 'number') payload.context_window_size = contextWindowSize;
    if (typeof rerankerTopK === 'number') payload.reranker_top_k = rerankerTopK;
    if (typeof searchType === 'string') payload.search_type = searchType;
    if (typeof denseWeight === 'number') payload.dense_weight = denseWeight;
    if (typeof forceRag === 'boolean') payload.force_rag = forceRag;
    if (typeof provencePrune === 'boolean') payload.provence_prune = provencePrune;
    if (params.filters) payload.filters = params.filters;
    if (typeof params.agentic === 'boolean') payload.agentic = params.agentic;
    if (typeof reflect === 'boolean') payload.reflect = reflect;
    if (typeof rewriteQuery === 'boolean') payload.rewrite_query = rewriteQuery;
    if (reflectionModel) payload.reflection_model = reflectionModel;
    if (typeof reflectionMaxLoops === 'number') payload.reflection_max_loops = reflectionMaxLoops;

    const resp = await fetch(`${API_BASE_URL}/rag/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    });

    if (!resp.ok || !resp.body) {
      throw new Error(`Stream request failed: ${resp.status}`);
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
            // Both events are terminal on the server side; close the
            // stream so the caller unblocks
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

/**
 * Subscribe to live indexing progress via SSE.
 * Calls onEvent for each progress update; resolves when the job finishes.
 * Returns a cleanup function that aborts the stream.
 */
export function streamIndexJob(
  jobId: string,
  onEvent: (data: { id?: string; index_id?: string; status: string; stage: string; progress: number; message: string; files: ApiRecord[] }) => void,
): { cancel: () => void; promise: Promise<void> } {
  const ctrl = new AbortController();
  const promise = (async () => {
    const resp = await fetch(`${API_BASE_URL}/index-jobs/${jobId}/stream`, { signal: ctrl.signal });
    if (!resp.ok || !resp.body) throw new Error(`SSE stream failed: ${resp.status}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() ?? '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let parsed: { type?: string; data?: { message?: string } } | null = null;
        try {
          parsed = JSON.parse(line.slice(6));
        } catch { continue; /* ignore malformed */ }
        if (parsed?.type === 'progress') {
          onEvent(parsed.data as Parameters<typeof onEvent>[0]);
        } else if (parsed?.type === 'error') {
          throw new Error(parsed.data?.message || 'Index build stream reported an error');
        }
      }
    }
  })();
  return { cancel: () => ctrl.abort(), promise };
}
