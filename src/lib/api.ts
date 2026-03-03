const API_BASE_URL = 'http://localhost:8000';

/**
 * Generates a UUID string for client-side message identification.
 * Uses the native crypto.randomUUID() when available, otherwise falls back to a custom implementation.
 * @returns {string} A UUID string in the format xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
 */
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

/**
 * Represents a step in a multi-step process or workflow.
 */
export interface Step {
  /** Unique identifier for the step */
  key: string;
  /** Human-readable label for the step */
  label: string;
  /** Current status of the step */
  status: 'pending' | 'active' | 'done';
  /** Additional details or data associated with the step */
  details: any;
}

/**
 * Represents a chat message in the conversation.
 */
export interface ChatMessage {
  /** Unique identifier for the message */
  id: string;
  /** Content of the message, can be text, structured data, or steps */
  content: string | Array<Record<string, any>> | { steps: Step[] };
  /** Who sent the message */
  sender: 'user' | 'assistant';
  /** ISO timestamp when the message was created */
  timestamp: string;
  /** Whether the message is currently being processed */
  isLoading?: boolean;
  /** Additional metadata associated with the message */
  metadata?: Record<string, unknown>;
}

/**
 * Represents a chat session containing multiple messages.
 */
export interface ChatSession {
  /** Unique identifier for the session */
  id: string;
  /** Title or name of the session */
  title: string;
  /** ISO timestamp when the session was created */
  created_at: string;
  /** ISO timestamp when the session was last updated */
  updated_at: string;
  /** The AI model used in this session */
  model_used: string;
  /** Total number of messages in the session */
  message_count: number;
}

/**
 * Request payload for sending a chat message.
 */
export interface ChatRequest {
  /** The message content to send */
  message: string;
  /** Optional AI model to use for the response */
  model?: string;
  /** Optional conversation history for context */
  conversation_history?: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

/**
 * Response from the chat API after sending a message.
 */
export interface ChatResponse {
  /** The AI's response message */
  response: string;
  /** The model that generated the response */
  model: string;
  /** Total number of messages in the conversation */
  message_count: number;
}

/**
 * Response from the health check endpoint.
 */
export interface HealthResponse {
  /** Overall status of the service */
  status: string;
  /** Whether the Ollama service is running */
  ollama_running: boolean;
  /** List of available AI models */
  available_models: string[];
  /** Optional database statistics */
  database_stats?: {
    total_sessions: number;
    total_messages: number;
    most_used_model: string | null;
  };
}

/**
 * Response containing available AI models.
 */
export interface ModelsResponse {
  /** Models available for text generation */
  generation_models: string[];
  /** Models available for text embedding */
  embedding_models: string[];
}

/**
 * Response containing a list of chat sessions.
 */
export interface SessionResponse {
  /** Array of chat sessions */
  sessions: ChatSession[];
  /** Total number of sessions */
  total: number;
}

/**
 * Response from sending a message within a specific session.
 */
export interface SessionChatResponse {
  /** The AI's response message */
  response: string;
  /** Updated session information */
  session: ChatSession;
  /** ID of the user's message */
  user_message_id: string;
  /** ID of the AI's response message */
  ai_message_id: string;
}

/**
 * API client for interacting with the chat service backend.
 * Provides methods for health checks, messaging, session management, and file operations.
 */
class ChatAPI {
  /**
   * Checks the health status of the chat service.
   * @returns {Promise<HealthResponse>} Health status information including available models and database stats
   * @throws {Error} When the health check request fails
   */
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

  /**
   * Sends a chat message and receives a response.
   * @param {ChatRequest} request - The chat request containing message and optional parameters
   * @returns {Promise<ChatResponse>} The AI's response to the message
   * @throws {Error} When the chat request fails
   */
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

  /**
   * Converts an array of ChatMessage objects to conversation history format.
   * Filters out non-string content and empty messages.
   * @param {ChatMessage[]} messages - Array of chat messages to convert
   * @returns {Array<{role: 'user' | 'assistant', content: string}>} Formatted conversation history
   */
  messagesToHistory(messages: ChatMessage[]): Array<{ role: 'user' | 'assistant'; content: string }> {
    return messages
      .filter(msg => typeof msg.content === 'string' && msg.content.trim())
      .map(msg => ({
        role: msg.sender,
        content: msg.content as string,
      }));
  }

  /**
   * Retrieves all chat sessions for the user.
   * @returns {Promise<SessionResponse>} List of sessions with metadata
   * @throws {Error} When the request fails
   */
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

  /**
   * Creates a new chat session.
   * @param {string} [title='New Chat'] - Title for the new session
   * @param {string} [model='llama3.2:latest'] - AI model to use for the session
   * @returns {Promise<ChatSession>} The created session object
   * @throws {Error} When session creation fails
   */
  async createSession(title: string = 'New Chat', model: string = 'llama3.2:latest'): Promise<ChatSession> {
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

  /**
   * Retrieves a specific session and its messages.
   * @param {string} sessionId - ID of the session to retrieve
   * @returns {Promise<{session: ChatSession, messages: ChatMessage[]}>} Session data and message history
   * @throws {Error} When the session cannot be retrieved
   */
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

  /**
   * Sends a message within a specific session with advanced options.
   * @param {string} sessionId - ID of the session to send the message to
   * @param {string} message - The message content to send
   * @param {Object} [opts={}] - Optional parameters for message processing
   * @param {string} [opts.model] - AI model to use for this message
   * @param {boolean} [opts.composeSubAnswers] - Whether to compose sub-answers
   * @param {boolean} [opts.decompose] - Whether to decompose the query
   * @param {boolean} [opts.aiRerank] - Whether to use AI reranking
   * @param {boolean} [opts.contextExpand] - Whether to expand context
   * @param {boolean} [opts.verify] - Whether to verify the response
   * @param {number} [opts.retrievalK] - Number of documents to retrieve
   * @param {number} [opts.contextWindowSize] - Size of the context window
   * @param {number} [opts.rerankerTopK] - Top K for reranking
   * @param {string} [opts.searchType] - Type of search to perform
   * @param {number} [opts.denseWeight] - Weight for dense retrieval
   * @param {boolean} [opts.forceRag] - Whether to force RAG usage
   * @param {boolean} [opts.provencePrune] - Whether to prune provenance
   * @returns {Promise<SessionChatResponse & {source_documents: any[]}>} Response with source documents
   * @throws {Error} When the message sending fails
   */
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
      provencePrune?: boolean;
    } = {}
  ): Promise<SessionChatResponse & { source_documents: any[] }> {
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

  /**
   * Deletes a chat session permanently.
   * @param {string} sessionId - ID of the session to delete
   * @returns {Promise<{message: string, deleted_session_id: string}>} Confirmation of deletion
   * @throws {Error} When the deletion fails
   */
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

  /**
   * Renames an existing chat session.
   * @param {string} sessionId - ID of the session to rename
   * @param {string} newTitle - New title for the session
   * @returns {Promise<{message: string, session: ChatSession}>} Updated session information
   * @throws {Error} When the rename operation fails
   */
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

  /**
   * Removes empty sessions that have no messages.
   * @returns {Promise<{message: string, cleanup_count: number}>} Number of sessions cleaned up
   * @throws {Error} When the cleanup operation fails
   */
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

  /**
   * Uploads files to a specific session.
   * @param {string} sessionId - ID of the session to upload files to
   * @param {File[]} files - Array of files to upload
   * @returns {Promise<{message: string, uploaded_files: {filename: string, stored_path: string}[]}>} Upload results
   * @throws {Error} When the upload fails
   */
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

  /**
   * Triggers indexing of documents for a session to enable search and retrieval.
   * @param {string} sessionId - ID of the session to index documents for
   * @returns {Promise<{message: string}>} Indexing status message
   * @throws {Error} When the indexing fails
   */
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

  /**
   * Legacy method for uploading PDF files with detailed processing information.
   * @deprecated Use uploadFiles instead
   * @param {string} sessionId - ID of the session to upload PDFs to
   * @param {File[]} files - Array of PDF files to upload
   * @returns {Promise<{message: string, uploaded_files: any[], processing_results: any[], session_documents: any[], total_session_documents: number}>} Detailed upload results
   * @throws {Error} When the upload fails or files exceed size limits
   */
  async uploadPDFs(sessionId: string, files: File[]): Promise<{ 
    message: string; 
    uploaded_files: any[]; 
    processing_results: any[];
    session_documents: any[];
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

  /**
   * Converts a database message record to a ChatMessage object.
   * @param {Record<string, unknown>} dbMessage - Raw message data from database
   * @returns {ChatMessage} Formatted chat message object
   */
  convertDbMessage(dbMessage: Record<string, unknown>): ChatMessage {
    return {
      id: dbMessage.id as string,
      content: dbMessage.content as string,
      sender: dbMessage.sender as 'user' | 'assistant',
      timestamp: dbMessage.timestamp as string,
      metadata: dbMessage.metadata as Record<string, unknown> | undefined,
    };
  }

  /**
   * Creates a new ChatMessage object with a generated UUID.
   * @param {string} content - Message content
   * @param {'user' | 'assistant'} sender - Who is sending the message
   * @param {boolean} [isLoading=false] - Whether the message is in a loading state
   * @returns {ChatMessage} New chat message object
   */
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

  /**
   * Retrieves the list of available AI models.
   * @returns {Promise<ModelsResponse>} Available generation and embedding models
   * @throws {Error} When the models list cannot be fetched
   */
  async getModels(): Promise<ModelsResponse> {
    const resp = await fetch(`${API_BASE_URL}/models`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch models list: ${resp.status}`);
    }
    return resp.json();
  }

  /**
   * Retrieves documents associated with a specific session.
   * @param {string} sessionId - ID of the session to get documents for
   * @returns {Promise<{files: string[], file_count: number, session: ChatSession}>} Session documents information
   * @throws {Error} When the documents cannot be retrieved
   */
  async getSessionDocuments(sessionId: string): Promise<{ files: string[]; file_count: number; session: ChatSession }> {
    const resp = await fetch(`${API_BASE_URL}/sessions/${sessionId}/documents`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch session documents: ${resp.status}`);
    }
    return resp.json();
  }

  /**
   * Creates a new document index for organizing and searching documents.
   * @param {string} name - Name of the index
   * @param {string} [description] - Optional description of the index
   * @param {Record<string, unknown>} [metadata={}] - Additional metadata for the index
   * @returns {Promise<{index_id: string}>} ID of the created index
   * @throws {Error} When index creation fails
   */
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

  /**
   * Uploads files to a specific document index.
   * @param {string} indexId - ID of the index to upload files to
   * @param {File[]} files - Array of files to upload
   * @returns {Promise<{message: string, uploaded_files: any[]}>} Upload results
   * @throws {Error} When the upload to index fails
   */
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

  /**
   * Builds a document index with specified processing options.
   * @param {string} indexId - ID of the index to build
   * @param {Object} [opts={}] - Build configuration options
   * @param {boolean} [opts.latechunk] - Whether to use late chunking
   * @param {boolean} [opts.doclingChunk] - Whether to use docling chunking
   * @param {number} [opts.chunkSize] - Size of text chunks
   * @param {number} [opts.chunkOverlap] - Overlap between chunks
   * @param {string} [opts.retrievalMode] - Retrieval mode to use
   * @param {number} [opts.windowSize] - Context window size
   * @param {boolean} [opts.enableEnrich] - Whether to enable enrichment
   * @param {string} [opts.embeddingModel] - Model for embeddings
   * @param {string} [opts.enrichModel] - Model for enrichment
   * @param {string} [opts.overviewModel] - Model for overviews
   * @param {number} [opts.batchSizeEmbed] - Batch size for embedding
   * @param {number} [opts.batchSizeEnrich] - Batch size for enrichment
   * @returns {Promise<{message: string}>} Build status message
   * @throws {Error} When index building fails
   */
  async buildIndex(indexId: string, opts: { 
    latechunk?: boolean; 
    doclingChunk?: boolean;
    chunkSize?: number;
    chunkOverlap?: number;
    retrievalMode?: string;
    windowSize?: number;
    enableEnrich?: boolean;
    embeddingModel?: string;
    enrichModel?: string;
    overviewModel?: string;
    batchSizeEmbed?: number;
    batchSizeEnrich?: number;
  } = {}): Promise<{ message: string }> {
    try {
      const response = await fetch(`${API_BASE_URL}/indexes/${indexId}/build`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          latechunk: opts.latechunk ?? false,
          doclingChunk: opts.doclingChunk ?? false,
          chunkSize: opts.chunkSize ?? 512,
          chunkOverlap: opts.chunkOverlap ?? 64,
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

  /**
   * Links an existing index to a chat session for document retrieval.
   * @param {string} sessionId - ID of the session to link the index to
   * @param {string} indexId - ID of the index to link
   * @returns {Promise<{message: string}>} Link status message
   * @throws {Error} When linking fails
   */
  async linkIndexToSession(sessionId: string, indexId: string): Promise<{ message: string }> {
    const resp = await fetch(`${API_BASE_URL}/sessions/${sessionId}/indexes/${indexId}`, { method: 'POST' });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(`Link index error: ${err.error || resp.statusText}`);
    }
    return resp.json();
  }

  /**
   * Retrieves all available document indexes.
   * @returns {Promise<{indexes: any[], total: number}>} List of indexes with total count
   * @throws {Error} When the indexes cannot be retrieved
   */
  async listIndexes(): Promise<{ indexes: any[]; total: number }> {
    const resp = await fetch(`${API_BASE_URL}/indexes`);
    if (!resp.ok) {
      throw new Error(`Failed to list indexes: ${resp.status}`);
    }
    return resp.json();
  }

  /**
   * Retrieves indexes linked to a specific session.
   * @param {string} sessionId - ID of the session to get indexes for
   * @returns {Promise<{indexes: any[], total: number}>} Session's linked indexes
   * @throws {Error} When the session indexes cannot be retrieved
   */
  async getSessionIndexes(sessionId: string): Promise<{ indexes: any[]; total: number }> {
    const resp = await fetch(`${API_BASE_URL}/sessions/${sessionId}/indexes`);
    if (!resp.ok) throw new Error(`Failed to get session indexes: ${resp.status}`);
    return resp.json();
  }

  /**
   * Permanently deletes a document index and all its data.
   * @param {string} indexId - ID of the index to delete
   * @returns {Promise<{message: string}>} Deletion confirmation message
   * @throws {Error} When the index deletion fails
   */
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

  /**
   * Streams a chat session message using Server-Sent Events over fetch.
   * @param {Object} params - Streaming parameters
   * @param {string} params.query - The query/message to send
   * @param {string} [params.model] - AI model to use
   * @param {string} [params.session_id] - Session ID for context
   * @param {string} [params.table_name] - Table name for data retrieval
   * @param {boolean} [params.composeSubAnswers] - Whether to compose sub-answers
   * @param {boolean} [params.decompose] - Whether to decompose the query
   * @param {boolean} [params.aiRerank] - Whether to use AI reranking
   * @param {boolean} [params.contextExpand] - Whether to expand context
   * @param {boolean} [params.verify] - Whether to verify responses
   * @param {number} [params.retrievalK] - Number of documents to retrieve
   * @param {number} [params.contextWindowSize] - Context window size
   * @param {number} [params.rerankerTopK] - Top K for reranking
   * @param {string} [params.searchType] - Type of search to perform
   * @param {number} [params.denseWeight] - Weight for dense retrieval
   * @param {boolean} [params.forceRag] - Whether to force RAG usage
   * @param {boolean} [params.provencePrune] - Whether to prune provenance
   * @param {function} onEvent - Callback function to handle streaming events
   * @returns {Promise<void>} Resolves when streaming completes
   * @throws {Error} When the streaming request fails
   */
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
    },
    onEvent: (event: { type: string; data: any }) => void,
  ): Promise<void> {
    const { query, model, session_id, table_name, composeSubAnswers, decompose, aiRerank, contextExpand, verify, retrievalK, contextWindowSize, rerankerTopK, searchType, denseWeight, forceRag, provencePrune } = params;

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

    const resp = await fetch('http://localhost:8001/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
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

      const parts = buffer.split('\\n\\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.replace(/^data:\\s*/, '');
        try {
          const evt = JSON.parse(jsonStr);
          onEvent(evt);
          if (evt.type === 'complete') {
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

/** Singleton instance of the ChatAPI for use throughout the application */
export const chatAPI = new ChatAPI();