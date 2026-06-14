// Maps chat-UI state names to the API's retrieval/chat parameter names. Both
// the streaming and non-streaming send paths build their request from this, so
// a knob (reflection, rewrite, …) can't be wired on one path and forgotten on
// the other — and the mapping is unit-testable without rendering the chat.

export interface ChatUiState {
  composeSubAnswers: boolean;
  enableDecompose: boolean;
  enableAiRerank: boolean;
  enableContextExpand: boolean;
  enableVerify: boolean;
  selectedModel: string;
  retrievalK: number;
  contextWindowSize: number;
  rerankerTopK: number;
  searchType: string;
  forceDocs: boolean;
  provencePrune: boolean;
  agenticMode: boolean;
  enableReflect: boolean;
  enableRewrite: boolean;
  reflectionModel: string;
  reflectionMaxLoops: number;
  filters?: Record<string, unknown>;
}

export interface ChatRequestSettings {
  composeSubAnswers: boolean;
  decompose: boolean;
  aiRerank: boolean;
  contextExpand: boolean;
  verify: boolean;
  model: string;
  retrievalK: number;
  contextWindowSize: number;
  rerankerTopK: number;
  searchType: string;
  forceRag: boolean;
  provencePrune: boolean;
  agentic: boolean;
  reflect: boolean;
  rewriteQuery: boolean;
  reflectionModel: string;
  reflectionMaxLoops: number;
  filters?: Record<string, unknown>;
}

export function buildChatRequestSettings(s: ChatUiState): ChatRequestSettings {
  return {
    composeSubAnswers: s.composeSubAnswers,
    decompose: s.enableDecompose,
    aiRerank: s.enableAiRerank,
    contextExpand: s.enableContextExpand,
    verify: s.enableVerify,
    model: s.selectedModel,
    retrievalK: s.retrievalK,
    contextWindowSize: s.contextWindowSize,
    rerankerTopK: s.rerankerTopK,
    searchType: s.searchType,
    forceRag: s.forceDocs,
    provencePrune: s.provencePrune,
    agentic: s.agenticMode,
    reflect: s.enableReflect,
    rewriteQuery: s.enableRewrite,
    reflectionModel: s.reflectionModel,
    reflectionMaxLoops: s.reflectionMaxLoops,
    filters: s.filters,
  };
}
