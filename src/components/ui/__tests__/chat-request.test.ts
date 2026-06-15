import { describe, it, expect } from 'vitest'
import { buildChatRequestSettings, type ChatUiState } from '@/components/ui/chat-request'

// Both send paths (streaming + non-streaming) build their request from this
// mapping, so these tests are the integration guard that the reflection/rewrite
// knobs actually reach the API call with the right param names.
const base: ChatUiState = {
  composeSubAnswers: true,
  enableDecompose: true,
  enableAiRerank: false,
  enableContextExpand: false,
  enableVerify: false,
  selectedModel: 'qwen3:8b',
  retrievalK: 20,
  contextWindowSize: 1,
  rerankerTopK: 10,
  searchType: 'hybrid',
  forceDocs: false,
  provencePrune: false,
  agenticMode: false,
  enableReflect: false,
  enableRewrite: false,
  enableReport: false,
  reflectionModel: '',
  reflectionMaxLoops: 2,
  relevanceThreshold: 1,
  groundednessThreshold: 1,
}

describe('buildChatRequestSettings', () => {
  it('maps the reflection + rewrite knobs to API param names', () => {
    const out = buildChatRequestSettings({
      ...base,
      enableReflect: true,
      enableRewrite: true,
      enableReport: true,
      reflectionModel: 'qwen3:0.6b',
      reflectionMaxLoops: 3,
      relevanceThreshold: 2,
      groundednessThreshold: 0,
    })
    expect(out).toMatchObject({
      reflect: true,
      rewriteQuery: true,
      report: true,
      reflectionModel: 'qwen3:0.6b',
      reflectionMaxLoops: 3,
      relevanceThreshold: 2,
      groundednessThreshold: 0,
    })
  })

  it('carries the off-by-default values (so the knobs are always present)', () => {
    const out = buildChatRequestSettings(base)
    expect(out).toMatchObject({
      reflect: false,
      rewriteQuery: false,
      reflectionModel: '',
      reflectionMaxLoops: 2,
      forceRag: false,
      agentic: false,
    })
    // Every reflection field is present (not undefined) regardless of state.
    for (const key of ['reflect', 'rewriteQuery', 'reflectionModel', 'reflectionMaxLoops'] as const) {
      expect(out[key]).not.toBeUndefined()
    }
  })

  it('renames the non-obvious UI->API fields', () => {
    const out = buildChatRequestSettings({ ...base, forceDocs: true, enableDecompose: false })
    expect(out.forceRag).toBe(true) // forceDocs -> forceRag
    expect(out.decompose).toBe(false) // enableDecompose -> decompose
  })
})
