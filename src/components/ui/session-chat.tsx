"use client"

import * as React from "react"
import { ConversationPage } from "./conversation-page"
import { ChatInput } from "./chat-input"
import { ApiRecord, ChatMessage, ChatSession, IndexSummary, PREFERRED_CHAT_MODELS, SourceDocument, chatAPI, generateUUID, pickDefaultChatModel } from "@/lib/api"
import { AttachedFile } from "@/lib/types"
import { useEffect, useState, forwardRef, useImperativeHandle, useCallback, useRef } from "react"
import { Button } from "./button"
import type { Step } from '@/lib/api'
import { foldActivityEvents, type RawEvent } from '@/lib/activity-trace'
import { buildChatRequestSettings } from '@/components/ui/chat-request'
import { ChatSettingsModal } from '@/components/ui/chat-settings-modal'
import { IndexForm } from '@/components/IndexForm'
import IndexPicker from '@/components/IndexPicker'
import SessionIndexInfo from '@/components/SessionIndexInfo'
import { useConfirm, useAlert } from '@/components/ui/confirm-dialog'
import { Database } from 'lucide-react'

interface SessionChatProps {
  sessionId?: string
  onSessionChange?: (session: ChatSession) => void
  onNewMessage?: (message: ChatMessage) => void
  className?: string
}

// Export sendMessage function for parent components
export interface SessionChatRef {
  sendMessage: (content: string, attachedFiles?: AttachedFile[]) => Promise<void>
  currentSession: ChatSession | null
}

// Helper to shorten long titles
const truncate = (str: string, n: number = 18) => str.length > n ? str.slice(0, n) + '…' : str;

// Timing breakdown (ms) the RAG server returns when LOCALGPT_TIMINGS is on.
type TimingsMs = {
  retrieval?: number;
  rerank?: number;
  context_expand?: number;
  prune?: number;
  generation?: number;
  total?: number;
}

// Self-reflection summary, present only when the reflect flag was set.
type ReflectionInfo = {
  rounds: number;
  relevance: number | null;
  groundedness: number | null;
}

type SubQueryDetail = {
  question: string;
  answer: string;
  source_documents?: SourceDocument[];
  timings_ms?: TimingsMs;
  reflection?: ReflectionInfo;
}

const getIndexId = (index: IndexSummary) => index.index_id || index.id || null;

export const SessionChat = forwardRef<SessionChatRef, SessionChatProps>(({ 
  sessionId,
  onSessionChange,
  onNewMessage,
  className = ""
}, ref) => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploadedFiles, setUploadedFiles] = useState<{filename: string, stored_path: string}[]>([])
  const [isIndexed, setIsIndexed] = useState(false)
  const [composeSubAnswers, setComposeSubAnswers] = useState<boolean>(true)
  const [enableDecompose, setEnableDecompose] = useState<boolean>(true)
  const [enableAiRerank, setEnableAiRerank] = useState<boolean>(true)
  const [enableContextExpand, setEnableContextExpand] = useState<boolean>(true)
  const [enableStream, setEnableStream] = useState<boolean>(true)
  const [enableVerify, setEnableVerify] = useState<boolean>(true)
  // Force RAG toggle
  const [forceDocs, setForceDocs] = useState<boolean>(false)
  // Provence pruning toggle
  const [provencePrune, setProvencePrune] = useState<boolean>(false)
  // Agentic mode: plan-and-execute with evidence-driven retry. Opt-in,
  // default off (adds latency). See rag_system/agent/agentic.py.
  const [agenticMode, setAgenticMode] = useState<boolean>(false)
  // Self-reflection loop: trades latency for answer quality. Default off.
  const [enableReflect, setEnableReflect] = useState<boolean>(false)
  // Standalone multi-turn query rewrite. Default off.
  const [enableRewrite, setEnableRewrite] = useState<boolean>(false)
  // Long-form local report mode (plan -> retrieve -> draft -> compile). Off.
  const [enableReport, setEnableReport] = useState<boolean>(false)
  // Reflection advanced controls. Empty model = judge with the answer model;
  // pick a small fast model to keep scoring cheap. Loops cap the retry depth.
  const [reflectionModel, setReflectionModel] = useState<string>('')
  const [reflectionMaxLoops, setReflectionMaxLoops] = useState<number>(2)
  // Min acceptable relevance/groundedness scores (0-2) before reflection
  // rewrites/regenerates. Seeded from the backend defaults on mount.
  const [relevanceThreshold, setRelevanceThreshold] = useState<number>(1)
  const [groundednessThreshold, setGroundednessThreshold] = useState<number>(1)
  // Typed metadata filters, e.g. "project=Antapaccay, year>=2020" — parsed
  // client-side into a filters object; types are validated server-side
  // against the index's metadata schema
  const [metadataFilters, setMetadataFilters] = useState<string>('')

  const parseMetadataFilters = (raw: string): Record<string, unknown> | undefined => {
    const filters: Record<string, unknown> = {}
    for (const part of raw.split(',')) {
      const entry = part.trim()
      if (!entry) continue
      const m = entry.match(/^([a-z][a-z0-9_]*)\s*(>=|<=|!=|==|=|>|<)\s*(.+)$/i)
      if (!m) continue
      const [, field, op, value] = m
      if (op === '=' || op === '==') filters[field.toLowerCase()] = value.trim()
      else filters[field.toLowerCase()] = { [op]: value.trim() }
    }
    return Object.keys(filters).length ? filters : undefined
  }
  
  // ✨ NEW RETRIEVAL PARAMETERS
  const [retrievalK, setRetrievalK] = useState<number>(20)
  const [contextWindowSize, setContextWindowSize] = useState<number>(1)
  const [rerankerTopK, setRerankerTopK] = useState<number>(10)
  const [searchType, setSearchType] = useState<string>('hybrid')
  const [generationModels,setGenerationModels]=useState<string[]>([])
  const [selectedModel,setSelectedModel]=useState<string>(PREFERRED_CHAT_MODELS[0])
  const [currentIndexId, setCurrentIndexId] = useState<string | null>(null)
  const [currentIndexName, setCurrentIndexName] = useState<string | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [showIndexForm, setShowIndexForm] = useState(false)
  const [showIndexInfo, setShowIndexInfo] = useState(false)
  const [showIndexSwitcher, setShowIndexSwitcher] = useState(false)

  const { showConfirm, dialog: confirmDialog } = useConfirm()
  const { showAlert, dialog: alertDialog } = useAlert()

  const apiService = chatAPI

  // Active answer stream — aborted on unmount and on session switch so a
  // stale stream can't keep updating state for a conversation we left
  const streamAbortRef = useRef<AbortController | null>(null)
  // Raw SSE events for the in-flight answer, folded into the activity trace on
  // completion. A ref (not state) so accumulation never affects updater purity.
  const activityRef = useRef<RawEvent[]>([])
  useEffect(() => {
    return () => { streamAbortRef.current?.abort() }
  }, [])

  const ensureIndexHealthyForChat = async (idxId: string | null, indexName?: string | null) => {
    if (!idxId) return;
    const diagnostics = await apiService.getIndexDiagnostics(idxId);
    const label = indexName || diagnostics.name || idxId.slice(0, 8);
    if (diagnostics.health === 'unhealthy') {
      throw new Error(`Cannot chat with "${label}" because its index is unhealthy. Run diagnostics and repair it before chatting.`);
    }
    if (diagnostics.health === 'warning' && !await showConfirm(`"${label}" has diagnostics warnings. Continue chatting anyway?`)) {
      throw new Error('Chat cancelled because the linked index has warnings.');
    }
  };

  // Define loadSession with useCallback before useEffect
  const loadSession = useCallback(async (id: string) => {
    try {
      setError(null)
      const { session, messages: sessionMessages } = await apiService.getSession(id)
      
      const convertedMessages = sessionMessages.map((msg: unknown) => apiService.convertDbMessage(msg as Record<string, unknown>))
      setMessages(convertedMessages)
      setCurrentSession(session)
      
      if (onSessionChange) {
        onSessionChange(session)
      }

      // Fetch linked indexes to know table name for streaming
      try {
        const idxResp = await apiService.getSessionIndexes(id)
        if (idxResp.indexes && idxResp.indexes.length > 0) {
          const lastIdxObj = idxResp.indexes[idxResp.indexes.length - 1]
          const idxId = getIndexId(lastIdxObj)
          setCurrentIndexId(idxId ?? null)
          setCurrentIndexName(lastIdxObj.name ?? lastIdxObj.title ?? idxId?.slice(0,8) ?? null)
        }
      } catch {}
    } catch (error) {
      console.error('Failed to load session:', error)
      setError('Failed to load session')
    }
  }, [apiService, onSessionChange])

  const loadedSessionId = currentSession?.id
  useEffect(() => {
    if (sessionId) {
      if (loadedSessionId !== sessionId) {
        streamAbortRef.current?.abort()
        loadSession(sessionId)
      }
    } else {
      streamAbortRef.current?.abort()
      setMessages([])
      setCurrentSession(null)
    }
  }, [sessionId, loadedSessionId, loadSession])

  // Fetch available models on mount
  useEffect(()=>{
    (async()=>{
      try{
        const resp=await apiService.getModels();
        setGenerationModels(resp.generation_models||[])
        if(resp.generation_models&&resp.generation_models.length>0){
          setSelectedModel(pickDefaultChatModel(resp.generation_models))
        }
      }catch(e){console.warn('Failed to load models',e)}
    })()
  },[apiService])

  // Source the reflection max-loops default from the backend (single source of
  // truth); falls back to the local default if the request fails. Runs once on
  // mount, before the settings panel can be opened.
  useEffect(()=>{
    apiService.getReflectionDefaults()
      .then(d=>{
        if(typeof d.max_loops==='number') setReflectionMaxLoops(d.max_loops)
        if(typeof d.relevance_threshold==='number') setRelevanceThreshold(d.relevance_threshold)
        if(typeof d.groundedness_threshold==='number') setGroundednessThreshold(d.groundedness_threshold)
      })
      .catch(()=>{})
  },[apiService])

  const sendMessage = async (content: string, attachedFiles?: AttachedFile[]) => {
    // --- Guard Clauses ---
    // If files are being indexed, do nothing.
    if (uploadedFiles.length > 0 && !isIndexed) {
      console.warn("sendMessage called while waiting for indexing. Action blocked.");
      return;
    }
    // If no content and no files, do nothing.
    if (!content.trim() && (!attachedFiles || attachedFiles.length === 0)) return;

    try {
      setError(null)
      
      // Prefer the loaded session: after the first message in a brand-new chat,
      // the sessionId prop hasn't propagated back yet, so regenerate/resend must
      // reuse currentSession rather than create a second session.
      let activeSessionId = sessionId || currentSession?.id
      if (!activeSessionId) {
        try {
          const newSession = await apiService.createSession()
          activeSessionId = newSession.id
          setCurrentSession(newSession)
          if (onSessionChange) {
            onSessionChange(newSession)
          }
        } catch (error) {
          console.error('Failed to create session:', error)
          setError('Failed to create session')
          return
        }
      }

      // --- Action Router: Decide if this is an upload or a chat message ---
      
      // A) UPLOAD ACTION: If files are attached, this action's priority is to upload. Ignore any text content.
      if (attachedFiles && attachedFiles.length > 0) {
        setIsLoading(true)
        try {
          const files = attachedFiles.map(af => af.file)
          const uploadResult = await apiService.uploadFiles(activeSessionId, files)
          console.log('✅ Files uploaded successfully:', uploadResult)
          
          setUploadedFiles(uploadResult.uploaded_files)
          setIsIndexed(false)

          const uploadMessage = apiService.createMessage(
            `📎 Uploaded ${uploadResult.uploaded_files.length} file(s): ${uploadResult.uploaded_files.map(f => f.filename).join(', ')}. Please click 'Index Documents' to chat with them.`,
            'assistant'
          )
          setMessages(prev => [...prev, uploadMessage])
        } catch (error) {
          console.error('❌ Failed to upload files:', error)
          const errorMessage = apiService.createMessage('❌ Failed to upload files. Please try again.', 'assistant')
          setMessages(prev => [...prev, errorMessage])
        } finally {
          setIsLoading(false)
        }
        return; // End the function here.
      }

      // B) CHAT ACTION: If no files, it's a standard chat message.
      if (!content.trim()) return;

      const userMessage = apiService.createMessage(content, 'user')
      setMessages(prev => [...prev, userMessage])
      if (onNewMessage) onNewMessage(userMessage)

      setIsLoading(true)

      // Ensure we know the index id for table_name; fetch if missing
      let idxId = currentIndexId;
      if (!idxId) {
        try {
          const idxResp = await apiService.getSessionIndexes(activeSessionId as string);
          if (idxResp.indexes && idxResp.indexes.length > 0) {
            const lastIdxObj = idxResp.indexes[idxResp.indexes.length - 1];
            idxId = getIndexId(lastIdxObj);
            setCurrentIndexId(idxId ?? null);
            setCurrentIndexName(lastIdxObj.name ?? lastIdxObj.title ?? idxId?.slice(0,8) ?? null);
          }
        } catch {}
      }

      await ensureIndexHealthyForChat(idxId, currentIndexName);

      if (enableStream) {
        // Stepwise progress structure
        const steps: Step[] = [
          { key: 'analyze', label: 'Analyzing user question', status: 'pending' as const, details: '' },
          { key: 'decompose', label: 'Generating sub-queries', status: 'pending' as const, details: '' },
          { key: 'retrieval', label: 'Retrieving context', status: 'pending' as const, details: '' },
          { key: 'rerank', label: 'Reranking results', status: 'pending' as const, details: '' },
          { key: 'expand', label: 'Expanding context window', status: 'pending' as const, details: '' },
          { key: 'answer', label: 'Answering sub-queries', status: 'pending' as const, details: [] },
          { key: 'synthesize', label: 'Putting everything together', status: 'pending' as const, details: '' },
          { key: 'final', label: 'Final answer', status: 'pending' as const, details: '' },
        ];
        const placeholder: ChatMessage = {
          id: generateUUID(),
          content: { steps },
          sender: 'assistant',
          timestamp: new Date().toISOString(),
          isLoading: false,
          metadata: { message_type: 'in_progress' }
        }
        setMessages(prev => {
          const withoutLoaders = prev.filter(m => m.metadata?.message_type !== 'in_progress' && !m.isLoading)
          return [...withoutLoaders, placeholder]
        })
        // keep global isLoading true so input disabled until completion

        streamAbortRef.current?.abort()
        const streamController = new AbortController()
        streamAbortRef.current = streamController
        activityRef.current = []  // fresh trace per answer

        await apiService.streamSessionMessage(
          {
            query: content,
            session_id: activeSessionId,
            // table_name deliberately omitted: the RAG server searches all
            // of the session's linked indexes (multi-collection retrieval)
            ...buildChatRequestSettings({
              composeSubAnswers, enableDecompose, enableAiRerank, enableContextExpand,
              enableVerify, selectedModel, retrievalK, contextWindowSize, rerankerTopK,
              searchType, forceDocs, provencePrune, agenticMode, enableReflect,
              enableRewrite, enableReport, reflectionModel, reflectionMaxLoops,
              relevanceThreshold, groundednessThreshold,
              filters: parseMetadataFilters(metadataFilters),
            }),
          },
          (evt) => {
            console.log('STREAM EVENT:', evt.type, evt.data); // Debug log for SSE events
            // Accumulate every event for the generic activity trace (ref write
            // is a side effect, safe outside the pure updater below).
            activityRef.current.push({ type: evt.type, data: evt.data as Record<string, unknown> });
            // Side effects stay outside the setMessages updater — updaters
            // must be pure (StrictMode invokes them twice).
            if (['token', 'sub_query_token', 'final_answer', 'single_query_result', 'complete', 'error'].includes(evt.type)) {
              setIsLoading(false);
            }
            if (evt.type === 'error') {
              const detail = evt.data?.error;
              setError(typeof detail === 'string' && detail ? detail : 'The server reported an error while answering.');
            }
            if (evt.type === 'complete' && activeSessionId) {
              // 🔄 REFRESH SESSION: refresh session data so updated title & message count are reflected in the UI
              setTimeout(async () => {
                // If the user switched sessions in this window, the stream was
                // aborted by the session-switch effect — don't clobber the now-
                // current session with the just-finished (old) one.
                if (streamController.signal.aborted) return;
                try {
                  const { session } = await apiService.getSession(activeSessionId as string);
                  if (streamController.signal.aborted) return;
                  setCurrentSession(session);
                  if (onSessionChange) {
                    onSessionChange(session);
                  }
                } catch (error) {
                  console.error('Failed to refresh session after completion:', error);
                }
              }, 100); // Small delay to ensure backend has processed the title update
            }
            setMessages(prev => prev.map(m => {
              if (m.id !== placeholder.id) return m;
              // Copy each step object: they are shared with the previous state
              // snapshot, and mutating them in place breaks updater purity.
              const steps = ((m.content as { steps: Step[] }).steps).map(s => ({ ...s }));
              if (evt.type === 'analyze') {
                steps[0].status = 'active';
                steps[0].details = 'Analyzing your question...';
                return { ...m, content: { steps } };
              }
              if (evt.type === 'decomposition') {
                steps[0].status = 'done';
                steps[1].status = 'active';
                steps[1].details = (evt.data.sub_queries || []);
                return { ...m, content: { steps } };
              }
              if (evt.type === 'retrieval_started') {
                steps[1].status = 'done';
                steps[2].status = 'active';
                steps[2].details = 'Retrieving relevant documents...';
                return { ...m, content: { steps } };
              }
              if (evt.type === 'retrieval_done') {
                const ridx = steps.findIndex(s => s.key === 'retrieval');
                if (ridx !== -1) {
                  steps[ridx].status = 'done';
                  steps[ridx].details = 'Retrieval complete.';
                }
                const rrxIdx = steps.findIndex(s => s.key === 'rerank');
                if (rrxIdx !== -1) {
                  steps[rrxIdx].status = 'active';
                  steps[rrxIdx].details = 'Reranking results...';
                }
                return { ...m, content: { steps } };
              }
              if (evt.type === 'rerank_started') {
                const rrxIdx = steps.findIndex(s => s.key === 'rerank');
                if (rrxIdx !== -1) {
                  steps[rrxIdx].status = 'active';
                  steps[rrxIdx].details = 'Reranking results...';
                }
                return { ...m, content: { steps } };
              }
              if (evt.type === 'rerank_done') {
                const rrxIdx = steps.findIndex(s => s.key === 'rerank');
                if (rrxIdx !== -1) {
                  steps[rrxIdx].status = 'done';
                  steps[rrxIdx].details = 'Reranking complete.';
                }
                return { ...m, content: { steps } };
              }
              if (evt.type === 'context_expand_started') {
                const eidx = steps.findIndex(s => s.key === 'expand');
                if (eidx !== -1) {
                  steps[eidx].status = 'active';
                  steps[eidx].details = 'Expanding context window...';
                }
                return { ...m, content: { steps } };
              }
              if (evt.type === 'context_expand_done') {
                const eidx = steps.findIndex(s => s.key === 'expand');
                if (eidx !== -1) {
                  steps[eidx].status = 'done';
                  steps[eidx].details = 'Context expansion complete.';
                }
                // Activate answering sub-queries stage to show spinner while we wait
                const ansIdx = steps.findIndex(s => s.key === 'answer');
                if (ansIdx !== -1 && steps[ansIdx].status === 'pending') {
                  steps[ansIdx].status = 'active';
                  steps[ansIdx].details = 'Answering sub-queries...';
                }
                return { ...m, content: { steps } };
              }
              if (evt.type === 'sub_query_result') {
                if (steps.length < 8) return m; // not the RAG step layout (e.g. direct-answer)
                steps[5].status = 'active';
                const existing = Array.isArray(steps[5].details) ? steps[5].details : [];
                if (!existing.some((d) => (d as SubQueryDetail).question === evt.data.query)) {
                  steps[5].details = [...existing, {
                    question: evt.data.query,
                    answer: evt.data.answer,
                    source_documents: evt.data.source_documents || []
                  }];
                } else {
                  steps[5].details = existing; // no change if duplicate
                }
                return { ...m, content: { steps } };
              }
              if (evt.type === 'final_answer' || evt.type === 'single_query_result') {
                if (steps.length < 8) return m; // not the RAG step layout (e.g. direct-answer)
                steps[5].status = 'done';
                steps[6].status = 'active';
                steps[6].details = 'Synthesizing final answer...';
                return { ...m, content: { steps } };
              }
              if (evt.type === 'token') {
                // Determine final step index dynamically (7 for RAG, 0 for direct)
                const finalIdx = steps.findIndex(s => s.key === 'final' || s.key === 'direct');
                if (finalIdx === -1) return m;
                if (steps[finalIdx].key !== 'direct') {
                  steps[6].status = 'done';
                  steps[7].status = 'active';
                } else {
                  steps[0].status = 'active';
                }
                let current = '' as string;
                const detHolder = steps[finalIdx].details;
                if (detHolder && typeof detHolder === 'object' && !Array.isArray(detHolder)) {
                  current = String((detHolder as ApiRecord).answer || '');
                } else if (typeof detHolder === 'string') {
                  current = detHolder;
                }
                const tok: string = (evt.data.text || '') as string;
                if (!tok) {
                  return m;
                }
                // Append verbatim: whitespace-only chunks are paragraph breaks
                // and repeated tokens ("had had") are legitimate output.
                // Whitespace normalization happens at render time.
                const updated = current + tok;
                if (steps[finalIdx].key === 'direct') {
                  steps[0].details = updated;
                } else {
                  steps[7].details = { answer: updated, source_documents: [] };
                }
                steps[finalIdx].details = updated;
                // Mark "Putting everything together" step as done once tokens start
                const synthIdx = steps.findIndex(s => s.key === 'synthesize');
                if (synthIdx !== -1 && steps[synthIdx].status !== 'done') {
                  steps[synthIdx].status = 'done';
                }
                return { ...m, content: { steps } };
              }
              if (evt.type === 'sub_query_token') {
                const idx = evt.data.index as number;
                const tok = String(evt.data.text || '');
                if (!tok) return m;
                steps[5].status = 'active';
                // Copy the array and its entries — they are shared with the previous state
                const detailsArr: SubQueryDetail[] = (Array.isArray(steps[5].details) ? steps[5].details as SubQueryDetail[] : []).map(d => ({ ...d }));
                while (detailsArr.length <= idx) {
                  detailsArr.push({ question: String(evt.data.question || `Sub-query ${idx+1}`), answer: '' });
                }
                detailsArr[idx].answer = (detailsArr[idx].answer || '') + tok;
                steps[5].details = detailsArr;
                return { ...m, content: { steps } };
              }
              if (evt.type === 'complete') {
                const finalIdx = steps.findIndex(s => s.key === 'final' || s.key === 'direct');
                if (finalIdx === -1) return m;
                steps[finalIdx].status = 'done';

                if (steps[finalIdx].key === 'direct') {
                  // Direct answer: details is plain string
                  steps[finalIdx].details = evt.data.answer;
                } else {
                  steps[finalIdx].details = {
                    answer: evt.data.answer,
                    source_documents: evt.data.source_documents || [],
                    timings_ms: evt.data.timings_ms,
                    reflection: evt.data.reflection,
                    activity: foldActivityEvents(activityRef.current),
                  };
                }

                // Make sure any lingering steps are marked done
                steps.forEach(s => {
                  if (s.status !== 'done') s.status = 'done';
                });

                return { ...m, content: { steps }, metadata: { message_type: 'complete' } };
              }
              if (evt.type === 'direct_answer') {
                const stepsDir: Step[] = [
                  { key: 'direct', label: 'Answering directly', status: 'active' as const, details: '' }
                ];
                return { ...m, content: { steps: stepsDir } };
              }
              if (evt.type === 'error') {
                // Terminal failure: flag the step that was running so the
                // spinner stops, and mark the message as no longer in progress.
                steps.forEach(s => {
                  if (s.status === 'active') s.status = 'error';
                });
                return { ...m, content: { steps }, metadata: { message_type: 'complete' } };
              }
              return m;
            }));
          },
          streamController.signal,
        )
      } else {
        const response = await apiService.sendSessionMessage(activeSessionId, content,
          buildChatRequestSettings({
            composeSubAnswers, enableDecompose, enableAiRerank, enableContextExpand,
            enableVerify, selectedModel, retrievalK, contextWindowSize, rerankerTopK,
            searchType, forceDocs, provencePrune, agenticMode, enableReflect,
            enableRewrite, enableReport, reflectionModel, reflectionMaxLoops,
            relevanceThreshold, groundednessThreshold,
            filters: parseMetadataFilters(metadataFilters),
          }))
      
      const aiMessage: ChatMessage = {
        id: response.ai_message_id || generateUUID(),
        content: response.response,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
          metadata: { 
            message_type: 'sub_answer',
            source_documents: response.source_documents || [] 
          }
      }
      setMessages(prev => [...prev, aiMessage])
      
        if (response.session) {
          const sess = response.session
          setCurrentSession(sess)
          if (onSessionChange) onSessionChange(sess)
        }
        if (onNewMessage) onNewMessage(aiMessage)
      }

    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        // Deliberate cancellation (unmount or session switch) — not an error
        return
      }
      console.error('Failed to send message:', error)
      setError(error instanceof Error ? error.message : 'Failed to send message')
    } finally {
      setIsLoading(false)
    }
  }

  const handleIndexDocuments = async () => {
    if (!currentSession) return;

    setIsLoading(true);
    setError(null);
    try {
      const result = await apiService.indexDocuments(currentSession.id);
      console.log('✅ Indexing complete:', result);

      const indexMessage = apiService.createMessage(
        `✅ ${result.message}`,
        'assistant'
      );
      setMessages(prev => [...prev, indexMessage]);
      setIsIndexed(true);
      setUploadedFiles([]); // Clear uploaded files after indexing

    } catch (error) {
      console.error('❌ Failed to index documents:', error);
      const errorMessage = apiService.createMessage(
        '❌ Failed to index documents. Please try again.',
        'assistant'
      );
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }

  // Expose functions to parent component
  useImperativeHandle(ref, () => ({
    sendMessage,
    currentSession
  }))

  const handleAction = async (action: string, messageId: string, messageContent: string | Record<string, unknown>[] | { steps: Step[] }) => {
    console.log(`Action ${action} on message ${messageId}`)
    
    switch (action) {
      case 'copy':
        await navigator.clipboard.writeText(typeof messageContent === 'string' ? messageContent : JSON.stringify(messageContent, null, 2))
        break
      case 'regenerate':
        // Find the user message before this AI message and resend it
        const messageIndex = messages.findIndex(m => m.id === messageId)
        if (messageIndex > 0 && messages[messageIndex].sender === 'assistant') {
          const userMessage = messages[messageIndex - 1]
          if (userMessage.sender === 'user') {
            // Remove the AI message and the old user message — sendMessage
            // re-appends the user message, so keeping it would duplicate it
            setMessages(prev => prev.filter(m => m.id !== messageId && m.id !== userMessage.id))
            await sendMessage(userMessage.content as string)
          }
        }
        break
      default:
        // Handle other actions
        break
    }
  }

  const showEmptyState = (!sessionId || messages.length === 0) && !isLoading

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {error && (
        <div className="bg-red-900 text-red-200 px-4 py-2 text-sm flex-shrink-0">
          {error}
        </div>
      )}
      
      {showEmptyState ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-6 min-h-0">
          <div className="text-center text-2xl font-semibold text-gray-300 select-none">What can I help you find today?</div>
          <div className="w-full max-w-2xl px-4">
            <ChatInput
              onSendMessage={sendMessage}
              disabled={isLoading}
              placeholder="Ask anything"
              onOpenSettings={()=>setShowSettings(true)}
              onAddIndex={()=>setShowIndexForm(true)}
              leftExtras={currentIndexId && currentIndexName ? (
                <>
                  <button
                    type="button"
                    onClick={()=>setShowIndexInfo(true)}
                    title="View index info"
                    className="flex items-center gap-1 p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-full transition-colors"
                  >
                    <Database className="w-5 h-5" />
                    <span className="text-xs hidden sm:inline">{truncate(currentIndexName,12)}</span>
                  </button>
                  {currentSession && (
                    <button
                      type="button"
                      onClick={()=>setShowIndexSwitcher(true)}
                      title="Switch index"
                      className="px-2 py-1 text-xs text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
                    >
                      Switch
                    </button>
                  )}
                </>
              ) : undefined}
            />
          </div>
        </div>
      ) : (
        <>
          <ConversationPage 
            messages={messages}
            isLoading={isLoading}
            onAction={handleAction}
            className="flex-1 overflow-y-auto"
          />

          {/* Bottom input when chat active */}
          <div className="flex-shrink-0">
            {uploadedFiles.length > 0 && !isIndexed && (
              <div className="p-2 text-center bg-yellow-100 dark:bg-yellow-900 border-t border-b border-gray-200 dark:border-gray-700">
                <Button onClick={handleIndexDocuments} disabled={isLoading}>
                  {isLoading ? 'Indexing...' : 'Index Documents to Enable Chat'}
                </Button>
              </div>
            )}
            <ChatInput
              onSendMessage={sendMessage}
              disabled={isLoading || (uploadedFiles.length > 0 && !isIndexed)}
              placeholder="Message localGPT..."
              onOpenSettings={()=>setShowSettings(true)}
              onAddIndex={()=>setShowIndexForm(true)}
              leftExtras={currentIndexId && currentIndexName ? (
                <>
                  <button
                    type="button"
                    onClick={()=>setShowIndexInfo(true)}
                    title="View index info"
                    className="flex items-center gap-1 p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-full transition-colors"
                  >
                    <Database className="w-5 h-5" />
                    <span className="text-xs hidden sm:inline">{truncate(currentIndexName,12)}</span>
                  </button>
                  {currentSession && (
                    <button
                      type="button"
                      onClick={()=>setShowIndexSwitcher(true)}
                      title="Switch index"
                      className="px-2 py-1 text-xs text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
                    >
                      Switch
                    </button>
                  )}
                </>
              ) : undefined}
            />
          </div>
        </>
      )}

      {showSettings && (
        <ChatSettingsModal
          onClose={()=>setShowSettings(false)}
          options={[
            // General Settings
            {type: 'toggle', label:'Query decomposition', checked: enableDecompose, setter: setEnableDecompose},
            {type: 'toggle', label:'Compose sub-answers', checked: composeSubAnswers, setter: setComposeSubAnswers},
            {type: 'toggle', label:'Verify answer', checked: enableVerify, setter: setEnableVerify},
            {type: 'toggle', label:'Self-reflection', checked: enableReflect, setter: setEnableReflect},
            {type: 'toggle', label:'Multi-turn query rewrite', checked: enableRewrite, setter: setEnableRewrite},
            {type: 'toggle', label:'Long-form report', checked: enableReport, setter: setEnableReport},
            {type: 'dropdown', label:'Reflection model', value: reflectionModel, setter: setReflectionModel, options: [{value:'',label:'Same as answer model'}, ...generationModels.map(m=>({value:m,label:m}))]},
            {type: 'slider', label:'Max reflection loops', value: reflectionMaxLoops, setter: setReflectionMaxLoops, min: 1, max: 3, unit: ' loops'},
            {type: 'slider', label:'Relevance threshold', value: relevanceThreshold, setter: setRelevanceThreshold, min: 0, max: 2, unit: '/2'},
            {type: 'slider', label:'Groundedness threshold', value: groundednessThreshold, setter: setGroundednessThreshold, min: 0, max: 2, unit: '/2'},
            {type: 'toggle', label:'Stream phases', checked: enableStream, setter: setEnableStream},
            
            // Retrieval Settings
            {type: 'dropdown', label:'LLM model', value: selectedModel, setter: setSelectedModel, options: generationModels.map(m=>({value:m,label:m}))},
            {type: 'dropdown', label:'Search type', value: searchType, setter: setSearchType, options: [
              {value: 'hybrid', label: 'Hybrid (Vector + FTS)'},
              {value: 'vector_only', label: 'Vector Only'},
              {value: 'bm25_only', label: 'FTS Only'}
            ]},
            {type: 'slider', label:'Retrieval chunks', value: retrievalK, setter: setRetrievalK, min: 5, max: 50, unit: ' chunks'},
            
            // Reranking & Context
            {type: 'toggle', label:'AI reranker', checked: enableAiRerank, setter: setEnableAiRerank},
            {type: 'slider', label:'Reranker top chunks', value: rerankerTopK, setter: setRerankerTopK, min: 3, max: 20, unit: ' chunks'},
            {type: 'toggle', label:'Expand context window', checked: enableContextExpand, setter: setEnableContextExpand},
            {type: 'slider', label:'Context window size', value: contextWindowSize, setter: setContextWindowSize, min: 0, max: 5, unit: ' chunks'},
            {type: 'toggle', label:'Prune irrelevant sentences', checked: provencePrune, setter: setProvencePrune},
            {type: 'toggle', label:'Always search documents', checked: forceDocs, setter: setForceDocs},
            {type: 'toggle', label:'Agentic mode', checked: agenticMode, setter: setAgenticMode},
            {type: 'text', label:'Metadata filters', value: metadataFilters, setter: setMetadataFilters, placeholder: 'project=Antapaccay, year>=2020'},
          ]}
        />
      )}

      {showIndexForm && (
        <IndexForm
          onClose={()=>setShowIndexForm(false)}
          onIndexed={(s)=>{
            setShowIndexForm(false);
            setCurrentSession(s);
            if(onSessionChange) onSessionChange(s);
          }}
        />
      )}

      {showIndexInfo && currentSession && (
        <SessionIndexInfo sessionId={currentSession.id} onClose={()=>setShowIndexInfo(false)} />
      )}

      {showIndexSwitcher && (
        <IndexPicker
          onClose={()=>setShowIndexSwitcher(false)}
          onSelect={async (idxId) => {
            if (currentSession) {
              try {
                await chatAPI.linkIndexToSession(currentSession.id, idxId)
                const idxResp = await chatAPI.getSessionIndexes(currentSession.id)
                const linked = idxResp.indexes.find((i: IndexSummary) => (i.index_id || i.id) === idxId)
                setCurrentIndexId(idxId)
                setCurrentIndexName(linked?.name ?? linked?.title ?? idxId.slice(0, 8))
              } catch (e) {
                await showAlert(e instanceof Error ? e.message : 'Failed to switch index')
              }
            }
            setShowIndexSwitcher(false)
          }}
        />
      )}

      {confirmDialog}
      {alertDialog}
    </div>
  )
})

SessionChat.displayName = "SessionChat"  
