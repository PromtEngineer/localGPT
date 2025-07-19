"use client"

import * as React from "react"
import { ConversationPage } from "./conversation-page"
import { ChatInput } from "./chat-input"
import { EmptyChatState } from "./empty-chat-state"
import { ChatMessage, ChatSession, chatAPI, generateUUID } from "@/lib/api"
import { AttachedFile } from "@/lib/types"
import { useEffect, useState, forwardRef, useImperativeHandle, useCallback } from "react"
import { normalizeStreamingToken } from "@/utils/textNormalization"
import { Button } from "./button"
import type { Step } from '@/lib/api'
import { ChatSettingsModal } from '@/components/ui/chat-settings-modal'
import { IndexForm } from '@/components/IndexForm'
import SessionIndexInfo from '@/components/SessionIndexInfo'
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
  
  // ✨ NEW RETRIEVAL PARAMETERS
  const [retrievalK, setRetrievalK] = useState<number>(20)
  const [contextWindowSize, setContextWindowSize] = useState<number>(1)
  const [rerankerTopK, setRerankerTopK] = useState<number>(10)
  const [searchType, setSearchType] = useState<string>('hybrid')
  const [generationModels,setGenerationModels]=useState<string[]>([])
  const [selectedModel,setSelectedModel]=useState<string>('qwen3:8b')
  const [currentIndexId, setCurrentIndexId] = useState<string | null>(null)
  const [currentIndexName, setCurrentIndexName] = useState<string | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [showIndexForm, setShowIndexForm] = useState(false)
  const [showIndexInfo, setShowIndexInfo] = useState(false)
  
  const apiService = chatAPI

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
          const lastIdxObj = idxResp.indexes[idxResp.indexes.length - 1] as any
          const idxId = (lastIdxObj.index_id ?? lastIdxObj.id) as string
          setCurrentIndexId(idxId ?? null)
          setCurrentIndexName(lastIdxObj.name ?? lastIdxObj.title ?? idxId.slice(0,8))
        }
      } catch {}
    } catch (error) {
      console.error('Failed to load session:', error)
      setError('Failed to load session')
    }
  }, [apiService, onSessionChange])

  // Load session when sessionId changes
  useEffect(() => {
    if (sessionId) {
      // Only load session if we don't already have the current session
      // This prevents overriding messages when a new session is created
      if (!currentSession || currentSession.id !== sessionId) {
        loadSession(sessionId)
      }
    } else {
      // Clear messages if no session
      setMessages([])
      setCurrentSession(null)
    }
  }, [sessionId, currentSession, loadSession]) // Added missing dependencies

  // Fetch available models on mount
  useEffect(()=>{
    (async()=>{
      try{
        const resp=await apiService.getModels();
        setGenerationModels(resp.generation_models||[])
        if(resp.generation_models&&resp.generation_models.length>0){
          const def = resp.generation_models.find((m:string)=>m==='qwen3:8b');
          setSelectedModel(def || resp.generation_models[0])
        }
      }catch(e){console.warn('Failed to load models',e)}
    })()
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
      
      let activeSessionId = sessionId
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
            const lastIdxObj = idxResp.indexes[idxResp.indexes.length - 1] as any;
            idxId = (lastIdxObj.index_id ?? lastIdxObj.id) as string;
            setCurrentIndexId(idxId ?? null);
            setCurrentIndexName(lastIdxObj.name ?? lastIdxObj.title ?? idxId.slice(0,8));
          }
        } catch {}
      }

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

        await apiService.streamSessionMessage(
          {
            query: content,
            session_id: activeSessionId,
            table_name: idxId ? `text_pages_${idxId}` : undefined,
            composeSubAnswers,
            decompose: enableDecompose,
            aiRerank: enableAiRerank,
            contextExpand: enableContextExpand,
            verify: enableVerify,
            model: selectedModel,
            // ✨ NEW RETRIEVAL PARAMETERS
            retrievalK,
            contextWindowSize,
            rerankerTopK,
            searchType,
            forceRag: forceDocs,
            provencePrune,
          },
          (evt) => {
            console.log('STREAM EVENT:', evt.type, evt.data); // Debug log for SSE events
            setMessages(prev => prev.map(m => {
              if (m.id !== placeholder.id) return m;
              const steps = [...(m.content as any).steps];
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
                steps[5].status = 'active';
                const existing = Array.isArray(steps[5].details) ? steps[5].details : [];
                if (!existing.some((d: any) => d.question === evt.data.query)) {
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
                steps[5].status = 'done';
                steps[6].status = 'active';
                steps[6].details = 'Synthesizing final answer...';
                if (isLoading) setIsLoading(false);
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
                  current = (detHolder as any).answer || '';
                } else if (typeof detHolder === 'string') {
                  current = detHolder;
                }
                const tok: string = (evt.data.text || '') as string;
                if (!tok.trim()) {
                  return m; // skip empty/whitespace-only chunks
                }
                let updated = current.endsWith(tok) ? current : current + tok;
                updated = normalizeStreamingToken('', updated);
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
                if (isLoading) setIsLoading(false);
                return { ...m, content: { steps } };
              }
              if (evt.type === 'sub_query_token') {
                const idx = evt.data.index as number;
                const tok: string = evt.data.text || '';
                if (!tok.trim()) return m;
                steps[5].status = 'active';
                let detailsArr: any[] = Array.isArray(steps[5].details) ? steps[5].details as any[] : [];
                while (detailsArr.length <= idx) {
                  detailsArr.push({ question: evt.data.question || `Sub-query ${idx+1}`, answer: '' });
                }
                const curAns: string = detailsArr[idx].answer || '';
                if (!curAns.endsWith(tok)) {
                  let updatedAnswer = curAns + tok;
                  updatedAnswer = normalizeStreamingToken('', updatedAnswer);
                  detailsArr[idx].answer = updatedAnswer;
                }
                steps[5].details = detailsArr;
                if (isLoading) setIsLoading(false);
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
                    source_documents: evt.data.source_documents || []
                  };
                }

                setIsLoading(false);
                // Make sure any lingering steps are marked done
                steps.forEach(s => {
                  if (s.status !== 'done') s.status = 'done';
                });
                
                // 🔄 REFRESH SESSION: After completion, refresh session data to get updated title
                if (activeSessionId) {
                  // Always refresh session data so updated title & message count are reflected in the UI
                  setTimeout(async () => {
                    try {
                      const { session } = await apiService.getSession(activeSessionId as string);
                      setCurrentSession(session);
                      if (onSessionChange) {
                        onSessionChange(session);
                      }
                    } catch (error) {
                      console.error('Failed to refresh session after completion:', error);
                    }
                  }, 100); // Small delay to ensure backend has processed the title update
                }
                
                return { ...m, content: { steps }, metadata: { message_type: 'complete' } };
              }
              if (evt.type === 'direct_answer') {
                const stepsDir: Step[] = [
                  { key: 'direct', label: 'Answering directly', status: 'active' as const, details: '' }
                ];
                return { ...m, content: { steps: stepsDir } };
              }
              return m;
            }));
          }
        )
      } else {
        const response = await apiService.sendSessionMessage(activeSessionId, content, { 
          composeSubAnswers, 
          decompose: enableDecompose, 
          aiRerank: enableAiRerank, 
          contextExpand: enableContextExpand, 
          verify: enableVerify,
          model: selectedModel,
          // ✨ NEW RETRIEVAL PARAMETERS
          retrievalK,
          contextWindowSize,
          rerankerTopK,
          searchType,
          forceRag: forceDocs,
          provencePrune,
        })
      
      const aiMessage: ChatMessage = {
        id: response.ai_message_id || generateUUID(),
        content: response.response,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
          metadata: { 
            message_type: 'sub_answer',
            source_documents: (response as any).source_documents || [] 
          }
      }
      setMessages(prev => [...prev, aiMessage])
      
        if ((response as any).session) {
          const sess = (response as any).session as ChatSession
          setCurrentSession(sess)
          if (onSessionChange) onSessionChange(sess)
        }
        if (onNewMessage) onNewMessage(aiMessage)
      }

    } catch (error) {
      console.error('Failed to send message:', error)
      setError('Failed to send message')
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

  const handleAction = async (action: string, messageId: string, messageContent: string | Record<string, any>[] | { steps: Step[] }) => {
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
            // Remove the AI message and resend the user message
            setMessages(prev => prev.filter(m => m.id !== messageId))
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
                <button
                  type="button"
                  onClick={()=>setShowIndexInfo(true)}
                  title="View index info"
                  className="flex items-center gap-1 p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-full transition-colors"
                >
                  <Database className="w-5 h-5" />
                  <span className="text-xs hidden sm:inline">{truncate(currentIndexName,12)}</span>
                </button>
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
                <button
                  type="button"
                  onClick={()=>setShowIndexInfo(true)}
                  title="View index info"
                  className="flex items-center gap-1 p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-full transition-colors"
                >
                  <Database className="w-5 h-5" />
                  <span className="text-xs hidden sm:inline">{truncate(currentIndexName,12)}</span>
                </button>
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

      {/* Index info modal */}
      {showIndexInfo && currentSession && (
        <SessionIndexInfo sessionId={currentSession.id} onClose={()=>setShowIndexInfo(false)} />
      )}
    </div>
  )
})

SessionChat.displayName = "SessionChat"  