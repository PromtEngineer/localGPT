"use client";

import React, { useState, useEffect } from 'react';
import { ChatInput } from '@/components/ui/chat-input';
import { chatAPI, ChatMessage, DEFAULT_GENERATION_MODEL, generateUUID } from '@/lib/api';
import { ConversationPage } from '@/components/ui/conversation-page';
import { ChatSettingsModal } from '@/components/ui/chat-settings-modal';

interface QuickChatProps {
  sessionId?: string;
  onSessionChange?: (s: any) => void;
  className?: string;
}

const QC_MODEL_KEY = 'localgpt.quickChat.model';

export function QuickChat({ sessionId: externalSessionId, onSessionChange, className="" }: QuickChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  // Starts undefined even when a session id is provided at mount, so the sync
  // effect below passes its guard and loads that session's messages.
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [generationModels, setGenerationModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    if (typeof window === 'undefined') return '';
    return window.localStorage.getItem(QC_MODEL_KEY) || '';
  });
  const [showSettings, setShowSettings] = useState(false);
  const api = chatAPI;

  // Persist the chosen model so it survives unmount/mode switches
  useEffect(() => {
    if (selectedModel) {
      try { window.localStorage.setItem(QC_MODEL_KEY, selectedModel); } catch {}
    }
  }, [selectedModel]);

  // 🔄 Sync prop -> state: when sidebar selects a different session, update local session and reset chat window
  useEffect(() => {
    if (externalSessionId && externalSessionId !== sessionId) {
      setSessionId(externalSessionId);
      // Fetch existing messages for the selected session
      (async () => {
        try {
          const data = await api.getSession(externalSessionId);
          // Convert DB messages to ChatMessage format expected by UI helper
          const msgs: ChatMessage[] = data.messages.map((m: any) => api.convertDbMessage(m));
          setMessages(msgs);
        } catch (err) {
          console.error('Failed to load messages for session', err);
          setMessages([]);
        }
      })();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [externalSessionId]);

  // Fetch available models
  useEffect(()=>{
    (async()=>{
      try{
        const resp = await api.getModels();
        setGenerationModels(resp.generation_models||[]);
        if(resp.generation_models && resp.generation_models.length>0){
          const def = resp.generation_models.find((m:string)=>m===DEFAULT_GENERATION_MODEL);
          // Keep a persisted selection only if the backend still offers it
          setSelectedModel(prev => (prev && resp.generation_models.includes(prev)) ? prev : (def || resp.generation_models[0]));
        }
      }catch(e){console.warn('Failed to load models',e);}
    })();
  },[api]);

  const sendMessage = async (content: string, _files?: any) => {
    if (!content.trim()) return;

    const userMsg: ChatMessage = {
      id: generateUUID(),
      content,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);

    setIsLoading(true);

    // Ensure we have a backend session to preserve history on the agent side
    let activeSessionId = sessionId;
    if (!activeSessionId) {
      try {
        const newSess = await api.createSession('Quick Chat');
        activeSessionId = newSess.id;
        setSessionId(activeSessionId);
        if(onSessionChange){
          onSessionChange(newSess);
        }
      } catch (err) {
        console.error('Failed to create quick-chat session', err);
      }
    }

    try {
      const history = api.messagesToHistory(messages);

      // Stream token-by-token. The placeholder is appended first and each
      // token event replaces it with a NEW message object — the memoized
      // bubbles re-render only when their message identity changes.
      const assistantId = generateUUID();
      setMessages((prev) => [...prev, {
        id: assistantId,
        content: '',
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      }]);

      let finalText = '';
      let streamError: string | null = null;
      await api.streamChatMessage(
        { message: content, conversation_history: history, model: selectedModel },
        (evt) => {
          if (evt.type === 'token') {
            const tok: string = evt.data?.text || '';
            if (!tok) return;
            finalText += tok;
            const snapshot = finalText;
            setMessages((prev) => prev.map((m) =>
              m.id === assistantId ? { ...m, content: snapshot } : m));
          } else if (evt.type === 'complete') {
            const answer = typeof evt.data?.response === 'string' && evt.data.response
              ? evt.data.response : finalText;
            finalText = answer;
            setMessages((prev) => prev.map((m) =>
              m.id === assistantId ? { ...m, content: answer } : m));
          } else if (evt.type === 'error') {
            streamError = typeof evt.data?.error === 'string' ? evt.data.error : 'Quick chat failed';
          }
        },
      );
      if (streamError) throw new Error(streamError);
      const resp = { response: finalText };

      // /chat itself persists nothing — save the turn against the session so it
      // survives reloads and shows up in the sidebar. Skip gracefully if the
      // session could not be created above.
      if (activeSessionId) {
        try {
          const saved = await api.saveStreamedTurn(activeSessionId, content, resp.response);
          if (onSessionChange && saved?.session) {
            onSessionChange(saved.session);
          }
        } catch (saveErr) {
          console.error('Failed to persist quick-chat turn', saveErr);
        }
      }
    } catch (err) {
      console.error('Quick chat failed', err);
      const errText = err instanceof Error ? err.message : 'Quick chat failed';
      setMessages((prev) => [...prev, {
        id: generateUUID(),
        content: `❌ ${errText}`,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const showEmptyState = messages.length === 0 && !isLoading

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {showEmptyState ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-6">
          <div className="text-center text-2xl font-semibold text-gray-300 select-none">What can I help you find today?</div>
          <div className="w-full max-w-2xl px-4">
            <ChatInput onSendMessage={sendMessage} disabled={isLoading} placeholder="Ask anything…" onOpenSettings={()=>setShowSettings(true)} />
          </div>
        </div>
      ) : (
        <>
          <ConversationPage messages={messages} isLoading={isLoading} className="flex-1 overflow-y-auto" />
          <div className="flex-shrink-0">
            <ChatInput onSendMessage={sendMessage} disabled={isLoading} placeholder="Ask anything…" onOpenSettings={()=>setShowSettings(true)} />
          </div>
        </>
      )}
      {showSettings && (
        <ChatSettingsModal
          onClose={()=>setShowSettings(false)}
          options={[
            { type:'dropdown', label:'LLM model', value:selectedModel, setter:setSelectedModel, options:generationModels.map(m=>({value:m,label:m})) }
          ]}
        />
      )}
    </div>
  );
} 