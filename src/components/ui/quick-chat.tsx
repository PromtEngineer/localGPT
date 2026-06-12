"use client";

import React, { useState, useEffect } from 'react';
import { ChatInput } from '@/components/ui/chat-input';
import { chatAPI, ChatMessage, ChatSession, pickDefaultChatModel } from '@/lib/api';
import { ConversationPage } from '@/components/ui/conversation-page';
import { ChatSettingsModal } from '@/components/ui/chat-settings-modal';

interface QuickChatProps {
  sessionId?: string;
  onSessionChange?: (s: ChatSession) => void;
  className?: string;
}

export function QuickChat({ sessionId: externalSessionId, onSessionChange, className="" }: QuickChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(externalSessionId);
  const [generationModels, setGenerationModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [showSettings, setShowSettings] = useState(false);
  const api = chatAPI;

  // 🔄 Sync prop -> state: when sidebar selects a different session, update local session and reset chat window
  useEffect(() => {
    if (externalSessionId && externalSessionId !== sessionId) {
      setSessionId(externalSessionId);
      // Fetch existing messages for the selected session
      (async () => {
        try {
          const data = await api.getSession(externalSessionId);
          // Convert DB messages to ChatMessage format expected by UI helper
          const msgs: ChatMessage[] = data.messages.map((m) => api.convertDbMessage(m as unknown as Record<string, unknown>));
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
          setSelectedModel(pickDefaultChatModel(resp.generation_models));
        }
      }catch(e){console.warn('Failed to load models',e);}
    })();
  },[api]);

  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
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
      let responseText: string;
      if (activeSessionId) {
        // Session endpoint persists both turns; force the direct-LLM path
        // since quick chat has no documents to search.
        const resp = await api.sendSessionMessage(activeSessionId, content, {
          model: selectedModel,
          forceDirect: true,
        });
        responseText = resp.response;
        if (resp.session && onSessionChange) onSessionChange(resp.session);
      } else {
        // Session creation failed — fall back to the stateless legacy endpoint
        const history = api.messagesToHistory(messages);
        const resp = await api.sendMessage({ message: content, conversation_history: history, model: selectedModel });
        responseText = resp.response;
      }

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        content: responseText,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error('Quick chat failed', err);
      const errorMsg: ChatMessage = {
        id: crypto.randomUUID(),
        content: `❌ ${err instanceof Error ? err.message : 'Failed to send message.'}`,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMsg]);
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
