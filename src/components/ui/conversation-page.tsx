"use client"

import * as React from "react"
import { useRef, useEffect, useState } from "react"
import {
  ChatBubbleAvatar,
} from "@/components/ui/chat-bubble"
import { Copy, RefreshCcw, ThumbsUp, ThumbsDown, Volume2, MoreHorizontal, ChevronDown, Loader2, CheckCircle, XOctagon } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChatMessage } from "@/lib/api"
import { cn } from "@/lib/utils"
import Markdown from "@/components/Markdown"
import { normalizeWhitespace } from "@/utils/textNormalization"

interface ConversationPageProps {
  messages: ChatMessage[]
  isLoading?: boolean
  className?: string
  onAction?: (action: string, messageId: string, messageContent: string) => void
}

const actionIcons = [
  { icon: Copy, type: "Copy", action: "copy" },
  { icon: ThumbsUp, type: "Like", action: "like" },
  { icon: ThumbsDown, type: "Dislike", action: "dislike" },
  { icon: Volume2, type: "Speak", action: "speak" },
  { icon: RefreshCcw, type: "Regenerate", action: "regenerate" },
  { icon: MoreHorizontal, type: "More", action: "more" },
]

// Citation block toggle component
function Citation({doc, idx}: {doc:any, idx:number}){
  const [open,setOpen]=React.useState(false);
  const preview = (doc.text||'').replace(/\s+/g,' ').trim().slice(0,160) + ((doc.text||'').length>160?'…':'');
  return (
    <div onClick={()=>setOpen(!open)} className="text-xs text-gray-300 bg-gray-900/60 rounded p-2 cursor-pointer hover:bg-gray-800 transition">
      <span className="font-semibold mr-1">[{idx+1}]</span>{open?doc.text:preview}
    </div>
  );
}

// NEW: Expandable list of citations per assistant message
function CitationsBlock({docs}:{docs:any[]}){
  const scored = docs.filter(d => d.rerank_score || d.score || d._distance)
  scored.sort((a, b) => (b.rerank_score ?? b.score ?? 1/b._distance) - (a.rerank_score ?? a.score ?? 1/a._distance))
  const [expanded, setExpanded] = useState(false);

  if (scored.length === 0) return null;

  const visibleDocs = expanded ? scored : scored.slice(0, 5);

  return (
    <div className="mt-2 text-xs text-gray-400">
      <p className="font-semibold mb-1">Sources:</p>
      <div className="grid grid-cols-1 gap-2">
        {visibleDocs.map((doc, i) => <Citation key={doc.chunk_id || i} doc={doc} idx={i} />)}
      </div>
      {scored.length > 5 && (
        <button 
          onClick={() => setExpanded(!expanded)} 
          className="text-blue-400 hover:text-blue-300 mt-2 text-xs"
        >
          {expanded ? 'Show less' : `Show ${scored.length-5} more`}
        </button>
      )}
    </div>
  );
}

function StepIcon({ status }: { status: 'pending' | 'active' | 'done' | 'error' }) {
  switch (status) {
    case 'pending':
      return <MoreHorizontal className="w-4 h-4 text-neutral-600" />
    case 'active':
      return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
    case 'done':
      return <CheckCircle className="w-4 h-4 text-green-400" />
    case 'error':
      return <XOctagon className="w-4 h-4 text-red-400" />
    default:
      return null
  }
}

const statusBorder: Record<string, string> = {
  pending: 'border-neutral-800',
  active: 'border-blue-400 animate-pulse',
  done: 'border-green-400',
  error: 'border-red-400'
}

// Component to handle <think> tokens and render them in a collapsible block
function ThinkingText({ text }: { text: string }) {
  const regex = /<think>([\s\S]*?)<\/think>/g;
  const thinkSegments: string[] = [];
  const visibleText = text.replace(regex, (_, p1) => {
    thinkSegments.push(p1.trim());
    return ""; // remove thinking content from main text
  });

  return (
    <>
      {thinkSegments.length > 0 && (
        <details className="thinking-block inline-block align-baseline mr-2" open={false}>
          <summary className="cursor-pointer text-xs text-gray-400 uppercase select-none">Thinking</summary>
          <div className="mt-1 space-y-1 text-xs text-gray-400 italic">
            {thinkSegments.map((seg, idx) => (
              <div key={idx}>{seg}</div>
            ))}
          </div>
        </details>
      )}
      {visibleText.trim() && (
        <Markdown text={normalizeWhitespace(visibleText)} className="whitespace-pre-wrap" />
      )}
    </>
  );
}

function StructuredMessageBlock({ content }: { content: Array<Record<string, any>> | { steps: any[] } }) {
  const steps: any[] = Array.isArray(content) ? content : (content as any).steps;
  // Determine if sub-query answers are present
  const hasSubAnswers = steps.some((s: any) => s.key === 'answer' && Array.isArray(s.details) && s.details.length > 0);
  // Compute the last index that has started (status !== 'pending') so we only
  // render steps that are in progress or completed. This avoids showing the
  // whole plan upfront and reveals each stage sequentially.
  const lastRevealedIdx = (() => {
    for (let i = steps.length - 1; i >= 0; i--) {
      if (steps[i].status && steps[i].status !== 'pending') {
        return i;
      }
    }
    return -1; // nothing started yet
  })();

  const visibleSteps = lastRevealedIdx >= 0 ? steps.slice(0, lastRevealedIdx + 1) : [];

  return (
    <div className="flex flex-col">
      {visibleSteps.map((step: any, index: number) => {
        if (step.key && step.label) {
          const borderCls = statusBorder[step.status] || statusBorder['pending']
          const statusClass = `timeline-card card my-1 py-2 pl-3 pr-2 bg-[#0d0d0d] rounded border-l-2 ${borderCls}`
          
          return (
            <div key={step.key} className={statusClass}>
              <div className="flex items-center gap-2 mb-1">
                <StepIcon status={step.status} />
                <span className="text-sm font-medium text-neutral-100">{step.label}</span>
              </div>
              {/* Details for each step */}
              {step.key === 'final' && step.details && typeof step.details === 'object' && !Array.isArray(step.details) ? (
                <div className="space-y-3">
                  <div className="whitespace-pre-wrap text-gray-100">
                    <ThinkingText text={normalizeWhitespace(step.details.answer)} />
                  </div>
                  {!hasSubAnswers && step.details.source_documents && step.details.source_documents.length > 0 && (
                    <CitationsBlock docs={step.details.source_documents} />
                  )}
                </div>
              ) : step.key === 'final' && step.details && typeof step.details === 'string' ? (
                <div className="whitespace-pre-wrap text-gray-100">
                  <ThinkingText text={normalizeWhitespace(step.details)} />
                </div>
              ) : Array.isArray(step.details) ? (
                step.key === 'decompose' && step.details.every((d: any)=> typeof d === 'string') ? (
                  // Render list of sub-query strings
                  <ul className="list-disc list-inside space-y-1 text-neutral-200">
                    {step.details.map((q: string, idx:number)=>(
                      <li key={idx}>{q}</li>
                    ))}
                  </ul>
                ) : (
                  // Handle array of sub-answers
                  <div className="space-y-2">
                    {step.details.map((detail: any, idx: number) => (
                      <div key={idx} className="border-l-2 border-blue-400 pl-2">
                        <div className="font-semibold">{detail.question}</div>
                        <div><ThinkingText text={normalizeWhitespace(detail.answer)} /></div>
                        {detail.source_documents && detail.source_documents.length > 0 && (
                          <CitationsBlock docs={detail.source_documents} />
                        )}
                      </div>
                    ))}
                  </div>
                )
              ) : (
                // Handle string details
                <ThinkingText text={normalizeWhitespace(step.details as string)} />
              )}
            </div>
          );
        }
        return null;
      })}
    </div>
  );
}

export function ConversationPage({ 
  messages, 
  isLoading = false,
  className = "",
  onAction
}: ConversationPageProps) {
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [showScrollButton, setShowScrollButton] = useState(false)
  const [isUserNearBottom,setIsUserNearBottom]=useState(true)

  // Track if user is near bottom so we don't interrupt manual scrolling
  useEffect(() => {
    if(isUserNearBottom){
    scrollToBottom()
    }
  }, [messages, isLoading])

  // Monitor scroll position to show/hide scroll button
  useEffect(() => {
    const scrollContainer = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]')
    if (!scrollContainer) return

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = scrollContainer
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100
      setShowScrollButton(!isNearBottom)
      setIsUserNearBottom(isNearBottom)
    }

    scrollContainer.addEventListener('scroll', handleScroll)
    handleScroll() // Check initial state

    return () => scrollContainer.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToBottom = () => {
    // Try multiple methods to ensure scrolling works
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
    
    // Fallback: scroll the container directly
    setTimeout(() => {
      if (scrollAreaRef.current) {
        const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]') || scrollAreaRef.current
        if (scrollContainer) {
          scrollContainer.scrollTop = scrollContainer.scrollHeight
        }
      }
    }, 100)
  }

  const handleAction = (action: string, messageId: string, messageContent: string) => {
    if (onAction) {
      // For structured messages, we'll just join the text parts for copy/paste
      let contentToPass: string;
      if (typeof messageContent === 'string') {
        contentToPass = messageContent;
      } else if (Array.isArray(messageContent)) {
        contentToPass = (messageContent as any[]).map((s: any) => s.text || s.answer || '').join('\n');
      } else if (messageContent && typeof messageContent === 'object' && Array.isArray((messageContent as any).steps)) {
        // For {steps: Step[]} structure
        contentToPass = (messageContent as any).steps.map((s: any) => s.label + (s.details ? (typeof s.details === 'string' ? (': ' + s.details) : '') : '')).join('\n');
      } else {
        contentToPass = '';
      }
      onAction(action, messageId, contentToPass)
      return
    }
    
    console.log(`Action ${action} clicked for message ${messageId}`)
    // Handle different actions here
    switch (action) {
      case 'copy':
        navigator.clipboard.writeText(messageContent)
        break
      case 'regenerate':
        // Regenerate AI response
        break
      case 'like':
        // Add like reaction
        break
      case 'dislike':
        // Add dislike reaction
        break
      case 'speak':
        // Text to speech
        break
      case 'more':
        // Show more options
        break
    }
  }

  return (
    <div className={`flex flex-col h-full bg-black relative overflow-hidden ${className}`}>
      <ScrollArea ref={scrollAreaRef} className="flex-1 h-full px-4 pt-4 pb-6 min-h-0">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((message) => {
            const isUser = message.sender === "user"
            
            return (
              <div key={message.id} className="w-full group">
                <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
                  {!isUser && (
                    <ChatBubbleAvatar 
                      fallback="AI" 
                      className="mt-1 flex-shrink-0 text-black"
                    />
                  )}
                  
                  <div className={`flex flex-col space-y-2 ${isUser ? 'items-end' : 'items-start'} max-w-full md:max-w-3xl`}>
                    <div
                      className={`rounded-2xl px-5 py-4 ${
                        isUser 
                          ? "bg-white text-black" 
                          : "bg-gray-800 text-gray-100"
                      }`}
                    >
                      {message.isLoading ? (
                        <div className="flex items-center space-x-2">
                          <div className="flex space-x-1">
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                          </div>
                        </div>
                      ) : (
                        <div className="whitespace-pre-wrap text-base leading-relaxed">
                          {typeof message.content === 'string' 
                              ? <ThinkingText text={normalizeWhitespace(message.content)} />
                              : <StructuredMessageBlock content={message.content} />
                          }
                        </div>
                      )}
                    </div>
                    
                    {!isUser && !message.isLoading && (
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        {actionIcons.map(({ icon: Icon, type, action }) => (
                          <button
                            key={action}
                            onClick={() => {
                              const content = typeof message.content === 'string' ? message.content : (message.content as any[]).map(s => s.text || s.answer).join('\\n');
                              handleAction(action, message.id, content)
                            }}
                            className="p-1.5 hover:bg-gray-700 rounded-md transition-colors text-gray-400 hover:text-gray-200"
                            title={type}
                          >
                            <Icon className="w-3.5 h-3.5" />
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Global citations only for plain-string messages */}
                    {(!isUser &&
                      !message.isLoading &&
                      typeof message.content === 'string' &&
                      Array.isArray((message as any).metadata?.source_documents) &&
                      (message as any).metadata.source_documents.length > 0) && (
                        <CitationsBlock docs={(message as any).metadata.source_documents} />
                    )}
                  </div>

                  {isUser && (
                    <ChatBubbleAvatar 
                      className="mt-1 flex-shrink-0 text-black"
                      src="https://i.pravatar.cc/40?u=user"
                      fallback="User"
                    />
                  )}
                </div>
              </div>
            )
          })}
          
          {/* Loading indicator for new message */}
          {isLoading && (
            <div className="w-full group">
              <div className="flex gap-3 justify-start">
                <ChatBubbleAvatar fallback="AI" className="mt-1 flex-shrink-0 text-black" />
                <div className="flex flex-col space-y-2 items-start max-w-[80%]">
                  <div className="rounded-2xl px-4 py-3 bg-gray-800 text-gray-100">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
                      )}
          
          {/* Invisible element to scroll to */}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>
      
      {/* Scroll to bottom button - only show when not at bottom */}
      {showScrollButton && (
        <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 z-10">
          <button
            onClick={scrollToBottom}
            className="p-2 bg-gray-800 border border-gray-700 rounded-full hover:bg-gray-700 transition-all duration-200 shadow-lg group animate-in fade-in slide-in-from-bottom-2"
            title="Scroll to bottom"
          >
            <ChevronDown className="w-4 h-4 text-gray-400 group-hover:text-gray-200 transition-colors" />
          </button>
        </div>
      )}
    </div>
  )
}  