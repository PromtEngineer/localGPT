"use client"

import * as React from "react"
import { useState, useEffect } from "react"
import { Plus, MessageSquare, MoreVertical } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChatSession, chatAPI } from "@/lib/api"

interface SessionSidebarRef {
  refreshSessions: () => Promise<void>
}

interface SessionSidebarProps {
  currentSessionId?: string
  onSessionSelect: (sessionId: string) => void
  onNewSession: () => void
  onSessionDelete?: (sessionId: string) => void
  onSessionCreated?: (ref: SessionSidebarRef) => void
  className?: string
}

export function SessionSidebar({
  currentSessionId,
  onSessionSelect,
  onNewSession,
  onSessionDelete,
  onSessionCreated,
  className = ""
}: SessionSidebarProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null)

  // Load sessions on mount
  useEffect(() => {
    loadSessions()
  }, [])

  const loadSessions = React.useCallback(async () => {
    try {
      setError(null)
      const response = await chatAPI.getSessions()
      setSessions(response.sessions)
    } catch (error) {
      console.error('Failed to load sessions:', error)
      setError('Failed to load sessions')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleNewSession = () => {
    // Don't create session immediately - just trigger empty state
    onNewSession()
  }

  // Refresh sessions when a new session is created
  const refreshSessions = React.useCallback(async () => {
    await loadSessions()
  }, [loadSessions])

  // Expose refresh function to parent
  React.useEffect(() => {
    if (onSessionCreated) {
      onSessionCreated({ refreshSessions })
    }
  }, [onSessionCreated, refreshSessions])

  const handleDeleteSession = async (sessionId: string, event: React.MouseEvent) => {
    event.stopPropagation() // Prevent session selection when clicking delete
    
    if (!confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
      return
    }

    try {
      await chatAPI.deleteSession(sessionId)
      setSessions(prev => prev.filter(s => s.id !== sessionId))
      
      // If the deleted session was currently selected, notify parent
      if (currentSessionId === sessionId && onSessionDelete) {
        onSessionDelete(sessionId)
      }
    } catch (error) {
      console.error('Failed to delete session:', error)
      setError('Failed to delete session')
    }
  }

  const handleRenameSession = async (sessionId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    const current = sessions.find(s => s.id === sessionId);
    const newTitle = prompt('Enter new title', current?.title || '');
    if (!newTitle || newTitle.trim() === '' || newTitle === current?.title) {
      return;
    }
    try {
      const result = await chatAPI.renameSession(sessionId, newTitle.trim());
      // Update local state with new session data
      setSessions(prev => prev.map(s => s.id === sessionId ? result.session : s));
      // If this is the currently open session, notify parent to refresh
      if (currentSessionId === sessionId && onSessionSelect) {
        onSessionSelect(sessionId);
      }
      setMenuOpenId(null);
    } catch (error) {
      console.error('Failed to rename session:', error);
      setError('Failed to rename session');
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60)
    
    if (diffInHours < 24) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } else if (diffInHours < 24 * 7) {
      return date.toLocaleDateString([], { weekday: 'short' })
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
    }
  }

  const truncateTitle = (title: string, maxLength: number = 25) => {
    return title.length > maxLength ? title.substring(0, maxLength) + '...' : title
  }

  return (
    <div className={`w-64 h-full min-h-0 bg-black border-r border-gray-800 flex flex-col ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-white">Chats</h2>
          <Button
            onClick={handleNewSession}
            size="sm"
            className="h-8 w-8 p-0 bg-gray-700 hover:bg-gray-600 text-white"
            title="New Chat"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Sessions List */}
      <ScrollArea className="flex-1 min-h-0 overflow-y-auto">
        <div className="p-2">
          {error && (
            <div className="mb-4 p-3 bg-red-900 text-red-200 text-sm rounded-lg">
              {error}
              <Button
                onClick={loadSessions}
                size="sm"
                className="ml-2 h-6 px-2 text-xs bg-red-800 hover:bg-red-700"
              >
                Retry
              </Button>
            </div>
          )}

          {isLoading ? (
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-12 bg-gray-900 rounded-lg animate-pulse" />
              ))}
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No conversations yet</p>
              <p className="text-xs mt-1">Start a new chat to begin</p>
            </div>
          ) : (
            <div className="space-y-px">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  className={`relative group pl-1 rounded transition-colors ${
                    currentSessionId === session.id
                      ? 'bg-gray-700/60 text-white border-l-2 border-white'
                      : 'hover:bg-gray-800 text-gray-300'
                  }`}
                >
                  <button
                    onClick={() => onSessionSelect(session.id)}
                    className="w-full pl-3 pr-8 py-2 text-left text-sm"
                  >
                    <p className="truncate">
                      {truncateTitle(session.title)}
                    </p>
                  </button>
                  
                  {/* Overflow menu */}
                  <div className="absolute right-2 top-2 index-row-menu">
                    <button onClick={(e)=>{e.stopPropagation(); setMenuOpenId(menuOpenId===session.id?null:session.id);}} className="p-1 text-gray-400 hover:text-white opacity-0 group-hover:opacity-100 transition">
                      <MoreVertical className="w-4 h-4" />
                    </button>
                    {menuOpenId===session.id && (
                      <div className="absolute right-0 top-full mt-1 bg-black/90 backdrop-blur border border-white/10 rounded shadow-lg py-1 w-32 text-sm z-50">
                        <button onClick={(e)=>{e.stopPropagation(); onSessionSelect(session.id); setMenuOpenId(null);}} className="block w-full text-left px-4 py-2 hover:bg-white/10">Open</button>
                        <button onClick={(e)=>handleRenameSession(session.id,e)} className="block w-full text-left px-4 py-2 hover:bg-white/10">Rename</button>
                        <button onClick={(e)=>handleDeleteSession(session.id,e)} className="block w-full text-left px-4 py-2 hover:bg-white/10 text-red-400 hover:text-red-500">Delete</button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Footer with stats */}
      {sessions.length > 0 && (
        <div className="p-4 border-t border-gray-800 text-xs text-gray-400 bg-black">
          <div className="flex justify-between">
            <span>{sessions.length} conversations</span>
            <span>
              {sessions.reduce((sum, s) => sum + s.message_count, 0)} messages
            </span>
          </div>
        </div>
      )}
    </div>
  )
} 