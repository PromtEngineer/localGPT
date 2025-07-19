"use client"

import {
  ChatBubble,
  ChatBubbleAvatar,
  ChatBubbleMessage
} from "@/components/ui/chat-bubble"
import { Copy, RefreshCcw } from "lucide-react"

const messages = [
  {
    id: 1,
    message: "Help me with my essay.",
    sender: "user",
  },
  {
    id: 2,
    message: "I can help you with that. What do you need help with?",
    sender: "bot",
  },
]

const actionIcons = [
  { icon: Copy, type: "Copy" },
  { icon: RefreshCcw, type: "Regenerate" },
]

export function ChatBubbleVariants() {
  return (
    <div className="max-w-md space-y-4 p-4">
      <ChatBubble variant="sent">
        <ChatBubbleAvatar fallback="US" src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=64&h=64&q=80&crop=faces&fit=crop" />
        <ChatBubbleMessage variant="sent">
          I have a question about the library.
        </ChatBubbleMessage>
      </ChatBubble>

      <ChatBubble variant="received">
        <ChatBubbleAvatar fallback="AI" src="https://images.unsplash.com/photo-1677442136019-21780ecad995?w=64&h=64&q=80&crop=faces&fit=crop"  />
        <ChatBubbleMessage>
          Sure, I&apos;d be happy to help!
        </ChatBubbleMessage>
      </ChatBubble>
    </div>
  )
}

export function ChatBubbleAiLayout() {
  return (
    <div className="max-w-md divide-y">
      {messages.map((message, index) => {
        const variant = message.sender === "user" ? "sent" : "received"
        return (
          <div key={message.id} className="py-6 first:pt-0 last:pb-0">
            <div className="flex gap-3">
              <ChatBubbleAvatar 
                src={variant === "sent" 
                  ? "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=64&h=64&q=80&crop=faces&fit=crop"
                  : "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=64&h=64&q=80&crop=faces&fit=crop"
                }
                fallback={variant === "sent" ? "US" : "L"} 
              />
              <div className="flex-1">
                {message.message}
                {message.sender === "bot" && (
                  <div className="flex gap-2 mt-2">
                    {actionIcons.map(({ icon: Icon, type }) => (
                      <button
                        key={type}
                        onClick={() => console.log(`Action ${type} clicked for message ${index}`)}
                        className="p-1 hover:bg-muted rounded-md transition-colors"
                      >
                        <Icon className="size-3" />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function ChatBubbleStates() {
  return (
    <div className="max-w-md space-y-4 p-4">
      <ChatBubble variant="received">
        <ChatBubbleAvatar fallback="L" />
        <ChatBubbleMessage isLoading />
      </ChatBubble>

      <ChatBubble variant="received">
        <ChatBubbleAvatar fallback="L" />
        <ChatBubbleMessage className="bg-destructive/10 text-destructive">
          Error processing request
        </ChatBubbleMessage>
      </ChatBubble>
    </div>
  )
} 