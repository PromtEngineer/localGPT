"use client"

import { cn } from "@/lib/utils"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

interface ChatBubbleAvatarProps {
  src?: string
  fallback?: string
  className?: string
}

export function ChatBubbleAvatar({
  src,
  fallback = "AI",
  className,
}: ChatBubbleAvatarProps) {
  return (
    <Avatar className={cn("h-8 w-8", className)}>
      {src && <AvatarImage src={src} />}
      {/* Explicit light scheme: the shadcn default (bg-muted + text-black) is
          invisible in this app's dark theme once there is no image on top. */}
      <AvatarFallback className="bg-white text-black text-xs font-medium">
        {fallback}
      </AvatarFallback>
    </Avatar>
  )
}
