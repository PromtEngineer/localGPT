// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-nocheck
'use client'

import dynamic from 'next/dynamic'
import React, { useMemo } from 'react'
import remarkGfm from 'remark-gfm'

// Dynamically import react-markdown to avoid SSR issues
const ReactMarkdown: any = dynamic(() => import('react-markdown') as any, { ssr: false })

interface MarkdownProps {
  text: string
  className?: string
}

export default function Markdown({ text, className = '' }: MarkdownProps) {
  const plugins = useMemo(() => [remarkGfm], [])
  return (
    <div className={`prose prose-invert max-w-none ${className}`}>
      {/* @ts-ignore – react-markdown type doesn't recognise remarkPlugins array */}
    <ReactMarkdown
        remarkPlugins={plugins}
        components={{
          a: ({ node, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
        }}
    >
      {text}
    </ReactMarkdown>
    </div>
  )
} 