'use client'

import dynamic from 'next/dynamic'
import React, { useMemo } from 'react'
import remarkGfm from 'remark-gfm'
import type { Options } from 'react-markdown'

// Dynamically import react-markdown to avoid SSR issues
const ReactMarkdown = dynamic<Options>(() => import('react-markdown').then((mod) => mod.default), { ssr: false })

interface MarkdownProps {
  text: string
  className?: string
}

export default function Markdown({ text, className = '' }: MarkdownProps) {
  const plugins = useMemo(() => [remarkGfm], [])
  return (
    <div className={`prose prose-invert max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={plugins}
        components={{
          a: (props) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
} 
