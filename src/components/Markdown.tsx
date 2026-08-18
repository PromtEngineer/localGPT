'use client'

import dynamic from 'next/dynamic'
import React, { useMemo } from 'react'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'

// Dynamically import react-markdown to avoid SSR issues
const ReactMarkdown = dynamic(() => import('react-markdown'), { ssr: false })

interface MarkdownProps {
  text: string
  className?: string
}

export default function Markdown({ text, className = '' }: MarkdownProps) {
  // remark-breaks renders single newlines as <br>: model answers use them
  // as intentional line breaks, and without the plugin markdown collapses
  // them into spaces. This replaces the old whitespace-pre-wrap approach,
  // which rendered every newline TWICE (markdown paragraph + literal break).
  const plugins = useMemo(() => [remarkGfm, remarkBreaks], [])
  return (
    <div className={`prose prose-invert max-w-none prose-p:my-2 prose-headings:mt-3 prose-headings:mb-2 prose-ul:my-2 prose-ol:my-2 prose-pre:my-2 prose-hr:my-3 ${className}`}>
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
