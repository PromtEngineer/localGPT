"use client";
import React from 'react';

interface Props {
  title: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export function AccordionGroup({ title, children, defaultOpen }: Props) {
  return (
    <details open={defaultOpen} className="border-t border-white/10 py-4 group">
      <summary className="cursor-pointer select-none list-none text-xs uppercase tracking-wide text-gray-400 mb-3 flex items-center gap-2">
        {title}
        <svg
          className="w-3 h-3 text-gray-400 ml-auto transition-transform group-open:rotate-90"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M6 6l6 4-6 4V6z" />
        </svg>
      </summary>
      <div className="space-y-4 pl-1">{children}</div>
    </details>
  );
} 