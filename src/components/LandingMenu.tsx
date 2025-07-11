"use client";

import React from 'react';

interface Props {
  onSelect: (mode: 'INDEX' | 'CHAT_EXISTING' | 'QUICK_CHAT') => void;
}

export function LandingMenu({ onSelect }: Props) {
  const Tile = ({ label, mode, icon }: { label: string; mode: Props["onSelect"] extends (m: infer U)=>void ? U: never; icon: React.ReactNode;}) => (
    <button
      onClick={() => onSelect(mode)}
      className="w-56 h-44 rounded-xl bg-white/5 backdrop-blur border border-white/10 hover:border-white/30 text-white flex flex-col items-center justify-center gap-2 transition"
    >
      {icon}
      <span className="text-sm font-medium">{label}</span>
    </button>
  );

  const FileIcon = (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );

  const DbIcon = (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v6c0 1.7 4 3 9 3s9-1.3 9-3V5" />
      <path d="M3 11v6c0 1.7 4 3 9 3s9-1.3 9-3v-6" />
    </svg>
  );

  const ChatIcon = (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );

  return (
    <div className="flex gap-8">
      <Tile label="Create new index" mode={"INDEX"} icon={FileIcon} />
      <Tile label="Chat with index" mode={"CHAT_EXISTING"} icon={DbIcon} />
      <Tile label="LLM Chat" mode={"QUICK_CHAT"} icon={ChatIcon} />
    </div>
  );
} 