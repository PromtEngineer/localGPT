"use client";
import React, { SelectHTMLAttributes } from 'react';

export function GlassSelect(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={`w-full rounded bg-white/5 hover:bg-white/10 focus:bg-white/10 px-2 py-1 text-sm font-sans text-white outline-none focus:ring-2 focus:ring-white/20 transition ${props.className || ''}`}
    >
      {props.children}
    </select>
  );
} 