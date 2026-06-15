'use client';

import { useState } from 'react';
import type { ActivityStage } from '@/lib/activity-trace';

/**
 * Compact, expandable activity trace for a completed answer — a collapsed
 * "Activity · N/M steps" summary that opens to the per-stage list. Deliberately
 * minimal (an observability strip, not an agent builder).
 */
export function ActivityTrace({ stages }: { stages?: ActivityStage[] }) {
  const [open, setOpen] = useState(false);
  if (!stages || stages.length === 0) return null;
  const done = stages.filter((s) => s.status === 'done').length;

  return (
    <div className="text-xs text-gray-400">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 hover:text-gray-200"
        aria-expanded={open}
      >
        <span aria-hidden>{open ? '▾' : '▸'}</span>
        <span>Activity · {done}/{stages.length} steps</span>
      </button>
      {open && (
        <ul className="mt-1 ml-3 space-y-0.5">
          {stages.map((s) => (
            <li key={s.key} className="flex items-center gap-2">
              <span aria-hidden>{s.status === 'done' ? '✓' : '◌'}</span>
              <span>{s.label}</span>
              {s.detail && <span className="text-gray-500">— {s.detail}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
