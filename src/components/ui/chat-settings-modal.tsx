"use client";

import { GlassToggle } from '@/components/ui/GlassToggle';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

export interface ToggleOption {
  type: 'toggle';
  label: string;
  checked: boolean;
  setter: (v: boolean) => void;
}

export interface SliderOption {
  type: 'slider';
  label: string;
  value: number;
  setter: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
}

export interface DropdownOption {
  type: 'dropdown';
  label: string;
  value: string;
  setter: (v: string) => void;
  options: { value: string; label: string }[];
}

export type SettingOption = ToggleOption | SliderOption | DropdownOption;

interface Props {
  options: SettingOption[];
  onClose: () => void;
}

const optionHelp: Record<string,string> = {
  'Query decomposition':'Breaks a complex question into sub-queries to improve recall (adds latency).',
  'Compose sub-answers':'Merges answers from decomposed sub-queries into a single response.',
  'Pruning':'Removes sentences deemed irrelevant by a lightweight model before synthesis.',
  'RAG (no-triage)':'Force retrieval on every query; disables index-selection triage.',
  'Verify answer':'Runs an extra LLM pass to self-critique the draft answer.',
  'Streaming':'Send tokens to the UI as they are generated.',
  'AI reranker':'Re-orders retrieved chunks with a cross-encoder (higher quality, more latency).',
  'Expand context window':'Adds neighbour chunks around each top chunk to provide more context.',
  'Context window size':'How many neighbour chunks to include on each side.',
  'Retrieval chunks':'Number of chunks fetched before reranking.',
  'LLM':'Select which model generates the final answer.',
  'Search type':'Choose retrieval strategy (Hybrid recommended).',
  'Reranker top chunks':'Limit how many chunks are re-ranked to speed up processing.'
};

export function ChatSettingsModal({ options, onClose }: Props) {
  const renderOption = (opt: SettingOption) => {
    switch (opt.type) {
      case 'toggle':
        return (
          <div key={opt.label} className="flex items-center justify-between">
            <span className="text-sm text-gray-300 flex items-center gap-1 whitespace-nowrap">
              {displayName(opt.label)}
              {optionHelp[displayName(opt.label)] && <InfoTooltip text={optionHelp[displayName(opt.label)]} size={12} />}
            </span>
            <GlassToggle checked={opt.checked} onChange={opt.setter} />
          </div>
        );
      
      case 'slider':
        return (
          <div key={opt.label} className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-300 flex items-center gap-1">{displayName(opt.label)}{optionHelp[displayName(opt.label)] && <InfoTooltip text={optionHelp[displayName(opt.label)]} size={12} />}</span>
              <span className="text-sm text-gray-400">
                {opt.value}{opt.unit || ''}
              </span>
            </div>
            <input
              type="range"
              min={opt.min}
              max={opt.max}
              step={opt.step || 1}
              value={opt.value}
              onChange={(e) => opt.setter(Number(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${((opt.value - opt.min) / (opt.max - opt.min)) * 100}%, #374151 ${((opt.value - opt.min) / (opt.max - opt.min)) * 100}%, #374151 100%)`
              }}
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>{opt.min}{opt.unit || ''}</span>
              <span>{opt.max}{opt.unit || ''}</span>
            </div>
          </div>
        );
      
      case 'dropdown':
        return (
          <div key={opt.label} className="space-y-2">
            <span className="text-sm text-gray-300 flex items-center gap-1">{displayName(opt.label)}{optionHelp[displayName(opt.label)] && <InfoTooltip text={optionHelp[displayName(opt.label)]} size={12} />}</span>
            <select
              value={opt.value}
              onChange={(e) => opt.setter(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {opt.options.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        );
      
      default:
        return null;
    }
  };

  const gridToggleLabels: string[] = [
    'Query decomposition',
    'Compose sub-answers',
    'Prune irrelevant sentences',
    'Always search documents', // will be displayed as RAG (no-triage)
    'Verify answer',
    'Stream phases',
  ];

  const retrievalGridLabels = ['LLM model','Search type'];

  const displayName = (label: string) => {
    if (label === 'Always search documents') return 'RAG (no-triage)';
    if (label === 'LLM model') return 'LLM';
    if (label === 'Prune irrelevant sentences') return 'Pruning';
    if (label === 'Stream phases') return 'Streaming';
    return label;
  };

  const renderOptionOrdered = (label: string) => {
    const opt = options.find(o => o.label === label);
    if (!opt) return null;
    // Clone option with display label override
    const clone = { ...opt, label: displayName(label) } as SettingOption;
    return renderOption(clone);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white/5 backdrop-blur rounded-xl w-full max-w-xl max-h-full overflow-y-auto p-6 text-white space-y-6">
        <h2 className="text-lg font-semibold mb-6">Chat Settings</h2>

        <div className="space-y-6">
          {/* High-level Settings */}
          <div>
            <h3 className="text-md font-medium text-gray-200 mb-4 flex items-center gap-1">General Settings <InfoTooltip text="High-level toggles that affect how the assistant thinks and whether it always performs RAG." /></h3>
            {/* Two-column grid for key toggles */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              {gridToggleLabels.map(renderOptionOrdered)}
            </div>
            {/* No additional general options after grid */}
          </div>

          {/* Retrieval Settings */}
          <div>
            <h3 className="text-md font-medium text-gray-200 mb-4 flex items-center gap-1">Retrieval Settings <InfoTooltip text="Configure which LLM answers and how the system searches your indexes." /></h3>
            {/* LLM + Search type grid */}
            {(() => {
              const arr: SettingOption[] = retrievalGridLabels
                .map(lbl => {
                  const opt = options.find(o=>o.label===lbl);
                  return opt ? ({...opt, label: displayName(lbl) } as SettingOption) : undefined;
                })
                .filter((o): o is SettingOption => !!o);
              return <div className="grid grid-cols-2 gap-4 mb-4">{arr.map(renderOption)}</div>;
            })()}
            {/* Sliders */}
            <div className="space-y-4">
              {options.filter(opt => ['Retrieval chunks'].includes(opt.label)).map(renderOption)}
            </div>
          </div>

          {/* Reranking Settings */}
          <div>
            <h3 className="text-md font-medium text-gray-200 mb-4 flex items-center gap-1">Reranking & Context <InfoTooltip text="Controls post-retrieval reordering, context window expansion and pruning (may add latency)." /></h3>
            <div className="space-y-4">
              {options.filter(opt => 
                ['AI reranker', 'Reranker top chunks', 'Expand context window', 'Context window size'].includes(opt.label)
              ).map(renderOption)}
            </div>
          </div>
        </div>

        <div className="flex justify-end pt-6 border-t border-white/10 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
} 