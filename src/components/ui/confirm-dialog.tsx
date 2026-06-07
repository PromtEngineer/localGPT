"use client";
import { useCallback, useState } from 'react';

interface ConfirmDialogProps {
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({ message, onConfirm, onCancel }: ConfirmDialogProps) {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm z-[80] p-4">
      <div className="bg-gray-900 border border-white/10 rounded-xl p-6 w-full max-w-sm text-white shadow-2xl space-y-4">
        <p className="text-sm text-gray-200 whitespace-pre-wrap leading-relaxed">{message}</p>
        <div className="flex justify-end gap-3">
          <button onClick={onCancel} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm transition-colors">
            Cancel
          </button>
          <button onClick={onConfirm} className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 text-sm transition-colors">
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}

interface AlertDialogProps {
  message: string;
  onClose: () => void;
}

export function AlertDialog({ message, onClose }: AlertDialogProps) {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm z-[80] p-4">
      <div className="bg-gray-900 border border-white/10 rounded-xl p-6 w-full max-w-sm text-white shadow-2xl space-y-4">
        <p className="text-sm text-gray-200 whitespace-pre-wrap leading-relaxed">{message}</p>
        <div className="flex justify-end">
          <button onClick={onClose} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm transition-colors">
            OK
          </button>
        </div>
      </div>
    </div>
  );
}

interface PromptDialogProps {
  message: string;
  defaultValue?: string;
  onConfirm: (value: string) => void;
  onCancel: () => void;
}

export function PromptDialog({ message, defaultValue = '', onConfirm, onCancel }: PromptDialogProps) {
  const [value, setValue] = useState(defaultValue);
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm z-[80] p-4">
      <div className="bg-gray-900 border border-white/10 rounded-xl p-6 w-full max-w-sm text-white shadow-2xl space-y-4">
        <p className="text-sm text-gray-200">{message}</p>
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') onConfirm(value);
            if (e.key === 'Escape') onCancel();
          }}
          autoFocus
          className="w-full px-3 py-2 bg-black/40 border border-white/20 rounded text-sm focus:outline-none focus:border-white/40"
        />
        <div className="flex justify-end gap-3">
          <button onClick={onCancel} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm transition-colors">
            Cancel
          </button>
          <button onClick={() => onConfirm(value)} className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 text-sm transition-colors">
            OK
          </button>
        </div>
      </div>
    </div>
  );
}

export function useConfirm() {
  const [state, setState] = useState<{ message: string; resolve: (v: boolean) => void } | null>(null);

  const showConfirm = useCallback((message: string): Promise<boolean> => {
    return new Promise((resolve) => setState({ message, resolve }));
  }, []);

  const dialog = state ? (
    <ConfirmDialog
      message={state.message}
      onConfirm={() => { state.resolve(true); setState(null); }}
      onCancel={() => { state.resolve(false); setState(null); }}
    />
  ) : null;

  return { showConfirm, dialog };
}

export function useAlert() {
  const [state, setState] = useState<{ message: string; resolve: () => void } | null>(null);

  const showAlert = useCallback((message: string): Promise<void> => {
    return new Promise((resolve) => setState({ message, resolve }));
  }, []);

  const dialog = state ? (
    <AlertDialog
      message={state.message}
      onClose={() => { state.resolve(); setState(null); }}
    />
  ) : null;

  return { showAlert, dialog };
}

export function usePrompt() {
  const [state, setState] = useState<{
    message: string;
    defaultValue: string;
    resolve: (v: string | null) => void;
  } | null>(null);

  const showPrompt = useCallback((message: string, defaultValue = ''): Promise<string | null> => {
    return new Promise((resolve) => setState({ message, defaultValue, resolve }));
  }, []);

  const dialog = state ? (
    <PromptDialog
      message={state.message}
      defaultValue={state.defaultValue}
      onConfirm={(value) => { state.resolve(value); setState(null); }}
      onCancel={() => { state.resolve(null); setState(null); }}
    />
  ) : null;

  return { showPrompt, dialog };
}
