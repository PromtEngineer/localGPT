"use client";
import { useState } from 'react';
import { ModelSelect } from '@/components/ModelSelect';

interface Props {
  onClose: () => void;
}

export function IndexWizard({ onClose }: Props) {
  const [files, setFiles] = useState<FileList | null>(null);
  const [chunkSize, setChunkSize] = useState(512);
  const [chunkOverlap, setChunkOverlap] = useState(64);
  const [embeddingModel, setEmbeddingModel] = useState<string>();
  // TODO: more params

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFiles(e.target.files);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur flex items-center justify-center z-50">
      <div className="bg-gray-900 w-[600px] max-h-[90vh] overflow-auto rounded-xl p-6 text-white space-y-6">
        <h2 className="text-lg font-semibold">Create new index</h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm mb-1">Document files</label>
            <input type="file" accept="application/pdf,.docx,.doc,.html,.htm,.md,.txt" multiple onChange={handleFile} className="text-sm" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm mb-1">Chunk size</label>
              <input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(parseInt(e.target.value))}
                className="w-full bg-gray-800 rounded px-2 py-1"
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Chunk overlap</label>
              <input
                type="number"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(parseInt(e.target.value))}
                className="w-full bg-gray-800 rounded px-2 py-1"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm mb-1">Embedding model</label>
            <ModelSelect type="embedding" value={embeddingModel} onChange={setEmbeddingModel} />
          </div>
        </div>

        <div className="flex justify-end gap-3 pt-4 border-t border-white/10">
          <button onClick={onClose} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm">
            Cancel
          </button>
          <button
            disabled={!files || !embeddingModel}
            className="px-4 py-2 bg-green-600 rounded disabled:opacity-40 text-sm"
          >
            Start indexing
          </button>
        </div>
      </div>
    </div>
  );
}    