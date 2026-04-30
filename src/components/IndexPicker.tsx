import { useEffect, useRef, useState } from 'react';
import { ApiRecord, chatAPI, IndexSummary } from '@/lib/api';

interface Props {
  onSelect: (indexId: string) => void;
  onClose: () => void;
}

export default function IndexPicker({ onSelect, onClose }: Props) {
  const [indexes, setIndexes] = useState<IndexSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [busyId, setBusyId] = useState<string | null>(null);
  const [uploadTargetId, setUploadTargetId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await chatAPI.listIndexes();
        setIndexes(data.indexes);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Failed to load indexes');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const filtered = indexes.filter(i => (i.name || '').toLowerCase().includes(search.toLowerCase()));
  const indexId = (idx: IndexSummary) => idx.id || idx.index_id || '';

  const buildOptions = (idx: IndexSummary) => {
    const meta = (idx.metadata || {}) as ApiRecord;
    return {
      latechunk: typeof meta.latechunk === 'boolean' ? meta.latechunk : false,
      doclingChunk: typeof meta.docling_chunk === 'boolean' ? meta.docling_chunk : true,
      chunkSize: typeof meta.chunk_size === 'number' ? meta.chunk_size : 512,
      chunkOverlap: typeof meta.chunk_overlap === 'number' ? meta.chunk_overlap : 64,
      retrievalMode: typeof meta.retrieval_mode === 'string' ? meta.retrieval_mode : 'hybrid',
      windowSize: typeof meta.window_size === 'number' ? meta.window_size : 2,
      enableEnrich: typeof meta.enable_enrich === 'boolean' ? meta.enable_enrich : true,
      embeddingModel: typeof meta.embedding_model === 'string' ? meta.embedding_model : undefined,
      enrichModel: typeof meta.enrich_model === 'string' ? meta.enrich_model : undefined,
      overviewModel: typeof meta.overview_model === 'string' ? meta.overview_model : undefined,
      batchSizeEmbed: typeof meta.batch_size_embed === 'number' ? meta.batch_size_embed : 50,
      batchSizeEnrich: typeof meta.batch_size_enrich === 'number' ? meta.batch_size_enrich : 25,
      forceReindex: true,
    };
  };

  async function handleDelete(idxId: string, name: string) {
    if (!confirm(`Delete index "${name}"? This cannot be undone.`)) return;
    try {
      await chatAPI.deleteIndex(idxId);
      setIndexes(prev => prev.filter(i => (i.id || i.index_id)!==idxId));
      setMenuOpenId(null);
    } catch (e: unknown){
      alert(e instanceof Error ? e.message : 'Failed to delete index');
    }
  }

  async function handleRebuild(idx: IndexSummary) {
    const id = indexId(idx);
    if (!id) return;
    const docCount = idx.documents?.length || 0;
    if (docCount === 0) {
      alert('This index has no files. Add files first, then rebuild.');
      return;
    }
    if (!confirm(`Force rebuild "${idx.name || 'Untitled index'}" from ${docCount} file(s)?`)) return;
    setBusyId(id);
    setMenuOpenId(null);
    try {
      await chatAPI.buildIndex(id, buildOptions(idx));
      const data = await chatAPI.listIndexes();
      setIndexes(data.indexes);
      alert('Rebuild complete.');
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to rebuild index');
    } finally {
      setBusyId(null);
    }
  }

  function handleAddFiles(idx: IndexSummary) {
    const id = indexId(idx);
    if (!id) return;
    setUploadTargetId(id);
    setMenuOpenId(null);
    fileInputRef.current?.click();
  }

  async function handleUploadAndRebuild(files: FileList | null) {
    if (!files || !uploadTargetId) return;
    const idx = indexes.find((item) => indexId(item) === uploadTargetId);
    if (!idx) return;
    setBusyId(uploadTargetId);
    try {
      await chatAPI.uploadFilesToIndex(uploadTargetId, Array.from(files));
      await chatAPI.buildIndex(uploadTargetId, buildOptions(idx));
      const data = await chatAPI.listIndexes();
      setIndexes(data.indexes);
      alert('Files added and index rebuilt.');
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to add files and rebuild index');
    } finally {
      setBusyId(null);
      setUploadTargetId(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  useEffect(() => {
    function handleOutside(e: MouseEvent) {
      if ((e.target as Element).closest('.index-row-menu') === null) {
        setMenuOpenId(null);
      }
    }
    if (menuOpenId) {
      document.addEventListener('click', handleOutside);
    }
    return () => document.removeEventListener('click', handleOutside);
  }, [menuOpenId]);

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white/5 backdrop-blur rounded-xl w-full max-w-xl max-h-full overflow-y-auto p-6 text-white space-y-6">
        <h2 className="text-lg font-semibold">Select an index</h2>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf,.docx,.doc,.html,.htm,.md,.txt"
          multiple
          className="hidden"
          onChange={(e)=>handleUploadAndRebuild(e.target.files)}
        />
        <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search…" className="w-full px-3 py-2 rounded bg-black/30 border border-white/20 focus:outline-none" />
        {busyId && <p className="text-xs text-green-300">Rebuilding index… keep both backend terminals running.</p>}
        {loading && <p className="text-sm text-gray-300">Loading…</p>}
        {error && <p className="text-sm text-red-400">{error}</p>}
        {!loading && !error && (
          <ul className="space-y-2">
            {filtered.map(idx => (
              <li key={indexId(idx)}>
                <div className="relative group">
                  <button disabled={busyId===indexId(idx)} onClick={()=>onSelect(indexId(idx))} className="w-full px-4 py-3 bg-white/10 hover:bg-white/20 rounded transition flex justify-between items-center pr-10 disabled:opacity-50">
                    <span className="font-medium truncate max-w-[60%]">{idx.name}</span>
                    <span className="text-xs text-gray-400">{busyId===indexId(idx) ? 'rebuilding…' : `${idx.documents?.length || 0} files`}</span>
                  </button>

                  <button disabled={busyId===indexId(idx)} onClick={(e)=>{e.stopPropagation(); const id = indexId(idx); setMenuOpenId(menuOpenId===id?null:id);}} title="More actions" className="absolute right-4 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 text-gray-400 hover:text-white transition text-lg leading-none font-bold disabled:opacity-40">
                    …
                  </button>

                  {menuOpenId===indexId(idx) && (
                    <div className="index-row-menu absolute right-0 top-full mt-1 bg-black/80 backdrop-blur border border-white/10 rounded shadow-lg py-1 w-44 text-sm z-50">
                      <button onClick={()=>{onSelect(indexId(idx)); setMenuOpenId(null);}} className="block w-full text-left px-4 py-2 hover:bg-white/10">Open</button>
                      <button onClick={()=>handleAddFiles(idx)} className="block w-full text-left px-4 py-2 hover:bg-white/10">Add files + rebuild</button>
                      <button onClick={()=>handleRebuild(idx)} className="block w-full text-left px-4 py-2 hover:bg-white/10">Force rebuild</button>
                      <button onClick={()=>handleDelete(indexId(idx), idx.name || 'Untitled index')} className="block w-full text-left px-4 py-2 hover:bg-white/10 text-red-400 hover:text-red-500">Delete</button>
                    </div>
                  )}
                </div>
              </li>
            ))}
            {filtered.length===0 && <p className="text-sm text-gray-400">No indexes found.</p>}
          </ul>
        )}
        <div className="pt-4 border-t border-white/10 flex justify-end">
          <button onClick={onClose} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm">Close</button>
        </div>
      </div>
    </div>
  );
} 
