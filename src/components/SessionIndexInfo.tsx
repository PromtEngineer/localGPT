import { useEffect, useRef, useState } from 'react';
import { ApiRecord, IndexJob, chatAPI } from '@/lib/api';

interface Props {
  sessionId: string;
  onClose: () => void;
}

export default function SessionIndexInfo({ sessionId, onClose }: Props) {
  const [files, setFiles] = useState<string[]>([]);
  const [indexMeta, setIndexMeta] = useState<ApiRecord | null>(null);
  const [sessionTitle, setSessionTitle] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [storedIndexId, setStoredIndexId] = useState<string | null>(null);

  const [rebuilding, setRebuilding] = useState(false);
  const [rebuildMessage, setRebuildMessage] = useState<string | null>(null);
  const [rebuildError, setRebuildError] = useState<string | null>(null);
  const [rebuildSuccess, setRebuildSuccess] = useState<string | null>(null);

  const mountedRef = useRef(true);
  useEffect(() => { return () => { mountedRef.current = false; }; }, []);

  useEffect(() => {
    (async () => {
      try {
        const data = await chatAPI.getSessionIndexes(sessionId);
        const first = data.indexes[0];
        if(first){
          setSessionTitle(first.session?.title || first.name || first.title || 'Untitled index');
          setFiles(first.documents?.map((d)=>d.filename) || []);
          setIndexMeta(first.metadata || {});
          setStoredIndexId(first.index_id || first.id || null);
        } else {
          setError('No indexes linked to this chat');
        }
      } catch (e: unknown){ setError(e instanceof Error ? e.message : 'Failed to load'); }
      finally{ setLoading(false);}
    })();
  }, [sessionId]);

  const meta = indexMeta || {};
  const hasMetadata = Object.keys(meta).length > 0;
  const isInferredMetadata = meta.metadata_source === 'lancedb_inspection';
  const indexStatus = typeof meta.status === 'string' ? meta.status : undefined;
  const textValue = (value: unknown) => (
    typeof value === 'string' || typeof value === 'number' ? String(value) : ''
  );
  const numberValue = (value: unknown) => (typeof value === 'number' ? value : undefined);
  const dateValue = (value: unknown) => (
    typeof value === 'string' || typeof value === 'number' || value instanceof Date ? value : undefined
  );
  const stringArrayValue = (value: unknown) => (
    Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
  );

  const getStatusMessage = () => {
    if (!hasMetadata) {
      return {
        type: 'warning',
        title: '⚠️ No Configuration Data',
        message: 'This index was created before metadata tracking was implemented. Configuration details are not available.'
      };
    }
    if (indexStatus === 'incomplete') {
      return {
        type: 'error',
        title: '❌ Index Incomplete',
        message: textValue(meta.issue) || 'The index appears to be incomplete or was never properly built.'
      };
    }
    if (indexStatus === 'empty') {
      return {
        type: 'error',
        title: '❌ Index Empty',
        message: 'The vector table exists but contains no data. The index may need to be rebuilt.'
      };
    }
    if (indexStatus === 'legacy') {
      return {
        type: 'warning',
        title: '⚠️ Legacy Index',
        message: textValue(meta.issue) || 'This index was created before metadata tracking was implemented. Configuration details are not available.'
      };
    }
    if (isInferredMetadata) {
      return {
        type: 'info',
        title: '🔍 Metadata Inferred',
        message: 'This metadata was inferred from the vector database structure. Some configuration details may be incomplete.'
      };
    }
    if (indexStatus === 'functional') {
      const hasCompleteConfig = meta.chunk_size &&
                               meta.chunk_overlap !== undefined &&
                               meta.retrieval_mode &&
                               meta.embedding_model;
      if (meta.inspection_limitation && !hasCompleteConfig) {
        return {
          type: 'info',
          title: '🔍 Limited Configuration Data',
          message: 'This index is functional but detailed configuration inspection requires direct RAG system access. Basic information is shown below.'
        };
      }
      return null;
    }
    return null;
  };

  const handleRebuild = async (forceReindex: boolean) => {
    if (!storedIndexId) return;
    setRebuilding(true);
    setRebuildError(null);
    setRebuildSuccess(null);
    setRebuildMessage('Running preflight check…');
    try {
      const options = {
        latechunk: typeof meta.latechunk === 'boolean' ? meta.latechunk : false,
        doclingChunk: typeof meta.docling_chunk === 'boolean' ? meta.docling_chunk : true,
        chunkSize: typeof meta.chunk_size === 'number' ? meta.chunk_size : 512,
        chunkOverlap: typeof meta.chunk_overlap === 'number' ? meta.chunk_overlap : 64,
        retrievalMode: typeof meta.retrieval_mode === 'string' ? meta.retrieval_mode : 'hybrid',
        windowSize: typeof meta.window_size === 'number' ? meta.window_size : 2,
        enableEnrich: typeof meta.enable_enrich === 'boolean' ? meta.enable_enrich : false,
        embeddingModel: typeof meta.embedding_model === 'string' ? meta.embedding_model : undefined,
        enrichModel: typeof meta.enrich_model === 'string' ? meta.enrich_model : undefined,
        overviewModel: typeof meta.overview_model === 'string' ? meta.overview_model : undefined,
        batchSizeEmbed: typeof meta.batch_size_embed === 'number' ? meta.batch_size_embed : 8,
        batchSizeEnrich: typeof meta.batch_size_enrich === 'number' ? meta.batch_size_enrich : 4,
        forceReindex,
      };
      const preflight = await chatAPI.preflightIndexBuild(storedIndexId, options);
      if (!preflight.ok) {
        throw new Error(preflight.errors.join(' ') || 'Preflight check failed.');
      }
      const started = await chatAPI.startIndexBuild(storedIndexId, options);
      if (!mountedRef.current) return;
      setRebuildMessage(`Build started — polling for completion…`);
      let job: IndexJob;
      while (mountedRef.current) {
        job = await chatAPI.getIndexJob(started.job_id);
        if (!mountedRef.current) break;
        setRebuildMessage(job.message || 'Rebuilding…');
        if (job.status === 'completed') break;
        if (job.status === 'failed') throw new Error(job.error || 'Rebuild failed.');
        if (job.status === 'cancelled') throw new Error('Rebuild was cancelled.');
        await new Promise(r => setTimeout(r, 1500));
      }
      if (!mountedRef.current) return;
      const data = await chatAPI.getSessionIndexes(sessionId);
      const first = data.indexes[0];
      if (first) {
        setSessionTitle(first.session?.title || first.name || first.title || 'Untitled index');
        setFiles(first.documents?.map((d: { filename: string }) => d.filename) || []);
        setIndexMeta(first.metadata || {});
      }
      setRebuildSuccess('Index rebuilt successfully.');
    } catch (e: unknown) {
      if (mountedRef.current) {
        setRebuildError(e instanceof Error ? e.message : 'Rebuild failed.');
      }
    } finally {
      if (mountedRef.current) {
        setRebuilding(false);
        setRebuildMessage(null);
      }
    }
  };

  const showRebuildButton = storedIndexId && !rebuilding && (
    indexStatus === 'incomplete' || indexStatus === 'empty' || indexStatus === 'legacy'
  );
  const forceForStatus = indexStatus === 'incomplete' || indexStatus === 'empty';

  const statusMessage = getStatusMessage();

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-50 p-4">
      <div className="relative bg-white/5 backdrop-blur rounded-xl p-8 w-full max-w-2xl text-white space-y-6 overflow-y-auto max-h-full">
        <h2 className="text-lg font-semibold">Index details</h2>

        {loading && <p className="text-sm text-gray-300">Loading…</p>}
        {error && <p className="text-sm text-red-400">{error}</p>}

        {rebuildError && (
          <div className="rounded-lg p-3 bg-red-900/20 border border-red-600/30">
            <p className="text-sm text-red-300">{rebuildError}</p>
          </div>
        )}
        {rebuildSuccess && (
          <div className="rounded-lg p-3 bg-green-900/20 border border-green-600/30">
            <p className="text-sm text-green-300">{rebuildSuccess}</p>
          </div>
        )}
        {rebuilding && rebuildMessage && (
          <div className="rounded-lg p-3 bg-blue-900/20 border border-blue-600/30 flex items-center gap-3">
            <div className="w-4 h-4 border-2 border-blue-400/40 border-t-blue-400 rounded-full animate-spin shrink-0" />
            <p className="text-sm text-blue-300">{rebuildMessage}</p>
          </div>
        )}

        {(!loading && !error) && (
          <>
            <div>
              <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Name</span>
              <p className="text-sm">{sessionTitle}</p>
            </div>

            {statusMessage && (
              <div className={`rounded-lg p-4 ${
                statusMessage.type === 'error' ? 'bg-red-900/20 border border-red-600/30' :
                statusMessage.type === 'warning' ? 'bg-yellow-900/20 border border-yellow-600/30' :
                'bg-blue-900/20 border border-blue-600/30'
              }`}>
                <p className={`text-sm font-medium mb-1 ${
                  statusMessage.type === 'error' ? 'text-red-200' :
                  statusMessage.type === 'warning' ? 'text-yellow-200' :
                  'text-blue-200'
                }`}>
                  {statusMessage.title}
                </p>
                <p className={`text-sm ${
                  statusMessage.type === 'error' ? 'text-red-300' :
                  statusMessage.type === 'warning' ? 'text-yellow-300' :
                  'text-blue-300'
                }`}>
                  {statusMessage.message}
                </p>
              </div>
            )}

            {hasMetadata && (indexStatus === 'functional' || indexStatus === 'created' || !indexStatus) && (
              <>
                <div className="grid grid-cols-2 gap-4">
                  {Boolean(meta.embedding_model || meta.embedding_model_inferred) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Embedding model</span>
                      <p className="text-sm break-words">
                        {textValue(meta.embedding_model || meta.embedding_model_inferred)}
                        {Boolean(meta.embedding_model_inferred) && <span className="text-gray-400"> (inferred)</span>}
                      </p>
                    </div>
                  )}
                  {Boolean(meta.retrieval_mode || meta.retrieval_mode_inferred) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Retrieval mode</span>
                      <p className="text-sm capitalize">
                        {textValue(meta.retrieval_mode || meta.retrieval_mode_inferred)}
                        {Boolean(meta.retrieval_mode_inferred) && <span className="text-gray-400"> (inferred)</span>}
                      </p>
                    </div>
                  )}
                  {numberValue(meta.vector_dimensions) !== undefined && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Vector dimensions</span>
                      <p className="text-sm">{numberValue(meta.vector_dimensions)}</p>
                    </div>
                  )}
                  {numberValue(meta.total_chunks) !== undefined && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Total chunks</span>
                      <p className="text-sm">{numberValue(meta.total_chunks)?.toLocaleString()}</p>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {(typeof meta.chunk_size==='number' || Boolean(meta.chunk_size_inferred)) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Chunk size</span>
                      <p className="text-sm">
                        {typeof meta.chunk_size==='number' ? `${meta.chunk_size} tokens` : textValue(meta.chunk_size_inferred)}
                        {Boolean(meta.chunk_size_inferred) && <span className="text-gray-400"> (estimated)</span>}
                      </p>
                    </div>
                  )}
                  {typeof meta.chunk_overlap==='number' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Chunk overlap</span>
                      <p className="text-sm">{meta.chunk_overlap} tokens</p>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {typeof meta.window_size==='number' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Context window</span>
                      <p className="text-sm">{meta.window_size}</p>
                    </div>
                  )}
                  {typeof meta.enable_enrich==='boolean' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Contextual enrichment</span>
                      <p className="text-sm">{meta.enable_enrich ? 'Enabled' : 'Disabled'}</p>
                    </div>
                  )}
                  {Boolean(meta.has_contextual_enrichment) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Contextual enrichment</span>
                      <p className="text-sm">Detected</p>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {typeof meta.latechunk==='boolean' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Late-chunk vectors</span>
                      <p className="text-sm">{meta.latechunk ? 'Enabled' : 'Disabled'}</p>
                    </div>
                  )}
                  {typeof meta.docling_chunk==='boolean' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">High-recall chunking</span>
                      <p className="text-sm">{meta.docling_chunk ? 'Enabled' : 'Disabled'}</p>
                    </div>
                  )}
                  {Boolean(meta.has_fts_index) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Full-text search</span>
                      <p className="text-sm">Available</p>
                    </div>
                  )}
                  {Boolean(meta.has_document_structure) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Document structure</span>
                      <p className="text-sm">Organized</p>
                    </div>
                  )}
                </div>

                {Boolean(meta.enrich_model || meta.overview_model) && (
                  <>
                    <div className="border-t border-white/10 pt-4">
                      <h3 className="text-sm font-medium text-gray-300 mb-3">LLM Models</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {Boolean(meta.enrich_model) && (
                          <div>
                            <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Enrichment LLM</span>
                            <p className="text-sm break-words">{textValue(meta.enrich_model)}</p>
                          </div>
                        )}
                        {Boolean(meta.overview_model) && (
                          <div>
                            <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Overview LLM</span>
                            <p className="text-sm break-words">{textValue(meta.overview_model)}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}

                {(typeof meta.batch_size_embed==='number' || typeof meta.batch_size_enrich==='number') && (
                  <>
                    <div className="border-t border-white/10 pt-4">
                      <h3 className="text-sm font-medium text-gray-300 mb-3">Batch Configuration</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {typeof meta.batch_size_embed==='number' && (
                          <div>
                            <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Embedding batch size</span>
                            <p className="text-sm">{meta.batch_size_embed}</p>
                          </div>
                        )}
                        {typeof meta.batch_size_enrich==='number' && (
                          <div>
                            <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Enrichment batch size</span>
                            <p className="text-sm">{meta.batch_size_enrich}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}

                {isInferredMetadata && dateValue(meta.metadata_inferred_at) && (
                  <div className="border-t border-white/10 pt-4">
                    <h3 className="text-sm font-medium text-gray-300 mb-3">Metadata Information</h3>
                    <div className="text-xs text-gray-400 space-y-1">
                      <p>Inferred at: {new Date(dateValue(meta.metadata_inferred_at)!).toLocaleString()}</p>
                      <p>Source: LanceDB table inspection</p>
                      {numberValue(meta.sample_chunk_length) !== undefined && (
                        <p>Sample chunk length: {numberValue(meta.sample_chunk_length)} characters</p>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}

            {hasMetadata && indexStatus === 'legacy' && (
              <>
                <div className="grid grid-cols-2 gap-4">
                  {typeof meta.documents_count === 'number' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Documents</span>
                      <p className="text-sm">{meta.documents_count}</p>
                    </div>
                  )}
                  {dateValue(meta.created_at) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Created</span>
                      <p className="text-sm">{new Date(dateValue(meta.created_at)!).toLocaleDateString()}</p>
                    </div>
                  )}
                  {Boolean(meta.vector_table_name) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Vector table</span>
                      <p className="text-gray-400 text-xs break-all">{textValue(meta.vector_table_name)}</p>
                    </div>
                  )}
                </div>
                {meta.note && (
                  <div className="border-t border-white/10 pt-4">
                    <h3 className="text-sm font-medium text-gray-300 mb-3">Technical Note</h3>
                    <p className="text-xs text-gray-400">{textValue(meta.note)}</p>
                  </div>
                )}
              </>
            )}

            {indexStatus === 'incomplete' && stringArrayValue(meta.available_tables).length > 0 && (
              <div className="border-t border-white/10 pt-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Debug Information</h3>
                <div className="text-xs text-gray-400 space-y-1">
                  <p>Expected table: {textValue(meta.vector_table_expected)}</p>
                  <p>Available tables: {stringArrayValue(meta.available_tables).join(', ') || 'None'}</p>
                </div>
              </div>
            )}

            <div className="border-t border-white/10 pt-4">
              <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Files ({files.length})</span>
              <ul className="list-disc list-inside space-y-1 text-sm max-h-32 overflow-y-auto">
                {files.map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
            </div>
          </>
        )}

        <div className="flex items-center justify-between pt-4 border-t border-white/10">
          <div className="flex gap-2">
            {showRebuildButton && (
              <button
                onClick={() => handleRebuild(forceForStatus)}
                className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 text-sm transition-colors"
              >
                {forceForStatus ? 'Force rebuild' : 'Rebuild index'}
              </button>
            )}
            {rebuilding && (
              <span className="text-xs text-blue-300 self-center">Rebuilding in progress…</span>
            )}
          </div>
          <button onClick={onClose} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm">Close</button>
        </div>
      </div>
    </div>
  );
}
