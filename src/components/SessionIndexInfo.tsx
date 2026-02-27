import { useEffect, useState } from 'react';
import { chatAPI, ChatSession } from '@/lib/api';

interface Props {
  sessionId: string;
  onClose: () => void;
}

export default function SessionIndexInfo({ sessionId, onClose }: Props) {
  const [files, setFiles] = useState<string[]>([]);
  const [indexMeta, setIndexMeta] = useState<any | null>(null);
  const [session, setSession] = useState<ChatSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await chatAPI.getSessionIndexes(sessionId);
        const first = data.indexes[0];
        if(first){
          setSession(first.session??{...first, title:first.name, model_used:first.model_used||''});
          setFiles(first.documents?.map((d:any)=>d.filename) || []);
          setIndexMeta(first.metadata || {});
        } else {
          setError('No indexes linked to this chat');
        }
      } catch (e:any){ setError(e.message||'Failed to load'); }
      finally{ setLoading(false);}
    })();
  }, [sessionId]);

  const hasMetadata = indexMeta && Object.keys(indexMeta).length > 0;
  const isInferredMetadata = indexMeta?.metadata_source === 'lancedb_inspection';
  const indexStatus = indexMeta?.status;

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
        message: indexMeta.issue || 'The index appears to be incomplete or was never properly built.'
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
        message: indexMeta.issue || 'This index was created before metadata tracking was implemented. Configuration details are not available.'
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
      // Check if we have complete configuration metadata
      const hasCompleteConfig = indexMeta.chunk_size && 
                               indexMeta.chunk_overlap !== undefined &&
                               indexMeta.retrieval_mode &&
                               indexMeta.embedding_model;
      
      // Only show limited message if we truly have limited data
      if (indexMeta.inspection_limitation && !hasCompleteConfig) {
        return {
          type: 'info',
          title: '🔍 Limited Configuration Data',
          message: 'This index is functional but detailed configuration inspection requires direct RAG system access. Basic information is shown below.'
        };
      }
      
      // Don't show any status message for functional indexes with complete metadata
      return null;
    }
    
    return null;
  };

  const statusMessage = getStatusMessage();

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-50 p-4">
      <div className="relative bg-white/5 backdrop-blur rounded-xl p-8 w-full max-w-2xl text-white space-y-6 overflow-y-auto max-h-full">
        <h2 className="text-lg font-semibold">Index details</h2>

        {loading && <p className="text-sm text-gray-300">Loading…</p>}
        {error && <p className="text-sm text-red-400">{error}</p>}

        {(!loading && !error) && (
          <>
            <div>
              <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Name</span>
              <p className="text-sm">{session?.title}</p>
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
                {/* Basic Information */}
                <div className="grid grid-cols-2 gap-4">
                  {(indexMeta.embedding_model || indexMeta.embedding_model_inferred) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Embedding model</span>
                      <p className="text-sm break-words">
                        {indexMeta.embedding_model || indexMeta.embedding_model_inferred}
                        {indexMeta.embedding_model_inferred && <span className="text-gray-400"> (inferred)</span>}
                      </p>
                    </div>
                  )}
                  {(indexMeta.retrieval_mode || indexMeta.retrieval_mode_inferred) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Retrieval mode</span>
                      <p className="text-sm capitalize">
                        {indexMeta.retrieval_mode || indexMeta.retrieval_mode_inferred}
                        {indexMeta.retrieval_mode_inferred && <span className="text-gray-400"> (inferred)</span>}
                      </p>
                    </div>
                  )}
                  {indexMeta.vector_dimensions && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Vector dimensions</span>
                      <p className="text-sm">{indexMeta.vector_dimensions}</p>
                    </div>
                  )}
                  {indexMeta.total_chunks && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Total chunks</span>
                      <p className="text-sm">{indexMeta.total_chunks.toLocaleString()}</p>
                    </div>
                  )}
                </div>

                {/* Chunk Configuration */}
                <div className="grid grid-cols-2 gap-4">
                  {(typeof indexMeta.chunk_size==='number' || indexMeta.chunk_size_inferred) && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Chunk size</span>
                      <p className="text-sm">
                        {typeof indexMeta.chunk_size==='number' ? `${indexMeta.chunk_size} tokens` : indexMeta.chunk_size_inferred}
                        {indexMeta.chunk_size_inferred && <span className="text-gray-400"> (estimated)</span>}
                      </p>
                    </div>
                  )}
                  {typeof indexMeta.chunk_overlap==='number' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Chunk overlap</span>
                      <p className="text-sm">{indexMeta.chunk_overlap} tokens</p>
                    </div>
                  )}
                </div>

                {/* Context and Features */}
                <div className="grid grid-cols-2 gap-4">
                  {typeof indexMeta.window_size==='number' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Context window</span>
                      <p className="text-sm">{indexMeta.window_size}</p>
                    </div>
                  )}
                  {typeof indexMeta.enable_enrich==='boolean' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Contextual enrichment</span>
                      <p className="text-sm">{indexMeta.enable_enrich ? '✅ Enabled' : '❌ Disabled'}</p>
                    </div>
                  )}
                  {indexMeta.has_contextual_enrichment && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Contextual enrichment</span>
                      <p className="text-sm">🔍 Detected</p>
                    </div>
                  )}
                </div>

                {/* Advanced features */}
                <div className="grid grid-cols-2 gap-4">
                  {typeof indexMeta.latechunk==='boolean' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Late-chunk vectors</span>
                      <p className="text-sm">{indexMeta.latechunk ? '✅ Enabled' : '❌ Disabled'}</p>
                    </div>
                  )}
                  {typeof indexMeta.docling_chunk==='boolean' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">High-recall chunking</span>
                      <p className="text-sm">{indexMeta.docling_chunk ? '✅ Enabled' : '❌ Disabled'}</p>
                    </div>
                  )}
                  {indexMeta.has_fts_index && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Full-text search</span>
                      <p className="text-sm">🔍 Available</p>
                    </div>
                  )}
                  {indexMeta.has_document_structure && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Document structure</span>
                      <p className="text-sm">🔍 Organized</p>
                    </div>
                  )}
                </div>

                {/* LLM Models section */}
                {(indexMeta.enrich_model || indexMeta.overview_model) && (
                  <>
                    <div className="border-t border-white/10 pt-4">
                      <h3 className="text-sm font-medium text-gray-300 mb-3">LLM Models</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {indexMeta.enrich_model && (
                          <div>
                            <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Enrichment LLM</span>
                            <p className="text-sm break-words">{indexMeta.enrich_model}</p>
                          </div>
                        )}
                        {indexMeta.overview_model && (
                          <div>
                            <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Overview LLM</span>
                            <p className="text-sm break-words">{indexMeta.overview_model}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}

                {/* Batch sizes section */}
                {(typeof indexMeta.batch_size_embed==='number' || typeof indexMeta.batch_size_enrich==='number') && (
                  <>
                    <div className="border-t border-white/10 pt-4">
                      <h3 className="text-sm font-medium text-gray-300 mb-3">Batch Configuration</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {typeof indexMeta.batch_size_embed==='number' && (
                          <div>
                            <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Embedding batch size</span>
                            <p className="text-sm">{indexMeta.batch_size_embed}</p>
                          </div>
                        )}
                        {typeof indexMeta.batch_size_enrich==='number' && (
                          <div>
                            <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Enrichment batch size</span>
                            <p className="text-sm">{indexMeta.batch_size_enrich}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}

                {/* Metadata info */}
                {isInferredMetadata && indexMeta.metadata_inferred_at && (
                  <div className="border-t border-white/10 pt-4">
                    <h3 className="text-sm font-medium text-gray-300 mb-3">Metadata Information</h3>
                    <div className="text-xs text-gray-400 space-y-1">
                      <p>Inferred at: {new Date(indexMeta.metadata_inferred_at).toLocaleString()}</p>
                      <p>Source: LanceDB table inspection</p>
                      {indexMeta.sample_chunk_length && (
                        <p>Sample chunk length: {indexMeta.sample_chunk_length} characters</p>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Legacy index information */}
            {hasMetadata && indexStatus === 'legacy' && (
              <>
                <div className="grid grid-cols-2 gap-4">
                  {typeof indexMeta.documents_count === 'number' && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Documents</span>
                      <p className="text-sm">{indexMeta.documents_count}</p>
                    </div>
                  )}
                  {indexMeta.created_at && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Created</span>
                      <p className="text-sm">{new Date(indexMeta.created_at).toLocaleDateString()}</p>
                    </div>
                  )}
                  {indexMeta.vector_table_name && (
                    <div>
                      <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Vector table</span>
                      <p className="text-sm text-gray-400 text-xs break-all">{indexMeta.vector_table_name}</p>
                    </div>
                  )}
                </div>
                
                {indexMeta.note && (
                  <div className="border-t border-white/10 pt-4">
                    <h3 className="text-sm font-medium text-gray-300 mb-3">Technical Note</h3>
                    <p className="text-xs text-gray-400">{indexMeta.note}</p>
                  </div>
                )}
              </>
            )}

            {/* Debug info for incomplete indexes */}
            {indexStatus === 'incomplete' && indexMeta.available_tables && (
              <div className="border-t border-white/10 pt-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Debug Information</h3>
                <div className="text-xs text-gray-400 space-y-1">
                  <p>Expected table: {indexMeta.vector_table_expected}</p>
                  <p>Available tables: {indexMeta.available_tables.join(', ') || 'None'}</p>
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

        <div className="flex justify-end pt-4 border-t border-white/10">
          <button onClick={onClose} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm">Close</button>
        </div>
      </div>
    </div>
  );
} 