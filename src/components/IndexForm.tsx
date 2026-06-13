"use client";
import { useState, useRef, useEffect } from 'react';
import { GlassInput } from '@/components/ui/GlassInput';
import { GlassToggle } from '@/components/ui/GlassToggle';
import { AccordionGroup } from '@/components/ui/AccordionGroup';
import { ModelSelect } from '@/components/ModelSelect';
import { chatAPI, streamIndexJob, ChatSession, EnrichProvider, IndexBuildOptions, IndexJob } from '@/lib/api';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { useAlert, useConfirm } from '@/components/ui/confirm-dialog';

interface Props {
  onClose: () => void;
  onIndexed?: (session: ChatSession) => void;
}

type IndexingProfile = 'fast' | 'balanced' | 'maximum';

const DEFAULT_INDEXING_LLM = 'qwen3:8b';
const LARGE_INDEXING_MODEL_RE = /(gpt-oss|120b|70b|large|cloud)/i;

const ENRICH_PROVIDERS: { id: EnrichProvider; label: string; defaultModel: string; hint: string }[] = [
  { id: 'ollama',    label: 'Ollama',   defaultModel: 'qwen3:8b',               hint: 'Local model, no API key needed' },
  { id: 'groq',     label: 'Groq',     defaultModel: 'llama-3.1-8b-instant',      hint: 'Free cloud tier, very fast' },
  { id: 'openai',   label: 'ChatGPT',  defaultModel: 'gpt-4o-mini',               hint: 'OpenAI API key required' },
  { id: 'anthropic', label: 'Claude',  defaultModel: 'claude-haiku-4-5-20251001', hint: 'Anthropic API key required' },
];

const INDEXING_PROFILES: Record<IndexingProfile, {
  label: string;
  description: string;
  chunkSize: number;
  chunkOverlap: number;
  windowSize: number;
  enableEnrich: boolean;
  enableLateChunk: boolean;
  enableDoclingChunk: boolean;
  batchSizeEmbed: number;
  batchSizeEnrich: number;
}> = {
  fast: {
    label: 'Fast',
    description: 'Lowest risk for large uploads. Skips per-chunk LLM enrichment.',
    chunkSize: 768,
    chunkOverlap: 96,
    windowSize: 1,
    enableEnrich: false,
    enableLateChunk: false,
    enableDoclingChunk: false,
    batchSizeEmbed: 32,
    batchSizeEnrich: 2,
  },
  balanced: {
    label: 'Balanced',
    description: 'Good default. Better chunking, no per-chunk LLM calls unless enabled below.',
    chunkSize: 768,
    chunkOverlap: 96,
    windowSize: 1,
    enableEnrich: false,
    enableLateChunk: false,
    enableDoclingChunk: true,
    batchSizeEmbed: 32,
    batchSizeEnrich: 4,
  },
  maximum: {
    label: 'Maximum',
    description: 'Slow, opt-in accuracy mode. Uses per-chunk LLM enrichment.',
    chunkSize: 512,
    chunkOverlap: 64,
    windowSize: 1,
    enableEnrich: true,
    enableLateChunk: false,
    enableDoclingChunk: true,
    batchSizeEmbed: 24,
    batchSizeEnrich: 2,
  },
};

function isLargeIndexingModel(model?: string) {
  return Boolean(model && LARGE_INDEXING_MODEL_RE.test(model));
}

const fileStatusSummary = (job: IndexJob | null) => {
  const files = job?.files || [];
  if (!files.length) return null;
  const counts = files.reduce<Record<string, number>>((acc, file) => {
    acc[file.status] = (acc[file.status] || 0) + 1;
    return acc;
  }, {});
  return `${counts.done || 0} done - ${counts.processing || 0} active - ${counts.skipped || 0} skipped - ${counts.failed || 0} failed - ${counts.pending || 0} pending`;
};

export function IndexForm({ onClose, onIndexed }: Props) {
  const [files, setFiles] = useState<FileList | null>(null);
  const [indexName, setIndexName] = useState('');
  const [profile, setProfile] = useState<IndexingProfile>('balanced');
  const [chunkSize, setChunkSize] = useState(INDEXING_PROFILES.balanced.chunkSize);
  const [chunkOverlap, setChunkOverlap] = useState(INDEXING_PROFILES.balanced.chunkOverlap);
  const [windowSize, setWindowSize] = useState(INDEXING_PROFILES.balanced.windowSize);
  const [enableEnrich, setEnableEnrich] = useState(INDEXING_PROFILES.balanced.enableEnrich);
  const [retrievalMode, setRetrievalMode] = useState<'hybrid' | 'vector' | 'fts'>('hybrid');
  const [embeddingModel, setEmbeddingModel] = useState<string>();
  const [enrichModel, setEnrichModel] = useState<string>(DEFAULT_INDEXING_LLM);
  const [overviewModel, setOverviewModel] = useState<string>(DEFAULT_INDEXING_LLM);
  const [batchSizeEmbed, setBatchSizeEmbed] = useState(INDEXING_PROFILES.balanced.batchSizeEmbed);
  const [batchSizeEnrich, setBatchSizeEnrich] = useState(INDEXING_PROFILES.balanced.batchSizeEnrich);
  const [loading, setLoading] = useState(false);
  const [buildJob, setBuildJob] = useState<IndexJob | null>(null);
  const cancelStreamRef = useRef<(() => void) | null>(null);
  useEffect(() => () => { cancelStreamRef.current?.(); }, []);
  const [enableLateChunk, setEnableLateChunk] = useState(INDEXING_PROFILES.balanced.enableLateChunk);
  const [enableDoclingChunk, setEnableDoclingChunk] = useState(INDEXING_PROFILES.balanced.enableDoclingChunk);
  const [enrichProvider, setEnrichProvider] = useState<EnrichProvider>('ollama');
  const [enrichApiKey, setEnrichApiKey] = useState('');
  const [metadataSchemaJson, setMetadataSchemaJson] = useState('');
  const [documentMetadataJson, setDocumentMetadataJson] = useState('');

  const { showAlert, dialog: alertDialog } = useAlert();
  const { showConfirm, dialog: confirmDialog } = useConfirm();

  const selectedFiles = files ? Array.from(files) : [];
  const totalBytes = selectedFiles.reduce((sum, file) => sum + file.size, 0);
  const estimatedChunks = Math.max(
    selectedFiles.length,
    Math.ceil(totalBytes / Math.max(chunkSize * 4, 1))
  );
  const estimatedLlmCalls = selectedFiles.length + (enableEnrich ? estimatedChunks : 0);
  const hasLargeIndexingModel = isLargeIndexingModel(enrichModel) || isLargeIndexingModel(overviewModel);
  const isHighRiskJob = estimatedLlmCalls > 250 || hasLargeIndexingModel;

  const applyProfile = (nextProfile: IndexingProfile) => {
    const next = INDEXING_PROFILES[nextProfile];
    setProfile(nextProfile);
    setChunkSize(next.chunkSize);
    setChunkOverlap(next.chunkOverlap);
    setWindowSize(next.windowSize);
    setEnableEnrich(next.enableEnrich);
    setEnableLateChunk(next.enableLateChunk);
    setEnableDoclingChunk(next.enableDoclingChunk);
    setBatchSizeEmbed(next.batchSizeEmbed);
    setBatchSizeEnrich(next.batchSizeEnrich);
    setEnrichModel(DEFAULT_INDEXING_LLM);
    setOverviewModel(DEFAULT_INDEXING_LLM);
  };

  const handleProviderChange = (p: EnrichProvider) => {
    setEnrichProvider(p);
    setEnrichApiKey('');
    const providerDef = ENRICH_PROVIDERS.find(x => x.id === p);
    if (providerDef) setEnrichModel(providerDef.defaultModel);
  };

  const buildOptions = (): IndexBuildOptions => ({
    latechunk: enableLateChunk,
    doclingChunk: enableDoclingChunk,
    chunkSize,
    chunkOverlap,
    retrievalMode: retrievalMode === 'fts' ? 'bm25' : retrievalMode,
    windowSize,
    enableEnrich,
    embeddingModel,
    enrichModel,
    enrichProvider,
    enrichApiKey: enrichApiKey || undefined,
    overviewModel,
    batchSizeEmbed,
    batchSizeEnrich: Math.max(1, Math.min(batchSizeEnrich, 8)),
  });

  const waitForBuildJob = async (jobId: string) => {
    return new Promise<IndexJob>((resolve, reject) => {
      let lastJob: IndexJob | null = null;
      const { cancel, promise } = streamIndexJob(jobId, (data) => {
        const job: IndexJob = {
          id: data.id ?? jobId,
          index_id: data.index_id ?? lastJob?.index_id ?? '',
          ...data,
        } as IndexJob;
        lastJob = job;
        setBuildJob(job);
        if (data.status === 'completed') { cancel(); resolve(job); }
        if (data.status === 'failed') { cancel(); reject(new Error(data.message || 'Index build failed')); }
        if (data.status === 'cancelled') { cancel(); reject(new Error('Index build was cancelled')); }
      });
      cancelStreamRef.current = cancel;
      promise.then(
        () => {
          // Stream closed without a terminal status event (server restart,
          // proxy timeout): poll the job once so we settle instead of hanging.
          chatAPI.getIndexJob(jobId).then((job) => {
            setBuildJob(job);
            if (job.status === 'completed') resolve(job);
            else reject(new Error(job.error || job.message || 'Build stream ended unexpectedly'));
          }).catch(() => reject(new Error('Build stream ended unexpectedly')));
        },
        (err) => {
          if ((err as Error).name !== 'AbortError') reject(err);
        },
      );
    });
  };

  const handleCancelBuild = async () => {
    if (!buildJob) return;
    try {
      const job = await chatAPI.cancelIndexJob(buildJob.id);
      setBuildJob(job);
    } catch (e) {
      console.error('Cancel failed', e);
      await showAlert('Cancel request failed. See console for details.');
    }
  };

  const handleSubmit = async () => {
    if (!files) return;
    let metadataSchema: Array<Record<string, unknown>> | undefined;
    let documentMetadata: Record<string, unknown> | Record<string, Record<string, unknown>> | undefined;
    try {
      if (metadataSchemaJson.trim()) {
        const parsed = JSON.parse(metadataSchemaJson);
        if (!Array.isArray(parsed) || parsed.length === 0) {
          throw new Error('Metadata schema must be a non-empty JSON array.');
        }
        metadataSchema = parsed;
      }
      if (documentMetadataJson.trim()) {
        const parsed = JSON.parse(documentMetadataJson);
        if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
          throw new Error('Document metadata must be a JSON object.');
        }
        if (!metadataSchema) {
          throw new Error('Define a metadata schema before adding document metadata.');
        }
        documentMetadata = parsed;
      }
    } catch (e) {
      await showAlert(e instanceof Error ? e.message : 'Metadata JSON is invalid.');
      return;
    }
    if (hasLargeIndexingModel) {
      await showAlert('Large chat models such as gpt-oss:120b-cloud are blocked for indexing enrichment/overview. Use qwen3:8b for indexing, then use the large model for chat.');
      return;
    }
    if (isHighRiskJob && !await showConfirm(`This index may run about ${estimatedLlmCalls.toLocaleString()} LLM call(s). Continue?`)) {
      return;
    }
    setLoading(true);
    try {
      const { index_id } = await chatAPI.createIndex(indexName);

      if (metadataSchema) {
        await chatAPI.setIndexMetadataSchema(index_id, metadataSchema);
      }
      await chatAPI.uploadFilesToIndex(index_id, Array.from(files), documentMetadata);

      const preflight = await chatAPI.preflightIndexBuild(index_id, buildOptions());
      if (!preflight.ok) {
        throw new Error(preflight.errors.join(' ') || 'Index build preflight failed.');
      }

      const started = await chatAPI.startIndexBuild(index_id, buildOptions());
      setBuildJob({
        id: started.job_id,
        index_id,
        status: 'queued',
        stage: 'queued',
        progress: 0,
        message: started.message,
      });
      await waitForBuildJob(started.job_id);

      const session = await chatAPI.createSession(indexName);
      await chatAPI.linkIndexToSession(session.id, index_id);

      if (onIndexed) onIndexed(session);
    } catch (e) {
      console.error('Indexing failed', e);
      setLoading(false);
      setBuildJob(null);
      await showAlert(e instanceof Error ? `Indexing failed: ${e.message}` : 'Indexing failed. See console for details.');
    }
  };

  return (
    <div className="relative bg-white/5 backdrop-blur rounded-xl p-6 w-[640px] text-white space-y-6">
      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center rounded-xl z-20">
          <div className="w-10 h-10 border-4 border-white/30 border-t-transparent rounded-full animate-spin"></div>
          <p className="mt-4 text-sm text-gray-200">{buildJob?.message || 'Starting index build…'}</p>
          {buildJob && (
            <div className="mt-4 w-72">
              <div className="h-2 rounded bg-white/10 overflow-hidden">
                <div className="h-full bg-green-500 transition-all" style={{ width: `${Math.max(0, Math.min(buildJob.progress || 0, 100))}%` }} />
              </div>
              <div className="mt-2 flex justify-between text-xs text-gray-300">
                <span>{buildJob.stage}</span>
                <span>{buildJob.progress || 0}%</span>
              </div>
              {fileStatusSummary(buildJob) && (
                <p className="mt-2 text-xs text-gray-300">{fileStatusSummary(buildJob)}</p>
              )}
              {buildJob.cancel_requested && <p className="mt-2 text-xs text-yellow-200">Cancel requested. Waiting for the active indexing step to finish.</p>}
              {(() => {
                const failed = (buildJob.files || []).filter(f => f.status === 'failed' && f.error);
                if (!failed.length) return null;
                return (
                  <div className="mt-3 max-h-32 overflow-y-auto space-y-1">
                    {failed.map((f, i) => (
                      <div key={i} className="rounded bg-red-900/40 px-2 py-1 text-[11px] text-red-300">
                        <span className="font-medium">{f.filename || 'Unknown file'}</span>
                        {f.error && <span className="block text-red-400/80 truncate">{f.error}</span>}
                      </div>
                    ))}
                  </div>
                );
              })()}
              {buildJob.status !== 'completed' && buildJob.status !== 'failed' && buildJob.status !== 'cancelled' && (
                <button onClick={handleCancelBuild} className="mt-4 w-full rounded bg-red-700/80 px-3 py-2 text-xs hover:bg-red-700">
                  Cancel build
                </button>
              )}
            </div>
          )}
        </div>
      )}

      <h2 className="text-lg font-semibold">Create new index</h2>

      <div>
        <label className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Index name</label>
        <GlassInput placeholder="My project docs" value={indexName} onChange={(e)=>setIndexName(e.target.value)} />
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-xs uppercase tracking-wide text-gray-300 mb-1">PDF files</label>
          <label
            htmlFor="file-upload"
            className="flex flex-col items-center justify-center w-full h-32 border border-dashed border-white/20 rounded cursor-pointer hover:border-white/40 transition"
            onDragOver={(e)=>e.preventDefault()}
            onDrop={(e)=>{e.preventDefault(); if(e.dataTransfer.files) setFiles(e.dataTransfer.files)}}
          >
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mb-2 text-white/80"><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/><polyline points="7 10 12 5 17 10"/><line x1="12" y1="5" x2="12" y2="16"/></svg>
            <span className="text-xs text-gray-400">Drag & Drop documents here or click to browse</span>
            <input id="file-upload" type="file" accept="application/pdf,.docx,.doc,.html,.htm,.md,.txt" multiple className="hidden" onChange={(e)=>setFiles(e.target.files)} />
          </label>
          {files && <p className="mt-1 text-xs text-green-400">{files.length} file(s) selected</p>}
        </div>

        <div>
          <label className="block text-xs uppercase tracking-wide text-gray-300 mb-2">Indexing profile</label>
          <div className="grid grid-cols-3 gap-2">
            {(Object.keys(INDEXING_PROFILES) as IndexingProfile[]).map((key) => {
              const item = INDEXING_PROFILES[key];
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => applyProfile(key)}
                  className={`rounded border px-3 py-2 text-left transition ${
                    profile === key ? 'border-green-400 bg-green-500/15' : 'border-white/10 bg-white/5 hover:bg-white/10'
                  }`}
                >
                  <span className="block text-xs font-medium">{item.label}</span>
                  <span className="block text-[11px] leading-4 text-gray-400">{item.description}</span>
                </button>
              );
            })}
          </div>
        </div>

        {selectedFiles.length > 0 && (
          <div className={`rounded border p-3 text-xs ${isHighRiskJob ? 'border-yellow-500/40 bg-yellow-500/10 text-yellow-100' : 'border-white/10 bg-white/5 text-gray-300'}`}>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <span className="block text-gray-400">Size</span>
                <span>{(totalBytes / (1024 * 1024)).toFixed(1)} MB</span>
              </div>
              <div>
                <span className="block text-gray-400">Est. chunks</span>
                <span>{estimatedChunks.toLocaleString()}</span>
              </div>
              <div>
                <span className="block text-gray-400">Est. LLM calls</span>
                <span>{estimatedLlmCalls.toLocaleString()}</span>
              </div>
            </div>
            {hasLargeIndexingModel && (
              <p className="mt-2 text-yellow-200">Large generation models are blocked for indexing. Use {DEFAULT_INDEXING_LLM} here and keep gpt-oss:120b-cloud for chat.</p>
            )}
          </div>
        )}

        <div>
          <label className="flex items-center gap-1 text-xs uppercase tracking-wide text-gray-300 mb-1">Retrieval mode <InfoTooltip text="Choose how chunks are found. Hybrid combines full-text search with vectors; FTS uses textual matching only; Vector relies purely on dense similarity." /></label>
          <div className="flex gap-3">
            {(['hybrid','vector','fts'] as const).map((m)=>(
              <button key={m} onClick={()=>setRetrievalMode(m)} className={`px-3 py-1 rounded text-xs font-sans ${retrievalMode===m?'bg-white/20':'bg-white/10 hover:bg-white/20'}`}>{m==='fts' ? 'FTS' : m}</button>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-4 mt-3">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-400">Late-chunk vectors <InfoTooltip text="Split chunks into sub-vectors to improve recall, then merge them back after retrieval." size={12} /></span>
              <GlassToggle checked={enableLateChunk} onChange={setEnableLateChunk} />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-400">High-recall chunking <InfoTooltip text="Advanced sentence-level packing with Docling features for maximum recall. Both modes use token-based sizing." size={12} /></span>
              <GlassToggle checked={enableDoclingChunk} onChange={setEnableDoclingChunk} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div>
              <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">Chunk size <InfoTooltip text="Maximum token length for each chunk. Both legacy and high-recall modes now use token-based sizing." size={12} /></label>
              <GlassInput type="number" value={chunkSize} onChange={(e) => setChunkSize(parseInt(e.target.value))} />
            </div>
            <div>
              <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">Chunk overlap <InfoTooltip text="Tokens reused between adjacent chunks to preserve context." size={12} /></label>
              <GlassInput
                type="number"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4">
            <div>
              <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">Embedding model <InfoTooltip text="Model used to generate dense vectors stored in the index." size={12} /></label>
              <ModelSelect
                value={embeddingModel}
                onChange={setEmbeddingModel}
                type="embedding"
                placeholder="Select embedding model"
              />
            </div>
            <div>
              <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">Overview LLM <InfoTooltip text="Use a small model here. This runs during indexing and should not use large chat models." size={12} /></label>
              <ModelSelect
                value={overviewModel}
                onChange={setOverviewModel}
                type="generation"
                placeholder="Select overview LLM"
              />
            </div>
          </div>
        </div>

        <AccordionGroup title={<><span>Contextual Retrieval</span> <InfoTooltip text="Adds neighbour chunks into each original chunk then enriches with LLM – improves semantic continuity but increases indexing latency." /></>}>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-400">Enable</span>
            <GlassToggle checked={enableEnrich} onChange={setEnableEnrich} />
          </div>

          {enableEnrich && (
            <div className="mt-3 space-y-3">
              <div>
                <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">
                  Enrichment provider
                  <InfoTooltip text="Ollama runs locally (free, slower). Cloud providers are fast and offload GPU usage so embedding stays fast. API key is used only during indexing." size={12} />
                </label>
                <div className="flex gap-2 flex-wrap">
                  {ENRICH_PROVIDERS.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      title={p.hint}
                      onClick={() => handleProviderChange(p.id)}
                      className={`px-3 py-1.5 rounded text-xs font-medium transition ${
                        enrichProvider === p.id
                          ? 'bg-green-500/25 border border-green-400 text-white'
                          : 'bg-white/5 border border-white/10 text-gray-300 hover:bg-white/10'
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
                {enrichProvider !== 'ollama' && (
                  <p className="mt-1 text-[11px] text-green-300">
                    {ENRICH_PROVIDERS.find(p => p.id === enrichProvider)?.hint}
                  </p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">Context window <InfoTooltip text="Number of neighbour chunks included when enriching context." size={12} /></label>
                  <GlassInput type="number" value={windowSize} onChange={(e)=>setWindowSize(parseInt(e.target.value))} />
                </div>
                <div>
                  <label className="block text-xs mb-1 text-gray-400">Model</label>
                  {enrichProvider === 'ollama' ? (
                    <ModelSelect
                      value={enrichModel}
                      onChange={setEnrichModel}
                      type="generation"
                      placeholder="Select retrieval LLM"
                    />
                  ) : (
                    <GlassInput
                      value={enrichModel}
                      onChange={(e) => setEnrichModel(e.target.value)}
                      placeholder={ENRICH_PROVIDERS.find(p => p.id === enrichProvider)?.defaultModel}
                    />
                  )}
                </div>
              </div>

              {enrichProvider !== 'ollama' && (
                <div>
                  <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">
                    API key
                    <InfoTooltip text="Used only during indexing. Not stored in the database." size={12} />
                  </label>
                  <GlassInput
                    type="password"
                    value={enrichApiKey}
                    onChange={(e) => setEnrichApiKey(e.target.value)}
                    placeholder={`${enrichProvider.toUpperCase()}_API_KEY (or set env var)`}
                  />
                </div>
              )}
            </div>
          )}

          {!enableEnrich && (
            <div className="grid grid-cols-2 gap-4 mt-3">
              <div>
                <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">Context window <InfoTooltip text="Number of neighbour chunks included when enriching context." size={12} /></label>
                <GlassInput type="number" value={windowSize} onChange={(e)=>setWindowSize(parseInt(e.target.value))} />
              </div>
            </div>
          )}
        </AccordionGroup>

        <AccordionGroup title={<><span>Document Metadata</span> <InfoTooltip text="Optional typed fields stored with every document and available as retrieval filters." /></>}>
          <div className="space-y-3">
            <div>
              <label className="block text-xs mb-1 text-gray-400">Schema</label>
              <textarea
                value={metadataSchemaJson}
                onChange={(e) => setMetadataSchemaJson(e.target.value)}
                placeholder={'[{"name":"project","type":"string","required":true}]'}
                rows={4}
                spellCheck={false}
                className="w-full resize-y rounded border border-white/10 bg-white/5 px-3 py-2 font-mono text-xs text-white outline-none focus:border-green-400"
              />
            </div>
            <div>
              <label className="block text-xs mb-1 text-gray-400">Upload values</label>
              <textarea
                value={documentMetadataJson}
                onChange={(e) => setDocumentMetadataJson(e.target.value)}
                placeholder={'{"project":"Aurora"}'}
                rows={4}
                spellCheck={false}
                className="w-full resize-y rounded border border-white/10 bg-white/5 px-3 py-2 font-mono text-xs text-white outline-none focus:border-green-400"
              />
            </div>
          </div>
        </AccordionGroup>
      </div>

      <AccordionGroup title={<><span>Batch Size</span> <InfoTooltip text="Control the number of chunks processed per batch. Larger values speed up indexing but require more memory." /></>}>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">Embedding batch size <InfoTooltip text="Chunks processed per batch when producing embeddings." size={12} /></label>
            <GlassInput
              type="number"
              value={batchSizeEmbed}
              onChange={(e) => setBatchSizeEmbed(parseInt(e.target.value))}
            />
          </div>
          <div>
              <label className="flex items-center gap-1 text-xs mb-1 text-gray-400">Context retrieval batch size <InfoTooltip text="Keep this small. Enrichment still makes one LLM call per chunk." size={12} /></label>
            <GlassInput
              type="number"
              value={batchSizeEnrich}
              onChange={(e) => setBatchSizeEnrich(parseInt(e.target.value))}
            />
          </div>
        </div>
      </AccordionGroup>

      <div className="flex justify-end gap-3 pt-4 border-t border-white/10">
        <button onClick={onClose} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm">
          Cancel
        </button>
        <button
          disabled={loading || !files || !indexName.trim()}
          onClick={handleSubmit}
          className="px-4 py-2 bg-green-600 rounded disabled:opacity-40 text-sm"
        >
          {loading ? 'Indexing…' : 'Start indexing'}
        </button>
      </div>
      {alertDialog}
      {confirmDialog}
    </div>
  );
}
