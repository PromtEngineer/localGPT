import { useEffect, useRef, useState } from 'react';
import { ApiRecord, BuildIndexResponse, chatAPI, IndexDiagnostics, IndexDiagnosticsSummary, IndexJob, IndexSummary, MaintenanceHealthReport } from '@/lib/api';

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
  const [busyMessage, setBusyMessage] = useState<string | null>(null);
  const [buildJob, setBuildJob] = useState<IndexJob | null>(null);
  const [uploadTargetId, setUploadTargetId] = useState<string | null>(null);
  const [showFileDetails, setShowFileDetails] = useState(false);
  const [diagnosticsById, setDiagnosticsById] = useState<Record<string, IndexDiagnosticsSummary>>({});
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false);
  const [healthReport, setHealthReport] = useState<MaintenanceHealthReport | null>(null);
  const [healthReportLoading, setHealthReportLoading] = useState(false);
  const [showHealthReport, setShowHealthReport] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await chatAPI.listIndexes();
        setIndexes(data.indexes);
        refreshDiagnostics();
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Failed to load indexes');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const filtered = indexes.filter(i => (i.name || '').toLowerCase().includes(search.toLowerCase()));
  const indexId = (idx: IndexSummary) => idx.id || idx.index_id || '';

  async function refreshDiagnostics() {
    setDiagnosticsLoading(true);
    try {
      const data = await chatAPI.getIndexesDiagnostics();
      const next: Record<string, IndexDiagnosticsSummary> = {};
      data.diagnostics.forEach((item) => {
        next[item.index_id] = item;
      });
      setDiagnosticsById(next);
    } catch (e) {
      console.warn('Index health refresh failed', e);
    } finally {
      setDiagnosticsLoading(false);
    }
  }

  async function handleMaintenanceReport() {
    if (showHealthReport) {
      setShowHealthReport(false);
      return;
    }
    setShowHealthReport(true);
    setHealthReportLoading(true);
    try {
      const report = await chatAPI.getMaintenanceHealthReport();
      setHealthReport(report);
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to load maintenance health report');
      setShowHealthReport(false);
    } finally {
      setHealthReportLoading(false);
    }
  }

  const buildOptions = (idx: IndexSummary, forceReindex = false) => {
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
      forceReindex,
    };
  };

  const formatBuildSummary = (result: BuildIndexResponse) => {
    const stats = result.indexing_result;
    if (!stats) return 'Rebuild complete.';
    const parts = [
      `${stats.files_processed ?? 0}/${stats.total_files_considered ?? 0} files processed`,
      `${stats.unchanged_files ?? 0} unchanged skipped`,
      `${stats.chunks_generated ?? 0} chunks`,
    ];
    if ((stats.chunk_cache_hits ?? 0) > 0) {
      parts.push(`${stats.chunk_cache_hits} cache hits`);
    }
    return `Rebuild complete: ${parts.join(', ')}.`;
  };

  const formatDiagnostics = (diagnostics: IndexDiagnostics) => {
    const lines = [
      `Health: ${diagnostics.health}`,
      `Files: ${diagnostics.document_count} (${diagnostics.total_size})`,
      `Vectors: ${diagnostics.vector_table?.exists ? `${diagnostics.vector_table.row_count ?? 'unknown'} rows` : 'missing'}`,
      `Recommended action: ${diagnostics.recommended_action.replace('_', ' ')}`,
    ];
    if (diagnostics.errors.length) lines.push(`Errors: ${diagnostics.errors.join(' ')}`);
    if (diagnostics.warnings.length) lines.push(`Warnings: ${diagnostics.warnings.join(' ')}`);
    if (diagnostics.recommendations.length) lines.push(`Next: ${diagnostics.recommendations.join(' ')}`);
    return lines.join('\n');
  };

  const waitForBuildJob = async (jobId: string): Promise<BuildIndexResponse> => {
    while (true) {
      const job = await chatAPI.getIndexJob(jobId);
      setBuildJob(job);
      setBusyMessage(job.message || 'Rebuilding index...');

      if (job.status === 'completed') {
        return job.result || { message: job.message || 'Rebuild complete.' };
      }
      if (job.status === 'failed') {
        throw new Error(job.error || job.message || 'Index rebuild failed.');
      }
      if (job.status === 'cancelled') {
        throw new Error('Index rebuild was cancelled.');
      }

      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
  };

  const rebuildInBackground = async (idxId: string, idx: IndexSummary, forceReindex: boolean) => {
    const options = buildOptions(idx, forceReindex);
    const preflight = await chatAPI.preflightIndexBuild(idxId, options);
    if (!preflight.ok) {
      throw new Error(preflight.errors.join(' ') || 'Index build preflight failed.');
    }
    const started = await chatAPI.startIndexBuild(idxId, options);
    setBuildJob({
      id: started.job_id,
      index_id: idxId,
      status: 'queued',
      stage: 'queued',
      progress: 0,
      message: started.message,
    });
    return waitForBuildJob(started.job_id);
  };

  const fileStatusSummary = (job: IndexJob | null) => {
    const files = job?.files || [];
    if (!files.length) return null;
    const counts = files.reduce<Record<string, number>>((acc, file) => {
      acc[file.status] = (acc[file.status] || 0) + 1;
      return acc;
    }, {});
    return `${counts.done || 0} done · ${counts.processing || 0} active · ${counts.skipped || 0} skipped · ${counts.failed || 0} failed · ${counts.pending || 0} pending`;
  };

  const formatFileDetails = (job: IndexJob | null) => {
    const files = job?.files || [];
    if (!files.length) return null;
    return files.map((file) => (
      <li key={`${file.filename}-${file.id}`} className="border-b border-white/10 py-2 last:border-b-0">
        <div className="flex items-center justify-between gap-3 text-sm text-gray-200">
          <div className="min-w-0 truncate">
            <div className="font-medium">{file.filename || 'Unnamed file'}</div>
            <div className="text-xs text-gray-400">{file.stage ? `Stage: ${file.stage}` : 'Stage unknown'}</div>
          </div>
          <span className={`rounded-full px-2 py-1 text-[11px] ${file.status === 'done' ? 'bg-green-500/20 text-green-200' : file.status === 'failed' ? 'bg-red-500/20 text-red-200' : file.status === 'processing' ? 'bg-yellow-500/20 text-yellow-200' : 'bg-white/10 text-gray-200'}`}>
            {file.status}
          </span>
        </div>
        {file.error && <p className="mt-1 text-xs text-red-300">Error: {file.error}</p>}
        {typeof file.chunks_generated === 'number' && <p className="mt-1 text-xs text-gray-400">Chunks generated: {file.chunks_generated}</p>}
      </li>
    ));
  };

  const healthBadge = (diagnostics?: IndexDiagnosticsSummary) => {
    if (!diagnostics) return { label: diagnosticsLoading ? 'checking' : 'unknown', className: 'bg-white/10 text-gray-300 border-white/10' };
    if (diagnostics.health === 'healthy') return { label: 'healthy', className: 'bg-green-500/15 text-green-300 border-green-400/20' };
    if (diagnostics.health === 'warning') return { label: 'warning', className: 'bg-yellow-500/15 text-yellow-200 border-yellow-400/20' };
    return { label: 'unhealthy', className: 'bg-red-500/15 text-red-300 border-red-400/20' };
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

  async function handleRebuild(idx: IndexSummary, forceReindex = false) {
    const id = indexId(idx);
    if (!id) return;
    const docCount = idx.documents?.length || 0;
    if (docCount === 0) {
      alert('This index has no files. Add files first, then rebuild.');
      return;
    }
    const actionLabel = forceReindex ? 'Force rebuild' : 'Rebuild changed files for';
    if (!confirm(`${actionLabel} "${idx.name || 'Untitled index'}" from ${docCount} file(s)?`)) return;
    setBusyId(id);
    setBusyMessage(`${forceReindex ? 'Force rebuilding' : 'Checking changed files for'} "${idx.name || 'Untitled index'}"…`);
    setMenuOpenId(null);
    try {
      const result = await rebuildInBackground(id, idx, forceReindex);
      const data = await chatAPI.listIndexes();
      setIndexes(data.indexes);
      refreshDiagnostics();
      alert(formatBuildSummary(result));
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to rebuild index');
    } finally {
      setBusyId(null);
      setBusyMessage(null);
      setBuildJob(null);
    }
  }

  async function handleOpenIndex(idx: IndexSummary) {
    const id = indexId(idx);
    if (!id) return;
    setMenuOpenId(null);
    let summary = diagnosticsById[id];
    if (!summary) {
      setBusyId(id);
      setBusyMessage(`Checking "${idx.name || 'Untitled index'}"...`);
      try {
        const diagnostics = await chatAPI.getIndexDiagnostics(id);
        summary = {
          index_id: diagnostics.index_id,
          name: diagnostics.name,
          health: diagnostics.health,
          ok: diagnostics.ok,
          recommended_action: diagnostics.recommended_action,
          can_repair: diagnostics.can_repair,
          error_count: diagnostics.errors.length,
          warning_count: diagnostics.warnings.length,
          document_count: diagnostics.document_count,
          total_size: diagnostics.total_size,
          vector_exists: Boolean(diagnostics.vector_table?.exists),
          vector_rows: diagnostics.vector_table?.row_count,
          metadata_status: diagnostics.metadata_status,
        };
        setDiagnosticsById((prev) => ({ ...prev, [id]: summary as IndexDiagnosticsSummary }));
      } catch (e: unknown) {
        alert(e instanceof Error ? e.message : 'Failed to check index health');
        setBusyId(null);
        setBusyMessage(null);
        return;
      }
      setBusyId(null);
      setBusyMessage(null);
    }

    if (summary.health === 'unhealthy') {
      const canRepair = summary.can_repair;
      const message = `"${idx.name || 'Untitled index'}" is unhealthy and should not be opened for chat.\n\nRecommended action: ${summary.recommended_action.replace('_', ' ')}.`;
      if (canRepair && confirm(`${message}\n\nRun diagnose + repair now?`)) {
        await handleDiagnostics(idx, true);
      } else if (!canRepair) {
        alert(`${message}\n\nRe-upload or fix the source files first.`);
      }
      return;
    }

    if (summary.health === 'warning' && !confirm(`"${idx.name || 'Untitled index'}" has diagnostics warnings. Open it anyway?`)) {
      return;
    }

    onSelect(id);
  }

  async function handleDiagnostics(idx: IndexSummary, offerRepair = false) {
    const id = indexId(idx);
    if (!id) return;
    setBusyId(id);
    setBusyMessage(`Checking "${idx.name || 'Untitled index'}"...`);
    setMenuOpenId(null);
    try {
      const diagnostics = await chatAPI.getIndexDiagnostics(id);
      const details = formatDiagnostics(diagnostics);
      if (!offerRepair || diagnostics.recommended_action === 'none') {
        alert(details);
        return;
      }
      if (!diagnostics.can_repair) {
        alert(details);
        return;
      }
      const forceReindex = diagnostics.recommended_action === 'force_rebuild';
      const repairLabel = forceReindex ? 'Force rebuild now?' : 'Rebuild changed files now?';
      if (!confirm(`${details}\n\n${repairLabel}`)) return;
      setBusyMessage(`${forceReindex ? 'Force rebuilding' : 'Rebuilding'} "${idx.name || 'Untitled index'}"...`);
      const result = await rebuildInBackground(id, idx, forceReindex);
      const data = await chatAPI.listIndexes();
      setIndexes(data.indexes);
      refreshDiagnostics();
      alert(formatBuildSummary(result));
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to inspect index');
    } finally {
      setBusyId(null);
      setBusyMessage(null);
      setBuildJob(null);
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
    setBusyMessage(`Uploading ${files.length} file(s)…`);
    try {
      await chatAPI.uploadFilesToIndex(uploadTargetId, Array.from(files));
      setBusyMessage(`Rebuilding "${idx.name || 'Untitled index'}" with the added file(s)…`);
      const result = await rebuildInBackground(uploadTargetId, idx, false);
      setBusyMessage('Refreshing indexes…');
      const data = await chatAPI.listIndexes();
      setIndexes(data.indexes);
      refreshDiagnostics();
      alert(`Files added. ${formatBuildSummary(result)}`);
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to add files and rebuild index');
    } finally {
      setBusyId(null);
      setBusyMessage(null);
      setBuildJob(null);
      setUploadTargetId(null);
      setShowFileDetails(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function handleCancelBuild() {
    if (!buildJob) return;
    try {
      const job = await chatAPI.cancelIndexJob(buildJob.id);
      setBuildJob(job);
      setBusyMessage(job.message || 'Cancel requested.');
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to cancel rebuild');
    }
  }

  async function handleResumeBuild() {
    if (!buildJob) return;
    setBusyId(buildJob.index_id || buildJob.id);
    try {
      const result = await chatAPI.resumeIndexJob(buildJob.id);
      setBusyMessage(result.message || 'Resume requested.');
      const job = await chatAPI.getIndexJob(buildJob.id);
      setBuildJob(job);
      if (job.status === 'queued' || job.status === 'running') {
        setBusyMessage('Resuming build…');
      }
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Failed to resume build');
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
        <div className="flex gap-2">
          <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search…" className="min-w-0 flex-1 px-3 py-2 rounded bg-black/30 border border-white/20 focus:outline-none" />
          <button onClick={refreshDiagnostics} disabled={diagnosticsLoading || !!busyId} className="shrink-0 rounded bg-white/10 px-3 py-2 text-xs text-gray-200 hover:bg-white/20 disabled:opacity-50">
            {diagnosticsLoading ? 'Checking' : 'Refresh health'}
          </button>
          <button onClick={handleMaintenanceReport} disabled={healthReportLoading || !!busyId} className="shrink-0 rounded bg-white/10 px-3 py-2 text-xs text-gray-200 hover:bg-white/20 disabled:opacity-50">
            {healthReportLoading ? 'Loading report' : showHealthReport ? 'Hide maintenance report' : 'Maintenance report'}
          </button>
        </div>
        {showHealthReport && (
          <div className="rounded-lg border border-white/10 bg-black/60 p-3 text-xs text-gray-200">
            {healthReportLoading && <p className="text-gray-300">Loading maintenance health report…</p>}
            {!healthReportLoading && healthReport && (
              healthReport.error ? (
                <p className="text-red-300">Error: {healthReport.error}</p>
              ) : (
                <>
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <p className="text-sm font-medium text-white">Maintenance health report</p>
                    <span className="shrink-0 text-[11px] text-gray-400">As of {new Date(healthReport.timestamp).toLocaleString()}</span>
                  </div>
                  <p className="mb-2 text-gray-300">
                    {healthReport.summary.healthy} healthy · {healthReport.summary.warning} warning · {healthReport.summary.unhealthy} unhealthy ({healthReport.summary.total} total)
                  </p>
                  <ul className="space-y-2 max-h-56 overflow-y-auto">
                    {healthReport.indexes.map((entry) => (
                      <li key={entry.index_id} className="border-b border-white/10 py-2 last:border-b-0">
                        <div className="flex items-center justify-between gap-3">
                          <span className="min-w-0 truncate font-medium text-gray-100">{entry.name || 'Untitled index'}</span>
                          <span className={`shrink-0 rounded-full px-2 py-1 text-[11px] ${entry.health === 'healthy' ? 'bg-green-500/20 text-green-200' : entry.health === 'warning' ? 'bg-yellow-500/20 text-yellow-200' : 'bg-red-500/20 text-red-200'}`}>
                            {entry.health}
                          </span>
                        </div>
                        <p className="mt-1 text-gray-400">
                          {entry.documents} document(s){entry.status ? ` · status: ${entry.status}` : ''}
                        </p>
                        {entry.latest_job?.status && (
                          <p className="mt-1 text-gray-400">
                            Latest job: {entry.latest_job.status}{entry.latest_job.error ? ` — ${entry.latest_job.error}` : ''}
                          </p>
                        )}
                      </li>
                    ))}
                    {healthReport.indexes.length === 0 && <p className="text-gray-400">No indexes to report on.</p>}
                  </ul>
                </>
              )
            )}
          </div>
        )}
        {busyId && (
          <div className="space-y-2 text-xs text-green-300">
            <p>{busyMessage || 'Rebuilding index…'} Keep both backend terminals running.</p>
            {buildJob && (
              <div className="space-y-1">
                <div className="h-1.5 overflow-hidden rounded bg-white/10">
                  <div className="h-full bg-green-500 transition-all" style={{ width: `${Math.max(0, Math.min(buildJob.progress || 0, 100))}%` }} />
                </div>
                <div className="flex items-center justify-between text-gray-300">
                  <span>{buildJob.stage}</span>
                  <span>{buildJob.progress || 0}%</span>
                </div>
                {fileStatusSummary(buildJob) && <p className="text-gray-300">{fileStatusSummary(buildJob)}</p>}
                {buildJob.status === 'paused' && (
                  <p className="text-yellow-200">Build paused. Resume to continue indexing.</p>
                )}
                <div className="flex flex-wrap items-center gap-2">
                  {buildJob.files && buildJob.files.length > 0 && (
                    <button onClick={() => setShowFileDetails((prev) => !prev)} className="rounded bg-white/10 px-3 py-1 text-xs text-gray-200 hover:bg-white/20">
                      {showFileDetails ? 'Hide file details' : 'Show file details'}
                    </button>
                  )}
                  {buildJob.status === 'paused' && (
                    <button onClick={handleResumeBuild} className="rounded bg-blue-500/80 px-3 py-1 text-white hover:bg-blue-500">
                      Resume rebuild
                    </button>
                  )}
                  {buildJob.status !== 'completed' && buildJob.status !== 'failed' && buildJob.status !== 'cancelled' && (
                    <button onClick={handleCancelBuild} className="rounded bg-red-500/80 px-3 py-1 text-white hover:bg-red-500">
                      Cancel rebuild
                    </button>
                  )}
                </div>
                {showFileDetails && buildJob.files && buildJob.files.length > 0 && (
                  <div className="mt-3 rounded-lg border border-white/10 bg-black/60 p-3 text-xs text-gray-200">
                    <p className="mb-2 text-sm font-medium text-white">File progress details</p>
                    <ul className="space-y-2 max-h-56 overflow-y-auto">
                      {formatFileDetails(buildJob)}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        {loading && <p className="text-sm text-gray-300">Loading…</p>}
        {error && <p className="text-sm text-red-400">{error}</p>}
        {!loading && !error && (
          <ul className="space-y-2">
            {filtered.map(idx => (
              <li key={indexId(idx)}>
                <div className="relative group">
                  {(() => {
                    const id = indexId(idx);
                    const diagnostics = diagnosticsById[id];
                    const badge = healthBadge(diagnostics);
                    return (
                      <button disabled={busyId===id} onClick={()=>handleOpenIndex(idx)} className="w-full px-4 py-3 bg-white/10 hover:bg-white/20 rounded transition flex justify-between items-center gap-3 pr-10 disabled:opacity-50">
                        <span className="min-w-0 flex-1 font-medium truncate">{idx.name}</span>
                        <span className={`shrink-0 rounded border px-2 py-0.5 text-[11px] ${badge.className}`}>{badge.label}</span>
                        <span className="shrink-0 text-xs text-gray-400">{busyId===id ? 'rebuilding...' : `${idx.documents?.length || 0} files`}</span>
                      </button>
                    );
                  })()}

                  <button disabled={busyId===indexId(idx)} onClick={(e)=>{e.stopPropagation(); const id = indexId(idx); setMenuOpenId(menuOpenId===id?null:id);}} title="More actions" className="absolute right-4 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 text-gray-400 hover:text-white transition text-lg leading-none font-bold disabled:opacity-40">
                    …
                  </button>

                  {menuOpenId===indexId(idx) && (
                    <div className="index-row-menu absolute right-0 top-full mt-1 bg-black/80 backdrop-blur border border-white/10 rounded shadow-lg py-1 w-44 text-sm z-50">
                      <button onClick={()=>handleOpenIndex(idx)} className="block w-full text-left px-4 py-2 hover:bg-white/10">Open</button>
                      <button onClick={()=>handleAddFiles(idx)} className="block w-full text-left px-4 py-2 hover:bg-white/10">Add files + rebuild</button>
                      <button onClick={()=>handleDiagnostics(idx)} className="block w-full text-left px-4 py-2 hover:bg-white/10">Run diagnostics</button>
                      <button onClick={()=>handleDiagnostics(idx, true)} className="block w-full text-left px-4 py-2 hover:bg-white/10">Diagnose + repair</button>
                      <div className="px-4 py-2 text-xs text-gray-400 border-t border-white/10">
                        Tip: Use diagnostics to check health and repair indexes before opening them for chat.
                      </div>
                      <button onClick={()=>handleRebuild(idx, false)} className="block w-full text-left px-4 py-2 hover:bg-white/10">Rebuild changed only</button>
                      <button onClick={()=>handleRebuild(idx, true)} className="block w-full text-left px-4 py-2 hover:bg-white/10">Force rebuild</button>
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
