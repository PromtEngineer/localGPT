// Compact, muted footer surfacing per-answer timing + self-reflection data.
// Both inputs are optional: timings_ms only exists when LOCALGPT_TIMINGS is on,
// reflection only when the reflect flag was set for the request.
export type TimingsMs = {
  retrieval?: number; rerank?: number; context_expand?: number;
  prune?: number; generation?: number; total?: number;
}
export type ReflectionInfo = { rounds: number; relevance: number | null; groundedness: number | null }

export function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

export function MetricsFooter({ timings, reflection }: { timings?: TimingsMs; reflection?: ReflectionInfo }) {
  if (!timings && !reflection) return null

  const stages: Array<[string, number | undefined]> = timings ? [
    ['retrieval', timings.retrieval],
    ['rerank', timings.rerank],
    ['context expand', timings.context_expand],
    ['prune', timings.prune],
    ['generation', timings.generation],
  ] : []
  const breakdown = stages
    .filter(([, v]) => typeof v === 'number')
    .map(([label, v]) => `${label} ${fmtMs(v as number)}`)
    .join(' · ')

  return (
    <div className="text-xs text-gray-500 space-y-0.5 pt-1">
      {reflection && (
        <div>
          ↻ {reflection.rounds} reflection round{reflection.rounds === 1 ? '' : 's'}
          {reflection.relevance !== null && ` · relevance ${reflection.relevance}/2`}
          {reflection.groundedness !== null && ` · groundedness ${reflection.groundedness}/2`}
        </div>
      )}
      {timings && typeof timings.total === 'number' && (
        <div>⏱ {fmtMs(timings.total)} total{breakdown ? ` · ${breakdown}` : ''}</div>
      )}
    </div>
  )
}
