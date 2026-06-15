/**
 * Generic activity trace for the RAG pipeline.
 *
 * The backend already streams every stage over SSE using a uniform
 * `<stage>_started` / `<stage>_done` naming convention (retrieval, rerank,
 * context_expand, prune, generation, ...). Rather than hard-code each event to
 * a fixed UI slot, this folds the raw event stream into an ordered list of
 * stages by that convention — so any new `foo_started`/`foo_done` the pipeline
 * emits shows up automatically, humanized, with no UI change.
 */

export type ActivityStatus = 'active' | 'done';

export interface ActivityStage {
  key: string;
  label: string;
  status: ActivityStatus;
  detail?: string;
}

export type RawEvent = { type: string; data?: Record<string, unknown> };

// Friendly labels for known stage keys; unknown keys fall back to title-case so
// the trace degrades gracefully as the pipeline grows new stages.
const STAGE_LABELS: Record<string, string> = {
  index_selection: 'Index selection',
  query_rewrite: 'Query rewrite',
  rewrite: 'Query rewrite',
  decomposition: 'Sub-query planning',
  retrieval: 'Retrieval',
  rerank: 'Reranking',
  context_expand: 'Context expansion',
  prune: 'Context pruning',
  generation: 'Answer generation',
  reflection: 'Self-reflection',
  verification: 'Verification',
  web_search: 'Web search',
  skill: 'Skill loading',
};

// Streaming/terminal events that carry no stage meaning for the trace.
const IGNORED = new Set([
  'token',
  'sub_query_token',
  'complete',
  'error',
  'final_answer',
  'direct_answer',
]);

export function humanizeStage(key: string): string {
  return (
    STAGE_LABELS[key] ??
    key.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())
  );
}

function detailFromPayload(data?: Record<string, unknown>): string | undefined {
  if (!data) return undefined;
  if (typeof data.count === 'number') {
    return `${data.count} chunk${data.count === 1 ? '' : 's'}`;
  }
  return undefined;
}

/**
 * Fold an ordered SSE event stream into deduplicated, ordered stages. A
 * `<key>_started` opens a stage (status 'active'); the matching `<key>_done`
 * closes it (status 'done'). Stages appear in first-seen order.
 */
export function foldActivityEvents(events: RawEvent[]): ActivityStage[] {
  const order: string[] = [];
  const byKey: Record<string, ActivityStage> = {};

  const upsert = (key: string, status: ActivityStatus, detail?: string) => {
    const existing = byKey[key];
    if (!existing) {
      byKey[key] = { key, label: humanizeStage(key), status, detail };
      order.push(key);
      return;
    }
    existing.status = status;
    if (detail !== undefined) existing.detail = detail;
  };

  for (const evt of events) {
    if (!evt || typeof evt.type !== 'string' || IGNORED.has(evt.type)) continue;
    const m = evt.type.match(/^(.+)_(started|done)$/);
    if (!m) continue;
    const [, key, phase] = m;
    upsert(key, phase === 'done' ? 'done' : 'active', detailFromPayload(evt.data));
  }

  return order.map((k) => byKey[k]);
}
