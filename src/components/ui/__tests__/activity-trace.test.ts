import { describe, it, expect } from 'vitest'
import { foldActivityEvents, humanizeStage } from '@/lib/activity-trace'

// The fold is the generic contract: pair <stage>_started/_done by suffix, keep
// first-seen order, ignore streaming noise, and humanize unknown stages.
describe('foldActivityEvents', () => {
  it('pairs started/done into one stage and marks it done', () => {
    const stages = foldActivityEvents([
      { type: 'retrieval_started', data: {} },
      { type: 'retrieval_done', data: { count: 12 } },
    ])
    expect(stages).toHaveLength(1)
    expect(stages[0]).toMatchObject({ key: 'retrieval', label: 'Retrieval', status: 'done' })
    expect(stages[0].detail).toBe('12 chunks')
  })

  it('keeps an unfinished stage active', () => {
    const stages = foldActivityEvents([{ type: 'rerank_started', data: {} }])
    expect(stages[0]).toMatchObject({ key: 'rerank', status: 'active' })
  })

  it('preserves first-seen order across multiple stages', () => {
    const stages = foldActivityEvents([
      { type: 'retrieval_started' },
      { type: 'retrieval_done' },
      { type: 'rerank_started' },
      { type: 'rerank_done' },
      { type: 'generation_started' },
      { type: 'generation_done' },
    ])
    expect(stages.map((s) => s.key)).toEqual(['retrieval', 'rerank', 'generation'])
  })

  it('ignores streaming/terminal noise', () => {
    const stages = foldActivityEvents([
      { type: 'token', data: { text: 'hi' } },
      { type: 'complete', data: {} },
      { type: 'error', data: {} },
    ])
    expect(stages).toEqual([])
  })

  it('humanizes an unknown stage key via title-case fallback', () => {
    expect(humanizeStage('some_new_stage')).toBe('Some new stage')
    const stages = foldActivityEvents([{ type: 'some_new_stage_started' }])
    expect(stages[0].label).toBe('Some new stage')
  })

  it('singularizes the chunk count', () => {
    const stages = foldActivityEvents([{ type: 'prune_done', data: { count: 1 } }])
    expect(stages[0].detail).toBe('1 chunk')
  })
})
