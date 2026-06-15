import { describe, it, expect } from 'vitest'
import { render, fireEvent } from '@testing-library/react'
import { ActivityTrace } from '@/components/ui/activity-trace'
import type { ActivityStage } from '@/lib/activity-trace'

const STAGES: ActivityStage[] = [
  { key: 'retrieval', label: 'Retrieval', status: 'done', detail: '12 chunks' },
  { key: 'rerank', label: 'Reranking', status: 'active' },
]

describe('ActivityTrace', () => {
  it('renders nothing without stages', () => {
    const { container } = render(<ActivityTrace />)
    expect(container.firstChild).toBeNull()
    const { container: c2 } = render(<ActivityTrace stages={[]} />)
    expect(c2.firstChild).toBeNull()
  })

  it('shows a collapsed done/total summary and hides the list', () => {
    const { container, queryByText } = render(<ActivityTrace stages={STAGES} />)
    expect((container.textContent || '')).toContain('Activity · 1/2 steps')
    expect(queryByText('Reranking')).toBeNull() // collapsed by default
  })

  it('expands to the per-stage list on click', () => {
    const { getByRole, getByText } = render(<ActivityTrace stages={STAGES} />)
    fireEvent.click(getByRole('button'))
    expect(getByText('Retrieval')).toBeTruthy()
    expect(getByText('Reranking')).toBeTruthy()
    expect(getByText(/12 chunks/)).toBeTruthy()
  })
})
