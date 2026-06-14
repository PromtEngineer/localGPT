import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import { MetricsFooter } from '@/components/ui/metrics-footer'

// Guards the per-answer footer's display logic: reflection round wording and
// the ms->s timing format. Uses container.textContent because the strings are
// composed from several JSX text nodes.
const norm = (el: HTMLElement) => (el.textContent || '').replace(/\s+/g, ' ').trim()

describe('MetricsFooter', () => {
  it('renders nothing when neither timings nor reflection is present', () => {
    const { container } = render(<MetricsFooter />)
    expect(container.firstChild).toBeNull()
  })

  it('pluralizes reflection rounds and shows the scores', () => {
    const { container } = render(
      <MetricsFooter reflection={{ rounds: 2, relevance: 2, groundedness: 1 }} />,
    )
    const text = norm(container as HTMLElement)
    expect(text).toContain('2 reflection rounds')
    expect(text).toContain('relevance 2/2')
    expect(text).toContain('groundedness 1/2')
  })

  it('uses the singular "round" for a single round and omits null scores', () => {
    const { container } = render(
      <MetricsFooter reflection={{ rounds: 1, relevance: null, groundedness: null }} />,
    )
    expect(norm(container as HTMLElement)).toBe('↻ 1 reflection round')
  })

  it('formats timings ms->s (>=1000ms) and ms (<1000ms)', () => {
    const { container } = render(
      <MetricsFooter
        timings={{ total: 274430, retrieval: 1690, context_expand: 19 }}
      />,
    )
    const text = norm(container as HTMLElement)
    expect(text).toContain('274.4s total')
    expect(text).toContain('retrieval 1.7s')
    expect(text).toContain('context expand 19ms')
  })
})
