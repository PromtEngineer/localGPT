import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ChatSettingsModal, type SettingOption } from '@/components/ui/chat-settings-modal'

// Render smoke for the curated settings layout. The modal renders a hand-laid
// layout (gridToggleLabels + per-section label filters), NOT the raw options
// array — so a control can be present in the array yet never displayed. These
// tests pin the reflection controls to the actual rendered DOM so a future
// toggle can't silently go missing again.

function reflectionOptions(reflectChecked: boolean): SettingOption[] {
  return [
    { type: 'toggle', label: 'Self-reflection', checked: reflectChecked, setter: vi.fn() },
    { type: 'toggle', label: 'Multi-turn query rewrite', checked: false, setter: vi.fn() },
    {
      type: 'dropdown',
      label: 'Reflection model',
      value: '',
      setter: vi.fn(),
      options: [
        { value: '', label: 'Same as answer model' },
        { value: 'qwen3:0.6b', label: 'qwen3:0.6b' },
      ],
    },
    { type: 'slider', label: 'Max reflection loops', value: 2, setter: vi.fn(), min: 1, max: 3, unit: ' loops' },
    { type: 'slider', label: 'Relevance threshold', value: 1, setter: vi.fn(), min: 0, max: 2, unit: '/2' },
    { type: 'slider', label: 'Groundedness threshold', value: 1, setter: vi.fn(), min: 0, max: 2, unit: '/2' },
  ]
}

describe('ChatSettingsModal — reflection controls', () => {
  it('renders both reflection toggles (regression: they were in the array but unrendered)', () => {
    render(<ChatSettingsModal options={reflectionOptions(false)} onClose={() => {}} />)
    expect(screen.getByText('Self-reflection')).toBeInTheDocument()
    expect(screen.getByText('Multi-turn query rewrite')).toBeInTheDocument()
  })

  it('hides the advanced controls until self-reflection is on', () => {
    render(<ChatSettingsModal options={reflectionOptions(false)} onClose={() => {}} />)
    expect(screen.queryByText('Reflection model')).not.toBeInTheDocument()
    expect(screen.queryByText('Max reflection loops')).not.toBeInTheDocument()
    expect(screen.queryByText('Relevance threshold')).not.toBeInTheDocument()
    expect(screen.queryByText('Groundedness threshold')).not.toBeInTheDocument()
  })

  it('reveals the advanced controls (model, loops, thresholds) when on', () => {
    render(<ChatSettingsModal options={reflectionOptions(true)} onClose={() => {}} />)
    expect(screen.getByText('Reflection model')).toBeInTheDocument()
    expect(screen.getByText('Max reflection loops')).toBeInTheDocument()
    expect(screen.getByText('Relevance threshold')).toBeInTheDocument()
    expect(screen.getByText('Groundedness threshold')).toBeInTheDocument()
  })

  it('omits the whole section when the host does not provide it (e.g. quick chat)', () => {
    const opts: SettingOption[] = [
      { type: 'toggle', label: 'Verify answer', checked: false, setter: vi.fn() },
    ]
    render(<ChatSettingsModal options={opts} onClose={() => {}} />)
    expect(screen.queryByText('Self-reflection')).not.toBeInTheDocument()
    expect(screen.queryByText(/Reflection & Multi-turn/)).not.toBeInTheDocument()
  })
})
