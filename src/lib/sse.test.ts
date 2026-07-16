import { describe, expect, it } from 'vitest';

import { parseSseFrames } from './sse';

describe('parseSseFrames', () => {
  it('preserves event ids so a durable stream can resume', () => {
    const parsed = parseSseFrames(
      'id: 41\nevent: tool.completed\ndata: {"tool":"search_knowledge"}\n\n' +
      'id: 42\nevent: run.completed\ndata: {"status":"completed"}\n\n',
    );

    expect(parsed.events).toEqual([
      { id: 41, type: 'tool.completed', data: { tool: 'search_knowledge' } },
      { id: 42, type: 'run.completed', data: { status: 'completed' } },
    ]);
    expect(parsed.remainder).toBe('');
  });
});
