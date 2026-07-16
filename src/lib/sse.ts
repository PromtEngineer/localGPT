export interface DurableEvent<T = Record<string, unknown>> {
  id: number;
  type: string;
  data: T;
}

export function parseSseFrames(
  buffer: string,
): { events: DurableEvent[]; remainder: string } {
  const frames = buffer.replace(/\r\n/g, '\n').split('\n\n');
  const remainder = frames.pop() ?? '';
  const events: DurableEvent[] = [];

  for (const frame of frames) {
    let id: number | undefined;
    let type = 'message';
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (!line || line.startsWith(':')) continue;
      const separator = line.indexOf(':');
      const field = separator === -1 ? line : line.slice(0, separator);
      const value = separator === -1 ? '' : line.slice(separator + 1).replace(/^ /, '');
      if (field === 'id') id = Number.parseInt(value, 10);
      if (field === 'event') type = value;
      if (field === 'data') data.push(value);
    }
    if (id === undefined || Number.isNaN(id) || data.length === 0) continue;
    events.push({ id, type, data: JSON.parse(data.join('\n')) as Record<string, unknown> });
  }
  return { events, remainder };
}
