/**
 * Whitespace normalization for model answers before markdown rendering.
 *
 * Runs OUTSIDE fenced code blocks only: collapsing space runs or blank lines
 * inside ``` fences would corrupt code indentation and ASCII tables that
 * answers frequently quote verbatim from documents.
 */

function normalizeProse(text: string): string {
  // Cap paragraph gaps at one blank line (with or without stray indentation).
  text = text.replace(/[ \t]*\n[ \t]*\n[\s]*\n/g, '\n\n');
  text = text.replace(/\n{3,}/g, '\n\n');
  // Trailing whitespace on lines (also neutralizes markdown two-space hard
  // breaks, which models emit accidentally — remark-breaks already renders
  // intentional single newlines as breaks).
  text = text.replace(/[ \t]+$/gm, '');
  // Long horizontal space runs read as layout accidents in prose.
  text = text.replace(/[ \t]{3,}/g, ' ');
  return text;
}

export function normalizeWhitespace(text: string): string {
  if (!text || typeof text !== 'string') {
    return '';
  }
  // Split on fenced blocks; even segments are prose, odd segments are code.
  const parts = text.split(/(```[\s\S]*?(?:```|$))/);
  const out = parts
    .map((seg, i) => (i % 2 === 1 ? seg : normalizeProse(seg)))
    .join('');
  return out.trim();
}

/**
 * Specialized normalization for streaming tokens to prevent accumulation
 * of excessive whitespace during real-time text generation.
 */
export function normalizeStreamingToken(currentText: string, newToken: string): string {
  if (!newToken || typeof newToken !== 'string') {
    return currentText;
  }
  return normalizeWhitespace(currentText + newToken);
}
