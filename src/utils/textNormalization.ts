/**
 * Comprehensive text normalization utility for cleaning up excessive whitespace
 * in streaming markdown responses to prevent large visual gaps in the UI.
 */

export function normalizeWhitespace(text: string): string {
  if (!text || typeof text !== 'string') {
    return '';
  }

  text = text.replace(/\n{3,}/g, '\n\n');
  
  text = text.replace(/[ \t]+$/gm, '');
  
  text = text.replace(/[ \t]{3,}/g, ' ');
  
  text = text.replace(/[ \t]*\n[ \t]*\n[ \t]*\n/g, '\n\n');
  
  text = text.replace(/[ \t]+\n/g, '\n');
  
  text = text.trim();
  
  return text;
}

/**
 * Specialized normalization for streaming tokens to prevent accumulation
 * of excessive whitespace during real-time text generation.
 */
export function normalizeStreamingToken(currentText: string, newToken: string): string {
  if (!newToken || typeof newToken !== 'string') {
    return currentText;
  }

  let combined = currentText + newToken;
  
  combined = normalizeWhitespace(combined);
  
  return combined;
}

/**
 * Check if text contains excessive whitespace that needs normalization
 */
export function hasExcessiveWhitespace(text: string): boolean {
  if (!text || typeof text !== 'string') {
    return false;
  }
  
  if (/\n{3,}/.test(text)) {
    return true;
  }
  
  if (/[ \t]{3,}/.test(text)) {
    return true;
  }
  
  if (/[ \t]*\n[ \t]*\n[ \t]*\n/.test(text)) {
    return true;
  }
  
  return false;
}
