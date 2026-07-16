import { describe, expect, it } from 'vitest';

import { getApiBaseUrl, isAllowedProxyOrigin } from './runtime';

describe('getApiBaseUrl', () => {
  it('uses the configured backend URL without a trailing slash', () => {
    expect(getApiBaseUrl('https://localgpt.example/')).toBe('https://localgpt.example');
  });

  it('defaults to the same-origin backend proxy', () => {
    expect(getApiBaseUrl()).toBe('/api/backend');
  });
});

describe('isAllowedProxyOrigin', () => {
  it('rejects cross-origin requests at the token-injecting proxy', () => {
    expect(isAllowedProxyOrigin('https://evil.example', 'localgpt.example')).toBe(false);
    expect(isAllowedProxyOrigin('https://localgpt.example', 'localgpt.example')).toBe(true);
    expect(isAllowedProxyOrigin(null, 'localgpt.example')).toBe(true);
  });
});
