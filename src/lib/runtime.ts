export function getApiBaseUrl(configuredUrl?: string): string {
  return (configuredUrl || '/api/backend').replace(/\/+$/, '');
}

export function apiHeaders(json = false): HeadersInit {
  const headers: Record<string, string> = {};
  if (json) headers['Content-Type'] = 'application/json';
  return headers;
}

export function isAllowedProxyOrigin(origin: string | null, requestHost: string): boolean {
  if (!origin) return true;
  try {
    return new URL(origin).host === requestHost;
  } catch {
    return false;
  }
}
