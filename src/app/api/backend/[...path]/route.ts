import type { NextRequest } from 'next/server';
import { isAllowedProxyOrigin } from '@/lib/runtime';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

type RouteContext = { params: Promise<{ path: string[] }> };

async function proxy(request: NextRequest, context: RouteContext): Promise<Response> {
  if (
    request.method !== 'GET'
    && request.method !== 'HEAD'
    && !isAllowedProxyOrigin(request.headers.get('origin'), request.nextUrl.host)
  ) {
    return Response.json({ error: 'Cross-origin mutation rejected' }, { status: 403 });
  }
  const { path } = await context.params;
  const backend = (process.env.BACKEND_INTERNAL_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
  const target = `${backend}/${path.map(encodeURIComponent).join('/')}${request.nextUrl.search}`;

  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.delete('content-length');
  headers.delete('connection');
  const token = process.env.LOCALGPT_API_TOKEN;
  if (token) headers.set('authorization', `Bearer ${token}`);

  const hasBody = request.method !== 'GET' && request.method !== 'HEAD';
  const upstreamInit: RequestInit & { duplex?: 'half' } = {
    method: request.method,
    headers,
    body: hasBody ? request.body : undefined,
    cache: 'no-store',
  };
  if (hasBody) upstreamInit.duplex = 'half';
  const upstream = await fetch(target, upstreamInit);

  const responseHeaders = new Headers();
  for (const name of ['content-type', 'cache-control']) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  return new Response(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
