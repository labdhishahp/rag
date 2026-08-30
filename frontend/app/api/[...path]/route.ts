/**
 * Server-side proxy to the Python API.
 *
 * The browser talks to this route, not to the backend directly. Two reasons,
 * both of which disappear if the browser calls the backend itself:
 *
 *  1. The API key stays secret. Anything the browser holds is readable in
 *     devtools, so a key shipped to the client authenticates the whole
 *     internet. BACKEND_API_KEY has no NEXT_PUBLIC_ prefix, so Next.js never
 *     puts it in the bundle; it is read here, on the server.
 *
 *  2. No CORS. These requests are same-origin from the browser's point of
 *     view, and the hop to Python is server-to-server, where CORS does not
 *     apply at all.
 *
 * It also forwards the caller's address as X-Client-Id so the backend can rate
 * limit per visitor rather than seeing every request come from this one server.
 */

import { NextRequest } from "next/server";

// Deployed: API_INTERNAL_URL is injected by the Vercel service binding declared
// in vercel.json. The FastAPI service has NO public rewrite, so that binding is
// the only way to reach it — the browser cannot call it even if it wants to.
// Internal calls skip the public request pipeline entirely.
//
// Locally there is no binding, so fall back to the address uvicorn is serving on.
const BACKEND_URL = (
  process.env.API_INTERNAL_URL ?? process.env.BACKEND_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

// Kept even though the service is already unreachable from the internet. The
// binding grants access but does not authenticate, so this stays as the second
// layer: if a public rewrite to the api service were ever added by mistake, the
// key still holds the line. It is server-side in both services and never
// reaches a browser.
const BACKEND_API_KEY = process.env.BACKEND_API_KEY ?? "";

// Headers that belong to the browser's connection to Next.js and would be
// wrong (or actively harmful) to replay on a new request to Python.
const STRIP = new Set([
  "host",
  "connection",
  "content-length",
  "accept-encoding",
  "x-api-key",       // never let a caller supply their own
  "x-client-id",     // set from the real address below, not from user input
]);

function clientAddress(request: NextRequest): string {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) return forwarded.split(",")[0].trim();
  return request.headers.get("x-real-ip") ?? "unknown";
}

async function proxy(request: NextRequest, path: string[]) {
  // A request for /api/api/... means the caller prefixed "/api" onto a path
  // that already had one. Forwarding it produces a 404 from Python that looks
  // like a missing document rather than a client bug, so name it here instead.
  if (path[0] === "api") {
    return Response.json(
      {
        detail:
          `Bad request path /api/${path.join("/")}: the "/api" prefix is added once by ` +
          `lib/api.ts. Request "/${path.slice(1).join("/")}" instead.`,
      },
      { status: 400 },
    );
  }

  const search = request.nextUrl.search;
  // /health is the one backend route that does not live under /api — it is
  // unauthenticated on purpose so uptime monitoring can reach it.
  const suffix = path.length === 1 && path[0] === "health" ? "health" : `api/${path.join("/")}`;
  const target = `${BACKEND_URL}/${suffix}${search}`;

  const headers = new Headers();
  request.headers.forEach((value, key) => {
    if (!STRIP.has(key.toLowerCase())) headers.set(key, value);
  });
  if (BACKEND_API_KEY) headers.set("x-api-key", BACKEND_API_KEY);
  headers.set("x-client-id", clientAddress(request));

  // GET/HEAD must not carry a body; everything else is streamed straight
  // through, which is what keeps file uploads working without buffering.
  const method = request.method;
  const hasBody = method !== "GET" && method !== "HEAD";

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method,
      headers,
      body: hasBody ? await request.arrayBuffer() : undefined,
      cache: "no-store",
    });
  } catch {
    return Response.json(
      { detail: "Can't reach the Knowledge Assistant API. Is the backend running?" },
      { status: 502 },
    );
  }

  const body = await upstream.arrayBuffer();
  const responseHeaders = new Headers();
  const contentType = upstream.headers.get("content-type");
  if (contentType) responseHeaders.set("content-type", contentType);
  const retryAfter = upstream.headers.get("retry-after");
  if (retryAfter) responseHeaders.set("retry-after", retryAfter);

  return new Response(body, { status: upstream.status, headers: responseHeaders });
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, ctx: Ctx) {
  return proxy(request, (await ctx.params).path);
}
export async function POST(request: NextRequest, ctx: Ctx) {
  return proxy(request, (await ctx.params).path);
}
export async function DELETE(request: NextRequest, ctx: Ctx) {
  return proxy(request, (await ctx.params).path);
}
