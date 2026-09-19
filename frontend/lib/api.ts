// Typed HTTP client for the Knowledge Assistant API. The frontend never
// imports Python RAG modules directly — every RAG capability is reached
// through these calls, over the boundary defined in backend/api/.

import type {
  ChatResponse,
  LlmProvider,
  ChunkEmbedding,
  ChunksResponse,
  HealthResponse,
  UploadResponse,
} from "./types";

// Same-origin: every call goes to this app's own route handler, which adds the
// API key server-side and forwards to Python (app/api/[...path]/route.ts).
//
// Deliberately not the backend URL. Pointing the browser straight at Python
// would mean shipping the API key in the bundle, where anyone can read it, and
// would put CORS back in the way. It also means the backend address is a
// server-side setting (BACKEND_URL) rather than one baked in at build time.
//
// The paths below are RESOURCE paths with no /api prefix of their own — this
// constant supplies it exactly once. Writing "/api/documents" here as well
// produced "/api/api/documents", which the proxy dutifully forwarded to a
// backend route that does not exist.
const API_BASE_URL = "/api";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** True when the request never reached a server at all (offline, wrong port, CORS). */
export function isBackendUnreachable(error: unknown): boolean {
  return error instanceof ApiError && error.status === 0;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      "Can't reach the Knowledge Assistant API. Make sure the backend is running and NEXT_PUBLIC_API_URL is correct.",
      0,
    );
  }

  if (!response.ok) {
    let detail = response.statusText || `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // Error body wasn't JSON; keep statusText.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<UploadResponse>("/documents", { method: "POST", body: formData });
}

/** `provider` selects which model answers. Omitted means the server default.
 *  Only the NAME crosses the wire — both API keys stay server-side. */
export function askQuestion(
  sessionId: string,
  question: string,
  provider?: LlmProvider,
  topK?: number,
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      question,
      provider: provider ?? null,
      top_k: topK ?? null,
    }),
  });
}

export function resetConversation(sessionId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/sessions/${sessionId}/reset`, { method: "POST" });
}

export function deleteSession(sessionId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/sessions/${sessionId}`, { method: "DELETE" });
}

/** Stored chunks plus a short preview of each embedding. */
export function getChunks(sessionId: string): Promise<ChunksResponse> {
  return request<ChunksResponse>(`/documents/${sessionId}/chunks`);
}

/** All 384 values for one chunk — requested only when a reader expands it. */
export function getChunkEmbedding(sessionId: string, chunkId: number): Promise<ChunkEmbedding> {
  return request<ChunkEmbedding>(`/documents/${sessionId}/chunks/${chunkId}/embedding`);
}
