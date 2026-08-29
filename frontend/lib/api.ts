// Typed HTTP client for the Knowledge Assistant API. The frontend never
// imports Python RAG modules directly — every RAG capability is reached
// through these calls, over the boundary defined in backend/api/.

import type { ChatResponse, HealthResponse, UploadResponse } from "./types";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

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
  return request<UploadResponse>("/api/documents", { method: "POST", body: formData });
}

export function askQuestion(sessionId: string, question: string, topK?: number): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question, top_k: topK ?? null }),
  });
}

export function resetConversation(sessionId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/sessions/${sessionId}/reset`, { method: "POST" });
}

export function deleteSession(sessionId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/sessions/${sessionId}`, { method: "DELETE" });
}
