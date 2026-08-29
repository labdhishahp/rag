// Mirrors backend/api/rag_bridge.py's JSON shapes exactly — this file is the
// contract between the frontend and the Python API. If the backend response
// shape changes, update it here first.

export interface DocumentMetadata {
  filename: string;
  document_id: string | null;
  page_label: string;
  page_count: number;
  chunk_count: number;
  embedding_dimension: number;
  status: string;
}

export interface UploadResponse {
  session_id: string;
  document: DocumentMetadata;
}

export type EvidenceLevel = "none" | "weak" | "ok";
export type AnswerDepth = "brief" | "normal" | "detailed";

export interface Passage {
  text: string;
  document_name: string | null;
  page_number: number;
  page_label: string;
  section: string | null;
  chunk_ids: number[];
  entry_chunk_ids: number[];
  is_pure_expansion: boolean;
  best_similarity: number;
}

export interface ChatResponse {
  question: string;
  retrieval_query: string;
  was_follow_up: boolean;
  answer: string;
  depth: AnswerDepth;
  wants_example: boolean;
  low_confidence: boolean;
  refused: boolean;
  llm_called: boolean;
  llm_model: string | null;
  evidence_level: EvidenceLevel;
  best_similarity: number;
  sources: number[];
  source_citations: string[];
  cited_labels: number[];
  cited_sources: string[];
  entry_chunk_ids: number[];
  expanded_chunk_ids: number[];
  dropped_chunk_ids: number[];
  context_chars: number;
  duplicate_chars_removed: number;
  context_formatted: string;
  passages: Passage[];
  top_k: number;
}

export interface HealthResponse {
  status: string;
  embedding_model_loaded: boolean;
  embedding_model_name: string | null;
  embedding_dimension: number | null;
  llm_configured: boolean;
  llm_error: string | null;
  active_sessions: number;
}

// One entry in the on-screen conversation. Kept client-side only — the
// backend's Conversation object is the one that matters for retrieval; this
// is purely for rendering the transcript.
export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  result?: ChatResponse;
}
