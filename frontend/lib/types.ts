// Mirrors backend/api/rag_bridge.py's JSON shapes exactly — this file is the
// contract between the frontend and the Python API. If the backend response
// shape changes, update it here first.

export interface DocumentMetadata {
  filename: string;
  document_id: string | null;
  page_label: string;
  page_count: number;
  chunk_count: number;
  /** Which embedding space this document's vectors live in. Recorded at index
   *  time and used for every later query — see src/embeddings.py. */
  embedding_provider: string | null;
  embedding_model: string | null;
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

export interface EmbeddingProviderHealth {
  available: boolean;
  model?: string;
  dimension?: number;
  error?: string;
}

export interface HealthResponse {
  /** "ok" only when storage AND at least one embedding provider are reachable. */
  status: "ok" | "degraded";
  embedding_primary: string;
  embedding_providers: Record<string, EmbeddingProviderHealth>;
  llm_configured: boolean;
  llm_error: string | null;
  /** Which storage backend is in use, e.g. "PostgresStorage". */
  storage: string;
  /** False when the database cannot be reached. An upload WILL fail. */
  storage_ok: boolean;
  /** Deliberately vague server-side; the detail stays in the server log. */
  storage_error: string | null;
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

// ---- RAG inspection --------------------------------------------------------
// What the document actually became: the stored chunks and their vectors,
// read back from the database rather than recomputed.

export interface StoredChunk {
  chunk_id: number;
  page_number: number | null;
  page_label: string;
  section: string | null;
  text: string;
  char_start: number | null;
  char_end: number | null;
  prev_chunk_id: number | null;
  next_chunk_id: number | null;
  /** The first few values of the vector; the rest is fetched on expand. */
  embedding_preview: number[];
  embedding_norm: number | null;
}

export interface ChunksResponse {
  document: DocumentMetadata;
  preview_values: number;
  chunks: StoredChunk[];
}

export interface ChunkEmbedding {
  chunk_id: number;
  dimension: number;
  model: string | null;
  provider: string | null;
  values: number[];
}
