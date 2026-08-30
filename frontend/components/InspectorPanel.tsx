"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError, getChunkEmbedding, getChunks } from "@/lib/api";
import type { ChunksResponse, StoredChunk } from "@/lib/types";

export type InspectorView = "chunks" | "embeddings";

/**
 * What the uploaded document actually became.
 *
 * Everything here is read back from storage — the same rows retrieval searches
 * — rather than re-chunked or re-embedded for display. If this panel and the
 * answers ever disagreed, the panel would be worse than useless.
 */
export default function InspectorPanel({
  sessionId,
  view,
  onChangeView,
  onClose,
}: {
  sessionId: string;
  view: InspectorView;
  onChangeView: (view: InspectorView) => void;
  onClose: () => void;
}) {
  const [data, setData] = useState<ChunksResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await getChunks(sessionId));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not load the stored chunks.");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const doc = data?.document;

  return (
    <div className="inspector-backdrop" onClick={onClose}>
      <aside className="inspector" onClick={(e) => e.stopPropagation()} aria-label="RAG inspector">
        <header className="inspector-head">
          <div className="inspector-tabs">
            <button
              type="button"
              className={view === "chunks" ? "tab active" : "tab"}
              onClick={() => onChangeView("chunks")}
            >
              Chunks
            </button>
            <button
              type="button"
              className={view === "embeddings" ? "tab active" : "tab"}
              onClick={() => onChangeView("embeddings")}
            >
              Embeddings
            </button>
          </div>
          <button type="button" className="inspector-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        {doc && (
          <p className="inspector-meta">
            <strong>{doc.filename}</strong> · {doc.chunk_count} chunks ·{" "}
            {doc.page_count} {doc.page_label}
            {doc.page_count === 1 ? "" : "s"}
            {view === "embeddings" && (
              <>
                {" "}
                · <code>{doc.embedding_model}</code> ({doc.embedding_provider}) ·{" "}
                <strong>{doc.embedding_dimension}-d</strong>
              </>
            )}
          </p>
        )}

        <div className="inspector-body">
          {loading && <p className="muted">Loading stored chunks…</p>}
          {error && <p className="inspector-error">{error}</p>}
          {!loading && !error && data?.chunks.length === 0 && (
            <p className="muted">This document has no stored chunks.</p>
          )}
          {!loading &&
            !error &&
            data?.chunks.map((chunk) =>
              view === "chunks" ? (
                <ChunkCard key={chunk.chunk_id} chunk={chunk} />
              ) : (
                <EmbeddingCard
                  key={chunk.chunk_id}
                  chunk={chunk}
                  sessionId={sessionId}
                  dimension={doc?.embedding_dimension ?? 0}
                  previewCount={data.preview_values}
                />
              ),
            )}
        </div>
      </aside>
    </div>
  );
}

function ChunkHeading({ chunk }: { chunk: StoredChunk }) {
  return (
    <div className="chunk-head">
      <span className="chunk-id">chunk {chunk.chunk_id}</span>
      <span className="chunk-where">
        {chunk.page_label} {chunk.page_number}
        {chunk.section ? <> · {chunk.section}</> : null}
      </span>
    </div>
  );
}

function ChunkCard({ chunk }: { chunk: StoredChunk }) {
  return (
    <article className="chunk-card">
      <ChunkHeading chunk={chunk} />
      <p className="chunk-text">{chunk.text}</p>
      <p className="chunk-foot">
        {chunk.text.length} chars
        {chunk.char_start !== null && <> · offsets {chunk.char_start}–{chunk.char_end}</>}
        {chunk.prev_chunk_id !== null && <> · prev {chunk.prev_chunk_id}</>}
        {chunk.next_chunk_id !== null && <> · next {chunk.next_chunk_id}</>}
      </p>
    </article>
  );
}

function EmbeddingCard({
  chunk,
  sessionId,
  dimension,
  previewCount,
}: {
  chunk: StoredChunk;
  sessionId: string;
  dimension: number;
  previewCount: number;
}) {
  const [full, setFull] = useState<number[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  async function expand() {
    setOpen(true);
    if (full || busy) return;
    setBusy(true);
    try {
      const result = await getChunkEmbedding(sessionId, chunk.chunk_id);
      setFull(result.values);
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not load the full vector.");
    } finally {
      setBusy(false);
    }
  }

  const shown = open && full ? full : chunk.embedding_preview;

  return (
    <article className="chunk-card">
      <ChunkHeading chunk={chunk} />
      {/* The text sits directly above its vector so the pairing is obvious. */}
      <p className="chunk-text embedding-source">{chunk.text}</p>

      <div className="vector-block">
        <div className="vector-label">
          embedding · {dimension} values
          {chunk.embedding_norm !== null && (
            <> · ‖v‖ = {chunk.embedding_norm.toFixed(4)}</>
          )}
        </div>
        <code className="vector-values">
          [{shown.map((v) => v.toFixed(4)).join(", ")}
          {!(open && full) && dimension > previewCount ? ", …" : ""}]
        </code>
        <div className="vector-actions">
          {!open || !full ? (
            <button type="button" className="disclosure" onClick={() => void expand()} disabled={busy}>
              {busy ? "Loading…" : `Show all ${dimension} values`}
            </button>
          ) : (
            <button type="button" className="disclosure" onClick={() => setOpen(false)}>
              Show first {previewCount}
            </button>
          )}
          {error && <span className="inspector-error"> {error}</span>}
        </div>
      </div>
    </article>
  );
}
