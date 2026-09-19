"use client";

import { useRef, useState } from "react";
import { ApiError, uploadDocument } from "@/lib/api";
import type { DocumentMetadata, UploadResponse } from "@/lib/types";

export default function DocumentUpload({
  document,
  onIndexed,
  onError,
  disabledReason = null,
}: {
  document: DocumentMetadata | null;
  onIndexed: (upload: UploadResponse) => void;
  onError: (message: string) => void;
  /** When set, uploading is blocked and this explains why. */
  disabledReason?: string | null;
}) {
  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setFileName(file.name);
    setUploading(true);
    try {
      const result = await uploadDocument(file);
      onIndexed(result);
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "Failed to upload the document.";
      onError(message);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="document-panel">
      <h2>Document</h2>
      <label className={disabledReason ? "upload-control is-disabled" : "upload-control"}>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx"
          disabled={uploading || disabledReason !== null}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void handleFile(file);
          }}
        />
        <span>
          {disabledReason
            ? "Upload unavailable"
            : uploading
              ? `Indexing ${fileName ?? "document"}…`
              : "Upload a PDF or Word document"}
        </span>
      </label>
      {disabledReason ? (
        <p className="warning-note">{disabledReason}</p>
      ) : (
        <p className="hint">One document at a time. A new upload starts a new conversation.</p>
      )}

      {uploading && <div className="spinner" role="status" aria-label="Indexing document" />}

      {!uploading && document && (
        <div className="document-status">
          <p className="status-ok">Indexed</p>
          <p className="document-meta">
            <strong>{document.filename}</strong>
            <br />
            {document.page_count} {document.page_label}
            {document.page_count === 1 ? "" : "s"} · {document.chunk_count} chunks ·{" "}
            {document.embedding_dimension}-d embeddings
          </p>
        </div>
      )}
    </div>
  );
}
