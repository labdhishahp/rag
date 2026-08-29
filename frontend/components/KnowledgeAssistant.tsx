"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  askQuestion,
  deleteSession,
  getHealth,
  isBackendUnreachable,
  resetConversation,
} from "@/lib/api";
import type { ChatMessage, DocumentMetadata, UploadResponse } from "@/lib/types";
import DocumentUpload from "./DocumentUpload";
import MessageBubble from "./MessageBubble";

type BackendStatus = "checking" | "up" | "down";

function newId(): string {
  return Math.random().toString(36).slice(2);
}

export default function KnowledgeAssistant() {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [llmConfigured, setLlmConfigured] = useState(true);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [document, setDocument] = useState<DocumentMetadata | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const [input, setInput] = useState("");
  const [asking, setAsking] = useState(false);
  const [showDebug, setShowDebug] = useState(true);

  const transcriptRef = useRef<HTMLDivElement>(null);

  const checkHealth = useCallback(async () => {
    try {
      const health = await getHealth();
      setBackendStatus("up");
      setLlmConfigured(health.llm_configured);
    } catch {
      setBackendStatus("down");
    }
  }, []);

  useEffect(() => {
    // One-time reachability probe of an external system (the API process) on
    // mount, not state derived from props/state — the case this lint rule
    // doesn't cover without a data-fetching library, which this project isn't
    // pulling in for a single health check.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void checkHealth();
  }, [checkHealth]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight });
  }, [messages]);

  function handleIndexed(upload: UploadResponse) {
    if (sessionId) {
      // Best-effort cleanup of the previous document's session; a new upload
      // always starts a new conversation, same as the Streamlit sidebar.
      void deleteSession(sessionId).catch(() => undefined);
    }
    setSessionId(upload.session_id);
    setDocument(upload.document);
    setMessages([]);
    setErrorBanner(null);
  }

  async function handleNewConversation() {
    if (!sessionId) return;
    try {
      await resetConversation(sessionId);
      setMessages([]);
      setErrorBanner(null);
    } catch (error) {
      setErrorBanner(error instanceof ApiError ? error.message : "Failed to start a new conversation.");
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || !sessionId || asking) return;

    setInput("");
    setErrorBanner(null);
    setMessages((prev) => [...prev, { id: newId(), role: "user", content: question }]);
    setAsking(true);

    try {
      const result = await askQuestion(sessionId, question);
      setMessages((prev) => [...prev, { id: newId(), role: "assistant", content: result.answer, result }]);
    } catch (error) {
      if (isBackendUnreachable(error)) {
        setBackendStatus("down");
      }
      const message =
        error instanceof ApiError ? error.message : "Something went wrong while generating the answer.";
      setMessages((prev) => [...prev, { id: newId(), role: "error", content: message }]);
    } finally {
      setAsking(false);
    }
  }

  if (backendStatus === "down") {
    return (
      <div className="backend-down">
        <h1>Knowledge Assistant</h1>
        <p>Can&apos;t reach the API backend. Make sure it&apos;s running, then try again.</p>
        <button
          type="button"
          onClick={() => {
            setBackendStatus("checking");
            void checkHealth();
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  const documentReady = sessionId !== null;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <DocumentUpload
          document={document}
          onIndexed={handleIndexed}
          onError={(message) => setErrorBanner(message)}
        />

        <div className="settings-panel">
          <h2>Settings</h2>
          <label className="toggle-row">
            <input type="checkbox" checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} />
            Show retrieval details under each answer
          </label>
          <button type="button" onClick={() => void handleNewConversation()} disabled={messages.length === 0}>
            New conversation
          </button>
          {!llmConfigured && (
            <p className="warning-note">
              The LLM isn&apos;t configured on the server — questions will fail until GEMINI_API_KEY is set.
            </p>
          )}
        </div>
      </aside>

      <main className="chat-main">
        <h1>Knowledge Assistant</h1>
        <p className="subtitle">
          Ask about your document. Follow-ups like &ldquo;explain that in more detail&rdquo; are understood.
        </p>

        {errorBanner && (
          <div className="error-banner" role="alert">
            {errorBanner}
            <button type="button" onClick={() => setErrorBanner(null)} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}

        <div className="transcript" ref={transcriptRef}>
          {messages.length === 0 && (
            <p className="empty-transcript">
              {documentReady ? "Ask a question about your document to get started." : "Upload a document to begin."}
            </p>
          )}
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} showDebug={showDebug} />
          ))}
          {asking && (
            <div className="message message-assistant">
              <div className="message-role">Assistant</div>
              <div className="message-content muted">Retrieving evidence and writing the answer…</div>
            </div>
          )}
        </div>

        <form className="composer" onSubmit={(e) => void handleSubmit(e)}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={documentReady ? "Ask a question about your document…" : "Upload a document to begin"}
            disabled={!documentReady || asking}
          />
          <button type="submit" disabled={!documentReady || asking || !input.trim()}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
