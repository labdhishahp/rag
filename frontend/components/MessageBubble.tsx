"use client";

import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "@/lib/types";
import SourcesPanel from "./SourcesPanel";

export default function MessageBubble({
  message,
  showDebug,
}: {
  message: ChatMessage;
  showDebug: boolean;
}) {
  const { role, content, result } = message;

  return (
    <div className={`message message-${role}`}>
      <div className="message-role">{role === "user" ? "You" : role === "error" ? "Error" : "Assistant"}</div>
      {role === "assistant" ? (
        <div className="message-content markdown">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      ) : (
        <div className="message-content">{content}</div>
      )}
      {role === "assistant" && result && (
        <>
          {result.low_confidence && !result.refused && (
            <p className="low-confidence-caption">⚠️ Low retrieval confidence — verify against the sources.</p>
          )}
          {showDebug && <SourcesPanel result={result} />}
        </>
      )}
    </div>
  );
}
