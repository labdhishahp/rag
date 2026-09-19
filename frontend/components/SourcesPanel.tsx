"use client";

import { useState } from "react";
import type { ChatResponse } from "@/lib/types";

/**
 * "Sources and retrieval details" — the same inspectable pipeline view the
 * Streamlit app rendered per answer: which passages were matched vs. added
 * by expansion, the evidence level, and the exact text sent to the model.
 */
export default function SourcesPanel({ result }: { result: ChatResponse }) {
  const [open, setOpen] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const citedSet = new Set(result.cited_labels);

  return (
    <div className="sources-panel">
      <button type="button" className="disclosure" onClick={() => setOpen((v) => !v)}>
        {open ? "▾" : "▸"} Sources and retrieval details
      </button>
      {open && (
        <div className="disclosure-body">
          <div className="sources-block">
            <h4>Sources</h4>
            {result.source_citations.length === 0 ? (
              <p className="muted">No passages retrieved.</p>
            ) : (
              <ul className="citation-list">
                {result.source_citations.map((citation, i) => {
                  const label = i + 1;
                  return (
                    <li key={label}>
                      <span className={citedSet.has(label) ? "cite-mark used" : "cite-mark"}>
                        {citedSet.has(label) ? "✓" : "·"}
                      </span>{" "}
                      <code>[S{label}]</code> {citation}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          <div className="retrieval-block">
            <h4>Retrieval</h4>
            <ul className="detail-list">
              <li>
                Request read as: <code>{result.depth}</code>
                {result.wants_example ? " · asked for an example" : ""}
              </li>
              {result.was_follow_up && (
                <li>
                  Follow-up detected. Retrieval searched for: <em>{result.retrieval_query}</em>
                </li>
              )}
              <li>
                Matched chunks: <code>{JSON.stringify(result.entry_chunk_ids)}</code> · added
                neighbours: <code>{JSON.stringify(result.expanded_chunk_ids)}</code>
                {result.dropped_chunk_ids.length > 0 && (
                  <>
                    {" "}
                    · dropped for budget: <code>{JSON.stringify(result.dropped_chunk_ids)}</code>
                  </>
                )}
              </li>
              <li>
                Evidence level: <code>{result.evidence_level}</code> (best similarity{" "}
                {result.best_similarity.toFixed(3)})
              </li>
              <li>
                Context sent: {result.context_chars} chars in {result.passages.length} passage(s); overlap
                removed: {result.duplicate_chars_removed} chars
              </li>
              <li>
                LLM called: {result.llm_called ? "yes" : "no (declined on evidence)"}
                {result.llm_provider ? <> · <code>{result.llm_provider}</code></> : null}
                {result.llm_model ? <> · model <code>{result.llm_model}</code></> : null}
              </li>
            </ul>
          </div>

          <button type="button" className="disclosure nested" onClick={() => setShowEvidence((v) => !v)}>
            {showEvidence ? "▾" : "▸"} Evidence passages exactly as sent to the model
          </button>
          {showEvidence && <pre className="evidence-block">{result.context_formatted}</pre>}
        </div>
      )}
    </div>
  );
}
