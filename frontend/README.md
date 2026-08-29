# Frontend — Knowledge Assistant

Next.js (App Router, TypeScript) chat UI. Talks to the backend (`../backend/`)
only over HTTP via `lib/api.ts` — no Python is imported here.

## Run

```bash
npm install
cp .env.example .env.local   # NEXT_PUBLIC_API_URL, default http://localhost:8000
npm run dev
```

Requires the backend running (see `../backend/README.md`).

## Structure

```
app/                  # App Router: layout, page, global styles
components/           # KnowledgeAssistant (orchestrator), DocumentUpload,
                       # MessageBubble, SourcesPanel
lib/api.ts            # typed fetch client — the only place that calls the API
lib/types.ts          # response shapes, mirrors backend/api/rag_bridge.py
e2e/                  # live Playwright end-to-end test
```

## Test

```bash
npx tsc --noEmit      # types
npm run lint          # eslint
npm run build         # production build
npm run test:e2e      # live: starts backend+frontend, drives a real browser,
                       # calls the real Gemini API
```

`test:e2e` needs a working `GEMINI_API_KEY` in the project-root `.env` — it's a
deliberate, occasional full-stack check, not something to run on every save.

## Notes

- Answers render as Markdown (`react-markdown`) to match what Streamlit's
  `st.markdown` produced; inline LaTeX (`$...$`) is not rendered as math —
  Streamlit had that via a KaTeX plugin, deliberately not reproduced here to
  avoid an extra dependency for a cosmetic feature.
- Upload progress is a single spinner, not per-step text. The Streamlit
  sidebar showed live "Extracting… / Chunking… / Embedding…" because it ran
  in-process with a callback; over HTTP that would need streaming (SSE), which
  felt like unnecessary complexity for a few seconds of upload time.
