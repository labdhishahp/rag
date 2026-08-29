# Frontend — Knowledge Assistant

Next.js (App Router, TypeScript) chat UI. Talks to the backend only over HTTP
through `lib/api.ts` — no Python is imported here.

## Run

```bash
npm install
cp .env.example .env.local   # NEXT_PUBLIC_API_URL, default http://localhost:8000
npm run dev
```

Requires the backend running (see `../backend/README.md`).

## Structure

```
app/                  layout, page, global styles
components/
  KnowledgeAssistant  top-level state: session, messages, errors
  DocumentUpload      file input + indexing status
  MessageBubble       one message, Markdown-rendered
  SourcesPanel        citations + what retrieval actually did
lib/api.ts            the only place that calls the API
lib/types.ts          response shapes, mirroring backend/api/rag_bridge.py
```

## Checks

```bash
npx tsc --noEmit   # types
npm run lint       # eslint
npm run build      # production build
```

## Notes

- Answers render as Markdown (`react-markdown`). Inline LaTeX is not rendered
  as maths — deliberately not worth an extra dependency here.
- Upload progress is a single spinner rather than per-step text. Showing live
  "extracting / chunking / embedding" would need streaming (SSE), which is more
  machinery than a few seconds of upload justifies.
- `NEXT_PUBLIC_API_URL` is read at **build** time, so changing it requires a
  rebuild, not just an environment change.
