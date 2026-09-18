"""
A knowledge base of several documents behind one retriever.

Before Phase 5 the app held exactly one document: a new upload wiped the
index. VectorStore already keyed chunks by (document_id, chunk_id) and add()
already appended, so multi-document support is a registry and a filter, not
a new store.

    KnowledgeBase
      ├── embedding_model        one model for every document (same vector space)
      ├── vector_store           one numpy matrix, chunks from all documents
      ├── retriever              pure similarity search, optional document filter
      └── documents              {document_id: DocumentInfo}   the registry

The registry is what lets a question mention a document BY NAME
("what does the engineering guideline say about ...") and have retrieval
scoped to it, and what lets the assistant list what it knows.

ONE EMBEDDING MODEL, DELIBERATELY
  Every document here is embedded by the same provider. That is not a
  simplification, it is a correctness requirement: a 384-d bge-small vector
  and a 768-d Gemini vector are different coordinate systems, and a similarity
  between them is meaningless even when the arithmetic runs. See embeddings.py.
  A knowledge base spanning two providers would have to be two stores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from chunker import chunk_pages
from document_loader import load_document_from_bytes
from embeddings import EmbeddingModel
from pipeline import DocumentProcessingError, make_document_id
from retriever import Retriever
from vector_store import VectorStore


@dataclass
class DocumentInfo:
    document_id: str
    name: str
    page_label: str
    page_count: int
    chunk_count: int
    sections: list[str] = field(default_factory=list)   # in reading order, unique

    @property
    def stem(self) -> str:
        return re.sub(r"\.(pdf|docx)$", "", self.name, flags=re.IGNORECASE)


class KnowledgeBase:
    def __init__(self, embedding_model: EmbeddingModel | None = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = VectorStore(dimension=self.embedding_model.dimension)
        self.retriever = Retriever(self.embedding_model, self.vector_store)
        self.documents: dict[str, DocumentInfo] = {}

    # ---- ingestion -------------------------------------------------------

    def add_document(self, file_bytes: bytes, filename: str,
                     chunk_size: int = 500, chunk_overlap: int = 50) -> DocumentInfo:
        """
        Index one document. Re-adding identical bytes is a no-op (content
        hash), so the same file uploaded twice does not double its chunks.
        """
        if not file_bytes:
            raise DocumentProcessingError("The uploaded file is empty.")
        document_id = make_document_id(file_bytes)
        if document_id in self.documents:
            return self.documents[document_id]

        try:
            pages = load_document_from_bytes(file_bytes, filename)
        except ValueError as exc:
            raise DocumentProcessingError(str(exc)) from exc

        chunks = chunk_pages(
            pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            document_name=filename, document_id=document_id,
        )
        embeddings = self.embedding_model.embed_texts([c["text"] for c in chunks])
        self.vector_store.add(embeddings, chunks)

        sections: list[str] = []
        for c in chunks:
            if c["section"] and (not sections or sections[-1] != c["section"]):
                sections.append(c["section"])

        info = DocumentInfo(
            document_id=document_id,
            name=filename,
            page_label=pages[0].get("page_label", "page") if pages else "page",
            page_count=len(pages),
            chunk_count=len(chunks),
            sections=sections,
        )
        self.documents[document_id] = info
        print(f"\n=== KNOWLEDGE BASE === added {filename} ({document_id}): "
              f"{len(pages)} {info.page_label}s, {len(chunks)} chunks, {len(sections)} sections; "
              f"total documents: {len(self.documents)}")
        return info

    # ---- lookup ----------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return not self.documents

    def list_documents(self) -> list[DocumentInfo]:
        return list(self.documents.values())

    def chunks_for(self, document_id: str) -> list[dict]:
        return self.vector_store.chunks_for_document(document_id)

    def resolve_documents(self, text: str) -> list[DocumentInfo]:
        """
        Which documents does this text mention by name?

        Returns every match at the STRONGEST tier that fired, so the caller can
        still tell "one document named" from "genuinely ambiguous" from "none".

        ----------------------------------------------------------------------
        WHY TIERS, AND NOT "EVERY DOCUMENT WHOSE NAME SHARES A WORD"
        ----------------------------------------------------------------------
        The first version returned every document sharing any 4+ character word
        of its filename. Measured against two real test documents, that was
        wrong for every question that named one of them:

            sample.pdf          -> words ['sample']
            formula_sample.pdf  -> words ['formula', 'sample']

        'sample' belongs to both, so "according to sample.pdf" matched BOTH
        documents and the router either asked a pointless clarifying question
        or searched the wrong file. Six of seventeen tests failed on this one
        rule.

        The fix is to rank how a name matched rather than only whether it did:

            tier 3  the filename itself, extension and all   "sample.pdf"
            tier 2  the whole stem, however it was spaced    "formula sample"
            tier 1  one distinctive word of the name         "nist", "bert"

        Highest tier wins outright; within a tier the LONGEST match wins, so
        "formula sample" (tier 2, 14 chars) beats "sample" (tier 2, 6 chars).
        Only a true tie returns several documents — budget_2023.pdf and
        budget_2024.pdf against "the budget" — which is exactly when the
        router SHOULD ask which one.

        Word boundaries use lookaround rather than \\b because '_' is a word
        character: that is what stops 'sample.pdf' matching inside
        'formula_sample.pdf'.
        """
        low = " ".join(text.lower().split())
        scored: list[tuple[int, int, DocumentInfo]] = []
        for info in self.documents.values():
            tier, length = _match_strength(info, low)
            if tier:
                scored.append((tier, length, info))
        if not scored:
            return []
        best_tier = max(tier for tier, _, _ in scored)
        best_length = max(length for tier, length, _ in scored if tier == best_tier)
        return [info for tier, length, info in scored
                if tier == best_tier and length == best_length]


# Filename words too generic to identify a document on their own. Without this,
# "is this the final version of the report?" would match q3_report_final.pdf and
# silently scope a question that meant to search everything. A document whose
# WHOLE name is one of these (report.pdf) still matches at tier 2 — it is only
# barred from identifying a document by one fragment of a longer name.
_GENERIC_NAME_WORDS = frozenset({
    "copy", "data", "doc", "docs", "document", "documents", "download",
    "draft", "export", "file", "files", "final", "latest", "new", "old",
    "output", "page", "pages", "paper", "report", "sample", "samples",
    "scan", "scanned", "temp", "test", "untitled", "updated", "version",
})

# A word boundary that treats '_' as part of a word, unlike \b.
_EDGE_LEFT = r"(?<!\w)"
_EDGE_RIGHT = r"(?!\w)"


def _match_strength(info: DocumentInfo, low: str) -> tuple[int, int]:
    """
    How strongly does `low` name this document? -> (tier, matched length)

    tier 0 means no match at all. See resolve_documents for what the tiers are.
    """
    name = info.name.lower()
    # Tier 3 — the filename exactly as it is on disk.
    if re.search(_EDGE_LEFT + re.escape(name) + _EDGE_RIGHT, low):
        return 3, len(name)

    parts = [p for p in re.split(r"[\s_\-.]+", info.stem.lower()) if p]
    if not parts:
        return 0, 0

    # Tier 2 — the whole stem, however the user spaced or punctuated it:
    # "formula_sample", "formula sample" and "formula-sample" all count.
    phrase = r"[\s_\-.]+".join(re.escape(p) for p in parts)
    match = re.search(_EDGE_LEFT + phrase + _EDGE_RIGHT, low)
    if match:
        return 2, len(match.group(0))

    # Tier 1 — one distinctive word of the name. Short, numeric and generic
    # words cannot carry a match on their own.
    longest = 0
    for token in parts:
        if len(token) < 4 or token.isdigit() or token in _GENERIC_NAME_WORDS:
            continue
        if re.search(_EDGE_LEFT + re.escape(token) + _EDGE_RIGHT, low):
            longest = max(longest, len(token))
    return (1, longest) if longest else (0, 0)
