"""
Phase 1 RAG demo: ingest one PDF, then ask questions and inspect retrieved chunks.

No LLM answer generation — only retrieval for learning and manual inspection.
"""

from pathlib import Path

from chunker import chunk_pages
from document_loader import load_pdf
from embeddings import EmbeddingModel
from retriever import Retriever
from vector_store import VectorStore

# --- Configurable settings ---
PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


def print_results(question: str, results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"QUESTION:\n{question}\n")

    if not results:
        print("No results returned.")
        return

    for result in results:
        print(f"RETRIEVED CHUNK {result['rank']}")
        print(f"Page: {result['page_number']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Text: {result['text']}\n")


def build_retriever() -> Retriever:
    print(f"Loading PDF: {PDF_PATH}")
    pages = load_pdf(PDF_PATH)
    print(f"  Pages loaded: {len(pages)}")

    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  Chunks created: {len(chunks)}")

    print("Loading embedding model (first run downloads weights)...")
    embedding_model = EmbeddingModel()
    print(f"  Model dimension: {embedding_model.dimension}")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.embed_texts(texts)

    store = VectorStore(dimension=embedding_model.dimension)
    store.add(embeddings, chunks)
    print(f"  Vectors in FAISS index: {store.index.ntotal}")

    return Retriever(embedding_model, store)


def print_suggested_tests() -> None:
    print("\nSuggested manual test questions:")
    print("  1. What was the company's revenue in 2024?")
    print("  2. How much money did the firm make in twenty twenty-four?")
    print("  3. Who is the CEO?  (answer on a different page)")
    print("  4. What is the stock price on Mars?  (not in document)")
    print("  5. Tell me about the company.  (vague)")
    print("  6. What are the main products and services?  (multiple chunks)")


def main() -> None:
    retriever = build_retriever()
    print_suggested_tests()
    print("\nType a question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        results = retriever.retrieve(question, top_k=TOP_K)
        print_results(question, results)


if __name__ == "__main__":
    main()
