"""
Fetch the real-document validation corpus into data/real/ (gitignored).

We validate ingestion against documents we did not write, with the layout
features our synthetic test PDF lacks: two columns, running headers and
footers, page numbers, footnotes, tables, hyphenated line breaks.

The files are publicly available but are not ours to redistribute, so they
are downloaded on demand rather than committed. Tests that need them skip
cleanly when they are absent.

Run:  ./.venv/bin/python scripts/fetch_real_docs.py
"""

import sys
import urllib.request
from pathlib import Path

REAL_DIR = Path(__file__).resolve().parent.parent / "data" / "real"

DOCUMENTS = {
    # Two-column IEEE-style survey with tables, footnotes, a drop cap.
    "rag_survey.pdf": "https://arxiv.org/pdf/2312.10997",
    # Two-column ACL-style paper with figures and result tables.
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",
    # Single-column US government report (public domain): running header and
    # rotated footer on every page, roman-numeral front matter, numbered
    # headings and lists.
    "nist_sp800-207.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-207.pdf",
}


def main() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, url in DOCUMENTS.items():
        target = REAL_DIR / name
        if target.exists() and target.stat().st_size > 10_000:
            print(f"  present  {name}")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 10_000 or not data.startswith(b"%PDF"):
                raise ValueError(f"response does not look like a PDF ({len(data)} bytes)")
            target.write_bytes(data)
            print(f"  fetched  {name}  ({len(data) // 1024} KB)")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAILED   {name}: {exc}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
