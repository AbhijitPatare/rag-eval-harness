"""Load PDFs -> chunk per page -> drop garbled chunks -> embed -> store in pgvector.
Usage: python -m app.ingest --chunker fixed|recursive [--keep-garbled]
"""
import argparse
import re
import sys
from pathlib import Path

import pymupdf
from sentence_transformers import SentenceTransformer

from app.chunking import chunk
from app.config import DATA_DIR, EMBED_MODEL
from app.db import get_conn, init_db

STOP = {"the", "of", "and", "to", "in", "a", "is", "for", "shall", "be", "by", "or", "that", "with", "as", "on", "are", "any"}


def is_garbled(text: str, min_ratio: float = 0.10) -> bool:
    """Legacy-font Hindi extracts as Latin gibberish: very few real English stopwords."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(words) < 20:
        return False
    return sum(w in STOP for w in words) / len(words) < min_ratio


def load_pages(path: Path):
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().replace("\x00", "").strip()
            if text:
                yield i, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", choices=["fixed", "recursive", "recursive_small"], required=True)
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--keep-garbled", action="store_true")
    args = ap.parse_args()

    pdfs = sorted(Path(args.data).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.data}/")

    init_db()
    rows = []
    for pdf in pdfs:
        for page_no, text in load_pages(pdf):
            for c in chunk(text, args.chunker):
                rows.append((pdf.name, page_no, args.chunker, c))

    before = len(rows)
    if not args.keep_garbled:
        rows = [r for r in rows if not is_garbled(r[3])]
    print(f"{len(pdfs)} PDFs -> {before} chunks, dropped {before - len(rows)} garbled, kept {len(rows)} ({args.chunker})")

    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(
        [r[3] for r in rows], batch_size=64,
        normalize_embeddings=True, show_progress_bar=True,
    )

    with get_conn() as conn:
        conn.execute("DELETE FROM chunks WHERE chunker = %s", (args.chunker,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (doc_id, page, chunker, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                [(*r, e) for r, e in zip(rows, embs)],
            )
    print("Done.")


if __name__ == "__main__":
    main()
