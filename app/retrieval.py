"""Retrieval: vector, keyword, hybrid (RRF), optional cross-encoder rerank."""
import os
import re
from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import EMBED_MODEL
from app.db import get_conn

RERANK_MODEL = "BAAI/bge-reranker-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CANDIDATES = int(os.getenv("CANDIDATES", "20"))  # fetched per method before fusion / rerank


@lru_cache(maxsize=1)
def _embedder():
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    return CrossEncoder(RERANK_MODEL)


def _rows(sql, params):
    with get_conn() as conn:
        cur = conn.execute(sql, params)
        return [
            {"id": r[0], "doc_id": r[1], "page": r[2], "text": r[3], "score": float(r[4])}
            for r in cur.fetchall()
        ]


def vector_search(query: str, chunker: str, n: int = CANDIDATES):
    emb = _embedder().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    return _rows(
        "SELECT id, doc_id, page, text, 1 - (embedding <=> %s) AS score "
        "FROM chunks WHERE chunker = %s ORDER BY embedding <=> %s LIMIT %s",
        (emb, chunker, emb, n),
    )


def keyword_search(query: str, chunker: str, n: int = CANDIDATES):
    words = re.findall(r"\w+", query.lower())
    if not words:
        return []
    tsq = " | ".join(words)  # OR semantics: natural-language questions rarely match every word
    return _rows(
        "SELECT id, doc_id, page, text, ts_rank(tsv, q) AS score "
        "FROM chunks, to_tsquery('english', %s) q "
        "WHERE chunker = %s AND tsv @@ q ORDER BY score DESC LIMIT %s",
        (tsq, chunker, n),
    )


def hybrid_search(query: str, chunker: str, n: int = CANDIDATES, k: int = 60):
    """Reciprocal rank fusion of vector + keyword results."""
    fused, by_id = {}, {}
    for results in (vector_search(query, chunker, n), keyword_search(query, chunker, n)):
        for rank, r in enumerate(results, start=1):
            fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (k + rank)
            by_id[r["id"]] = r
    ranked = sorted(fused, key=fused.get, reverse=True)[:n]
    return [{**by_id[i], "score": fused[i]} for i in ranked]


def rerank(query: str, results: list[dict], top_k: int):
    if not results:
        return results
    scores = _reranker().predict([(query, r["text"]) for r in results])
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [{**r, "score": float(s)} for r, s in ranked[:top_k]]


def search(query: str, chunker: str = "recursive", mode: str = "hybrid",
           use_rerank: bool = False, top_k: int = 5):
    fetch = {"vector": vector_search, "keyword": keyword_search, "hybrid": hybrid_search}[mode]
    results = fetch(query, chunker, CANDIDATES)
    if use_rerank:
        return rerank(query, results, top_k)
    return results[:top_k]
