"""Retrieval metrics. A result matches if doc_id/page match and, when gold_snippet is set, the snippet is in the chunk text."""


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def is_match(r: dict, q: dict) -> bool:
    if r["doc_id"] != q["gold_doc"]:
        return False
    if q.get("gold_page") is not None and r["page"] != q["gold_page"]:
        return False
    if q.get("gold_snippet") and _norm(q["gold_snippet"]) not in _norm(r["text"]):
        return False
    return True


def hit_at_k(results, q, k):
    return any(is_match(r, q) for r in results[:k])


def doc_hit_at_k(results, q, k):
    return any(r["doc_id"] == q["gold_doc"] for r in results[:k])


def reciprocal_rank(results, q):
    for i, r in enumerate(results, start=1):
        if is_match(r, q):
            return 1.0 / i
    return 0.0
