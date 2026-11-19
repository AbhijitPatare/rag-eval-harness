"""Retrieval metrics.
strict:   doc_id + page (+ snippet if set) must match the gold.
evidence: the gold snippet appears in a retrieved chunk, from any document or page.
"""


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


def has_evidence(r: dict, q: dict) -> bool:
    if q.get("gold_snippet"):
        return _norm(q["gold_snippet"]) in _norm(r["text"])
    return is_match(r, q)


def hit_at_k(results, q, k):
    return any(is_match(r, q) for r in results[:k])


def doc_hit_at_k(results, q, k):
    return any(r["doc_id"] == q["gold_doc"] for r in results[:k])


def evidence_hit_at_k(results, q, k):
    return any(has_evidence(r, q) for r in results[:k])


def reciprocal_rank(results, q):
    for i, r in enumerate(results, start=1):
        if is_match(r, q):
            return 1.0 / i
    return 0.0


def evidence_rr(results, q):
    for i, r in enumerate(results, start=1):
        if has_evidence(r, q):
            return 1.0 / i
    return 0.0
