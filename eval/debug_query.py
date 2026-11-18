import sys

from app.retrieval import hybrid_search, keyword_search, vector_search

q, snip = sys.argv[1], sys.argv[2].lower()
for name, fn in (("vector", vector_search), ("keyword", keyword_search), ("hybrid", hybrid_search)):
    res = fn(q, "recursive", 100)
    ranks = [i for i, r in enumerate(res, 1) if snip in " ".join(r["text"].lower().split())]
    print(name, "rank of snippet:", ranks[:3] or "not in top 100")
