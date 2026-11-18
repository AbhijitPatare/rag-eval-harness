import sys

from app.retrieval import search

ch = sys.argv[1] if len(sys.argv) > 1 else "recursive"
qs = [
    "Is the UGC allowed to inspect a foreign university campus in India?",
    "Is the UGC allowed to inspect a foreign university's campus in India?",
    "Can the UGC inspect a foreign university campus in India?",
    "Does the UGC have the power to inspect foreign university campuses?",
]
for q in qs:
    res = search(q, ch, "hybrid", True, 8)
    ranks = [i for i, r in enumerate(res, 1) if "power to visit" in " ".join(r["text"].lower().split())]
    print(ranks or "not in top 8", "|", q)
