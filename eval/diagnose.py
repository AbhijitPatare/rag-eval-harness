import json
import sys

from app.db import get_conn
from app.retrieval import hybrid_search, rerank


def norm(s):
    return " ".join(s.lower().split())


name = sys.argv[1] if len(sys.argv) > 1 else "D_n36"
res = json.load(open(f"eval/results/{name}.json", encoding="utf-8"))
qs = {q["id"]: q for q in (json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8-sig") if l.strip())}
with get_conn() as conn:
    rows = conn.execute("SELECT doc_id, page, text FROM chunks WHERE chunker='recursive'").fetchall()


def first(lst, pred):
    return next((i for i, r in enumerate(lst, 1) if pred(r)), None)


for m in res["misses"]:
    q = qs[m["id"]]
    snip = norm(q["gold_snippet"])
    holders = sorted({(d, p) for d, p, t in rows if snip in norm(t)})
    pool = hybrid_search(q["question"], "recursive", 20)
    rr = rerank(q["question"], pool, 20)
    gold = lambda r: snip in norm(r["text"]) and r["doc_id"] == q["gold_doc"] and r["page"] == q["gold_page"]
    anyp = lambda r: snip in norm(r["text"])
    print(f"Q{q['id']}: snippet text appears on {holders}")
    print(f"     gold chunk: pool rank {first(pool, gold)} -> reranked {first(rr, gold)} | "
          f"any page with snippet: pool {first(pool, anyp)} -> reranked {first(rr, anyp)}\n")
