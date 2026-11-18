"""Usage: python -m eval.run_eval --chunker recursive --mode hybrid [--rerank] [--k 5]"""
import argparse
import json
import time
from pathlib import Path

from app.retrieval import search
from eval.metrics import doc_hit_at_k, hit_at_k, reciprocal_rank

ap = argparse.ArgumentParser()
ap.add_argument("--chunker", default="recursive")
ap.add_argument("--mode", default="hybrid", choices=["vector", "keyword", "hybrid"])
ap.add_argument("--rerank", action="store_true")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--name")
args = ap.parse_args()
name = args.name or f"{args.chunker}_{args.mode}{'_rerank' if args.rerank else ''}"

qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
qs = [q for q in qs if q.get("gold_doc")]  # unanswerable questions are for answer eval, not retrieval

hits = doc_hits = rr = 0.0
misses, t0 = [], time.time()
for q in qs:
    res = search(q["question"], args.chunker, args.mode, args.rerank, args.k)
    h = hit_at_k(res, q, args.k)
    hits += h
    doc_hits += doc_hit_at_k(res, q, args.k)
    rr += reciprocal_rank(res, q)
    if not h:
        misses.append({"id": q["id"], "question": q["question"],
                       "got": [(r["doc_id"], r["page"]) for r in res[:3]]})

n = len(qs)
out = {
    "name": name, "n": n, "k": args.k,
    f"hit@{args.k}": round(hits / n, 3),
    f"doc_hit@{args.k}": round(doc_hits / n, 3),
    "mrr": round(rr / n, 3),
    "avg_latency_s": round((time.time() - t0) / n, 3),
    "misses": misses,
}
Path("eval/results").mkdir(exist_ok=True)
Path(f"eval/results/{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print({k: v for k, v in out.items() if k != "misses"})
