import json
import sys

from app.db import get_conn


def norm(s):
    return " ".join(s.lower().split())


qs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "eval/questions.jsonl", encoding="utf-8-sig") if l.strip()]
with get_conn() as conn:
    rows = conn.execute("SELECT doc_id, page, text FROM chunks WHERE chunker='recursive'").fetchall()
by = {}
for d, p, t in rows:
    by.setdefault((d, p), []).append(norm(t))

bad = 0
for q in qs:
    if not q.get("gold_doc"):
        continue
    snip = q.get("gold_snippet")
    if not snip:
        print("NO SNIPPET", q["id"]); bad += 1; continue
    if not any(norm(snip) in t for t in by.get((q["gold_doc"], q["gold_page"]), [])):
        print("SNIPPET NOT FOUND", q["id"], q["gold_doc"], q["gold_page"], repr(snip)); bad += 1
answerable = sum(1 for q in qs if q.get("gold_doc"))
print(f"{len(qs)} questions, {answerable} answerable, {bad} problems")

