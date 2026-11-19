import json

from app.db import get_conn


def norm(s):
    return " ".join(s.lower().split())


qs = [json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8-sig") if l.strip()]
with get_conn() as conn:
    rows = conn.execute("SELECT doc_id, page, text FROM chunks WHERE chunker='recursive'").fetchall()
for q in qs:
    if not q.get("gold_snippet"):
        continue
    s = norm(q["gold_snippet"])
    where = sorted({(d, p) for d, p, t in rows if s in norm(t)})
    if len(where) > 1:
        print(f"Q{q['id']}: snippet in {len(where)} places: {where}")
