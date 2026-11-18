"""Usage: python -m eval.sample_chunks [per_doc] [outfile]  -> random clean chunks, evenly across docs."""
import sys

from app.db import get_conn

per_doc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
out = sys.argv[2] if len(sys.argv) > 2 else None
lines = []
with get_conn() as conn:
    docs = [r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM chunks WHERE chunker='recursive' ORDER BY 1")]
    for d in docs:
        rows = conn.execute(
            "SELECT page, text FROM chunks WHERE chunker='recursive' AND doc_id=%s "
            "AND length(text) > 400 ORDER BY random() LIMIT %s", (d, per_doc)
        ).fetchall()
        for page, text in rows:
            lines.append(f"\n=== {d} | page {page} ===\n{text[:600]}")

text = "\n".join(lines)
if out:
    open(out, "w", encoding="utf-8").write(text)
    print(f"Wrote {len(lines)} chunks to {out}")
else:
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
