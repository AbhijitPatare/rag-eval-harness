import sys

from app.db import get_conn

sys.stdout.reconfigure(encoding="utf-8")
doc, page, snip = sys.argv[1], int(sys.argv[2]), sys.argv[3].lower()
with get_conn() as conn:
    rows = conn.execute("SELECT text FROM chunks WHERE chunker='recursive' AND doc_id=%s AND page=%s", (doc, page)).fetchall()
for (t,) in rows:
    if snip in " ".join(t.lower().split()):
        print(t[:900]); print("-----")
