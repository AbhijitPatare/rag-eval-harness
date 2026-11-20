"""Copy the recursive chunks (with embeddings) from the local DB to a hosted pgvector DB."""
import os
import sys

import psycopg
from pgvector.psycopg import register_vector

SRC = os.environ.get("SOURCE_URL", "postgresql://rag:rag@localhost:5544/rag")
DST = os.environ.get("TARGET_URL")
if not DST:
    sys.exit("Set TARGET_URL first")
os.environ["DATABASE_URL"] = DST  # app.db reads this at import time
from app.db import get_conn, init_db  # noqa: E402

init_db()
with psycopg.connect(SRC) as s:
    register_vector(s)
    rows = s.execute(
        "SELECT doc_id, page, chunker, text, embedding FROM chunks WHERE chunker = 'recursive' ORDER BY id"
    ).fetchall()
print(len(rows), "chunks read from local DB")

with get_conn() as d:
    d.execute("DELETE FROM chunks WHERE chunker = 'recursive'")
    with d.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (doc_id, page, chunker, text, embedding) VALUES (%s, %s, %s, %s, %s)", rows
        )
    n = d.execute("SELECT count(*), count(DISTINCT doc_id) FROM chunks WHERE chunker = 'recursive'").fetchone()
print("remote now has", n[0], "chunks across", n[1], "docs")
