import os
import psycopg
from pgvector.psycopg import register_vector

DSN = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5544/rag")
EMBED_DIM = 384  # BAAI/bge-small-en-v1.5


def get_conn():
    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_db(reset: bool = False):
    with get_conn() as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS chunks (
                id        BIGSERIAL PRIMARY KEY,
                doc_id    TEXT NOT NULL,
                page      INT,
                chunker   TEXT NOT NULL,
                text      TEXT NOT NULL,
                embedding vector({EMBED_DIM}),
                tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            )
        ''')
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_chunker_idx ON chunks (chunker)")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )


if __name__ == "__main__":
    import sys
    init_db(reset="--reset" in sys.argv)
    print("DB ready.")
