import time

from fastapi import FastAPI
from pydantic import BaseModel

from app.generate import answer
from app.retrieval import search

app = FastAPI(title="RAG Eval Harness")


class Ask(BaseModel):
    question: str
    chunker: str = "recursive"
    mode: str = "hybrid"
    rerank: bool = True
    top_k: int = 8


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Ask):
    t0 = time.time()
    chunks = search(body.question, body.chunker, body.mode, body.rerank, body.top_k)
    out = answer(body.question, chunks)
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out
