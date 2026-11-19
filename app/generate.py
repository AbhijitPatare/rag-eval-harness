"""Answer generation: grounded prompt + citations + NOT_FOUND handling. Provider is isolated in llm()."""
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK = os.getenv("GEMINI_FALLBACK")
RETRYABLE = {429, 500, 503, 504}
_client = None

SYSTEM = """You answer questions using ONLY the numbered context passages provided.
- After each claim, cite the passage number in square brackets, e.g. [2]. Use [1][3] for several. Numbers only.
- If the passages do not contain the answer, reply with exactly: NOT_FOUND
- Never use outside knowledge. Be concise."""


def _call(model: str, prompt: str) -> str:
    resp = _client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0),
    )
    return (resp.text or "").strip()


def llm(prompt: str, attempts: int = 5) -> str:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=30000))
    for model in [m for m in (MODEL, FALLBACK) if m]:
        for i in range(attempts):
            try:
                return _call(model, prompt)
            except Exception as e:
                if isinstance(e, errors.APIError) and e.code not in RETRYABLE:
                    raise
                wait = min(2 ** i * 2, 30)
                print(f"[{model}] {type(e).__name__}, retry {i + 1}/{attempts} in {wait}s")
                time.sleep(wait)
        print(f"[{model}] still failing, moving on")
    raise RuntimeError("All Gemini models unavailable; try again later")


def answer(question: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(
        f"[{i}] {c['doc_id']}, page {c['page']}\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    text = llm(f"Context passages:\n{context}\n\nQuestion: {question}")
    if text.strip() == "NOT_FOUND":
        return {"answer": text, "found": False, "sources": []}

    cited = []
    for n in (int(x) for x in re.findall(r"\[(\d+)\]", text)):
        if 1 <= n <= len(chunks) and n not in cited:
            cited.append(n)
    picked = [chunks[n - 1] for n in cited] or chunks  # fall back to all if no citation parsed
    sources, seen = [], set()
    for c in picked:
        key = (c["doc_id"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"doc_id": c["doc_id"], "page": c["page"]})
    return {"answer": text, "found": True, "sources": sources}
