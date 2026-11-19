"""Answer-quality eval. Usage:
python -m eval.answer_eval --questions eval/questions.jsonl --prompts strict loose --name dev
python -m eval.answer_eval --questions eval/test_questions.jsonl eval/test_questions_b3.jsonl --prompts loose --name test
Resumable: rerun the same command after an interruption.
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from app import generate as g
from app.retrieval import search

sys.stdout.reconfigure(encoding="utf-8")

LOOSE = """You answer questions using ONLY the numbered context passages provided.
- After each claim, cite the passage number in square brackets, e.g. [2]. Use [1][3] for several. Numbers only.
- Treat synonyms and paraphrases as equivalent (e.g. 'inspect' and 'visit and examine'). If the passages support an answer even though the wording differs, answer it.
- Reply with exactly NOT_FOUND only if no passage addresses the question.
- Never use outside knowledge. Be concise."""
STRICT = """You answer questions using ONLY the numbered context passages provided.
- After each claim, cite the passage number in square brackets, e.g. [2]. Use [1][3] for several. Numbers only.
- If the passages do not contain the answer, reply with exactly: NOT_FOUND
- Never use outside knowledge. Be concise."""
JUDGE = """You grade answers to questions about regulations. You get a question, a reference answer and a candidate answer. Reply with exactly one word:
CORRECT - the candidate conveys the reference answer's key facts and does not contradict it (extra detail is fine).
PARTIAL - some key facts are right but an important part is missing or wrong.
INCORRECT - wrong, contradicts the reference, or says nothing relevant.
Ignore citation brackets like [2]. Judge only against the reference answer."""
PROMPTS = {"strict": STRICT, "loose": LOOSE}


def norm(s):
    return " ".join(s.lower().split())


ap = argparse.ArgumentParser()
ap.add_argument("--questions", nargs="+", required=True)
ap.add_argument("--prompts", nargs="+", default=["loose"], choices=["strict", "loose"])
ap.add_argument("--name", required=True)
ap.add_argument("--top_k", type=int, default=8)
ap.add_argument("--sleep", type=float, default=1.5)
args = ap.parse_args()

qs = []
for f in args.questions:
    qs += [json.loads(l) for l in Path(f).read_text(encoding="utf-8-sig").splitlines() if l.strip()]

out_path = Path(f"eval/results/answers_{args.name}.jsonl")
rows, done = [], set()
if out_path.exists():
    for l in out_path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            rows.append(r)
            done.add((r["id"], r["prompt"]))


def judge(q, ans):
    prev = g.SYSTEM
    g.SYSTEM = JUDGE
    try:
        t = g.llm(f"Question: {q['question']}\nReference answer: {q['gold_answer']}\nCandidate answer: {ans}")
    finally:
        g.SYSTEM = prev
    w = t.strip().split()[0].upper().strip(".,:*") if t.strip() else ""
    return w if w in ("CORRECT", "PARTIAL", "INCORRECT") else "UNPARSED"


with out_path.open("a", encoding="utf-8") as fh:
    for q in qs:
        todo = [p for p in args.prompts if (q["id"], p) not in done]
        if not todo:
            continue
        chunks = search(q["question"], "recursive", "hybrid", True, args.top_k)
        snip = norm(q["gold_snippet"]) if q.get("gold_snippet") else None
        ctx_ev = any(snip in norm(c["text"]) for c in chunks) if snip else None
        for p in todo:
            g.SYSTEM = PROMPTS[p]
            res = g.answer(q["question"], chunks)
            time.sleep(args.sleep)
            row = {"id": q["id"], "prompt": p, "answerable": bool(q.get("gold_doc")),
                   "question": q["question"], "gold": q["gold_answer"], "answer": res["answer"], "model": g.MODEL,
                   "found": res["found"], "context_has_evidence": ctx_ev}
            if not q.get("gold_doc"):
                row["label"] = "hallucinated" if res["found"] else "correct_refusal"
            elif not res["found"]:
                row["label"] = "refused"
            else:
                cited = [int(x) for x in re.findall(r"\[(\d+)\]", res["answer"])]
                row["cited_has_evidence"] = (
                    any(1 <= n <= len(chunks) and snip in norm(chunks[n - 1]["text"]) for n in cited) if snip else None
                )
                row["label"] = judge(q, res["answer"])
                time.sleep(args.sleep)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rows.append(row)
            print(q["id"], p, row["label"])

ids = {q["id"] for q in qs}
print("\n=== SUMMARY ===")
for p in args.prompts:
    rs = [r for r in rows if r["prompt"] == p and r["id"] in ids]
    a = [r for r in rs if r["answerable"]]
    u = [r for r in rs if not r["answerable"]]
    c = Counter(r["label"] for r in a)
    answered = [r for r in a if r["label"] not in ("refused",)]
    cited_ok = [r for r in answered if r.get("cited_has_evidence")]
    print(f"[{p}] answerable n={len(a)}: " + ", ".join(f"{k}={c[k]}" for k in ("CORRECT", "PARTIAL", "INCORRECT", "refused", "UNPARSED")))
    if a:
        print(f"      correct rate {c['CORRECT'] / len(a):.3f} | answered-but-no-evidence-in-citation {len(answered) - len(cited_ok)}/{len(answered)}")
    cu = Counter(r["label"] for r in u)
    print(f"      unanswerable n={len(u)}: correct_refusal={cu['correct_refusal']}, hallucinated={cu['hallucinated']}")

judged = [r for r in rows if r["id"] in ids and r["label"] in ("CORRECT", "PARTIAL", "INCORRECT")]
print("\n=== SPOT-CHECK THESE (verify the judge by hand) ===")
for r in random.sample(judged, min(8, len(judged))):
    print(f"\nQ{r['id']} [{r['prompt']}] judge={r['label']}\n  gold: {r['gold'][:150]}\n  ans : {r['answer'][:200]}")

