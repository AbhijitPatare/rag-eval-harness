import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
name = sys.argv[1] if len(sys.argv) > 1 else "dev"
rows = [json.loads(l) for l in open(f"eval/results/answers_{name}.jsonl", encoding="utf-8") if l.strip()]
for r in rows:
    if r["label"] not in ("CORRECT", "correct_refusal") or r.get("cited_has_evidence") is False:
        print(f"Q{r['id']} [{r['prompt']}] {r['label']} | context_has_evidence={r['context_has_evidence']} cited_has_evidence={r.get('cited_has_evidence')}")
        print(f"  Q   : {r['question']}")
        print(f"  gold: {r['gold'][:160]}")
        print(f"  ans : {r['answer'][:300]}\n")
