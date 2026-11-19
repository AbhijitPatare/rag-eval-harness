import json
import math

rows = [json.loads(l) for l in open("eval/results/answers_test.jsonl", encoding="utf-8") if l.strip()]
a = [r for r in rows if r["answerable"]]
u = [r for r in rows if not r["answerable"]]
k, n = sum(r["label"] == "CORRECT" for r in a), len(a)
z = 1.96
d = 1 + z * z / n
c = (k / n + z * z / (2 * n)) / d
h = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / d
print(f"correct {k}/{n} = {k / n:.3f}  95% CI {c - h:.2f}-{c + h:.2f}")
print("unanswerable correct refusals:", sum(r["label"] == "correct_refusal" for r in u), "/", len(u))
for r in a:
    if r["label"] != "CORRECT":
        print("Q", r["id"], r["label"], "| evidence in context:", r["context_has_evidence"])
print("models:", {r.get("model") for r in rows})
