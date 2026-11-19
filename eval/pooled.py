import json
import math


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


names = {"B": "recursive, vector", "C": "recursive, hybrid", "D": "recursive, hybrid + rerank"}
print("| Config | n | hit@5 | 95% CI | evidence@5 | MRR | latency |\n|---|---|---|---|---|---|---|")
for cfg in "BCD":
    a = json.load(open(f"eval/results/TEST_{cfg}.json", encoding="utf-8"))
    b = json.load(open(f"eval/results/TEST3_{cfg}.json", encoding="utf-8"))
    n = a["n"] + b["n"]
    w = lambda key: (a[key] * a["n"] + b[key] * b["n"]) / n
    hit, ev = round(w("hit@5") * n), round(w("evidence_hit@5") * n)
    lo, hi = wilson(hit, n)
    print(f"| {names[cfg]} | {n} | {hit}/{n} ({hit / n:.3f}) | {lo:.2f}-{hi:.2f} | {ev}/{n} ({ev / n:.3f}) | {w('mrr'):.3f} | {w('avg_latency_s'):.2f}s |")
