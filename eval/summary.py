import json
rows = []
for n in "ABCDEF":
    r = json.load(open(f"eval/results/{n}_n37.json", encoding="utf-8"))
    rows.append(r)
print("| Run | hit@5 | evidence@5 | doc_hit@5 | MRR | latency |\n|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['name']} | {r['hit@5']} | {r['evidence_hit@5']} | {r['doc_hit@5']} | {r['mrr']} | {r['avg_latency_s']}s |")
