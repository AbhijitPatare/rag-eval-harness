import json
from math import comb


def load(n):
    return json.load(open(f"eval/results/{n}.json", encoding="utf-8"))


for other, label in (("C", "hybrid"), ("B", "vector")):
    d_only = o_only = 0
    for a, b in (("TEST_D", f"TEST_{other}"), ("TEST3_D", f"TEST3_{other}")):
        am, bm = set(load(a)["evidence_misses"]), set(load(b)["evidence_misses"])
        d_only += len(bm - am)  # D hit, other missed
        o_only += len(am - bm)  # other hit, D missed
    n, k = d_only + o_only, min(d_only, o_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    print(f"D vs {label}: D-only hits {d_only}, {label}-only hits {o_only}, exact sign-test p={p:.2f}")
