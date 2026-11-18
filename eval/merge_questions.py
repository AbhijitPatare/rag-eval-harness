import json

S = {1: "funds shall not be diverted", 2: "audited annually", 3: "may appeal to the commission for redressal",
     5: "not exceeding 360 days", 6: "enter into any franchise agreement", 7: "transparency, fairness, honesty",
     8: "notarized affidavit", 9: "clauses 5.1 to 5.4", 10: "experience of at least ten years in physical education and sports"}
p = "eval/questions.jsonl"
qs = [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
for q in qs:
    if q["id"] in S and q.get("gold_doc"):
        q["gold_snippet"] = S[q["id"]]
new = [json.loads(l) for l in open("eval/new_qs.jsonl", encoding="utf-8-sig") if l.strip()]
have = {q["id"] for q in qs}
qs += [q for q in new if q["id"] not in have]
open(p, "w", encoding="utf-8").write("\n".join(json.dumps(q, ensure_ascii=False) for q in qs) + "\n")
print(len(qs), "questions")
