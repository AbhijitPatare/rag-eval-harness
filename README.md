# RAG Eval Harness

Question answering over a small corpus of UGC (University Grants Commission) regulations, built to be measured. Retrieval variants are compared on a frozen held-out set, and answer quality is graded separately from retrieval.

**Live demo:** https://rag-eval-harness-cxfjzuze2xmdqlzyg99lhd.streamlit.app/

## Pipeline

PDF text (pymupdf) -> chunking -> embeddings (`BAAI/bge-small-en-v1.5`) -> Postgres + pgvector (HNSW cosine) and Postgres full-text search -> reciprocal rank fusion -> cross-encoder rerank (`BAAI/bge-reranker-base`, top 20 candidates) -> top 8 passages to Gemini with a strict citation prompt. The model must cite passage numbers, and replies `NOT_FOUND` when the passages lack the answer.

- Corpus: 9 public UGC PDFs (regulations, rules, gazette notifications). Bilingual gazettes extract Hindi as garbled Latin text, so an ingest filter drops those chunks (366 of 985 recursive chunks dropped, 619 indexed).
- Stack: Python, FastAPI, Postgres/pgvector (Docker), sentence-transformers, Gemini API.

## Evaluation design

- **Dev set:** 37 answerable and 6 unanswerable questions. All pipeline decisions were made here.
- **Held-out test set:** 31 answerable and 8 unanswerable questions from two batches of randomly sampled chunks, written before any pipeline results were seen. Retrieval was frozen at git tag `pipeline-frozen`, and the headline config (D) was declared before the test run. The generation prompt was chosen on dev before any test answer was generated.
- **Retrieval metrics (k=5):** `hit@5` needs the right document, page and gold snippet in a retrieved chunk. `evidence@5` needs the gold snippet in any retrieved chunk. `doc_hit@5` needs only the right document. MRR uses the strict match.
- **Answer metrics:** LLM-judged correctness against a reference answer, a deterministic check that the cited passage contains the gold snippet, and refusal behaviour on unanswerable questions. Generator and judge are the same model (`gemini-3.5-flash-lite`), so treat correctness as model-graded.
- Question format: `{"id": 4, "question": "...", "gold_answer": "...", "gold_doc": "x.pdf", "gold_page": 10, "gold_snippet": "exact phrase"}`

## Results

### Retrieval, dev set (n=37 answerable)

| Run | Config | hit@5 | evidence@5 | doc_hit@5 | MRR | Latency |
|---|---|---|---|---|---|---|
| A | fixed 500/50, vector | 0.649 | 0.676 | 0.892 | 0.528 | 0.13s |
| B | recursive 800/100, vector | 0.676 | 0.703 | 0.892 | 0.541 | 0.12s |
| C | recursive 800/100, hybrid (RRF) | 0.757 | 0.784 | 0.919 | 0.619 | 0.17s |
| D | recursive 800/100, hybrid + rerank | 0.838 | 0.865 | 0.946 | 0.709 | 4.36s |
| E | recursive 400/50, hybrid | 0.703 | 0.730 | 0.919 | 0.538 | 0.16s |
| F | recursive 400/50, hybrid + rerank | 0.784 | 0.811 | 0.973 | 0.590 | 2.48s |

### Retrieval, held-out test set (pooled, n=31 answerable)

| Run | Config | hit@5 | 95% CI | MRR | Latency |
|---|---|---|---|---|---|
| B | recursive 800/100, vector | 24/31 (0.774) | 0.60-0.89 | 0.585 | 0.22s |
| C | recursive 800/100, hybrid (RRF) | 24/31 (0.774) | 0.60-0.89 | 0.595 | 0.28s |
| D | recursive 800/100, hybrid + rerank | 26/31 (0.839) | 0.67-0.93 | 0.663 | 4.82s |

### Answer quality, dev set (n=37 answerable, 6 unanswerable)

| Prompt | Correct | 95% CI | Partial | Incorrect | Refused | Unanswerable refused correctly |
|---|---|---|---|---|---|---|
| strict | 34/37 (0.919) | 0.79-0.97 | 1 | 1 | 1 | 6/6 |
| loose | 34/37 (0.919) | 0.79-0.97 | 1 | 2 | 0 | 6/6 |

### Answer quality, held-out test set (n=31 answerable, 8 unanswerable)

| Prompt | Correct | 95% CI | Partial | Incorrect | Refused | Unanswerable refused correctly |
|---|---|---|---|---|---|---|
| strict | 27/31 (0.871) | 0.71-0.95 | 0 | 0 | 4 | 8/8 |

Citation check: on the test set every correct answer cited a passage that contained the gold snippet (27/27).

## Findings

1. **Reranking is ahead in every comparison but not statistically established.** Dev hit@5 0.838 vs 0.757 for hybrid alone. On the held-out set 26/31 vs 24/31. A paired sign test on questions where configs disagreed gave p=0.62. It costs roughly 17-26x the latency on CPU (about 4.5s vs about 0.2s per query).
2. **Hybrid vs vector-only is not separable** at this sample size.
3. **800-character chunks beat 400-character chunks** in both dev pairs. The 400-character config with reranking has the best doc_hit@5 (0.973) and about half the rerank latency.
4. **The garbled-text filter made no measurable accuracy difference on dev.** It removes about a third of chunks from the index regardless.
5. **A looser "treat synonyms as equivalent" prompt did not beat the strict one** on dev (34/37 correct for both). It answered Q17 incorrectly where the strict prompt refused, so the strict prompt shipped.
6. **Retrieval misses mostly became refusals.** All four test refusals had no gold evidence in the model's context. The exception is dev Q28, where a lookalike passage produced a confident wrong answer, so refusal on missing evidence is not guaranteed.

## Failure analysis

- **Synonym gap (Q4):** the question says "inspect" and the source says "power to visit the campus and examine its operations". Neither vector nor keyword search ranked it highly, and the chunk's rank moved between 4 and a miss across rephrasings of the same question. Widening the generator's context from 5 to 8 passages fixed the answer.
- **Reranker demotions (Q20, Q28):** the reranker moved correct chunks from rank 1 and 7 down to rank 6 and 9, outside the context window, while helping elsewhere. Its net gain hides these losses.
- **Recall miss (Q17):** the question never names the deemed-university context, and the answer chunk was outside the top 20 candidates.
- **Duplicated provisions (Q15):** the same grant-withholding clause appears in two documents, which is why the evidence metric exists next to the strict one.
- **Near-duplicate documents:** the 1974 and 1975 grants-fitness rules read almost alike and account for several held-out misses (108, 110, 111). This is suspected but not isolated.
- **Pool size:** raising the candidate pool from 20 to 50 pushed the Q4 chunk out of the top 8 on all four phrasings tested, so the default stays at 20.

## Limitations

- Small evaluation sets. Confidence intervals are wide, and most differences between configs are within noise.
- One judge model, the same family as the generator.
- 9 documents only. Retrieval difficulty at larger scale is untested.
- Questions were drafted with LLM assistance from randomly sampled chunks. Gold snippets were validated against the index by script.
- Devanagari text in bilingual gazettes is not filtered and still occupies index slots.
- Latency was measured on CPU.
- The metric definition (evidence@5) was added after reading dev results (Q15). The strict metric is unchanged and reported alongside it.

## Run it

```bash
cp .env.example .env            # add your Gemini key
docker compose up -d            # Postgres + pgvector on port 5544
pip install -r requirements.txt
python -m app.db
python -m app.ingest --chunker recursive
uvicorn app.main:app --reload   # POST /ask at http://127.0.0.1:8000/docs
python -m eval.run_eval --chunker recursive --mode hybrid --rerank --name demo
```

PDFs go in `data/` (gitignored). Eval scripts are in `eval/`, and raw results are in `eval/results/`.
