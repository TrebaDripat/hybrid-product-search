# Observed evaluation: 2026-11-17

This record preserves the initial search implementation. A later experiment evaluated partial lexical matches and two BGE budgets on both this catalog and external Wayfair judgments. [Paired measurements](quality.md)

[Raw per-query results](../reports/evaluation-2026-11-17.json) were generated from
clean source commit `41c0e30`, with the previously frozen catalog and judgments.
All 18 evaluation queries are included. Metrics are macro averages at k=5; one
query has no relevant product and contributes zero, so the maximum macro MRR and
Recall here is 17/18, approximately 0.9444.

| Mode | nDCG@5 | MRR@5 | Recall@5 | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| PostgreSQL FTS | 0.1111 | 0.1111 | 0.1111 | 3.920 ms | 7.033 ms |
| E5 cosine | 0.9429 | 0.9444 | 0.9444 | 9.807 ms | 17.094 ms |
| RRF hybrid | 0.9429 | 0.9444 | 0.9444 | 7.602 ms | 12.772 ms |
| BGE reranked | 0.9417 | 0.9444 | 0.9444 | 179.094 ms | 1001.000 ms |

These are small synthetic measurements, not an independent IR benchmark. The
lexical baseline uses `websearch_to_tsquery`, whose implicit AND semantics are
restrictive for long natural-language questions. It is not a tuned BM25 or
query-expansion baseline. The gap cannot be attributed solely to a general
superiority of semantic retrieval. OR-based candidate expansion and a larger,
independently judged catalog would be appropriate follow-up experiments.

BGE did **not** improve this fixture. On `eval-13` (large desktop backup drive),
it moved a USB flash drive above a more relevant portable SSD, reducing that
query's nDCG from 0.9721 to 0.9508. The top relevant result remained correct, so
MRR and Recall did not change. Reranking is exposed as an explicit mode, not
presented as a guaranteed quality improvement.

The modes ran sequentially in the same order with warmed models, two ONNX CPU
threads, four CPU equivalents of cgroup quota and a 16 GiB memory limit. Small
sample size, differing filter selectivity and cache effects make the latency
numbers illustrative; the lower observed hybrid p50 does not establish that it
is faster than dense retrieval. Download/startup time is excluded.

No parameters or relevance labels were adjusted after observing these results.
Future experiments should preserve this report and use a new, clearly versioned
evaluation set. [Metric definitions and limitations](evaluation.md).
