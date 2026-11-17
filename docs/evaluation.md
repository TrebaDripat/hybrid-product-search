# Evaluation protocol

The demo dataset contains 30 authored synthetic English products, six development queries and 18 evaluation queries. Products, queries and grade assignments were committed in `f7d9795` before the first model experiment. A second evaluation uses external Wayfair WANDS judgments on a fixed, reproducible 875-product pool. The pool selection was frozen in `86585e3` before the retrieval changes. [WANDS sampling](wands.md)

## Metrics

- **nDCG@k** uses gain `2^grade - 1` and a logarithmic rank discount. The ideal ordering uses the eligible judgments for the query.
- **MRR@k** averages the reciprocal position of the first result with a positive grade.
- **Recall@k** divides the number of distinct relevant hits by the number of eligible products with a positive judgment for the query.
- Empty rankings and queries with zero eligible relevant products score zero. One of the 18 synthetic evaluation queries has no relevant product, giving a ceiling of 17/18 on all three macro metrics. The semantic modes can still return candidates for it.

## Timing and provenance

The evaluator warms both ONNX sessions before timing and rotates mode order by query and pass. Elapsed time covers query encoding, SQL retrieval, fusion and requested reranking. Downloads, indexing, model warmup and HTTP transit are outside the timed interval. Each percentile is the nearest rank across all query measurements for a mode. For repeated passes every query has the same number of measurements.

The report retains per-query rankings, query IDs, relevant counts, scores and elapsed times. New runs also record phase timings and how many pairs BGE scored. Dataset hashes, model revisions, thread count, cgroup limits, catalog fingerprint, candidate limit and the running checkout revision accompany the results. Evaluation stops if a manifest hash or the indexed catalog fingerprint differs. Custom inputs supplied without a manifest are labeled as a custom catalog.

```sh
uv run hybrid-search index
uv run hybrid-search evaluate --passes 3 --output reports/local-evaluation.json
python3 scripts/compare_runs.py reports/quality-v2/before-demo.json reports/quality-v2/after-demo.json
```

The comparison script works with the Python standard library. It checks matching inputs, model settings, environment, query IDs and passes, then recalculates the table from the individual samples. On the WANDS pool, unjudged products receive zero gain. Numbers on either catalog describe these queries on the recorded host. [Paired results](quality.md) · [Original evaluation](results.md)
