# Evaluation protocol

`data/catalog.json` contains 30 authored synthetic English products. The 24
queries and graded relevance judgments in `data/queries.json` were committed
before model experiments: six development queries and 18 evaluation queries.
The initial fixture commit is `f7d9795`. These are demonstration judgments, not
independent human annotation or customer search logs.

The CLI compares FTS, dense cosine retrieval, RRF and BGE reranking on the same
indexed snapshot and eligible judgments. Both candidate branches apply the
filters before their limit; the fused set is bounded to 30 by default. RRF uses
`1 / (60 + rank)` with one-based ranks. BGE reranks that set, not the whole catalog.
An embedding model change requires reindexing; it cannot silently mix spaces.

- nDCG@5 uses graded gain `2^grade - 1` and logarithmic rank discount.
- MRR@5 is the mean of the reciprocal position of the first result with grade > 0.
- Recall@5 counts distinct relevant returned IDs divided by all eligible relevant
  IDs for the query. A partial ranking is not padded with invented hits.
- Queries with no judgments and queries with no results remain in the report;
  their three metrics are zero. The evaluation split deliberately has one query
  with no relevant product, so even perfect results for the other 17 have a macro
  score ceiling of 17/18. This system does not implement semantic abstention.

Models are warmed before latency timing. Downloads, ONNX session startup and
indexing are excluded; query encoding, SQL candidate retrieval, fusion and any
reranking are included. Latency percentiles use nearest rank over a single
sequential pass. The report retains every query's rankings and scores, dataset
SHA-256 hashes, immutable model revisions, CPU thread count, environment and
source revision. Errors fail the evaluation instead of dropping queries.

The quantized CPU exports are E5-small-v2 and BGE-reranker-base, not their original
full-precision execution. Small curated fixtures can expose correctness and
ranking behavior; they cannot establish production relevance, throughput,
multilingual quality or a commercial improvement. No evaluation judgments are
used to train these models or to select parameters during this implementation.

```sh
uv run hybrid-search index
uv run hybrid-search evaluate --output reports/local-evaluation.json
```

Pure metric tests use hand-calculated examples. PostgreSQL/API tests use an
explicit deterministic fixture model only for plumbing and transaction checks;
it is never selectable from the CLI or used to generate the evaluation report.
