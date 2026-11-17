# Hybrid Product Search

A new portfolio implementation created in November 2026: PostgreSQL full-text
search plus E5 embeddings in pgvector, reciprocal rank fusion, and BGE
cross-encoder reranking. Models run locally on CPU through quantized ONNX weights.
No paid LLM API, card data, private catalog or historical production result is used.

This repository reconstructs the engineering idea described in my CV. It is not
the lost original source or proof of historical deployment. The catalog and
relevance judgments are authored synthetic fixtures, not customer search logs or
an independently annotated benchmark.

The fixed evaluation set is committed before retrieval experiments. Development
queries and evaluation queries are separated. Measured results include
model revisions, dataset hashes, retrieval parameters and environment details.

## Run with Docker

```sh
docker compose up --build --detach --wait --wait-timeout 600
```

Then open `http://127.0.0.1:8080/docs`. The first run downloads pinned model
artifacts, indexes 30 synthetic products and warms both models. The API is bound
to loopback; PostgreSQL stays on the internal Compose network. See the
[operations guide](docs/operations.md) for shutdown, cache reuse and test commands.

## Local development

Requires Python 3.12, uv, and PostgreSQL 17 with pgvector 0.8.1. Set `DATABASE_URL`
to a disposable database, then run:

```sh
uv sync --frozen
uv run hybrid-search download-models
uv run hybrid-search index
uv run hybrid-search search 'noise cancelling headphones'
uv run hybrid-search serve
```

The first model command downloads immutable quantized E5/BGE exports from Hugging
Face; later runs reuse the local cache. Inference runs on CPU, without a GPU or
paid API. Both upstream model cards declare MIT licenses:
[E5](https://huggingface.co/intfloat/e5-small-v2) and
[BGE](https://huggingface.co/BAAI/bge-reranker-base). Export repositories and exact
revisions are in [models.json](src/hybrid_search/models.json).

## API and evaluation

The API listens on `127.0.0.1:8080` by default. Interactive documentation is at
`/docs`, the OpenAPI contract at `/openapi.json`, readiness at `/health`, and model
metadata at `/models`. Search modes are `fts`, `dense`, `hybrid` and `rerank`:

```sh
curl http://127.0.0.1:8080/search -H 'Content-Type: application/json' -d \
  '{"query":"coffee","mode":"rerank","k":5,"filters":{"category":"kitchen","max_price_cents":5000,"in_stock":true}}'
uv run hybrid-search evaluate --output reports/local-evaluation.json
```

Results include product metadata, branch ranks, lexical score, cosine similarity,
RRF score and BGE logit where applicable. Scores from different stages have
different scales; a BGE logit is not a probability. At most 100 candidates per
branch are permitted, with a bounded fused reranking set. Four concurrent search
requests are admitted; additional requests return 503 with `Retry-After`.

See [design decisions](docs/design.md) and [the evaluation protocol](docs/evaluation.md).
Indexing validates a whole catalog snapshot, replaces it atomically, removes
absent IDs and reuses embeddings when only price or availability changes. Failed
inference rolls back the snapshot; readers retain the previous committed view.

## Verified results

The [retained evaluation](docs/results.md) reports nDCG@5 of 0.9429 for dense and
hybrid retrieval on 18 authored queries. BGE did not improve this fixture
(0.9417) and added latency. Results include the weak lexical baseline and the
unanswerable query; they are not production or independently judged quality
claims. [Verification details](docs/verification.md) distinguish fixture tests
from the real-model Docker check.

## Interview walkthrough

1. Run the same query with `fts`, `dense`, `hybrid` and `rerank`; inspect ranks,
   scores and latency instead of assuming the most complex mode is best.
2. Explain why E5 query/passage prefixes and normalization matter, and why a new
   embedding signature requires reindexing.
3. Follow the SQL filters in both branches and the one-snapshot transaction.
   Discuss how HNSW's approximate candidate coverage differs at a larger scale.
4. Change only a price, reindex, and inspect the zero-embedding update count.
   Explain what concurrent readers see while a new catalog is being prepared.
5. Calculate one graded nDCG example from the retained rankings. Explain the
   zero-relevance convention, restrictive FTS baseline, and observed reranker
   regression before proposing a larger independently judged experiment.
