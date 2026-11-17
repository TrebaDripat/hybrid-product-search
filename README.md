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
queries and evaluation queries are separated. Measured results will include
model revisions, dataset hashes, retrieval parameters and environment details.

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
