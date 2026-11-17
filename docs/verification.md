# Verification record

Observed locally on 2026-11-17, Linux amd64, Python 3.12.14, uv 0.13.6,
PostgreSQL 17 with pgvector 0.8.1, Docker 29.9.2 and Compose v2.40.3.

| Check | Result |
| --- | --- |
| Locked dependency installation | `uv sync --frozen` passed |
| Lint and formatting | Ruff checks passed |
| `pytest -q` with disposable PostgreSQL | 11 passed; no integration cases skipped |
| Atomic indexing | Price-only updates reuse vectors, missing IDs are removed, invalid vectors roll back, readers see the previous snapshot during a blocked reindex |
| Search/API contracts | Both branches filter before candidate limits, ranks fuse without duplicate IDs, invalid requests return 422, missing/mismatched catalog returns 503 |
| Real E5/BGE inference | Pinned quantized exports run on CPU; E5 outputs normalized 384-dimensional vectors and BGE returns finite scalar logits |
| Evaluation | All 18 evaluation queries across four modes retained; hashes, source commit, model revisions and per-query results saved in `reports/evaluation-2026-11-17.json` |
| Compose smoke | Non-root application image, real models, 30 indexed products, four search modes, filters, invalid input, zero re-embeddings on an unchanged snapshot, persistence after restart |

The local container run reused the same pinned public-model cache read-only and
offline. Build downloads used the environment's trusted CA bundle through a
temporary BuildKit secret. Neither artifact is baked into the image. The CI
workflow is configured to run PostgreSQL fixture tests and a separate Compose
job that downloads and runs the real models. GitHub CI is not claimed as passed
until the repository is published and its workflow actually completes.

The deterministic test model is confined to `tests/conftest.py`; it is not an
application mode or a fallback for unavailable models. Ranking-quality numbers
come only from the real ONNX models. Tests validate transaction and retrieval
contracts, not arbitrary production workloads.

Not exercised: a large catalog/HNSW recall benchmark, simultaneous multi-host
indexers, real customer traffic, non-English queries, GPU inference, production
authentication, or recovery from host/storage failure. The small authored fixture
and its restrictive lexical baseline limit the interpretation of the quality
numbers; see [the results](results.md).
