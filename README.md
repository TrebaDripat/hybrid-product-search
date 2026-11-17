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

Implementation proceeds in small commits: fixtures and contracts; indexing and
hybrid retrieval; API and evaluation; reproducible container/CI verification.
