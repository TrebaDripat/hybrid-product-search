# Retrieval contract

1. Validate the complete synthetic catalog before writing anything.
2. Index document text with E5's `passage: ` prefix and normalized mean pooling.
   Query vectors use `query: `. PostgreSQL holds both the metadata and vectors.
3. Apply category, price and availability filters to both candidate branches.
   Retrieve lexical candidates with PostgreSQL FTS and dense candidates with
   cosine distance. Fuse ranks with RRF (`k=60`) and deterministic ID tie-breaking.
4. Rerank a bounded fused candidate set with BGE; return scores with their actual
   meaning and provenance. Raw cross-encoder logits are not probabilities.
5. Compare lexical, dense, hybrid and reranked results on the same fixed eligible
   judgments. Report graded nDCG@k, reciprocal rank@k and binary Recall@k. Include
   failed and zero-result queries; do not select only favorable cases.

The demonstration uses English text, 384-dimensional E5 embeddings, one catalog
and one currency (USD). Prices use integer cents. A model-signature mismatch must
fail explicitly rather than query incompatible vectors. Ingestion publishes a
whole validated catalog snapshot atomically and reuses vectors for unchanged
embedding text. Search uses one database snapshot for both retrieval branches.

The small fixture may use PostgreSQL's exact sequential scan even though a cosine
HNSW index is present. Approximate-index scale, multilingual relevance, production
latency and business improvements require separate representative evaluation.

Model exports are pinned to immutable revisions, with the original model family
identified in the code. Downloaded files are model weights/configuration only;
remote Python code is never trusted or executed.
