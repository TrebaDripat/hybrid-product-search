# Retrieval and indexing

## How a query reaches the top five

1. The API validates the query, filters, requested result count and inference budget. Each search reads one repeatable-read PostgreSQL snapshot. Category, availability and price constraints apply in both candidate branches before either limit.
2. The lexical branch prioritizes an eligible exact catalog ID, then matches containing all normalized terms. For an ordinary English phrase, PostgreSQL also retrieves partial matches and orders them by term coverage and text rank. Quoted phrases, `OR` and exclusions keep PostgreSQL web-search semantics within the lexical branch. The dense branch interprets the query as text.
3. E5 encodes queries with the `query: ` prefix and products with `passage: `. Attention-mask mean pooling and L2 normalization produce 384-dimensional vectors. PostgreSQL retrieves the nearest products by cosine distance using pgvector.
4. Fusion adds `weight / (60 + rank)` for each branch where a product appears. Dense candidates and strict lexical matches have weight 1. Partial lexical matches use the fraction of distinct normalized query terms present in the product. Ties break on product ID. An exact eligible ID stays first in FTS, hybrid and reranked results.
5. BGE scores the leading fused candidates when `mode=rerank`. Its default depth is `min(candidate_limit, max(8, k))`, enough for one eight-pair batch at the default `k=5`. Callers can request any depth from `k` through `candidate_limit`. A BGE score is a raw logit. Its magnitude is meaningful within the model’s ordering of a given query.

Responses include the catalog fingerprint, candidates before the BGE cut, actual pairs scored, and wall time in query encoding, retrieval, fusion and reranking. Branch ranks and raw scores let the caller trace a result back to its candidate sources. The explorer compares two sequential requests and checks that their catalog fingerprints match.

## Publishing a catalog

The indexer validates the entire snapshot, locks concurrent writers with a transaction-level advisory lock and compares the catalog fingerprint and embedding signature. Replaying an identical catalog performs zero product writes and preserves its publication time. A metadata-only change updates the row in PostgreSQL and keeps its existing vector there.

New or changed embedding text is encoded in batches of 32 inside one transaction. Absent IDs are deleted and the catalog state is published in that transaction. A failure in a later batch rolls back earlier writes. Readers continue to see the previous committed snapshot. A model-signature change requires a rebuild before query vectors can be used in the new embedding space.

## Operating scope

This demonstration uses English catalogs and USD prices in integer cents. It ships with a cosine HNSW index. PostgreSQL may choose an exact sequential scan for small catalogs. The retained experiments cover pools of 30 and 875 products on one CPU host. HNSW recall at larger scale and latency under concurrent production traffic need different measurements. [Model revisions](../src/hybrid_search/models.json) · [Evaluation scope](evaluation.md)
