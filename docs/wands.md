# External relevance judgments

The external check uses product queries and human relevance judgments from Wayfair WANDS. The sampling rule is fixed before changing the retrieval implementation. Both versions search the same pool for every query.

* Rank query IDs by SHA-256 of a fixed namespace and the ID, then take the first 24. For each query, sample up to 32 judged products by hashing their IDs independently of the grade. Add 128 products sampled from the full catalog by ID. Take the union and retain every available upstream judgment for each query within that union.
* Grade Exact as 2, Partial as 1 and Irrelevant as 0. Unjudged products receive zero gain. Recall uses the judged relevant products within the shared pool as its denominator. The pool is smaller and more densely judged than the full collection of 42,994 products.
* Import titles, descriptions and normalized product classes. Prices and availability are absent from WANDS. The adapter supplies constant placeholder values to the storage contract and runs all queries without filters.

```sh
python3 scripts/prepare_wands.py
# Use a separate disposable DATABASE_URL. Indexing replaces its catalog.
uv run hybrid-search index --catalog .cache/wands/fixed-v1/catalog.json
uv run hybrid-search evaluate --catalog .cache/wands/fixed-v1/catalog.json \
  --queries .cache/wands/fixed-v1/queries.json --manifest .cache/wands/fixed-v1/manifest.json \
  --output reports/local-wands.json
```

The script verifies all three source files against pinned SHA-256 digests. The generated catalog and queries stay under `.cache/`. The comparison records their hashes, the upstream revision and product IDs so others can reproduce the run.

Yan Chen, Shujian Liu, Zheng Liu, Weiyi Sun, Linas Baltrunas and Benjamin Schroeder, *WANDS: Dataset for Product Search Relevance Assessment*, ECIR 2022. [Upstream dataset](https://github.com/wayfair/WANDS/tree/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5) · [MIT license, Copyright (c) 2021 ecir2022](https://github.com/wayfair/WANDS/blob/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/LICENSE)
