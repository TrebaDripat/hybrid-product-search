import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from hybrid_search.domain import SearchRequest, parse_catalog
from hybrid_search.search import SearchService, fuse
from hybrid_search.store import CatalogNotReady

PRODUCTS = parse_catalog(json.loads((Path(__file__).parents[1] / "data/catalog.json").read_text()))


def test_rrf_uses_ranks_and_deterministic_ties():
    a, b = [dict(p.model_dump(), score=0.5) for p in PRODUCTS[:2]]
    hits = fuse([a, b], [b, a])
    assert [h.product.id for h in hits] == sorted([a["id"], b["id"]])
    assert hits[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert len(hits) == 2


@pytest.mark.integration
def test_index_reuse_price_updates_removal_and_model_mismatch(store, models):
    assert store.index(PRODUCTS, models)["embedded"] == len(PRODUCTS)
    assert store.index(PRODUCTS, models)["embedded"] == 0
    altered = [PRODUCTS[0].model_copy(update={"price_cents": 1}), *PRODUCTS[1:]]
    result = store.index(altered, models)
    assert result["embedded"] == 0 and result["updated"] == 1
    assert store.index(altered[:-1], models)["deleted"] == 1
    with pytest.raises(CatalogNotReady, match="model changed"):
        store.state("different-model")


@pytest.mark.integration
def test_filters_apply_to_both_branches_before_top_k(store, models):
    store.index(PRODUCTS, models)
    service = SearchService(store, models)
    for mode in ("fts", "dense", "hybrid", "rerank"):
        request = SearchRequest(
            query="coffee",
            mode=mode,
            k=2,
            candidate_limit=2,
            filters={"category": "kitchen", "max_price_cents": 5000, "in_stock": True},
        )
        result = service.search(request)
        assert len(result.hits) == 2
        assert len({h.product.id for h in result.hits}) == 2
        assert all(request.filters.allows(h.product) for h in result.hits)
    assert not service.search(SearchRequest(query="coffee", filters={"max_price_cents": 0})).hits


@pytest.mark.integration
def test_failed_model_output_cannot_replace_a_committed_catalog(store, models):
    store.index(PRODUCTS, models)
    previous = store.state(models.signature)["catalog_fingerprint"]
    models.passages = lambda texts: np.full((len(texts), 384), np.nan)
    changed = [PRODUCTS[0].model_copy(update={"title": "New product text"}), *PRODUCTS[1:]]
    with pytest.raises(ValueError, match="finite"):
        store.index(changed, models)
    assert store.state(models.signature)["catalog_fingerprint"] == previous


@pytest.mark.integration
def test_readers_see_previous_catalog_while_reindexing(store, models):
    store.index(PRODUCTS, models)
    original = store.state(models.signature)["catalog_fingerprint"]
    changed = [PRODUCTS[0].model_copy(update={"title": "New product text"}), *PRODUCTS[1:]]
    entered, release = Event(), Event()
    encode = models.passages

    def delayed(texts):
        entered.set()
        assert release.wait(5)
        return encode(texts)

    models.passages = delayed
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(store.index, changed, models)
        try:
            assert entered.wait(5)
            response = SearchService(store, models).search(
                SearchRequest(query="headphones", mode="fts")
            )
            assert response.catalog_fingerprint == original
        finally:
            release.set()
        assert future.result()["catalog_fingerprint"] != original
