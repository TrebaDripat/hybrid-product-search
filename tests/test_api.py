import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hybrid_search.api import create_app
from hybrid_search.domain import parse_catalog
from hybrid_search.search import SearchService


@pytest.mark.integration
def test_api_readiness_filters_validation_and_openapi(store, models):
    with TestClient(create_app(SearchService(store, models))) as client:
        assert client.get("/health").status_code == 503
        products = parse_catalog(
            json.loads((Path(__file__).parents[1] / "data/catalog.json").read_text())
        )
        store.index(products, models)
        assert client.get("/health").json()["product_count"] == 30
        response = client.post(
            "/search", json={"query": "coffee", "filters": {"category": "kitchen"}}
        )
        assert response.status_code == 200
        assert all(hit["product"]["category"] == "kitchen" for hit in response.json()["hits"])
        assert client.post("/search", json={"query": " "}).status_code == 422
        for fields in (
            {"k": 5, "rerank_limit": 4},
            {"candidate_limit": 6, "rerank_limit": 8},
            {"mode": "dense", "rerank_limit": 8},
        ):
            assert client.post("/search", json={"query": "coffee", **fields}).status_code == 422
        assert (
            client.post("/search", json={"query": "x", "candidate_limit": 1, "k": 2}).status_code
            == 422
        )
        spec = client.get("/openapi.json").json()
        assert spec["paths"]["/search"]["post"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"].endswith("SearchResponse")
        models.signature = "different-model"
        assert client.get("/health").status_code == 503
        assert client.post("/search", json={"query": "coffee"}).status_code == 503
