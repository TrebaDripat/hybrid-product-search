import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hybrid_search.domain import Filters, SearchRequest, parse_catalog

DATA = Path(__file__).parents[1] / "data"


def test_frozen_judgments_reference_eligible_catalog_products():
    products = {p.id: p for p in parse_catalog(json.loads((DATA / "catalog.json").read_text()))}
    queries = json.loads((DATA / "queries.json").read_text())
    assert len({q["id"] for q in queries}) == len(queries)
    assert {q["split"] for q in queries} == {"development", "evaluation"}
    for q in queries:
        request = SearchRequest(query=q["query"], filters=q["filters"])
        for identifier, grade in q["relevance"].items():
            assert grade in (1, 2, 3)
            assert request.filters.allows(products[identifier])


def test_reject_invalid_filters_and_search_budgets():
    for values in (
        {"query": " "},
        {"query": "mouse", "k": 0},
        {"query": "mouse", "k": 10, "candidate_limit": 5},
        {"query": "mouse", "filters": {"min_price_cents": 100, "max_price_cents": 50}},
        {"query": "mouse", "filters": {"max_price_cents": -1}},
        {"query": "mouse", "unknown": True},
    ):
        with pytest.raises(ValidationError):
            SearchRequest.model_validate(values)
    assert Filters(max_price_cents=0).max_price_cents == 0


def test_bad_catalog_never_silently_drops_products():
    rows = json.loads((DATA / "catalog.json").read_text())
    with pytest.raises(ValueError, match="duplicate"):
        parse_catalog(rows + [rows[0]])
    with pytest.raises(ValueError, match="empty"):
        parse_catalog([])
